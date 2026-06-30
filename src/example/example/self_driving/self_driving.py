import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common

# from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState

# TODO : LED 구현
# from gpiozero import LED


class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name
        self.is_running = True
        # [튜닝] 차선추종 직선 보정용 PID 게인 (P, I, D)
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get("MACHINE_TYPE")
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )
        self.result_publisher = self.create_publisher(Image, "~/image_result", 1)

        self.create_service(Trigger, "~/enter", self.enter_srv_callback)
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)  # exit the game
        self.create_service(SetBool, "~/set_running", self.set_running_srv_callback)
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, "/yolov5_ros2/init_finish")
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(
            Trigger, "/yolov5/start", callback_group=timer_cb_group
        )
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(
            Trigger, "/yolov5/stop", callback_group=timer_cb_group
        )
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(
            0.0, self.init_process, callback_group=timer_cb_group
        )

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter("only_line_follow").value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)

        if 1:  # self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign
        self.park_area = 0  # 현재 프레임에서 인식된 park 박스 면적(px^2)
        self.park_area_threshold = 1000  # 박스 면적이 이 값보다 크면 가까워진 것으로 판단


        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        # [우회전 동작] 우회전 표지판 인식(turn_right) 후 횡단보도 정지 → 우회전 수행.
        self.doing_turn_right = (
            False  # 우회전 동작 수행 중(이 동안 차선추종은 제어 양보)
        )
        self.turn_right_speed = 0.05  # 우회전 시 전진 속도
        self.turn_right_angular = (
            -0.7
        )  # 우회전 각속도(음수=우회전). 절댓값 ↑ = 더 급하게 돔
        self.turn_right_duration = (
            2.0  # 우회전 동작 시간(초). 덜 돌면 ↑, 과하게 돌면 ↓ (90도 맞춰 튜닝)
        )

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # stopping sign
        self.stop_reason = None #TODO: stop 이유 추가 
        self.start_park = False  # start parking sign 했는지 안했는지 여부

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        # [횡단보도 정지] 규칙: 횡단보도 앞 반드시 정지 후 출발. (기존 코드는 감속만 했고
        #   slow_down_speed가 normal_speed와 같아 감속조차 안 보였음)
        self.crosswalk_stop_dist = 200  # crosswalk_distance가 이 값보다 크면(가까우면) 정지. 값↑=더 가까이서 멈춤.
        # TODO : (150→200: 멀리서 미리 멈춰 신호등을 못 보던 문제 해결)
        self.crosswalk_stop_duration = 2.0  # 정지 유지 시간(초)
        self.crosswalk_stopping = False  # 현재 횡단보도에서 정지 중인가
        self.crosswalk_stop_time = 0  # 정지 시작 시각
        self.crosswalk_passed = False  # 이번 횡단보도 통과 처리 완료(중복 정지 방지)

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.3  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        # ===== [1단계] 차선추종(Lane Keeping) 튜닝 파라미터 =====
        # 기존에 main() 안에 하드코딩되어 있던 값들을 여기로 모음. 동작은 기존과 동일.
        # 실차 주행 후 아래 값들만 조정하면 차선추종 성향을 바꿀 수 있음.
        #
        # lane_setpoint: 로봇이 차선 중앙일 때의 목표 lane_x 픽셀값(PID 목표점).
        #   값을 키우면 차가 더 오른쪽, 줄이면 더 왼쪽으로 붙어 주행함.
        self.lane_setpoint = 130
        self.lane_setpoint = 140
        # turn_threshold: 급회전 진입 임계값. lane_x가 이 값보다 크면 코너로 판단해 고정 회전.
        #   코너를 못 돌고 직진해 이탈하면 ↓, 직선에서 불필요하게 꺾이면 ↑.
        #   코너를 너무 빨리/일찍 도는 증상 → ↑ (진입 늦춤). lane_setpoint(130)보다 충분히 커야 함.
        #   캘리브레이션 개선 후 150→180→200 으로 단계적 상향.
        #   ※ 아래 main()의 lane_x 로그로 직선/코너 실제값을 보고 정밀 조정할 것.
        self.turn_threshold = 210
        self.turn_threshold = 200
        # turn_angular_z: 급회전 구간의 고정 회전 각속도(rad/s, 음수=우회전).
        #   코너 안쪽으로 파고들면 절댓값 ↓, 못 돌고 바깥으로 나가면 절댓값 ↑.
        #   [복원] 실차 결과 초기값이 더 안정적이라 -0.38 → -0.45(원래)로 되돌림.
        self.turn_angular_z = -0.8
        # angular_z_limit: 직선 PID 보정 출력의 최대 회전 각속도(rad/s) 제한.
        #   직선에서 좌우 흔들림(진동)이 크면 ↓.
        self.angular_z_limit = 0.1
        # lane_deadband: 직선 보정 데드밴드(픽셀). |lane_x - lane_setpoint|가 이 값 이내면
        #   조향하지 않고 직진(미세 진동 제거). 0이면 비활성(원래 동작).
        #   [복원] 효과가 뚜렷하지 않아 6 → 0(비활성)으로 되돌림. 필요시 4~8로 재시도 가능.
        self.lane_deadband = 0
        # [3단계] turn_confirm_count: 회전 진입 확정에 필요한 연속 검출 프레임 수.
        #   값 ↑ 이면 코너를 더 신중히(늦게) 진입해 오검출 방지, 값 ↓ 이면 민감하게 빨리 진입.
        self.turn_confirm_count = 7 #TODO: 5->7
        self.turn_confirm_count = 5 #TODO: 5->7
        # turn_recover_time: 회전 시작 후 PID 직선보정으로 복귀하기까지의 유지 시간(초).
        #   회전 직후 치우치면 ↓(예: 1.0), 회전이 덜 끝난 채 흔들리면 ↑.
        #   [복원] 1.5 → 2.0(원래)으로 되돌림.
        self.turn_recover_time = 2.0

        self.traffic_signs_status = None  # record the state of the traffic lights

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "self driving enter")
        with self.lock:
            self.start = False
            camera = "depth_cam"  # self.get_parameter('depth_camera_name').value
            self.create_subscription(
                Image, "/ascamera/camera_publisher/rgb0/image", self.image_callback, 1
            )
            self.create_subscription(
                ObjectsInfo, "/yolov5_ros2/object_detect", self.get_object_callback, 1
            )
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info("\033[1;32m%s\033[0m" % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self):  # press 'ctrl+c' to close the program
        self.is_running = False

    def image_callback(self, ros_image):  # callback target checking
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)

    # parking processing
    def park_action(self):
        if self.machine_type == "MentorPi_Mecanum":
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38 / 0.2)
        self.mecanum_pub.publish(Twist())
        self.shutdown()

    # 우회전 동작 (우회전 표지판 + 횡단보도 정지 후 실행).
    # TODO : park_action처럼 별도 스레드로 동작.
    def turn_right_action(self):
        twist = Twist()

        # HARD - 횡단보도 정차 후 직진
        twist.linear.x = self.normal_speed
        self.mecanum_pub.publish(twist)
        time.sleep(1)

        # HARD - 우회전
        twist.linear.x = self.turn_right_speed  # 전진하며
        twist.angular.z = self.turn_right_angular  # 우회전
        self.mecanum_pub.publish(twist)
        time.sleep(self.turn_right_duration)  # 90도 맞춰 튜닝
        self.mecanum_pub.publish(Twist())  # 정지
        # self.doing_turn_right = False  # 차선추종 재개
        self.have_turn_right = True

    def main(self):
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                # 횡단보도 정지 처리 (규칙: 횡단보도 앞 반드시 정지 후 출발, 신호등 빨강이면 계속 정지)
                # [디버그 로그] crosswalk=거리, stopping=정지중, passed=통과처리됨, sign=신호등상태
                self.get_logger().info(
                    "\033[1;33mcrosswalk=%d stopping=%s passed=%s sign=%s\033[0m"
                    % (
                        self.crosswalk_distance,
                        self.crosswalk_stopping,
                        self.crosswalk_passed,
                        (
                            self.traffic_signs_status.class_name
                            if self.traffic_signs_status is not None
                            else "none"
                        ),
                    )
                )
                self.get_logger().info(
                    "\033[1;33mpark_x:%s crosswalk:%s count_park:%s park_area:%s\033[0m"
                    % (
                        self.park_x, 
                        self.crosswalk_distance,
                        self.count_park,
                        self.park_area)) #TODO:

                twist.linear.x = self.normal_speed  # 기본 직진 속도

                # TODO : 우회전 표지 감지 시 Parking까지 수행 후 shutdown

                if self.turn_right and not self.doing_turn_right:
                    self.mecanum_pub.publish(Twist())
                    time.sleep(1)
                    self.turn_right = False
                    self.doing_turn_right = True
                    self.turn_right_action()
                    twist.linear.x = self.normal_speed
                    self.mecanum_pub.publish(twist)
                    # time.sleep(6)

                else:
                    if (
                        not self.start_park
                        and not self.doing_turn_right
                        and self.crosswalk_distance > self.crosswalk_stop_dist
                        and not self.crosswalk_passed
                    ):
                        # 횡단보도가 충분히 가까움 → 정지 단계
                        if not self.crosswalk_stopping:
                            self.crosswalk_stopping = True
                            self.crosswalk_stop_time = (
                                time.time()
                            )  # 정지 시작 시각 기록
                        # 신호등이 빨강이면 계속 정지, 빨강이 아니면(초록/없음) 정해진 시간 정지 후 통과 허용
                        is_red = (
                            self.traffic_signs_status is not None
                            and self.traffic_signs_status.class_name == "red"
                        )
                        stopped_enough = (
                            time.time() - self.crosswalk_stop_time
                        ) > self.crosswalk_stop_duration
                        if stopped_enough and not is_red:
                            self.crosswalk_passed = (
                                True  # 통과 허용 → 이후 차선추종으로 진행
                            )
                            self.crosswalk_stopping = False
                            self.stop = False
                            self.stop_reason = None
                        else:
                            self.stop = True  # 정지 유지
                            self.stop_reason = "crosswalk"
                            self.mecanum_pub.publish(Twist())
                    else:
                        # 횡단보도에서 멀어지면(사라지면) 다음 횡단보도를 위해 상태 리셋
                        if self.crosswalk_distance < 70:
                            self.crosswalk_passed = False
                            self.crosswalk_stopping = False
                        if self.stop_reason == "crosswalk":
                            self.stop = False
                            self.stop_reason = None

                self.get_logger().info(
                            "parking trigger: reason=%s park_x=%s park_area=%s count_park=%s"
                            % (
                                self.stop_reason,
                                self.park_x,
                                self.park_area,
                                self.count_park,
                            )
                        )


                # [주차 전 정지 원인 확인]
                if not self.start_park:
                    if self.park_x > 0 and self.park_area > self.park_area_threshold:
                        self.count_park += 1
                    # else:
                    #     self.count_park = 0
                    if self.count_park >= 10:
                        self.start_park = True
                        self.stop = True
                        self.stop_reason = "park"
                        self.count_park = 0
                        self.mecanum_pub.publish(Twist())
                        threading.Thread(target=self.park_action, daemon=True).start()
                        # self.shutdown()

                # # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                # if 0 < self.park_x and 135 < self.crosswalk_distance:
                #     twist.linear.x = self.slow_down_speed
                #     if (
                #         not self.start_park and 180 < self.crosswalk_distance
                #     ):  # When the robot is close enough to the crosswalk, it will start parking
                #         self.count_park += 1
                #         if self.count_park >= 15:
                #             self.mecanum_pub.publish(Twist())
                #             self.start_park = True
                #             self.stop = True
                #             threading.Thread(target=self.park_action).start()
                #     else:
                #         self.count_park = 0

                # line following processing
                # [핵심수정] 회전/보정 판단을 '가까운 ROI 기준'(near)으로 변경.
                #   기존 lane_x는 max_center_x(=far)로, 먼 ROI가 앞쪽 코너를 미리 봐서 회전이 너무 일찍 트리거됐음.
                #   → lane_x 에 near 값을 받아 이후 로직(회전 threshold, PID)은 그대로 두고 판단 기준만 바꿈.
                #   되돌리려면 lane_x_far 를 lane_x 로 받으면 기존 동작.
                result_image, lane_angle, lane_x_far, lane_x = self.lane_detect(
                    binary_image, image.copy()
                )
                # [튜닝 로그] 필요시 주석 해제. near=회전판단 기준값, far=기존 max값.
                # self.get_logger().info('\033[1;36mlane_x(near)=%d  far=%d  (turn_threshold=%d)\033[0m' % (lane_x, lane_x_far, self.turn_threshold))
                if (
                    lane_x >= 0 and not self.stop and not self.doing_turn_right
                ):  # 우회전 동작 중엔 차선추종 양보
                    if (
                        lane_x > self.turn_threshold
                    ):  # [튜닝] 급회전 진입 임계값 (param_init의 turn_threshold)
                        self.count_turn += 1
                        if (
                            self.count_turn > self.turn_confirm_count
                            and not self.start_turn
                        ):  # [3단계] 회전 진입 확정 (param_init의 turn_confirm_count)
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != "MentorPi_Acker":
                            twist.angular.z = (
                                self.turn_angular_z
                            )  # [튜닝] 고정 회전 각속도 (param_init의 turn_angular_z)
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0
                        if (
                            time.time() - self.start_turn_time_stamp
                            > self.turn_recover_time
                            and self.start_turn
                        ):  # [3단계] 회전 후 PID 복귀까지 유지 시간 (param_init의 turn_recover_time)
                            self.start_turn = False
                        if not self.start_turn:
                            self.pid.SetPoint = (
                                self.lane_setpoint
                            )  # [튜닝] 차선 중앙 목표점 (param_init의 lane_setpoint)
                            # [2단계] 데드밴드: 차선 오차가 lane_deadband 이내면 조향하지 않고 직진.
                            #   프레임마다 1~2px씩 떨리는 측정 노이즈로 인한 미세 진동(꼬물거림)을 제거함.
                            if abs(lane_x - self.lane_setpoint) < self.lane_deadband:
                                self.pid.clear()  # PID 내부 상태 초기화로 데드밴드 이탈 시 튐 방지
                                twist.angular.z = 0.0
                            else:
                                self.pid.update(lane_x)
                                if self.machine_type != "MentorPi_Acker":
                                    twist.angular.z = common.set_range(
                                        self.pid.output,
                                        -self.angular_z_limit,
                                        self.angular_z_limit,
                                    )  # [튜닝] 출력 제한 (param_init의 angular_z_limit)
                                else:
                                    twist.angular.z = (
                                        twist.linear.x
                                        * math.tan(
                                            common.set_range(
                                                self.pid.output,
                                                -self.angular_z_limit,
                                                self.angular_z_limit,
                                            )
                                        )
                                        / 0.145
                                    )
                        else:
                            if self.machine_type == "MentorPi_Acker":
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()

                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )

            else:
                time.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    # Obtain the target detection result
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        self.crosswalk_distance = 0
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (
                    int((i.box[0] + i.box[2]) / 2),
                    int((i.box[1] + i.box[3]) / 2),
                )

                if class_name == "crosswalk":
                    if (
                        center[1] > min_distance
                    ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
                elif class_name == "right":  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if (
                        self.count_right >= 8
                    ):  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif (
                    class_name == "park"
                ):  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                    box = i.box
                    width = abs(box[2] - box[0])
                    height = abs(box[3] - box[1])
                    self.park_area = width * height

                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light
                    self.traffic_signs_status = i

            self.get_logger().info("\033[1;32m%s\033[0m" % class_name)
            self.crosswalk_distance = min_distance


def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":