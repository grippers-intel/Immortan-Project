#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
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
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get("MACHINE_TYPE")
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )
        self.result_publisher = self.create_publisher(Image, "~/image_result", 1)

        self.create_service(
            Trigger, "~/enter", self.enter_srv_callback
        )  # enter the game
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)  # exit the game
        self.create_service(SetBool, "~/set_running", self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
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

        if True:  # self.get_parameter('start').value:
            self.start_delay_time = time.time()  # TODO 01 : 시작 시간 기록
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        # self.park_action()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

    def param_init(self):
        self.stop_time = 0  # TODO 00 : 교차로 인식 시 일단 정지
        self.start_delay = True  # TODO 01 : 시작 딜레이 (오작동 우회전 방지)
        self.start_delay_time = 0  # TODO 01 : 시작 딜레이 (오작동 우회전 방지)
        # TODO 01 : 상황별 주행 파라미터 추가(~113행)
        self.drive_params = {
            "straight": {
                "linear_x": 0.3,
                "angular_z": 0.0,
                "pid_p": 0.4,
                "pid_d": 0.05,
            },
            "turn_right": {
                "linear_x": 0.2,
                "angular_z": -0.2,
                "pid_p": 0.4,
                "pid_d": 0.05,
            },
            "slow_down": {
                "linear_x": 0.15,
                "angular_z": 0.0,
                "pid_p": 0.4,
                "pid_d": 0.05,
            },
        }
        self.current_drive_mode = "straight"

        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_ignore = False  # TODO 00 : 횡단보도 무시 플래그 → 추가
        self.crosswalk_ignore_time = 0  # TODO 00 : 무시 시작 시간 → 추가
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.1  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

    # TODO 01 : 상황별 주행 모드 전환 함수
    def set_drive_mode(self, mode):
        if self.current_drive_mode == mode:  # 이미 같은 모드면 무시
            return
        self.current_drive_mode = mode
        params = self.drive_params[mode]
        self.pid.Kp = params["pid_p"]
        self.pid.Kd = params["pid_d"]
        self.get_logger().info(f"Drive mode changed: {mode}")

    # TODO 02 : 교차로 우회전 동작 함수
    def turn_right_action(self):
        self.stop = True  # 주행 로직 잠시 멈춤

        twist = Twist()
        twist.linear.x = self.drive_params["turn_right"]["linear_x"]
        twist.angular.z = self.drive_params["turn_right"]["angular_z"]
        self.mecanum_pub.publish(twist)
        time.sleep(1)  # 90도 회전 시간 (테스트 후 튜닝)

        self.mecanum_pub.publish(Twist())  # 정지
        self.stop = False  # 주행 로직 재개
        self.turn_right = False  # 표지판 플래그 리셋

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

    def shutdown(self, signum, frame):  # press 'ctrl+c' to close the program
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
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38 / 0.2)
        elif self.machine_type == "MentorPi_Acker":
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -twist.linear.x * math.tan(-0.5061) / 0.145
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        else:
            twist = Twist()
            twist.angular.z = -1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.65 / 0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
        self.mecanum_pub.publish(Twist())

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

                # if detecting the zebra crossing, start to slow down
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                self.get_logger().info(f"crosswalk_distance: {self.crosswalk_distance}, crosswalk_ignore: {self.crosswalk_ignore}")  # ← 추가
                if self.crosswalk_ignore:  # TODO 00 : ignore 체크 → 추가
                    if time.time() - self.crosswalk_ignore_time > 3.5:
                        self.crosswalk_ignore = False
                if (
                    300 < self.crosswalk_distance
                    and not self.start_slow_down
                    and not self.crosswalk_ignore
                ):  # TODO 00 : ignore 조건 추가
                    self.count_crosswalk += 1
                    if (
                        self.count_crosswalk == 3
                    ):  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        self.start_slow_down = True  # sign for slowing down
                        self.count_slow_down = (
                            time.time()
                        )  # fixing time for slowing down
                else:
                    if not self.start_slow_down:
                        self.count_crosswalk = 0

                # deceleration processing
                # TODO 00 : 교차로 인식 시 일단 정지
                if self.start_slow_down:
                    if not self.stop:
                        self.mecanum_pub.publish(
                            Twist()
                        )  # 횡단보도 감지하면 일단 무조건 정지
                        self.stop = True
                        self.stop_time = time.time()  # 정지한 시간 기록
                        self.count_turn = 0  # TODO 00 : 우회전 카운트 리셋
                        self.start_turn = False  # TODO 00 : 우회전 플래그 리셋

                    if self.traffic_signs_status is not None:
                        area = abs(
                            self.traffic_signs_status.box[0]
                            - self.traffic_signs_status.box[2]
                        ) * abs(
                            self.traffic_signs_status.box[1]
                            - self.traffic_signs_status.box[3]
                        )
                        if (
                            self.traffic_signs_status.class_name == "red"
                            and area < 1000
                        ):
                            pass  # 빨간불이면 계속 대기
                        elif self.traffic_signs_status.class_name == "green":
                            self.stop = False  # 초록불이면 출발
                            self.start_slow_down = False
                            self.crosswalk_distance = 0  # TODO 00 : 거리 초기화 → 추가
                            self.crosswalk_ignore = True  # TODO 00 : 무시 시작 → 추가
                            self.crosswalk_ignore_time = (
                                time.time()
                            )  # TODO 00 : 무시 시작 시간 → 추가

                    else:
                        # 신호등 없으면 1초 후 출발
                        if (
                            time.time() - self.stop_time > 1.0
                        ):  # TODO 00 숫자 변경 = 정지 시간 변경(x.0)
                            if self.turn_right:  # TODO 02 : 우회전 표지판 확인
                                self.start_slow_down = False
                                threading.Thread(
                                    target=self.turn_right_action
                                ).start()  # TODO 02 : 우회전 표지판 확인

                            else:
                                self.stop = False
                                self.start_slow_down = False
                                self.crosswalk_distance = (
                                    0  # TODO 00 : 거리 초기화 → 추가
                                )
                                self.crosswalk_ignore = (
                                    True  # TODO 00 : 무시 시작 → 추가
                                )
                                self.crosswalk_ignore_time = (
                                    time.time()
                                )  # TODO 00 : 무시 시작 시간 → 추가

                    if not self.stop:
                        self.set_drive_mode("slow_down")  # TODO 01
                        twist.linear.x = self.drive_params["slow_down"][
                            "linear_x"
                        ]  # TODO 01
                        # twist.linear.x = self.slow_down_speed

                # if self.start_slow_down:
                #     if self.traffic_signs_status is not None:
                #         area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                #         if self.traffic_signs_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                #             self.mecanum_pub.publish(Twist())
                #             self.stop = True
                #         elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                #             twist.linear.x = self.slow_down_speed
                #             self.stop = False
                #     if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                #         twist.linear.x = self.slow_down_speed
                #         if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                #             self.start_slow_down = False

                else:
                    self.set_drive_mode("straight")  # TODO 01 : 직진 모드 전환
                    twist.linear.x = self.drive_params["straight"][
                        "linear_x"
                    ]  # TODO 01 : 직진 속도 적용

                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                if 0 < self.park_x and 135 < self.crosswalk_distance:
                    twist.linear.x = self.slow_down_speed
                    if (
                        not self.start_park and 180 < self.crosswalk_distance
                    ):  # When the robot is close enough to the crosswalk, it will start parking
                        self.count_park += 1
                        if self.count_park >= 15:
                            self.mecanum_pub.publish(Twist())
                            self.start_park = True
                            self.stop = True
                            threading.Thread(target=self.park_action).start()
                    else:
                        self.count_park = 0

                # line following processing
                # TODO 01 : 시작 딜레이 체크
                if self.start_delay:
                    if (
                        time.time() - self.start_delay_time > 3.0
                    ):  # TODO 01 : 3초 후 딜레이 해제
                        self.start_delay = False
                    else:
                        continue
                result_image, lane_angle, lane_x, center_x = self.lane_detect(
                    binary_image, image.copy()
                )  # TODO 01 : center_x 추가
                self.get_logger().info(f"lane_x: {lane_x}")
                # TODO 02 : 우회전 디버그 로그
                self.get_logger().info(
                    f"center_x: {center_x}"
                )  # TODO 01 : 디버그 로그 추가
                if not self.stop and not self.start_delay:  # TODO 01 : 딜레이 조건 추가
                    if (
                        len(center_x) >= 5 and center_x[0] == -1 and center_x[2] == -1
                    ):  # TODO 01 : 박스 2번 없을 때 우회전 감지
                        self.count_turn += 1
                        if (
                            self.count_turn > 10 and not self.start_turn
                        ):  # TODO 01 : 10프레임 연속 → 수정
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                            self.start_slow_down = (
                                False  # TODO 01 : 우회전 시작 시 횡단보도 플래그 리셋
                            )
                            self.stop = False  # TODO 01 : 정지 플래그 리셋
                    else:
                        self.count_turn = 0  # TODO 01 : else로 리셋 → 추가
                    if self.start_turn:
                        if self.machine_type != "MentorPi_Acker":
                            twist.angular.z = self.drive_params["turn_right"][
                                "angular_z"
                            ]  # -0.25
                            twist.linear.x = self.drive_params["turn_right"][
                                "linear_x"
                            ]  # TODO 01 : 코너 속도 0.15 적용
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0

                        if not self.start_turn:
                            self.pid.SetPoint = 185  # TODO 도로 중앙값 조절( 130 -> 170 좀더 왼쪽으로 붙어서감)
                            self.pid.update(lane_x)
                            if self.machine_type != "MentorPi_Acker":
                                twist.angular.z = common.set_range(
                                    self.pid.output, -0.1, 0.1
                                )
                            else:
                                twist.angular.z = (
                                    twist.linear.x
                                    * math.tan(
                                        common.set_range(self.pid.output, -0.1, 0.1)
                                    )
                                    / 0.145
                                )
                        else:
                            if self.machine_type == "MentorPi_Acker":
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()
                    # TODO 01 : start_turn 탈출 조건 else 밖으로 이동 (차선 없어도 탈출 가능)
                if self.start_turn:
                    if (
                        time.time() - self.start_turn_time_stamp > 4.0
                    ):  # TODO 01 : 4초 후 강제 탈출
                        self.start_turn = False
                        self.count_turn = 0  # TODO 01 : 카운트 동시 리셋

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
                    box_area = abs(i.box[0] - i.box[2]) * abs(
                        i.box[1] - i.box[3]
                    )  # TODO 00 : 박스 크기 계산
                    if box_area > 5000:  # TODO 00 : 오인식 방지 기준값 (튜닝 필요)
                        if (
                            center[1] > min_distance
                        ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                            min_distance = center[1]
                            self.get_logger().info(
                                f"crosswalk box_area: {box_area}"
                            )  # TODO 00 : 사이즈 로그 → 추가
                elif class_name == "right":  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if (
                        self.count_right >= 5
                    ):  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == "park":
                    box_area = abs(i.box[0] - i.box[2]) * abs(
                        i.box[1] - i.box[3]
                    )  # TODO 00 : 박스 크기 계산
                    if box_area > 1000:  # TODO 00 : 오인식 방지
                        self.park_x = center[0]
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
    main()
