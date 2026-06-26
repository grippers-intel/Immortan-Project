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
# Heart : Watchdog 역할. 테스트를 원활하게 동작시키기 위해 주석 처리. 필요 시 app/common.py에서 Heart 클래스 확인 후 사용
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
    # 퍼블리셔/서비스 생성 후 YOLO 노드가 켜질 때까지 대기
    # (/yolov5_ros2/init_finish, /yolov5/start, /yolov5/stop 서비스 wait_for_service())
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name
        self.is_running = True
        # PID Gain 설정: P=0.4, I=0.0, D=0.05
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
        # lane_detect.py의 LaneDetector 클래스 인스턴스 생성
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

    # 타이머 1회 실행
    # - `only_line_follow`가 False면 `/yolov5/start` 호출 → YOLO 추론 시작
    # - `enter_srv_callback()` → 카메라 이미지 구독 + YOLO 결과 구독 등록
    # - `set_running_srv_callback(True)` → 주행 플래그 ON
    # - `threading.Thread(target=self.main)` → 메인 주행 루프 시작
    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter("only_line_follow").value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(3)

        if 1:  # self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        # self.park_action()
        # self.right_action()
        # 위 두 주석은 주차/우회전 기능을 시작과 동시에 수행하도록 함
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        # TODO :우회전 구현
        self.have_turn_right = False
        self.detect_turn_right = False
        self.right_x = -1
        self.count_right = 0
        self.count_right_miss = 0  # right sign이 잠시 탐지되지 않을 때를 대비한 카운트
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.3  # normal driving speed
        self.slow_down_speed = 0.3  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

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

    # TODO : shutdown() 구현 필요. 현재는 ctrl+c로 종료 시 rclpy.shutdown() 호출되어 노드 종료됨
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

        # 안 쓰는 모델
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

    # TODO : right action 구현
    def right_action(self):
        twist = Twist()

        # TODO : crosswalk stop 수행 시 이 부분 삭제할 것.
        self.mecanum_pub.publish(Twist())
        time.sleep(1)  # TODO : 횡단보도 앞 1초 stop

        twist.linear.x = 0.2
        twist.angular.z = -0.4
        self.mecanum_pub.publish(twist)
        time.sleep(3)  # TODO : 회전 시간, 속도, 각도 조절 필요

        self.mecanum_pub.publish(Twist())
        time.sleep(0.1)  # 0.1초 stop

        self.have_turn_right = False
        self.right_x = -1
        self.count_right = 0
        self.count_right_miss = 0
        self.detect_turn_right = (
            False  # have_turn_right = False로 설정하여 main 충돌 방지
        )

    # main() 루프 (주행 두뇌): 이미지 큐 → 차선 이진화 → 차선 중심좌표/각도 계산
    # → 횡단보도/신호/주차 상태 따라 감속·정지·주차 → PID로 조향 → cmd_vel 발행
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
                # TODO : 횡단보도 로직 구현
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                # TODO TODO : 횡단보도 정차 조건 70 -> 200~300, 3회 -> 8회
                if (
                    200 < self.crosswalk_distance < 300 and not self.start_slow_down
                ):  # The robot starts to slow down only when it is close enough to the zebra crossing
                    self.count_crosswalk += 1
                    if (
                        self.count_crosswalk > 5
                    ):  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        # TODO TODO
                        # self.start_slow_down = True  # sign for slowing down
                        # self.count_slow_down = (
                        #     time.time()
                        # )  # fixing time for slowing down
                        self.mecanum_pub.publish(Twist())
                        time.sleep(1)
                else:  # need to detect continuously, otherwise reset
                    self.count_crosswalk = 0

                # deceleration processing
                # TODO TODO : slow_down 조건 생략
                # if self.start_slow_down:
                if self.traffic_signs_status is not None:
                    area = abs(
                        self.traffic_signs_status.box[0]
                        - self.traffic_signs_status.box[2]
                    ) * abs(
                        self.traffic_signs_status.box[1]
                        - self.traffic_signs_status.box[3]
                    )
                    # 신호등이 빨간색일 때, 로봇은 정지 (red 인식, 면적 1000px 이하)
                    if (
                        self.traffic_signs_status.class_name == "red" and area < 1000
                    ):  # If the robot detects a red traffic light, it will stop
                        self.mecanum_pub.publish(Twist())
                        self.stop = True
                    # TODO : green 신호등일 때 같은 조건에서 정상 속도로 출발
                    elif (
                        self.traffic_signs_status.class_name == "green" and area < 1000
                    ):  # If the traffic light is green, the robot will slow down and pass through
                        # TODO : green 속도 올림
                        twist.linear.x = self.normal_speed
                        self.stop = False
                #     if (
                #         not self.stop
                #     ):  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                #         twist.linear.x = self.slow_down_speed
                #         if (
                #             time.time() - self.count_slow_down
                #             > self.crosswalk_length / twist.linear.x
                #         ):
                #             self.start_slow_down = False
                # else:
                twist.linear.x = self.normal_speed  # go straight with normal speed

                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                # 주차 조건
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
                #         self.count_park = 0  => 문제잠: 주차 앞쪽 cross walk 가 아니라 옆쪽이나 가까운 곳 crosswalk 봄

                # TODO TODO : parking logic

                if not self.start_park:
                    if 0 < self.park_x:
                        # and 150 < self.crosswalk_distance:
                        # When the robot is close enough to the crosswalk, increment counter
                        self.count_park += 1
                    # else:
                    #     # reset counter when condition not met
                    #     self.count_park = 0

                    # trigger parking only once when counter reaches threshold
                    if self.count_park >= 25:
                        self.mecanum_pub.publish(Twist())
                        self.start_park = True
                        self.stop = True
                        threading.Thread(target=self.park_action).start()

                # line following processing
                # self_driving.py가 lane_detect에서 계산한 lane_x를 PID 목표값(130px)과 비교해 조향
                result_image, lane_angle, lane_x = self.lane_detect(
                    binary_image, image.copy()
                )  # the coordinate of the line while the robot is in the middle of the lane

                # TODO : have_right_turn을 포함하여 elif로 바꿈
                if self.have_turn_right:
                    continue

                elif lane_x >= 0 and not self.stop:
                    # 150 px 이상이면 차선이 로봇 중앙에서 벗어난 것으로 판단하고, 회전 속도를 -0.6로 고정해 회전
                    # TODO TODO : 코너링 조건 추가
                    if lane_x > 150:
                        self.count_turn += 1
                        if self.count_turn > 5 and not self.start_turn:
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        # TODO : 코너링 각 조절 (0.45 -> 0.7)
                        if self.machine_type != "MentorPi_Acker":
                            twist.angular.z = -0.7  # turning speed
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0
                        if (
                            time.time() - self.start_turn_time_stamp > 2
                            and self.start_turn
                        ):
                            self.start_turn = False
                        if not self.start_turn:
                            # 차선이 로봇 중앙에 있을 때의 좌표(130px)를 PID 목표값으로 설정하고,
                            # 현재 차선 중심 좌표(lane_x)와 비교하여 PID 출력값을 twist.angular.z에 적용
                            self.pid.SetPoint = 130  # the coordinate of the line while the robot is in the middle of the lane
                            self.pid.update(lane_x)
                            if self.machine_type != "MentorPi_Acker":
                                twist.angular.z = common.set_range(
                                    self.pid.output, -0.1, 0.1
                                )
                            else:
                                # driver/sdk/sdk/common의 set_range()로 PID 출력값을 -0.1~0.1로 제한하고,
                                # 이를 로봇의 조향각으로 변환해 twist.angular.z에 적용
                                twist.angular.z = (
                                    twist.linear.x
                                    * math.tan(
                                        common.set_range(self.pid.output, -0.1, 0.1)
                                    )
                                    / 0.145
                                )
                        else:
                            # 안 쓰는 부분
                            if self.machine_type == "MentorPi_Acker":
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()
                    # TODO TODO
                    twist.linear.x = 0.1
                    self.mecanum_pub.publish(twist)  # TODO: 오른쪽 무한루프 제거 추가

                # TODO : 우회전 구현
                if self.right_x > 0 and not self.have_turn_right:
                    self.count_right += 1
                    if self.count_right >= 5:
                        self.count_right = 0
                        self.have_turn_right = (
                            True  # 제어권 잠금 (차선 인식 블록 건너뜀)
                        )
                        threading.Thread(target=self.right_action).start()

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
    # YOLO 결과(crosswalk/right/park/red/green)로 상태 변수 갱신
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

                # 횡단보도 y좌표(crosswalk_distance) 계산
                if class_name == "crosswalk":
                    if (
                        center[1] > min_distance
                    ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]

                # 우회전 카운트(count_right >= 5)
                # TODO : 우회전 sign을 발견한 경우 miss 값 0으로 초기화
                elif class_name == "right":  # obtain the right turning sign
                    self.right_x = center[0]
                    self.count_right_miss = 0
                    self.detect_turn_right = True
                # 주차 표지 x좌표(park_x)
                elif (
                    class_name == "park"
                ):  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                # 신호등 상태(traffic_signs_status)
                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light
                    self.traffic_signs_status = i
                # TODO : 프레임에서 우회전 표시를 인식 못한 경우
                elif not self.detect_turn_right and not self.have_turn_right:
                    self.count_right_miss += 1
                    if self.count_right_miss > 5:  # 5프레임 이상 미탐지 시
                        self.count_right = 0  # 카운트 리셋
                        self.right_x = -1

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
