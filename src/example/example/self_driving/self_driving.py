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
from ros_robot_controller_msgs.msg import (
    BuzzerState,
    SetPWMServoState,
    PWMServoState,
    RGBState,
    RGBStates,
    ButtonState,
)
import RPi.GPIO as GPIO

# GPIO 전구 핀 설정
PIN_GREEN = 24
PIN_RED = 25
PIN_YELLOW_LEFT = 23
PIN_YELLOW_RIGHT = 18
GPIO_ALL_PINS = [PIN_GREEN, PIN_RED, PIN_YELLOW_LEFT, PIN_YELLOW_RIGHT]
LED_ON = 1  # 반대면 0↔1 스왑
LED_OFF = 0
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for _pin in GPIO_ALL_PINS:
    GPIO.setup(_pin, GPIO.OUT)
    GPIO.output(_pin, LED_OFF)


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
        self.binary_publisher = self.create_publisher(Image, "~/image_binary", 1)
        self.rgb_pub = self.create_publisher(
            RGBStates, "/ros_robot_controller/set_rgb", 10
        )

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

        self.display = True
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        self.create_subscription(
            ButtonState, "/ros_robot_controller/button", self.button_callback, 1
        )
        self.get_logger().info(
            "\033[1;32m%s\033[0m" % "버튼 대기 중 - 아무 버튼이나 클릭하면 시작"
        )

        # self.park_action()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

    def param_init(self):
        self.stop_time = 0  # TODO 00 : 교차로 인식 시 일단 정지
        self.start_delay = True  # TODO 01 : 시작 딜레이 (오작동 우회전 방지)
        self.start_delay_time = 0  # TODO 01 : 시작 딜레이 (오작동 우회전 방지)
        self._last_btn_time = 0  # 수동 더블클릭 감지용
        # TODO 01 : 상황별 주행 파라미터 추가(~113행)
        self.drive_params = {
            "straight": {
                "linear_x": 0.5,
                "angular_z": 0.0,
                "pid_p": 0.4,
                "pid_d": 0.05,
            },
            "turn_right": {
                "linear_x": 0.45,  # TODO : 0.30→0.375→0.45 (직진 0.5에서 급감속 없애기)
                "angular_z": -1.4,  # TODO : -0.55→-1.35→-1.0→-1.2→-1.4 (늦은트리거가 원인, box5=200으로 조기트리거 복구 후 -1.4 유지)
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
        self.turn_count_start_box1 = (
            -1
        )  # TODO : 회전 카운트 시작 시점 박스1 값 (드리프트 감지용) → 추가
        self.count_turn_exit = 0  # TODO : 회전 종료 판단용 카운트 → 추가
        self.turn_exit_time = (
            0  # TODO : 회전 종료 시점 (교차로 차선 끊김 대비 강제 직진용) → 추가
        )
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_total_count = (
            0  # 총 횡단보도 감지 횟수 (홀수=정지, 짝수=통과, 9=주차)
        )
        self.just_stopped_crosswalk = (
            False  # 구버전: 정지 후 다음 횡단보도 강제 무시 플래그
        )
        self.accel_ramp_active = False  # 정지 후 재출발 가속 구간 플래그
        self.accel_ramp_start_time = 0  # 가속 구간 시작 시간
        self.crosswalk_ignore = False  # TODO 00 : 횡단보도 무시 플래그 → 추가
        self.crosswalk_ignore_time = 0  # TODO 00 : 무시 시작 시간 → 추가
        self.turn_guard_time = 0  # 우회전 트리거 가드 시간 (실제 정지/우회전 완료 때만 리셋, 구버전 강제무시는 리셋 안함)
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_box_height = (
            0  # YOLO box_height of closest crosswalk (proximity proxy)
        )
        self.crosswalk_last_seen_time = 0  # TODO : 깜빡임 방지용 마지막 감지 시간
        self.crosswalk_raw_last_seen_time = (
            0  # TODO : 필터 통과 여부 상관없이 YOLO가 crosswalk를 본 마지막 시간 → 추가
        )
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.1  # normal driving speed
        self.slow_down_speed = 0.15  # slowing down speed (0.05→0.15: 너무 느려서 7초에 0.35m밖에 못 가 cw_h가 50 미달)
        self.pre_slow_down = False  # count=1 감지 시 pre-decel 플래그

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

    def set_rgb(self, r, g, b):
        msg = RGBStates()
        msg.states = [
            RGBState(index=1, red=r, green=g, blue=b),
            RGBState(index=2, red=r, green=g, blue=b),
        ]
        self.rgb_pub.publish(msg)

    def button_callback(self, msg):
        now = time.time()
        if msg.state in (1, 5):
            if self.start and (now - self._last_btn_time) < 0.5:
                # 빠른 연속 클릭 = 수동 더블클릭 리셋
                self.get_logger().info(
                    "\033[1;32m%s\033[0m" % "연속클릭 - 대기 상태로 초기화"
                )
                self.mecanum_pub.publish(Twist())
                self.param_init()
            elif not self.start:
                self.get_logger().info("\033[1;32m%s\033[0m" % "버튼 입력 - 주행 시작")
                self.start_delay_time = now
                request = SetBool.Request()
                request.data = True
                self.set_running_srv_callback(request, SetBool.Response())
            self._last_btn_time = now
        elif msg.state == 6:  # 하드웨어 더블클릭 이벤트 (백업)
            self.get_logger().info(
                "\033[1;32m%s\033[0m" % "더블클릭 - 대기 상태로 초기화"
            )
            self.mecanum_pub.publish(Twist())
            self.param_init()

    # TODO 02 : 교차로 우회전 동작 함수
    def turn_right_action(self):
        self.stop = True  # 주행 로직 잠시 멈춤

        twist = Twist()
        twist.linear.x = self.drive_params["turn_right"]["linear_x"]
        twist.angular.z = self.drive_params["turn_right"]["angular_z"]
        self.mecanum_pub.publish(twist)

        turn_start = time.time()
        crosswalk_hit = False
        while (
            time.time() - turn_start < 1.0
        ):  # TODO : 0.6→1.0초 ≈ 80도 (angular_z=-1.35 기준, 최소 70~80도 보장)
            if self.crosswalk_box_height > 30:  # 회전 중 횡단보도 감지 시 즉시 중단
                crosswalk_hit = True
                break
            time.sleep(0.03)

        self.mecanum_pub.publish(Twist())  # 정지
        self.stop = False  # 주행 로직 재개
        self.turn_right = False  # 표지판 플래그 리셋
        self.accel_ramp_active = True  # 우회전 후 재출발 가속 구간
        self.accel_ramp_start_time = time.time()

        if not crosswalk_hit:  # 정상 종료 시 횡단보도 재감지 방지
            self.crosswalk_ignore = True
            self.crosswalk_ignore_time = time.time()
            self.turn_guard_time = time.time()  # 우회전 직후 false-turn 방지
            self.pre_slow_down = False

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
            binary_image = self.lane_detect.get_binary(
                image
            )  # 대기 중에도 바이너리 발행용
            self.binary_publisher.publish(
                self.bridge.cv2_to_imgmsg(binary_image, "mono8")
            )
            if self.start:
                # TODO 01 : 시작 딜레이를 가장 먼저 체크
                if self.start_delay:
                    if time.time() - self.start_delay_time > 0.5:
                        self.start_delay = False
                        self.accel_ramp_active = True  # 출발 가속 구간 시작
                        self.accel_ramp_start_time = time.time()
                        self.get_logger().info(
                            "\033[1;32m%s\033[0m" % "start_delay 해제 - 주행 시작!"
                        )
                    else:
                        self.mecanum_pub.publish(Twist())  # 명시적으로 정지 명령
                        continue

                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                # if detecting the zebra crossing, start to slow down
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                self.get_logger().info(
                    f"crosswalk_distance: {self.crosswalk_distance}, crosswalk_ignore: {self.crosswalk_ignore}"
                )  # ← 추가
                if self.crosswalk_ignore:  # TODO 00 : ignore 체크 → 추가
                    if (
                        time.time() - self.crosswalk_ignore_time > 1.2
                    ):  # TODO : 3.5→6.0→1.5→0.65→0.8→1.2 - just_stopped_crosswalk 플래그가 코너 횡단보도 담당, ignore는 현재 횡단보도 통과만 커버 (1.2s≈57cm)
                        self.crosswalk_ignore = False
                if (
                    60 < self.crosswalk_distance
                    and not self.start_slow_down
                    and not self.crosswalk_ignore
                    and not self.start_turn  # 우회전 중 횡단보도 감지로 인한 회전 중단 방지
                ):
                    if not self.pre_slow_down:
                        # 새 횡단보도 최초 감지 - 번호 부여
                        self.crosswalk_total_count += 1
                        self.get_logger().info(
                            f"\033[1;36m횡단보도 #{self.crosswalk_total_count} 감지\033[0m"
                        )
                        if self.just_stopped_crosswalk:
                            # 구버전: 정지 직후 다음 횡단보도는 무조건 강제 무시 (카운트 유지)
                            self.get_logger().info(
                                "\033[1;36m구버전: 정지 후 첫 횡단보도 강제 무시\033[0m"
                            )
                            self.just_stopped_crosswalk = False
                            self.crosswalk_ignore = True
                            self.crosswalk_ignore_time = time.time()
                            self.count_crosswalk = 0
                        elif self.crosswalk_total_count % 2 == 0:
                            # 짝수 (2,4,6,8번) → 즉시 통과 (정지 없음)
                            self.get_logger().info(
                                "\033[1;36m짝수 횡단보도 → 통과\033[0m"
                            )
                            self.crosswalk_ignore = True
                            self.crosswalk_ignore_time = time.time()
                            self.count_crosswalk = 0
                        elif self.crosswalk_total_count == 9:
                            # 9번 횡단보도 → 주차 (park_x 감지 시 park_action 실행, 기존 로직)
                            self.get_logger().info(
                                "\033[1;35m%s\033[0m" % "9번 횡단보도 → 주차 준비"
                            )
                            self.pre_slow_down = True
                        else:
                            # 홀수 (1,3,5,7번) → 정지 로직 진입
                            self.pre_slow_down = True

                    if self.pre_slow_down:  # 홀수 횡단보도: 기존 정지 로직 유지
                        if (
                            self.crosswalk_box_height > 15
                        ):  # 30→60→50→20→15→12→15: 15가 적당한 정지 위치
                            self.count_crosswalk += 1
                            if self.count_crosswalk >= 2:
                                self.count_crosswalk = 0
                                self.start_slow_down = True
                                self.pre_slow_down = False
                                self.count_slow_down = time.time()
                        else:  # 원거리 → 아직 count 안함, 0.05m/s로만 감속 유지
                            self.count_crosswalk = 0
                else:
                    if not self.start_slow_down:
                        self.count_crosswalk = 0
                        self.pre_slow_down = False

                # deceleration processing
                # TODO 00 : 교차로 인식 시 완전 정지
                if self.start_slow_down:
                    if not self.stop:
                        self.mecanum_pub.publish(Twist())  # 횡단보도 감지하면 완전 정지
                        self.stop = True
                        self.stop_time = time.time()
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
                            self.stop = False
                            self.start_slow_down = False
                            self.crosswalk_distance = 0  # TODO 00 : 거리 초기화 → 추가
                            self.crosswalk_ignore = True  # TODO 00 : 무시 시작 → 추가
                            self.crosswalk_ignore_time = (
                                time.time()
                            )  # TODO 00 : 무시 시작 시간 → 추가
                            self.turn_guard_time = (
                                time.time()
                            )  # 실제 정지 재출발 → 우회전 가드 리셋
                            self.pre_slow_down = False
                            self.just_stopped_crosswalk = (
                                True  # 구버전: 다음 횡단보도 강제 무시
                            )
                            self.accel_ramp_active = True  # 초록불 재출발 가속 구간
                            self.accel_ramp_start_time = time.time()

                    else:
                        if (
                            time.time() - self.stop_time > 0.7
                        ):  # TODO 00 숫자 변경 = 정지 시간 변경(x.0) 1.0→0.7
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
                                self.turn_guard_time = (
                                    time.time()
                                )  # 실제 정지 재출발 → 우회전 가드 리셋
                                self.pre_slow_down = False
                                self.just_stopped_crosswalk = (
                                    True  # 구버전: 다음 횡단보도 강제 무시
                                )
                                self.accel_ramp_active = (
                                    True  # 횡단보도 재출발 가속 구간
                                )
                                self.accel_ramp_start_time = time.time()

                    if not self.stop:
                        self.set_drive_mode("slow_down")
                        twist.linear.x = self.drive_params["slow_down"]["linear_x"]

                else:
                    self.set_drive_mode("straight")  # TODO 01 : 직진 모드 전환
                    # 속도는 차선 감지 후 아래에서 설정

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
                result_image, lane_angle, lane_x, center_x = self.lane_detect(
                    binary_image, image.copy()
                )  # TODO 01 : center_x 추가
                self.get_logger().info(f"lane_x: {lane_x}")
                # TODO 02 : 우회전 디버그 로그
                self.get_logger().info(
                    f"center_x: {center_x}"
                )  # TODO 01 : 디버그 로그 추가

                # 차선이 보일 때만 직진 속도 설정 (차선 없으면 정지 유지)
                if not self.start_slow_down and lane_x is not None and lane_x > 0:
                    twist.linear.x = self.drive_params["straight"]["linear_x"]
                elif (
                    not self.start_slow_down
                    and not self.start_turn
                    and time.time() - self.turn_exit_time < 1.5
                ):  # TODO : 회전 종료 직후 1.5초간은 차선 안 보여도 강제 직진 - 교차로 구간엔 차선이 안 그려져 있어 회전 직후 잠깐 끊기는 게 정상이라 그 구간을 버티고 지나가도록 → 추가
                    twist.linear.x = self.drive_params["straight"]["linear_x"]

                self.get_logger().info(
                    f"[DEBUG] stop:{self.stop}, start_turn:{self.start_turn}, count_turn:{self.count_turn}, start_slow_down:{self.start_slow_down}"
                )  # TODO : 우회전 안되는 원인 추적용 → 추가
                if not self.start_delay and (
                    not self.stop or (self.start_slow_down and self.turn_right)
                ):  # TODO : 크리핑(stop=False) 또는 횡단보도정지+우회전표지판 조합일 때만 count_turn 허용 - turn_right_action 실행 중(start_slow_down=False)엔 차단
                    if (
                        len(center_x) >= 5
                        and center_x[0] != -1
                        and center_x[3]
                        == -1  # TODO : Box3 조건 제거 - 코너에서 Box3이 계속 살아있어 count 미도달 → Box4=Box5=-1만 요구하도록 완화
                        and center_x[4] == -1
                        and time.time() - self.turn_guard_time
                        > 1.1  # 실제 정지/우회전 완료 후 1.1s 가드 (구버전 강제무시는 타이머 리셋 안함 → 코너 직전 #2 감지 시 지연 방지)
                        and not self.pre_slow_down  # 횡단보도 접근 중 false turn 방지
                    ):
                        self.count_turn += 1  # TODO : drift 감지 제거 - 0.5m/s에서 진짜 코너 drift(88px)가 기준(70px) 초과해서 카운트 리셋되는 문제 → 수정
                        if (
                            self.count_turn > 1
                            and not self.start_turn  # 4→1: 횡단보도 직후 코너 window 2프레임으로 축소
                        ):  # TODO 01 : 10->5 (깜빡임 감안해서 낮춤)
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                            self.start_slow_down = (
                                False  # TODO 01 : 우회전 시작 시 횡단보도 플래그 리셋
                            )
                            self.stop = False  # TODO 01 : 정지 플래그 리셋
                    # else: count 유지 (감소 제거) - TODO : Box1 깜빡임으로 count가 1-2 사이 진동, 5 미도달 → 감소 없애고 누적되도록 수정
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
                    elif self.stop:  # 완전 정지 중 - count_turn만 허용, PID 명령 차단
                        self.pid.clear()
                    else:  # use PID algorithm to correct turns on a straight road
                        if not self.start_turn:
                            self.pid.SetPoint = 170  # TODO 도로 중앙값 조절 (245->230->250->230->240->250->270->255->190->180->170)
                            if (
                                self.crosswalk_ignore
                                and time.time() - self.crosswalk_ignore_time < 1.0
                            ):  # 1.0초간 PID 없이 강제 직진 - 정지 후 PID 바로 재개하면 차선 줄무늬 오인식으로 좌우 비틀림 발생 (0.7s+차선보임조건 → 1.0s 무조건)
                                twist.linear.x = self.drive_params["straight"][
                                    "linear_x"
                                ]
                                self.pid.clear()
                            elif (
                                lane_x is not None and lane_x > 0
                            ):  # 차선 보일 때만 PID 적용
                                self.pid.update(lane_x)
                                if self.machine_type != "MentorPi_Acker":
                                    twist.angular.z = common.set_range(
                                        self.pid.output, -0.15, 0.15
                                    )  # TODO : 0.18->0.13->0.18->0.13->0.15, SetPoint=240 + 클램프=0.15 (4cm 왼쪽 보정용)
                                else:
                                    twist.angular.z = (
                                        twist.linear.x
                                        * math.tan(
                                            common.set_range(self.pid.output, -0.1, 0.1)
                                        )
                                        / 0.145
                                    )
                            else:
                                self.pid.clear()
                        else:
                            if self.machine_type == "MentorPi_Acker":
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    if (
                        self.pre_slow_down
                        and not self.start_turn
                        and self.crosswalk_box_height > 13
                    ):
                        twist.linear.x = (
                            self.slow_down_speed
                        )  # cw_h>13일 때 감속: 정지(15) 직전만 감속, 너무 이른 감속 방지 (10→13)
                    if self.accel_ramp_active and not self.start_turn:
                        elapsed = time.time() - self.accel_ramp_start_time
                        if elapsed >= 0.15:
                            self.accel_ramp_active = False
                        else:
                            twist.linear.x = (elapsed / 0.15) * self.drive_params[
                                "straight"
                            ]["linear_x"]
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()
                    # TODO 01 : start_turn 탈출 조건 else 밖으로 이동 (차선 없어도 탈출 가능)
                if self.start_turn:
                    turn_elapsed = time.time() - self.start_turn_time_stamp
                    # TODO : 최소 1초는 회전 유지 (시작 직후 깜빡임으로 바로 탈출 방지)
                    if (
                        turn_elapsed > 0.8
                        and len(center_x) >= 4
                        and center_x[3] != -1
                        and center_x[3] < 180
                    ):  # <180: 실제 차선(x≈50~100), 횡단보도 stripe(x≈229) 제외
                        self.count_turn_exit += 1  # TODO : 박스5->박스4 기준 변경 - 박스5는 끝까지 안 돌아오는 경우가 많아서 너무 늦게 탈출/영영 탈출 못함
                        if (
                            self.count_turn_exit > 0
                        ):  # 1프레임만 보여도 탈출 - 1코너에서 1프레임만 보이다 소실, count>1이면 못 빠져나와 1.8s 과회전
                            self.start_turn = False
                            self.count_turn = 0
                            self.count_turn_exit = 0
                            self.turn_exit_time = (
                                time.time()
                            )  # TODO : 회전 종료 시점 기록 (교차로 구간 차선 끊김 대비 강제 직진용) → 추가
                    else:
                        self.count_turn_exit = max(0, self.count_turn_exit - 1)

                    if (
                        turn_elapsed > 1.1
                    ):  # 2.6→1.8→1.3→1.1초: angular_z=-1.2로 올리면서 시간 줄여 과회전 방지 (1.2×1.1=1.32rad≈76°)
                        self.start_turn = False
                        self.count_turn = 0  # TODO 01 : 카운트 동시 리셋
                        self.count_turn_exit = 0
                        self.turn_exit_time = (
                            time.time()
                        )  # TODO : 회전 종료 시점 기록 (교차로 구간 차선 끊김 대비 강제 직진용) → 추가

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

                # RGB 등화 제어 (LED1: 정지등/주행등, LED2: 방향지시등)
                led1 = (255, 0, 0) if (self.stop or self.pre_slow_down) else (0, 255, 0)
                if self.turn_right or self.start_turn:
                    blink_on = (time.time() % 0.5) < 0.25
                    led2 = (255, 200, 0) if blink_on else (0, 0, 0)
                else:
                    led2 = led1  # 방향지시 없을 때 LED2도 LED1과 동일
                msg = RGBStates()
                msg.states = [
                    RGBState(index=1, red=led1[0], green=led1[1], blue=led1[2]),
                    RGBState(index=2, red=led2[0], green=led2[1], blue=led2[2]),
                ]
                self.rgb_pub.publish(msg)

                # GPIO 전구 제어
                is_stop = self.stop or self.pre_slow_down
                GPIO.output(PIN_GREEN, LED_OFF if is_stop else LED_ON)
                GPIO.output(PIN_RED, LED_ON if is_stop else LED_OFF)
                if self.turn_right or self.start_turn:
                    gpio_blink = (time.time() % 0.6) < 0.3
                    GPIO.output(PIN_YELLOW_LEFT, LED_ON if gpio_blink else LED_OFF)
                    GPIO.output(PIN_YELLOW_RIGHT, LED_ON if gpio_blink else LED_OFF)
                else:
                    GPIO.output(PIN_YELLOW_LEFT, LED_OFF)
                    GPIO.output(PIN_YELLOW_RIGHT, LED_OFF)

            else:
                blink_on = (time.time() % 0.6) < 0.3  # 대기 중 교대 점멸 (50% 듀티)
                msg = RGBStates()
                msg.states = [
                    RGBState(index=1, red=255 if blink_on else 0, green=0, blue=0),
                    RGBState(index=2, red=0 if blink_on else 255, green=0, blue=0),
                ]
                self.rgb_pub.publish(msg)
                # GPIO 대기 중 빨간불 교대 점멸 (RGB와 동일 주기)
                GPIO.output(PIN_RED, LED_ON if blink_on else LED_OFF)
                GPIO.output(PIN_GREEN, LED_OFF)
                GPIO.output(PIN_YELLOW_LEFT, LED_OFF)
                GPIO.output(PIN_YELLOW_RIGHT, LED_OFF)
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
            if (
                time.time() - self.crosswalk_last_seen_time > 0.3
            ):  # TODO : 깜빡임 방지, 0.3초간 이전값 유지
                self.crosswalk_distance = 0
        else:
            min_distance = 0
            min_box_height = 0
            found_traffic_light = False
            for i in self.objects_info:
                class_name = i.class_name
                center = (
                    int((i.box[0] + i.box[2]) / 2),
                    int((i.box[1] + i.box[3]) / 2),
                )

                if class_name == "crosswalk":
                    box_height = abs(i.box[1] - i.box[3])  # TODO 00 : 박스 높이 계산
                    box_width = abs(i.box[0] - i.box[2])  # TODO : 박스 너비 계산 → 추가
                    self.crosswalk_raw_last_seen_time = (
                        time.time()
                    )  # TODO : 필터 통과 여부 상관없이 본 시간 기록 → 추가
                    self.get_logger().info(
                        f"[crosswalk raw] box_height: {box_height}, box_width: {box_width}"
                    )  # TODO : 필터링 전 모든 크기 로그 → 튜닝용
                    if (
                        10 < box_height < 180
                        and 150 < box_width
                        and box_width > 2 * box_height
                    ):  # height 15->10: 먼 거리 감지용, width>150: 노이즈 차단, width>2*height: 정사각형 오인식 차단 (실제 횡단보도는 가로>>세로)
                        if (
                            center[1] > min_distance
                        ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                            min_distance = center[1]
                            min_box_height = box_height
                            self.get_logger().info(
                                f"crosswalk box_height: {box_height}, box_width: {box_width}"
                            )  # TODO 00 : 높이/너비 로그 → 추가
                elif class_name == "right":  # obtain the right turning sign
                    sign_h = abs(i.box[1] - i.box[3])
                    sign_w = abs(i.box[0] - i.box[2])
                    self.get_logger().info(
                        f"[right raw] box_height: {sign_h}, box_width: {sign_w}"
                    )
                    self.count_right += 1
                    self.count_right_miss = 0
                    if (
                        self.count_right >= 5
                    ):  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == "park":
                    sign_h = abs(i.box[1] - i.box[3])
                    sign_w = abs(i.box[0] - i.box[2])
                    box_area = sign_w * sign_h  # TODO 00 : 박스 크기 계산
                    self.get_logger().info(
                        f"[park raw] box_height: {sign_h}, box_width: {sign_w}, box_area: {box_area}"
                    )  # TODO : park 오인식 추적용 → 추가
                    if (
                        box_area > 5000
                    ):  # TODO 00 : 오인식 방지 1000→5000 (원거리 오인식 차단, 약 70×70px 이상만 허용)
                        self.park_x = center[0]
                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light
                    sign_h = abs(i.box[1] - i.box[3])
                    sign_w = abs(i.box[0] - i.box[2])
                    self.get_logger().info(
                        f"[{class_name} raw] box_height: {sign_h}, box_width: {sign_w}"
                    )
                    self.traffic_signs_status = i
                    found_traffic_light = True

            if not found_traffic_light:
                self.traffic_signs_status = None

            all_classes = [i.class_name for i in self.objects_info]
            self.get_logger().info(
                f"\033[1;32m[YOLO] {all_classes} | cw_dist:{self.crosswalk_distance} cw_h:{self.crosswalk_box_height} ignore:{self.crosswalk_ignore} total#{self.crosswalk_total_count}\033[0m"
            )
            if (
                self.crosswalk_ignore
            ):  # TODO : 정지 직후 보호기간 동안은 횡단보도 거리값 추적 자체를 무시 (두번째 횡단보도 영향 차단)
                self.crosswalk_distance = 0
                self.crosswalk_box_height = 0
            elif (
                min_distance > 0
            ):  # TODO : 횡단보도 발견 시 시간 기록, 못 찾으면 0.3초간 이전값 유지
                self.crosswalk_distance = min_distance
                self.crosswalk_box_height = min_box_height
                self.crosswalk_last_seen_time = time.time()
            elif time.time() - self.crosswalk_last_seen_time > 0.3:
                self.crosswalk_distance = 0
                self.crosswalk_box_height = 0


def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
