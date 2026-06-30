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
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)
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

        # self.park_action()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

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

    def image_callback(self, ros_image):  # callback target checking
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)

    def param_init(self):
        self.start = False
        self.enter = False

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
        self.turn_right_speed = 0.1  # 우회전 시 전진 속도
        self.turn_right_angular = (
            -0.7
        )  # 우회전 각속도(음수=우회전). 절댓값 ↑ = 더 급하게 돔
        self.turn_right_duration = (
            1.5  # 우회전 동작 시간(초). 덜 돌면 ↑, 과하게 돌면 ↓ (90도 맞춰 튜닝)
        )
        self.have_turn_right = False

        self.count_park = 0
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign
        self.park_area = 0  # obtain the area of the parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing

        # [횡단보도 정지] 규칙: 횡단보도 앞 반드시 정지 후 출발
        self.crosswalk_stop_dist = 200  # crosswalk_distance가 이 값보다 크면(가까우면) 정지. 값↑=더 가까이서 멈춤.
        self.crosswalk_stop_duration = 1.0  # 정지 유지 시간(초)
        self.crosswalk_stopping = False  # 현재 횡단보도에서 정지 중인가
        self.crosswalk_stop_time = 0  # 정지 시작 시각
        self.crosswalk_passed = False  # 이번 횡단보도 통과 처리 완료(중복 정지 방지)

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.3  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        # ===== [1단계] 차선추종(Lane Keeping) 튜닝 파라미터 =====
        self.lane_setpoint = 130
        self.turn_threshold = 200
        self.turn_angular_z = -0.8
        self.angular_z_limit = 0.1
        self.lane_deadband = 0
        self.turn_confirm_count = 5
        self.turn_recover_time = 2.0

        self.traffic_signs_status = None  # record the state of the traffic lights

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

    def shutdown(self):  # press 'ctrl+c' to close the program
        self.is_running = False

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

    # 우회전 동작
    def turn_right_action(self):
        twist = Twist()
        twist.linear.x = self.turn_right_speed  # 전진하며
        twist.angular.z = self.turn_right_angular  # 우회전
        self.mecanum_pub.publish(twist)
        time.sleep(self.turn_right_duration)  # 90도 맞춰 튜닝
        self.mecanum_pub.publish(Twist())  # 정지
        self.doing_turn_right = False  # 차선추종 재개
        # self.have_turn_right = True

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
            twist = Twist()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

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

                twist.linear.x = self.normal_speed  # 기본 직진 속도

                # 우회전 동작 처리 (규칙: 우회전 표지판 인식 후 횡단보도 정지 → 우회전 수행)

                if self.turn_right and not self.doing_turn_right:
                    self.turn_right = False
                    self.doing_turn_right = True
                    self.turn_right_action()

                # 횡단보도 정지 처리 (규칙: 횡단보도 앞 반드시 정지 후 출발, 신호등 빨강이면 계속 정지)
                else:
                    if (
                        self.crosswalk_distance > self.crosswalk_stop_dist
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
                            time.sleep(0.1)
                        else:
                            self.stop = True  # 정지 유지
                            self.mecanum_pub.publish(Twist())
                    else:
                        # 횡단보도에서 멀어지면(사라지면) 다음 횡단보도를 위해 상태 리셋
                        if self.crosswalk_distance < 70:
                            self.crosswalk_passed = False
                            self.crosswalk_stopping = False
                        self.stop = False

                self.get_logger().info(
                    "parking trigger: park_x=%s park_area=%s count_park=%s"
                    % (
                        self.park_x,
                        self.park_area,
                        self.count_park,
                    )
                )

                # Parking Process
                if not self.start_park:
                    if self.park_x > 0 and self.park_area > 1000:
                        self.count_park += 1
                    if self.count_park >= 10:
                        self.start_park = True
                        self.stop = True
                        self.count_park = 0
                        self.mecanum_pub.publish(Twist())
                        threading.Thread(target=self.park_action, daemon=True).start()

                # line following processing
                result_image, lane_angle, lane_x_far, lane_x = self.lane_detect(
                    binary_image, image.copy()
                )
                if (
                    lane_x >= 0
                    and not self.stop
                    and not self.doing_turn_right
                    and not self.start_park
                ):  # 우회전 동작, 주차 중엔 차선추종 양보
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
                    self.mecanum_pub.publish(twist)

                # TODO : 우회전 직후 수행할 기능 (차선 정렬 및 오른쪽 라인 detect)
                # elif self.have_turn_right:
                # twist.linear.x = self.normal_speed
                # self.mecanum_pub.publish(twist)
                # time.sleep(6)
                # threading.Thread(target=self.park_action, daemon=True).start()
                # self.have_turn_right = False

                else:
                    # TODO - 차선 인식 실패 시 정지 or 감속 or 회전 등 처리
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

    def is_valid_crosswalk(self, box):
        width = abs(box[2] - box[0])
        height = abs(box[3] - box[1])
        area = width * height
        aspect_ratio = width / height if height > 0 else 0

        return width >= 80 and height >= 20 and area >= 3500 and aspect_ratio >= 1.8

    # Obtain the target detection result
    def get_object_callback(self, msg):
        valid_objects = []
        for i in msg.objects:
            if i.class_name == "crosswalk" and not self.is_valid_crosswalk(i.box):
                continue
            valid_objects.append(i)

        self.objects_info = valid_objects

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
                        self.count_right >= 5
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
    main()
