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
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        # [튜닝] 차선추종 직선 보정용 PID 게인 (P, I, D)
        #   - P(0.4): 차선 중심에서 벗어난 만큼 즉시 조향. 반응이 둔하면 ↑, 좌우로 흔들리면 ↓
        #   - I(0.0): 정상상태 오차 누적 보정. 한쪽으로 치우쳐 달리면 아주 조금만 ↑ (와인드업 주의)
        #   - D(0.05): 흔들림 감쇠(댐핑). 직선에서 좌우 진동이 크면 ↑
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()  
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")
        # [추가] 우회전 후 주차장까지 갈 때 쓰는 '우측 라인' 전용 검출기.
        #   기본 lane_detect는 ROI가 좌측 절반(x 0~320)만 봐서 좌측 라인을 추종함.
        #   주차 경로(중앙 복도)에선 좌측에 교차로 갈림길이 있어 좌측 라인을 따라가면 이탈 →
        #   우측 절반(x 320~640) ROI로 우측 라인을 보게 함. near_x는 우측 절반 내 0~320 좌표.
        self.lane_detect_right = lane_detect.LaneDetector("yellow")
        self.lane_detect_right.set_roi(((338, 360, 320, 640, 0.7), (292, 315, 320, 640, 0.2), (248, 270, 320, 640, 0.1)))

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)
        
        if 1:#self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        #self.park_action() 
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign
        self.park_area = 0       # 주차 표지판 박스 면적(px^2). 클수록 표지판에 가까움(거리 지표)
        self.park_min_area = 1000  # 이 면적 이상일 때만 주차 시작(표지판에 충분히 가까움). 너무 멀리서 주차하면 ↑, 가까이서도 안하면 ↓
                                   #   (1500→1000: 실측 park_area가 320~1036이라 1500은 절대 안 걸림. 로그 보고 정밀 조정)
        self.park_forward_time = 1.0  # 주차 시작 전 똑바로 직진하는 시간(초). 주차칸 앞까지 더 가서 주차하도록
        self.going_to_park = False  # 우회전 완료 후 주차장까지 가는 중. 이 동안은 '우측 라인' 추종(좌측 갈림길 이탈 방지)
        self.park_lane_setpoint = 190  # 우측 라인 추종 목표 x(우측 절반 0~320 좌표). 로그(right_x) 보고 튜닝. 우측 라인을 이 값에 맞춰 유지

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        # [우회전 동작] 우회전 표지판 인식(turn_right) 후 횡단보도 정지 → 우회전 수행.
        self.doing_turn_right = False    # 우회전 동작 수행 중(이 동안 차선추종은 제어 양보)
        self.turn_right_speed = 0.15     # 우회전 시 전진 속도
        self.turn_right_angular = -0.5   # 우회전 각속도(음수=우회전). 절댓값 ↑ = 더 급하게 돔
        self.turn_right_forward_time = 1.5  # 우회전 '전' 똑바로 직진하는 시간(초). 너무 일찍 꺾이면 ↑
        self.turn_right_duration = 3.0   # 우회전 동작 시간(초). 덜 돌면 ↑, 과하게 돌면 ↓ (90도 맞춰 튜닝)

        self.last_park_detect = False
        self.count_park = 0  
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        # [횡단보도 정지] 규칙: 횡단보도 앞 반드시 정지 후 출발. (기존 코드는 감속만 했고
        #   slow_down_speed가 normal_speed와 같아 감속조차 안 보였음)
        self.crosswalk_stop_dist = 350      # crosswalk_distance가 이 값보다 크면(가까우면) 정지. 값↑=더 가까이서 멈춤.
                                            #   (150→350: 멀리서 미리 멈춰 신호등을 못 보던 문제 해결)
        self.crosswalk_min_area = 3000      # 횡단보도 박스 면적이 이 값 이상일 때만 인정. 바닥 허연 부분을
                                            #   한프레임씩 횡단보도로 오검출하던 것 제거. 진짜 횡단보도 미인식이면 ↓
        self.crosswalk_stop_duration = 2.0  # 정지 유지 시간(초)
        self.crosswalk_stopping = False     # 현재 횡단보도에서 정지 중인가
        self.crosswalk_stop_time = 0        # 정지 시작 시각
        self.crosswalk_passed = False       # 이번 횡단보도 통과 처리 완료(중복 정지 방지)

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.15  # normal driving speed (완주시간 가산점 위해 0.1→0.15 상향. 코너링 불안정하면 ↓)
        self.slow_down_speed = 0.1  # slowing down speed

        # ===== [1단계] 차선추종(Lane Keeping) 튜닝 파라미터 =====
        # 기존에 main() 안에 하드코딩되어 있던 값들을 여기로 모음. 동작은 기존과 동일.
        # 실차 주행 후 아래 값들만 조정하면 차선추종 성향을 바꿀 수 있음.
        #
        # lane_setpoint: 로봇이 차선 중앙일 때의 목표 lane_x 픽셀값(PID 목표점).
        #   값을 키우면 차가 더 오른쪽, 줄이면 더 왼쪽으로 붙어 주행함.
        self.lane_setpoint = 130
        # turn_threshold: 급회전 진입 임계값. lane_x가 이 값보다 크면 코너로 판단해 고정 회전.
        #   코너를 못 돌고 직진해 이탈하면 ↓, 직선에서 불필요하게 꺾이면 ↑.
        #   코너를 너무 빨리/일찍 도는 증상 → ↑ (진입 늦춤). lane_setpoint(130)보다 충분히 커야 함.
        #   캘리브레이션 개선 후 150→180→200 으로 단계적 상향.
        #   ※ 아래 main()의 lane_x 로그로 직선/코너 실제값을 보고 정밀 조정할 것.
        self.turn_threshold = 200
        # turn_angular_z: 급회전 구간의 고정 회전 각속도(rad/s, 음수=우회전).
        #   코너 안쪽으로 파고들면 절댓값 ↓, 못 돌고 바깥으로 나가면 절댓값 ↑.
        #   [복원] 실차 결과 초기값이 더 안정적이라 -0.38 → -0.45(원래)로 되돌림.
        self.turn_angular_z = -0.45
        # angular_z_limit: 직선 PID 보정 출력의 최대 회전 각속도(rad/s) 제한.
        #   직선에서 좌우 흔들림(진동)이 크면 ↓.
        self.angular_z_limit = 0.1
        # lane_deadband: 직선 보정 데드밴드(픽셀). |lane_x - lane_setpoint|가 이 값 이내면
        #   조향하지 않고 직진(미세 진동 제거). 0이면 비활성(원래 동작).
        #   [복원] 효과가 뚜렷하지 않아 6 → 0(비활성)으로 되돌림. 필요시 4~8로 재시도 가능.
        self.lane_deadband = 0
        # [3단계] turn_confirm_count: 회전 진입 확정에 필요한 연속 검출 프레임 수.
        #   값 ↑ 이면 코너를 더 신중히(늦게) 진입해 오검출 방지, 값 ↓ 이면 민감하게 빨리 진입.
        self.turn_confirm_count = 5
        # turn_recover_time: 회전 시작 후 PID 직선보정으로 복귀하기까지의 유지 시간(초).
        #   회전 직후 치우치면 ↓(예: 1.0), 회전이 덜 끝난 채 흔들리면 ↑.
        #   [복원] 1.5 → 2.0(원래)으로 되돌림.
        self.turn_recover_time = 2.0

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
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            camera = 'depth_cam'#self.get_parameter('depth_camera_name').value
            self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image' , self.image_callback, 1)
            self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
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
        if self.machine_type == 'MentorPi_Mecanum':
            # [추가] 주차 전 똑바로 1초 직진 — 주차칸 앞까지 더 들어간 뒤 옆으로 주차(우회전 동작과 동일 패턴)
            twist = Twist()
            twist.linear.x = self.normal_speed
            self.mecanum_pub.publish(twist)
            time.sleep(self.park_forward_time)
            # 옆으로 이동(메카넘 횡이동)하여 주차칸에 진입
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38/0.2)
        
        # 안 쓰는 모델
        elif self.machine_type == 'MentorPi_Acker':
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
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
            time.sleep(0.65/0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
        self.mecanum_pub.publish(Twist())

    # 우회전 동작 (우회전 표지판 + 횡단보도 정지 후 실행). park_action처럼 별도 스레드로 동작.
    def turn_right_action(self):
        # 1단계: 회전 없이 똑바로 직진 — 교차로 안쪽까지 더 들어간 뒤 돌게 함(일찍 꺾임 방지)
        twist = Twist()
        twist.linear.x = self.turn_right_speed
        twist.angular.z = 0.0
        self.mecanum_pub.publish(twist)
        time.sleep(self.turn_right_forward_time)
        # 2단계: 전진하며 우회전
        twist.angular.z = self.turn_right_angular
        self.mecanum_pub.publish(twist)
        time.sleep(self.turn_right_duration)       # 90도 맞춰 튜닝
        self.mecanum_pub.publish(Twist())          # 정지
        self.doing_turn_right = False              # 차선추종 재개
        self.going_to_park = True                  # 이후 주차장까지는 직진만(좌측 라인 이탈 방지)

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
                self.get_logger().info('\033[1;33mcrosswalk=%d stopping=%s passed=%s sign=%s\033[0m' % (
                    self.crosswalk_distance, self.crosswalk_stopping, self.crosswalk_passed,
                    self.traffic_signs_status.class_name if self.traffic_signs_status is not None else 'none'))

                twist.linear.x = self.normal_speed  # 기본 직진 속도

                if self.crosswalk_distance > self.crosswalk_stop_dist and not self.crosswalk_passed:
                    # 횡단보도가 충분히 가까움 → 정지 단계
                    if not self.crosswalk_stopping:
                        self.crosswalk_stopping = True
                        self.crosswalk_stop_time = time.time()  # 정지 시작 시각 기록
                    # 신호등이 빨강이면 계속 정지, 빨강이 아니면(초록/없음) 정해진 시간 정지 후 통과 허용
                    is_red = (self.traffic_signs_status is not None and self.traffic_signs_status.class_name == 'red')
                    stopped_enough = (time.time() - self.crosswalk_stop_time) > self.crosswalk_stop_duration
                    if stopped_enough and not is_red:
                        self.crosswalk_passed = True   # 통과 허용 → 이후 차선추종으로 진행
                        self.crosswalk_stopping = False
                        self.stop = False
                        # [우회전] 우회전 표지판을 본 상태(turn_right)면, 정지 후 우회전 동작 실행
                        if self.turn_right and not self.doing_turn_right:
                            self.turn_right = False
                            self.doing_turn_right = True
                            threading.Thread(target=self.turn_right_action, daemon=True).start()
                    else:
                        self.stop = True               # 정지 유지
                        self.mecanum_pub.publish(Twist())
                else:
                    # 횡단보도에서 멀어지면(사라지면) 다음 횡단보도를 위해 상태 리셋
                    if self.crosswalk_distance < 70:
                        self.crosswalk_passed = False
                        self.crosswalk_stopping = False
                    self.stop = False

                # [수정] 주차 표지판 검출 시 처리. 기존엔 crosswalk_distance에만 의존해
                #   표지판을 멀리서 보기만 해도 주차했음 → 표지판 박스 면적(park_area=거리지표)으로 게이트.
                # [디버그 로그] 주차 표지판 보일 때 면적 출력 → park_min_area 튜닝용.
                if 0 < self.park_x:
                    self.get_logger().info('\033[1;35mpark_x=%d park_area=%d (min=%d)\033[0m' % (
                        self.park_x, self.park_area, self.park_min_area))
                if 0 < self.park_x and self.park_area > self.park_min_area:
                    # 표지판에 충분히 가까움 → 감속하며 주차 준비
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park:  # 주차 시작 (표지판이 가까워 면적 임계 통과)
                        self.count_park += 1
                        if self.count_park >= 8:  # 연속 8프레임 가까우면 주차 시작 (15→8: 검출이 드문드문이라 완화)
                            self.mecanum_pub.publish(Twist())
                            self.start_park = True
                            self.stop = True
                            self.going_to_park = False  # 주차 시작하므로 직진 모드 종료
                            threading.Thread(target=self.park_action).start()
                else:
                    self.count_park = 0

                # line following processing
                # [핵심수정] 회전/보정 판단을 '가까운 ROI 기준'(near)으로 변경.
                #   기존 lane_x는 max_center_x(=far)로, 먼 ROI가 앞쪽 코너를 미리 봐서 회전이 너무 일찍 트리거됐음.
                #   → lane_x 에 near 값을 받아 이후 로직(회전 threshold, PID)은 그대로 두고 판단 기준만 바꿈.
                #   되돌리려면 lane_x_far 를 lane_x 로 받으면 기존 동작.
                result_image, lane_angle, lane_x_far, lane_x = self.lane_detect(binary_image, image.copy())
                # [튜닝 로그] 필요시 주석 해제. near=회전판단 기준값, far=기존 max값.
                # self.get_logger().info('\033[1;36mlane_x(near)=%d  far=%d  (turn_threshold=%d)\033[0m' % (lane_x, lane_x_far, self.turn_threshold))
                if (self.going_to_park or self.park_x > 0) and not self.stop:
                    # [추가] 우회전 후(going_to_park) 또는 주차 표지판이 보일 때(park_x>0)는 '우측 라인'을 PID로 추종.
                    #   좌측엔 중앙 교차로 갈림길이 있어 기존(좌측) 차선추종은 좌측 길로 이탈했음.
                    #   park_x>0 조건 덕분에 어떻게 도달했든(실주행/수동) 주차 표지판만 보이면 접근 모드로 들어감.
                    #   우측 절반 ROI 검출기로 우측 라인 x(right_x)를 구해 park_lane_setpoint에 맞춤.
                    #   우측 라인이 안 보이면(-1) 직진 폴백. 주차 표지판이 가까워지면 위 주차 블록에서 종료.
                    _, _, _, right_x = self.lane_detect_right(binary_image, result_image)
                    self.get_logger().info('\033[1;34mgoing_to_park right_x=%d (setpoint=%d)\033[0m' % (
                        right_x, self.park_lane_setpoint))
                    twist.linear.x = self.normal_speed
                    if right_x >= 0:
                        self.pid.SetPoint = self.park_lane_setpoint
                        self.pid.update(right_x)
                        twist.angular.z = common.set_range(self.pid.output, -self.angular_z_limit, self.angular_z_limit)
                    else:
                        self.pid.clear()
                        twist.angular.z = 0.0  # 우측 라인 미검출 시 직진 유지
                    self.mecanum_pub.publish(twist)
                elif lane_x >= 0 and not self.stop and not self.doing_turn_right:  # 우회전 동작 중엔 차선추종 양보
                    if lane_x > self.turn_threshold:  # [튜닝] 급회전 진입 임계값 (param_init의 turn_threshold)
                        self.count_turn += 1
                        if self.count_turn > self.turn_confirm_count and not self.start_turn:  # [3단계] 회전 진입 확정 (param_init의 turn_confirm_count)
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != 'MentorPi_Acker':
                            twist.angular.z = self.turn_angular_z  # [튜닝] 고정 회전 각속도 (param_init의 turn_angular_z)
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0
                        if time.time() - self.start_turn_time_stamp > self.turn_recover_time and self.start_turn:  # [3단계] 회전 후 PID 복귀까지 유지 시간 (param_init의 turn_recover_time)
                            self.start_turn = False
                        if not self.start_turn:
                            self.pid.SetPoint = self.lane_setpoint  # [튜닝] 차선 중앙 목표점 (param_init의 lane_setpoint)
                            # [2단계] 데드밴드: 차선 오차가 lane_deadband 이내면 조향하지 않고 직진.
                            #   프레임마다 1~2px씩 떨리는 측정 노이즈로 인한 미세 진동(꼬물거림)을 제거함.
                            if abs(lane_x - self.lane_setpoint) < self.lane_deadband:
                                self.pid.clear()  # PID 내부 상태 초기화로 데드밴드 이탈 시 튐 방지
                                twist.angular.z = 0.0
                            else:
                                self.pid.update(lane_x)
                                if self.machine_type != 'MentorPi_Acker':
                                    twist.angular.z = common.set_range(self.pid.output, -self.angular_z_limit, self.angular_z_limit)  # [튜닝] 출력 제한 (param_init의 angular_z_limit)
                                else:
                                    twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -self.angular_z_limit, self.angular_z_limit)) / 0.145
                        else:
                            if self.machine_type == 'MentorPi_Acker':
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
        # [수정] 주차 표지판은 매 프레임 새로 판단(사라지면 0으로). 멀리서 한 번 본 값이 남아 오작동하던 문제 방지.
        self.park_x = -1
        self.park_area = 0
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                
                if class_name == 'crosswalk':
                    # [수정] 바닥 허연 자국을 횡단보도로 오검출(한프레임씩)하는 것 방지.
                    #   박스 면적이 crosswalk_min_area 이상인 '충분히 큰' 검출만 인정.
                    cw_area = (i.box[2] - i.box[0]) * (i.box[3] - i.box[1])
                    self.get_logger().info('\033[1;36mcrosswalk area=%d (min=%d)\033[0m' % (cw_area, self.crosswalk_min_area))
                    if cw_area >= self.crosswalk_min_area and center[1] > min_distance:  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
                elif class_name == 'right':  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 5:  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'park':  # obtain the center coordinate of the parking sign
                    # [수정] 박스 면적 = 가로*세로. 표지판에 가까울수록 큼 → 거리 지표로 사용.
                    #   한 프레임에 park 박스가 여러 개(진짜+오검출) 잡히면 값이 큰박스↔작은박스로 튐 →
                    #   가장 큰(=가장 가까운/신뢰도 높은) 박스만 사용해 안정화.
                    area = (i.box[2] - i.box[0]) * (i.box[3] - i.box[1])
                    if area > self.park_area:
                        self.park_area = area
                        self.park_x = center[0]
                elif class_name == 'red' or class_name == 'green':  # obtain the status of the traffic light
                    self.traffic_signs_status = i
               

            self.get_logger().info('\033[1;32m%s\033[0m' % class_name)
            self.crosswalk_distance = min_distance

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

    
