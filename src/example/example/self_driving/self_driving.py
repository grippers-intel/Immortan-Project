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
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState, RGBStates, RGBState, ButtonState

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
        # [튜닝] 직진 휘청거림(뱅뱅 진동) 수정: 기존 P=0.4,D=0.05는 출력 클램프(±angular_z_limit=0.25)
        #   대비 수십 배 커서 오차 1px에도 최대 조향으로 포화 → 좌우로 계속 튕김.
        #   비례제어가 되도록 대폭 낮춤. error 50px→0.5(클램프 0.25, 실편차엔 여전히 풀조향),
        #   error 15px→0.15(부드럽게), error<deadband→직진. 여전히 흔들리면 P 더 ↓ 또는 deadband ↑.
        #   반대로 완만한 굽이에서 못 따라가면 P 다시 ↑.
        self.pid = pid.PID(0.01, 0.0, 0.002)
        # [② 우회전 후 복귀용 PID] 직진용 self.pid(0.01)는 휘청 방지로 아주 약함 → 회전 직후 우측라인으로
        #   당겨오는 힘이 부족. going_to_park 우측라인 추종은 더 단단한 전용 게인을 씀(파고듦 빠르게 복구).
        #   여전히 파고들면 P ↑, 우측라인 넘어 반대로 튀면 P ↓.
        self.pid_park = pid.PID(0.02, 0.0, 0.003)
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
        # [LED] 온보드 RGB LED 2개(RGB1=index1, RGB2=index2) 제어용 발행자.
        self.rgb_pub = self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        # [스위치 출발] 온보드 버튼(확장보드 key1/key2)으로 주행 시작.
        #   버튼 이벤트는 /ros_robot_controller/button 으로 발행됨(ButtonState: id=1/2, state).
        #   규칙: 스위치로 출발(자동출발 -5 감점 방지). 버튼 누르면 button_callback에서 self.start=True.
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 1)

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
            # [스위치 출발] 자동출발 금지. enter로 카메라/YOLO 구독만 준비하고 start=False로 대기.
            #   온보드 버튼을 누르면 button_callback에서 self.start=True 가 되어 주행 시작한다.
            request = SetBool.Request()
            request.data = False
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
        self.park_min_area = 1200  # 이 면적 이상일 때만 주차 시작(표지판에 충분히 가까움). 너무 멀리서 주차하면 ↑, 가까이서도 안하면 ↓
                                   #   (실측 로그 기반으로 1200 설정)
        self.park_forward_time = 1.0  # 주차 시작 전 똑바로 직진하는 시간(초). 주차칸 앞까지 더 가서 주차하도록
        self.park_forward_speed = 0.3  # 주차 전 직진 속도(순항속도와 분리!). 예전 0.3에서 잘 됐던 거리(0.3m). 라인 넘으면 ↓
        self.going_to_park = False  # 우회전 완료 후 주차장까지 가는 중. 이 동안은 '우측 라인' 추종(좌측 갈림길 이탈 방지)
        self.park_lane_setpoint = 190  # 우측 라인 추종 목표 x(우측 절반 0~320 좌표). 로그(right_x) 보고 튜닝. 우측 라인을 이 값에 맞춰 유지
        self.park_angular_limit = 0.4  # [②] 우회전 후 우측라인 복구 각속도 제한(직진용 0.25보다 큼). 파고듦 복구 힘. 너무 휘청이면 ↓

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign
        self.right_min_area = 1000  # 우회전 표지판이 이 면적 이상(가까움)일 때만 인정. 너무 일찍 켜지면 ↑, 아예 안 켜지면 ↓ (로그 보고 튜닝)

        # [LED] 화살표(직진) 표지판 인식 시 노란 LED 점멸용. go 표지판 본 뒤 일정 시간 점멸.
        self.count_go = 0
        self.go_signal_time = 0          # 직진 표지판 마지막 인식 시각
        self.go_signal_duration = 3.0    # 직진 표지판 인식 후 노란불 점멸 유지 시간(초)
        self.last_led_state = None       # 마지막으로 발행한 LED 색(변화 시에만 발행해 토픽 과다 방지)

        # [우회전 동작] 우회전 표지판 인식(turn_right) 후 횡단보도 정지 → 우회전 수행.
        self.doing_turn_right = False    # 우회전 동작 수행 중(이 동안 차선추종은 제어 양보)
        self.turn_right_speed = 0.15     # 우회전 시 전진 속도
        self.turn_right_angular = -0.5   # 우회전 각속도(음수=우회전). 절댓값 ↑ = 더 급하게 돔
        self.turn_right_forward_time = 1.1  # 우회전 '전' 똑바로 직진하는 시간(초). 너무 일찍 꺾이면 ↑
                                            #   (0.8→1.1: 조금 일찍 돌아 안쪽 라인 밟던 것 → 더 들어간 뒤 회전)
        self.turn_right_duration = 3.3   # 우회전 동작 시간(초). 덜 돌면 ↑, 과하게 돌면 ↓ (90도 맞춰 튜닝)
                                         #   (3.0→3.2→3.5→3.3: [②] 살짝 덜 돌려 '오른쪽 파고듦' 방지. 회전 후
                                         #    going_to_park 우측라인 PID가 마무리로 당겨옴. 못 돌면 ↑, 파고들면 ↓)
        # [③ 시작점 정규화] 우회전은 개방루프라 정지 위치가 매번 달라지면 도착 라인도 달라짐.
        #   최소 직진(turn_right_forward_time) 후, 횡단보도가 완전히 지나갈 때까지(거리<pass_dist) 추가 전진 →
        #   항상 '횡단보도를 막 지난 지점'에서 회전 시작 → 시작점 일정. (검출 실패 대비 타임아웃 있음)
        self.turn_right_pass_dist = 150   # crosswalk_distance가 이 값 미만이면 '횡단보도 지나감'으로 판단
        self.turn_right_forward_max = 2.6 # 정규화 전진 최대 시간(초, 타임아웃). 횡단보도 검출 실패해도 여기서 회전

        self.last_park_detect = False
        self.count_park = 0  
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign
        self.parked = False  # 주차 완료(이후 영구 정지). 주차가 마지막 미션이므로 끝나면 안 움직임

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        # [횡단보도 정지] 규칙: 횡단보도 앞 반드시 정지 후 출발. (기존 코드는 감속만 했고
        #   slow_down_speed가 normal_speed와 같아 감속조차 안 보였음)
        self.crosswalk_stop_dist = 320      # crosswalk_distance가 이 값보다 크면(가까우면) 정지. 값↑=더 가까이서 멈춤.
                                            #   (210→320: 횡단보도가 y≈300에 처음 잡혀 바로 멈춰 '약간 일찍'이던 것 →
                                            #    더 가까이(320)서 멈춤. 지나치면 ↓, 여전히 이르면 ↑)
        self.crosswalk_min_area = 1800      # 횡단보도 박스 면적이 이 값 이상일 때만 인정. 바닥 허연 부분(≈1200)은
                                            #   여전히 걸러짐. (2200→1800: 더 멀리서 미리 잡아 정지 여유 확보. 오검출 생기면 ↑)
        self.crosswalk_min_aspect = 2.0     # [종횡비 필터] 박스 가로/세로 비가 이 값 이상일 때만 인정.
                                            #   실제 횡단보도는 가로로 긴 줄무늬(가로≫세로)라 통과하고, 바닥 까진 자국은
                                            #   덩어리/정사각형이라 걸러짐. 진짜 횡단보도도 걸러지면 ↓(1.7), 오검출 남으면 ↑(2.5)
        self.crosswalk_stop_duration = 2.0  # 정지 유지 시간(초)
        self.crosswalk_approach_dist = 180  # 횡단보도가 이 거리 이상(가까워지기 시작)이면 접근 감속 시작. 값↓=더 멀리서부터 감속.
        self.crosswalk_approach_speed = 0.2 # 횡단보도 접근 중 속도(관성 오버슛↓). 여전히 지나치면 ↓, 너무 굼뜨면 ↑.
        self.crosswalk_stopping = False     # 현재 횡단보도에서 정지 중인가
        self.crosswalk_stop_time = 0        # 정지 시작 시각
        self.crosswalk_passed = False       # 이번 횡단보도 통과 처리 완료(중복 정지 방지)

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.45  # normal driving speed (0.6은 카메라 15fps로 비전제어 한계 초과→미션 실패. 0.45로 타협)
        self.corner_speed = 0.25  # 코너 직후 복귀 동안 속도(순항보다 ↓). 코너 직후 갑툭튀 횡단보도를 제때 멈추려고 (0.3→0.25)
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
        #   속도와 같은 비율로 스케일(반경 = speed/|angular|). 0.3→0.45라 -0.9→-1.35.
        #   ※ normal_speed를 바꾸면 이 값도 같은 비율로 바꿔야 함.
        self.turn_angular_z = -1.35
        # angular_z_limit: 직선 PID 보정 출력의 최대 회전 각속도(rad/s) 제한.
        #   직선에서 좌우 흔들림(진동)이 크면 ↓.
        self.angular_z_limit = 0.25
        # lane_deadband: 직선 보정 데드밴드(픽셀). |lane_x - lane_setpoint|가 이 값 이내면
        #   조향하지 않고 직진(미세 진동 제거). 0이면 비활성(원래 동작).
        #   [복원] 효과가 뚜렷하지 않아 6 → 0(비활성)으로 되돌림. 필요시 4~8로 재시도 가능.
        #   [재활성] P를 비례제어로 낮춘 것과 함께 미세 jitter 제거용으로 10 적용. 더 흔들리면 ↑(12~15).
        self.lane_deadband = 10
        # [3단계] turn_confirm_count: 회전 진입 확정에 필요한 연속 검출 프레임 수.
        #   값 ↑ 이면 코너를 더 신중히(늦게) 진입해 오검출 방지, 값 ↓ 이면 민감하게 빨리 진입.
        self.turn_confirm_count = 5
        # turn_recover_time: 회전 시작 후 PID 직선보정으로 복귀하기까지의 유지 시간(초).
        #   회전 직후 치우치면 ↓(예: 1.0), 회전이 덜 끝난 채 흔들리면 ↑.
        #   [복원] 1.5 → 2.0. 코너 직후 감속(corner_speed) 지속시간도 이 값 → 2.5로 늘려 코너 직후
        #   갑툭튀 횡단보도를 느린 상태로 만나 제때 멈추게 함.
        self.turn_recover_time = 2.5

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0
        # [신호등] 초록불을 멀어서 못 잡아 못 출발하던 문제 → '빨강 신선도' 방식.
        #   빨강이 최근(red_hold_time 이내)에 보였으면 빨강으로 간주. 초록으로 바뀌면 빨강이 사라지고
        #   red_hold_time 후 is_red=False → 출발(초록을 직접 검출하지 않아도 됨). 빨강은 잘 잡히는 전제.
        self.red_last_seen_time = 0
        self.red_hold_time = 1.5   # 빨강 마지막 검출 후 이 시간(초)까지 빨강으로 유지. 너무 빨리 출발하면 ↑
        # [신호등 정지 트리거] 규칙: 신호등 인식하면 우선 멈춤. 횡단보도 검출이 끊겨도 '가까운 빨강'을
        #   보면 정지하도록 별도 트리거. '가까운' 판단은 박스 면적(멀리 있는 빨강엔 길 한복판서 안 멈추게).
        self.red_close_time = 0    # 가까운 빨강 마지막 검출 시각
        self.red_min_area = 800    # 빨강 박스가 이 면적 이상이면 '가까운 빨강'으로 보고 정지 트리거 (로그 보고 튜닝)

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

    def button_callback(self, msg):
        # [스위치 출발] 온보드 버튼이 눌리면 주행 시작.
        #   state=1(PRESSED 눌림), 5(CLICK 클릭) 둘 다 시작 트리거로 인정. id(key1/key2)는 무시(둘 중 아무 버튼).
        #   enter(초기화)가 끝난 뒤에만, 아직 출발 전(not start)일 때만 반응 → 주행 중 오작동 방지.
        if msg.state in (1, 5) and self.enter and not self.start:
            self.get_logger().info('\033[1;32m%s\033[0m' % ('START button pressed (id=%d) -> 주행 시작' % msg.id))
            with self.lock:
                self.start = True

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
            # [추가] 주차 전 똑바로 직진 — 주차칸 앞까지 더 들어간 뒤 옆으로 주차.
            #   [수정] normal_speed(순항)가 아니라 고정 park_forward_speed 사용 — 순항속도 올리면 전진거리가
            #   같이 늘어 주차라인을 넘어가던 문제 방지.
            twist = Twist()
            twist.linear.x = self.park_forward_speed
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
        self.parked = True  # 주차 완료 → main 루프가 이후 계속 정지 유지

    # 우회전 동작 (우회전 표지판 + 횡단보도 정지 후 실행). park_action처럼 별도 스레드로 동작.
    def turn_right_action(self):
        # 1단계: 회전 없이 똑바로 직진 — 교차로 안쪽까지 더 들어간 뒤 돌게 함(일찍 꺾임 방지)
        twist = Twist()
        twist.linear.x = self.turn_right_speed
        twist.angular.z = 0.0
        # 1-a) 최소 직진(기존 동작 보장): 오검출로 너무 일찍 회전하는 것 방지.
        t0 = time.time()
        while time.time() - t0 < self.turn_right_forward_time:
            self.mecanum_pub.publish(twist)
            time.sleep(0.03)
        # 1-b) [③ 시작점 정규화] 횡단보도가 완전히 지나갈 때까지(거리<pass_dist, 3프레임 연속) 추가 전진.
        #   정지 위치가 매번 달라도 '횡단보도를 막 지난 지점'에서 회전이 시작됨 → 시작점 일정.
        #   검출이 안 끊기거나 실패해도 turn_right_forward_max 타임아웃으로 반드시 회전으로 넘어감.
        gone = 0
        while time.time() - t0 < self.turn_right_forward_max:
            self.mecanum_pub.publish(twist)
            if self.crosswalk_distance < self.turn_right_pass_dist:
                gone += 1
                if gone >= 3:
                    break
            else:
                gone = 0
            time.sleep(0.03)
        self.get_logger().info('\033[1;41mTURN RIGHT: forward done in %.2fs (cw=%d) -> rotating\033[0m' % (
            time.time() - t0, self.crosswalk_distance))
        # 2단계: 전진하며 우회전
        twist.angular.z = self.turn_right_angular
        self.mecanum_pub.publish(twist)
        time.sleep(self.turn_right_duration)       # 90도 맞춰 튜닝(② 살짝 덜 돌림)
        self.mecanum_pub.publish(Twist())          # 정지
        self.doing_turn_right = False              # 차선추종 재개
        self.going_to_park = True                  # 이후 주차장까지는 직진만(좌측 라인 이탈 방지)

    # [LED] 두 RGB LED 색 발행. 색이 바뀔 때만 발행(점멸/상태변화 시에만).
    def publish_leds(self, c1, c2):
        state = (c1, c2)
        if state == self.last_led_state:
            return
        self.last_led_state = state
        self.get_logger().info('\033[1;32mLED L=%s R=%s\033[0m' % (c1, c2))  # [진단] LED 색 바뀔 때만
        msg = RGBStates()
        msg.states = [
            RGBState(index=1, red=int(c1[0]), green=int(c1[1]), blue=int(c1[2])),
            RGBState(index=2, red=int(c2[0]), green=int(c2[1]), blue=int(c2[2])),
        ]
        self.rgb_pub.publish(msg)

    # [LED] 현재 주행 상태에 맞춰 LED 색 결정. main 루프에서 매 프레임 호출.
    #   규칙: 주행=녹색 / 정지=빨강 / 화살표 방향=노란 점멸 / 주차완료=전체 점멸.
    def update_leds(self):
        GREEN = (0, 255, 0); RED = (255, 0, 0); YELLOW = (255, 255, 0); OFF = (0, 0, 0)
        blink = int(time.time() / 0.3) % 2 == 0  # 약 1.6Hz 점멸
        if self.parked:
            # 주차 완료 → 모든 LED 점멸
            c = YELLOW if blink else OFF
            led1 = led2 = c
        elif self.stop:
            # 정지 → 빨강
            led1 = led2 = RED
        elif self.doing_turn_right or self.turn_right:
            # 우회전 표지판 인식/회전 중 → 우측(index2) 노란 점멸
            led1 = OFF
            led2 = YELLOW if blink else OFF
        elif self.go_signal_time and (time.time() - self.go_signal_time) < self.go_signal_duration:
            # 직진 표지판 인식 → 양쪽 노란 점멸
            led1 = led2 = YELLOW if blink else OFF
        else:
            # 주행 → 녹색
            led1 = led2 = GREEN
        self.publish_leds(led1, led2)

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
                self.update_leds()  # [LED] 주행 상태에 맞춰 LED 갱신(매 프레임)
            else:
                self.publish_leds((255, 0, 0), (255, 0, 0))  # [스위치 출발] 대기 중 = 정지 상태이므로 빨강
            if self.start and self.parked:
                # [추가] 주차 완료 후엔 어떤 제어도 하지 않고 계속 정지(앞으로 새는 것 방지). 주차가 마지막 미션.
                self.mecanum_pub.publish(Twist())
            elif self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                # 횡단보도 정지 처리 (규칙: 횡단보도 앞 반드시 정지 후 출발, 신호등 빨강이면 계속 정지)
                # [디버그 로그] crosswalk=거리, stopping=정지중, passed=통과처리됨, sign=신호등상태
                self.get_logger().info('\033[1;33mcrosswalk=%d stopping=%s passed=%s sign=%s turn_right=%s doing=%s\033[0m' % (
                    self.crosswalk_distance, self.crosswalk_stopping, self.crosswalk_passed,
                    self.traffic_signs_status.class_name if self.traffic_signs_status is not None else 'none',
                    self.turn_right, self.doing_turn_right))

                twist.linear.x = self.normal_speed  # 기본 직진 속도

                # 신호등이 빨강이면 계속 정지, 빨강이 아니면(초록/없음) 정해진 시간 정지 후 통과 허용
                # [수정] '빨강 신선도' 방식: 빨강이 최근에 보였는지로 판단(초록 미검출 의존 제거).
                is_red = (time.time() - self.red_last_seen_time) < self.red_hold_time          # 모든 빨강 → 출발 막기(안전)
                red_close = (time.time() - self.red_close_time) < self.red_hold_time           # 가까운 빨강 → 정지 트리거
                # [수정] 정지 트리거: 횡단보도가 가깝거나(거리) OR 가까운 빨강을 봤을 때(규칙: 신호 인식시 우선 정지).
                #   횡단보도 검출이 끊겨도 빨강이면 멈추게 함.
                # [수정] 정지 래치(self.crosswalk_stopping): 일단 정지에 들어가면 거리(320) 근처에서
                #   crosswalk_distance가 요동쳐도(예: 454→316→337) 정지가 풀리지 않게 함. 예전엔 316처럼
                #   임계값 아래로 잠깐 떨어지면 정지가 한 프레임 풀려 차선추종이 로봇을 앞으로 밀어 '덜컥'거림.
                #   정지는 stopped_enough(2초) 후 passed=True 될 때만 해제된다.
                if (self.crosswalk_distance > self.crosswalk_stop_dist or red_close or self.crosswalk_stopping) and not self.crosswalk_passed:
                    # 횡단보도가 충분히 가까움 → 정지 단계
                    if not self.crosswalk_stopping:
                        self.crosswalk_stopping = True
                        self.crosswalk_stop_time = time.time()  # 정지 시작 시각 기록
                    stopped_enough = (time.time() - self.crosswalk_stop_time) > self.crosswalk_stop_duration
                    if stopped_enough and not is_red:
                        self.crosswalk_passed = True   # 통과 허용 → 이후 차선추종으로 진행
                        self.crosswalk_stopping = False
                        self.stop = False
                        # [우회전] 우회전 표지판을 본 상태(turn_right)면, 정지 후 우회전 동작 실행
                        #   [수정] going_to_park/start_park 중엔 실행 금지 — 주차장 부근에서 turn_right가
                        #   재무장돼 주차 중에 두 번째 우회전이 실행되어 주차를 망치던 버그 방지.
                        if self.turn_right and not self.doing_turn_right and not self.going_to_park and not self.start_park:
                            self.turn_right = False
                            self.doing_turn_right = True
                            threading.Thread(target=self.turn_right_action, daemon=True).start()
                    elif not self.start_park:          # 주차 동작 중이면 횡단보도 정지가 cmd_vel을 덮어쓰지 않게
                        self.stop = True               # 정지 유지
                        self.mecanum_pub.publish(Twist())
                else:
                    # 횡단보도에서 멀어지면(사라지면) 다음 횡단보도를 위해 상태 리셋
                    if self.crosswalk_distance < 70:
                        self.crosswalk_passed = False
                        self.crosswalk_stopping = False
                    self.stop = False

                # [추가] 횡단보도 접근 감속: 검출됐고(거리≥approach_dist) 아직 통과/정지 전이면 미리 감속해
                #   정지 명령 후 관성 오버슛을 줄인다. 특히 출발 직후 '바로 앞' 횡단보도는 늦게(가까이서)
                #   검출돼 풀속(0.45)이면 지나치므로 접근 속도를 낮춘다. (정지지점 자체는 crosswalk_stop_dist 그대로)
                #   [가드] not start_turn: 실제 코너 회전/복귀 중엔 감속 금지(고정 각속도라 감속하면 반경이
                #   좁아져 과회전·불안정). 코너 직후 갑툭튀 횡단보도는 코너-복귀 감속(아래)이 이미 처리한다.
                if (self.crosswalk_distance >= self.crosswalk_approach_dist
                        and not self.crosswalk_passed and not self.crosswalk_stopping
                        and not self.start_turn):
                    twist.linear.x = min(twist.linear.x, self.crosswalk_approach_speed)

                # [수정] 주차 표지판 검출 시 처리. 기존엔 crosswalk_distance에만 의존해
                #   표지판을 멀리서 보기만 해도 주차했음 → 표지판 박스 면적(park_area=거리지표)으로 게이트.
                # [디버그 로그] 주차 표지판 보일 때 면적 출력 → park_min_area 튜닝용.
                if 0 < self.park_x:
                    self.get_logger().info('\033[1;35mpark_x=%d park_area=%d (min=%d)\033[0m' % (
                        self.park_x, self.park_area, self.park_min_area))
                if self.going_to_park and 0 < self.park_x and self.park_area > self.park_min_area:
                    # [수정] going_to_park(우회전 이후)일 때만 주차. 출발 지점에서 park 표지판이 보여도
                    #   주차/접근 모드에 안 들어가게(다른 팀이 신호등을 치워 출발선에서 park가 보이던 문제).
                    # 표지판에 충분히 가까움 → 감속하며 주차 준비
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park:  # 주차 시작 (표지판이 가까워 면적 임계 통과)
                        self.count_park += 1
                        # [진단 로그] 카운트 진행상황 확인 → 8에 도달하는지 보기
                        self.get_logger().info('\033[1;41mPARK COUNT=%d/8 (area=%d>min)\033[0m' % (self.count_park, self.park_area))
                        if self.count_park >= 8:  # 8프레임 이상 가까우면 주차 시작
                            self.get_logger().info('\033[1;41m=== PARK START ===\033[0m')
                            self.mecanum_pub.publish(Twist())
                            self.start_park = True
                            self.stop = True
                            self.going_to_park = False  # 주차 시작하므로 직진 모드 종료
                            threading.Thread(target=self.park_action).start()
                else:
                    # [수정] 검출이 한 프레임 끊겨도 0으로 리셋하지 않고 1씩만 감소 → 드문드문 검출에 강하게.
                    #   (기존 self.count_park=0 은 한 번만 놓쳐도 처음부터 다시 세서 8을 못 채웠음)
                    if self.count_park > 0:
                        self.count_park -= 1

                # line following processing
                # [핵심수정] 회전/보정 판단을 '가까운 ROI 기준'(near)으로 변경.
                #   기존 lane_x는 max_center_x(=far)로, 먼 ROI가 앞쪽 코너를 미리 봐서 회전이 너무 일찍 트리거됐음.
                #   → lane_x 에 near 값을 받아 이후 로직(회전 threshold, PID)은 그대로 두고 판단 기준만 바꿈.
                #   되돌리려면 lane_x_far 를 lane_x 로 받으면 기존 동작.
                result_image, lane_angle, lane_x_far, lane_x = self.lane_detect(binary_image, image.copy())
                # [튜닝 로그] 필요시 주석 해제. near=회전판단 기준값, far=기존 max값.
                # self.get_logger().info('\033[1;36mlane_x(near)=%d  far=%d  (turn_threshold=%d)\033[0m' % (lane_x, lane_x_far, self.turn_threshold))
                # [진단 로그] 코너 회전 중 멈춤 추적용. 매 프레임 출력(어느 블록이 실행되든).
                #   start_turn=코너 급회전 확정 상태, lane_x(near)=회전 판단값, far=먼 ROI, thr=회전 임계값,
                #   stop=정지플래그(참이면 아래 차선/코너 블록이 통째로 스킵됨), doing=표지판 우회전 동작 중,
                #   go2park=주차경로 모드, start_park=주차동작 중.
                self.get_logger().info('\033[1;35mTURN? start_turn=%s lane_x=%d far=%d thr=%d stop=%s doing=%s go2park=%s start_park=%s\033[0m' % (
                    self.start_turn, lane_x, lane_x_far, self.turn_threshold,
                    self.stop, self.doing_turn_right, self.going_to_park, self.start_park))
                if not self.start_park and self.going_to_park and not self.stop:
                    # [수정] '우회전 이후(going_to_park)'에만 우측 라인 접근 모드. (park_x>0 조건 제거 —
                    #   출발선에서 park 표지판이 보이면 오작동하던 문제). '우측 라인'을 PID로 추종.
                    #   [중요] not self.start_park 가드: 주차 동작 시작 후엔 main 루프가 cmd_vel을 발행하면
                    #   park_action 스레드의 횡이동을 덮어써서 주차가 안 됨(직진해 표지판 지나침). 그래서 양보.
                    #   좌측엔 중앙 교차로 갈림길이 있어 기존(좌측) 차선추종은 좌측 길로 이탈했음.
                    #   park_x>0 조건 덕분에 어떻게 도달했든(실주행/수동) 주차 표지판만 보이면 접근 모드로 들어감.
                    #   우측 절반 ROI 검출기로 우측 라인 x(right_x)를 구해 park_lane_setpoint에 맞춤.
                    #   우측 라인이 안 보이면(-1) 직진 폴백. 주차 표지판이 가까워지면 위 주차 블록에서 종료.
                    _, _, _, right_x = self.lane_detect_right(binary_image, result_image)
                    self.get_logger().info('\033[1;34mgoing_to_park right_x=%d (setpoint=%d)\033[0m' % (
                        right_x, self.park_lane_setpoint))
                    # 주차 표지판이 보이면(park_x>0) 감속해 정밀 접근(지나침 방지). 아직 안 보이면 복도를 순항속도로.
                    twist.linear.x = self.slow_down_speed if self.park_x > 0 else self.normal_speed
                    if right_x >= 0:
                        # [②] 전용 PID(pid_park)로 우측라인 추종 — 회전 직후 파고듦을 단단히 복구.
                        #   클램프도 park_angular_limit(0.4)로 넓혀 회복 각속도 확보(직진용 0.25보다 큼).
                        self.pid_park.SetPoint = self.park_lane_setpoint
                        self.pid_park.update(right_x)
                        twist.angular.z = common.set_range(self.pid_park.output, -self.park_angular_limit, self.park_angular_limit)
                    else:
                        self.pid_park.clear()
                        twist.angular.z = 0.0  # 우측 라인 미검출 시 직진 유지
                    self.mecanum_pub.publish(twist)
                elif not self.start_park and lane_x >= 0 and not self.stop and not self.doing_turn_right:  # 우회전/주차 동작 중엔 차선추종 양보
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
                    # [수정] 회전 '직후 복귀' 구간에서만 감속(코너 직후 갑툭튀 횡단보도 대비).
                    #   실제 급회전 중(lane_x>threshold)엔 감속 안 함 — 감속하면 반경이 좁아져 과회전/불안정.
                    if self.start_turn and lane_x <= self.turn_threshold:
                        twist.linear.x = self.corner_speed
                    # [진단 로그] 차선추종/코너 블록이 실제로 로봇에 내보내는 속도. 코너 중 lin이 0으로
                    #   떨어지거나 ang이 안 나가면 여기서 잡힘. (이 로그가 안 찍히면 stop 때문에 블록이 스킵된 것)
                    self.get_logger().info('\033[1;32mDRIVE lin=%.2f ang=%.2f (start_turn=%s lane_x=%d)\033[0m' % (
                        twist.linear.x, twist.angular.z, self.start_turn, lane_x))
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
                    #   ① 면적 게이트: crosswalk_min_area 이상인 '충분히 큰' 검출만.
                    #   ② [종횡비 필터] 가로 > 세로 * crosswalk_min_aspect 인 '납작한 줄무늬'만 인정.
                    #      실제 횡단보도는 가로로 길고, 바닥 자국은 정사각형/덩어리라 종횡비에서 걸러짐.
                    box_w = i.box[2] - i.box[0]
                    box_h = i.box[3] - i.box[1]
                    cw_area = box_w * box_h
                    aspect = (box_w / box_h) if box_h > 0 else 0
                    self.get_logger().info('\033[1;36mcrosswalk area=%d aspect=%.1f (min_area=%d min_asp=%.1f)\033[0m' % (
                        cw_area, aspect, self.crosswalk_min_area, self.crosswalk_min_aspect))
                    if (cw_area >= self.crosswalk_min_area
                            and box_w > self.crosswalk_min_aspect * box_h  # 종횡비: 가로가 세로의 N배 이상
                            and center[1] > min_distance):  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
                elif class_name == 'go':  # [LED] 직진 화살표 표지판 → 노란 LED 점멸 트리거
                    self.count_go += 1
                    if self.count_go >= 3:
                        self.go_signal_time = time.time()
                        self.count_go = 0
                elif class_name == 'right':  # obtain the right turning sign
                    # [수정] 표지판이 '충분히 가까울 때'(박스 큼)만 카운트 → 멀리서 일찍 turn_right가 켜져
                    #   엉뚱한 앞 코너에서 깜빡이가 켜지거나 일반 코너링이 우회전을 가로채던 문제 방지.
                    right_area = (i.box[2] - i.box[0]) * (i.box[3] - i.box[1])
                    self.get_logger().info('\033[1;31mright sign area=%d (min=%d) count=%d\033[0m' % (
                        right_area, self.right_min_area, self.count_right))
                    # [수정] 우회전 이후(going_to_park)·주차 중엔 turn_right 재무장 금지(주차장서 두 번째 우회전 방지).
                    if right_area >= self.right_min_area and not self.going_to_park and not self.start_park and not self.parked:
                        self.count_right += 1
                        # (3→1: 정지 자세에서 표지판이 멀어 YOLO가 ~26프레임 중 1번만 검출 → 3회 못 채워 우회전 미발동.
                        #  가까운(area≥right_min_area) 'right'를 1번만 봐도 트리거. 멀리서 오작동은 area 게이트가 막음)
                        if self.count_right >= 1:
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
                    if class_name == 'red':
                        self.red_last_seen_time = time.time()  # [신호등] 빨강 신선도 갱신(모든 빨강 → 출발 막기)
                        red_area = (i.box[2] - i.box[0]) * (i.box[3] - i.box[1])
                        self.get_logger().info('\033[1;31mred area=%d (min=%d)\033[0m' % (red_area, self.red_min_area))
                        if red_area >= self.red_min_area:        # 가까운 빨강 → 정지 트리거 갱신
                            self.red_close_time = time.time()
               

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

    
