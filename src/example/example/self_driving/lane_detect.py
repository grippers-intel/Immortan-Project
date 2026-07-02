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
        # [Mecanum 우회전 보정용] 직진/코너 PID(self.pid)와 별개 인스턴스.
        # execute_turn_right()에서 회전 중 좌우 스트레이프(linear.y) 보정에만 사용.
        self.turn_pid = pid.PID(0.4, 0.0, 0.05)
        # param_init()이 machine_type을 참조하므로(Mecanum 여부에 따라 코너 보정 on/off)
        # 반드시 param_init()보다 먼저 설정되어야 함
        self.machine_type = os.environ.get("MACHINE_TYPE")
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        # TODO(competition, 규정7 - 화살표 인식): YOLO 모델 클래스에 화살표(직진/좌/우) 라벨이
        # 없어 "도로에 적힌 화살표 인식 후 노란 LED 점멸" 요구사항은 이 클래스 목록만으로는
        # 구현 불가. 모델 재학습 또는 별도 OpenCV 화살표 검출이 필요 - 팀 논의 필요.
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # TODO(competition, 규정3/4 추가 구현): 출발 스위치 게이팅 및 주행중 LED 상태 표시
        self.require_start_button = True
        self.start_button_id = 1  # Tips 슬라이드 기준 Button1 (RRC 보드)
        self.last_wait_reminder_time = (
            0  # "버튼 대기중" 로그 스팸 방지용(3초 간격 제한)
        )
        self.moving_led_state = (
            None  # 마지막으로 publish한 LED 상태(중복 publish 방지용)
        )

        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )
        self.result_publisher = self.create_publisher(Image, "~/image_result", 1)
        # 규정4(주행중 초록ON/빨강OFF, 정지중 반대), 규정11(주차완료 전체 점멸)에 사용
        # TODO(competition, 규정5 - 가산점 +15): 이 퍼블리셔는 보드 내장 RGB1/RGB2만 제어함.
        # "빵판 + Raspberry ROS <-> Host 통신" 요구사항(외부 LED 별도 구동)은 하드웨어 배선이
        # 정해진 뒤 별도 노드로 붙여야 하는 부분이라 여기서는 구현하지 않음.
        self.rgb_pub = self.create_publisher(
            RGBStates, "/ros_robot_controller/set_rgb", 1
        )
        # 규정3: 경연자가 로봇에 부착된 스위치(버튼)로 출발 신호를 전달해야 함
        self.create_subscription(
            ButtonState, "/ros_robot_controller/button", self.button_callback, 1
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
        # TODO(competition, 규정2/3 - 변경됨): 기존에는 여기서 곧바로 set_running(True)를 호출해
        # 버튼 입력 없이 자동 주행을 시작했음 -> "경연자가 로봇에 부착된 스위치를 이용해 출발
        # 신호를 전달한다"는 규정3을 위반하는 구조였음. require_start_button=True(기본값)이면
        # 물리 버튼(button_callback, Button1)이 눌릴 때까지 대기 상태(정지 LED)로 두고,
        # 버튼 이벤트에서 set_running(True)을 호출하도록 변경함.
        # 벤치 테스트 등으로 버튼 없이 바로 돌리고 싶으면 __init__의
        # self.require_start_button = False 로 바꿀 것.
        if self.require_start_button:
            self.update_status_led(False)  # 대기중: 정지 상태 LED (규정4)
            self.get_logger().info(
                "\033[1;33m%s\033[0m" % "waiting for start button (Button1) press..."
            )
        else:
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        # self.park_action()
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
        self.park_area = (
            0  # 주차 표지판 박스 면적(px^2). 클수록 표지판에 가까움(거리 지표)
        )
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
        self.go_signal_time = 0  # 직진 표지판 마지막 인식 시각
        self.go_signal_duration = 3.0  # 직진 표지판 인식 후 노란불 점멸 유지 시간(초)
        self.last_led_state = (
            None  # 마지막으로 발행한 LED 색(변화 시에만 발행해 토픽 과다 방지)
        )

        # [우회전 동작] 우회전 표지판 인식(turn_right) 후 횡단보도 정지 → 우회전 수행.
        self.doing_turn_right = (
            False  # 우회전 동작 수행 중(이 동안 차선추종은 제어 양보)
        )
        self.turn_right_speed = 0.15  # 우회전 시 전진 속도
        self.turn_right_angular = (
            -0.5
        )  # 우회전 각속도(음수=우회전). 절댓값 ↑ = 더 급하게 돔
        self.turn_right_forward_time = (
            1.1  # 우회전 '전' 똑바로 직진하는 시간(초). 너무 일찍 꺾이면 ↑
        )
        #   (0.8→1.1: 조금 일찍 돌아 안쪽 라인 밟던 것 → 더 들어간 뒤 회전)
        self.turn_right_duration = (
            3.3  # 우회전 동작 시간(초). 덜 돌면 ↑, 과하게 돌면 ↓ (90도 맞춰 튜닝)
        )
        #   (3.0→3.2→3.5→3.3: [②] 살짝 덜 돌려 '오른쪽 파고듦' 방지. 회전 후
        #    going_to_park 우측라인 PID가 마무리로 당겨옴. 못 돌면 ↑, 파고들면 ↓)
        # [③ 시작점 정규화] 우회전은 개방루프라 정지 위치가 매번 달라지면 도착 라인도 달라짐.
        #   최소 직진(turn_right_forward_time) 후, 횡단보도가 완전히 지나갈 때까지(거리<pass_dist) 추가 전진 →
        #   항상 '횡단보도를 막 지난 지점'에서 회전 시작 → 시작점 일정. (검출 실패 대비 타임아웃 있음)
        self.turn_right_pass_dist = (
            150  # crosswalk_distance가 이 값 미만이면 '횡단보도 지나감'으로 판단
        )
        self.turn_right_forward_max = 2.6  # 정규화 전진 최대 시간(초, 타임아웃). 횡단보도 검출 실패해도 여기서 회전

        # [Mecanum 코너 보정] 사각형 트랙 코너를 돌 때, 전진(x)+회전(z)은 위 개방루프
        # 튜닝값을 그대로 쓰되, Mecanum 휠 특유의 좌우 스트레이프(y)로 실시간 보정을 얹음.
        # 회전 중에도 카메라에 차선이 계속 보이므로, main()이 매 프레임 갱신하는
        # self.latest_lane_x를 읽어 PID로 옆걸음질 -> 안쪽으로 파고들거나 바깥으로
        # 벌어지는 걸 즉시 보정(직진용 turn_angular_z 각속도 자체를 흔들지 않아서
        # 기존에 실차로 맞춘 회전량이 깨지지 않음).
        self.latest_lane_x = -1  # main()이 매 프레임 갱신하는 최신 lane_x(공유 상태)
        # turn_right_strafe_limit: 코너 중 좌우 스트레이프 최대 속도(m/s).
        #   너무 크면 회전이 옆으로 밀려나가 코너를 벗어남. 보정이 약하면 ↑, 휘청이면 ↓.
        self.turn_right_strafe_limit = 0.12
        # turn_right_strafe_sign: 실차에서 스트레이프 방향이 반대로 나오면 -1로 뒤집을 것.
        #   (직진 PID의 lane_x 오차 -> angular.z 부호와 동일한 관례로 우선 +1 적용.
        #    첫 실차 테스트에서 코너 중 반대쪽으로 밀리면 이 값만 -1로 바꾸면 됨)
        self.turn_right_strafe_sign = 1
        # Mecanum이 아닌 모델(MentorPi_Acker 등)은 옆으로 못 움직이므로 이 보정을 끔
        self.turn_right_use_strafe_pid = self.machine_type == "MentorPi_Mecanum"

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
        self.crosswalk_stop_dist = 320  # crosswalk_distance가 이 값보다 크면(가까우면) 정지. 값↑=더 가까이서 멈춤.
        #   (210→320: 횡단보도가 y≈300에 처음 잡혀 바로 멈춰 '약간 일찍'이던 것 →
        #    더 가까이(320)서 멈춤. 지나치면 ↓, 여전히 이르면 ↑)
        self.crosswalk_min_area = (
            1800  # 횡단보도 박스 면적이 이 값 이상일 때만 인정. 바닥 허연 부분(≈1200)은
        )
        #   여전히 걸러짐. (2200→1800: 더 멀리서 미리 잡아 정지 여유 확보. 오검출 생기면 ↑)
        self.crosswalk_min_aspect = (
            2.0  # [종횡비 필터] 박스 가로/세로 비가 이 값 이상일 때만 인정.
        )
        #   실제 횡단보도는 가로로 긴 줄무늬(가로≫세로)라 통과하고, 바닥 까진 자국은
        #   덩어리/정사각형이라 걸러짐. 진짜 횡단보도도 걸러지면 ↓(1.7), 오검출 남으면 ↑(2.5)
        self.crosswalk_stop_duration = 2.0  # 정지 유지 시간(초)
        self.crosswalk_approach_dist = 180  # 횡단보도가 이 거리 이상(가까워지기 시작)이면 접근 감속 시작. 값↓=더 멀리서부터 감속.
        self.crosswalk_approach_speed = 0.2  # 횡단보도 접근 중 속도(관성 오버슛↓). 여전히 지나치면 ↓, 너무 굼뜨면 ↑.
        self.crosswalk_stopping = False  # 현재 횡단보도에서 정지 중인가
        self.crosswalk_stop_time = 0  # 정지 시작 시각
        self.crosswalk_passed = False  # 이번 횡단보도 통과 처리 완료(중복 정지 방지)

        # NOTE(competition): start_slow_down/slow_down_speed/crosswalk_length는 옛 "횡단보도+
        # 신호등 통합 감속" 로직에서 쓰던 값. main()을 규정6(횡단보도 무조건 정지)/규정9(신호등
        # '빨강 신선도') 독립 트리거로 재작성하면서 대체됨(crosswalk_stop_*, red_close_time 등
        # 참고) - 지금은 main()에서 읽지 않는 죽은 값이라 삭제해도 되지만, 혹시 몰라 남겨둠.
        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.45  # normal driving speed (0.6은 카메라 15fps로 비전제어 한계 초과→미션 실패. 0.45로 타협)
        self.corner_speed = 0.25  # 코너 직후 복귀 동안 속도(순항보다 ↓). 코너 직후 갑툭튀 횡단보도를 제때 멈추려고 (0.3→0.25)
        self.slow_down_speed = 0.1  # slowing down speed (현재 미사용, 위 NOTE 참고)

        # ===== [1단계] 차선추종(Lane Keeping) 튜닝 파라미터 =====
        # 기존에 main() 안에 하드코딩되어 있던 값들을 여기로 모음. 동작은 기존과 동일.
        # 실차 주행 후 아래 값들만 조정하면 차선추종 성향을 바꿀 수 있음.
        #
        # lane_setpoint: 로봇이 차선 중앙일 때의 목표 lane_x 픽셀값(PID 목표점).
        #   값을 키우면 차가 더 오른쪽, 줄이면 더 왼쪽으로 붙어 주행함.
        self.lane_setpoint = 130
        # turn_right_lane_setpoint: 우회전 코너 중 좌우 스트레이프 보정의 목표 lane_x.
        #   우선 직진 setpoint와 동일하게 시작. 회전 중 카메라 각도가 달라져 실제 차선
        #   위치가 다르게 보이면 트랙에서 로그 보고 별도 값으로 조정할 것.
        self.turn_right_lane_setpoint = self.lane_setpoint
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

        # NOTE(competition): traffic_signs_status/red_loss_count도 마찬가지로 옛 로직의 잔재.
        # 지금은 get_object_callback()에서 값만 채워질 뿐 main()에서 읽지 않음(대신 아래
        # red_last_seen_time/red_close_time을 씀). 다른 용도로 재사용할 계획이 없다면 정리 대상.
        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0
        # [신호등] 초록불을 멀어서 못 잡아 못 출발하던 문제 → '빨강 신선도' 방식.
        #   빨강이 최근(red_hold_time 이내)에 보였으면 빨강으로 간주. 초록으로 바뀌면 빨강이 사라지고
        #   red_hold_time 후 is_red=False → 출발(초록을 직접 검출하지 않아도 됨). 빨강은 잘 잡히는 전제.
        self.red_last_seen_time = 0
        self.red_hold_time = 1.5  # 빨강 마지막 검출 후 이 시간(초)까지 빨강으로 유지. 너무 빨리 출발하면 ↑
        # [신호등 정지 트리거] 규칙: 신호등 인식하면 우선 멈춤. 횡단보도 검출이 끊겨도 '가까운 빨강'을
        #   보면 정지하도록 별도 트리거. '가까운' 판단은 박스 면적(멀리 있는 빨강엔 길 한복판서 안 멈추게).
        self.red_close_time = 0  # 가까운 빨강 마지막 검출 시각
        self.red_min_area = 800  # 빨강 박스가 이 면적 이상이면 '가까운 빨강'으로 보고 정지 트리거 (로그 보고 튜닝)

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
                self.update_status_led(False)  # 규정4: 정지 상태 LED
        response.success = True
        response.message = "set_running"
        return response

    def button_callback(self, msg):
        # 규정3: 경연자가 로봇에 부착된 스위치(버튼)로 출발 신호를 전달.
        # [버그 수정] 1차 실차 테스트에서 버튼을 눌러도 차가 전혀 출발하지 않고
        # 로그에는 YOLO crosswalk 인식 로그(get_object_callback)만 계속 찍히는 문제가
        # 있었음. 원인은 여기가 아니라 ros_robot_controller_node.py의 pub_button_data가
        # 가진 벤더 버그였음: Board.get_button()은 이미 (key_id, 0=클릭, 1=눌림)으로
        # 단순화해서 반환하는데, pub_button_data가 이 값을 다시 원본
        # PacketReportKeyEvents(enum, 값 전부 1 이상) 기준 표로 조회해서, "클릭"에
        # 해당하는 정수 0은 항상 매칭 실패 -> ButtonState 메시지 자체가 발행되지 않고
        # "Unhandled button event: 0" 에러만 찍혔음(=버튼을 눌렀다 떼는 일반적인 클릭이
        # 통째로 씹힘). 그 파일도 함께 수정해 이제는 0/1 두 상태 모두 정상 publish됨.
        # 이 클래스에서는 그에 맞춰 기존에 잘못 가정했던 (1, 5) 대신 실제로 나올 수 있는
        # 두 값 (0=클릭 완료, 1=눌린 상태) 모두를 시작 신호로 인정하도록 수정.
        self.get_logger().info(
            "\033[1;36m%s\033[0m"
            % f"button msg received: id={msg.id}, state={msg.state}"
        )  # 진단용: 버튼 입력 자체가 도달하는지 로그로 바로 확인 가능하게 함
        if msg.id != self.start_button_id:
            return
        if msg.state not in (0, 1):
            return
        if self.enter and not self.start:
            self.get_logger().info(
                "\033[1;32m%s\033[0m" % "start button pressed -> begin driving"
            )
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

    def set_rgb(self, red, green, blue):
        # 보드 내장 RGB1/RGB2 두 개를 동일 색상으로 설정
        # TODO(competition, 규정5 - +15 가산점 미구현): 빵판을 이용한 외부 LED를
        # Raspberry ROS <-> Host 통신으로 별도 구동하는 부분은 하드웨어 배선이 정해진
        # 뒤 별도 노드/토픽으로 추가해야 함. 여기서는 보드 내장 LED만 다룸.
        msg = RGBStates()
        msg.states = [
            RGBState(index=1, red=red, green=green, blue=blue),
            RGBState(index=2, red=red, green=green, blue=blue),
        ]
        self.rgb_pub.publish(msg)

    def update_status_led(self, moving):
        # 규정4: 주행중 -> 초록 ON/빨강 OFF, 정지중 -> 초록 OFF/빨강 ON
        if self.moving_led_state == moving:
            return  # 상태 변화가 없으면 재전송하지 않음
        self.moving_led_state = moving
        if moving:
            self.set_rgb(0, 255, 0)
        else:
            self.set_rgb(255, 0, 0)

    def blink_all_rgb(self, times=6, interval=0.25, color=(255, 255, 0)):
        # 규정11: 주차 완료 의미로 모든 LED 점멸
        for _ in range(times):
            self.set_rgb(*color)
            time.sleep(interval)
            self.set_rgb(0, 0, 0)
            time.sleep(interval)
        self.moving_led_state = (
            None  # 다음 update_status_led 호출이 강제로 반영되도록 초기화
        )

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
        # 규정11: 주차 완료 의미로 모든 LED 점멸
        # TODO(competition): 주차 정밀도(규정10, -10점)는 이 고정 시간 스트레이프 동작에
        # 의존함 - 실제 주차 라인 폭에 맞게 time.sleep 값들을 트랙에서 재조정 필요.
        self.blink_all_rgb()

    def execute_turn_right(self):
        # 규정8: 우회전 표지판 인식 확정 후 우회전을 수행.
        # main()에서 self.doing_turn_right=True로 세팅한 뒤 이 스레드를 실행함.
        # [③ 시작점 정규화] 최소 직진(turn_right_forward_time) 후, 횡단보도를 완전히
        # 지날 때까지(또는 타임아웃까지) 추가 직진하여 매번 비슷한 지점에서 회전을 시작.
        twist = Twist()
        twist.linear.x = self.turn_right_speed
        self.mecanum_pub.publish(twist)
        time.sleep(self.turn_right_forward_time)

        wait_start = time.time()
        extra_timeout = max(
            0.0, self.turn_right_forward_max - self.turn_right_forward_time
        )
        while (
            self.crosswalk_distance >= self.turn_right_pass_dist
            and time.time() - wait_start < extra_timeout
        ):
            time.sleep(0.05)

        # [Mecanum 코너] 사각형 트랙 코너를 도는 구간. 전진(x)/회전(z)은 기존에 실차로
        # 맞춘 개방루프 값을 그대로 쓰고(과거 튜닝 기록: turn_right_duration 3.0→3.2→
        # 3.5→3.3 등, 이 값을 흔들면 그동안의 튜닝이 무의미해짐), Mecanum 휠 고유의
        # 좌우 스트레이프(linear.y)만 카메라의 lane_x 피드백으로 실시간 PID 보정해서
        # '오른쪽으로 파고듦' 같은 편차를 회전 도중에 바로잡는다.
        # main()의 이미지 큐(self.image_queue)는 main() 루프가 단독으로 소비 중이라
        # 여기서 또 큐를 읽으면 서로 프레임을 뺏는 경쟁이 생김 -> main()이 매 프레임
        # 갱신하는 self.latest_lane_x(공유 상태)만 읽는다.
        self.turn_pid.clear()
        turn_start = time.time()
        while time.time() - turn_start < self.turn_right_duration:
            twist = Twist()
            twist.linear.x = self.turn_right_speed
            twist.angular.z = self.turn_right_angular

            if self.turn_right_use_strafe_pid:
                lane_x = self.latest_lane_x
                if lane_x >= 0:
                    self.turn_pid.SetPoint = self.turn_right_lane_setpoint
                    self.turn_pid.update(lane_x)
                    strafe = common.set_range(
                        self.turn_pid.output,
                        -self.turn_right_strafe_limit,
                        self.turn_right_strafe_limit,
                    )
                    twist.linear.y = self.turn_right_strafe_sign * strafe

            self.mecanum_pub.publish(twist)
            time.sleep(0.03)

        self.mecanum_pub.publish(Twist())
        self.going_to_park = (
            True  # 회전 후에는 우측 라인 추종으로 주차장까지 진입(규정10 준비)
        )
        # [버그 수정] 회전이 끝나자마자 doing_turn_right=False가 되면 바로 다음
        # 프레임부터 blocked=False -> normal_speed(0.45)로 곧장 복귀했음. 회전 직후
        # 바로 횡단보도가 나타나는 구간에서는 이 속도가 너무 빨라 count_crosswalk가
        # 3프레임 채워지기 전에 crosswalk_stop_dist를 넘겨버려 정지를 놓치는 문제가
        # 있었음(실차 확인). 일반 차선 커브 직후에 이미 쓰던 감속 구간(start_turn ->
        # corner_speed, turn_recover_time초)을 재사용해서 회전 직후에도 잠시
        # 저속으로 진입 -> 횡단보도 인식에 필요한 프레임 수를 확보한다.
        # turn_recover_time(2.5s)@corner_speed(0.25)로도 여전히 못 잡으면 이 구간을
        # 우회전 전용으로 더 길게/느리게 분리(turn_right_recover_time 등)하는 것도 검토.
        self.start_turn = True
        self.start_turn_time_stamp = time.time()
        self.doing_turn_right = False

    def approach_and_park(self):
        # 규정10: 주차 표지판이 충분히 가까워지면(면적 기준) main()이 self.start_park=True로
        # 세팅한 뒤 이 스레드를 실행. 잠시 더 직진해 주차 라인 앞까지 접근한 뒤
        # park_action()(수평 스트레이프 + 규정11 LED 점멸)을 수행.
        twist = Twist()
        twist.linear.x = self.park_forward_speed
        self.mecanum_pub.publish(twist)
        time.sleep(self.park_forward_time)
        self.mecanum_pub.publish(Twist())

        self.park_action()

        self.going_to_park = False
        self.parked = True  # 주차 완료 -> 이후 main()에서 영구 정지(규정10 마지막 미션)

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

                # ---- 규정9: 신호등 정지('빨강 신선도' 방식, get_object_callback에서 갱신) ----
                red_stop = (time.time() - self.red_close_time) < self.red_hold_time

                # ---- 규정6: 횡단보도 정지(신호등 유무와 무관하게 반드시 정지 후 출발) ----
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                if not self.crosswalk_stopping and not self.crosswalk_passed:
                    if self.crosswalk_distance > self.crosswalk_stop_dist:
                        self.count_crosswalk += 1
                        if (
                            self.count_crosswalk >= 3
                        ):  # judge multiple times to prevent false detection
                            self.count_crosswalk = 0
                            self.crosswalk_stopping = True
                            self.crosswalk_stop_time = time.time()
                    else:
                        self.count_crosswalk = 0
                if (
                    self.crosswalk_passed
                    and self.crosswalk_distance < self.turn_right_pass_dist
                ):
                    self.crosswalk_passed = (
                        False  # 이번 횡단보도를 벗어남 -> 다음 횡단보도를 위해 리셋
                    )

                if self.crosswalk_stopping:
                    if (
                        time.time() - self.crosswalk_stop_time
                        < self.crosswalk_stop_duration
                    ):
                        self.stop = True
                    else:
                        self.crosswalk_stopping = False
                        self.crosswalk_passed = True
                        self.stop = False
                elif red_stop:
                    self.stop = True
                elif not self.doing_turn_right and not self.start_park:
                    self.stop = False

                # ---- 규정8: 우회전 표지판 확정 + 횡단보도를 막 지난 시점에 개방루프 우회전 실행 ----
                if (
                    self.turn_right
                    and not self.doing_turn_right
                    and not self.start_park
                    and not self.parked
                    and not self.stop
                    and self.crosswalk_passed
                ):
                    self.turn_right = False
                    self.doing_turn_right = True
                    threading.Thread(
                        target=self.execute_turn_right, daemon=True
                    ).start()

                # ---- 규정10: 주차 표지판이 충분히 가까우면(면적) 주차 시퀀스 시작 ----
                if (
                    self.park_area >= self.park_min_area
                    and not self.start_park
                    and not self.doing_turn_right
                    and not self.parked
                    and not self.stop
                ):
                    self.count_park += 1
                    if self.count_park >= 15:
                        self.count_park = 0
                        self.start_park = True
                        threading.Thread(
                            target=self.approach_and_park, daemon=True
                        ).start()
                elif not self.start_park:
                    self.count_park = 0

                # ---- LED: 규정7(go 신호 점멸)이 최우선, 아니면 규정4(주행/정지 상태) ----
                go_signal_active = (
                    time.time() - self.go_signal_time
                ) < self.go_signal_duration
                if go_signal_active:
                    phase = int(time.time() / 0.3) % 2  # 0.3초 간격 점멸
                    led_color = (255, 255, 0) if phase == 0 else (0, 0, 0)
                    if self.last_led_state != led_color:
                        self.last_led_state = led_color
                        self.set_rgb(*led_color)
                    self.moving_led_state = (
                        None  # 점멸 종료 후 강제로 재반영되도록 초기화
                    )
                else:
                    if self.last_led_state is not None:
                        self.last_led_state = None
                    if self.parked:
                        is_moving = False
                    elif self.doing_turn_right or self.start_park:
                        is_moving = (
                            True  # 우회전/주차 동작 스레드가 실제로 로봇을 움직이는 중
                        )
                    else:
                        is_moving = not self.stop
                    self.update_status_led(is_moving)

                # 정지/우회전/주차/주차완료 중에는 차선추종 제어를 양보(각 동작 스레드가 직접 모터를 제어)
                # maneuver_active: execute_turn_right()/approach_and_park() 스레드가 지금
                # 직접 mecanum_pub에 twist를 publish하고 있는 구간. 이 동안 main()이 별도로
                # 0속도를 publish하면 두 스레드가 같은 토픽에 번갈아 써서 움직임이 끊기므로
                # (경쟁), 아래 명시적 0속도 publish에서는 이 구간을 제외한다.
                maneuver_active = self.doing_turn_right or (
                    self.start_park and not self.parked
                )
                blocked = (
                    self.parked or self.start_park or self.doing_turn_right or self.stop
                )

                if not blocked:
                    if self.crosswalk_distance > self.crosswalk_approach_dist:
                        twist.linear.x = (
                            self.crosswalk_approach_speed
                        )  # 횡단보도 접근 감속
                    elif (
                        self.start_turn
                        and time.time() - self.start_turn_time_stamp
                        < self.turn_recover_time
                    ):
                        twist.linear.x = self.corner_speed  # 코너 직후 복귀 구간 감속
                    else:
                        twist.linear.x = (
                            self.normal_speed
                        )  # go straight with normal speed

                # line following processing
                result_image, lane_angle, lane_x = self.lane_detect(
                    binary_image, image.copy()
                )  # the coordinate of the line while the robot is in the middle of the lane
                # execute_turn_right()가 image_queue를 따로 소비하지 않고도 최신 차선
                # 위치를 읽을 수 있도록 공유 상태로 남겨둠(회전/정지 중에도 계속 갱신)
                self.latest_lane_x = lane_x
                if lane_x >= 0 and not blocked:
                    lane_setpoint = (
                        self.park_lane_setpoint
                        if self.going_to_park
                        else self.lane_setpoint
                    )
                    angular_limit = (
                        self.park_angular_limit
                        if self.going_to_park
                        else self.angular_z_limit
                    )
                    if lane_x > self.turn_threshold:
                        self.count_turn += 1
                        if (
                            self.count_turn > self.turn_confirm_count
                            and not self.start_turn
                        ):
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != "MentorPi_Acker":
                            twist.angular.z = self.turn_angular_z  # turning speed
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0
                        if (
                            time.time() - self.start_turn_time_stamp
                            > self.turn_recover_time
                            and self.start_turn
                        ):
                            self.start_turn = False
                        if not self.start_turn:
                            if (
                                abs(lane_x - lane_setpoint) < self.lane_deadband
                            ):  # 데드밴드: 미세 진동 제거
                                twist.angular.z = 0.0
                                self.pid.clear()
                            else:
                                self.pid.SetPoint = lane_setpoint  # the coordinate of the line while the robot is in the middle of the lane
                                self.pid.update(lane_x)
                                if self.machine_type != "MentorPi_Acker":
                                    twist.angular.z = common.set_range(
                                        self.pid.output, -angular_limit, angular_limit
                                    )
                                else:
                                    twist.angular.z = (
                                        twist.linear.x
                                        * math.tan(
                                            common.set_range(
                                                self.pid.output,
                                                -angular_limit,
                                                angular_limit,
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
                    if blocked and not maneuver_active:
                        # self.stop(횡단보도/신호등 정지) 또는 self.parked(영구 정지)일 때만
                        # 여기서 0속도를 publish. maneuver_active(우회전/주차 진행 중)에는
                        # 해당 스레드가 이미 자체적으로 twist를 publish하고 있으므로 여기서
                        # 또 publish하면 서로 경쟁해 움직임이 끊긴다.
                        self.mecanum_pub.publish(Twist())

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
                # [진단용] 예전에는 버튼을 기다리는 동안 아무 로그도 안 찍혀서, 버튼 입력이
                # 안 들어오는 상황과 다른 문제를 구분하기 어려웠음(1차 실차 테스트에서
                # crosswalk 인식 로그만 계속 보여 헷갈렸던 원인 중 하나). 3초마다 한 번씩
                # "대기중" 알림을 남겨서 로그만 보고도 상태를 바로 알 수 있게 함.
                if self.require_start_button and not self.start:
                    now = time.time()
                    if now - self.last_wait_reminder_time > 3.0:
                        self.last_wait_reminder_time = now
                        self.get_logger().info(
                            "\033[1;33m%s\033[0m"
                            % "still waiting for start button (Button1) press..."
                        )
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
                box = i.box
                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                # 규정6/8/9 필터링에 쓰는 박스 크기(거리 지표) 및 종횡비
                width = abs(box[2] - box[0])
                height = abs(box[3] - box[1])
                area = width * height
                aspect = width / height if height > 0 else 0

                if class_name == "crosswalk":
                    # [종횡비/면적 필터] 규정6: 바닥 얼룩 등 오검출 제거, 실제 횡단보도
                    # (가로로 긴 줄무늬)만 거리 판단에 반영
                    if (
                        area >= self.crosswalk_min_area
                        and aspect >= self.crosswalk_min_aspect
                    ):
                        if (
                            center[1] > min_distance
                        ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                            min_distance = center[1]
                elif class_name == "right":  # obtain the right turning sign
                    # 규정8: 표지판이 충분히 가까울 때만 카운트(멀리서 오검출 방지).
                    # 확정되면 turn_right=True -> main()의 execute_turn_right()가 소비함
                    if area >= self.right_min_area:
                        self.count_right += 1
                        self.count_right_miss = 0
                        if (
                            self.count_right >= 5
                        ):  # If it is detected multiple times, take the right turning sign to true
                            self.turn_right = True
                            self.count_right = 0
                elif class_name == "go":
                    # 규정7: 직진 화살표 표지판 인식 -> 일정 시간 노란 LED 점멸(main()에서 소비)
                    self.count_go += 1
                    if (
                        self.count_go >= 3
                    ):  # judge multiple times to prevent false detection
                        self.go_signal_time = time.time()
                        self.count_go = 0
                elif (
                    class_name == "park"
                ):  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                    self.park_area = (
                        area  # 규정10: 표지판 거리(면적) 기반 주차 트리거에 사용
                    )
                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light
                    self.traffic_signs_status = i
                    if class_name == "red":
                        # [신호등 정지] 규정9: '빨강 신선도' 방식(main()의 is_red_light 참고).
                        # 가까운 빨강(면적 충분)은 별도로 기록해 무조건 정지 트리거에 사용
                        self.red_last_seen_time = time.time()
                        if area >= self.red_min_area:
                            self.red_close_time = time.time()

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
