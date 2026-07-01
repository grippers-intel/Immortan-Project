#!/usr/bin/env python3
# encoding: utf-8
# =============================================================================
# self_driving.py  —  FSM 기반 자율주행 메인 노드 v3 (검수 수정본)
# =============================================================================
# v2 → v3 수정사항:
#   [🔴 크래시 수정]
#   1. twist.angular_z 오타 → twist.angular.z 로 전면 수정
#   2. S1 신호등 대기 오류 → crosswalk_stage == 2 (S3)만 TRAFFIC_LIGHT
#   3. 주차 수직이동 방향 오류 → linear_y = +0.2 (진행방향 기준 우측)
#   [🟡 논리/안정성 수정]
#   4. 급커브 조향각 -0.8 → -1.2, 복귀 시간 0.2 → 0.3초
#   5. PARK_COUNT 3 → 6 (오감지 방지)
#   6. 주차 정지 조건: 타이머 의존 → park_cy 기반 + 타임아웃 안전장치
#   7. _scan_objects()에 park_cy 키 추가
#   8. CROSSWALK_WAIT 1.0 → 2.0초 (규칙 6 준수)
# =============================================================================
#
# 코스 정지 지점 (crosswalk_stage 순서):
#   Stage 0 (S1): 하단 횡단보도 — 신호등 있음 → green 신호 후 출발
#   Stage 1 (S2): 좌측 횡단보도 — 신호등 없음 → 2초 정지
#   Stage 2 (S3): 상단 횡단보도 — 신호등 있음 → green 신호 후 출발
#   Stage 3 (S4): 우측 횡단보도 — 신호등 없음 → 2초 정지 후 우회전
#
# ★ 표시 항목은 실측 후 조정 필요
# =============================================================================

import os
import cv2
import time
import queue
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from math import atan2, pi
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from interfaces.msg import ObjectsInfo
from ros_robot_controller_msgs.msg import ButtonState

import sdk.pid as pid
import sdk.fps as fps
import sdk.common as common
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect


# =============================================================================
# FSM 상태 정의
# =============================================================================
class State:
    WAIT_START = "WAIT_START"
    LINE_FOLLOW = "LINE_FOLLOW"
    CROSSWALK_STOP = "CROSSWALK_STOP"
    ARROW_SIGNAL = "ARROW_SIGNAL"
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    INTERSECTION = "INTERSECTION"
    PARKING = "PARKING"
    DONE = "DONE"


# =============================================================================
# 주행 상수  (★ = 실측 후 조정 권장)
# =============================================================================
NORMAL_SPEED = 0.60  # 고속 직진 속도 (m/s)
SLOW_SPEED = 0.20  # 감속 속도
TURN_SPEED = 0.20  # 교차로 직진 속도
PARK_SPEED = 0.15  # 주차 접근 속도
PARK_SIDE_SPEED = 0.20  # 주차 수직이동 속도

TURN_ANGULAR = 1.30  # 우회전 각속도 (rad/s)  ★
TURN_ANGLE_DEG = 80.0  # 우회전 목표 각도 (°) — 오도메트리 기반  ★
TURN_TIMEOUT = 4.0  # 오도메트리 실패 시 안전 타임아웃 (초)  ★

CROSSWALK_WAIT = 2.0  # 신호등 없는 횡단보도 정지 시간 (초) — 규칙 6
CROSSWALK_EXIT_T = 1.0  # 횡단보도 탈출 맹목 전진 시간 (초)  ★
ARROW_BLINK_TIME = 1.5  # 황색 LED 점멸 지속 시간 (초)
TRAFFIC_TIMEOUT = 15.0  # 신호등 최대 대기 시간 (초)

PARK_SIDE_TIME = 2.5  # 주차 수직이동 지속 시간 (초)  ★ (이전팀 0.7m 기준 참고)
PARK_STOP_Y = 350  # park 표지판 중심 Y 임계값 (px)  ★
PARK_TIMEOUT = 5.0  # 주차 직진 최대 시간 (초, 안전장치)

# 감지 임계값
CROSSWALK_NEAR_Y = 200  # crosswalk Y픽셀 임계 (300→200: 더 멀리서 감지)
CROSSWALK_COUNT = 3  # 오탐 방지: 3프레임 연속 확인 (2→3)
ARROW_COUNT = 3  # 오탐 방지: 3프레임 연속 확인 (2→3)
PARK_COUNT = 6  # ★ 오감지 방지: 6프레임 연속 확인
TRAFFIC_AREA_MIN = 200  # 신호등 최소 감지 면적 (px²) (400→200)
PARK_AREA_MIN = 200  # park 최소 감지 면적 (px²) (400→200)
ARROW_AREA_MIN = 200  # 화살표 최소 감지 면적 (px²) (400→200)

# 급커브 처리
SHARP_TURN_X = 140  # 이 값 초과 시 급커브 판정
SHARP_TURN_Z = -1.2  # 급커브 조향각  ★ (이전 -0.8 → -1.2 강화)
SHARP_TURN_TIME = 0.3  # 급커브 유지 시간 (초)  ★ (이전 0.2 → 0.3)
SHARP_TURN_COUNT = 2  # 급커브 진입 연속 프레임

# 코스 횡단보도 설정
MAX_CROSSWALK_STAGE = 4  # S1~S4 총 4회
TRAFFIC_STAGE = {0, 2}  # 신호등 있는 횡단보도: S1(stage=0), S3(stage=2)


# =============================================================================
# 메인 노드
# =============================================================================
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
        self.bridge = CvBridge()
        self.fps_ = fps.FPS()
        self.pid = pid.PID(0.4, 0.0, 0.1)  # 고속용 D값 상향
        self.lock = threading.RLock()
        self.machine_type = os.environ.get("MACHINE_TYPE", "MentorPi_Mecanum")
        self.test_mode = (
            self.declare_parameter("test_mode", "false").value.lower() == "true"
        )

        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # ── FSM ──────────────────────────────────────────────────────────────
        self._fsm_state = State.WAIT_START
        self._state_entry_time = time.time()

        # ── 감지 카운터 ───────────────────────────────────────────────────────
        self._cnt_crosswalk = 0
        self._cnt_arrow = 0
        self._cnt_park = 0

        # ── 횡단보도 상태 ─────────────────────────────────────────────────────
        self.crosswalk_stage = 0  # 현재까지 처리 완료한 횡단보도 수
        self._crosswalk_done = False  # 현재 횡단보도 처리 완료 플래그

        # ── 기타 플래그 ───────────────────────────────────────────────────────
        self._arrow_direction = None
        self._traffic_color = None
        self._park_triggered = False

        # ── 급커브 상태 ───────────────────────────────────────────────────────
        self._cnt_turn = 0
        self._is_turning = False
        self._turn_start_time = 0.0

        # ── 오도메트리 (우회전 각도 추종) ────────────────────────────────────
        self.yaw = 0.0
        self.degree = 0.0
        self._turn_start_yaw = 0.0

        self.objects_info = []

        # ── 퍼블리셔 ──────────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.led_pub = self.create_publisher(String, "/led/cmd", 10)
        self.result_pub = self.create_publisher(Image, "~/image_result", 1)

        # ── 서비스 ────────────────────────────────────────────────────────────
        self.create_service(Trigger, "~/enter", self._enter_cb)
        self.create_service(Trigger, "~/exit", self._exit_cb)
        self.create_service(SetBool, "~/set_running", self._set_running_cb)
        self.create_subscription(
            ButtonState, "/ros_robot_controller/button", self._button_cb, 10
        )
        self.create_subscription(Odometry, "odom", self._odom_cb, 10)

        # ── YOLO 클라이언트 ───────────────────────────────────────────────────
        timer_cb_group = ReentrantCallbackGroup()
        self._yolo_start_client = self.create_client(
            Trigger, "/yolov5/start", callback_group=timer_cb_group
        )
        self._yolo_stop_client = self.create_client(
            Trigger, "/yolov5/stop", callback_group=timer_cb_group
        )

        # ── 백그라운드 시퀀스 (데드락 방지) ──────────────────────────────────
        threading.Thread(target=self._main_sequence, daemon=True).start()

    # =========================================================================
    # 메인 시퀀스 스레드 — 초기화 후 버튼 대기
    # =========================================================================
    def _main_sequence(self):
        self.get_logger().info("시스템 초기화 중...")
        self.cmd_vel_pub.publish(Twist())
        self._led("red_on")
        time.sleep(2.0)

        # YOLO 시작
        if self._yolo_start_client.wait_for_service(timeout_sec=5.0):
            self._yolo_start_client.call_async(Trigger.Request())
            time.sleep(1.0)
        else:
            self.get_logger().warn("YOLO 서비스 연결 실패. 계속 진행합니다.")

        # 구독 시작
        self._enter_cb(Trigger.Request(), Trigger.Response())

        if self.test_mode:
            # 시험 운행: 3초 카운트다운 후 자동 출발
            for i in range(3, 0, -1):
                self.get_logger().info(
                    f"\033[1;33m[시험모드] {i}초 후 자동 출발\033[0m"
                )
                time.sleep(1.0)
            self.get_logger().info("\033[1;32m[시험모드] 출발!\033[0m")
            self._transition(State.LINE_FOLLOW)
        else:
            # 대회 모드: 버튼 대기 (규칙 3)
            self.get_logger().info(
                "\033[1;33mFSM v3 Ready: Button1을 눌러 출발하세요\033[0m"
            )

        # FSM 루프
        self._main_loop()

    # =========================================================================
    # 서비스 콜백
    # =========================================================================
    def _enter_cb(self, request, response):
        with self.lock:
            self.create_subscription(
                Image, "/ascamera/camera_publisher/rgb0/image", self._image_cb, 1
            )
            self.create_subscription(
                ObjectsInfo, "/yolov5_ros2/object_detect", self._objects_cb, 1
            )
            self.cmd_vel_pub.publish(Twist())
        response.success = True
        return response

    def _exit_cb(self, request, response):
        self.cmd_vel_pub.publish(Twist())
        self._led("all_off")
        self._fsm_state = State.WAIT_START
        response.success = True
        return response

    def _set_running_cb(self, request, response):
        if not request.data:
            self.cmd_vel_pub.publish(Twist())
            self._led("red_on")
        response.success = True
        return response

    def _button_cb(self, msg: ButtonState):
        # state: 1=PRESSED, 5=CLICK (ros_robot_controller_node.py state_map 참조)
        if msg.state in (1, 5) and self._fsm_state == State.WAIT_START:
            self.get_logger().info(f"\033[1;32m[버튼 id={msg.id}] 출발 신호\033[0m")
            self._transition(State.LINE_FOLLOW)

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.degree = self.yaw * 180.0 / pi

    def _image_cb(self, ros_image):
        img = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        img = np.array(img, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(img)

    def _objects_cb(self, msg):
        self.objects_info = msg.objects

    # =========================================================================
    # 헬퍼
    # =========================================================================
    def _led(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.led_pub.publish(msg)

    def _move(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.linear.y = float(linear_y)
        twist.angular.z = float(angular_z)  # ✅ 수정 1: .angular.z (오타 수정)
        self.cmd_vel_pub.publish(twist)

    def _stop(self):
        self.cmd_vel_pub.publish(Twist())

    def _elapsed(self) -> float:
        return time.time() - self._state_entry_time

    def _angle_diff_deg(self, a: float, b: float) -> float:
        """두 각도(°) 사이의 최단 거리 (0 ~ 180)"""
        return abs((a - b + 180.0) % 360.0 - 180.0)

    # =========================================================================
    # FSM 상태 전환
    # =========================================================================
    def _transition(self, new_state: str):
        old = self._fsm_state
        self._fsm_state = new_state
        self._state_entry_time = time.time()
        self.get_logger().info(f"\033[1;36m[FSM] {old} → {new_state}\033[0m")

        if new_state == State.LINE_FOLLOW:
            self._led("green_on")
        elif new_state in (State.CROSSWALK_STOP, State.TRAFFIC_LIGHT, State.WAIT_START):
            self._led("red_on")
        elif new_state == State.ARROW_SIGNAL:
            self._led("yellow_blink")
        elif new_state == State.INTERSECTION:
            self._turn_start_yaw = self.degree  # 우회전 시작 각도 저장
        elif new_state == State.DONE:
            self._stop()
            self._led("all_blink")

    # =========================================================================
    # 메인 FSM 루프
    # =========================================================================
    def _main_loop(self):
        while self.is_running:
            t_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            result_image = image.copy()

            if self._fsm_state == State.WAIT_START:
                self._stop()
            elif self._fsm_state == State.LINE_FOLLOW:
                result_image = self._state_line_follow(image, result_image)
            elif self._fsm_state == State.CROSSWALK_STOP:
                result_image = self._state_crosswalk_stop(image, result_image)
            elif self._fsm_state == State.TRAFFIC_LIGHT:
                result_image = self._state_traffic_light(image, result_image)
            elif self._fsm_state == State.ARROW_SIGNAL:
                result_image = self._state_arrow_signal(image, result_image)
            elif self._fsm_state == State.INTERSECTION:
                result_image = self._state_intersection(image, result_image)
            elif self._fsm_state == State.PARKING:
                result_image = self._state_parking(image, result_image)
            elif self._fsm_state == State.DONE:
                self._stop()

            result_image = self._draw_objects(result_image)
            bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            self.fps_.update()
            bgr = self.fps_.show_fps(bgr)
            self.result_pub.publish(self.bridge.cv2_to_imgmsg(bgr, "bgr8"))

            sleep_t = 0.033 - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._stop()
        rclpy.shutdown()

    # =========================================================================
    # 상태: LINE_FOLLOW
    # =========================================================================
    def _state_line_follow(self, image, result_image):
        binary = self.lane_detect.get_binary(image)
        result_image, lane_angle, lane_x = self.lane_detect(binary, result_image)

        # None 방어
        if lane_angle is None:
            lane_angle = 0.0
        if lane_x is None:
            lane_x = -1.0

        detected = self._scan_objects()

        # 1) 주차 표지판
        if detected["park"]:
            self._cnt_park += 1
            if self._cnt_park >= PARK_COUNT and not self._park_triggered:
                self._cnt_park = 0
                self._park_triggered = True
                self._transition(State.PARKING)
                return result_image
        else:
            self._cnt_park = 0

        # 2) 횡단보도 감지
        cw_y = detected["crosswalk_y"]
        if (
            cw_y > CROSSWALK_NEAR_Y
            and not self._crosswalk_done
            and self.crosswalk_stage < MAX_CROSSWALK_STAGE
        ):
            self._cnt_crosswalk += 1
            if self._cnt_crosswalk >= CROSSWALK_COUNT:
                self._cnt_crosswalk = 0
                self.get_logger().info(
                    f"\033[1;33m[CW] Stage {self.crosswalk_stage} 횡단보도 감지\033[0m"
                )

                # ✅ 수정 2: stage==2(S3)만 신호등 대기, 나머지는 단순 정지
                if self.crosswalk_stage in TRAFFIC_STAGE:
                    self._traffic_color = detected["traffic"]
                    self._transition(State.TRAFFIC_LIGHT)
                else:
                    self._transition(State.CROSSWALK_STOP)
                return result_image
        else:
            if cw_y <= CROSSWALK_NEAR_Y:
                self._cnt_crosswalk = 0

        # 3) 화살표 감지
        arrow = detected["arrow"]
        if arrow in ("go", "right"):
            self._cnt_arrow += 1
            if self._cnt_arrow >= ARROW_COUNT:
                self._cnt_arrow = 0
                self._arrow_direction = arrow
                self._transition(State.ARROW_SIGNAL)
                return result_image
        else:
            self._cnt_arrow = 0

        # 4) 동적 감속: 표지판/횡단보도/화살표 감지 시 감속
        near_event = detected["park"] or cw_y > 100 or arrow
        target_speed = SLOW_SPEED if near_event else NORMAL_SPEED
        self._do_line_follow(lane_x, lane_angle, target_speed)
        return result_image

    # =========================================================================
    # 상태: CROSSWALK_STOP — 신호등 없는 횡단보도 (S2, S4)
    # =========================================================================
    def _state_crosswalk_stop(self, image, result_image):
        """2초 정지 → 1초 맹목 전진(횡단보도 탈출) → LINE_FOLLOW"""
        elapsed = self._elapsed()
        if elapsed < CROSSWALK_WAIT:
            self._stop()
        elif elapsed < CROSSWALK_WAIT + CROSSWALK_EXIT_T:
            self._move(linear_x=SLOW_SPEED)
        else:
            self.crosswalk_stage += 1
            self._crosswalk_done = True
            self.get_logger().info(
                f"[CW_STOP] 탈출 완료 → Stage {self.crosswalk_stage}"
            )
            self._transition(State.LINE_FOLLOW)
            threading.Thread(target=self._reset_crosswalk_flag, daemon=True).start()
        return result_image

    # =========================================================================
    # 상태: TRAFFIC_LIGHT — 신호등 있는 횡단보도 (S3)
    # =========================================================================
    def _state_traffic_light(self, image, result_image):
        """
        정지 유지 → green 신호 감지 시 출발.
        TRAFFIC_TIMEOUT 초과 시 강제 출발.
        """
        self._stop()
        detected = self._scan_objects()
        if detected["traffic"]:
            self._traffic_color = detected["traffic"]

        if self._traffic_color == "green":
            self.crosswalk_stage += 1
            self._crosswalk_done = True
            self.get_logger().info(
                f"[TRAFFIC] 녹색 신호 → Stage {self.crosswalk_stage}"
            )
            self._transition(State.LINE_FOLLOW)
            threading.Thread(target=self._reset_crosswalk_flag, daemon=True).start()
        elif self._elapsed() > TRAFFIC_TIMEOUT:
            self.get_logger().warn("[TRAFFIC] 타임아웃 → 강제 출발")
            self.crosswalk_stage += 1
            self._crosswalk_done = True
            self._transition(State.LINE_FOLLOW)
            threading.Thread(target=self._reset_crosswalk_flag, daemon=True).start()

        return result_image

    def _reset_crosswalk_flag(self):
        """5초 후 crosswalk_done 해제 → 다음 횡단보도 처리 가능"""
        time.sleep(5.0)
        self._crosswalk_done = False

    # =========================================================================
    # 상태: ARROW_SIGNAL — 황색 LED 점멸 (규칙 7)
    # =========================================================================
    def _state_arrow_signal(self, image, result_image):
        if self._elapsed() < ARROW_BLINK_TIME:
            self._stop()
        else:
            self._led("yellow_off")
            self.get_logger().info(
                f"[ARROW] 방향={self._arrow_direction} → INTERSECTION"
            )
            self._transition(State.INTERSECTION)
        return result_image

    # =========================================================================
    # 상태: INTERSECTION — 교차로 우회전 (규칙 8)
    # =========================================================================
    def _state_intersection(self, image, result_image):
        elapsed = self._elapsed()
        binary = self.lane_detect.get_binary(image)
        result_image, lane_angle, lane_x = self.lane_detect(binary, result_image)

        if lane_angle is None:
            lane_angle = 0.0
        if lane_x is None:
            lane_x = -1.0

        if self._arrow_direction == "go":
            self._do_line_follow(lane_x, lane_angle, NORMAL_SPEED)
            if lane_x >= 0:
                self._transition(State.LINE_FOLLOW)

        elif self._arrow_direction == "right":
            turned = self._angle_diff_deg(self.degree, self._turn_start_yaw)
            if turned < TURN_ANGLE_DEG and elapsed < TURN_TIMEOUT:
                self._move(linear_x=TURN_SPEED, angular_z=-TURN_ANGULAR)
            else:
                if elapsed >= TURN_TIMEOUT:
                    self.get_logger().warn(
                        f"[INTERSECTION] 오도메트리 타임아웃 ({turned:.1f}°) → 강제 전환"
                    )
                if lane_x >= 0:
                    self._transition(State.LINE_FOLLOW)
                else:
                    self._move(linear_x=TURN_SPEED, angular_z=-0.3)

        return result_image

    # =========================================================================
    # 상태: PARKING — 주차 (규칙 10·11)
    # =========================================================================
    def _state_parking(self, image, result_image):
        """
        1단계: park_cy > PARK_STOP_Y 될 때까지 라인 추종 직진 (타임아웃 안전장치)
        2단계: 우측 수직이동 (진행방향 기준 우측 = linear.y 양수)
        3단계: 정지 → DONE
        """
        elapsed = self._elapsed()
        detected = self._scan_objects()
        park_cy = detected["park_cy"]

        binary = self.lane_detect.get_binary(image)
        result_image, lane_angle, lane_x = self.lane_detect(binary, result_image)
        if lane_angle is None:
            lane_angle = 0.0
        if lane_x is None:
            lane_x = -1.0

        # 1단계: 표지판까지 직진 (park_cy 기반 + 타임아웃)
        if park_cy < PARK_STOP_Y and elapsed < PARK_TIMEOUT:
            self._do_line_follow(lane_x, lane_angle, PARK_SPEED)

        # 2단계: 우측 수직이동
        elif elapsed < PARK_TIMEOUT + PARK_SIDE_TIME:
            # ROS REP-103: linear.y 양수=좌측, 음수=우측 (제조사 원본 코드 기준 일치)
            self._move(linear_x=0.0, linear_y=-PARK_SIDE_SPEED, angular_z=0.0)

        # 3단계: 정지 → DONE
        else:
            self._stop()
            self.get_logger().info("[PARKING] 주차 완료 → DONE")
            self._transition(State.DONE)

        return result_image

    # =========================================================================
    # 공통: 라인 추종 (PID + 피드포워드 + 급커브 처리)
    # =========================================================================
    def _do_line_follow(self, lane_x: float, lane_angle: float, speed: float):
        twist = Twist()
        twist.linear.x = float(speed)

        if lane_x < 0:
            self._stop()
            self.pid.clear()
            return

        # 급커브 진입 판단
        if not self._is_turning:
            if lane_x > SHARP_TURN_X:
                self._cnt_turn += 1
                if self._cnt_turn >= SHARP_TURN_COUNT:
                    self._is_turning = True
                    self._turn_start_time = time.time()
                    self._cnt_turn = 0
            else:
                self._cnt_turn = 0

        # 급커브 처리
        if self._is_turning:
            elapsed_turn = time.time() - self._turn_start_time
            if elapsed_turn > SHARP_TURN_TIME:  # ✅ 수정 4: 0.2 → 0.3초
                self._is_turning = False
            else:
                twist.linear.x = float(SLOW_SPEED)
                twist.angular.z = SHARP_TURN_Z  # ✅ 수정 4: -0.8 → -1.2
                self.cmd_vel_pub.publish(twist)
                return

        # PID + lane_angle 피드포워드
        self.pid.SetPoint = 130
        self.pid.update(lane_x)
        twist.angular.z = common.set_range(
            self.pid.output + (lane_angle * 0.005), -0.22, 0.22
        )
        self.cmd_vel_pub.publish(twist)

    # =========================================================================
    # 공통: YOLO 결과 파싱
    # =========================================================================
    def _scan_objects(self) -> dict:
        result = {
            "park": False,
            "park_cy": 0,  # ✅ 수정 7: park_cy 키 추가
            "crosswalk_y": 0,
            "traffic": None,
            "arrow": None,
        }

        for obj in self.objects_info:
            name = obj.class_name
            cy = int((obj.box[1] + obj.box[3]) / 2)
            area = abs(obj.box[2] - obj.box[0]) * abs(obj.box[3] - obj.box[1])

            if name == "park" and area > PARK_AREA_MIN:
                result["park"] = True
                result["park_cy"] = max(result["park_cy"], cy)

            elif name == "crosswalk":
                result["crosswalk_y"] = max(result["crosswalk_y"], cy)

            elif name in ("red", "green") and area > TRAFFIC_AREA_MIN:
                result["traffic"] = name

            elif name in ("go", "right") and area > ARROW_AREA_MIN:
                result["arrow"] = name

        return result

    # =========================================================================
    # 공통: YOLO 박스 시각화
    # =========================================================================
    def _draw_objects(self, image):
        for obj in self.objects_info:
            if obj.class_name not in self.classes:
                continue
            cls_id = self.classes.index(obj.class_name)
            color = colors(cls_id, True)
            plot_one_box(
                obj.box, image, color=color, label=f"{obj.class_name}:{obj.score:.2f}"
            )
        return image

    # =========================================================================
    # 종료
    # =========================================================================
    def shutdown(self):
        self.is_running = False
        self._stop()
        self._led("all_off")


# =============================================================================
# 엔트리포인트
# =============================================================================
def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
