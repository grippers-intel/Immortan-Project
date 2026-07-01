#!/usr/bin/env python3
# encoding: utf-8
# =============================================================================
# self_driving_launch.py  —  자율주행 런치 파일 (LED 노드 추가)
# =============================================================================
# 변경사항 (기존 대비):
#   1. led_controller 노드 추가 (규칙 4·5·7·11)
#   2. only_line_follow 파라미터 유지
#   3. web_video_server 주석 유지 (필요 시 활성화)
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchService
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    # ── 컴파일 여부에 따른 경로 분기 ─────────────────────────────────────────
    compiled = os.environ.get("need_compile", "False")

    if compiled == "True":
        peripherals_package_path = get_package_share_directory("peripherals")
        controller_package_path = get_package_share_directory("controller")
        yolov5_package_path = get_package_share_directory("yolov5_ros2")
    else:
        peripherals_package_path = "/home/ubuntu/ros2_ws/src/peripherals"
        controller_package_path = "/home/ubuntu/ros2_ws/src/driver/controller"
        yolov5_package_path = "/home/ubuntu/ros2_ws/src/yolov5_ros2"

    # ── 런치 인자 ─────────────────────────────────────────────────────────────
    start = LaunchConfiguration("start", default="true")
    start_arg = DeclareLaunchArgument("start", default_value=start)

    only_line_follow = LaunchConfiguration("only_line_follow", default="false")
    only_line_follow_arg = DeclareLaunchArgument(
        "only_line_follow", default_value=only_line_follow
    )

    test_mode = LaunchConfiguration("test_mode", default="false")
    test_mode_arg = DeclareLaunchArgument("test_mode", default_value=test_mode)

    # ── 서브 런치: 뎁스 카메라 ───────────────────────────────────────────────
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, "launch/depth_camera.launch.py")
        ),
    )

    # ── 서브 런치: 컨트롤러 (모터 드라이버 + 오도메트리) ──────────────────────
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, "launch/controller.launch.py")
        ),
    )

    # ── YOLO 감지 노드 ────────────────────────────────────────────────────────
    yolov5_node = Node(
        package="yolov5_ros2",
        executable="yolo_detect",
        output="screen",
        parameters=[
            {
                "classes": ["go", "right", "park", "red", "green", "crosswalk"],
                "device": "cpu",
                "model": "traffic_signs_640s_7_0",
                "image_topic": "/ascamera/camera_publisher/rgb0/image",
                "camera_info_topic": "/camera/camera_info",
                "camera_info_file": f"{yolov5_package_path}/config/camera_info.yaml",
                "pub_result_img": True,
            }
        ],
    )

    # ── LED 제어 노드 (신규 추가) ─────────────────────────────────────────────
    # 규칙 4: 주행=녹색ON/적색OFF, 정지=녹색OFF/적색ON
    # 규칙 5: Raspberry Pi GPIO → 브레드보드 LED
    # 규칙 7: 화살표 감지 시 황색 점멸
    # 규칙 11: 주차 완료 시 전체 LED 점멸
    led_controller_node = Node(
        package="example",
        executable="led_controller",
        output="screen",
    )

    # ── 자율주행 메인 노드 ────────────────────────────────────────────────────
    self_driving_node = Node(
        package="example",
        executable="self_driving",
        output="screen",
        parameters=[
            {"start": start},
            {"only_line_follow": only_line_follow},
            {"test_mode": test_mode},
        ],
    )

    # ── web_video_server (필요 시 주석 해제) ──────────────────────────────────
    # web_video_server_node = Node(
    #     package='web_video_server',
    #     executable='web_video_server',
    #     output='screen',
    # )

    return [
        start_arg,
        only_line_follow_arg,
        test_mode_arg,
        depth_camera_launch,
        controller_launch,
        yolov5_node,
        led_controller_node,  # ← 신규 추가
        self_driving_node,
        # web_video_server_node,
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])


if __name__ == "__main__":
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
