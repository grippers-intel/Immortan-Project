import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    start = LaunchConfiguration('start', default='true')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    only_line_follow = LaunchConfiguration('only_line_follow', default='false')
    only_line_follow_arg = DeclareLaunchArgument('only_line_follow', default_value=only_line_follow)
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
        controller_package_path = get_package_share_directory('controller')
        package_share_directory = get_package_share_directory('yolov5_ros2')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        package_share_directory = '/home/ubuntu/ros2_ws/src/yolov5_ros2'

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        output='screen',
    )
    yolov5_node = Node(
        package='yolov5_ros2',
        executable='yolo_detect',
        output='screen',
        parameters=[{'classes': ['go', 'right', 'park', 'red', 'green', 'crosswalk']},
            {"device": "cpu",
            "model": "traffic_signs_640s_7_0",
            "image_topic": "/ascamera/camera_publisher/rgb0/image",
            "camera_info_topic": "/camera/camera_info",
            "camera_info_file": f"{package_share_directory}/config/camera_info.yaml",
            # "show_result": True,
            # [수정] pub_result_img를 끔(True→False). YOLO가 매 추론마다 결과이미지를 그려
            #   발행하느라 추론 FPS가 떨어져, 주행 중 신호등을 잡는 프레임이 줄던 문제 해결.
            #   박스 시각화는 self_driving이 /self_driving/image_result에 이미 그려주므로 디버깅엔 지장 없음.
            #   (다시 보고 싶으면 True로)
            "pub_result_img": False}]
    )

    self_driving_node = Node(
        package='example',
        executable='self_driving',
        output='screen',
        parameters=[{'start': start}, {'only_line_follow': only_line_follow}],
    )

    return [start_arg,
            only_line_follow_arg,
            depth_camera_launch,
            controller_launch,
            #web_video_server_node,
            yolov5_node, 
            self_driving_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

