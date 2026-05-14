#!/usr/bin/env python3
"""
fly_straight.py
---------------
讓無人機從當前位置 A 飛直線到目標點 B。

使用方式:
    ros2 run nsysu_drone_control fly_straight       # 飛到預設目標點
    或
    python3 fly_straight.py                         # 直接用 python 執行

前置條件:
    1. Gazebo 模擬已經啟動 (launch_drone)
    2. 無人機已經起飛:
       ros2 topic pub /simple_drone/takeoff std_msgs/msg/Empty {} --once
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty


class FlyStraight(Node):
    """讓無人機沿直線飛到目標點的 ROS 2 Node。"""

    def __init__(self):
        # 初始化 Node,給它一個名字
        super().__init__('fly_straight_controller')

        # ========== 1. 設定目標點 B ==========
        # (x, y, z) 單位是公尺,你可以自由修改
        self.target_x = 5.0
        self.target_y = 3.0
        self.target_z = 2.0

        # ========== 2. 控制參數 ==========
        self.Kp = 0.5          # 比例增益 (Proportional gain)
        self.max_speed = 1.0   # 最大速度限制,單位: m/s
        self.tolerance = 0.2   # 誤差容忍值,單位: 公尺 (差多少算抵達)

        # ========== 3. 紀錄當前位置 ==========
        self.current_x = None
        self.current_y = None
        self.current_z = None

        # ========== 4. 建立 Publisher (發布控制指令) ==========
        # topic 名稱對應 Gazebo 裡無人機訂閱的 /simple_drone/cmd_vel
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/simple_drone/cmd_vel', 10
        )

        # ========== 5. 建立 Subscriber (讀取無人機位置) ==========
        # /simple_drone/gt_pose 是 ground truth 位置 (模擬環境中最準確的)
        self.pose_sub = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self.pose_callback, 10
        )

        # ========== 6. 建立定時器 (每 0.1 秒執行一次控制迴圈) ==========
        self.timer = self.create_timer(0.1, self.control_loop)

        # 標記是否已抵達,避免重複列印訊息
        self.arrived = False

        self.get_logger().info(
            f'🎯 目標位置: ({self.target_x}, {self.target_y}, {self.target_z})'
        )
        self.get_logger().info('⏳ 等待讀到無人機當前位置...')

    # ------------------------------------------------------------
    # Callback: 每次收到 /gt_pose 訊息就更新當前位置
    # ------------------------------------------------------------
    def pose_callback(self, msg: Pose):
        """每次 Gazebo 發布無人機位置,這個函式就會被呼叫。"""
        self.current_x = msg.position.x
        self.current_y = msg.position.y
        self.current_z = msg.position.z

    # ------------------------------------------------------------
    # 控制迴圈: 每 0.1 秒執行一次
    # ------------------------------------------------------------
    def control_loop(self):
        """核心控制邏輯。"""
        # 如果還沒讀到位置,先不動作
        if self.current_x is None:
            return

        # ===== Step 1: 計算 A 到 B 的向量 (誤差) =====
        error_x = self.target_x - self.current_x
        error_y = self.target_y - self.current_y
        error_z = self.target_z - self.current_z

        # ===== Step 2: 計算還差多遠 (歐幾里得距離) =====
        distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)

        # ===== Step 3: 判斷是否抵達 =====
        if distance < self.tolerance:
            if not self.arrived:
                self.get_logger().info(
                    f'✅ 抵達目標! 最終位置: '
                    f'({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})'
                )
                self.arrived = True
            # 抵達後發送停止指令 (速度全部設 0)
            self.publish_velocity(0.0, 0.0, 0.0)
            return

        # 如果曾經抵達過但又飄走,重置旗標
        self.arrived = False

        # ===== Step 4: 用比例控制 (P control) 計算速度 =====
        # 誤差越大 → 速度越快;誤差越小 → 速度越慢,自動減速
        vx = self.Kp * error_x
        vy = self.Kp * error_y
        vz = self.Kp * error_z

        # ===== Step 5: 限制最大速度 (避免過快衝過頭) =====
        vx, vy, vz = self.clamp_velocity(vx, vy, vz)

        # ===== Step 6: 發布速度指令 =====
        self.publish_velocity(vx, vy, vz)

        # 每秒印一次 debug 資訊 (避免洗版)
        self.get_logger().info(
            f'📍 位置: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | '
            f'剩餘: {distance:.2f}m | 速度: ({vx:.2f}, {vy:.2f}, {vz:.2f})',
            throttle_duration_sec=1.0
        )

    # ------------------------------------------------------------
    # 輔助函式: 發布速度指令到 /cmd_vel
    # ------------------------------------------------------------
    def publish_velocity(self, vx: float, vy: float, vz: float):
        """把 x,y,z 速度打包成 Twist 訊息並發布。"""
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz
        # 不需要旋轉,所以 angular 全設 0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------
    # 輔助函式: 限制速度不超過 max_speed
    # ------------------------------------------------------------
    def clamp_velocity(self, vx: float, vy: float, vz: float):
        """
        如果速度向量長度超過 max_speed,就把它縮短到 max_speed。
        這樣方向不變,只是變慢,不會超速衝過頭。
        """
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
            vz *= scale
        return vx, vy, vz


# ================================================================
# 程式進入點
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = FlyStraight()
    try:
        rclpy.spin(node)   # 持續執行,直到 Ctrl+C
    except KeyboardInterrupt:
        node.get_logger().info('🛑 使用者中斷,停止控制器。')
    finally:
        # 結束前發送一個停止指令,讓無人機停住
        node.publish_velocity(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
