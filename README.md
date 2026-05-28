# nsysu_drone

以 ROS 2 + Gazebo Classic 為基礎的四旋翼無人機模擬套件，源自 [tum_simulator](http://wiki.ros.org/tum_simulator)。

本版本內建完整的 Docker 工作流程，將 Gazebo 與 RViz 執行在容器內，透過 GPU 加速 3D 渲染，並以 VNC 將圖形介面傳送給使用者。適合沒有本地 X server 或僅能 SSH 連線的伺服器環境使用。

---

## 目錄

- [環境需求](#環境需求)
- [Repository 結構](#repository-結構)
- [啟動（Docker + VNC）](#啟動docker--vnc)
- [WSL 使用方式](#wsl-使用方式)
- [安裝 Python 套件](#安裝-python-套件)
- [訓練 RL 模型（train.py）](#訓練-rl-模型trainpy)
- [測試模型（test.py）](#測試模型testpy)
- [無人機 Topics](#無人機-topics)
- [常用指令](#常用指令)
- [設定 Plugin 參數](#設定-plugin-參數)
- [非 Docker 安裝（Native）](#非-docker-安裝native)
- [常見問題排除](#常見問題排除)
- [參考資料](#參考資料)

---

## 環境需求

測試環境：

| 項目 | 規格 |
|---|---|
| **Host OS** | Ubuntu 22.04 / Windows 11（WSL2） |
| **NVIDIA Driver** | 525+（已驗證 580） |
| **GPU** | 支援 OpenGL/EGL（已驗證 RTX A6000、RTX 5080） |
| **Docker** | 24.x + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **ROS 2** | iron（預設），亦支援 humble、rolling |
| **VNC Client** | 任意 VNC Viewer，推薦 [TurboVNC Viewer](https://sourceforge.net/projects/turbovnc/)（延遲最低） |

---

## Repository 結構

```
.
├── Dockerfile                   # 以 ROS 2 為基底建置容器
├── run_docker.sh                # 一鍵啟動容器（含 GPU / port 設定）
├── nsysu_drone_description/     # URDF、meshes、Gazebo world、感測器 plugins
├── nsysu_drone_bringup/         # Launch 檔案、RViz config、參數檔
├── nsysu_drone_control/         # Teleop 與控制節點
├── drone_env.py                 # Gymnasium RL 環境（含 ROS 2 介面）
├── train.py                     # PPO 課程學習訓練腳本
└── test.py                      # 模型評估與 Baseline 比較腳本
```

---

## 啟動（Docker + VNC）

### 1. 建置 Docker Image

第一次建置約需 15～30 分鐘。

```bash
cd /path/to/nsysu_drone
docker build -t nsysu_drone_vnc:iron .
```

若需要其他 ROS 2 版本：

```bash
docker build --build-arg ROS_DISTRO=humble  -t nsysu_drone_vnc:humble .
docker build --build-arg ROS_DISTRO=rolling -t nsysu_drone_vnc:rolling .
```

### 2. 設定 SSH Tunnel

VNC 流量透過 SSH 加密傳輸，不需對外開放任何 port。

**OpenSSH（Linux/macOS/WSL）：**

```bash
ssh -L 5901:localhost:5901 user@remote-host
```

**PuTTY（Windows）：**

> Session → Connection → SSH → Tunnels  
> Source port: `5901` / Destination: `localhost:5901` / Type: Local → **Add**

### 3. 啟動容器

在遠端主機執行：

```bash
./run_docker.sh
```

透過環境變數指定 GPU 或 port：

```bash
GPU_ID=3 ./run_docker.sh                    # 使用 GPU 3（預設）
GPU_ID=4 VNC_PORT=5902 ./run_docker.sh      # 使用 GPU 4，對應 host port 5902
```

容器啟動後會顯示：

```
======================================================
 TurboVNC ready  (NSYSU Drone)
   * Host port : 5901  (tunnel via SSH)
   * Password  : nsysudrone
   * Display   : :1 (1920x1080)
   * VGL_DISPLAY=egl  (NVIDIA GPU hardware rendering)
======================================================
```

### 4. 以 VNC Viewer 連線

推薦使用 **[TurboVNC Viewer](https://sourceforge.net/projects/turbovnc/)**，相較一般 VNC client 延遲更低、畫質更佳。

連線資訊：

- **Server**：`localhost:5901`
- **Password**：`nsysudrone`

連線後會出現 XFCE 桌面，Gazebo、RViz、xterm 等 GUI 視窗均在此環境內運行。

### 5. 啟動模擬

在容器的 shell 中執行：

```bash
launch_drone
```

---

## WSL 使用方式

若使用 Windows 的 WSL2 環境，流程如下：

### 查詢可用 GPU

```bash
nvidia-smi --list-gpus
```

輸出範例：

```
GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-xxxxxxxx)
GPU 1: NVIDIA GeForce RTX 3080 (UUID: GPU-yyyyyyyy)
```

根據輸出的編號，在啟動指令中指定 `GPU_ID`。

### 啟動容器

```bash
user@DESKTOP:~$ cd Assignment2/
user@DESKTOP:~/Assignment2$ GPU_ID=0 ./run_docker.sh
```

### 連線 VNC

WSL2 與 Windows 共享 localhost，因此直接開啟 TurboVNC Viewer 連線：

- **Server**：`localhost:5901`
- **Password**：`nsysudrone`

> 如果 WSL2 的 localhost 無法連線，可嘗試將 Server 改為 WSL2 的實際 IP（執行 `ip addr show eth0` 查詢）。

---

## 安裝 Python 套件

進入容器後，安裝訓練所需的 Python 套件：

```bash
apt update && apt install python3-pip -y

# 安裝核心套件（指定 numpy<2 以避免與 stable-baselines3 衝突）
pip install "numpy<2"
pip install stable-baselines3 gymnasium matplotlib tensorboard
```

> **版本說明**：`numpy>=2.0` 與 stable-baselines3 目前存在 API 不相容問題，需固定使用 `numpy<2`。

---

## 訓練 RL 模型（train.py）

### 環境說明（drone_env.py）

`DroneGymEnv` 是符合 Gymnasium 介面的 RL 環境，包含以下設計：

| 項目 | 說明 |
|---|---|
| **任務** | 隨機目標導航（Random Target Navigation） |
| **觀測空間** | 13 維：`[pos(3), target(3), rel_pos(3), vel(3), distance(1)]` |
| **動作空間** | 3 維速度指令 `[vx, vy, vz]`，範圍 `[-1.0, 1.0]` m/s |
| **成功判定** | 在目標圈內（dist < ARRIVE\_DIST）且速度低於 HOVER\_MAX\_SPEED，連續達到 HOVER\_STEPS 步 |
| **課程學習** | Level 1～7，難度逐步提升（目標距離、邊界範圍、懸停時間要求） |

**課程難度一覽：**

| Level | 目標範圍 XY | 到達距離 | 懸停步數 | 最大懸停速度 |
|---|---|---|---|---|
| L1 | ±1.0 m | 1.50 m | 5 步 | 0.80 m/s |
| L2 | ±2.5 m | 1.20 m | 8 步 | 0.60 m/s |
| L3 | ±4.0 m | 1.00 m | 11 步 | 0.45 m/s |
| L4 | ±5.5 m | 0.90 m | 14 步 | 0.35 m/s |
| L5 | ±7.0 m | 0.80 m | 16 步 | 0.28 m/s |
| L6 | ±8.5 m | 0.70 m | 18 步 | 0.24 m/s |
| L7 | ±10.0 m | 0.60 m | 20 步 | 0.20 m/s |

### 執行訓練

確保 Gazebo 模擬已在 VNC 桌面內啟動，再開啟另一個 terminal 執行：

```bash
# 從 Level 1 開始（預設）
python3 train.py

# 從指定 Level 繼續訓練（會自動載入對應模型）
python3 train.py --level 3
```

### 輸出檔案

| 路徑 | 說明 |
|---|---|
| `models/model_level{N}.zip` | 各 Level 的模型權重 |
| `ppo_drone_curriculum_resume.zip` | 強制終止後的存檔點（可中斷後繼續） |
| `best_model.zip` | 最新存檔點（可中斷後繼續） |
| `logs/rewards.csv` | 每回合獎勵紀錄 |
| `logs/training_curve.png` | 全局學習曲線 |
| `logs/training_console.log` | 終端機輸出 |
| `logs/rewards.csv` | 每回合紀錄 |
| `logs/level{N}/` | 各 Level 的詳細數據與曲線 |


### 訓練升級機制

每 50 個回合進行 30 次確定性評估（`deterministic=True`）：

- 成功率 ≥ **65%** → 自動升至下一 Level
- Level 7 成功率 ≥ **80%** → 訓練完成，自動儲存並停止

---

## 測試模型（test.py）

### 執行測試

```bash
# 使用預設模型（best_model.zip），Level 1，10 個 episode
python3 test.py

# 指定模型、Level、episode 數
python3 test.py --model models/model_level5 --level 5 --episodes 20

# 使用隨機策略（預設為 deterministic）
python3 test.py --stochastic

# 與 P 控制器 Baseline 比較
python3 test.py --baseline --level 3 --episodes 15
```

### 參數說明

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--model` | `best_model` | 模型檔名（不含 `.zip`） |
| `--level` | `1` | 課程難度（1～7） |
| `--episodes` | `10` | 測試回合數 |
| `--baseline` | `False` | 改跑 P 控制器 Baseline |
| `--stochastic` | `False` | 使用隨機策略（預設 deterministic） |

### 輸出指標

| 指標 | 說明 |
|---|---|
| **成功率** | 完成 HOVER\_STEPS 步懸停的比例 |
| **EffectiveSpeed** | `initial_dist / ep_steps`，越高代表飛得越快越穩 |
| **Mean Reward** | 平均每回合總獎勵 |
| **Mean Steps** | 平均每回合步數 |

---

## 無人機 Topics

預設命名空間為 `/simple_drone`（可設定，參見[設定 Plugin 參數](#設定-plugin-參數)）。

### 感測器（Published）

| Topic | 訊息型別 |
|---|---|
| `~/front/image_raw` | `sensor_msgs/msg/Image` |
| `~/bottom/image_raw` | `sensor_msgs/msg/Image` |
| `~/sonar/out` | `sensor_msgs/msg/Range` |
| `~/imu/out` | `sensor_msgs/msg/Imu` |
| `~/gps/nav` | `sensor_msgs/msg/NavSatFix` |
| `~/gps/vel` | `geometry_msgs/msg/TwistStamped` |
| `~/joint_states` | `sensor_msgs/msg/JointState` |

### 控制（Subscribed）

| Topic | 訊息型別 | 功能 |
|---|---|---|
| `~/takeoff` | `std_msgs/msg/Empty` | 起飛 |
| `~/land` | `std_msgs/msg/Empty` | 降落 |
| `~/cmd_vel` | `geometry_msgs/msg/Twist` | 速度控制 |
| `~/reset` | `std_msgs/msg/Empty` | 重置位置 |
| `~/posctrl` | `std_msgs/msg/Bool` | 切換位置/速度控制模式 |
| `~/dronevel_mode` | `std_msgs/msg/Bool` | 切換速度/傾角控制 |
| `~/cmd_mode` | `std_msgs/msg/Bool` | 發布當前控制模式 |
| `~/state` | `std_msgs/msg/Int8` | 狀態（0=降落, 1=飛行, 2=懸停） |

### Ground Truth，Published

| Topic | 訊息型別 |
|---|---|
| `~/gt_pose` | `geometry_msgs/msg/Pose` |
| `~/gt_vel` | `geometry_msgs/msg/Twist` |
| `~/gt_acc` | `geometry_msgs/msg/Twist` |

---

## 常用指令

開啟容器的第二個 shell：

```bash
docker exec -it nsysu_drone_vnc bash
```

基本控制：

```bash
# 起飛
ros2 topic pub /simple_drone/takeoff std_msgs/msg/Empty {} --once

# 降落
ros2 topic pub /simple_drone/land std_msgs/msg/Empty {} --once

# 前進（0.3 m/s）
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --once

# 重置
ros2 topic pub /simple_drone/reset std_msgs/msg/Empty {} --once
```

---

## 設定 Plugin 參數

編輯 `nsysu_drone_bringup/config/` 下對應的 YAML 檔，再重新 build workspace：

```yaml
namespace: /simple_drone

rollpitchProportionalGain: 10.0
rollpitchDifferentialGain: 5.0
rollpitchLimit: 0.5

yawProportionalGain: 5.0
yawDifferentialGain: 1.0
yawLimit: 3.0

velocityXYProportionalGain: 10.0
velocityXYDifferentialGain: 4.0
velocityXYLimit: 2

velocityZProportionalGain: 5.0
velocityZIntegralGain: 0.0
velocityZDifferentialGain: 1.0
velocityZLimit: -1

positionXYProportionalGain: 1.1
positionXYDifferentialGain: 0.0
positionXYIntegralGain: 0.0
positionXYLimit: 5

positionZProportionalGain: 1.0
positionZDifferentialGain: 0.2
positionZIntegralGain: 0.0
positionZLimit: -1

maxForce: 50
motionSmallNoise: 0.00
motionDriftNoise: 0.00
motionDriftNoiseTime: 50
```

---

## 非 Docker 安裝（Native）

若不使用 Docker，可直接在主機上安裝：

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone <your-repo-url> nsysu_drone
cd ~/ros2_ws
rosdep install -r -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
colcon build --packages-select-regex nsysu.*
source install/setup.bash
ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py
```

請確認已安裝 Gazebo 通用模型，詳見 [`nsysu_drone_description/README.md`](./nsysu_drone_description/README.md)。

---

## 常見問題排除

**`cannot open display` / `GLX` 錯誤**

未使用 `vglrun` 直接啟動 GUI 程式。請務必使用 `launch_drone` alias 或在指令前加上 `vglrun`。

**`[VGL] ERROR: Could not open display :0.`**

VirtualGL 找不到 X server。確認 `VGL_DISPLAY=egl` 已設定（Dockerfile 會自動寫入 `/root/.bashrc`）。

**`Unable to start server [bind: Address already in use]`**

前一次 Gazebo 未正常關閉，執行以下指令後重試：

```bash
pkill -9 gzserver gzclient rviz2
```

**Gazebo 畫面卡頓**

確認硬體加速是否正常運作：

```bash
vglrun glxinfo | grep "OpenGL renderer"
```

預期結果應顯示 GPU 型號，若顯示 `llvmpipe` 代表未使用 GPU 渲染，請確認 `VGL_DISPLAY=egl` 與 `--gpus` 參數是否正確設定。

**`colcon build` 缺少依賴**

通常是 `rosdep` 快取過舊，在容器內執行：

```bash
rosdep update
rosdep install --from-paths /ros2_ws/src --ignore-src -r -y
```

**VNC 連線後黑畫面**

XFCE session 可能已崩潰，在容器 shell 中執行：

```bash
export DISPLAY=:1
dbus-launch xfce4-session &
```

---

## 參考資料

- 上游專案：[tum_simulator](http://wiki.ros.org/tum_simulator)
- VirtualGL：<https://virtualgl.org/>
- TurboVNC：<https://turbovnc.org/>
- NVIDIA Container Toolkit：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/>
- Tan & Karaköse (2023). [PPO-based distributed deep reinforcement learning for drone tracking](<refer/A new approach for drone tracking with drone using Proximal Policy Optimization based distributed deep reinforcement learning.pdf>)
- Zhang et al (2024). [AirPilot: Interpretable PPO-based DRL Auto-Tuned Nonlinear PID Drone Controller](<refer/AirPilot Interpretable PPO-based DRL Auto Tuned Nonlinear PID Drone Controller for Robust Autonomous Flights .pdf>)
- Shen & Huang (2024). [Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions](<refer/Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions.pdf>)