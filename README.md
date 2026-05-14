# nsysu_drone

A quadrotor simulation package for ROS 2 + Gazebo Classic, derived from [tum_simulator](http://wiki.ros.org/tum_simulator).

This version is maintained at National Sun Yat-sen University (NSYSU) and ships with a self-contained Docker workflow that runs the whole simulator (Gazebo + RViz + teleop) inside a container, with 3D rendering accelerated by GPUs and the GUI delivered to the user over VNC. It is designed for headless servers accessed via SSH, so users do not need a local X server or X11 forwarding.

# Requirements

Tested on:
- **Host**: Ubuntu 22.04, NVIDIA driver 525+ (driver 580 verified)
- **GPU**: Any GPU with OpenGL/EGL support (Nvidia RTX A6000, RTX 5080 verified)
- **Docker**: 24.x with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- **ROS 2**: iron (default), humble and rolling also supported via `--build-arg`
- **Client**: Any VNC viewer; [TurboVNC Viewer](https://sourceforge.net/projects/turbovnc/) recommended for best performance

# Repository layout

```
.
├── Dockerfile                   # Builds the container from ROS 2 base + source
├── run_docker.sh                # Convenience launcher with GPU/port flags
├── nsysu_drone_description/     # URDF, meshes, Gazebo world, sensor plugins
├── nsysu_drone_bringup/         # Launch files, RViz configs, parameters
└── nsysu_drone_control/         # Teleop and control nodes
```

# Quick start (Docker + VNC)

### 1. Build the image

The image is built entirely from source. Expected build time is 15–30 minutes on first build.

```bash
cd /path/to/nsysu_drone
docker build -t nsysu_drone_vnc:iron .
```

To build against a different ROS 2 distro:

```bash
docker build --build-arg ROS_DISTRO=humble  -t nsysu_drone_vnc:humble .
docker build --build-arg ROS_DISTRO=rolling -t nsysu_drone_vnc:rolling .
```

### 2. Set up the SSH tunnel

VNC traffic is routed through SSH so no ports need to be publicly exposed. On your local machine (PuTTY example):

> Session → Connection → SSH → Tunnels
> Source port: `5901`
> Destination: `localhost:5901`
> Type: Local → **Add**

For OpenSSH users:

```bash
ssh -L 5901:localhost:5901 user@remote-host
```

### 3. Launch the container

On the remote host:

```bash
./run_docker.sh
```

Environment variables control GPU selection and port mapping:

```bash
GPU_ID=3  ./run_docker.sh                    # use GPU 3 (default)
GPU_ID=4  VNC_PORT=5902 ./run_docker.sh      # use GPU 4, map to host port 5902
```

When ready, the container prints:

```
======================================================
 TurboVNC ready  (NSYSU Drone)
   * Host port : 5901  (tunnel via SSH)
   * Password  : nsysudrone
   * Display   : :1 (1920x1080)
   * VGL_DISPLAY=egl  (NVIDIA GPU hardware rendering)
======================================================
```

### 4. Connect with a VNC viewer

On your local machine, open TurboVNC Viewer (or any VNC client) and connect to:

- **Server**: `localhost:5901`
- **Password**: `nsysudrone`

An XFCE desktop will appear — all GUI windows (Gazebo, RViz, xterm) will spawn inside it.

### 5. Start the simulation

- Inside the container's shell (the one you launched with `run_docker.sh`):

```bash
launch_drone
```

- This is an alias for:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py
```

- `vglrun` intercepts OpenGL calls and routes them to the NVIDIA GPU for hardware-accelerated rendering. **Always prefix GUI ROS commands with `vglrun`** — running Gazebo or RViz without it will fail with GLX errors.

- Use teleop window to move the drone.

# Drone topics

The namespace is `/simple_drone` by default (configurable, see [Configure Plugin](#configure-plugin)).

### Sensors

| Topic | Type |
|---|---|
| `~/front/image_raw` | `sensor_msgs/msg/Image` |
| `~/bottom/image_raw` | `sensor_msgs/msg/Image` |
| `~/sonar/out` | `sensor_msgs/msg/Range` |
| `~/imu/out` | `sensor_msgs/msg/Imu` |
| `~/gps/nav` | `sensor_msgs/msg/NavSatFix` |
| `~/gps/vel` | `geometry_msgs/msg/TwistStamped` |
| `~/joint_states` | `sensor_msgs/msg/JointState` |

### Control (subscribed)

| Topic | Type | Purpose |
|---|---|---|
| `~/takeoff` | `std_msgs/msg/Empty` | Start the drone |
| `~/land` | `std_msgs/msg/Empty` | Land the drone |
| `~/cmd_vel` | `geometry_msgs/msg/Twist` | Steer the drone |
| `~/reset` | `std_msgs/msg/Empty` | Reset the drone |
| `~/posctrl` | `std_msgs/msg/Bool` | Toggle position control (give drone a pose via cmd_vel) and normal control (only use cmd_vel) |
| `~/dronevel_mode` | `std_msgs/msg/Bool` | Toggle velocity vs. tilt control (normal mode) |
| `~/cmd_mode` | `std_msgs/msg/Bool` | Publishes current control mode |
| `~/state` | `std_msgs/msg/Int8` | Current state (0 = landed, 1 = flying, 2 = hovering) |

### Ground truth (published)

| Topic | Type |
|---|---|
| `~/gt_pose` | `geometry_msgs/msg/Pose` |
| `~/gt_vel` | `geometry_msgs/msg/Twist` |
| `~/gt_acc` | `geometry_msgs/msg/Twist` |

# Common commands

Once the simulation is running, any of these can be executed from a second terminal (either in the VNC desktop's xterm, or by execing into the container):


To open a second shell inside the running container:

```bash
docker exec -it nsysu_drone_vnc bash
```


```bash
# Take off
ros2 topic pub /simple_drone/takeoff std_msgs/msg/Empty {} --once

# Land
ros2 topic pub /simple_drone/land std_msgs/msg/Empty {} --once

# Send a velocity command (forward at 0.3 m/s)
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --once

# Reset
ros2 topic pub /simple_drone/reset std_msgs/msg/Empty {} --once
```



# Configure plugin

The `plugin_drone` plugin accepts the following parameters. Edit them in the appropriate YAML under `nsysu_drone_bringup/config/` and rebuild the workspace.

```yaml
# ROS namespace. All topics and tf frames are prefixed with this.
namespace: /simple_drone

# Roll/pitch PID
rollpitchProportionalGain: 10.0
rollpitchDifferentialGain: 5.0
rollpitchLimit: 0.5

# Yaw PID
yawProportionalGain: 2.0
yawDifferentialGain: 1.0
yawLimit: 1.5

# Horizontal velocity PID
velocityXYProportionalGain: 5.0
velocityXYDifferentialGain: 2.3
velocityXYLimit: 2

# Vertical velocity PID
velocityZProportionalGain: 5.0
velocityZIntegralGain: 0.0
velocityZDifferentialGain: 1.0
velocityZLimit: -1

# Horizontal position PID
positionXYProportionalGain: 1.1
positionXYDifferentialGain: 0.0
positionXYIntegralGain: 0.0
positionXYLimit: 5

# Vertical position PID
positionZProportionalGain: 1.0
positionZDifferentialGain: 0.2
positionZIntegralGain: 0.0
positionZLimit: -1

# Physical limits / noise
maxForce: 30
motionSmallNoise: 0.00
motionDriftNoise: 0.00
motionDriftNoiseTime: 50
```

# Native (non-Docker) install

If you prefer to run directly on the host:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone <your-repo-url> nsysu_drone
cd ~/ros2_ws
rosdep install -r -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
colcon build --packages-select-regex nsysu.*
source install/setup.bash
ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py
```

Make sure the common Gazebo models are installed — see [`nsysu_drone_description/README.md`](./nsysu_drone_description/README.md).

# Troubleshooting

### `cannot open display` / `GLX` errors
Launching GUI apps without `vglrun` will fail because the container has no real display server for OpenGL. Always use the `launch_drone` alias or prefix commands with `vglrun`.

### `[VGL] ERROR: Could not open display :0.`
VirtualGL is trying to use an X server that doesn't exist. Make sure `VGL_DISPLAY=egl` is set (the Dockerfile adds this to `/root/.bashrc`, so it's automatic in interactive shells).

### `Unable to start server [bind: Address already in use]`
A previous Gazebo process didn't shut down cleanly. Run:
```bash
pkill -9 gzserver gzclient rviz2
```
then retry.

### Gazebo runs but feels slow / stutters
Check that hardware acceleration is actually engaged:
```bash
vglrun glxinfo | grep "OpenGL renderer"
```
Expected: `NVIDIA RTX A6000/…` (or your GPU).
If you see `llvmpipe`, EGL is not active — verify `VGL_DISPLAY=egl` and that the container was launched with `--gpus`.

### `colcon build` fails with missing dependencies
Usually caused by a stale `rosdep` cache. Inside the container:
```bash
rosdep update
rosdep install --from-paths /ros2_ws/src --ignore-src -r -y
```

### Black screen on VNC connect
The XFCE session may have crashed. In the container's shell:
```bash
export DISPLAY=:1
dbus-launch xfce4-session &
```

# References
- Upstream ancestor: [tum_simulator](http://wiki.ros.org/tum_simulator)
- VirtualGL: <https://virtualgl.org/>
- TurboVNC: <https://turbovnc.org/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/>
