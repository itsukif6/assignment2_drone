ARG ROS_DISTRO=iron
FROM ros:${ROS_DISTRO}-ros-core-jammy
ARG ROS_DISTRO

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# ---------- 基本工具 ----------
RUN apt-get update && apt-get install -y \
    wget curl unzip \
    lsb-release \
    mesa-utils \
    build-essential \
    gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- Gazebo Classic + ROS 2 gazebo bridge ----------
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list && \
    wget -qO- http://packages.osrfoundation.org/gazebo.key | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
        gazebo \
        ros-${ROS_DISTRO}-gazebo-ros-pkgs \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-joint-state-publisher \
        ros-${ROS_DISTRO}-joint-state-publisher-gui \
        ros-${ROS_DISTRO}-teleop-twist-keyboard \
        ros-${ROS_DISTRO}-joy \
        ros-${ROS_DISTRO}-joy-teleop \
        xterm \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

# ---------- Copy nsysu_drone source ----------
RUN mkdir -p /ros2_ws/src
COPY ./nsysu_drone_description /ros2_ws/src/nsysu_drone_description
COPY ./nsysu_drone_bringup    /ros2_ws/src/nsysu_drone_bringup
COPY ./nsysu_drone_control    /ros2_ws/src/nsysu_drone_control

# ---------- Gazebo models ----------
RUN curl -fsSL https://github.com/osrf/gazebo_models/archive/refs/heads/master.zip \
        -o /tmp/gazebo_models.zip && \
    unzip -q /tmp/gazebo_models.zip -d /tmp && \
    mkdir -p /root/.gazebo/models && \
    mv /tmp/gazebo_models-master/* /root/.gazebo/models/ && \
    rm -rf /tmp/gazebo_models.zip /tmp/gazebo_models-master

# ---------- Build workspace ----------
WORKDIR /ros2_ws
RUN apt-get update && \
    bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
             rosdep install --from-paths src --ignore-src -r -y && \
             colcon build --symlink-install" && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
#   以上:ROS + Gazebo + nsysu_drone（自己 build 的 base)
#   以下:疊 TurboVNC + VirtualGL + 桌面環境
# ============================================================

# ---------- 桌面與 GL ----------
RUN apt-get update && apt-get install -y \
    xfce4 xfce4-terminal dbus-x11 \
    libglu1-mesa \
    libxv1 libxtst6 libxmu6 \
    libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- VirtualGL ----------
RUN wget -q https://github.com/VirtualGL/virtualgl/releases/download/3.1/virtualgl_3.1_amd64.deb \
        -O /tmp/virtualgl.deb && \
    apt-get update && apt-get install -y /tmp/virtualgl.deb && \
    rm /tmp/virtualgl.deb && rm -rf /var/lib/apt/lists/*

# ---------- TurboVNC ----------
RUN wget -q https://github.com/TurboVNC/turbovnc/releases/download/3.1.1/turbovnc_3.1.1_amd64.deb \
        -O /tmp/turbovnc.deb && \
    apt-get update && apt-get install -y /tmp/turbovnc.deb && \
    rm /tmp/turbovnc.deb && rm -rf /var/lib/apt/lists/*

# ---------- 驗證 ----------
RUN test -x /opt/TurboVNC/bin/vncserver && \
    test -x /opt/TurboVNC/bin/vncpasswd && \
    test -x /opt/VirtualGL/bin/vglrun && \
    echo "TurboVNC and VirtualGL installed OK"

ENV PATH=/opt/TurboVNC/bin:/opt/VirtualGL/bin:$PATH
ENV ROS_DISTRO=${ROS_DISTRO}

# ---------- bashrc ----------
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash"                                         >> /root/.bashrc && \
    echo "if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi" >> /root/.bashrc && \
    echo "export VGL_DISPLAY=egl"                                                           >> /root/.bashrc && \
    echo "export DISPLAY=:1"                                                                >> /root/.bashrc && \
    echo "alias launch_drone='vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py'" >> /root/.bashrc

# ---------- 啟動腳本 ----------
RUN printf '%s\n' \
'#!/bin/bash' \
'set -e' \
'mkdir -p /root/.vnc' \
'echo "nsysudrone" | /opt/TurboVNC/bin/vncpasswd -f > /root/.vnc/passwd' \
'chmod 600 /root/.vnc/passwd' \
'rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true' \
'/opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 -noxstartup' \
'export DISPLAY=:1' \
'export VGL_DISPLAY=egl' \
'dbus-launch xfce4-session >/dev/null 2>&1 &' \
'sleep 2' \
'echo ""' \
'echo "======================================================"' \
'echo " TurboVNC ready  (NSYSU Drone)"' \
'echo "   * Host port : 5901  (tunnel via SSH)"' \
'echo "   * Password  : nsysudrone"' \
'echo "   * Display   : :1 (1920x1080)"' \
'echo "   * VGL_DISPLAY=egl  (NVIDIA GPU hardware rendering)"' \
'echo ""' \
'echo " Quick start:"' \
'echo "   launch_drone     # vglrun ros2 launch nsysu_drone_bringup ..."' \
'echo "======================================================"' \
'echo ""' \
'exec bash' \
    > /start_vnc.sh && chmod +x /start_vnc.sh

WORKDIR /ros2_ws
CMD ["/start_vnc.sh"]
