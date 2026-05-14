#!/bin/bash
# ---------------------------------------------------------------
# 啟動 nsysu_drone container (TurboVNC + VirtualGL + NVIDIA GPU)
#
# 使用方式:
#   ./run_docker.sh                # 預設用 GPU 3, VNC port 5901
#   GPU_ID=4 ./run_docker.sh       # 改用 GPU 4
#   VNC_PORT=5902 ./run_docker.sh  # 改用 port 5902
# ---------------------------------------------------------------

set -e

GPU_ID=${GPU_ID:-3}
VNC_PORT=${VNC_PORT:-5901}
IMAGE=${IMAGE:-nsysu_drone_vnc:iron}
CONTAINER_NAME=${CONTAINER_NAME:-nsysu_drone_vnc}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "=================================================="
echo " Launching ${CONTAINER_NAME}"
echo "   Image     : ${IMAGE}"
echo "   GPU       : ${GPU_ID}"
echo "   VNC port  : ${VNC_PORT} (host) -> 5901 (container)"
echo "=================================================="

docker run \
    -it --rm \
    --gpus "\"device=${GPU_ID}\"" \
    -p ${VNC_PORT}:5901 \
    --env=QT_X11_NO_MITSHM=1 \
    --privileged \
    --name="${CONTAINER_NAME}" \
    --hostname="$(hostname)" \
    "${IMAGE}"
