# GS-LIVO 安装与配置

Ubuntu 20.04 + ROS Noetic + CUDA 环境下配置、编译和运行 GS-LIVO 

## 1. 已验证环境

- Ubuntu 20.04.6 LTS
- ROS Noetic
- CMake 3.24.4
- CUDA 12.8
- PyTorch/libtorch 2.8.0 + CUDA 12.8
- Sophus non-templated/double-only 版本，安装位置为 `/usr/local/lib/libSophus.so`
- Livox ROS Driver workspace：`/home/lurvelly/Workspace/Livox-ROS-Driver/devel`

## 2. 系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential git cmake python3-dev python3-pip \
  ros-noetic-desktop-full \
  ros-noetic-cv-bridge ros-noetic-image-transport \
  ros-noetic-pcl-ros ros-noetic-eigen-conversions \
  libopencv-dev libpcl-dev libeigen3-dev libboost-all-dev \
  libtbb-dev libyaml-cpp-dev
```

如果 ROS 已安装，只需要补齐缺失的 `ros-noetic-*` 和 C++ 依赖即可。

## 3. Sophus

本项目使用 FAST-LIVO2/vikit 依赖的旧版 Sophus API，例如 `Sophus::SE3`。不要使用只有 `Sophus::SE3d`/模板 API 的新版 header-only Sophus。

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

检查：

```bash
test -f /usr/local/lib/libSophus.so
test -f /usr/local/include/sophus/se3.h
```

## 4. Torch/libtorch

任选一种方式提供 Torch CMake 包。

方式 A：使用已有 Python torch：

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.utils.cmake_prefix_path)
PY
```

编译时把 `Torch_DIR` 指向输出路径下的 `Torch` 目录，例如本机验证用：

```bash
export Torch_DIR=/home/lurvelly/miniconda3/envs/yolo/lib/python3.10/site-packages/torch/share/cmake/Torch
```

方式 B：下载官方 libtorch，并设置：

```bash
export Torch_DIR=/path/to/libtorch/share/cmake/Torch
```

Torch 的 CUDA 版本应与系统 CUDA 主版本兼容。本机使用 CUDA 12.8 + torch 2.8.0 cu128。

## 5. Livox ROS Driver

先编译 Livox ROS Driver，并让 GS-LIVO overlay 它：

```bash
cd /home/lurvelly/Workspace/Livox-ROS-Driver
catkin_make
source devel/setup.zsh
```

新机器上路径不同的话，后续 `CMAKE_PREFIX_PATH` 中替换为你的 Livox workspace `devel` 路径。

## 6. rpg_vikit

`src/rpg_vikit` 是本项目需要的 catkin 依赖。当前仓库把它记录为 submodule。重新克隆后先初始化它，再应用本仓库提供的适配 patch：

```bash
cd /home/lurvelly/Workspace/GS-LIVO
git submodule update --init src/rpg_vikit
git -C src/rpg_vikit apply ../../patches/rpg_vikit_gs_livo.patch
```

本项目的 CMake 已对 vikit 做了两点适配：

- 显式链接 `/usr/local/lib/libSophus.so`
- 默认关闭 vikit 测试可执行文件，只构建 `vikit_common` 和 `vikit_ros`

## 7. 编译 GS-LIVO

在仓库根目录执行：

```bash
cd /home/lurvelly/Workspace/GS-LIVO
source /opt/ros/noetic/setup.zsh

export Torch_DIR=/home/lurvelly/miniconda3/envs/yolo/lib/python3.10/site-packages/torch/share/cmake/Torch
export TORCH_CUDA_ARCH_LIST=8.9
export CMAKE_PREFIX_PATH=/home/lurvelly/Workspace/Livox-ROS-Driver/devel:/opt/ros/noetic

catkin_make -j4 -DTorch_DIR=$Torch_DIR
```

`TORCH_CUDA_ARCH_LIST` 需要按 GPU 调整：

- RTX 4090：`8.9`
- RTX 30 系列：`8.6`
- Jetson Orin：`8.7`

编译成功后应生成：

```bash
devel/lib/fast_livo/fastlivo_mapping
devel/lib/lib3dgs_lib.so
devel/lib/libvikit_common.so
devel/lib/libvikit_ros.so
```

## 8. 验证

```bash
source devel/setup.zsh
rospack find fast_livo
roslaunch --files fast_livo mapping_avia.launch
roslaunch --nodes fast_livo mapping_avia.launch
ldd devel/lib/fast_livo/fastlivo_mapping | grep 'not found' || true
```

本机验证结果：

- `catkin_make -j4 ...` 成功，`fastlivo_mapping` 构建到 100%
- `rospack find fast_livo` 返回 `src/gs-livo`
- `roslaunch --files fast_livo mapping_avia.launch` 能找到 launch 文件
- `roslaunch --nodes fast_livo mapping_avia.launch` 返回 `/laserMapping`、`/rviz`、`/republish`
- `ldd` 未发现 `not found`

直接 `rosrun fast_livo fastlivo_mapping` 或 `roslaunch` 需要 ROS master/socket 权限；在当前沙箱中网络 socket 被禁用，因此未做在线运行验证。

## 9. 运行

```bash
cd /home/lurvelly/Workspace/GS-LIVO
source devel/setup.zsh
roslaunch fast_livo mapping_avia.launch
```

如果不需要 RViz：

```bash
roslaunch fast_livo mapping_avia.launch rviz:=false
```

默认话题在 `src/gs-livo/config/avia.yaml`：

- Image: `/left_camera/image`
- LiDAR: `/livox/lidar`
- IMU: `/livox/imu`

使用自己的 rosbag 或设备时，需要同步修改 `avia.yaml` 中的话题、外参、时间偏移和相机内参文件 `camera_pinhole.yaml`。


