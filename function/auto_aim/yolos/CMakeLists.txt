cmake_minimum_required(VERSION 3.10)
project(rm_vision_system)

set(CMAKE_CXX_STANDARD 17)

# 1. 查找库
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

# 2. 海康 SDK 路径 (请根据实际安装位置修改)
set(MVS_INCLUDE_DIR /opt/MVS/include)
set(MVS_LIB_DIR /opt/MVS/lib/64) 

include_directories(${OpenCV_INCLUDE_DIRS} ${MVS_INCLUDE_DIR})
link_directories(${MVS_LIB_DIR})

# 3. 包含子模块源文件
file(GLOB_RECURSE SOURCES 
    "src/main.cpp"
    "src/modules/detector/*.cpp"
    # HikDriver 是头文件实现，不需要 .cpp
)

add_executable(rm_bot ${SOURCES})

# 4. 链接库
# 注意 MvCameraControl 和 X11 是必须的
target_link_libraries(rm_bot PRIVATE 
    ${OpenCV_LIBS} 
    openvino::runtime 
    MvCameraControl 
    X11 dl pthread
)