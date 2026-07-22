cmake_minimum_required(VERSION 3.50)
project(nn_modulo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置优化选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 添加头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 收集源文件（排除 main.cpp 和 predict.cpp）
file(GLOB COMMON_SOURCES 
    "src/matrix.cpp"
    "src/network.cpp"
    "src/layer.cpp"
    "src/activations.cpp"
    "src/utils.cpp"
)

# 训练程序
add_executable(nn_modulo 
    src/main.cpp
    ${COMMON_SOURCES}
)

# 预测程序
add_executable(nn_predict 
    src/predict.cpp
    ${COMMON_SOURCES}
)

# 设置输出目录
set_target_properties(nn_modulo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

set_target_properties(nn_predict PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)