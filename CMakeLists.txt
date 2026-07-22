cmake_minimum_required(VERSION 3.50)
project(nn_modulo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 优化编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 头文件路径
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# 主训练程序
add_executable(nn_modulo
    src/main.cpp
    src/network.cpp
    src/utils.cpp
    src/matrix.cpp
)
target_include_directories(nn_modulo PRIVATE ${INCLUDE_DIR})

# 独立推理程序
add_executable(predict
    src/predict.cpp
    src/network.cpp
    src/utils.cpp
    src/matrix.cpp
)
target_include_directories(predict PRIVATE ${INCLUDE_DIR})