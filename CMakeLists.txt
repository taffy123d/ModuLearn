cmake_minimum_required(VERSION 3.50)
project(nn_modulo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 包含头文件目录 (假设你的 matrix.h 和 network.h 在 include 文件夹下)
include_directories(include)

add_executable(nn_modulo 
    src/main.cpp 
    src/network.cpp 
    src/matrix.cpp   # 【关键】确保加上了你优秀的 matrix.cpp
    src/utils.cpp
)

add_executable(predict   
    src/predict.cpp 
    src/network.cpp 
    src/matrix.cpp   # 【关键】同上
    src/utils.cpp
)