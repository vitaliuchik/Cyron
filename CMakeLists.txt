cmake_minimum_required(VERSION 3.17)
project(Cyron)

set(CMAKE_CXX_STANDARD 17)

add_executable(Cyron main.cpp src/standardnn.cpp includes/standardnn.h src/exploration.cpp includes/exploration.h
        includes/time_counting.h)

TARGET_LINK_LIBRARIES(Cyron -ltbb)