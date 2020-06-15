cmake_minimum_required(VERSION 3.17)
project(Cyron)

set(CMAKE_CXX_STANDARD 17)
#-Werror -Wall -Wextra
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(Cyron main.cpp src/standardnn.cpp includes/standardnn.h src/exploration.cpp includes/exploration.h
        includes/time_counting.h src/convolutionalnn.cpp includes/convolutionalnn.h tutorials/snn.cpp tutorials/snn.h tutorials/cnn.cpp tutorials/cnn.h)

TARGET_LINK_LIBRARIES(Cyron -ltbb)