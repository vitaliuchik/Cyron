cmake_minimum_required(VERSION 3.15)
project(Cyron)
#set(Eigen_INCLUDE_DIR "/usr/local/Cellar/eigen/3.3.7/include/")
set(CMAKE_CXX_STANDARD 17)
INCLUDE_DIRECTORIES(${Eigen_INCLUDE_DIR})

find_package(TBB 2020 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
find_package(Eigen3 REQUIRED)

SET(Cyron_FILES main.cpp src/standardnn.cpp includes/standardnn.h src/exploration.cpp includes/exploration.h
        includes/time_counting.h src/convolutionalnn.cpp includes/convolutionalnn.h tutorials/cnn.cpp
        tutorials/cnn.h tutorials/snn.cpp tutorials/snn.h)

add_library(Cyron_lib SHARED ${Cyron_FILES} )

TARGET_LINK_LIBRARIES(Cyron_lib Eigen3::Eigen -ltbb)