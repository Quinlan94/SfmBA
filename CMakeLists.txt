cmake_minimum_required( VERSION 2.8 )
project( SFM )


set( CMAKE_CXX_FLAGS "-std=c++11" )
set(CMAKE_BUILD_TYPE "Debug")


find_package( OpenCV REQUIRED )
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Ceres REQUIRED)
find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#这个库好像有点问题
find_package(PCL REQUIRED COMPONET common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${DEFINITIONS})
list (REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
# 添加头文件
#include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PROJECT_SOURCE_DIR}/include )
include_directories(/usr/include/eigen3)

add_subdirectory(src)


add_executable( sfm main.cpp include/Feature.h)

target_link_libraries( sfm mysfm ${OpenCV_LIBS}   ${PCL_LIBRARIES} ${CERES_LIBRARIES})
