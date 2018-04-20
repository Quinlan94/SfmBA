cmake_minimum_required( VERSION 2.8 )
project( SFM )


set( CMAKE_CXX_FLAGS "-std=c++11" )


find_package( OpenCV REQUIRED )
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory(src)

add_executable( sfm main.cpp )

target_link_libraries( sfm mysfm ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
