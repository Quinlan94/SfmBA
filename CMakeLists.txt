cmake_minimum_required( VERSION 2.8 )
project( SFM )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory(src)

add_executable( sfm main.cpp )
# 链接OpenCV库
target_link_libraries( sfm mysfm ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
