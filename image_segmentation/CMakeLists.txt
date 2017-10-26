cmake_minimum_required(VERSION 2.8.3)  
project(image_segmentation)  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")  
if(WIN32 AND NOT CYGWIN)  
    set(HOME $ENV{PROFILE})  
else()  
    set(HOME $ENV{HOME})  
endif()  
  
find_package(catkin REQUIRED COMPONENTS  
 roscpp  
 rospy  
 std_msgs  
  
 sensor_msgs  
 cv_bridge  
 image_transport  
)  
 
set(OpenCV_DIR /opt/ros/kinetic/include)    
find_package(OpenCV REQUIRED)  



catkin_package(CATKIN_DEPENDS  
roscpp  
std_msgs  
sensor_msgs  
)  
  
include_directories(  
include  
${OpenCV_INCLUDE_DIRS}
${catkin_INCLUDE_DIRS}  
)  

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS}) 

  
### find files  
  
#file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*)  
#file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)  
  
add_executable(depth_seg  src/depth_seg.cpp
${HEADER_FILES}  
${SOURCE_FILES}  
)  
  
target_link_libraries(depth_seg  
${catkin_LIBRARIES}  
${OpenCV_LIBRARIES}  
${PCL_LIBRARIES} 
)  


