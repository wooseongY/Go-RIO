# SPDX-License-Identifier: BSD-2-Clause
cmake_minimum_required(VERSION 2.8.3)
project(gorio)

# Can we use C++17 in indigo?
add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
# add_definitions("-Wall -g")
ADD_COMPILE_OPTIONS(-std=c++17 )
ADD_COMPILE_OPTIONS(-std=c++17 )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

# set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# pcl 1.7 causes a segfault when it is built with debug mode
set(CMAKE_BUILD_TYPE "RELEASE")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  pcl_ros
  geodesy
  nmea_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  interactive_markers
  fast_gicp REQUIRED
  image_transport
  cv_bridge
  # rosopencv
)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "version: ${OpenCV_VERSION}")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

find_package(G2O REQUIRED)
include_directories(SYSTEM ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})
# link_libraries(${G2O_LIBRARIES})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Ceres)

find_library(VGICP_CUDA_FOUND NAMES fast_vgicp_cuda)
message(STATUS "VGICP_CUDA_FOUND:" ${VGICP_CUDA_FOUND})
if(VGICP_CUDA_FOUND)
  add_definitions(-DUSE_VGICP_CUDA)
endif()

########################
## message generation ##
########################
add_service_files(FILES
  SaveMap.srv
  DumpGraph.srv
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES radar_graph_slam_nodelet
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
  DEPENDS system_lib OpenCV
)

###########
## Build ##
###########
include_directories(include)
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  "~/sensor_ws/devel/include"
)

# nodelets

# Dataset-specific preprocessing frontends. The radar point type, topic layout and
# extrinsics differ per dataset, so there is one nodelet per dataset -- but both
# are built and the launch file picks which one loads:
#
#   ntu_*.launch         radar_graph_slam/PreprocessingNodeletNTU
#   msc_*.launch         radar_graph_slam/PreprocessingNodeletMSC
#
# They used to be mutually exclusive behind a GORIO_DATASET cache variable, which
# is sticky: building MSC once left every later catkin_make producing the MSC
# nodelet, so NTU runs came up subscribing to the wrong message type and silently
# produced nothing. Switching datasets now needs no rebuild.

add_library(preprocessing_nodelet_ntu
  apps/preprocessing_nodelet_ntu.cpp
  src/radar_ego_velocity_estimator.cpp
  #src/lib/radar_body_velocity_estimator.cpp
  #src/lib/radar_body_velocity_estimator_ros.cpp
)
target_link_libraries(preprocessing_nodelet_ntu
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES}
)
add_dependencies(preprocessing_nodelet_ntu ${PROJECT_NAME}_gencpp)

add_library(preprocessing_nodelet_msc
  apps/preprocessing_nodelet_msc.cpp
  src/radar_ego_velocity_estimator.cpp
)
target_link_libraries(preprocessing_nodelet_msc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES}
)
add_dependencies(preprocessing_nodelet_msc ${PROJECT_NAME}_gencpp)


add_library(scan_matching_odometry_nodelet
  apps/scan_matching_odometry_nodelet.cpp
  src/radar_graph_slam/registrations.cpp
  # scan_matching_odometry_nodelet.cpp constructs KeyFrame directly, but
  # keyframe.cpp was only compiled into radar_graph_slam_nodelet. The shared
  # library still linked (undefined symbols are allowed) and then failed at
  # dlopen time with an undefined KeyFrame constructor, so the nodelet could
  # never actually be loaded.
  src/radar_graph_slam/keyframe.cpp
)
target_link_libraries(scan_matching_odometry_nodelet
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # KeyFrame holds a g2o::VertexSE3*, so this target needs g2o's typeinfo for
  # g2o::VertexSE3 and g2o::HyperGraph::Vertex. Without it the library loaded
  # only when radar_graph_slam_nodelet happened to be dlopen'd first in the same
  # nodelet manager, which is load-order dependent.
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_TYPES_SLAM3D}
)
add_dependencies(scan_matching_odometry_nodelet ${PROJECT_NAME}_gencpp)

add_library(radar_graph_slam_nodelet
  apps/radar_graph_slam_nodelet.cpp
  src/radar_graph_slam/graph_slam.cpp
  src/radar_graph_slam/loop_detector.cpp
  src/radar_graph_slam/Scancontext.cpp
  src/radar_graph_slam/keyframe.cpp
  src/radar_graph_slam/map_cloud_generator.cpp
  src/radar_graph_slam/registrations.cpp
  src/radar_graph_slam/information_matrix_calculator.cpp
  src/g2o/robust_kernel_io.cpp
)
target_link_libraries(radar_graph_slam_nodelet
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}   # be aware of that CSPARSE is released under LGPL
  ${G2O_SOLVER_CHOLMOD}   # be aware of that cholmod is released under GPL
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
  ${CERES_LIBRARIES}
)
add_dependencies(radar_graph_slam_nodelet ${PROJECT_NAME}_gencpp)



install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS
  preprocessing_nodelet_ntu
  preprocessing_nodelet_msc
  scan_matching_odometry_nodelet
  radar_graph_slam_nodelet
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
