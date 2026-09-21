// SPDX-License-Identifier: BSD-2-Clause

#ifndef KEYFRAME_HPP
#define KEYFRAME_HPP

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/optional.hpp>

#include <geometry_msgs/Transform.h>
#include <sensor_msgs/Imu.h>


namespace g2o {
class VertexSE3;
class HyperGraph;
class SparseOptimizer;
}  // namespace g2o

namespace radar_graph_slam {

/**
 * @brief KeyFrame (pose node)
 */
struct KeyFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PointT = pcl::PointXYZINormal;
  using Ptr = std::shared_ptr<KeyFrame>;

  KeyFrame(const size_t index, const ros::Time& stamp, const Eigen::Isometry3d& odom_scan2scan, double accum_distance, const pcl::PointCloud<PointT>::ConstPtr& cloud);
  KeyFrame(const std::string& directory, g2o::HyperGraph* graph);
  virtual ~KeyFrame();

  void save(const std::string& directory);
  bool load(const std::string& directory, g2o::HyperGraph* graph);

  long id() const;
  Eigen::Isometry3d estimate() const;

public:
  size_t index;
  ros::Time stamp;                                // timestamp
  Eigen::Isometry3d odom_scan2scan;               // odometry (estimated by scan_matching_odometry)
  Eigen::Isometry3d odom_scan2map;
  double accum_distance;                          // accumulated distance from the first node (by scan_matching_odometry)
  pcl::PointCloud<PointT>::ConstPtr cloud;        // point cloud

  boost::optional<Eigen::Vector3d> acceleration;    //
  boost::optional<Eigen::Quaterniond> orientation;  //

  geometry_msgs::Transform trans_integrated; // relative transform obtained by imu preintegration

  // Covariance of trans_integrated, as the GP propagated it, in the library's
  // own [rotation, translation] ordering. g2o's EdgeSE3 uses the opposite
  // ordering, so it must be permuted before use as an information matrix --
  // see test/g2o_information_ordering_test.cpp.
  Eigen::Matrix<double, 6, 6> preint_cov = Eigen::Matrix<double, 6, 6>::Identity();
  bool preint_cov_valid = false;

  // Jacobian of the preintegrated rotation with respect to the gyro bias, and
  // the bias the measurement was integrated with. Together these let the bias
  // be a graph vertex rather than a value fed back from outside the graph.

  // Covariance of odom_scan2scan relative to the previous keyframe, accumulated
  // from the per-frame registration covariances the front end publishes on
  // /odom. Already in ROS/g2o [translation, rotation] order.
  Eigen::Matrix<double, 6, 6> odom_cov = Eigen::Matrix<double, 6, 6>::Zero();
  bool odom_cov_valid = false;
  int odom_cov_frames = 0;
  // Path length the ego-velocity accounts for since the previous keyframe.
  // A straight-line displacement cannot exceed it.
  double egovel_path = 0.0;

  // Gravity-corrected attitude from the IMU's AHRS at this keyframe's time,
  // expressed relative to the first keyframe so it lives in the map frame.
  boost::optional<Eigen::Quaterniond> ahrs_attitude;

  bool trust_omega = true; // trust the omega value of the IMU preintegration

  boost::optional<sensor_msgs::Imu> imu; // the IMU message close to keyframe_

  g2o::VertexSE3* node;  // node instance
};

/**
 * @brief KeyFramesnapshot for map cloud generation
 */
struct KeyFrameSnapshot {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PointT = KeyFrame::PointT;
  using Ptr = std::shared_ptr<KeyFrameSnapshot>;

  KeyFrameSnapshot(const KeyFrame::Ptr& key);
  KeyFrameSnapshot(const Eigen::Isometry3d& pose, const pcl::PointCloud<PointT>::ConstPtr& cloud);

  ~KeyFrameSnapshot();

public:
  Eigen::Isometry3d pose;                   // pose estimated by graph optimization
  pcl::PointCloud<PointT>::ConstPtr cloud;  // point cloud
};

}  // namespace radar_graph_slam

#endif  // KEYFRAME_HPP
