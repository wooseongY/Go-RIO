// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <chrono>
#include <mutex>
#include <atomic>
#include <memory>
#include <set>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>

#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/String.h>
#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/octree/octree_search.h>

#include <Eigen/Dense>

#include <radar_graph_slam/ros_utils.hpp>
#include <radar_graph_slam/registrations.hpp>
#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/fast_apdgicp.hpp>
#include <radar_graph_slam/keyframe.hpp>
#include <radar_graph_slam/keyframe_updater.hpp>
#include <radar_graph_slam/graph_slam.hpp>
#include <radar_graph_slam/information_matrix_calculator.hpp>

#include "utility_radar.h"
#include "dbscan/DBSCAN_kdtree.h"

using namespace std;

namespace radar_graph_slam {

class ScanMatchingOdometryNodelet : public nodelet::Nodelet, public ParamServer {
public:
  typedef pcl::PointXYZINormal PointT;
  typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::TwistWithCovarianceStamped, sensor_msgs::PointCloud2> ApproxSyncPolicy;
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::TransformStamped> ApproxSyncPolicy2;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdometryNodelet() {}
  virtual ~ScanMatchingOdometryNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params(); // this

    //******** Subscribers **********
    ego_vel_sub.reset(new message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>(mt_nh, "/eagle_data/twist", 1000000));
    points_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 1000000));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *ego_vel_sub, *points_sub));
    sync->registerCallback(boost::bind(&ScanMatchingOdometryNodelet::pointcloud_callback, this, _1, _2));
    // "/imu" was hardcoded here, but the topic is a parameter and NTU publishes
    // /vectornav/imu, so this queue never received anything on that dataset --
    // get_closest_imu() always failed and every IMU-dependent path was dead.
    imu_sub = nh.subscribe(imuTopic, 256, &ScanMatchingOdometryNodelet::imu_callback, this);
    command_sub = nh.subscribe("/command", 10, &ScanMatchingOdometryNodelet::command_callback, this);

    //******** Publishers **********
    // Odometry of Radar scan-matching_
    odom_pub = nh.advertise<nav_msgs::Odometry>(odomTopic, 1000000);
    // Transformation of Radar scan-matching_
    trans_pub = nh.advertise<geometry_msgs::TransformStamped>("/scan_matching_odometry/transform", 1000000);
    aligned_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 1000000);
    submap_pub = nh.advertise<sensor_msgs::PointCloud2>("/radar_graph_slam/submap", 1000000);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    auto& pnh = private_nh;
    points_topic = pnh.param<std::string>("points_topic", "/radar_enhanced_pcl");
    use_ego_vel = pnh.param<bool>("use_ego_vel", false);

    // The minimum tranlational distance and rotation angle between keyframes_.
    // If this value is zero, frames are always compared with the previous frame
    keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);

    // Registration validation by thresholding
    enable_transform_thresholding = pnh.param<bool>("enable_transform_thresholding", false);
    max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);
    max_diff_trans = pnh.param<double>("max_diff_trans", 1.0);
    max_diff_angle = pnh.param<double>("max_diff_angle", 1.0);
    max_egovel_cum = pnh.param<double>("max_egovel_cum", 1.0);

    map_cloud_resolution = pnh.param<double>("map_cloud_resolution", 0.05);
    keyframe_updater.reset(new KeyframeUpdater(pnh));


    // graph_slam.reset(new GraphSLAM(pnh.param<std::string>("g2o_solver_type", "lm_var")));

    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else if(downsample_method == "DECIMATE") {
      // Keep one point in every N by index, which thins the cloud without
      // changing its spatial distribution -- unlike a voxel grid, which
      // equalises density and so removes proportionally more from the dense
      // near field where the ground returns are.
      //
      // Worth its own test because the GICP Hessian's absolute scale is
      // proportional to the number of correspondences, so decimating the cloud
      // de-weights the odometry edge against the preintegration by that factor.
      // It is the one lever that rebalances the fusion without touching either
      // factor's covariance model, and the cluster weight's 1/N normalisation
      // scales with it too.
      decimate_stride_ = std::max(1, pnh.param<int>("decimate_stride", 4));
      std::cout << "downsample: DECIMATE keep 1 of " << decimate_stride_ << std::endl;
      downsample_filter.reset();
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }
    registration_s2s = select_registration_method(pnh);
    // Typed view of the same object, for getFinalHessian(). select_registration_method
    // returns the PCL base pointer, and NDT/ICP are not LsqRegistration, so this
    // is null for those and the pose covariance is simply not published.
    lsq_s2s = dynamic_cast<fast_gicp::LsqRegistration<PointT, PointT>*>(registration_s2s.get());
    // Same object again, as the APD type, so the cluster-label diagnostic can
    // be read back. Null for any other registration method.
    apd_s2s = dynamic_cast<fast_gicp::FastAPDGICP<PointT, PointT>*>(registration_s2s.get());
    if(lsq_s2s == nullptr) {
      ROS_WARN("registration method exposes no Hessian; /odom will carry no pose covariance");
    }
    pose_cov_min_hessian_eigenvalue = pnh.param<double>("pose_cov_min_hessian_eigenvalue", 1e-12);
    use_cluster_dof = pnh.param<bool>("pose_cov_use_cluster_dof", true);
  }

  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    
    Eigen::Quaterniond imu_quat_from(imu_msg->orientation.w, imu_msg->orientation.x, imu_msg->orientation.y, imu_msg->orientation.z);
    Eigen::Quaterniond imu_quat_deskew = imu_quat_from * extQRPY;
    imu_quat_deskew.normalize();

    double roll, pitch, yaw;
    // tf::quaternionMsgToTF(imu_odom_msg->orientation, orientation);
    tf::Quaternion orientation = tf::Quaternion(imu_quat_deskew.x(),imu_quat_deskew.y(),imu_quat_deskew.z(),imu_quat_deskew.w());
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    // The angular velocity was copied straight through while the orientation was
    // transformed, so the gyro stayed in the IMU frame -- the same defect the
    // back end had. extRot is roughly 180 deg about X here, which flips the sign
    // of w_y and w_z, and w_z is yaw for a ground vehicle.
    const sensor_msgs::Imu imu_conv = imuConverter(*imu_msg);

    sensor_msgs::ImuPtr imu(new sensor_msgs::Imu);
    imu->header = imu_msg->header;
    imu->angular_velocity = imu_conv.angular_velocity; imu->linear_acceleration = imu_conv.linear_acceleration;
    imu->angular_velocity_covariance = imu_msg->angular_velocity_covariance;
    imu->linear_acceleration_covariance, imu_msg->linear_acceleration_covariance;
    imu->orientation_covariance = imu_msg->orientation_covariance;
    imu->orientation.w=imu_quat_deskew.w(); imu->orientation.x = imu_quat_deskew.x(); imu->orientation.y = imu_quat_deskew.y(); imu->orientation.z = imu_quat_deskew.z();
    {
      std::lock_guard<std::mutex> lock(imu_queue_mutex);
      imu_queue.push_back(imu);
    }

    static bool logged_initial_rpy = false;
    if(!logged_initial_rpy) {
      logged_initial_rpy = true;
      ROS_INFO_STREAM("Initial IMU euler angles (RPY): "
            << RAD2DEG(roll) << ", " << RAD2DEG(pitch) << ", " << RAD2DEG(yaw));
    }
    
  }

  bool flush_imu_queue() {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(keyframes.empty() || imu_queue.empty()) {
      return false;
    }
    bool updated = false;
    auto imu_cursor = imu_queue.begin();

    for(size_t i=0; i < keyframes.size(); i++) {
      auto keyframe = keyframes.at(i);
      if(keyframe->stamp < (*imu_cursor)->header.stamp) {
        continue;
      }
      if(keyframe->stamp > imu_queue.back()->header.stamp) {
        break;
      }
      // find the imu data which is closest to the keyframe_
      auto closest_imu = imu_cursor;
      for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {
        auto dt = ((*closest_imu)->header.stamp - keyframe->stamp).toSec();
        auto dt2 = ((*imu)->header.stamp - keyframe->stamp).toSec();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }
        closest_imu = imu;
      }
      // if the time residual between the imu and keyframe_ is too large, skip it
      imu_cursor = closest_imu;
      if(0.2 < std::abs(((*closest_imu)->header.stamp - keyframe->stamp).toSec())) {
        continue;
      }
      sensor_msgs::Imu imu_;
      imu_.header = (*closest_imu)->header; imu_.orientation = (*closest_imu)->orientation;
      imu_.angular_velocity = (*closest_imu)->angular_velocity; imu_.linear_acceleration = (*closest_imu)->linear_acceleration;
      imu_.angular_velocity_covariance = (*closest_imu)->angular_velocity_covariance;
      imu_.linear_acceleration_covariance = (*closest_imu)->linear_acceleration_covariance;
      imu_.orientation_covariance = (*closest_imu)->orientation_covariance;
      keyframe->imu = imu_;
      updated = true;
    }
    auto remove_loc = std::upper_bound(imu_queue.begin(), imu_queue.end(), keyframes.back()->stamp, [=](const ros::Time& stamp, const sensor_msgs::ImuConstPtr& imupoint) { return stamp < imupoint->header.stamp; });
    imu_queue.erase(imu_queue.begin(), remove_loc);
    return updated;
  }

  // Attitude from the AHRS at an arbitrary time, interpolated between samples.
  // The queue holds orientations already rotated into base_link by extQRPY.
  bool ahrs_at(double t, Eigen::Quaterniond& out) {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(imu_queue.size() < 2) return false;
    if(t < imu_queue.front()->header.stamp.toSec() ||
       t > imu_queue.back()->header.stamp.toSec()) return false;
    auto hi = std::lower_bound(imu_queue.begin(), imu_queue.end(), t,
        [](const sensor_msgs::ImuConstPtr& m, double v) { return m->header.stamp.toSec() < v; });
    if(hi == imu_queue.begin()) { hi = std::next(hi); }
    auto lo = std::prev(hi);
    const double t0 = (*lo)->header.stamp.toSec(), t1 = (*hi)->header.stamp.toSec();
    const double a = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
    const Eigen::Quaterniond q0((*lo)->orientation.w, (*lo)->orientation.x,
                                (*lo)->orientation.y, (*lo)->orientation.z);
    const Eigen::Quaterniond q1((*hi)->orientation.w, (*hi)->orientation.x,
                                (*hi)->orientation.y, (*hi)->orientation.z);
    out = q0.slerp(a, q1).normalized();
    return true;
  }

  bool gyro_rotation_between(double t0, double t1, Eigen::Matrix3d& dR) {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(imu_queue.size() < 2 || t1 <= t0) return false;
    if(t0 < imu_queue.front()->header.stamp.toSec() ||
       t1 > imu_queue.back()->header.stamp.toSec()) return false;

    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    double covered = 0.0;
    for(size_t i = 1; i < imu_queue.size(); i++) {
      const double a = imu_queue[i - 1]->header.stamp.toSec();
      const double b = imu_queue[i]->header.stamp.toSec();
      const double lo = std::max(a, t0), hi = std::min(b, t1);
      if(hi <= lo) continue;
      const auto& w = imu_queue[i - 1]->angular_velocity;
      const Eigen::Vector3d omega(w.x, w.y, w.z);
      const double dt = hi - lo;
      const double angle = omega.norm() * dt;
      if(angle > 1e-12) {
        q *= Eigen::Quaterniond(Eigen::AngleAxisd(angle, omega.normalized()));
      }
      covered += dt;
    }
    // A gap in the stream would silently shrink the rotation, which is the very
    // error this prior exists to remove, so require the interval to be covered.
    if(covered < 0.9 * (t1 - t0)) return false;
    dR = q.normalized().toRotationMatrix();
    return true;
  }

  std::pair<bool, sensor_msgs::Imu> get_closest_imu(ros::Time frame_stamp) {
    sensor_msgs::Imu imu_;
    std::pair<bool, sensor_msgs::Imu> false_result {false, imu_};
    if(keyframes.empty() || imu_queue.empty())
      return false_result;
    bool updated = false;
    auto imu_cursor = imu_queue.begin();
    
    // find the imu data which is closest to the keyframe_
    auto closest_imu = imu_cursor;
    for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {
      auto dt = ((*closest_imu)->header.stamp - frame_stamp).toSec();
      auto dt2 = ((*imu)->header.stamp - frame_stamp).toSec();
      if(std::abs(dt) < std::abs(dt2)) {
        break;
      }
      closest_imu = imu;
    }
    // if the time residual between the imu and keyframe_ is too large, skip it
    imu_cursor = closest_imu;
    if(0.2 < std::abs(((*closest_imu)->header.stamp - frame_stamp).toSec()))
      return false_result;

    imu_.header = (*closest_imu)->header; imu_.orientation = (*closest_imu)->orientation;
    imu_.angular_velocity = (*closest_imu)->angular_velocity; imu_.linear_acceleration = (*closest_imu)->linear_acceleration;
    imu_.angular_velocity_covariance = (*closest_imu)->angular_velocity_covariance; 
    imu_.linear_acceleration_covariance = (*closest_imu)->linear_acceleration_covariance;
    imu_.orientation_covariance = (*closest_imu)->orientation_covariance;

    updated = true;
    // cout << (*closest_imu)->orientation <<endl;
    std::pair<bool, sensor_msgs::Imu> result {updated, imu_};
    return result;
  }


  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void pointcloud_callback(const geometry_msgs::TwistWithCovarianceStampedConstPtr& twistMsg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if(!ros::ok()) {
      return;
    }
    double this_cloud_time = cloud_msg->header.stamp.toSec();
    static double last_cloud_time = this_cloud_time;

    double dt = this_cloud_time - last_cloud_time;
    double egovel_cum_x = twistMsg->twist.twist.linear.x * dt;
    double egovel_cum_y = twistMsg->twist.twist.linear.y * dt;
    double egovel_cum_z = twistMsg->twist.twist.linear.z * dt;
    egovel_ = Eigen::Vector3d(twistMsg->twist.twist.linear.x, twistMsg->twist.twist.linear.y, twistMsg->twist.twist.linear.z);
    // If too large, set 0
    if (pow(egovel_cum_x,2)+pow(egovel_cum_y,2)+pow(egovel_cum_z,2) > pow(max_egovel_cum, 2));
    else egovel_cum.block<3, 1>(0, 3) = Eigen::Vector3d(egovel_cum_x, egovel_cum_y, egovel_cum_z);

    last_cloud_time = this_cloud_time;

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Matching
    Eigen::Matrix4d pose = matching(cloud_msg->header.stamp, cloud);
    // pose : accumulated location from the origin
    geometry_msgs::TwistWithCovariance twist = twistMsg->twist;
    // publish map to odom frame
    publish_odometry(cloud_msg->header.stamp, mapFrame, odometryFrame, pose, twist); // scan matching odometry

  }

  /**
   * @brief number of statistically independent observations in a scan
   *
   * The residual variance estimate sigma^2 = cost/(N-6) assumes N independent
   * residuals. GICP residuals are not independent: every point on one rigid
   * structure moves together, so a cluster of points contributes roughly one
   * independent constraint, not one per point. Using the point count therefore
   * understates sigma^2 and makes the covariance far too tight -- measured, it
   * gave a rotation sigma of 0.03 degrees for a 0.5 m radar keyframe step,
   * which would imply sub-metre drift over the whole sequence where the actual
   * error is metres.
   *
   * The preprocessing nodelet already assigns a cluster rank to each point in
   * normal_x, so the count is recoverable here without new plumbing. On cp the
   * ratio is about 200:1 -- 4661 points against 23 clusters at the median.
   */
  size_t effective_observations(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!use_cluster_dof || cloud->empty()) {
      return cloud->size();
    }
    std::set<int> ids;
    for(const auto& pt : cloud->points) {
      ids.insert(static_cast<int>(pt.normal_x));
    }
    // Fall back to the point count if the field is not carrying cluster ranks.
    return ids.size() > 1 ? ids.size() : cloud->size();
  }

  /**
   * @brief pose covariance of the last scan-to-scan registration
   *
   * The registration's own Gauss-Newton Hessian already describes which degrees
   * of freedom the geometry constrained and which it did not -- a radar scan in
   * a corridor is genuinely weak in yaw, and H says so, per keyframe. The
   * backend currently replaces that with a fitness-score heuristic that
   * saturates on radar data, handing every edge the same worst-case weight, plus
   * a hardcoded override of the yaw term.
   *
   * H alone is not a covariance: fast_gicp regularises the per-point GICP
   * covariances into shape descriptors, so H's absolute scale is arbitrary. The
   * standard nonlinear-least-squares estimate rescales it by the measured
   * residual level, sigma^2 = cost / (N - 6), giving Sigma = sigma^2 * H^-1.
   * That is one scalar with a statistical meaning, replacing six tuned
   * constants.
   *
   * Published in ROS pose-covariance order [x, y, z, rot_x, rot_y, rot_z].
   * fast_gicp orders its state [rotation, translation] -- d.head<3>() is the
   * rotation increment in step_gn/step_lm -- so the two 3-blocks are swapped
   * here. Same trap as ugpm's PreintMeas::cov; see
   * test/g2o_information_ordering_test.cpp.
   */
  bool compute_pose_covariance(size_t num_source_points, Eigen::Matrix<double, 6, 6>& cov_out) {
    if(lsq_s2s == nullptr) {
      return false;
    }

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    const double cost = lsq_s2s->evaluateCost(
        registration_s2s->getFinalTransformation(), &H, &b);
    if(!H.allFinite() || !std::isfinite(cost) || cost <= 0.0) {
      return false;
    }

    const double dof = static_cast<double>(num_source_points) - 6.0;
    if(dof <= 0.0) {
      return false;
    }
    const double sigma_sq = cost / dof;
    if(!std::isfinite(sigma_sq) || sigma_sq <= 0.0) {
      return false;
    }

    // Floor the eigenvalues before inverting: a near-unobservable direction
    // otherwise inverts to an enormous covariance, or to a negative one if the
    // Hessian came back slightly indefinite.
    Eigen::Matrix<double, 6, 6> sym = 0.5 * (H + H.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(sym);
    Eigen::Matrix<double, 6, 1> eigs = solver.eigenvalues();
    for(int i = 0; i < 6; ++i) {
      if(!(eigs(i) > pose_cov_min_hessian_eigenvalue)) {
        eigs(i) = pose_cov_min_hessian_eigenvalue;
        pose_cov_clamped++;
      }
    }
    const Eigen::Matrix<double, 6, 6> H_reg =
        solver.eigenvectors() * eigs.asDiagonal() * solver.eigenvectors().transpose();
    const Eigen::Matrix<double, 6, 6> cov_rt = sigma_sq * H_reg.inverse();

    // [rot, trans] -> [trans, rot]
    cov_out.setZero();
    cov_out.block<3, 3>(0, 0) = cov_rt.block<3, 3>(3, 3);
    cov_out.block<3, 3>(3, 3) = cov_rt.block<3, 3>(0, 0);
    cov_out.block<3, 3>(0, 3) = cov_rt.block<3, 3>(3, 0);
    cov_out.block<3, 3>(3, 0) = cov_rt.block<3, 3>(0, 3);
    return cov_out.allFinite();
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  /**
   * @brief thin the registration input, and only when index decimation is asked for
   *
   * The shipped configuration hands the registration the raw cloud: both call
   * sites in matching() were commented out, so downsample() never ran at all.
   * Routing the registration through downsample() unconditionally would put a
   * PassThrough copy in the NONE path and change nothing else, but this keeps
   * NONE on the identical pointer so the baseline is untouched.
   */
  pcl::PointCloud<PointT>::ConstPtr maybe_decimate(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    return (decimate_stride_ > 1) ? downsample(cloud) : cloud;
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(decimate_stride_ > 1) {
      pcl::PointCloud<PointT>::Ptr thinned(new pcl::PointCloud<PointT>());
      thinned->header = cloud->header;
      thinned->reserve(cloud->size() / decimate_stride_ + 1);
      for(size_t i = 0; i < cloud->size(); i += decimate_stride_) {
        thinned->points.push_back(cloud->points[i]);
      }
      thinned->width = thinned->size();
      thinned->height = 1;
      thinned->is_dense = cloud->is_dense;
      return thinned;
    }
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe_ cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe_ cloud
   */
  Eigen::Matrix4d matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(!keyframe_cloud_s2s) {
      prev_time = ros::Time();
      prev_trans_s2s.setIdentity();
      keyframe_pose_s2s.setIdentity();
      keyframe_stamp = stamp;
      keyframe_cloud_s2s = maybe_decimate(cloud);
      registration_s2s->setInputTarget(keyframe_cloud_s2s); // Scan-to-scan
      return Eigen::Matrix4d::Identity();
    }
    auto filtered = maybe_decimate(cloud);
    if(!reported_decimation_) {
      reported_decimation_ = true;
      std::cout << "registration input: " << cloud->size() << " -> " << filtered->size()
                << " points (stride " << decimate_stride_ << ")" << std::endl;
    }
    // Set Source Cloud
    registration_s2s->setInputSource(filtered);
    // registration_s2s->setSourceVelocity(egovel_);
    Eigen::Vector3d ext_trans(0.2, 0.17, 0.08);
    // registration_s2s->setSourceExtrinsic(ext_trans);

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    Eigen::Matrix4d odom_s2s_now;

    Eigen::Matrix4d delta_gyro = Eigen::Matrix4d::Identity();
    bool have_gyro_prior = false;
    if (last_frame_stamp > 0) {
      Eigen::Matrix3d dR;
      if (gyro_rotation_between(last_frame_stamp, stamp.toSec(), dR)) {
        delta_gyro.block<3, 3>(0, 0) = dR;
        have_gyro_prior = true;
        gyro_prior_hits++;
      } else {
        gyro_prior_misses++;
      }
    }
    const double prev_frame_stamp = last_frame_stamp;
    last_frame_stamp = stamp.toSec();

    Eigen::Matrix4d guess;
    if (use_ego_vel)
      guess = prev_trans_s2s * egovel_cum;
    else
      guess = prev_trans_s2s;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    registration_s2s->align(*aligned, guess.cast<float>());
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    double time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
    s2s_matching_time.push_back(time_used);


    // If not converged, use last transformation
    if(!registration_s2s->hasConverged()) {
      NODELET_INFO_STREAM("scan matching_ has not converged!!");
      NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
      return keyframe_pose_s2s * prev_trans_s2s;
    }
    Eigen::Matrix4d trans_s2s = registration_s2s->getFinalTransformation().cast<double>();
    odom_s2s_now = keyframe_pose_s2s * trans_s2s;

    last_pose_cov_valid = compute_pose_covariance(effective_observations(filtered), last_pose_cov);

    // Add abnormal judgment, that is, if the difference between the two frames matching point cloud 
    // transition matrix is too large, it will be discarded
    bool thresholded = false;
    if(enable_transform_thresholding) {
      Eigen::Matrix4d radar_delta = prev_trans_s2s.inverse() * trans_s2s;
      double dx_rd = radar_delta.block<3, 1>(0, 3).norm();
      // double da_rd = std::acos(Eigen::Quaterniond(radar_delta.block<3, 3>(0, 0)).w())*180/M_PI;
      Eigen::AngleAxisd rotation_vector;
      rotation_vector.fromRotationMatrix(radar_delta.block<3, 3>(0, 0));
      double da_rd = rotation_vector.angle();
      bool too_large_trans = dx_rd > max_acceptable_trans || da_rd > max_acceptable_angle;

      if (too_large_trans) {
          Eigen::Matrix4d mat_est(Eigen::Matrix4d::Identity());
          mat_est.block<3, 3>(0, 0) = delta_gyro.block<3, 3>(0, 0);
          mat_est.block<3, 1>(0, 3) = egovel_cum.block<3, 1>(0, 3).cast<double>();
          cout << "Too large transform!!  " << dx_rd << "[m] " << da_rd << "[degree]"
               << " falling back to ego-velocity for this frame (" << stamp << ")" << endl;
          transform_rejections_++;
          prev_trans_s2s = prev_trans_s2s * mat_est;
          thresholded = true;
          odom_s2s_now = keyframe_pose_s2s * prev_trans_s2s;
      }
      last_radar_delta = radar_delta;
    }
    prev_time = stamp;
    if (!thresholded) {
      prev_trans_s2s = trans_s2s;
    }
    
    //********** Decided whether to accept the frame as a key frame or not **********
    if(keyframe_updater->decide(Eigen::Isometry3d(odom_s2s_now), stamp)) {
      // std::cout<<"odom_s2s_now: "<<odom_s2s_now<<std::endl;

      keyframe_cloud_s2s = filtered;
      registration_s2s->setInputTarget(keyframe_cloud_s2s);
      keyframe_pose_s2s = odom_s2s_now;
      keyframe_stamp = stamp;
      prev_time = stamp;
      prev_trans_s2s.setIdentity();

      double accum_d = keyframe_updater->get_accum_distance();
      KeyFrame::Ptr keyframe(new KeyFrame(keyframe_index, stamp, Eigen::Isometry3d(odom_s2s_now.cast<double>()), accum_d, cloud));
      keyframe_index ++;
      keyframes.push_back(keyframe);

      // record keyframe's imu
      flush_imu_queue();

    }
    
    if (aligned_points_pub.getNumSubscribers() > 0)
    {
      pcl::transformPointCloud (*cloud, *aligned, odom_s2s_now);
      aligned->header.frame_id = odometryFrame;
      aligned_points_pub.publish(*aligned);
    }

    return odom_s2s_now;
  }


  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& father_frame_id, const std::string& child_frame_id, const Eigen::Matrix4d& pose_in, const geometry_msgs::TwistWithCovariance twist_in) {
    // publish transform stamped for IMU integration
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose_in, father_frame_id, child_frame_id); //"map" 
    trans_pub.publish(odom_trans);

    // broadcast the transform over TF
    map2odom_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = father_frame_id;   // frame: /odom
    odom.child_frame_id = child_frame_id;

    odom.pose.pose.position.x = pose_in(0, 3);
    odom.pose.pose.position.y = pose_in(1, 3);
    odom.pose.pose.position.z = pose_in(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;
    odom.twist = twist_in;

    // Registration pose covariance, so the backend can weight this edge by what
    // the geometry actually constrained instead of a saturating fitness-score
    // heuristic. Left at zero when unavailable, which the backend reads as
    // "fall back to the legacy weighting".
    if(last_pose_cov_valid) {
      for(int r = 0; r < 6; ++r) {
        for(int c = 0; c < 6; ++c) {
          odom.pose.covariance[6 * r + c] = last_pose_cov(r, c);
        }
      }
    }

    
    odom_pub.publish(odom);
  }


  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "time") {
      std::sort(s2s_matching_time.begin(), s2s_matching_time.end());
      double median = s2s_matching_time.at(size_t(s2s_matching_time.size() / 2));
      cout << "Scan Matching time cost (median): " << median << endl;
      if(apd_s2s != nullptr) {
        long agree = 0, total = 0;
        std::map<int, long> src, tgt;
        apd_s2s->clusterLabelStats(agree, total, src, tgt);
        if(total > 0) {
          // Chance rate: if the two labellings were independent, agreement would
          // be sum_l p_src(l) * p_tgt(l). The label is a rank by centroid
          // distance, which only corresponds across frames while the cluster
          // count and ordering hold, so this is the test of whether it does.
          double chance = 0.0;
          for(const auto& kv : src) {
            const auto it = tgt.find(kv.first);
            if(it != tgt.end()) {
              chance += (double(kv.second) / total) * (double(it->second) / total);
            }
          }
          cout << "Cluster label agreement: " << double(agree) / total * 100.0 << "% of "
               << total << " correspondences, chance " << chance * 100.0 << "%, lift "
               << (chance > 0.0 ? (double(agree) / total) / chance : 0.0) << "x, labels "
               << src.size() << " source / " << tgt.size() << " target" << endl;
        }
      }
    }
  }

private:
  // ROS topics
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  // ros::Subscriber points_sub;
  ros::Subscriber imu_sub;

  std::mutex imu_queue_mutex;
  std::deque<sensor_msgs::ImuConstPtr> imu_queue;
  sensor_msgs::Imu last_frame_imu;


  std::unique_ptr<message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>> ego_vel_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> points_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  // Submap
  ros::Publisher submap_pub;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::vector<KeyFrame::Ptr> keyframes;
  size_t keyframe_index = 0;
  double map_cloud_resolution;

  // std::unique_ptr<GraphSLAM> graph_slam;
  // std::unique_ptr<InformationMatrixCalculator> inf_calclator;
  
  ros::Publisher odom_pub;
  ros::Publisher trans_pub;
  // ros::Publisher keyframe_trans_pub;
  ros::Publisher aligned_points_pub;
  tf::TransformListener tf_listener;
  tf::TransformBroadcaster map2odom_broadcaster; // map => odom_frame

  std::string points_topic;


  // keyframe_ parameters
  double keyframe_delta_trans;  // minimum distance between keyframes_
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool enable_transform_thresholding;  //
  double max_acceptable_trans;  //
  double max_acceptable_angle;
  double max_diff_trans;
  double max_diff_angle;
  double max_egovel_cum;
  double last_frame_stamp = 0.0;
  long gyro_prior_hits = 0, gyro_prior_misses = 0;
  long transform_rejections_ = 0;
  Eigen::Matrix4d last_radar_delta = Eigen::Matrix4d::Identity();

  // odometry calculation

  Eigen::Matrix4d egovel_cum = Eigen::Matrix4d::Identity();
  Eigen::Vector3d egovel_;
  bool use_ego_vel;

  ros::Time prev_time;
  Eigen::Matrix4d prev_trans_s2s;                  // previous relative transform from keyframe_
  Eigen::Matrix4d keyframe_pose_s2s;               // keyframe_ pose
  ros::Time keyframe_stamp;                    // keyframe_ time
  pcl::PointCloud<PointT>::ConstPtr keyframe_cloud_s2s;  // keyframe_ point cloud

  // Registration
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration_s2s;    // Scan-to-Scan Registration
  fast_gicp::LsqRegistration<PointT, PointT>* lsq_s2s = nullptr;
  fast_gicp::FastAPDGICP<PointT, PointT>* apd_s2s = nullptr;  // same object, for getFinalHessian()
  Eigen::Matrix<double, 6, 6> last_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
  bool last_pose_cov_valid = false;
  double pose_cov_min_hessian_eigenvalue = 1e-6;
  long pose_cov_clamped = 0;
  bool use_cluster_dof = true;
  int decimate_stride_ = 1;
  mutable bool reported_decimation_ = false;

  // Time evaluation
  std::vector<double> s2s_matching_time;
  ros::Subscriber command_sub;
};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::ScanMatchingOdometryNodelet, nodelet::Nodelet)
