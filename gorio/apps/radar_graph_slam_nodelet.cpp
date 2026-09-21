// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <fstream>
#include <string>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/octree/octree_search.h>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <gorio/SaveMap.h>
#include <gorio/DumpGraph.h>
#include <radar_graph_slam/ros_utils.hpp>
#include <radar_graph_slam/ros_time_hash.hpp>
#include <radar_graph_slam/graph_slam.hpp>
#include <radar_graph_slam/keyframe.hpp>
#include <radar_graph_slam/keyframe_updater.hpp>
#include <radar_graph_slam/information_matrix_calculator.hpp>
#include <radar_graph_slam/loop_detector.hpp>
#include "scan_context/Scancontext.h"
#include <radar_graph_slam/map_cloud_generator.hpp>
#include "radar_graph_slam/polynomial_interpolation.hpp"
#include <radar_graph_slam/registrations.hpp>


#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/edge_se3_plane_relative.hpp>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

#include "VelInt/preint.h"

#include <unsupported/Eigen/Splines>


#include "utility_radar.h"

using namespace std;

namespace radar_graph_slam {

class RadarGraphSlamNodelet : public nodelet::Nodelet, public ParamServer {
public:
  typedef pcl::PointXYZINormal PointT;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> ApproxSyncPolicy;

  RadarGraphSlamNodelet() {}
  virtual ~RadarGraphSlamNodelet() {}

  virtual void onInit() {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    // init parameters
    map_cloud_resolution = private_nh.param<double>("map_cloud_resolution", 0.05);
    trans_odom2map.setIdentity();
    trans_aftmapped.setIdentity();
    trans_aftmapped_incremental.setIdentity();
    initial_pose.setIdentity();

    max_keyframes_per_update = private_nh.param<int>("max_keyframes_per_update", 10);
    fix_first_keyframe_ = private_nh.param<bool>("fix_first_keyframe", true);

    anchor_node = nullptr;
    anchor_edge = nullptr;
    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    keyframe_updater.reset(new KeyframeUpdater(private_nh));
    // Loop closure is off in every shipped configuration and in every reported
    // result; the implementation is kept so it can be switched on. Read once
    // rather than per call: the detector, the Scan Context descriptors it needs
    // and the loop factors are all built only when this is true, so with the
    // flag off none of that work is done at all.
    enable_loop_closure_ = private_nh.param<bool>("enable_loop_closure", false);
    if(enable_loop_closure_) {
      loop_detector.reset(new LoopDetector(private_nh));
      show_sphere_ = private_nh.param<bool>("show_sphere", false);
    }
    map_cloud_generator.reset(new MapCloudGenerator());
    inf_calclator.reset(new InformationMatrixCalculator(private_nh));


    // Preintegration Parameters
    enable_preintegration = private_nh.param<bool>("enable_preintegration", false);
    preinteg_window_margin = private_nh.param<double>("preinteg_window_margin", 0.8);
    preinteg_min_gyr_samples = private_nh.param<int>("preinteg_min_gyr_samples", 10);
    preinteg_min_vel_samples = private_nh.param<int>("preinteg_min_vel_samples", 5);
    // Largest permissible hole in the ego-velocity stream inside the window.
    // The nominal period is ~1/12 s, so this allows a few missed frames.
    preinteg_max_vel_gap = private_nh.param<double>("preinteg_max_vel_gap", 0.4);
    // Fraction of the keyframe interval the ego-velocity samples must span.
    preinteg_min_vel_coverage = private_nh.param<double>("preinteg_min_vel_coverage", 0.7);
    // Keyframe gaps wider than this are not preintegrated at all.
    preinteg_max_interval = private_nh.param<double>("preinteg_max_interval", 2.0);
    preinteg_vel_bias_std = private_nh.param<double>("preinteg_vel_bias_std", 0.05);
    preinteg_gyr_bias_std = private_nh.param<double>("preinteg_gyr_bias_std", 1e-3);
    imu_time_offset_ = private_nh.param<double>("imu_time_offset", 0.114);
    preinteg_edge_dofs_ = private_nh.param<std::string>("preinteg_edge_dofs", "full");
    preinteg_rot_weight_from_odom_ = private_nh.param<bool>("preinteg_rot_weight_from_odom", true);
    preinteg_weight_cap_yaw_only_ = private_nh.param<bool>("preinteg_weight_cap_yaw_only", true);
    {
      // Hz. Thin the gyro to one sample per decorrelation time so the GP's
      // white-noise assumption holds. 0 keeps every sample.
      const double rot_all = private_nh.param<double>("preinteg_rot_cov_inflation", 1.0);
      preinteg_cov_inflation_
          << std::max(rot_all, private_nh.param<double>("preinteg_cov_inflation_roll", 1.0)),
             std::max(rot_all, private_nh.param<double>("preinteg_cov_inflation_pitch", 1.0)),
             std::max(rot_all, private_nh.param<double>("preinteg_cov_inflation_yaw", 1.0)),
             private_nh.param<double>("preinteg_cov_inflation_fwd", 1.0),
             private_nh.param<double>("preinteg_cov_inflation_lat", 1.0),
             private_nh.param<double>("preinteg_cov_inflation_vert", 1.0);
      odom_cov_inflation_
          << private_nh.param<double>("odom_trans_cov_inflation_fwd", 1.0),
             private_nh.param<double>("odom_trans_cov_inflation_lat", 1.0),
             private_nh.param<double>("odom_trans_cov_inflation_vert", 1.0),
             private_nh.param<double>("odom_cov_inflation_roll", 1.0),
             private_nh.param<double>("odom_cov_inflation_pitch", 1.0),
             private_nh.param<double>("odom_cov_inflation_yaw", 1.0);
    }
    preinteg_yaw_cap_ratio_ = private_nh.param<double>("preinteg_yaw_cap_ratio", 0.1);
    // Ratios above 1 let the preintegration outweigh the scan matching on the
    // axes it measurably wins; 0 leaves an axis uncapped. Defaults reproduce
    // the yaw-only cap, so the per-axis path is opt-in via cap_yaw_only=false.
    preinteg_cap_ratio_trans_fwd_ = private_nh.param<double>("preinteg_cap_ratio_trans_fwd", 1.0);
    preinteg_cap_ratio_trans_lat_ = private_nh.param<double>("preinteg_cap_ratio_trans_lat", 1.0);
    preinteg_cap_ratio_trans_vert_ = private_nh.param<double>("preinteg_cap_ratio_trans_vert", 2.0);
    preinteg_cap_ratio_roll_ = private_nh.param<double>("preinteg_cap_ratio_roll", 2.0);
    preinteg_cap_ratio_pitch_ = private_nh.param<double>("preinteg_cap_ratio_pitch", 2.0);
    // Keyframes per bias vertex. 0 is one vertex for the whole run, which is
    // right when the bias is constant and wrong when it is not.
    odometry_yaw_gate_deg_ = private_nh.param<double>("odometry_yaw_gate_deg", 3.0);
    odometry_yaw_gate_deflate_ = private_nh.param<double>("odometry_yaw_gate_deflate", 0.01);
    odom_trans_weight_from_preint_ = private_nh.param<bool>("odom_trans_weight_from_preint", false);
    preinteg_yaw_weight_from_disagreement_ =
        private_nh.param<bool>("preinteg_yaw_weight_from_disagreement", false);
    preinteg_yaw_var_tau_ = private_nh.param<double>("preinteg_yaw_var_tau", 60.0);
    preinteg_yaw_var_min_samples_ = private_nh.param<int>("preinteg_yaw_var_min_samples", 30);
    preinteg_gyr_scale_std = private_nh.param<double>("preinteg_gyr_scale_std", 0.02);
    ugpm::g_state_freq_source_gyr =
        private_nh.param<bool>("gp_state_freq_from_gyro", false);
    // Kernel lengthscales in seconds, separately for rotation and velocity.
    // Zero keeps 3/state_freq for that signal, so the defaults change nothing.
    ugpm::g_rot_lengthscale = private_nh.param<double>("gp_rot_lengthscale", 0.0);
    ugpm::g_vel_lengthscale = private_nh.param<double>("gp_vel_lengthscale", 0.0);
    // Largest accepted variance on the accumulated registration covariance,
    // above which the edge falls back to the legacy weighting. 1 m^2 is already
    // far looser than any healthy keyframe pair, which measure 1e-4 to 7e-3.
    odometry_max_cov_diagonal = private_nh.param<double>("odometry_max_cov_diagonal", 1.0);
    odometry_cross_check_ = private_nh.param<bool>("odometry_cross_check", true);
    odometry_cross_check_thresh_ = private_nh.param<double>("odometry_cross_check_thresh", 0.15);
    odometry_cross_check_deflate_ = private_nh.param<double>("odometry_cross_check_deflate", 0.01);
    // Allow the straight-line displacement to reach this multiple of the
    // ego-velocity path before the edge is distrusted. Above 1.0 only to absorb
    // ego-velocity underestimation; the geometry itself allows no more than 1.0.
    odometry_kinematic_bound_ = private_nh.param<bool>("odometry_kinematic_bound", true);
    odometry_path_margin_ = private_nh.param<double>("odometry_path_margin", 1.5);
    preinteg_gyr_var = private_nh.param<double>("preinteg_gyr_var", 0.0);
    preinteg_vel_var = private_nh.param<double>("preinteg_vel_var", 0.0);
    use_egovel_preinteg_trans = private_nh.param<bool>("use_egovel_preinteg_trans", false);


    points_topic = private_nh.param<std::string>("points_topic", "/radar_enhanced_pcl");


    // Base path for trajectory output; "<prefix>.txt" and "<prefix>_ns.txt".
    result_prefix = private_nh.param<std::string>("result_prefix", "/root/catkin_ws/odom");

    registration = select_registration_method(private_nh);

    // subscribers
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, odomTopic, 1000000));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 1000000));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(1000000), *odom_sub, *cloud_sub));
    sync->registerCallback(boost::bind(&RadarGraphSlamNodelet::cloud_callback, this, _1, _2));
    
    // }
    if (enable_preintegration)
      imu_odom_sub = nh.subscribe("/imu_pre_integ/imu_odom_incre", 1024, &RadarGraphSlamNodelet::imu_odom_callback, this);
    imu_sub = nh.subscribe(imuTopic, 1410065408, &RadarGraphSlamNodelet::imu_callback, this);
    ROS_INFO_STREAM("radar_graph_slam: subscribing IMU on " << imuTopic);
    command_sub = nh.subscribe("/command", 10, &RadarGraphSlamNodelet::command_callback, this);

    //***** publishers ******
    markers_pub = mt_nh.advertise<visualization_msgs::MarkerArray>("/radar_graph_slam/markers", 1000000);
    // Transform RadarOdom_to_base
    odom2base_pub = mt_nh.advertise<geometry_msgs::TransformStamped>("/radar_graph_slam/odom2base", 1000000);
    aftmapped_odom_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/aftmapped_odom", 1000000);
    aftmapped_odom_incremenral_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/aftmapped_odom_incremental", 1000000);
    map_points_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/radar_graph_slam/map_points", 1000000, true);
    odom_frame2frame_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/odom_frame2frame", 1000000);

    dump_service_server = mt_nh.advertiseService("/radar_graph_slam/dump", &RadarGraphSlamNodelet::dump_service, this);
    save_map_service_server = mt_nh.advertiseService("/radar_graph_slam/save_map", &RadarGraphSlamNodelet::save_map_service, this);

    graph_updated = false;
    double graph_update_interval = private_nh.param<double>("graph_update_interval", 3.0);
    double map_cloud_update_interval = private_nh.param<double>("map_cloud_update_interval", 10.0);
    optimization_timer = mt_nh.createWallTimer(ros::WallDuration(graph_update_interval), &RadarGraphSlamNodelet::optimization_timer_callback, this);
    cloud_handler_timer = mt_nh.createWallTimer(ros::WallDuration(0.1), &RadarGraphSlamNodelet::cloud_handler_callback, this);
    map_publish_timer = mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval), &RadarGraphSlamNodelet::map_points_publish_timer_callback, this);
  

  }


private:
  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   */
  void cloud_callback(const nav_msgs::OdometryConstPtr& odom_msg, const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    // Scoped to the queue push so the 200 Hz IMU callback is not blocked.
    {
      std::lock_guard<std::mutex> lock(cloud_queue_mutex);
      cloud_queue.push_back(cloud_msg);
      odom_queue.push_back(odom_msg);
    }
    const ros::Time& stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom_now = odom2isometry(odom_msg); // scan matching odometry ( accumulated from the beginning )
    
    Eigen::Matrix4d matrix_map2base = Eigen::Matrix4d::Identity();
    // // Publish TF between /map and /base_link
    if(keyframes.size() > 0)
    {
      const KeyFrame::Ptr& keyframe_last = keyframes.back();
      Eigen::Isometry3d lastkeyframe_odom_incre =  keyframe_last->odom_scan2scan.inverse() * odom_now; // odom increment
      Eigen::Isometry3d keyframe_map2base_matrix = keyframe_last->node->estimate();

      // map2base = odom^(-1) * base
      matrix_map2base = (keyframe_map2base_matrix * lastkeyframe_odom_incre).matrix();
    }
    geometry_msgs::TransformStamped map2base_trans = matrix2transform(cloud_msg->header.stamp, matrix_map2base, mapFrame, baselinkFrame);
    if (pow(map2base_trans.transform.rotation.w,2)+pow(map2base_trans.transform.rotation.x,2)+
      pow(map2base_trans.transform.rotation.y,2)+pow(map2base_trans.transform.rotation.z,2) < pow(0.9,2)) 
      {map2base_trans.transform.rotation.w=1; map2base_trans.transform.rotation.x=0; map2base_trans.transform.rotation.y=0; map2base_trans.transform.rotation.z=0;}
    map2base_broadcaster.sendTransform(map2base_trans);
   
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(baselinkFrame.empty()) {
      baselinkFrame = cloud_msg->header.frame_id;
    }
    // Ego velocity with its covariance. The estimator writes its per-axis
    // sigma^2 into twist.covariance[0,7,14] and /odom carries it through; the
    // GP needs it, entering the kernel as K + sz2*I.
    geometry_msgs::TwistWithCovarianceStamped::Ptr twist_(new geometry_msgs::TwistWithCovarianceStamped);
    twist_->header.stamp = cloud_msg->header.stamp;
    twist_->twist = odom_msg->twist; // ego velocity and its covariance
    {
      std::lock_guard<std::mutex> lock(twist_queue_mutex);
      twist_queue.push_back(twist_);
    }
  }


  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    // std::cout<<"imu callback"<<std::endl;
    // imu_queue.push_back(imu_msg);

    // Transform to Radar's Frame
    geometry_msgs::QuaternionStamped::Ptr imu_quat(new geometry_msgs::QuaternionStamped);
    imu_quat->quaternion = imu_msg->orientation;
    Eigen::Quaterniond imu_quat_from(imu_quat->quaternion.w, imu_quat->quaternion.x, imu_quat->quaternion.y, imu_quat->quaternion.z);
    Eigen::Quaterniond imu_quat_deskew = imu_quat_from * extQRPY; // world to lidar(base)
    imu_quat_deskew.normalize();

    static int cnt = 0;
    if(cnt == 0) {
      double roll, pitch, yaw;
      tf::Quaternion orientation = tf::Quaternion(imu_quat_deskew.x(),imu_quat_deskew.y(),imu_quat_deskew.z(),imu_quat_deskew.w());
      tf::quaternionMsgToTF(imu_msg->orientation, orientation);
      tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
      // Eigen::Matrix3d imu_mat_deskew = imu_quat_deskew.toRotationMatrix();
      // Eigen::Vector3d eulerAngle = imu_mat_deskew.eulerAngles(0,1,2); // roll pitch yaw
      Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
      Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
      Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()));
      Eigen::Matrix3d imu_mat_final; imu_mat_final = yawAngle * pitchAngle * rollAngle;

      Eigen::Isometry3d isom_initial_pose;
      isom_initial_pose.setIdentity();
      isom_initial_pose.rotate(imu_mat_final); // Set rotation
      initial_pose = isom_initial_pose.matrix();
      ROS_INFO("Initial Position Matrix = ");
      std::cout << 
        initial_pose(0,0) << ", " << initial_pose(0,1) << ", " << initial_pose(0,2) << ", " << initial_pose(0,3) << ", " << std::endl <<
        initial_pose(1,0) << ", " << initial_pose(1,1) << ", " << initial_pose(1,2) << ", " << initial_pose(1,3) << ", " << std::endl <<
        initial_pose(2,0) << ", " << initial_pose(2,1) << ", " << initial_pose(2,2) << ", " << initial_pose(2,3) << ", " << std::endl <<
        initial_pose(3,0) << ", " << initial_pose(3,1) << ", " << initial_pose(3,2) << ", " << initial_pose(3,3) << ", " << std::endl << std::endl;
      cnt = 1;
    }
    

    sensor_msgs::Imu::Ptr imu_data =
        boost::make_shared<sensor_msgs::Imu>(imuConverter(*imu_msg));
    imu_data->header.frame_id = baselinkFrame.empty() ? std::string("base_link") : baselinkFrame;

    if(imu_time_offset_ != 0.0) {
      imu_data->header.stamp += ros::Duration(imu_time_offset_);
    }


    imu_queue.push_back(imu_data);
  }

  void imu_odom_callback(const nav_msgs::OdometryConstPtr& imu_odom_msg) {
    {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    imu_odom_queue.push_back(imu_odom_msg);
    }
    // std::cout<<"imu_odom_queue size: "<<imu_odom_queue.size()<<std::endl;
  }

  bool trust = false;

  geometry_msgs::Transform preIntegrationTransform(){

    trust = true;
    last_preint_cov_valid_ = false;

    // Identity, not default-constructed: a default geometry_msgs::Transform has
    // an all-zero quaternion, which is not a valid rotation. It was being
    // returned on the early-exit paths, published on
    // /radar_graph_slam/odom_frame2frame and stored as keyframe->trans_integrated.
    geometry_msgs::Transform trans_;
    trans_.rotation.w = 1.0;

    if(keyframes.size() == 0) return trans_;

    double lastImuTime = -1;
    double delta_t = 1.0;
    size_t imu_odom_end_index = 0; // The index of the last used IMU
    
    // pop old IMU orientation message

    // imu queue mutex???

    // Scoped locks only: this function has early returns.
    {
      std::lock_guard<std::mutex> lock(imu_queue_mutex);
      while (!imu_queue.empty() && imu_queue.front()->header.stamp.toSec() < lastKeyframeTime - delta_t){
        lastImuTime = imu_queue.front()->header.stamp.toSec();
        imu_queue.pop_front();
      }
    }

    // pop old Twist message
    {
      std::lock_guard<std::mutex> lock(twist_queue_mutex);
      while (!twist_queue.empty() && twist_queue.front()->header.stamp.toSec() < lastKeyframeTime - delta_t){
        twist_queue.pop_front();
      }
    }

    // repropogate
    Eigen::Isometry3d isom_frame2frame;  // Translation increment between last key frame and this IMU msg
    Eigen::Isometry3d isometry_rotation; // Rotation of IMU odom
    Eigen::Isometry3d isometry_translation;  // Translation of IMU odom
    isom_frame2frame.setIdentity();
    isometry_rotation.setIdentity();
    
    geometry_msgs::Vector3 translation_frame2frame; // Translation between two key frames
    geometry_msgs::Quaternion rotation_frame2frame; // Rotation between two key frames
    // std::cout<< "imu_queue size: " << imu_queue.size() << std::endl;
    if (!imu_queue.empty())
    {
      std::deque<sensor_msgs::Imu::ConstPtr> imu_queue_interp;
      std::deque<sensor_msgs::Imu::ConstPtr> imu_queue_between;
      // Held only for the copy out of imu_queue, not for the GP solve that
      // follows: that takes tens of milliseconds and would block the 200 Hz IMU
      // callback for the duration.
      {
        std::lock_guard<std::mutex> imu_lock(imu_queue_mutex);

        // delta_t is the collection margin, wider than the GP's own window
        // margin so the latter is never truncated by the former.
        for(size_t ii = 0;
            ii < imu_queue.size() &&
            imu_queue.at(ii)->header.stamp.toSec() < thisKeyframeTime + delta_t;
            ++ii) {
          imu_queue_interp.push_back(imu_queue.at(ii));
        }
      }

      double keyframe_time_diff = thisKeyframeTime - lastKeyframeTime;
      keyframe_time.push_back(keyframe_time_diff);

      if(thisKeyframeTime - lastKeyframeTime > preinteg_max_interval){
        ROS_WARN_STREAM_THROTTLE(2.0, "preintegration: keyframe gap "
                                 << (thisKeyframeTime - lastKeyframeTime)
                                 << "s exceeds " << preinteg_max_interval
                                 << "s; omitting this edge");
        trust = false;
        return trans_;
      }
      

      ugpm::GyroVelData imu_; // imu and velocity

      const double gp_lo = lastKeyframeTime - preinteg_window_margin;
      const double gp_hi = thisKeyframeTime + preinteg_window_margin;

      imu_.gyr_var = preinteg_gyr_var > 0.0
          ? preinteg_gyr_var
          : static_cast<double>(imuGyrNoise) * static_cast<double>(imuGyrNoise);

      Eigen::Vector3d vel_var_sum = Eigen::Vector3d::Zero();
      int vel_var_n = 0;
      double vel_first_t = -1.0;
      double vel_last_t = -1.0;
      double vel_max_gap = 0.0;

      for(size_t i = 0; i < imu_queue_interp.size(); i++){
        const double t = imu_queue_interp.at(i)->header.stamp.toSec();
        if(t < gp_lo || t > gp_hi){
          continue;
        }

        ugpm::DataSample imu_gyr;
        imu_gyr.t = t;
        imu_gyr.data[0] = imu_queue_interp.at(i)->angular_velocity.x;
        imu_gyr.data[1] = imu_queue_interp.at(i)->angular_velocity.y;
        imu_gyr.data[2] = imu_queue_interp.at(i)->angular_velocity.z;
        imu_.gyr.push_back(imu_gyr);
      }

      // Snapshot under the lock, then read: twist_queue is appended from
      // cloud_callback on another thread.
      std::deque<geometry_msgs::TwistWithCovarianceStampedConstPtr> twist_snapshot;
      {
        std::lock_guard<std::mutex> twist_lock(twist_queue_mutex);
        twist_snapshot = twist_queue;
      }

      for (size_t i = 0; i < twist_snapshot.size(); i++)
      {
        const double t = twist_snapshot.at(i)->header.stamp.toSec();
        if(t < gp_lo || t > gp_hi){
          continue;
        }

        ugpm::DataSample ego_vel;
        ego_vel.t = t;
        ego_vel.data[0] = twist_snapshot.at(i)->twist.twist.linear.x;
        ego_vel.data[1] = twist_snapshot.at(i)->twist.twist.linear.y;
        ego_vel.data[2] = twist_snapshot.at(i)->twist.twist.linear.z;
        imu_.vel.push_back(ego_vel);

        // The estimator's per-axis variance for this frame. Averaged over the
        // window because GyroVelData carries one value per axis for the whole
        // window, but no longer averaged across the axes.
        const auto& c = twist_snapshot.at(i)->twist.covariance;
        vel_var_sum += Eigen::Vector3d(c[0], c[7], c[14]);
        vel_var_n += 1;

        if(vel_first_t < 0.0) vel_first_t = t;
        if(vel_last_t >= 0.0) vel_max_gap = std::max(vel_max_gap, t - vel_last_t);
        vel_last_t = t;
      }

      // Velocity variance from the measurements. A floor is applied because a
      // zero or near-zero variance makes K + sz2*I singular, which is exactly
      // the failure this gate exists to prevent.
      // What the estimator reported for this window, untouched.
      const Eigen::Vector3d vel_var_raw = vel_var_n > 0
          ? (vel_var_sum / vel_var_n).eval()
          : Eigen::Vector3d::Constant(preinteg_min_vel_var);
      const double vmax = vel_var_raw.maxCoeff();
      imu_.vel_var = (vmax > 0.0 && vmax < preinteg_min_vel_var)
          ? (vel_var_raw * (preinteg_min_vel_var / vmax)).eval()
          : vel_var_raw.cwiseMax(preinteg_min_vel_var).eval();
      if(reported_vel_var_ < 5) {
        reported_vel_var_++;
        const Eigen::Vector3d raw = vel_var_raw.cwiseSqrt();
        std::cout << "ego-vel sigma raw " << raw.transpose()
                  << "  z/xy " << raw(2) / (0.5 * (raw(0) + raw(1)))
                  << "  floored " << imu_.vel_var.cwiseSqrt().transpose() << std::endl;
      }
      if(reported_vel_var_ <= 5) {
        std::cout << "   -> vel_var handed to the GP: "
                  << imu_.vel_var.cwiseSqrt().transpose() << " (sigma m/s)" << std::endl;
      }
      if(preinteg_vel_var > 0.0) {
        imu_.vel_var.setConstant(preinteg_vel_var);  // explicit override, for reproduction
      }

      const double interval = thisKeyframeTime - lastKeyframeTime;
      const double vel_coverage = (vel_first_t >= 0.0 && vel_last_t > vel_first_t)
          ? (vel_last_t - vel_first_t) : 0.0;

      if(imu_.gyr.size() < static_cast<size_t>(preinteg_min_gyr_samples) ||
         imu_.vel.size() < static_cast<size_t>(preinteg_min_vel_samples) ||
         vel_max_gap > preinteg_max_vel_gap ||
         vel_coverage < preinteg_min_vel_coverage * interval){
        ROS_WARN_STREAM_THROTTLE(2.0, "preintegration: data does not support the "
                                 << interval << "s interval -- " << imu_.gyr.size()
                                 << " gyro, " << imu_.vel.size() << " ego-vel spanning "
                                 << vel_coverage << "s, largest gap " << vel_max_gap
                                 << "s; omitting this edge");
        trust = false;
        return trans_;
      }
      
      ugpm::PreintPrior prior_bias;
      {
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        prior_bias.gyr_bias = {b.x(), b.y(), b.z()};
      }
      ugpm::PreintOption preint_opt;

      preint_opt.type = ugpm::UGPM;
      // preint_opt.quantum = 0.05;
      double start_t = lastKeyframeTime;
      double end_t = thisKeyframeTime;

      std::vector<std::vector<double> > t;
      std::vector<double> temp_t;
      temp_t.push_back(end_t);
      t.push_back(temp_t);

      // GP solve cost, recorded separately from the surrounding bookkeeping so the
      // oversized-window problem can be attributed. Also record how many samples
      // the solve was actually handed versus the interval being integrated.
      const ros::WallTime gp_t0 = ros::WallTime::now();

      ugpm::VelPreintegration preintegration(imu_, start_t, t, preint_opt, prior_bias, true);
      ugpm::PreintMeas preint_meas =
          preintegration.get(0, 0, preinteg_vel_bias_std, preinteg_gyr_bias_std,
                             preinteg_gyr_scale_std);
      gp_solve_time.push_back((ros::WallTime::now() - gp_t0).toSec());
      gp_gyr_samples.push_back(static_cast<double>(imu_.gyr.size()));
      gp_vel_samples.push_back(static_cast<double>(imu_.vel.size()));
      gp_interval.push_back(end_t - start_t);

      for(int i = 0; i < 6; ++i) {
        if(preinteg_cov_inflation_(i) > 1.0) {
          preint_meas.cov.row(i) *= std::sqrt(preinteg_cov_inflation_(i));
          preint_meas.cov.col(i) *= std::sqrt(preinteg_cov_inflation_(i));
        }
      }
      last_preint_cov_ = preint_meas.cov;
      last_preint_cov_valid_ = last_preint_cov_.allFinite();
      // Kept for the bias vertex: the Jacobian of the integrated rotation with
      // respect to the gyro bias, and the bias this measurement was integrated
      // with, so the edge can correct itself as the graph moves the bias.

      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(
            0.5 * (last_preint_cov_ + last_preint_cov_.transpose()));
        std::cout << "PCOV rot_diag=" << last_preint_cov_.diagonal().head<3>().transpose()
                  << " trans_diag=" << last_preint_cov_.diagonal().tail<3>().transpose()
                  << " eig_min=" << es.eigenvalues().minCoeff()
                  << " eig_max=" << es.eigenvalues().maxCoeff()
                  << " dur=" << (thisKeyframeTime - lastKeyframeTime)
                  << std::endl;
      }

      Eigen::Quaterniond q_gp(preint_meas.delta_R);

      rotation_frame2frame.x = q_gp.x();
      rotation_frame2frame.y = q_gp.y();
      rotation_frame2frame.z = q_gp.z();
      rotation_frame2frame.w = q_gp.w();

      Eigen::Vector3d p_gp(preint_meas.delta_p);

      translation_frame2frame.x = p_gp(0);
      translation_frame2frame.y = p_gp(1);
      translation_frame2frame.z = p_gp(2);

      std::cout << "imu integrated rotation: " << q_gp.toRotationMatrix().eulerAngles(2,1,0).transpose() * 180 / M_PI << std::endl;
      std::cout << "Integrated translation: " << p_gp.transpose() << std::endl;

      trans_.rotation = rotation_frame2frame;
      trans_.translation = translation_frame2frame;

      {
        const Eigen::Quaterniond q_pre(rotation_frame2frame.w, rotation_frame2frame.x,
                                       rotation_frame2frame.y, rotation_frame2frame.z);
        // The previous keyframe is the last one queued, not the last one
        // flushed: keyframes are flushed in batches by the optimisation timer,
        // so keyframes.back() lags by however many are waiting. Using it made
        // the logged scan matching increment span several keyframe intervals
        // while t0/t1 spanned one, which showed up as a spurious 25% failure
        // population in the scan matching translation.
        Eigen::Isometry3d prev_odom = Eigen::Isometry3d::Identity();
        bool have_prev = false;
        if(!keyframe_queue.empty()) {
          prev_odom = keyframe_queue.back()->odom_scan2scan;
          have_prev = true;
        } else if(!keyframes.empty()) {
          prev_odom = keyframes.back()->odom_scan2scan;
          have_prev = true;
        }
        const Eigen::Isometry3d rel_sm =
            have_prev ? Eigen::Isometry3d(prev_odom.inverse() * odom_now_for_log_)
                      : Eigen::Isometry3d::Identity();
        Eigen::Quaterniond q_sm(rel_sm.rotation());
        const Eigen::Vector3d t_sm = rel_sm.translation();

        const double dt_kf = thisKeyframeTime - lastKeyframeTime;
        if(have_prev && dt_kf > 0.05 && dt_kf < preinteg_max_interval) {
          const Eigen::AngleAxisd aa(
              (q_sm.normalized().inverse() * q_pre.normalized()).normalized());
          const Eigen::Vector3d d = aa.angle() * aa.axis();
          if(d.allFinite()) {
            {
              const double alpha_v = std::min(1.0, dt_kf / preinteg_yaw_var_tau_);
              const double dyaw = d(2);
              yaw_disagree_var_ = (1.0 - alpha_v) * yaw_disagree_var_ + alpha_v * dyaw * dyaw;
              yaw_disagree_n_++;
            }

          }
        }
        std::cout << std::setprecision(12)
                  << "PROT t0=" << lastKeyframeTime << " t1=" << thisKeyframeTime
                  << " pre=" << q_pre.x() << "," << q_pre.y() << "," << q_pre.z() << "," << q_pre.w()
                  << " sm=" << q_sm.x() << "," << q_sm.y() << "," << q_sm.z() << "," << q_sm.w()
                  << " pret=" << translation_frame2frame.x << "," << translation_frame2frame.y
                  << "," << translation_frame2frame.z
                  << " smt=" << t_sm.x() << "," << t_sm.y() << "," << t_sm.z()
                  << std::endl;
      }
    }
    return trans_;
  }

  /**
   * @brief this method adds all the keyframes_ in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe_ was added to the pose graph
   */
  bool flush_keyframe_queue() {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    if(keyframe_queue.empty()) {
      return false;
    }

    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    int num_processed = 0;
    // ********** Select number of keyframess to be optimized **********
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); ++i) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframess will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      Eigen::Isometry3d odom = odom2map * keyframe->odom_scan2scan;
      // ********** Vertex of keyframess is contructed here ***********
      keyframe->node = graph_slam->add_se3_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        if(private_nh.param<bool>("fix_first_node", false)) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
          std::stringstream sst(private_nh.param<std::string>("fix_first_node_stddev", "1 1 1 1 1 1"));
          for(int i = 0; i < 6; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }
          anchor_node = graph_slam->add_se3_node(Eigen::Isometry3d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se3_edge(anchor_node, keyframe->node, Eigen::Isometry3d::Identity(), inf);
        }
      }
      
      if(i == 0 && keyframes.empty()) {
        if(fix_first_keyframe_) {
          keyframe->node->setFixed(true);
        }
        continue;
      }

      /***** Scan-to-Scan Add edge to between consecutive keyframes *****/
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];
      // relative pose between odom of previous frame and this frame R2=R12*R1 => R12 = inv(R2) * R1
      Eigen::Isometry3d relative_pose = keyframe->odom_scan2scan.inverse() * prev_keyframe->odom_scan2scan;

      const bool odom_cov_sane_for_weight = keyframe->odom_cov_valid;
      Eigen::MatrixXd information = odometry_information(keyframe, prev_keyframe, relative_pose);
      const Eigen::MatrixXd information_claimed = information;

      if(odom_trans_weight_from_preint_ && keyframe->preint_cov_valid) {
        const Eigen::MatrixXd preint_info =
            preintegration_information(keyframe->preint_cov);
        for(int i = 0; i < 3; ++i) {
          const double cap = preint_info(i, i);
          if(cap > 0.0 && information(i, i) > cap) {
            const double sc = std::sqrt(cap / information(i, i));
            information.row(i) *= sc;
            information.col(i) *= sc;
            trans_weight_capped_++;
          }
        }
      }

      if(odometry_kinematic_bound_ && keyframe->egovel_path > 0.0) {
        const double claimed = relative_pose.translation().norm();
        const double bound = odometry_path_margin_ * keyframe->egovel_path;
        if(claimed > bound) {
          ROS_WARN_STREAM("scan matching claims " << claimed << " m where the ego-velocity "
                          "accounts for " << keyframe->egovel_path
                          << " m; deflating that edge");
          information *= odometry_cross_check_deflate_;
          odometry_bound_hits_++;
        }
      }

      if(odometry_cross_check_ && keyframe->trust_omega && enable_preintegration) {
        const Eigen::Isometry3d preint = transform2isometry(keyframe->trans_integrated);
        // relative_pose is T_cur_prev, the preintegration is T_prev_cur
        const double disagreement =
            (preint.translation() + relative_pose.translation()).norm();
        if(disagreement > odometry_cross_check_thresh_) {
          information *= odometry_cross_check_deflate_;
          odometry_cross_check_hits_++;
        }
      }
      if(odometry_yaw_gate_deg_ > 0.0 && keyframe->trust_omega && enable_preintegration) {
        const Eigen::Isometry3d preint = transform2isometry(keyframe->trans_integrated);
        // preint is T_prev_cur, relative_pose is T_cur_prev: their product is the disagreement
        const Eigen::Matrix3d Rd = preint.rotation() * relative_pose.rotation();
        const double dyaw_deg = std::abs(std::atan2(Rd(1, 0), Rd(0, 0))) * 180.0 / M_PI;
        if(dyaw_deg > odometry_yaw_gate_deg_) {
          information *= odometry_yaw_gate_deflate_;
          odometry_yaw_gate_hits_++;
        }
      }


      auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"), private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));


      if(enable_preintegration && keyframe->trust_omega && !keyframe->preint_cov_valid) {
        preint_cov_nonfinite_++;
      }
      if (enable_preintegration && keyframe->trust_omega && keyframe->preint_cov_valid){
        // Add Preintegration edge
        geometry_msgs::Transform relative_trans = keyframe->trans_integrated;
        g2o::SE3Quat relative_se3quat ( Eigen::Quaterniond(relative_trans.rotation.w, relative_trans.rotation.x, relative_trans.rotation.y, relative_trans.rotation.z), 
                                        Eigen::Vector3d(relative_trans.translation.x, relative_trans.translation.y, relative_trans.translation.z));
        Eigen::Isometry3d relative_isometry = transform2isometry(relative_trans);
        Eigen::MatrixXd information_integ =
            preintegration_information(keyframe->preint_cov);

        if(preinteg_yaw_weight_from_disagreement_ && odom_cov_sane_for_weight &&
           yaw_disagree_n_ >= preinteg_yaw_var_min_samples_) {
          const double var_sm = (information_claimed(5, 5) > 0.0) ? 1.0 / information_claimed(5, 5) : 0.0;
          const double var_gp = std::max(yaw_disagree_var_ - var_sm, preinteg_yaw_var_floor_);
          const double target = 1.0 / var_gp;
          if(information_integ(5, 5) > target) {
            const double sc = std::sqrt(target / information_integ(5, 5));
            information_integ.row(5) *= sc;
            information_integ.col(5) *= sc;
            yaw_weight_from_disagreement_++;
          }
        } else if(preinteg_rot_weight_from_odom_ && odom_cov_sane_for_weight) {
          const double axis_ratio[6] = {
              preinteg_cap_ratio_trans_fwd_,  // forward: tie, so equalise
              preinteg_cap_ratio_trans_lat_,  // lateral: slight GP edge
              preinteg_cap_ratio_trans_vert_, // vertical: GP clearly better
              preinteg_cap_ratio_roll_,       // roll: GP better, gravity-observed
              preinteg_cap_ratio_pitch_,      // pitch: same
              preinteg_yaw_cap_ratio_,        // yaw: no GP advantage
          };
          const int lo = preinteg_weight_cap_yaw_only_ ? 5 : 0;
          for(int i = lo; i < 6; ++i) {
            const double cap = axis_ratio[i] * information_claimed(i, i);
            if(cap <= 0.0) {
              continue;  // a non-positive ratio means "leave this axis alone"
            }
            if(information_integ(i, i) > cap) {
              const double sc = std::sqrt(cap / information_integ(i, i));
              information_integ.row(i) *= sc;
              information_integ.col(i) *= sc;
              rot_weight_capped_++;
            }
          }
        }

        if(preinteg_edge_dofs_ == "rotation_only") {
          information_integ.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-6;
          information_integ.block<3, 3>(0, 3).setZero();
          information_integ.block<3, 3>(3, 0).setZero();
        } else if(preinteg_edge_dofs_ == "translation_only") {
          information_integ.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-6;
          information_integ.block<3, 3>(0, 3).setZero();
          information_integ.block<3, 3>(3, 0).setZero();
        }

        // With the bias vertex the edge carries the Jacobian and re-evaluates
        // its own rotation as the graph moves the bias; otherwise it is the
        // plain SE3 edge and the bias stays wherever it was set.
 else {
          auto edge_integ = graph_slam->add_se3_edge(prev_keyframe->node, keyframe->node, relative_isometry, information_integ);
          graph_slam->add_robust_kernel(edge_integ, private_nh.param<std::string>("integ_edge_robust_kernel", "NONE"), private_nh.param<double>("integ_edge_robust_kernel_size", 1.0));
        }
      }
    }


    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    return true;
  }

  void cloud_handler_callback(const ros::WallTimerEvent& event) {
    sensor_msgs::PointCloud2::ConstPtr cloud_msg;
    nav_msgs::OdometryConstPtr odom_msg;
    {
      // Both queues are filled together in cloud_callback, so they stay the same
      // length; the empty check and both pops belong in one critical section.
      std::lock_guard<std::mutex> lock(cloud_queue_mutex);
      if(cloud_queue.empty() || odom_queue.empty()) {
        return;
      }
      cloud_msg = cloud_queue.front();
      cloud_queue.pop_front();
      odom_msg = odom_queue.front();
      odom_queue.pop_front();
    }

    const ros::Time& stamp = cloud_msg->header.stamp;

    // How far behind the sensor stream this frame is being processed. With
    // use_sim_time, ros::Time::now() is bag time, so this is the intake lag in
    // bag seconds; cloud_queue.size() is the backlog still waiting. Both grow
    // without bound once per-frame cost exceeds the frame period.
    intake_lag.push_back((ros::Time::now() - stamp).toSec());
    intake_backlog.push_back(static_cast<double>(cloud_queue.size()));

    {
      Eigen::Matrix<double, 6, 6> frame_cov;
      for(int r = 0; r < 6; ++r) {
        for(int c = 0; c < 6; ++c) {
          frame_cov(r, c) = odom_msg->pose.covariance[6 * r + c];
        }
      }
      if(frame_cov.allFinite() && frame_cov.diagonal().maxCoeff() > 0.0) {
        accum_odom_cov_ = frame_cov;
        accum_odom_frames_++;
      }

      // Path length the ego-velocity accounts for over the same span. A
      // straight-line displacement cannot exceed the distance actually
      // travelled, so this is a hard bound on the scan matching translation --
      // no tuning, just kinematics.
      const Eigen::Vector3d v(odom_msg->twist.twist.linear.x,
                              odom_msg->twist.twist.linear.y,
                              odom_msg->twist.twist.linear.z);
      const double dt = last_intake_stamp_ > 0.0 ? (stamp.toSec() - last_intake_stamp_) : 0.0;
      if(dt > 0.0 && dt < 1.0 && v.allFinite()) {
        accum_egovel_path_ += v.norm() * dt;
      }
      last_intake_stamp_ = stamp.toSec();
    }

    Eigen::Isometry3d odom_now = odom2isometry(odom_msg); // scan matching odometry in radar frame ( accumulated from the beginning )
    odom_now_for_log_ = odom_now;
    if(lastKeyframeTime == 0) {
      lastKeyframeTime = stamp.toSec();
    }
    

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);


    //********** Decided whether to accept the frame as a key frame or not **********
    if(!keyframe_updater->decide(odom_now, stamp)) {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      if(keyframe_queue.empty()) {
      }
      return;
    }
    std::cout << "***********************************************" << std::endl;
    std::cout << "----------- preintegration checking -----------" << std::endl;
    std::cout << "***********************************************" << std::endl;
    std::cout<<std::setprecision(15);
    
    // Get time of this key frame for Intergeration, to integerate between two key frames
    thisKeyframeTime = cloud_msg->header.stamp.toSec();

    if(keyframes.size() > 0) {
      const KeyFrame::Ptr& keyframe_last = keyframes.back();
      Eigen::Isometry3d lastkeyframe_odom_incre =  keyframe_last->odom_scan2scan.inverse() * odom_now;
      std::cout << "scan matching rot: " << lastkeyframe_odom_incre.rotation().eulerAngles(2,1,0).transpose() * 180 / M_PI<< std::endl;
      std::cout << "scan matching trans: "<< lastkeyframe_odom_incre.translation().x() << ", " << lastkeyframe_odom_incre.translation().y() << ", " << lastkeyframe_odom_incre.translation().z() << std::endl;
    }
    
    double accum_d = keyframe_updater->get_accum_distance();
    // Construct keyframe
    KeyFrame::Ptr keyframe(new KeyFrame(keyframe_index, stamp, odom_now, accum_d, cloud)); // location ( not delta )
    keyframe_index ++;

    keyframe->odom_cov = accum_odom_cov_;
    keyframe->odom_cov_valid = (accum_odom_frames_ > 0);
    keyframe->odom_cov_frames = accum_odom_frames_;
    keyframe->egovel_path = accum_egovel_path_;
    accum_egovel_path_ = 0.0;

    if(accum_odom_frames_ > 0) {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(
          0.5 * (accum_odom_cov_ + accum_odom_cov_.transpose()));
      std::cout << "OCOV frames=" << accum_odom_frames_
                << " trans_diag=" << accum_odom_cov_.diagonal().head<3>().transpose()
                << " rot_diag=" << accum_odom_cov_.diagonal().tail<3>().transpose()
                << " eig_min=" << es.eigenvalues().minCoeff()
                << " eig_max=" << es.eigenvalues().maxCoeff()
                << std::endl;
    }
    accum_odom_cov_.setZero();
    accum_odom_frames_ = 0;

    // egovel integration : delta
    
    if (enable_preintegration){
      // Intergerate translation of ego velocity, add rotation
      // Wall time, not clock(): clock() returns process CPU time summed over all
      // threads, so it over-reports anything that runs OpenMP/Ceres internally.
      const ros::WallTime integ_t0 = ros::WallTime::now();
      geometry_msgs::Transform transf_integ = preIntegrationTransform();
      integration_time.push_back((ros::WallTime::now() - integ_t0).toSec());
      
      static uint32_t sequ = 0;
      nav_msgs::Odometry odom_frame2frame;
      odom_frame2frame.pose.pose.orientation = transf_integ.rotation;
      odom_frame2frame.pose.pose.position.x = transf_integ.translation.x;
      odom_frame2frame.pose.pose.position.y = transf_integ.translation.y;
      odom_frame2frame.pose.pose.position.z = transf_integ.translation.z;
      odom_frame2frame.header.frame_id = "map";
      odom_frame2frame.header.stamp = cloud_msg->header.stamp;
      odom_frame2frame.header.seq = sequ; sequ ++;
      odom_frame2frame_pub.publish(odom_frame2frame);
      keyframe->trans_integrated = transf_integ;
      keyframe->preint_cov = last_preint_cov_;
      keyframe->preint_cov_valid = last_preint_cov_valid_;
      if(trust == true){
        keyframe->trust_omega = true;
      }
      else{
        keyframe->trust_omega = false;
      }
      // std::cout << "Integrated Translation: " << transf_integ.translation.x << ", " << transf_integ.translation.y << ", " << transf_integ.translation.z << std::endl;
    }

    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    keyframe_queue.push_back(keyframe);

    // Scan Context descriptor for this keyframe, which the detector matches
    // against. Upstream built it for every keyframe regardless of the flag; here
    // it is skipped entirely when loop closure is off, which is the only reason
    // the two configurations differ in cost.
    if(enable_loop_closure_) {
      loop_detector->scManager->makeAndSaveScancontextAndKeys(*cloud);
    }

    lastKeyframeTime = thisKeyframeTime;
  }


  /**
   * @brief Back-end Optimization. This methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback(const ros::WallTimerEvent& event) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add the keyframes in the queue to the pose graph
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) {
    }

    if(!keyframe_updated) {
      return;
    }
    
    if(enable_loop_closure_) {
      loop_detector->detect(keyframes, new_keyframes, *graph_slam);
    }

    // new_keyframes is what the detector searches against; copy it across only
    // after detection has run.
    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    if(enable_loop_closure_) {
      addLoopFactor();
    }

    // move the first node / position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && private_nh.param<bool>("fix_first_node_adaptive", true)) {
      Eigen::Isometry3d anchor_target = static_cast<g2o::VertexSE3*>(anchor_edge->vertices()[1])->estimate();
      anchor_node->setEstimate(anchor_target);
    }

    // optimize the pose graph
    int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
    // Wall time, not clock(): g2o/cholmod are multithreaded, so clock() sums CPU
    // time across threads and inflates the reported optimization cost.
    const ros::WallTime opt_t0 = ros::WallTime::now();
    graph_slam->optimize(num_iterations);
    opt_time.push_back((ros::WallTime::now() - opt_t0).toSec());

    //********** publish tf **********
    const auto& keyframe = keyframes.back();
    // RadarOdom_to_base = map_to_base * map_to_RadarOdom^(-1)
    Eigen::Isometry3d trans = keyframe->node->estimate() * keyframe->odom_scan2scan.inverse();
    Eigen::Isometry3d map2base_trans = keyframe->node->estimate();
    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix();
    // map2base_incremental = map2base_last^(-1) * map2base_this 
    trans_aftmapped_incremental = trans_aftmapped.inverse() * map2base_trans;
    trans_aftmapped = map2base_trans;
    trans_odom2map_mutex.unlock();

    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());
    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;

    // Publish After-Mapped Odometry
    nav_msgs::Odometry aft = isometry2odom(keyframe->stamp, trans_aftmapped, mapFrame, odometryFrame);
    aftmapped_odom_pub.publish(aft);

    // Publish After-Mapped Odometry Incrementation
    nav_msgs::Odometry aft_incre = isometry2odom(keyframe->stamp, trans_aftmapped_incremental, mapFrame, odometryFrame);
    aftmapped_odom_incremenral_pub.publish(aft_incre);

    // Publish /odom to /base_link
    if(odom2base_pub.getNumSubscribers()) {  // Returns the number of subscribers that are currently connected to this Publisher
      geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, trans.matrix(), mapFrame, odometryFrame);
      odom2base_pub.publish(ts);
    }

    if(markers_pub.getNumSubscribers()) {
      auto markers = create_marker_array(ros::Time::now());
      markers_pub.publish(markers);
    }
  }

  /**
   * @brief write the optimized keyframe poses in TUM format
   * @param path         destination file
   * @param time_scale   1.0 for seconds, 1e9 for nanoseconds
   *
   * Format matches the NTU ground truth files (gt_odom.txt), so evo can compare
   * them directly with no conversion.
   */
  void write_trajectory(const std::string& path, double time_scale) const {
    std::ofstream fout(path, std::ios::out);
    if (!fout.is_open()) {
      ROS_ERROR_STREAM("failed to open trajectory output file: " << path);
      return;
    }
    fout << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    fout.setf(std::ios::fixed, std::ios::floatfield);
    fout.precision(8);
    for(size_t i = 0; i < keyframes.size(); i++) {
      const Eigen::Isometry3d est = keyframes[i]->node->estimate();
      const Eigen::Vector3d pos_ = est.translation();
      const Eigen::Quaterniond quat_(est.rotation());
      fout << keyframes[i]->stamp.toSec() * time_scale << " "
           << pos_(0) << " " << pos_(1) << " " << pos_(2) << " "
           << quat_.x() << " " << quat_.y() << " " << quat_.z() << " " << quat_.w()
           << std::endl;
    }
    fout.close();
  }

  /**
   * @brief print median / mean / max / count of a collected metric
   *
   * The existing "time" handler sorts each vector in place and reads the median
   * only; median alone hides the tail, which is what actually causes the intake
   * path to fall behind. Takes a copy so repeated dumps stay valid.
   */
  static void report_stats(const std::string& label, std::vector<double> v) {
    if (v.empty()) {
      return;
    }
    std::sort(v.begin(), v.end());
    const double median = v.at(v.size() / 2);
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    const double p95 = v.at(static_cast<size_t>(0.95 * (v.size() - 1)));
    cout << label << ": median=" << median << " mean=" << mean
         << " p95=" << p95 << " max=" << v.back() << " n=" << v.size() << endl;
  }


  /**
   * @brief average the gyro over the opening stationary period to get its bias
   *
   * The preintegration was being handed a zero bias prior, so any constant gyro
   * offset was integrated straight into every delta_R. Nothing else in the
   * pipeline estimates it: there is no bias state in the graph, and the GP
   * treats the bias as known.
   *
   * cp opens nearly still -- 0.174 m of ground-truth motion in the first five
   * seconds -- which is the usual condition for this estimate. Samples whose
   * rate is high are skipped so that a deliberate rotation
   * during the window cannot be absorbed into the bias.
   */

  /**
   * @brief information matrix for a scan matching edge
   *
   * Inverts the registration covariance the front end accumulated for this
   * keyframe pair. That covariance comes from the GICP Gauss-Newton Hessian
   * rescaled by the measured residual, so it varies per edge and per degree of
   * freedom according to what the geometry actually constrained.
   *
   * When that covariance is missing or not invertible, mostly during the
   * startup accumulation, the edge falls back to calc_information_matrix with
   * a fixed yaw term. The fallback's adaptive weighting saturates on sparse
   * radar clouds, so the Hessian path is preferred wherever it is available.
   */
  Eigen::MatrixXd odometry_information(const KeyFrame::Ptr& keyframe,
                                       const KeyFrame::Ptr& prev_keyframe,
                                       const Eigen::Isometry3d& relative_pose) const {
    const bool odom_cov_sane =
        keyframe->odom_cov_valid &&
        keyframe->odom_cov.diagonal().maxCoeff() < odometry_max_cov_diagonal;

    if(odom_cov_sane) {
      Eigen::Matrix<double, 6, 6> sym = 0.5 * (keyframe->odom_cov + keyframe->odom_cov.transpose());
      for(int i = 0; i < 6; ++i) {
        if(odom_cov_inflation_(i) > 1.0) {
          const double f = std::sqrt(odom_cov_inflation_(i));
          sym.row(i) *= f;
          sym.col(i) *= f;
        }
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(sym);
      Eigen::Matrix<double, 6, 1> eigs = solver.eigenvalues();
      bool clamped = false;
      for(int i = 0; i < 6; ++i) {
        if(!(eigs(i) > odometry_min_information_eigenvalue)) {
          eigs(i) = odometry_min_information_eigenvalue;
          clamped = true;
        }
      }
      if(clamped) {
        odom_cov_clamped_++;
      }
      const Eigen::Matrix<double, 6, 6> reg =
          solver.eigenvectors() * eigs.asDiagonal() * solver.eigenvectors().transpose();
      const Eigen::Matrix<double, 6, 6> inf = reg.inverse();
      if(inf.allFinite()) {
        return inf;
      }
    }

    {
      odom_cov_fallbacks_++;
    }
    Eigen::MatrixXd information =
        inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
    information(5, 5) = 1.0;
    return information;
  }


  /**
   * @brief add the loop closures found since the last call
   *
   * The queues are drained here, so a loop closure is added exactly once.
   * Leaving them filled would re-add every closure on each optimisation cycle,
   * multiplying that constraint's effective information.
   */
  void addLoopFactor()
  {
    auto& idx_queue = loop_detector->loopIndexQueue;
    auto& pose_queue = loop_detector->loopPoseQueue;
    auto& info_queue = loop_detector->loopInfoQueue;

    if (idx_queue.empty())
      return;

    const std::string kernel =
        private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE");
    const double kernel_size =
        private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0);

    size_t added = 0;
    for (size_t i = 0; i < idx_queue.size(); ++i) {
      const int indexFrom = idx_queue[i].first;
      const int indexTo = idx_queue[i].second;
      if (indexFrom < 0 || indexTo < 0 ||
          indexFrom >= static_cast<int>(keyframes.size()) ||
          indexTo >= static_cast<int>(keyframes.size())) {
        ROS_WARN_STREAM("loop closure references keyframe " << indexFrom << "/" << indexTo
                        << " but only " << keyframes.size() << " exist; skipping");
        continue;
      }
      // Second guard against duplicates: the detector's own loopIndexContainer
      // should prevent them, but add_se3_edge does not check, so a repeat would
      // silently double that constraint's weight.
      const auto key = std::make_pair(indexFrom, indexTo);
      if (!added_loop_edges_.insert(key).second) {
        continue;
      }
      auto edge = graph_slam->add_se3_edge(keyframes[indexFrom]->node, keyframes[indexTo]->node,
                                           pose_queue[i], info_queue[i]);
      graph_slam->add_robust_kernel(edge, kernel, kernel_size);
      added++;
    }

    if (added > 0) {
      ROS_INFO_STREAM("added " << added << " loop closure edge(s); "
                      << added_loop_edges_.size() << " total");
    }

    idx_queue.clear();
    pose_queue.clear();
    info_queue.clear();
  }

  /**
   * @brief information matrix for a preintegration edge
   *
   * Inverts the covariance the GP already propagated. This is the
   * measured uncertainty of that particular keyframe pair, so it needs no tuning
   * constants and it varies with how well the interval was actually observed.
   *
   * There is no second source. The original fixed matrix was built from six
   * per-sequence preinteg_*_stddev constants using 1/stddev where the scan
   * matching edge uses 1/variance, so the two factor types were never on a
   * common scale; that mismatch is why the constants had to be so extreme
   * (roll 10000, giving an information of 1e-4 against scan matching's 25) and
   * why the orientation constraint contributed essentially nothing. A keyframe
   * whose covariance is not finite gets no edge at all rather than a constant.
   *
   * The covariance path has to permute: ugpm orders it [rotation, translation]
   * and g2o's EdgeSE3 orders its error [translation, rotation]. Confirmed by
   * measurement -- getting this wrong swaps the two weights without any
   * visible error.
   */
  Eigen::MatrixXd preintegration_information(const Eigen::Matrix<double, 6, 6>& cov) const {
    // Symmetrise, then floor the eigenvalues before inverting. A near-singular
    // direction would otherwise invert to an enormous weight and let one edge
    // dominate the graph, which is the failure mode this whole exercise started
    // from.
    Eigen::Matrix<double, 6, 6> sym = 0.5 * (cov + cov.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(sym);
    Eigen::Matrix<double, 6, 1> eigs = solver.eigenvalues();
    bool clamped = false;
    for(int i = 0; i < 6; ++i) {
      if(!(eigs(i) > preinteg_min_information_eigenvalue)) {
        eigs(i) = preinteg_min_information_eigenvalue;
        clamped = true;
      }
    }
    if(clamped) {
      preint_cov_clamped_++;
    }
    const Eigen::Matrix<double, 6, 6> regularised =
        solver.eigenvectors() * eigs.asDiagonal() * solver.eigenvectors().transpose();
    const Eigen::Matrix<double, 6, 6> inv = regularised.inverse();

    // [rot, trans] -> [trans, rot]
    Eigen::MatrixXd inf = Eigen::MatrixXd::Zero(6, 6);
    inf.block<3, 3>(0, 0) = inv.block<3, 3>(3, 3);
    inf.block<3, 3>(3, 3) = inv.block<3, 3>(0, 0);
    inf.block<3, 3>(0, 3) = inv.block<3, 3>(3, 0);
    inf.block<3, 3>(3, 0) = inv.block<3, 3>(0, 3);
    return inf;
  }


  /**
   * @brief generate map point cloud and publish it
   * @param event
   */
  void map_points_publish_timer_callback(const ros::WallTimerEvent& event) {
    if(!map_points_pub.getNumSubscribers() || !graph_updated) {
      return;
    }
    std::vector<KeyFrameSnapshot::Ptr> snapshot;
    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution);
    if(!cloud) {
      return;
    }
    cloud->header.frame_id = mapFrame;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*cloud, *cloud_msg);

    map_points_pub.publish(cloud_msg);
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::MarkerArray create_marker_array(const ros::Time& stamp) const {
    visualization_msgs::MarkerArray markers;
    // 0 loop edges, 1 nodes, 2 imu, 3 graph edges, and the search-radius sphere
    // as 4 when it is asked for. The loop marker stays an empty LINE_LIST while
    // loop closure is off, which costs nothing.
    markers.markers.resize(show_sphere_ ? 5 : 4);

    // loop edges
    visualization_msgs::Marker& loop_marker = markers.markers[0];
    loop_marker.header.frame_id = "map";
    loop_marker.header.stamp = stamp;
    loop_marker.action = visualization_msgs::Marker::ADD;
    loop_marker.type = visualization_msgs::Marker::LINE_LIST;
    loop_marker.ns = "loop_edges";
    loop_marker.id = 1;
    loop_marker.pose.orientation.w = 1;
    loop_marker.scale.x = 0.1; loop_marker.scale.y = 0.1; loop_marker.scale.z = 0.1;
    loop_marker.color.r = 0.9; loop_marker.color.g = 0.9; loop_marker.color.b = 0;
    loop_marker.color.a = 1;
    if (enable_loop_closure_) {
    for (auto it = loop_detector->loopIndexContainer.begin(); it != loop_detector->loopIndexContainer.end(); ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      Eigen::Vector3d pos = keyframes[key_cur]->node->estimate().translation();
      p.x = pos.x();
      p.y = pos.y();
      p.z = pos.z();
      loop_marker.points.push_back(p);
      pos = keyframes[key_pre]->node->estimate().translation();
      p.x = pos.x();
      p.y = pos.y();
      p.z = pos.z();
      loop_marker.points.push_back(p);
    }
    }

    // node markers
    visualization_msgs::Marker& traj_marker = markers.markers[1];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.3;

    visualization_msgs::Marker& imu_marker = markers.markers[2];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size());
    traj_marker.colors.resize(keyframes.size());
    for(size_t i = 0; i < keyframes.size(); i++) {
      Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = pos.z();

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 0.0;//1.0 - p;
      traj_marker.colors[i].g = 1.0;//p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();

        std_msgs::ColorRGBA color;
        color.r = 0.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 0.1;

        imu_marker.points.push_back(point);
        imu_marker.colors.push_back(color);
      }
    }

    // edge markers
    visualization_msgs::Marker& edge_marker = markers.markers[3];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);
      if(edge_se3) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
        
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 0.0;//1.0 - p1;
        edge_marker.colors[i * 2].g = 1.0;//p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 0.0;//1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = 1.0;//p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        if(std::abs(v1->id() - v2->id()) > 2) {
          // edge_marker.points[i * 2].z += 0.5;
          // edge_marker.points[i * 2 + 1].z += 0.5;
          edge_marker.colors[i * 2].r = 0.9;
          edge_marker.colors[i * 2].g = 0.9;
          edge_marker.colors[i * 2].b = 0.0;
          edge_marker.colors[i * 2 + 1].r = 0.9;
          edge_marker.colors[i * 2 + 1].g = 0.9;
          edge_marker.colors[i * 2 + 1].b = 0.0;
          edge_marker.colors[i * 2].a = 0.0;
          edge_marker.colors[i * 2 + 1].a += 0.0;
        }
        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }


    }

    if (show_sphere_ && enable_loop_closure_)
    {
      // sphere
      visualization_msgs::Marker& sphere_marker = markers.markers[4];
      sphere_marker.header.frame_id = "map";
      sphere_marker.header.stamp = stamp;
      sphere_marker.ns = "loop_close_radius";
      sphere_marker.id = 3;
      sphere_marker.type = visualization_msgs::Marker::SPHERE;

      if(!keyframes.empty()) {
        Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
        sphere_marker.pose.position.x = pos.x();
        sphere_marker.pose.position.y = pos.y();
        sphere_marker.pose.position.z = pos.z();
      }
      sphere_marker.pose.orientation.w = 1.0;
      sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

      sphere_marker.color.r = 1.0;
      sphere_marker.color.a = 0.3;
    }

    return markers;
  }

  /**
   * @brief dump all data to the current directory
   * @param req
   * @param res
   * @return
   */
  bool dump_service(gorio::DumpGraphRequest& req, gorio::DumpGraphResponse& res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req.destination;

    if(directory.empty()) {
      std::array<char, 64> buffer;
      buffer.fill(0);
      time_t rawtime;
      time(&rawtime);
      const auto timeinfo = localtime(&rawtime);
      strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    }

    if(!boost::filesystem::is_directory(directory)) {
      boost::filesystem::create_directory(directory);
    }

    std::cout << "all data dumped to:" << directory << std::endl;

    graph_slam->save(directory + "/graph.g2o");
    for(size_t i = 0; i < keyframes.size(); i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;

      keyframes[i]->save(sst.str());
    }


    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;

    res.success = true;
    return true;
  }

  /**
   * @brief save map data as pcd
   * @param req
   * @param res
   * @return
   */
  bool save_map_service(gorio::SaveMapRequest& req, gorio::SaveMapResponse& res) {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, req.resolution);
    if(!cloud) {
      res.success = false;
      return true;
    }


    cloud->header.frame_id = mapFrame;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;


    int ret = pcl::io::savePCDFileBinary(req.destination, *cloud);
    res.success = ret == 0;

    return true;
  }


  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "output_aftmapped") {
      // Two files: seconds (evo default) and nanoseconds, matching the
      // ground-truth variants shipped with the sequences.
      write_trajectory(result_prefix + ".txt", 1.0);
      write_trajectory(result_prefix + "_ns.txt", 1e9);
      ROS_INFO_STREAM("Optimized trajectory written to " << result_prefix
                      << ".txt and " << result_prefix << "_ns.txt ("
                      << keyframes.size() << " keyframes)");
    }
    else if (str_msg.data == "time") {
      if (enable_loop_closure_) {
        auto median_of = [](std::vector<double> v) {
          std::sort(v.begin(), v.end());
          return v.at(v.size() / 2);
        };
        if (loop_detector->pf_time.size() > 0)
          cout << "Pre-filtering Matching time cost (median): " << median_of(loop_detector->pf_time) << endl;
        if (loop_detector->sc_time.size() > 0)
          cout << "Scan Context time cost (median): " << median_of(loop_detector->sc_time) << endl;
        if (loop_detector->oc_time.size() > 0)
          cout << "Odometry Check time cost (median): " << median_of(loop_detector->oc_time) << endl;
      }
      if (opt_time.size() > 0) {
        std::sort(opt_time.begin(), opt_time.end());
        double median = opt_time.at(floor((double)opt_time.size() / 2));
        cout << "Optimization time cost (median): " << median << endl;
      }
      if (integration_time.size() > 0) {
        std::sort(integration_time.begin(), integration_time.end());
        double median = integration_time.at(floor((double)integration_time.size() / 2));
        cout << "Integration time cost (median): " << median << endl;
      }
      if (keyframe_time.size() > 0) {
        std::sort(keyframe_time.begin(), keyframe_time.end());
        double median = keyframe_time.at(floor((double)keyframe_time.size() / 2));
        cout << "Keyframe time diff (median): " << median << endl;
      }
      report_stats("GP solve time", gp_solve_time);
      report_stats("GP gyro samples", gp_gyr_samples);
      report_stats("GP ego-vel samples", gp_vel_samples);
      report_stats("GP integration interval", gp_interval);
      report_stats("Cloud intake lag", intake_lag);
      report_stats("Cloud intake backlog", intake_backlog);
      cout << "Odometry kinematic bound deflated " << odometry_bound_hits_
           << " edges" << endl;
      cout << "Odometry cross-check deflated " << odometry_cross_check_hits_
           << " edges" << endl;
      cout << "Odometry yaw gate deflated " << odometry_yaw_gate_hits_
           << " edges" << endl;
      cout << "Odometry covariance: eigenvalue clamp " << odom_cov_clamped_
           << ", fell back to fitness " << odom_cov_fallbacks_ << " times" << endl;
      cout << "Preintegration covariance: eigenvalue clamp fired "
           << preint_cov_clamped_ << " times, non-finite (edge skipped) "
           << preint_cov_nonfinite_ << " times" << endl;
      cout << "Pose graph: nodes=" << graph_slam->graph->vertices().size()
           << " edges=" << graph_slam->graph->edges().size()
           << " keyframes=" << keyframes.size() << endl;
    }
  }


private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  ros::WallTimer optimization_timer;
  ros::WallTimer map_publish_timer;
  ros::WallTimer cloud_handler_timer;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;


  ros::Subscriber imu_odom_sub;
  ros::Subscriber imu_sub;
  ros::Subscriber command_sub;

  ros::Publisher imu_odom_pub;
  ros::Publisher markers_pub;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix4d trans_odom2map; // keyframe->node->estimate() * keyframe->odom.inverse();
  Eigen::Isometry3d trans_aftmapped;  // Odometry from /map to /base_link
  Eigen::Isometry3d trans_aftmapped_incremental;
  ros::Publisher odom2base_pub;
  ros::Publisher aftmapped_odom_pub;
  ros::Publisher aftmapped_odom_incremenral_pub;
  ros::Publisher odom_frame2frame_pub;

  std::string points_topic;
  ros::Publisher map_points_pub;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster map2base_broadcaster; // odom_frame => base_frame

  ros::ServiceServer dump_service_server;
  ros::ServiceServer save_map_service_server; 

  // keyframe queue
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;
  std::deque<geometry_msgs::TwistWithCovarianceStampedConstPtr> twist_queue;
  std::deque<nav_msgs::OdometryConstPtr> imu_odom_queue;
  std::deque<sensor_msgs::Imu::Ptr> imu_queue;
  std::deque<sensor_msgs::PointCloud2::ConstPtr> cloud_queue;
  std::deque<nav_msgs::OdometryConstPtr> odom_queue;
  double thisKeyframeTime;
  double lastKeyframeTime;
  size_t keyframe_index = 0;

  // IMU / Ego Velocity Integration
  bool enable_preintegration;
  double preinteg_window_margin;
  int preinteg_min_gyr_samples;
  int preinteg_min_vel_samples;
  double preinteg_max_vel_gap;
  double preinteg_min_vel_coverage;
  double preinteg_max_interval;
  // Numerical guards, not tuning. Each one sits immediately before an inversion
  // or a linear solve and exists so that a degenerate input cannot produce an
  // enormous weight; none of them is a knob to trade accuracy against.
  //
  // The velocity variance floor enters the GP kernel as K + sz2*I, so it is the
  // Tikhonov regulariser: too small and the solve is singular whenever the state
  // is under-determined.
  static constexpr double preinteg_min_vel_var = 2.5e-3;
  static constexpr double preinteg_min_information_eigenvalue = 1e-12;
  Eigen::Matrix<double, 6, 6> last_preint_cov_ = Eigen::Matrix<double, 6, 6>::Identity();
  bool last_preint_cov_valid_ = false;
  mutable long preint_cov_clamped_ = 0;
  long preint_cov_nonfinite_ = 0;
  Eigen::Matrix<double, 6, 6> accum_odom_cov_ = Eigen::Matrix<double, 6, 6>::Zero();
  int accum_odom_frames_ = 0;
  static constexpr double odometry_min_information_eigenvalue = 1e-12;
  double odometry_max_cov_diagonal;
  double preinteg_vel_bias_std;
  double preinteg_gyr_bias_std;
  double preinteg_gyr_scale_std;
  double imu_time_offset_ = 0.114;
  std::string preinteg_edge_dofs_ = "full";
  bool preinteg_rot_weight_from_odom_ = true;
  bool preinteg_weight_cap_yaw_only_ = true;
  Eigen::Matrix<double, 6, 1> preinteg_cov_inflation_ = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 1> odom_cov_inflation_ = Eigen::Matrix<double, 6, 1>::Ones();
  double preinteg_yaw_cap_ratio_ = 0.1;
  double preinteg_cap_ratio_trans_fwd_ = 1.0;
  double preinteg_cap_ratio_trans_lat_ = 1.0;
  double preinteg_cap_ratio_trans_vert_ = 2.0;
  double preinteg_cap_ratio_roll_ = 2.0;
  double preinteg_cap_ratio_pitch_ = 2.0;
  double odometry_yaw_gate_deg_ = 3.0;
  double odometry_yaw_gate_deflate_ = 0.01;
  long odometry_yaw_gate_hits_ = 0;
  bool odom_trans_weight_from_preint_ = false;
  long trans_weight_capped_ = 0;
  bool preinteg_yaw_weight_from_disagreement_ = false;
  double preinteg_yaw_var_tau_ = 60.0;
  int preinteg_yaw_var_min_samples_ = 30;
  static constexpr double preinteg_yaw_var_floor_ = 1e-8;
  double yaw_disagree_var_ = 0.0;
  long yaw_disagree_n_ = 0, yaw_weight_from_disagreement_ = 0;
  long rot_weight_capped_ = 0;
  bool fix_first_keyframe_ = true;
  mutable long odom_cov_clamped_ = 0;
  mutable long odom_cov_fallbacks_ = 0;
  bool odometry_cross_check_ = true;
  double odometry_cross_check_thresh_ = 0.15;
  double odometry_cross_check_deflate_ = 0.01;
  long odometry_cross_check_hits_ = 0;
  bool odometry_kinematic_bound_ = true;
  double odometry_path_margin_ = 1.5;
  long odometry_bound_hits_ = 0;
  double accum_egovel_path_ = 0.0;
  double last_intake_stamp_ = -1.0;
  std::set<std::pair<int, int>> added_loop_edges_;
  double preinteg_gyr_var;
  double preinteg_vel_var;
  int reported_vel_var_ = 0;
  bool enable_imu_orientation;
  bool use_egovel_preinteg_trans;
  Eigen::Matrix4d initial_pose;
  std::mutex imu_queue_mutex;
  // One mutex per queue: the callback and timer threads both touch them.
  std::mutex cloud_queue_mutex;
  std::mutex twist_queue_mutex;

  std::string result_prefix;

  // Marker coefficients

  // for map cloud generation
  std::atomic_bool graph_updated;
  double map_cloud_resolution;
  std::mutex keyframes_snapshot_mutex;
  std::vector<KeyFrameSnapshot::Ptr> keyframes_snapshot;
  std::unique_ptr<MapCloudGenerator> map_cloud_generator;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  int max_keyframes_per_update;
  //  Used for Loop Closure detection source, 
  //  pushed form keyframe_queue at "flush_keyframe_queue()", 
  //  inserted to "keyframes_" before optimization
  std::deque<KeyFrame::Ptr> new_keyframes;
  //  Previous keyframes_
  std::vector<KeyFrame::Ptr> keyframes;
  std::unordered_map<ros::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;
  g2o::VertexSE3* anchor_node;
  g2o::EdgeSE3* anchor_edge;
  Eigen::Isometry3d odom_now_for_log_ = Eigen::Isometry3d::Identity();

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::unique_ptr<InformationMatrixCalculator> inf_calclator;
  std::unique_ptr<LoopDetector> loop_detector;
  bool enable_loop_closure_ = false;
  bool show_sphere_ = false;

  // Registration Method
  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::KdTreeFLANN<PointT>::Ptr kdtreeHistoryKeyPoses;

  std::vector<double> opt_time;
  std::vector<double> integration_time;
  // GP preintegration instrumentation (Phase 2 measurement harness)
  std::vector<double> gp_solve_time;    // seconds inside VelPreintegration
  std::vector<double> gp_gyr_samples;   // gyro samples handed to the GP
  std::vector<double> gp_vel_samples;   // ego-velocity samples handed to the GP
  std::vector<double> gp_interval;      // thisKeyframeTime - lastKeyframeTime
  std::vector<double> intake_lag;       // bag-time lag of the cloud intake path
  std::vector<double> intake_backlog;   // cloud_queue depth at processing time
  std::vector<double> keyframe_time;

  // gt

};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::RadarGraphSlamNodelet, nodelet::Nodelet)
