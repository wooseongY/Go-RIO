// SPDX-License-Identifier: BSD-2-Clause

#include <string>
#include <fstream>
#include <functional>
#include <sstream>
#include <numeric>
#include <mutex>
#include <algorithm>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/time.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>

#include <visualization_msgs/Marker.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "radar_ego_velocity_estimator.h"
#include "rio_utils/radar_point_cloud.h"
#include "utility_radar.h"

#include "patchworkpp/patchworkpp.hpp"
#include "dbscan/DBSCAN_kdtree.h"


boost::shared_ptr<PatchWorkpp<PointType>> PatchworkppGroundSeg;

using namespace std;

namespace radar_graph_slam {

class PreprocessingNodeletNTU : public nodelet::Nodelet, public ParamServer {
public: 
  // typedef pcl::PointXYZI PointT;
  typedef pcl::PointXYZINormal PointT;

  PreprocessingNodeletNTU() {}
  virtual ~PreprocessingNodeletNTU() {
    // Stop taking callbacks before the members they touch are destroyed.
    own_queue_.disable();
    if(spinner_) {
      spinner_->stop();
      spinner_.reset();
    }
  }

  virtual void onInit() {
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initializeTransformation();
    initializeParams();

    own_nh_ = nh;
    own_nh_.setCallbackQueue(&own_queue_);
    points_sub = own_nh_.subscribe(pointCloudTopic, 100000000, &PreprocessingNodeletNTU::cloud_callback, this);
    imu_sub = own_nh_.subscribe(imuTopic, 100000000, &PreprocessingNodeletNTU::imu_callback, this);
    command_sub = own_nh_.subscribe("/command", 10, &PreprocessingNodeletNTU::command_callback, this);

    points_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 100000000);
    ours_pub = nh.advertise<sensor_msgs::PointCloud2>("/ours_points", 100000000);
    patchwork_pub = nh.advertise<sensor_msgs::PointCloud2>("/patchwork_points", 100000000);
    ransac_pub = nh.advertise<sensor_msgs::PointCloud2>("/ransac_ground", 100000000);
    segmented_pub = nh.advertise<sensor_msgs::PointCloud2>("/segmented_points", 100000000);
    colored_pub = nh.advertise<sensor_msgs::PointCloud2>("/colored_points", 100000000);
    imu_pub = nh.advertise<sensor_msgs::Imu>("/imu", 100000000);
    gt_pub = nh.advertise<nav_msgs::Odometry>("/aftmapped_to_init", 16);
    pub_4d = nh.advertise<nav_msgs::Odometry>("/pose_4d", 100000000);
  
    std::string topic_twist = private_nh.param<std::string>("topic_twist", "/eagle_data/twist");
    std::string topic_inlier_pc2 = private_nh.param<std::string>("topic_inlier_pc2", "/eagle_data/inlier_pc2");
    std::string topic_outlier_pc2 = private_nh.param<std::string>("topic_outlier_pc2", "/eagle_data/outlier_pc2");
    pub_twist = nh.advertise<geometry_msgs::TwistWithCovarianceStamped>(topic_twist, 100000000);
    pub_stationary = nh.advertise<std_msgs::Bool>("/eagle_data/stationary", 100);
    pub_inlier_pc2 = nh.advertise<sensor_msgs::PointCloud2>(topic_inlier_pc2, 100000000);
    pub_outlier_pc2 = nh.advertise<sensor_msgs::PointCloud2>(topic_outlier_pc2, 100000000);
    pc2_raw_pub = nh.advertise<sensor_msgs::PointCloud2>("/eagle_data/pc2_raw",100000000);
    enable_dynamic_object_removal = private_nh.param<bool>("enable_dynamic_object_removal", false);
    power_threshold = private_nh.param<float>("power_threshold", 0);
    // Matches RadarEgoVelocityEstimatorConfig::thresh_zero_velocity, which is
    // what the estimator itself uses to decide a frame is stationary.
    zero_velocity_thresh = private_nh.param<double>("zero_velocity_thresh", 0.05);
    scan_matching_use_nonground_only_ =
        private_nh.param<bool>("scan_matching_use_nonground_only", false);
    min_nonground_points_ = private_nh.param<int>("min_nonground_points", 150);
    // Below this many ground returns the plane fit is not worth trusting.

    Params patchwork_parameters;
    patchwork_parameters.verbose = false;
    // Both default to the current local behaviour; setting them to the release
    // values reproduces the code the paper's numbers came from.
    patchwork_parameters.preserve_nonground_intensity =
        private_nh.param<bool>("patchwork_preserve_nonground_intensity", false);
    patchwork_parameters.erase_all_underground =
        private_nh.param<bool>("patchwork_erase_all_underground", true);
    patchwork_parameters.rnr_diag = private_nh.param<bool>("patchwork_rnr_diag", false);
    egovel_exclude_ground_ = private_nh.param<bool>("egovel_exclude_ground", false);
    egovel_min_nonground_ = private_nh.param<int>("egovel_min_nonground", 60);
    // Weight each Doppler return by how far the radar's elevation uncertainty
    // can move its residual. Zero keeps the unweighted fit.
    estimator.setElevationSigmaDeg(private_nh.param<double>("egovel_elevation_sigma_deg", 0.0));
    estimator.setDopplerSigma(private_nh.param<double>("egovel_doppler_sigma", 0.05));
    egovel_elevation_thresh_deg_ = private_nh.param<double>("egovel_elevation_thresh_deg", 22.5);
    estimator.setElevationThreshDeg(egovel_elevation_thresh_deg_);
    PatchworkppGroundSeg.reset(new PatchWorkpp<PointType>(patchwork_parameters));

    // Last, so nothing arrives before the members above are constructed.
    spinner_.reset(new ros::AsyncSpinner(1, &own_queue_));
    spinner_->start();
  }

private:
  void initializeTransformation(){
    static const double kComposed[16] = {
       0.998742069436, -0.021545931843,  0.045279799573,  0.296001694594,
       0.020570177936,  0.999548668692,  0.021904589263, -0.129886837458,
      -0.045731279730, -0.020945610263,  0.998733968425, -0.004543126257,
       0.0,             0.0,             0.0,             1.0};
    std::vector<double> m;
    if(!private_nh.getParam("radar_to_body", m) || m.size() != 16) {
      ROS_WARN_STREAM("radar_to_body not in the parameter server; using the built-in value");
      m.assign(kComposed, kComposed + 16);
    }
    Radar_to_livox = cv::Mat(4, 4, CV_64F);
    for(int i = 0; i < 4; ++i)
      for(int j = 0; j < 4; ++j) Radar_to_livox.at<double>(i, j) = m[4 * i + j];
    radar_lever_arm_ = Eigen::Vector3d(Radar_to_livox.at<double>(0, 3),
                                       Radar_to_livox.at<double>(1, 3),
                                       Radar_to_livox.at<double>(2, 3));
    ROS_INFO_STREAM("radar lever arm in body frame [m]: " << radar_lever_arm_.transpose());
  }
  void initializeParams() {
    std::string downsample_method = private_nh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);

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
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
    }

    std::string outlier_removal_method = private_nh.param<std::string>("outlier_removal_method", "STATISTICAL");
    if(outlier_removal_method == "STATISTICAL") {
      int mean_k = private_nh.param<int>("statistical_mean_k", 20);
      double stddev_mul_thresh = private_nh.param<double>("statistical_stddev", 1.0);
      std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

      pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
      sor->setMeanK(mean_k);
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter = sor;
    } else if(outlier_removal_method == "RADIUS") {
      double radius = private_nh.param<double>("radius_radius", 2);
      int min_neighbors = private_nh.param<int>("radius_min_neighbors", 2);
      std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;

      pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
      rad->setRadiusSearch(radius);
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter = rad;
    }
    else {
      std::cout << "outlier_removal: NONE" << std::endl;
    }

    use_distance_filter = private_nh.param<bool>("use_distance_filter", true);
    distance_near_thresh = private_nh.param<double>("distance_near_thresh", 1.0);
    distance_far_thresh = private_nh.param<double>("distance_far_thresh", 100.0);
    z_low_thresh = private_nh.param<double>("z_low_thresh", -5.0);
    z_high_thresh = private_nh.param<double>("z_high_thresh", 20.0);


    std::string file_4d = private_nh.param<std::string>("file_location_4d", "");

    ifstream file_in_4d(file_4d);
    if (!file_in_4d.is_open()) {
        cout << "Can not open this 4d file" << endl;
    }
    else{
      std::vector<std::string> vectorLines;
      std::string line;
      while (getline(file_in_4d, line)) {
          vectorLines.push_back(line);
      }
      
      for (size_t i = 0; i < vectorLines.size(); i++) {
          std::string line_ = vectorLines.at(i);
          double stamp,tx,ty,tz,qx,qy,qz,qw;
          stringstream data(line_);
          data >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
          nav_msgs::Odometry odom_msg_4d;
          odom_msg_4d.header.frame_id = mapFrame;
          odom_msg_4d.child_frame_id = baselinkFrame;
          odom_msg_4d.header.stamp = ros::Time().fromSec(stamp);
          odom_msg_4d.pose.pose.orientation.w = qw;
          odom_msg_4d.pose.pose.orientation.x = qx;
          odom_msg_4d.pose.pose.orientation.y = qy;
          odom_msg_4d.pose.pose.orientation.z = qz;
          odom_msg_4d.pose.pose.position.x = tx;
          odom_msg_4d.pose.pose.position.y = ty;
          odom_msg_4d.pose.pose.position.z = tz;
          std::lock_guard<std::mutex> lock(odom_queue_mutex);
          odom_msg_4ds.push_back(odom_msg_4d);
      }
    }
    file_in_4d.close();
  }

  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    sensor_msgs::Imu imu_data;
    imu_data.header.stamp = imu_msg->header.stamp;
    imu_data.header.seq = imu_msg->header.seq;
    imu_data.header.frame_id = "imu_frame";
    Eigen::Quaterniond q_ahrs(imu_msg->orientation.w,
                              imu_msg->orientation.x,
                              imu_msg->orientation.y,
                              imu_msg->orientation.z);
    Eigen::Quaterniond q_r = 
        Eigen::AngleAxisd( M_PI, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd( M_PI, Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd( 0.00000, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_rr = 
        Eigen::AngleAxisd( 0.00000, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd( 0.00000, Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd( M_PI, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_out =  q_r * q_ahrs * q_rr;
    imu_data.orientation.w = q_out.w();
    imu_data.orientation.x = q_out.x();
    imu_data.orientation.y = q_out.y();
    imu_data.orientation.z = q_out.z();
    imu_data.angular_velocity.x = imu_msg->angular_velocity.x;
    imu_data.angular_velocity.y = -imu_msg->angular_velocity.y;
    imu_data.angular_velocity.z = -imu_msg->angular_velocity.z;
    imu_data.linear_acceleration.x = imu_msg->linear_acceleration.x;
    imu_data.linear_acceleration.y = -imu_msg->linear_acceleration.y;
    imu_data.linear_acceleration.z = -imu_msg->linear_acceleration.z;
    imu_pub.publish(imu_data);

    // Keep the latest body-frame rate for the lever-arm correction below.
    {
      std::lock_guard<std::mutex> lock(omega_mutex_);
      latest_omega_ = Eigen::Vector3d(imu_data.angular_velocity.x,
                                      imu_data.angular_velocity.y,
                                      imu_data.angular_velocity.z);
      have_omega_ = true;
    }

    // imu_queue.push_back(imu_msg);
    double time_now = imu_msg->header.stamp.toSec();
    bool updated = false;
    // std::cout <<odom_msgs.size()<<std::endl;
    if (odom_msgs.size() != 0) {
      while (odom_msgs.front().header.stamp.toSec() + 0.001 < time_now) {
        std::lock_guard<std::mutex> lock(odom_queue_mutex);
        odom_msgs.pop_front();
        updated = true;
        if (odom_msgs.size() == 0)
          break;
      }
    }
    if (updated == true && odom_msgs.size() > 0){
      
      gt_pub.publish(odom_msgs.front());
    }

    bool updated_4d = false;
    if(odom_msg_4ds.size() != 0){
      while (odom_msg_4ds.front().header.stamp.toSec() + 0.001 < time_now) {
        std::lock_guard<std::mutex> lock(odom_queue_mutex);
        odom_msg_4ds.pop_front();
        updated_4d = true;
        if (odom_msg_4ds.size() == 0)
          break;
      }
    }
    if(updated_4d == true && odom_msg_4ds.size() > 0){
      pub_4d.publish(odom_msg_4ds.front());
    }


  }


  inline double hypot(double x, double y, double z){
    return sqrt(x*x+y*y+z*z);
  }

  int frame_num = 0;
  void cloud_callback(const sensor_msgs::PointCloud::ConstPtr&  eagle_msg) { // const pcl::PointCloud<PointT>& src_cloud_r
    
    RadarPointCloudType radarpoint_raw;
    PointT radarpoint_xyzi;
    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_raw( new pcl::PointCloud<RadarPointCloudType> );
    pcl::PointCloud<PointT>::Ptr radarcloud_xyzi( new pcl::PointCloud<PointT> );


    radarcloud_xyzi->header.frame_id = baselinkFrame;
    radarcloud_xyzi->header.seq = eagle_msg->header.seq;
    radarcloud_xyzi->header.stamp = eagle_msg->header.stamp.toSec() * 1e6;
    for(int i = 0; i < eagle_msg->points.size(); i++)
    {
        // cout << i << ":    " <<eagle_msg->points[i].x<<endl;
        if(eagle_msg->channels[2].values[i] > power_threshold) //"Power"
        {
            if (eagle_msg->points[i].x == NAN || eagle_msg->points[i].y == NAN || eagle_msg->points[i].z == NAN) continue;
            if (eagle_msg->points[i].x == INFINITY || eagle_msg->points[i].y == INFINITY || eagle_msg->points[i].z == INFINITY) continue;
            cv::Mat ptMat, dstMat;
            ptMat = (cv::Mat_<double>(4, 1) << eagle_msg->points[i].x, eagle_msg->points[i].y, eagle_msg->points[i].z, 1);    
            // Perform matrix multiplication and save as Mat_ for easy element access : in body frame(aka livox)
            cv::Mat Radar_to_livox_rot;
            Radar_to_livox_rot = Radar_to_livox;
            Radar_to_livox_rot.at<double>(0,3) = 0;
            Radar_to_livox_rot.at<double>(1,3) = 0;
            Radar_to_livox_rot.at<double>(2,3) = 0;
    
            dstMat= Radar_to_livox_rot * ptMat;
            radarpoint_raw.x = dstMat.at<double>(0,0);
            radarpoint_raw.y = dstMat.at<double>(1,0);
            radarpoint_raw.z = dstMat.at<double>(2,0);
            radarpoint_raw.intensity = eagle_msg->channels[2].values[i];
            radarpoint_raw.doppler = eagle_msg->channels[0].values[i];

            radarpoint_xyzi.x = dstMat.at<double>(0,0);
            radarpoint_xyzi.y = dstMat.at<double>(1,0);
            radarpoint_xyzi.z = dstMat.at<double>(2,0);
            radarpoint_xyzi.intensity = eagle_msg->channels[2].values[i];
            radarpoint_xyzi.curvature = eagle_msg->channels[0].values[i];

            radarcloud_raw->points.push_back(radarpoint_raw);
            radarcloud_xyzi->points.push_back(radarpoint_xyzi);
        }
    }
    // std::cout<<"eagle msg size: "<<eagle_msg->points.size()<<" radarcloud_raw size: "<<radarcloud_raw->size()<<" radarcloud_xyzi size: "<<radarcloud_xyzi->size()<<std::endl;

    //********** Publish PointCloud2 Format Raw Cloud **********
    sensor_msgs::PointCloud2 pc2_raw_msg;
    pcl::toROSMsg(*radarcloud_raw, pc2_raw_msg);
    pc2_raw_msg.header.stamp = eagle_msg->header.stamp;
    pc2_raw_msg.header.frame_id = baselinkFrame;
    pc2_raw_pub.publish(pc2_raw_msg);

    sensor_msgs::PointCloud2 pc2_for_egovel = pc2_raw_msg;
    if(egovel_exclude_ground_) {
      pcl::PointCloud<PointT>::Ptr pre(new pcl::PointCloud<PointT>());
      pre->reserve(radarcloud_raw->size());
      for(const auto& q : radarcloud_raw->points) {
        PointT t;
        t.x = q.x; t.y = q.y; t.z = q.z;
        t.intensity = q.intensity;
        t.curvature = q.doppler;
        pre->points.push_back(t);
      }
      pre->width = pre->size();
      pre->height = 1;
      pcl::PointCloud<PointT> g_pre, ng_pre;
      double t_pre = 0.0;
      PatchworkppGroundSeg->estimate_ground(*pre, Eigen::Vector3d::Zero(),
                                            g_pre, ng_pre, t_pre, 1);
      if(ng_pre.size() >= egovel_min_nonground_) {
        pcl::PointCloud<RadarPointCloudType> ng_radar;
        ng_radar.reserve(ng_pre.size());
        for(const auto& q : ng_pre.points) {
          RadarPointCloudType t;
          t.x = q.x; t.y = q.y; t.z = q.z;
          t.intensity = q.intensity;
          t.doppler = q.curvature;
          ng_radar.points.push_back(t);
        }
        ng_radar.width = ng_radar.size();
        ng_radar.height = 1;
        pcl::toROSMsg(ng_radar, pc2_for_egovel);
        pc2_for_egovel.header = pc2_raw_msg.header;
        egovel_ground_excluded_++;
      } else {
        // Too few non-ground returns to fit a velocity; fall back rather than
        // hand the estimator a cloud it cannot solve.
        egovel_ground_fallback_++;
      }
    }
    Eigen::Vector3d v_r, sigma_v_r;
    sensor_msgs::PointCloud2 inlier_radar_msg, outlier_radar_msg;
    // Wall time, not clock(): the RANSAC/LSQ estimate runs multithreaded.
    const ros::WallTime egovel_t0 = ros::WallTime::now();
    const bool egovel_ok =
        estimator.estimate(pc2_for_egovel, v_r, sigma_v_r, inlier_radar_msg, outlier_radar_msg);
    // Print what the gate actually kept, early and on the frame path, so a
    // narrowed elevation band is confirmed from the run's own log within
    // seconds rather than assumed or waited out to the end-of-run report.
    if(++egovel_frames_ == 100) {
      std::cout << "Ego-vel elevation gate: +-" << egovel_elevation_thresh_deg_
                << " deg, kept " << estimator.gateKept() << " of " << estimator.gateTotal()
                << " returns (" << (estimator.gateTotal() > 0
                       ? 100.0 * estimator.gateKept() / estimator.gateTotal() : 0.0)
                << "%) over 100 frames" << std::endl;
    }
    if (egovel_ok)
    {
      const bool stationary = (v_r.norm() < zero_velocity_thresh);
      if(stationary) {
        ROS_INFO_STREAM_THROTTLE(5.0, "zero velocity detected, publishing ZUPT "
                                 "(sigma " << sigma_v_r.transpose() << ")");
        zupt_frames++;
      }

      {
        Eigen::Vector3d omega = Eigen::Vector3d::Zero();
        bool have = false;
        {
          std::lock_guard<std::mutex> lock(omega_mutex_);
          omega = latest_omega_;
          have = have_omega_;
        }
        if(have) {
          const Eigen::Vector3d corr = omega.cross(radar_lever_arm_);
          v_r -= corr;
          lever_arm_applied_++;
          lever_arm_corr_sum_ += corr.norm();
          // Proof in the log that the correction reaches the velocity, and by how
          // much. Without this the only evidence is the parameter readback, and a
          // parameter arriving is not the same as a value being consumed.
          if(!reported_lever_arm_) {
            reported_lever_arm_ = true;
            ROS_INFO_STREAM("lever arm: first frame |omega x r| = " << corr.norm()
                            << " m/s at |omega| = " << omega.norm() << " rad/s");
          }
        }
      }

      egovel_time.push_back((ros::WallTime::now() - egovel_t0).toSec());
      // sigma_v_r is the estimator's own per-frame standard deviation of the ego
      // velocity, and it is what the GP preintegration uses as vel_var.
      egovel_sigma.push_back(sigma_v_r.norm());
      
      geometry_msgs::TwistWithCovarianceStamped twist;
      twist.header.stamp         = pc2_raw_msg.header.stamp;
      twist.twist.twist.linear.x = v_r.x();
      twist.twist.twist.linear.y = v_r.y();
      twist.twist.twist.linear.z = v_r.z();

      twist.twist.covariance.at(0)  = std::pow(sigma_v_r.x(), 2);
      twist.twist.covariance.at(7)  = std::pow(sigma_v_r.y(), 2);
      twist.twist.covariance.at(14) = std::pow(sigma_v_r.z(), 2);

      pub_twist.publish(twist);
      pub_inlier_pc2.publish(inlier_radar_msg);
      pub_outlier_pc2.publish(outlier_radar_msg);

      // Stationarity as an explicit signal, so consumers do not have to
      // re-derive it from the velocity magnitude and its threshold.
      std_msgs::Bool stationary_msg;
      stationary_msg.data = stationary;
      pub_stationary.publish(stationary_msg);
    }
    else{;}
    // std::cout << "ego vel : " << v_r.transpose() << std::endl;
    // std::cout << "sigma_v_r: "<<sigma_v_r.transpose()<<std::endl;

    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_inlier( new pcl::PointCloud<RadarPointCloudType> );
    pcl::fromROSMsg (inlier_radar_msg, *radarcloud_inlier);
    

    pcl::PointCloud<PointT>::Ptr radarcloud_inlier_xyzi(new pcl::PointCloud<PointT>());
    radarcloud_inlier_xyzi->header.frame_id = baselinkFrame;
    radarcloud_inlier_xyzi->header.seq = eagle_msg->header.seq;
    radarcloud_inlier_xyzi->header.stamp = eagle_msg->header.stamp.toSec() * 1e6;

    pcl::PointCloud<PointT>::ConstPtr src_cloud;
    
    if (enable_dynamic_object_removal){
      for(int i = 0; i < radarcloud_inlier->size(); i++){
        PointT tmp;
        tmp.x = radarcloud_inlier->points[i].x;
        tmp.y = radarcloud_inlier->points[i].y;
        tmp.z = radarcloud_inlier->points[i].z;
        tmp.intensity = radarcloud_inlier->points[i].intensity;
        tmp.curvature = radarcloud_inlier->points[i].doppler;
        radarcloud_inlier_xyzi->points.push_back(tmp);
      }
      src_cloud = radarcloud_inlier_xyzi;
    }
    else{
      src_cloud = radarcloud_xyzi;
    }
      
    if(src_cloud->empty()) {
      return;
    }


    // if baselinkFrame is defined, transform the input cloud to the frame
    if(!baselinkFrame.empty()) {
      if(!tf_listener.canTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0))) {
        std::cerr << "failed to find transform between " << baselinkFrame << " and " << src_cloud->header.frame_id << std::endl;
      }

      tf::StampedTransform transform;
      tf_listener.waitForTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0), ros::Duration(2.0));
      tf_listener.lookupTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0), transform);

      pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
      pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);
      transformed->header.frame_id = baselinkFrame;
      transformed->header.stamp = src_cloud->header.stamp;
      src_cloud = transformed;
    }
    const ros::WallTime cb_t0 = ros::WallTime::now();

    const ros::WallTime filt_t0 = ros::WallTime::now();
    pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);
    filtered = outlier_removal(filtered);
    filter_time.push_back((ros::WallTime::now() - filt_t0).toSec());
    points_in.push_back(static_cast<double>(src_cloud->size()));
    points_filtered.push_back(static_cast<double>(filtered->size()));

    // ground segmentation
    pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr non_ground(new pcl::PointCloud<PointT>());
    double groundseg_time = 0.0;

    pcl::PointCloud<PointT>::Ptr ground_cloud_1(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr non_ground_1(new pcl::PointCloud<PointT>());

    // Wall time, not clock(): patchwork++ uses OpenMP internally, so clock()
    // sums CPU time across threads and inflates the reported cost.
    //
    // Two timings are recorded because they measure different spans.
    // patchwork++ sets time_taken itself, but it stops its clock before
    // estimate_plane_cov and before the loop that erases under-ground multipath
    // artifacts from cloud_nonground -- both of which are inside the call. So
    // ground_time (the whole call) minus groundseg_time (patchwork's own span)
    // isolates that uncounted tail, which matters because the erase loop uses
    // vector::erase per element and is therefore quadratic in the number
    // removed.
    const ros::WallTime ground_t0 = ros::WallTime::now();
    PatchworkppGroundSeg->estimate_ground(*filtered, v_r, *ground_cloud_1, *non_ground_1, groundseg_time, 1);
    const double ground_total = (ros::WallTime::now() - ground_t0).toSec();
    ground_time.push_back(ground_total);
    ground_internal_time.push_back(groundseg_time);
    ground_tail_time.push_back(ground_total - groundseg_time);
    
    ground_points.push_back(static_cast<double>(ground_cloud_1->size()));
    nonground_points.push_back(static_cast<double>(non_ground_1->size()));

    pcl::PointCloud<PointT>::Ptr full_scan(new pcl::PointCloud<PointT>());
    if(scan_matching_use_nonground_only_) {
      *full_scan = *non_ground_1;
      if(full_scan->size() < static_cast<size_t>(min_nonground_points_)) {
        // Too little structure left to register against; fall back rather than
        // hand the front end a nearly empty scan.
        *full_scan = *ground_cloud_1 + *non_ground_1;
        nonground_fallbacks_++;
      }
    } else {
      *full_scan = *ground_cloud_1 + *non_ground_1;
    }

    frame_num++;
  

    const ros::WallTime clus_t0 = ros::WallTime::now();
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZINormal>());
    kdtree->setInputCloud(full_scan);

    std::vector<pcl::PointIndices> cluster_indices;
    DBSCANKdtreeCluster<PointT> ec;

    ec.setCorePointMinPts(10);
    ec.setClusterTolerance(0.9);
    ec.setMinClusterSize(20);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(kdtree);
    ec.setInputCloud(full_scan);
    ec.extract(cluster_indices);
    cluster_time.push_back((ros::WallTime::now() - clus_t0).toSec());
    cluster_count.push_back(static_cast<double>(cluster_indices.size()));


    std::vector<std::pair<int, float>> cluster_distances;
    cluster_distances.reserve(cluster_indices.size());  // Preallocate memory

    for (size_t i = 0; i < cluster_indices.size(); ++i)
    {
        float sum_x = 0, sum_y = 0, sum_z = 0;
        int num_points = cluster_indices[i].indices.size();

        for (int idx : cluster_indices[i].indices)
        {
          const PointT &point = full_scan->points[idx];  // Use const ref for faster access
          sum_x += point.x;
          sum_y += point.y;
          sum_z += point.z;
        }

        float centroid_x = sum_x / num_points;
        float centroid_y = sum_y / num_points;
        float centroid_z = sum_z / num_points;

        float distance = std::hypot(centroid_x, centroid_y, centroid_z);  // More efficient than sqrt()
        cluster_distances.emplace_back(i, distance);
    }

    std::sort(cluster_distances.begin(), cluster_distances.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    for (size_t rank = 0; rank < cluster_distances.size(); ++rank)
    {
      int cluster_id = cluster_distances[rank].first;
      for (int idx : cluster_indices[cluster_id].indices)
      {
        full_scan->points[idx].normal_x = static_cast<float>(rank + 1);
      }
    }
    
    full_scan->width = full_scan->size();
    full_scan->height = 1;
    full_scan->is_dense = true;

    sensor_msgs::PointCloud2 filtered_msg;
    pcl::toROSMsg(*full_scan, filtered_msg);
    filtered_msg.header.stamp = eagle_msg->header.stamp;
    filtered_msg.header.frame_id = baselinkFrame;

    points_pub.publish(filtered_msg);

    callback_time.push_back((ros::WallTime::now() - cb_t0).toSec());
  }


  pcl::PointCloud<PointT>::ConstPtr passthrough(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    PointT pt;
    for(int i = 0; i < cloud->size(); i++){
      if (cloud->at(i).z < 10 && cloud->at(i).z > -2){
        pt.x = (*cloud)[i].x;
        pt.y = (*cloud)[i].y;
        pt.z = (*cloud)[i].z;
        pt.intensity = (*cloud)[i].intensity;
        filtered->points.push_back(pt);
      }
    }
    filtered->header = cloud->header;
    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr removeNAN(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
      // Remove NaN/Inf points
      pcl::PointCloud<PointT>::Ptr cloudout(new pcl::PointCloud<PointT>());
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cloud, *cloudout, indices);
      
      return cloudout;
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      // Remove NaN/Inf points
      pcl::PointCloud<PointT>::Ptr cloudout(new pcl::PointCloud<PointT>());
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cloud, *cloudout, indices);
      
      return cloudout;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

    filtered->reserve(cloud->size());
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      double d = p.getVector3fMap().norm();
      double z = p.z;
      return d > distance_near_thresh && d < distance_far_thresh && z < z_high_thresh && z > z_low_thresh;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    filtered->header = cloud->header;

    return filtered;
  }


  /**
   * @brief print median / mean / p95 / max of a collected metric
   *
   * Takes a copy so repeated dumps stay valid, and reports the tail: the median
   * alone hides the frames that actually blow the budget.
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

  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "time") {
      report_stats("Preproc callback total", callback_time);
      report_stats("Preproc filter+outlier", filter_time);
      report_stats("Preproc ground seg", ground_time);
      report_stats("Preproc ground seg internal", ground_internal_time);
      report_stats("Preproc ground seg tail", ground_tail_time);
      report_stats("Preproc kdtree+DBSCAN", cluster_time);
      report_stats("Preproc ego-vel", egovel_time);
      report_stats("Preproc ego-vel sigma", egovel_sigma);
      report_stats("Preproc points in", points_in);
      report_stats("Preproc points filtered", points_filtered);
      report_stats("Preproc clusters", cluster_count);
      std::cout << "Ego-vel elevation gate: +-" << egovel_elevation_thresh_deg_
                << " deg, kept " << estimator.gateKept() << " of " << estimator.gateTotal()
                << " returns (" << (estimator.gateTotal() > 0
                       ? 100.0 * estimator.gateKept() / estimator.gateTotal() : 0.0)
                << "%)" << std::endl;
      if(egovel_exclude_ground_) {
        std::cout << "Ego velocity from non-ground returns on " << egovel_ground_excluded_
                  << " frames, fell back to all returns on " << egovel_ground_fallback_
                  << std::endl;
      }
      report_stats("Preproc ground points", ground_points);
      report_stats("Preproc non-ground points", nonground_points);
      cout << "Lever arm correction applied on " << lever_arm_applied_
           << " frames, mean |omega x r| = "
           << (lever_arm_applied_ > 0 ? lever_arm_corr_sum_ / lever_arm_applied_ : 0.0)
           << " m/s" << endl;
      cout << "Scan matching input: "
           << (scan_matching_use_nonground_only_ ? "non-ground only" : "ground + non-ground")
           << " (fell back " << nonground_fallbacks_ << " times)" << endl;
      cout << "ZUPT frames (published, previously dropped): " << zupt_frames << endl;
      if(egovel_time.size()>0){
        std::sort(egovel_time.begin(), egovel_time.end());
        double median = egovel_time.at(size_t(egovel_time.size() / 2));
        cout << "Ego velocity time cost (median): " << median << endl;
      }

      if(ground_time.size()>0){
        std::sort(ground_time.begin(), ground_time.end());
        double ground_median = ground_time.at(size_t(ground_time.size() / 2));
        cout << "Ground Segmentation time cost (median): " << ground_median << endl;
      }
    }
    else if (str_msg.data == "point_distribution") {
      Eigen::VectorXi data(100);
      for (size_t i = 0; i < num_at_dist_vec.size(); i++){ // N
        Eigen::VectorXi& nad = num_at_dist_vec.at(i);
        for (int j = 0; j< 100; j++){
          data(j) += nad(j);
        }
      }
      data /= num_at_dist_vec.size();
      for (int i=0; i<data.size(); i++){
        cout << data(i) << ", ";
      }
      cout << endl;
    }
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle own_nh_;
  ros::CallbackQueue own_queue_;
  boost::shared_ptr<ros::AsyncSpinner> spinner_;
  ros::NodeHandle private_nh;

  ros::Subscriber imu_sub;
  std::vector<sensor_msgs::ImuConstPtr> imu_queue;
  ros::Subscriber points_sub;

  ros::Publisher points_pub;
  ros::Publisher patchwork_pub;
  ros::Publisher ours_pub;
  ros::Publisher ransac_pub;
  ros::Publisher segmented_pub;
  ros::Publisher plane_pub;
  ros::Publisher colored_pub;
  ros::Publisher imu_pub;
  ros::Publisher gt_pub;
  ros::Publisher pub_4d;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;


  bool use_distance_filter;
  double distance_near_thresh;
  double distance_far_thresh;
  double z_low_thresh;
  double z_high_thresh;

  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Filter<PointT>::Ptr outlier_removal_filter;

  cv::Mat Radar_to_livox; // Transform Radar point cloud to LiDAR Frame
  rio::RadarEgoVelocityEstimator estimator;
  ros::Publisher pub_twist, pub_inlier_pc2, pub_outlier_pc2, pc2_raw_pub;

  float power_threshold;
  bool enable_dynamic_object_removal = false;

  std::mutex odom_queue_mutex;
  std::deque<nav_msgs::Odometry> odom_msgs;
  std::deque<nav_msgs::Odometry> odom_msg_4ds;

  ros::Subscriber command_sub;
  std::vector<double> egovel_time;
  std::vector<double> egovel_sigma;  // per-frame ego-velocity sigma (for Phase 5a)
  std::vector<double> ground_time;
  std::vector<double> callback_time;    // whole cloud_callback, post ego-velocity
  std::vector<double> filter_time;      // distance filter + radius outlier removal
  std::vector<double> cluster_time;     // kd-tree build + DBSCAN extract
  std::vector<double> points_in;        // points entering the filter chain
  std::vector<double> points_filtered;  // points surviving it
  std::vector<double> cluster_count;    // clusters found
  ros::Publisher pub_stationary;
  double zero_velocity_thresh;
  bool reported_lever_arm_ = false;
  long lever_arm_applied_ = 0;
  double lever_arm_corr_sum_ = 0.0;
  bool egovel_exclude_ground_ = false;
  int egovel_min_nonground_ = 60;
  long egovel_ground_excluded_ = 0;
  double egovel_elevation_thresh_deg_ = 22.5;
  long egovel_frames_ = 0;
  long egovel_ground_fallback_ = 0;
  Eigen::Vector3d radar_lever_arm_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d latest_omega_ = Eigen::Vector3d::Zero();
  bool have_omega_ = false;
  std::mutex omega_mutex_;
  long zupt_frames = 0;
  std::vector<double> ground_internal_time;  // patchwork++'s own time_taken
  std::vector<double> ground_tail_time;      // estimate_plane_cov + erase loop
  std::vector<double> ground_points;         // returns classified as ground
  std::vector<double> nonground_points;      // returns classified as non-ground
  bool scan_matching_use_nonground_only_ = false;
  int min_nonground_points_ = 150;
  long nonground_fallbacks_ = 0;

  std::vector<Eigen::VectorXi> num_at_dist_vec;
};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::PreprocessingNodeletNTU, nodelet::Nodelet)
