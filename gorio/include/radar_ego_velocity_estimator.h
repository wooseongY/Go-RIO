// This file is part of RIO - Radar Inertial Odometry and Radar ego velocity estimation.
// Copyright (C) 2021  Christopher Doer <christopher.doer@kit.edu>
// (Institute of Control Systems, Karlsruhe Institute of Technology)

// RIO is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// RIO is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with RIO.  If not, see <https://www.gnu.org/licenses/>.

#pragma once
 
#include <sensor_msgs/PointCloud2.h>

#include <rio_utils/data_types.h>
#include <rio_utils/ros_helper.h>

// #include <RadarEgoVelocityEstimatorConfig.h>

namespace rio
{

struct RadarEgoVelocityEstimatorConfig
{
 float min_dist = 0.5;
 float max_dist = 400;
 float min_db = -50.0;
 float elevation_thresh_deg = 22.5;
 float elevation_sigma_deg = 0.0;
 // Doppler noise, in m/s, which sets where the weighting saturates.
 float doppler_sigma = 0.05;
 float azimuth_thresh_deg = 60.0;
 float doppler_velocity_correction_factor = 1;
 
 float thresh_zero_velocity = 0.05; //Below this is recognized as inlier (m/s)
 float allowed_outlier_percentage = 0.30;
 float sigma_zero_velocity_x = 1.0e-03; // default Standard Deviation
 float sigma_zero_velocity_y = 3.2e-03;
 float sigma_zero_velocity_z = 1.0e-02;
 
 float sigma_offset_radar_x = 0;
 float sigma_offset_radar_y = 0;
 float sigma_offset_radar_z = 0;

 float max_sigma_x = 0.25;
 float max_sigma_y = 0.25;
 float max_sigma_z = 0.25;
 float max_r_cond;
 bool use_cholesky_instead_of_bdcsvd = true;

 bool use_ransac = true;
 float outlier_prob = 0.05; // Outlier Probability, to calculate ransac_iter_
 float success_prob = 0.995;
 float N_ransac_points = 5;
 float inlier_thresh = 0.5; // err(j) threshold, 0.1 too small, 1.0 too large
};

struct RadarEgoVelocityEstimatorIndices
{ 
  
  uint x_r          = 0;
  uint y_r          = 1;
  uint z_r          = 2;
  uint snr_db       = 3;
  uint doppler      = 4;
  uint range        = 5;
  uint azimuth      = 6;
  uint elevation    = 7;
  uint normalized_x = 8;
  uint normalized_y = 9;
  uint normalized_z = 10;
};

class RadarEgoVelocityEstimator
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief RadarEgoVelocityEstimator constructor
   */
  RadarEgoVelocityEstimator() {
    setRansacIter();
    }

  // configure() is a template over a dynamic-reconfigure type and is not called
  // anywhere, so the struct's defaults are what the estimator runs with. These
  // two are set directly from the nodelet instead.
  void setElevationSigmaDeg(double s) { config_.elevation_sigma_deg = static_cast<float>(s); }
  void setElevationThreshDeg(double s) { config_.elevation_thresh_deg = static_cast<float>(s); }
  // How many returns the range/power/azimuth/elevation gate has kept, and how
  // many it has seen. Reported by the nodelet so a narrowed band shows up in the
  // run's own log instead of being assumed.
  long gateKept() const { return gate_kept_; }
  long gateTotal() const { return gate_total_; }
  void setDopplerSigma(double s) { config_.doppler_sigma = static_cast<float>(s); }

  /**
   * @brief Reconfigure callback
   * @param config  has to contain RadarEgoVelocityEstimatorConfig
   * @return
   */
  template <class ConfigContainingRadarEgoVelocityEstimatorConfig>
  bool configure(ConfigContainingRadarEgoVelocityEstimatorConfig& config);

  /**
   * @brief Estimates the radar ego velocity based on a single radar scan
   * @param[in] radar_scan_msg       radar scan
   * @param[out] v_r                 estimated radar ego velocity
   * @param[out] sigma_v_r           estimated sigmas of ego velocity
   * @param[out] inlier_radar_scan   inlier point cloud
   * @returns true if estimation successful
   */
  bool estimate(const sensor_msgs::PointCloud2& radar_scan_msg, Vector3& v_r, Vector3& sigma_v_r);
  bool estimate(const sensor_msgs::PointCloud2& radar_scan_msg,
                Vector3& v_r,
                Vector3& sigma_v_r,
                sensor_msgs::PointCloud2& inlier_radar_msg,
                sensor_msgs::PointCloud2& outlier_radar_msg);


  mutable long gate_kept_ = 0;
  mutable long gate_total_ = 0;

private:
  /**
   * @brief Implementation of the ransac based estimation
   * @param[in] radar_data          matrix of parsed radar scan --> see RadarEgoVelocityEstimatorIndices
   * @param[out] v_r                estimated radar ego velocity
   * @param[out] sigma_v_r          estimated sigmas of ego velocity
   * @param[out] inlier_idx_best    idices of inlier
   * @returns true if estimation successful
   */
  bool
  solve3DFullRansac(const Matrix& radar_data, Vector3& v_r, Vector3& sigma_v_r, std::vector<uint>& inlier_idx_best, std::vector<uint>& outlier_idx_best);

  /**
   * @brief Estimates the radar ego velocity using all mesurements provided in radar_data
   * @param[in] radar_data          matrix of parsed radar scan --> see RadarEgoVelocityEstimatorIndices
   * @param[out] v_r                estimated radar ego velocity
   * @param[out] sigma_v_r          estimated sigmas of ego velocity
   * @param estimate_sigma          if true sigma will be estimated as well
   * @returns true if estimation successful
   */
  // radar_data may carry a fifth column of per-row weights; when it does the
  // fit is a weighted least squares. See the note at the weight computation.
  bool solve3DFull(const Matrix& radar_data, Vector3& v_r, Vector3& sigma_v_r, bool estimate_sigma = true);

  /**
   * @brief Helper function which estiamtes the number of RANSAC iterations
   */
  void setRansacIter()
  {
    ransac_iter_ = uint((std::log(1.0 - config_.success_prob)) /
                        std::log(1.0 - std::pow(1.0 - config_.outlier_prob, config_.N_ransac_points)));
    // The formula gives 3 for these probabilities, which is far too few
    // hypotheses to find a clean inlier set in a sparse radar scan. The
    // published version pins it at 50 and this line was lost in the refactor.
    ransac_iter_ = 50;
    ROS_INFO_STREAM(kPrefix << "Number of Ransac iterations: " << ransac_iter_);
  }

  const std::string kPrefix = "[RadarEgoVelocityEstimator]: ";
  const RadarEgoVelocityEstimatorIndices idx_;

  RadarEgoVelocityEstimatorConfig config_;
  uint ransac_iter_ = 0;
};

template <class ConfigContainingRadarEgoVelocityEstimatorConfig>
bool RadarEgoVelocityEstimator::configure(ConfigContainingRadarEgoVelocityEstimatorConfig& config)
{
  config_.min_dist                           = config.min_dist;
  config_.max_dist                           = config.max_dist;
  config_.min_db                             = config.min_db;
  config_.elevation_thresh_deg               = config.elevation_thresh_deg;
  config_.azimuth_thresh_deg                 = config.azimuth_thresh_deg;
  config_.doppler_velocity_correction_factor = config.doppler_velocity_correction_factor;

  config_.thresh_zero_velocity       = config.thresh_zero_velocity;
  config_.allowed_outlier_percentage = config.allowed_outlier_percentage;
  config_.sigma_zero_velocity_x      = config.sigma_zero_velocity_x;
  config_.sigma_zero_velocity_y      = config.sigma_zero_velocity_y;
  config_.sigma_zero_velocity_z      = config.sigma_zero_velocity_z;

  config_.sigma_offset_radar_x = config.sigma_offset_radar_x;
  config_.sigma_offset_radar_y = config.sigma_offset_radar_y;
  config_.sigma_offset_radar_z = config.sigma_offset_radar_z;

  config_.max_sigma_x                    = config.max_sigma_x;
  config_.max_sigma_y                    = config.max_sigma_y;
  config_.max_sigma_z                    = config.max_sigma_z;
  config_.max_r_cond                     = config.max_r_cond;
  config_.use_cholesky_instead_of_bdcsvd = config.use_cholesky_instead_of_bdcsvd;

  config_.use_ransac      = config.use_ransac;
  config_.outlier_prob    = config.outlier_prob;
  config_.success_prob    = config.success_prob;
  config_.N_ransac_points = config.N_ransac_points;
  config_.inlier_thresh   = config.inlier_thresh;

  setRansacIter();

  ROS_INFO_STREAM(kPrefix << "Number of Ransac iterations: " << ransac_iter_);
  return true;
}
}  // namespace rio
