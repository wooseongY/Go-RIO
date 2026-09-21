/**
 * @file patchworkpp.hpp
 * @author Seungjae Lee
 * @brief
 * @version 0.1
 * @date 2022-07-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef PATCHWORKPP_H
#define PATCHWORKPP_H

#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <boost/format.hpp>
#include <numeric>
#include <queue>
#include <mutex>
#include <ceres/ceres.h>
#include <patchworkpp/utils.hpp>

#define MARKER_Z_VALUE -2.2
#define UPRIGHT_ENOUGH 0.55
#define FLAT_ENOUGH 0.2
#define TOO_HIGH_ELEVATION 0.0
#define TOO_TILTED 1.0

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 3000

using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

using namespace std;

/*
    @brief PathWork ROS Node.
*/
template <typename PointT>
bool point_z_cmp(PointT a, PointT b) { return a.z < b.z; }

template <typename PointT>
struct RevertCandidate
{
    int concentric_idx;
    int sector_idx;
    double ground_flatness;
    double line_variable;
    Eigen::Vector4f pc_mean;
    pcl::PointCloud<PointT> regionwise_ground;

    RevertCandidate(int _c_idx, int _s_idx, double _flatness, double _line_var, Eigen::Vector4f _pc_mean, pcl::PointCloud<PointT> _ground)
     : concentric_idx(_c_idx), sector_idx(_s_idx), ground_flatness(_flatness), line_variable(_line_var), pc_mean(_pc_mean), regionwise_ground(_ground) {}
};

struct PointWithCovariance {
    Eigen::Vector3d point;
    Eigen::Matrix3d covariance;
};

// Cost function for Ceres optimization
struct PlaneFitCost {
    PlaneFitCost(const Eigen::Vector3d& point, const Eigen::Matrix3d& covariance)
        : point_(point), covariance_(covariance) {}

    template <typename T>
    bool operator()(const T* const plane, T* residual) const {
        Eigen::Matrix<T, 3, 1> normal(plane[0], plane[1], plane[2]);
        T d = plane[3];

        const T norm = normal.norm();
        const Eigen::Matrix<T, 3, 1> n_hat = normal / norm;
        const T d_hat = d / norm;

        Eigen::Matrix<T, 3, 1> point_T(point_.cast<T>());
        Eigen::Matrix<T, 3, 3> covariance_T(covariance_.cast<T>());

        // Ceres squares what is returned here, so the residual is the
        // Mahalanobis distance itself, not its square.
        const T distance = n_hat.dot(point_T) + d_hat;
        const T variance = (n_hat.transpose() * covariance_T * n_hat).trace();
        residual[0] = distance / ceres::sqrt(ceres::fmax(variance, T(1e-12)));
        return true;
    }

    const Eigen::Vector3d point_;
    const Eigen::Matrix3d covariance_;
};

struct Params 
{
    bool verbose;
    bool enable_RNR;
    bool enable_RVPF;
    bool enable_TGR;

    int num_iter;
    int num_lpr;
    int num_min_pts;
    int num_zones;
    int num_rings_of_interest;

    double RNR_ver_angle_thr;
    double RNR_intensity_thr;
    bool preserve_nonground_intensity;
    bool rnr_diag;
    bool erase_all_underground;

    double sensor_height;
    double th_seeds;
    double th_dist;
    double th_seeds_v;
    double th_dist_v;
    double max_range;
    double min_range;
    double uprightness_thr;
    double adaptive_seed_selection_margin;
    double intensity_thr;

    vector<int> num_sectors_each_zone;
    vector<int> num_rings_each_zone;
    
    int max_flatness_storage;
    int max_elevation_storage;

    vector<double> elevation_thr;
    vector<double> flatness_thr;

    
    Params() {
        verbose     = false;
        enable_RNR  = true;
        enable_RVPF = false;
        enable_TGR  = true;

        num_iter = 4;               // Number of iterations for ground plane estimation using PCA.
        num_lpr = 20;               // Maximum number of points to be selected as lowest points representative.
        num_min_pts = 10;           // Minimum number of points to be estimated as ground plane in each patch.
        num_zones = 4;              // Setting of Concentric Zone Model(CZM)
        num_rings_of_interest = 4;  // Number of rings to be checked with elevation and flatness values.

        RNR_ver_angle_thr = -15.0;  // Noise points vertical angle threshold. Downward rays of LiDAR are more likely to generate severe noise points.
        RNR_intensity_thr = 0.1;    // Noise points intensity threshold. The reflected points have relatively small intensity than others.
        preserve_nonground_intensity = false;
        rnr_diag = false;
        erase_all_underground = true;
        
        sensor_height = 0.7; // cp     
        // sensor_height = 1.5;      // loop
        th_seeds = 0.5;             // threshold for lowest point representatives using in initial seeds selection of ground points.
        // th_dist = 0.25;             // threshold for thickenss of ground.
        th_dist = 1.0;
        th_seeds_v = 0.25;          // threshold for lowest point representatives using in initial seeds selection of vertical structural points.
        // th_dist_v = 1.0;            // threshold for thickenss of vertical structure.
        th_dist_v = 2.0;            // threshold for thickenss of vertical structure.
        max_range = 50.0;           // max_range of ground estimation area
        min_range = 1.0;            // min_range of ground estimation area
        // uprightness_thr = 0.707;    // threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
        // adaptive_seed_selection_margin = -1.2; // parameter using in initial seeds selection

        uprightness_thr = 0.5;    // threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
        adaptive_seed_selection_margin = -1.2; // parameter using in initial seeds selection

        // num_sectors_each_zone = {16, 32, 54, 32};   // Setting of Concentric Zone Model(CZM)
        // num_rings_each_zone = {2, 4, 4, 4};         // Setting of Concentric Zone Model(CZM)

        max_flatness_storage = 1000;    // The maximum number of flatness storage
        max_elevation_storage = 1000;   // The maximum number of elevation storage
        elevation_thr = {0, 0, 0, 0};   // threshold of elevation for each ring using in GLE. Those values are updated adaptively.
        flatness_thr = {0, 0, 0, 0};    // threshold of flatness for each ring using in GLE. Those values are updated adaptively.

        num_sectors_each_zone = {3,1, 1, 3};   // Setting of Concentric Zone Model(CZM)
        num_rings_each_zone = {4, 4, 2, 2};         // Setting of Concentric Zone Model(CZM)
    }
};

template <typename PointT>
class PatchWorkpp {

public:
    typedef std::vector<pcl::PointCloud<PointT>> Ring;
    typedef std::vector<Ring> Zone;

    PatchWorkpp() {};

    PatchWorkpp(Params _params) : params_(_params) {
        // Init ROS related
        // ROS_INFO("Inititalizing PatchWork++...");

        verbose_ = params_.verbose;
        sensor_height_ = params_.sensor_height;
        num_iter_ = params_.num_iter;
        num_lpr_ = params_.num_lpr;
        num_min_pts_ = params_.num_min_pts;
        th_seeds_ = params_.th_seeds;
        th_dist_ = params_.th_dist;
        th_seeds_v_ = params_.th_seeds_v;
        th_dist_v_ = params_.th_dist_v;
        max_range_ = params_.max_range;
        min_range_ = params_.min_range;
        uprightness_thr_ = params_.uprightness_thr;
        adaptive_seed_selection_margin_ = params_.adaptive_seed_selection_margin;
        RNR_ver_angle_thr_ = params_.RNR_ver_angle_thr;
        RNR_intensity_thr_ = params_.RNR_intensity_thr;
        preserve_nonground_intensity_ = params_.preserve_nonground_intensity;
        rnr_diag_ = params_.rnr_diag;
        erase_all_underground_ = params_.erase_all_underground;
        max_flatness_storage_ = params_.max_flatness_storage;
        max_elevation_storage_ = params_.max_elevation_storage;
        enable_RNR_ = params_.enable_RNR;
        enable_RVPF_ = params_.enable_RVPF;
        enable_TGR_ = params_.enable_TGR;
        
        // ROS_INFO("Sensor Height: %f", sensor_height_);
        // ROS_INFO("Num of Iteration: %d", num_iter_);
        // ROS_INFO("Num of LPR: %d", num_lpr_);
        // ROS_INFO("Num of min. points: %d", num_min_pts_);
        // ROS_INFO("Seeds Threshold: %f", th_seeds_);
        // ROS_INFO("Distance Threshold: %f", th_dist_);
        // ROS_INFO("Max. range:: %f", max_range_);
        // ROS_INFO("Min. range:: %f", min_range_);
        // ROS_INFO("Normal vector threshold: %f", uprightness_thr_);
        // ROS_INFO("adaptive_seed_selection_margin: %f", adaptive_seed_selection_margin_);
        
        num_zones_ = params_.num_zones;
        num_sectors_each_zone_ = params_.num_sectors_each_zone;
        num_rings_each_zone_ = params_.num_rings_each_zone;
        elevation_thr_ = params_.elevation_thr;
        flatness_thr_ = params_.flatness_thr;
    
        // ROS_INFO("Num. zones: %d", num_zones_);

        if (num_zones_ != 4 || num_sectors_each_zone_.size() != num_rings_each_zone_.size()) {
            throw invalid_argument("Some parameters are wrong! Check the num_zones and num_rings/sectors_each_zone");
        }
        if (elevation_thr_.size() != flatness_thr_.size()) {
            throw invalid_argument("Some parameters are wrong! Check the elevation/flatness_thresholds");
        }

        // cout << (boost::format("Num. sectors: %d, %d, %d, %d") % num_sectors_each_zone_[0] % num_sectors_each_zone_[1] %
        //          num_sectors_each_zone_[2] %
        //          num_sectors_each_zone_[3]).str() << endl;
        // cout << (boost::format("Num. rings: %01d, %01d, %01d, %01d") % num_rings_each_zone_[0] %
        //          num_rings_each_zone_[1] %
        //          num_rings_each_zone_[2] %
        //          num_rings_each_zone_[3]).str() << endl;
        // cout << (boost::format("elevation_thr_: %0.4f, %0.4f, %0.4f, %0.4f ") % elevation_thr_[0] % elevation_thr_[1] %
        //          elevation_thr_[2] %
        //          elevation_thr_[3]).str() << endl;
        // cout << (boost::format("flatness_thr_: %0.4f, %0.4f, %0.4f, %0.4f ") % flatness_thr_[0] % flatness_thr_[1] %
        //          flatness_thr_[2] %
        //          flatness_thr_[3]).str() << endl;
        num_rings_of_interest_ = elevation_thr_.size();

        node_handle_.param("visualize", visualize_, true);


        revert_pc_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        ground_pc_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        regionwise_ground_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        regionwise_nonground_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

        min_range_z2_ = (7 * params_.min_range + params_.max_range) / 8.0;
        min_range_z3_ = (3 * params_.min_range + params_.max_range) / 4.0;
        min_range_z4_ = (params_.min_range + params_.max_range) / 2.0;
        min_ranges_ = {params_.min_range, min_range_z2_, min_range_z3_, min_range_z4_};

        ring_sizes_ = {(min_range_z2_ - params_.min_range) / params_.num_rings_each_zone.at(0),
                      (min_range_z3_ - min_range_z2_) / params_.num_rings_each_zone.at(1),
                      (min_range_z4_ - min_range_z3_) / params_.num_rings_each_zone.at(2),
                      (params_.max_range - min_range_z4_) / params_.num_rings_each_zone.at(3)};
        sector_sizes_ = {2 * M_PI / params_.num_sectors_each_zone.at(0),
                         2 * M_PI / params_.num_sectors_each_zone.at(1),
                         2 * M_PI / params_.num_sectors_each_zone.at(2),
                         2 * M_PI / params_.num_sectors_each_zone.at(3)};

        // cout << "RING SIZES: " << ring_sizes_[0] << " " << ring_sizes_[1] << " " << ring_sizes_[2] << " " << ring_sizes_[3] << endl;
        // cout << "SECTOR SIZES: " << sector_sizes_[0] << " " << sector_sizes_[1] << " " << sector_sizes_[2] << " " << sector_sizes_[3] << endl;

        // cout << "INITIALIZATION COMPLETE" << endl;

        for (int i = 0; i < num_zones_; i++) {
            Zone z;
            initialize_zone(z, num_sectors_each_zone_[i], num_rings_each_zone_[i]);
            ConcentricZoneModel_.push_back(z);
        }
    }

    void estimate_ground(pcl::PointCloud<PointT> cloud_in, Eigen::Vector3d ego_vel,
                         pcl::PointCloud<PointT> &cloud_ground, pcl::PointCloud<PointT> &cloud_nonground, double &time_taken, int id);

    /**
     * @brief the ground plane from the last estimate_ground call
     *
     * Returns nx, ny, nz, d with the normal unit-length and pointing up, so a
     * point p on the plane satisfies n.p + d = 0. estimate_plane_cov already
     * fits this by weighted least squares against per-point radar covariances;
     * it was only ever used internally to drop under-ground multipath, and
     * discarded afterwards.
     */
    Eigen::Vector4d getGroundPlane() const {
      return Eigen::Vector4d(normal_(0), normal_(1), normal_(2), d_);
    }

    /** @brief number of points the last plane was fitted to */
    size_t getGroundPlaneSupport() const { return ground_plane_support_; }

private:

    // Every private member variable is written with the undescore("_") in its end.

    ros::NodeHandle node_handle_;

    Params params_;

    std::recursive_mutex mutex_;

    int id_;

    // Stage profiling, off unless GORIO_PW_PROFILE is set in the environment.
    // Kept out of the launch files because it prints a line per frame and is
    // only wanted when investigating where the segmentation time goes.
    const bool pw_profile_ = (std::getenv("GORIO_PW_PROFILE") != nullptr);

    // Split of estimate_plane_cov()'s cost, to tell apart building the
    // per-point covariances, setting up the Ceres problem, and solving it.
    double pcov_build_ = 0.0, pcov_setup_ = 0.0, pcov_solve_ = 0.0;
    long pcov_calls_ = 0, pcov_iters_ = 0, pcov_points_ = 0;

    int num_iter_;
    int num_lpr_;
    int num_min_pts_;
    int num_zones_;
    int num_rings_of_interest_;

    double sensor_height_;
    double th_seeds_;
    double th_dist_;
    double th_seeds_v_;
    double th_dist_v_;
    double max_range_;
    double min_range_;
    double uprightness_thr_;
    double adaptive_seed_selection_margin_;
    double min_range_z2_; // 12.3625
    double min_range_z3_; // 22.025
    double min_range_z4_; // 41.35
    double RNR_ver_angle_thr_;
    double RNR_intensity_thr_;
    bool preserve_nonground_intensity_ = false;
    bool rnr_diag_ = false;
    bool erase_all_underground_ = true;

    bool verbose_;
    bool enable_RNR_;
    bool enable_RVPF_;
    bool enable_TGR_;

    int max_flatness_storage_, max_elevation_storage_;
    std::vector<double> update_flatness_[4];
    std::vector<double> update_elevation_[4];

    float d_;

    VectorXf normal_;
    size_t ground_plane_support_ = 0;
    MatrixXf pnormal_;
    VectorXf singular_values_;
    Eigen::Matrix3f cov_;
    Eigen::Vector4f pc_mean_;

    // For visualization
    bool visualize_;

    vector<int> num_sectors_each_zone_;
    vector<int> num_rings_each_zone_;

    vector<double> sector_sizes_;
    vector<double> ring_sizes_;
    vector<double> min_ranges_;
    vector<double> elevation_thr_;
    vector<double> flatness_thr_;

    queue<int> noise_idxs_;

    vector<Zone> ConcentricZoneModel_;

    ros::Publisher PlaneViz, pub_revert_pc, pub_reject_pc, pub_normal, pub_noise, pub_vertical;
    pcl::PointCloud<PointT> revert_pc_, reject_pc_, noise_pc_, vertical_pc_;
    pcl::PointCloud<PointT> ground_pc_;

    pcl::PointCloud<pcl::PointXYZINormal> normals_;

    pcl::PointCloud<PointT> regionwise_ground_, regionwise_nonground_;

    void initialize_zone(Zone &z, int num_sectors, int num_rings);

    void flush_patches_in_zone(Zone &patches, int num_sectors, int num_rings);
    void flush_patches(std::vector<Zone> &czm);

    void pc2czm(const pcl::PointCloud<PointT> &src, std::vector<Zone> &czm, pcl::PointCloud<PointT> &cloud_nonground);

    void reflected_noise_removal(pcl::PointCloud<PointT> &cloud, pcl::PointCloud<PointT> &cloud_nonground);

    void temporal_ground_revert(pcl::PointCloud<PointT> &cloud_ground, pcl::PointCloud<PointT> &cloud_nonground,
                                std::vector<double> ring_flatness, std::vector<RevertCandidate<PointT>> candidates,
                                int concentric_idx);

    void calc_mean_stdev(std::vector<double> vec, double &mean, double &stdev);

    void update_elevation_thr();
    void update_flatness_thr();

    double xy2theta(const double &x, const double &y);

    double xy2radius(const double &x, const double &y);

    void estimate_plane(const pcl::PointCloud<PointT> &ground);
    void estimate_plane_cov(const pcl::PointCloud<PointT> &ground);

    void extract_piecewiseground(
            const int zone_idx, const pcl::PointCloud<PointT> &src, Eigen::Vector3d ego_vel,
            pcl::PointCloud<PointT> &dst,
            pcl::PointCloud<PointT> &non_ground_dst);

    void extract_initial_seeds(
            const int zone_idx, const pcl::PointCloud<PointT> &p_sorted,
            pcl::PointCloud<PointT> &init_seeds);

    void extract_initial_seeds(
            const int zone_idx, const pcl::PointCloud<PointT> &p_sorted,
            pcl::PointCloud<PointT> &init_seeds, double th_seed);

    void set_ground_likelihood_estimation_status(
            const int zone_idx, const int ring_idx,
            const int concentric_idx,
            const double z_vec,
            const double z_elevation,
            const double ground_flatness);

};

Eigen::VectorXf normalize(const VectorXf& vec) {
    return vec.normalized();
}

// Function to calculate the cross product of two vectors
Eigen::Vector3f crossProduct(const Eigen::Vector3f& u, const Eigen::Vector3f& v) {
    return u.cross(v);
}

// Function to calculate the rotation matrix using Rodrigues' formula
Eigen::Matrix3f rotationMatrix(const VectorXf& n1, const VectorXf& n2) {
    Eigen::VectorXf n1_norm = normalize(n1);
    Eigen::VectorXf n2_norm = normalize(n2);
    Eigen::VectorXf u = crossProduct(n1_norm, n2_norm);
    u = normalize(u);
    double cosTheta = n1_norm.dot(n2_norm);
    double sinTheta = sqrt(1 - cosTheta * cosTheta);

    // Rodrigues' formula: R = I + sin(theta) * [u]_cross + (1 - cos(theta)) * [u]_cross^2
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f u_cross;
    u_cross <<    0, -u.z(),  u.y(),
               u.z(),     0, -u.x(),
              -u.y(),  u.x(),     0;

    Eigen::Matrix3f u_cross_squared = u_cross * u_cross;

    Eigen::Matrix3f R = I + sinTheta * u_cross + (1 - cosTheta) * u_cross_squared;
    return R;
}

template<typename PointT> inline
void PatchWorkpp<PointT>::initialize_zone(Zone &z, int num_sectors, int num_rings) {
    z.clear();
    pcl::PointCloud<PointT> cloud;
    cloud.reserve(1000);
    Ring ring;
    for (int i = 0; i < num_sectors; i++) {
        ring.emplace_back(cloud);
    }
    for (int j = 0; j < num_rings; j++) {
        z.emplace_back(ring);
    }
}

template<typename PointT> inline
void PatchWorkpp<PointT>::flush_patches_in_zone(Zone &patches, int num_sectors, int num_rings) {
    for (int i = 0; i < num_sectors; i++) {
        for (int j = 0; j < num_rings; j++) {
            if (!patches[j][i].points.empty()) patches[j][i].points.clear();
        }
    }
}

template<typename PointT> inline
void PatchWorkpp<PointT>::flush_patches(vector<Zone> &czm) {
    for (int k = 0; k < num_zones_; k++) {
        for (int i = 0; i < num_rings_each_zone_[k]; i++) {
            for (int j = 0; j < num_sectors_each_zone_[k]; j++) {
                if (!czm[k][i][j].points.empty()) czm[k][i][j].points.clear();
            }
        }
    }

    if( verbose_ ) cout << "Flushed patches" << endl;
}

template<typename PointT> inline
void PatchWorkpp<PointT>::estimate_plane(const pcl::PointCloud<PointT> &ground) {
    
    // Calculate the mean of the pointcloud
    pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);

    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);
    singular_values_ = svd.singularValues();

    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));

    if (normal_(2) < 0) { for(int i=0; i<3; i++) normal_(i) *= -1; }

    // mean ground seeds value
    // std::cout << "Mean: " << pc_mean_.transpose() << std::endl;
    // std::cout << "Normal: " << normal_.transpose() << std::endl;
    
    Eigen::Vector3f seeds_mean = pc_mean_.head<3>();
    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose() * seeds_mean)(0, 0);
}

template<typename PointT> inline
void PatchWorkpp<PointT>::estimate_plane_cov(const pcl::PointCloud<PointT> &ground) {
    // std::cout<<"Estimating plane using covariance matrix"<<std::endl;
    const double t_pcov0 = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    double azimuth_variance_ = 0.5;
    double elevation_variance_ = 1.0;
    double distance_variance_ = 0.86;
    std::vector<PointWithCovariance> points_with_cov;
    for (const auto& pt : ground) {
        Eigen::Vector3d point(pt.x, pt.y, pt.z);
        double dist = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        double s_x = dist * distance_variance_ / 400; // 0.00215
        double s_y = dist * sin(azimuth_variance_ / 180 * M_PI); // 0.00873
        double s_z = dist * sin(elevation_variance_ / 180 * M_PI); // 0.01745
        double elevation = atan2(sqrt(pt.x * pt.x + pt.y * pt.y), pt.z);
        double azimuth = atan2(pt.y, pt.x);
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(elevation, Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(azimuth, Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d R; // Rotation matrix
        R = yawAngle * pitchAngle;
        Eigen::Matrix3d S; // Scaling matix
        S << s_x, 0.0, 0.0,   0.0, s_y, 0.0,   0.0, 0.0, s_z;
        
        Eigen::Matrix3d A = R * S;
        Eigen::Matrix3d cov_r = A * A.transpose();
        points_with_cov.push_back({point, cov_r});
    }

    pcov_build_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count() - t_pcov0;
    pcov_calls_++;

    // Initial plane parameters: (normal_x, normal_y, normal_z, d)
    pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);

    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);
    singular_values_ = svd.singularValues();

    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));

    if (normal_(2) < 0) { for(int i=0; i<3; i++) normal_(i) *= -1; }
    Eigen::Vector3f seeds_mean = pc_mean_.head<3>();
    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose() * seeds_mean)(0, 0);

    double plane[4] = {normal_(0), normal_(1), normal_(2), d_};

    const double t_pcov1 = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    // Build the problem
    ceres::Problem problem;
    for (const auto& point_with_cov : points_with_cov) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PlaneFitCost, 1, 4>(
                new PlaneFitCost(point_with_cov.point, point_with_cov.covariance)),
            nullptr, plane);
    }

    // Configure and run the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 30;           // Set maximum number of iterations
    // options.function_tolerance = 1e-4;          // Set function tolerance
    // options.gradient_tolerance = 1e-4;         // Set gradient tolerance
    // options.parameter_tolerance = 1e-4;         // Set parameter tolerance

    const double t_solve0 = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    pcov_solve_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count() - t_solve0;
    pcov_setup_ += t_solve0 - t_pcov1;
    pcov_iters_ += summary.iterations.size();
    pcov_points_ += points_with_cov.size();
    if(plane[2] < 0) {
        plane[0] *= -1;
        plane[1] *= -1;
        plane[2] *= -1;
        plane[3] *= -1;
    }
    // normalize the normal vector

    double norm = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);

    plane[0] /= norm;
    plane[1] /= norm;
    plane[2] /= norm;
    plane[3] /= norm;

    normal_(0) = plane[0];
    normal_(1) = plane[1];
    normal_(2) = plane[2];
    d_ = plane[3];
    // std::cout<<"Normal: "<<normal_.transpose()<<std::endl;
}


template<typename PointT> inline
void PatchWorkpp<PointT>::extract_initial_seeds(
        const int zone_idx, const pcl::PointCloud<PointT> &p_sorted,
        pcl::PointCloud<PointT> &init_seeds, double th_seed) {
    init_seeds.points.clear();

    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;

    int init_idx = 0;
    if (zone_idx == 0) {
        for (int i = 0; i < p_sorted.points.size(); i++) {
            if (p_sorted.points[i].z < adaptive_seed_selection_margin_ * sensor_height_) {
                ++init_idx;
            } else {
                break;
            }
        }
    }

    // Calculate the mean height value.
    for (int i = init_idx; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0

    if(lpr_height > sensor_height_) return;

    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lpr_height + th_seed) {
            init_seeds.points.push_back(p_sorted.points[i]);
        }
    }
}

template<typename PointT> inline
void PatchWorkpp<PointT>::extract_initial_seeds(
        const int zone_idx, const pcl::PointCloud<PointT> &p_sorted,
        pcl::PointCloud<PointT> &init_seeds) {
    init_seeds.points.clear();

    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;

    int init_idx = 0;
    if (zone_idx == 0) {
        for (int i = 0; i < p_sorted.points.size(); i++) {
            if (p_sorted.points[i].z < adaptive_seed_selection_margin_ * sensor_height_) {
                ++init_idx;
            } else {
                break;
            }
        }
    }

    // Calculate the mean height value.
    for (int i = init_idx; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0

    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lpr_height + th_seeds_) {
            init_seeds.points.push_back(p_sorted.points[i]);
        }
    }
}

template<typename PointT> inline
void PatchWorkpp<PointT>::reflected_noise_removal(pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_nonground)
{
    for (int i=0; i<cloud_in.size(); i++)
    {
        double r = sqrt( cloud_in[i].x*cloud_in[i].x + cloud_in[i].y*cloud_in[i].y );
        double z = cloud_in[i].z;
        double ver_angle_in_deg = atan2(z, r)*180/M_PI;


        if ( ver_angle_in_deg < RNR_ver_angle_thr_ && z < -sensor_height_-0.8 && cloud_in[i].intensity < RNR_intensity_thr_)
        {
            cloud_nonground.push_back(cloud_in[i]);
            noise_pc_.push_back(cloud_in[i]);
            noise_idxs_.push(i);
        }
    }

    // The three conditions are ANDed, so knowing which one is never met matters:
    // the intensity threshold of 0.1 is a LiDAR convention and this is a radar.
    if(rnr_diag_) {
        static long n_frames = 0, n_angle = 0, n_z = 0, n_int = 0, n_all = 0, n_pts = 0;
        static double i_min = 1e18, i_max = -1e18, i_sum = 0.0;
        for(size_t k = 0; k < cloud_in.size(); ++k) {
            const double r = std::sqrt(cloud_in[k].x * cloud_in[k].x + cloud_in[k].y * cloud_in[k].y);
            const double va = std::atan2(cloud_in[k].z, r) * 180 / M_PI;
            const bool a = va < RNR_ver_angle_thr_;
            const bool b = cloud_in[k].z < -sensor_height_ - 0.8;
            const bool c = cloud_in[k].intensity < RNR_intensity_thr_;
            n_angle += a; n_z += b; n_int += c; n_all += (a && b && c); ++n_pts;
            i_min = std::min(i_min, double(cloud_in[k].intensity));
            i_max = std::max(i_max, double(cloud_in[k].intensity));
            i_sum += cloud_in[k].intensity;
        }
        if(++n_frames % 200 == 0) {
            cout << "RNRDIAG frames=" << n_frames << " points=" << n_pts
                 << " angle_ok=" << n_angle << " z_ok=" << n_z << " intensity_ok=" << n_int
                 << " all_three=" << n_all << " intensity min=" << i_min << " max=" << i_max
                 << " mean=" << (i_sum / std::max<long>(n_pts, 1)) << endl;
        }
    }
    if (verbose_) cout << "[ RNR ] Num of noises : " << noise_pc_.points.size() << endl;
}

/*
    @brief Velodyne pointcloud callback function. The main GPF pipeline is here.
    PointCloud SensorMsg -> Pointcloud -> z-value sorted Pointcloud
    ->error points removal -> extract ground seeds -> ground plane fit mainloop
*/

template<typename PointT> inline
void PatchWorkpp<PointT>::estimate_ground(
        pcl::PointCloud<PointT> cloud_in,
        Eigen::Vector3d ego_vel,
        pcl::PointCloud<PointT> &cloud_ground,
        pcl::PointCloud<PointT> &cloud_nonground,
        double &time_taken, int id) {
    
    id_ = id;

    unique_lock<recursive_mutex> lock(mutex_);

    static double start, t0, t1, t2, end;

    double pca_time_ = 0.0;

    // Wall time, not ros::Time. Under use_sim_time ros::Time::now() returns the
    // bag clock, which is not a measure of how long anything took and also
    // takes a lock shared with every other thread in the nodelet manager.
    const auto wall = []() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double w_start = wall();

    start = ros::Time::now().toSec();

    cloud_ground.clear();
    cloud_nonground.clear();

    // 1. Reflected Noise Removal (RNR)
    if (enable_RNR_) reflected_noise_removal(cloud_in, cloud_nonground);

    const double w_rnr = wall();
    t1 = ros::Time::now().toSec();

    // 2. Concentric Zone Model (CZM)
    flush_patches(ConcentricZoneModel_);
    const double w_flush = wall();
    pc2czm(cloud_in, ConcentricZoneModel_, cloud_nonground);
    const double w_czm = wall();

    t2 = ros::Time::now().toSec();

    int concentric_idx = 0;

    std::vector<RevertCandidate<PointT>> candidates;
    std::vector<double> ringwise_flatness;

    for (int zone_idx = 0; zone_idx < num_zones_; ++zone_idx) {

        auto zone = ConcentricZoneModel_[zone_idx];

        for (int ring_idx = 0; ring_idx < num_rings_each_zone_[zone_idx]; ++ring_idx) {
            for (int sector_idx = 0; sector_idx < num_sectors_each_zone_[zone_idx]; ++sector_idx) {

                if (zone[ring_idx][sector_idx].points.size() < num_min_pts_)
                {
                    cloud_nonground += zone[ring_idx][sector_idx];
                    continue;
                }

                // --------- region-wise sorting (faster than global sorting method) ---------------- //

                sort(zone[ring_idx][sector_idx].points.begin(), zone[ring_idx][sector_idx].points.end(), point_z_cmp<PointT>);

                // ---------------------------------------------------------------------------------- //

                extract_piecewiseground(zone_idx, zone[ring_idx][sector_idx], ego_vel, regionwise_ground_, regionwise_nonground_);

                // Status of each patch
                // used in checking uprightness, elevation, and flatness, respectively
                const double ground_uprightness = normal_(2);
                const double ground_elevation   = pc_mean_(2, 0);
                const double ground_flatness    = singular_values_.minCoeff();
                const double line_variable      = singular_values_(1) != 0 ? singular_values_(0)/singular_values_(1) : std::numeric_limits<double>::max();

                // const double velocity_dot_normal = normal_.dot(ego_vel)/ego_vel.norm();


                double heading = 0.0;
                for(int i=0; i<3; i++) heading += pc_mean_(i,0)*normal_(i);


                /*
                    About 'is_heading_outside' condidition, heading should be smaller than 0 theoretically.
                    ( Imagine the geometric relationship between the surface normal vector on the ground plane and
                        the vector connecting the sensor origin and the mean point of the ground plane )

                    However, when the patch is far awaw from the sensor origin,
                    heading could be larger than 0 even if it's ground due to lack of amount of ground plane points.

                    Therefore, we only check this value when concentric_idx < num_rings_of_interest ( near condition )
                */
                bool is_upright         = ground_uprightness > uprightness_thr_;
                bool is_not_elevated    = ground_elevation < elevation_thr_[concentric_idx];
                bool is_flat            = ground_flatness < flatness_thr_[concentric_idx];
                bool is_near_zone       = concentric_idx < num_rings_of_interest_;
                bool is_heading_outside = heading < 0.0;
                // bool is_perpendicular  = velocity_dot_normal < 0.1;


                /*
                    Store the elevation & flatness variables
                    for A-GLE (Adaptive Ground Likelihood Estimation)
                    and TGR (Temporal Ground Revert). More information in the paper Patchwork++.
                */
                if (is_upright && is_not_elevated && is_near_zone)
                {
                    update_elevation_[concentric_idx].push_back(ground_elevation);
                    update_flatness_[concentric_idx].push_back(ground_flatness);

                    ringwise_flatness.push_back(ground_flatness);
                }

                // Ground estimation based on conditions
                if (!is_upright)
                {
                    cloud_nonground += regionwise_ground_;
                }
                else if (!is_near_zone)
                {
                    cloud_ground += regionwise_ground_;
                }
                else if (!is_heading_outside)
                {
                    cloud_nonground += regionwise_ground_;
                }
                else if (is_not_elevated || is_flat)
                {
                    cloud_ground += regionwise_ground_;
                }
                else
                {
                    RevertCandidate<PointT> candidate(concentric_idx, sector_idx, ground_flatness, line_variable, pc_mean_, regionwise_ground_);
                    candidates.push_back(candidate);
                }
                // Every regionwise_nonground is considered nonground.
                cloud_nonground += regionwise_nonground_;

            }


            if (!candidates.empty())
            {
                if (enable_TGR_)
                {
                    temporal_ground_revert(cloud_ground, cloud_nonground, ringwise_flatness, candidates, concentric_idx);
                }
                else
                {
                    for (size_t i=0; i<candidates.size(); i++)
                    {
                        cloud_nonground += candidates[i].regionwise_ground;
                    }
                }

                candidates.clear();
                ringwise_flatness.clear();
            }


            concentric_idx++;
        }
    }

    const double w_loop = wall();
    double t_update = ros::Time::now().toSec();

    update_elevation_thr();
    update_flatness_thr();

    const double w_thr = wall();
    end = ros::Time::now().toSec();
    time_taken = end - start;
    if(id_ == 0)
        estimate_plane(cloud_ground);
    if(id_ == 1)
        estimate_plane_cov(cloud_ground);
    const double w_plane = wall();

    if (pw_profile_) std::cout << "PCOVSPLIT calls=" << pcov_calls_
              << " points=" << pcov_points_
              << " iters=" << pcov_iters_
              << " build=" << pcov_build_
              << " setup=" << pcov_setup_
              << " solve=" << pcov_solve_ << std::endl;
    pcov_build_ = pcov_setup_ = pcov_solve_ = 0.0;
    pcov_calls_ = pcov_iters_ = pcov_points_ = 0;

    if (pw_profile_) std::cout << "PWSTAGE n=" << cloud_in.size()
              << " rnr=" << (w_rnr - w_start)
              << " flush=" << (w_flush - w_rnr)
              << " czm=" << (w_czm - w_flush)
              << " loop=" << (w_loop - w_czm)
              << " thr=" << (w_thr - w_loop)
              << " plane=" << (w_plane - w_thr)
              << " total=" << (w_plane - w_start) << std::endl;
    ground_plane_support_ = cloud_ground.size();
    // std::cout << "cloud_nonground size: " << cloud_nonground.size() << std::endl;
    // std::cout << "cloud ground size: " << cloud_ground.size() << std::endl;
    // std::cout<<"======================================================================================="<<std::endl;

    // std::cout<<"plane eqation: "<< normal_.transpose() << " " << d_ << std::endl;

    for(int i=0; i<cloud_nonground.size(); ++i)
    {
        double x = cloud_nonground[i].x;
        double y = cloud_nonground[i].y;
        double z = cloud_nonground[i].z;
        double dist = normal_[0] * x + normal_[1] * y + normal_[2] * z + d_;
        // Two divergences from the published release live here, and the release
        // is what produced the paper's numbers.
        //
        // Overwriting the intensity discards the radar's return power for every
        // non-ground point. It is also why RNR flags nothing: its threshold is
        // "intensity < 0.1" and every value reaching it is exactly 1.
        if(!preserve_nonground_intensity_) {
            cloud_nonground[i].intensity = 1;
        }

        if(dist < -1.0) // under the  ground : multifact artifacts
        {   
            // if(dist < -1.0){
                // Eigen::Vector3d point = Eigen::Vector3d(x,y,z);
                // Eigen::Vector3d plane_point = Eigen::Vector3d(0,0,-d_/normal_[2]);
                // Eigen::Vector3d vec = point - plane_point;
                // Eigen::Vector3d normal = Eigen::Vector3d(normal_[0], normal_[1], normal_[2]);
                // Eigen::Vector3d reflected_point = point - 2 * vec.dot(normal) * normal;
                // // std::cout<<"dist: "<< dist << std::endl;
                // // std::cout<<"original_point: "<< point.transpose() << std::endl;
                // // std::cout<<"reflected_point: "<< reflected_point.transpose() << std::endl;

                // cloud_nonground[i].x = reflected_point[0];
                // cloud_nonground[i].y = reflected_point[1];
                // cloud_nonground[i].z = reflected_point[2];
                // cloud_nonground[i].curvature = -(ego_vel[0] * reflected_point[0] + ego_vel[1] * reflected_point[1] + ego_vel[2] * reflected_point[2]);
                // std::cout<<cloud_nonground[i].intensity<<std::endl;
            // }
            // else{
                // cloud_nonground[i].intensity = 2;
                
                cloud_nonground.erase(cloud_nonground.begin()+i);
                // Without this decrement the element that shifts into position
                // i is skipped, so only every other point of a consecutive run
                // is erased. The release has no decrement and so removes about
                // half of the under-ground artifacts. Keeping it is a real fix,
                // but it changes the cloud the registration is given.
                if(erase_all_underground_) {
                    i--;
                }

            // }
            

            // double z_cal = (x * ego_vel[0] + y * ego_vel[1]) * (-ego_vel[2] - cloud_ground.points[i].curvature * sqrt(x*x+y*y))
            //                             / (ego_vel[2] * ego_vel[2] - cloud_ground.points[i].curvature * cloud_ground.points[i].curvature * (x*x + y*y));
            
            // double A = ego_vel[2] * ego_vel[2] - cloud_ground.points[i].curvature * cloud_ground.points[i].curvature;
            // double B = (x * ego_vel[0] + y * ego_vel[1]) * cloud_ground.points[i].curvature;
            // double C = (x * ego_vel[0] + y * ego_vel[1]) * (x * ego_vel[0] + y * ego_vel[1]) - cloud_ground.points[i].curvature * cloud_ground.points[i].curvature * (x*x + y*y);

            // double z_cal1 = (-B - sqrt(B*B - A*C)) / A;
            // std::cout << "x: " << x << ", y: " << y << ", z: " << z << std::endl;
            // std::cout << "dist: " << dist << std::endl;
            // std::cout << "z_cal: " << z_cal << std::endl;
            // std::cout << "z_cal1: " << z_cal1 << std::endl;
            // cloud_ground.erase(cloud_ground.begin()+i);
            // i--;

        }
    }
    // if(ego_vel.norm() >= 0.0){
    //     for(int i=0; i < cloud_ground.size(); i++) {
            
    //         double x = cloud_ground.points[i].x;
    //         double y = cloud_ground.points[i].y;
    //         double z = cloud_ground.points[i].z;
    //         cloud_ground.points[i].intensity = 3;
    //         if(z > 0 || z < -sensor_height_-0.5){
    //             cloud_ground.erase(cloud_ground.begin()+i);
    //             i--;
    //         }
                

            // std::cout<<"----"<< std::endl;
            // std::cout <<"ego_vel: "<< ego_vel.transpose() << std::endl;
            
            // double A =  cloud_ground.points[i].curvature * cloud_ground.points[i].curvature - ego_vel[2] * ego_vel[2];
            // double B = -(x * ego_vel[0] + y * ego_vel[1]) * ego_vel[2];
            // double C = -(x * ego_vel[0] + y * ego_vel[1]) * (x * ego_vel[0] + y * ego_vel[1]) + cloud_ground.points[i].curvature * cloud_ground.points[i].curvature * (x*x + y*y);

            // double z_cal = (-B - sqrt(B*B - A*C)) / A;
            // if(B*B - A*C < 0) continue;
            // std::cout<<"det: "<< B*B - A*C << std::endl;

            // point to plane distance
            // double meas = x * normal_[0] + y * normal_[1] + z * normal_[2] + d_;
            // std::cout << "original: " << x << ", "<< y << ", " << z << std::endl;
            // std::cout << "refined : " << x << ", "<< y << ", " << z_cal << std::endl;
            // std::cout << "vd: " << cloud_ground.points[i].curvature << ", est: " << (x*ego_vel[0] + y*ego_vel[1] + z*ego_vel[2])/sqrt(x*x+y*y+z*z) << ", refine: " << (x*ego_vel[0] + y*ego_vel[1] + z_cal*ego_vel[2])/sqrt(x*x+y*y+z_cal*z_cal)<< std::endl;
            // std::cout << "dist: " << dist << std::endl;
            // cloud_ground.points[i].curvature = -(x*ego_vel[0] + y*ego_vel[1] + z_cal*ego_vel[2])/sqrt(x*x+y*y+z_cal*z_cal);
            // std::cout<<"curvature: "<< cloud_ground.points[i].curvature << std::endl;
            // double diff = z_cal - z;                     
            // dst.points[i].x -= (dist - meas) * normal_[0];
            // dst.points[i].y -= (dist - meas) * normal_[1];
            // double weight = ego_vel[0] * normal_[0] + ego_vel[1] * normal_[1] + ego_vel[2] * normal_[2];
            // std::cout << (dist - meas) * weight << std::endl;
            // cloud_ground.points[i].z = z_cal;


    //     }
    // }
    // std::cout<<"plane eqation: "<< normal_.transpose() << " " << d_ << std::endl;

    // cout << "Time taken : " << time_taken << endl;
    // cout << "Time taken to sort: " << t_sort << endl;
    // cout << "Time taken to pca : " << pca_time_ << endl;
    // cout << "Time taken to estimate: " << t_total_estimate << endl;
    // cout << "Time taken to Revert: " <<  t_revert << endl;
    // cout << "Time taken to update : " << end - t_update << endl;

    // if (visualize_)
    // {
    //     sensor_msgs::PointCloud2 cloud_ROS;
    //     pcl::toROSMsg(revert_pc_, cloud_ROS);
    //     cloud_ROS.header.stamp = ros::Time::now();
    //     cloud_ROS.header.frame_id = cloud_in.header.frame_id;
    //     pub_revert_pc.publish(cloud_ROS);

    //     pcl::toROSMsg(reject_pc_, cloud_ROS);
    //     cloud_ROS.header.stamp = ros::Time::now();
    //     cloud_ROS.header.frame_id = cloud_in.header.frame_id;
    //     pub_reject_pc.publish(cloud_ROS);

    //     pcl::toROSMsg(normals_, cloud_ROS);
    //     cloud_ROS.header.stamp = ros::Time::now();
    //     cloud_ROS.header.frame_id = cloud_in.header.frame_id;
    //     pub_normal.publish(cloud_ROS);

    //     pcl::toROSMsg(noise_pc_, cloud_ROS);
    //     cloud_ROS.header.stamp = ros::Time::now();
    //     cloud_ROS.header.frame_id = cloud_in.header.frame_id;
    //     pub_noise.publish(cloud_ROS);

    //     pcl::toROSMsg(vertical_pc_, cloud_ROS);
    //     cloud_ROS.header.stamp = ros::Time::now();
    //     cloud_ROS.header.frame_id = cloud_in.header.frame_id;
    //     pub_vertical.publish(cloud_ROS);
    // }


    revert_pc_.clear();
    reject_pc_.clear();
    normals_.clear();
    noise_pc_.clear();
    vertical_pc_.clear();
}

template<typename PointT> inline
void PatchWorkpp<PointT>::update_elevation_thr(void)
{
    for (int i=0; i<num_rings_of_interest_; i++)
    {
        if (update_elevation_[i].empty()) continue;

        double update_mean = 0.0, update_stdev = 0.0;
        calc_mean_stdev(update_elevation_[i], update_mean, update_stdev);
        if (i==0) {
            elevation_thr_[i] = update_mean + 3*update_stdev;
            sensor_height_ = -update_mean;
        }
        else elevation_thr_[i] = update_mean + 2*update_stdev;

        // if (verbose_) cout << "elevation threshold [" << i << "]: " << elevation_thr_[i] << endl;

        int exceed_num = update_elevation_[i].size() - max_elevation_storage_;
        if (exceed_num > 0) update_elevation_[i].erase(update_elevation_[i].begin(), update_elevation_[i].begin() + exceed_num);
    }

    if (verbose_)
    {
        cout << "sensor height: " << sensor_height_ << endl;
        cout << (boost::format("elevation_thr_  :   %0.4f,  %0.4f,  %0.4f,  %0.4f")
                % elevation_thr_[0] % elevation_thr_[1] % elevation_thr_[2] % elevation_thr_[3]).str() << endl;
    }

    return;
}

template<typename PointT> inline
void PatchWorkpp<PointT>::update_flatness_thr(void)
{
    for (int i=0; i<num_rings_of_interest_; i++)
    {
        if (update_flatness_[i].empty()) break;
        if (update_flatness_[i].size() <= 1) break;

        double update_mean = 0.0, update_stdev = 0.0;
        calc_mean_stdev(update_flatness_[i], update_mean, update_stdev);
        flatness_thr_[i] = update_mean+update_stdev;

        // if (verbose_) { cout << "flatness threshold [" << i << "]: " << flatness_thr_[i] << endl; }

        int exceed_num = update_flatness_[i].size() - max_flatness_storage_;
        if (exceed_num > 0) update_flatness_[i].erase(update_flatness_[i].begin(), update_flatness_[i].begin() + exceed_num);
    }

    if (verbose_)
    {
        cout << (boost::format("flatness_thr_   :   %0.4f,  %0.4f,  %0.4f,  %0.4f")
                % flatness_thr_[0] % flatness_thr_[1] % flatness_thr_[2] % flatness_thr_[3]).str() << endl;
    }

    return;
}

template<typename PointT> inline
void PatchWorkpp<PointT>::temporal_ground_revert(pcl::PointCloud<PointT> &cloud_ground, pcl::PointCloud<PointT> &cloud_nonground,
                                               std::vector<double> ring_flatness, std::vector<RevertCandidate<PointT>> candidates,
                                               int concentric_idx)
{
    if (verbose_) std::cout << "\033[1;34m" << "=========== Temporal Ground Revert (TGR) ===========" << "\033[0m" << endl;

    double mean_flatness = 0.0, stdev_flatness = 0.0;
    calc_mean_stdev(ring_flatness, mean_flatness, stdev_flatness);

    if (verbose_)
    {
        cout << "[" << candidates[0].concentric_idx << ", " << candidates[0].sector_idx << "]"
             << " mean_flatness: " << mean_flatness << ", stdev_flatness: " << stdev_flatness << std::endl;
    }

    for( size_t i=0; i<candidates.size(); i++ )
    {
        RevertCandidate<PointT> candidate = candidates[i];

        // Debug
        if(verbose_)
        {
            cout << "\033[1;33m" << candidate.sector_idx << "th flat_sector_candidate"
                 << " / flatness: " << candidate.ground_flatness
                 << " / line_variable: " << candidate.line_variable
                 << " / ground_num : " << candidate.regionwise_ground.size()
                 << "\033[0m" << endl;
        }

        double mu_flatness = mean_flatness + 1.5*stdev_flatness;
        double prob_flatness = 1/(1+exp( (candidate.ground_flatness-mu_flatness)/(mu_flatness/10) ));

        if (candidate.regionwise_ground.size() > 1500 && candidate.ground_flatness < th_dist_*th_dist_) prob_flatness = 1.0;

        double prob_line = 1.0;
        if (candidate.line_variable > 8.0 )//&& candidate.line_dir > M_PI/4)// candidate.ground_elevation > elevation_thr_[concentric_idx])
        {
            // if (verbose_) cout << "line_dir: " << candidate.line_dir << endl;
            prob_line = 0.0;
        }

        bool revert = prob_line*prob_flatness > 0.5;

        if ( concentric_idx < num_rings_of_interest_ )
        {
            if (revert)
            {
                if (verbose_)
                {
                    cout << "\033[1;32m" << "REVERT TRUE" << "\033[0m" << endl;
                }

                revert_pc_ += candidate.regionwise_ground;
                cloud_ground += candidate.regionwise_ground;
            }
            else
            {
                if (verbose_)
                {
                    cout << "\033[1;31m" << "FINAL REJECT" << "\033[0m" << endl;
                }
                reject_pc_ += candidate.regionwise_ground;
                cloud_nonground += candidate.regionwise_ground;
            }
        }
    }

    if (verbose_) std::cout << "\033[1;34m" << "====================================================" << "\033[0m" << endl;
}

// For adaptive
template<typename PointT> inline
void PatchWorkpp<PointT>::extract_piecewiseground(
        const int zone_idx, const pcl::PointCloud<PointT> &src, Eigen::Vector3d ego_vel,
        pcl::PointCloud<PointT> &dst,
        pcl::PointCloud<PointT> &non_ground_dst) {

    // 0. Initialization
    if (!ground_pc_.empty()) ground_pc_.clear();
    if (!dst.empty()) dst.clear();
    if (!non_ground_dst.empty()) non_ground_dst.clear();

    // 1. Region-wise Vertical Plane Fitting (R-VPF)
    // : removes potential vertical plane under the ground plane
    pcl::PointCloud<PointT> src_wo_verticals;
    src_wo_verticals = src;

    if (enable_RVPF_)
    {
        for (int i = 0; i < num_iter_; i++)
        {
            extract_initial_seeds(zone_idx, src_wo_verticals, ground_pc_, th_seeds_v_);
            if(id_ == 0)
                estimate_plane(ground_pc_); // calculate normal_ and d_
            if(id_ == 1)
                estimate_plane_cov(ground_pc_);

            if (zone_idx == 0 && normal_(2) < uprightness_thr_) // uprightness_thr_ : currently 1/sqrt(2), normal_ : plane normal vector
            {
                pcl::PointCloud<PointT> src_tmp;
                src_tmp = src_wo_verticals;
                src_wo_verticals.clear();

                Eigen::MatrixXf points(src_tmp.points.size(), 3);
                int j = 0;
                for (auto &p:src_tmp.points) {
                    points.row(j++) << p.x, p.y, p.z;
                }
                // ground plane model
                Eigen::VectorXf result = points * normal_;

                for (int r = 0; r < result.rows(); r++) {
                    if (result[r] < th_dist_v_ - d_ && result[r] > -th_dist_v_ - d_) {
                        non_ground_dst.points.push_back(src_tmp[r]);
                        vertical_pc_.points.push_back(src_tmp[r]);
                    } else {
                        src_wo_verticals.points.push_back(src_tmp[r]);
                    }
                }
            }
            else break;
        }
    }

    extract_initial_seeds(zone_idx, src_wo_verticals, ground_pc_);
    if(id_ == 0)
        estimate_plane(ground_pc_);
    if(id_ == 1)
        estimate_plane_cov(ground_pc_);

    // 2. Region-wise Ground Plane Fitting (R-GPF)
    // : fits the ground plane 

    //pointcloud to matrix
    Eigen::MatrixXf points(src_wo_verticals.points.size(), 3);
    int j = 0;
    for (auto &p:src_wo_verticals.points) {
        points.row(j++) << p.x, p.y, p.z;
    }

    for (int i = 0; i < num_iter_; i++) {

        ground_pc_.clear();

        // ground plane model
        Eigen::VectorXf result = points * normal_; // point to plane distance
        
        // threshold filter and refinement with velocity
        for (int r = 0; r < result.rows(); r++) {
            if (i < num_iter_ - 1) {
                if (result[r] < th_dist_ - d_ && src_wo_verticals[r].z < -sensor_height_ + 0.5) {

                    // double x = src_wo_verticals[r].x;
                    // double y = src_wo_verticals[r].y;
                    // double z = src_wo_verticals[r].z;  
                    // double vd = src_wo_verticals[r].curvature;

                    // double h = (x * ego_vel[0] + y * ego_vel[1]) * (-ego_vel[2] - vd * sqrt(x*x+y*y))
                    //                             / (ego_vel[2] - vd * vd * (x*x + y*y));

                    // double vd_tr = -(x * ego_vel[0] + y * ego_vel[1] + z * ego_vel[2])/sqrt(x*x + y*y + z*z);

                    // src_wo_verticals[r].z -= (h-z) * (vd-vd_tr);

                    ground_pc_.points.push_back(src_wo_verticals[r]);
                }
            } else { // Final stage
                if (result[r] < th_dist_ - d_  && src_wo_verticals[r].z < -sensor_height_ + 0.5) {

                    // double x = src_wo_verticals[r].x;
                    // double y = src_wo_verticals[r].y;
                    // double z = src_wo_verticals[r].z;  
                    // double vd = src_wo_verticals[r].curvature;

                    // double h = (x * ego_vel[0] + y * ego_vel[1]) * (-ego_vel[2] - vd * sqrt(x*x+y*y))
                    //                             / (ego_vel[2] - vd * vd * (x*x + y*y));

                    // double vd_tr = -(x * ego_vel[0] + y * ego_vel[1] + z * ego_vel[2])/sqrt(x*x + y*y + z*z);

                    // src_wo_verticals[r].z -= (h-z) * (vd-vd_tr);

                    dst.points.push_back(src_wo_verticals[r]);
                } else {
                    non_ground_dst.points.push_back(src_wo_verticals[r]);
                }
            }
        }
        

        if (i < num_iter_ -1){
            if(id_ == 0)
                estimate_plane(ground_pc_);
            if(id_ == 1)
                estimate_plane_cov(ground_pc_);
        }
        
        else{
            if(id_ == 0)
                estimate_plane(dst);
            if(id_ == 1)
                estimate_plane_cov(dst);
        }
    }
}

template<typename PointT> inline
void PatchWorkpp<PointT>::calc_mean_stdev(std::vector<double> vec, double &mean, double &stdev)
{
    if (vec.size() <= 1) return;

    mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();

    for (int i=0; i<vec.size(); i++) { stdev += (vec.at(i)-mean)*(vec.at(i)-mean); }
    stdev /= vec.size()-1;
    stdev = sqrt(stdev);
}

template<typename PointT> inline
double PatchWorkpp<PointT>::xy2theta(const double &x, const double &y) { // 0 ~ 2 * PI
    // if (y >= 0) {
    //     return atan2(y, x); // 1, 2 quadrant
    // } else {
    //     return 2 * M_PI + atan2(y, x);// 3, 4 quadrant
    // }

    double angle = atan2(y, x);
    return angle > 0 ? angle : 2*M_PI+angle;
}

template<typename PointT> inline
double PatchWorkpp<PointT>::xy2radius(const double &x, const double &y) {
    return sqrt(pow(x, 2) + pow(y, 2));
}

template<typename PointT> inline
void PatchWorkpp<PointT>::pc2czm(const pcl::PointCloud<PointT> &src, std::vector<Zone> &czm, pcl::PointCloud<PointT> &cloud_nonground) {

    for (int i=0; i<src.size(); i++) {
        if ((!noise_idxs_.empty()) &&(i == noise_idxs_.front())) {
            noise_idxs_.pop();
            continue;
        }

        PointT pt = src.points[i];

        double r = xy2radius(pt.x, pt.y);
        if ((r <= max_range_) && (r > min_range_)) {
            double theta = xy2theta(pt.x, pt.y);

            int zone_idx = 0;
            if ( r < min_ranges_[1] ) zone_idx = 0;
            else if ( r < min_ranges_[2] ) zone_idx = 1;
            else if ( r < min_ranges_[3] ) zone_idx = 2;
            else zone_idx = 3;

            int ring_idx = min(static_cast<int>(((r - min_ranges_[zone_idx]) / ring_sizes_[zone_idx])), num_rings_each_zone_[zone_idx] - 1);
            int sector_idx = min(static_cast<int>((theta / sector_sizes_[zone_idx])), num_sectors_each_zone_[zone_idx] - 1);

            czm[zone_idx][ring_idx][sector_idx].points.emplace_back(pt);
        }
        else {
            cloud_nonground.push_back(pt);
        }
    }

    if (verbose_) cout << "[ CZM ] Divides pointcloud into the concentric zone model" << endl;
}

#endif
