#ifndef FAST_GICP_FAST_IGICP_HPP
#define FAST_GICP_FAST_IGICP_HPP

#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>
#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

/**
 * @brief Fast GICP algorithm optimized for multi threading with OpenMP
 */ 
template<typename PointSource, typename PointTarget>
class FastAPDGICP : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<FastAPDGICP<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const FastAPDGICP<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<FastAPDGICP<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const FastAPDGICP<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

public:
  FastAPDGICP();
  virtual ~FastAPDGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);

  void setAzimuthVar(double var);
  void setElevationVar(double var);
  void setDistVar(double var);
  // Range-axis sigma in metres. A radar's range precision is set by its
  // bandwidth and does not grow with distance, unlike the two angular axes.
  // Zero keeps the legacy distance-proportional form.
  void setRangeSigma(double sigma);
  // Weight added when a correspondence's cluster labels agree. Zero keeps the
  // legacy 1/(number of correspondences), which is ~2e-4 and so inert.
  void setClusterWeight(double w);
  // Diagnostic: count how often a correspondence's cluster labels agree, and
  // the label histograms needed for the chance rate.
  void setClusterLabelStats(bool on) { cluster_label_stats_ = on; }
  void clusterLabelStats(long& agree, long& total,
                         std::map<int, long>& src, std::map<int, long>& tgt) const {
    agree = cl_agree_; total = cl_total_; src = cl_src_label_hist_; tgt = cl_tgt_label_hist_;
  }
  void resetClusterLabelStats() {
    cl_agree_ = 0; cl_total_ = 0; cl_src_label_hist_.clear(); cl_tgt_label_hist_.clear();
  }

  // void setLambda(double lambda);
  // void setSourceVelocity(const Eigen::Vector3d& vel);
  // void setTargetVelocity(const Eigen::Vector3d& vel);
  // void setSourceExtrinsic(const Eigen::Vector3d& ext);

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  virtual void setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getSourceCovariances() const {
    return source_covs_;
  }

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getTargetCovariances() const {
    return target_covs_;
  }

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual void update_correspondences(const Eigen::Isometry3d& trans);

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

  template<typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);
  // template <typename PointT>
  

protected:
  int num_threads_;
  int k_correspondences_;

  double lambda_;

  RegularizationMethod regularization_method_;

  std::shared_ptr<pcl::search::KdTree<PointSource>> source_kdtree_;
  std::shared_ptr<pcl::search::KdTree<PointTarget>> target_kdtree_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis_;

  std::vector<int> correspondences_;
  std::vector<float> sq_distances_;

  Eigen::Vector3d source_extrinsic_ = Eigen::Vector3d::Zero();

  double azimuth_variance_ = 0.5;
  double elevation_variance_ = 1.0;
  double distance_variance_ = 0.86;
  double range_sigma_ = 0.0;
  double cluster_weight_ = 0.0;
  bool cluster_label_stats_ = false;
  mutable long cl_agree_ = 0;
  mutable long cl_total_ = 0;
  mutable std::map<int, long> cl_src_label_hist_;
  mutable std::map<int, long> cl_tgt_label_hist_;

  Eigen::Vector3d source_vel_;
  Eigen::Vector3d target_vel_;
};
}  // namespace fast_gicp

#endif