#ifndef FAST_GICP_FAST_IGICP_IMPL_HPP
#define FAST_GICP_FAST_IGICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>
#include <pcl/features/normal_3d.h>

using namespace std;

#define debug_out false

namespace fast_gicp {

template <typename PointSource, typename PointTarget>
FastAPDGICP<PointSource, PointTarget>::FastAPDGICP() {
#ifdef _OPENMP 
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;
  reg_name_ = "FastAPDGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = RegularizationMethod::PLANE;
  source_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_.reset(new pcl::search::KdTree<PointTarget>);
}

template <typename PointSource, typename PointTarget>
FastAPDGICP<PointSource, PointTarget>::~FastAPDGICP() {}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setAzimuthVar(double var) {
  azimuth_variance_ = var;
}
template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setElevationVar(double var) {
  elevation_variance_ = var;
}
template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setDistVar(double var) {
  distance_variance_ = var;
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setRangeSigma(double sigma) {
  range_sigma_ = sigma;
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setClusterWeight(double w) {
  cluster_weight_ = w;
}


// template <typename PointSource, typename PointTarget>
// void FastAPDGICP<PointSource, PointTarget>::setLambda(double lambda) {
//   lambda_ = lambda;
// }
// template <typename PointSource, typename PointTarget>
// void FastAPDGICP<PointSource, PointTarget>::setSourceVelocity(const Eigen::Vector3d& vel) {
//   source_vel_ = vel;
// }
// template <typename PointSource, typename PointTarget>
// void FastAPDGICP<PointSource, PointTarget>::setTargetVelocity(const Eigen::Vector3d& vel) {
//   target_vel_ = vel;
// }

// template <typename PointSource, typename PointTarget>
// void FastAPDGICP<PointSource, PointTarget>::setSourceExtrinsic(const Eigen::Vector3d& ext) {
//   source_extrinsic_ = ext;
// }



template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree_.swap(target_kdtree_);
  source_covs_.swap(target_covs_);

  source_vel_.swap(target_vel_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
  source_vel_.setZero();
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
  target_vel_.setZero();
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  source_kdtree_->setInputCloud(cloud);
  source_covs_.clear();
  source_vel_.setZero();
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  target_kdtree_->setInputCloud(cloud);
  target_covs_.clear();
  target_vel_.setZero();
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (source_covs_.size() != input_->size()) {
    calculate_covariances(input_, *source_kdtree_, source_covs_);
  }
  if (target_covs_.size() != target_->size()) {
    calculate_covariances(target_, *target_kdtree_, target_covs_);
  }

  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void FastAPDGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    target_kdtree_->nearestKSearch(pt, 1, k_indices, k_sq_dists);

    sq_distances_[i] = k_sq_dists[0];
    /*****  Algorithum 1, line 5-9 In GICP paper  *****/
    // index of nearest point in target cloud
    correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;
 
    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];
    
    // Distance between the sensor origin and the point
    double dist = pt.getVector3fMap().template cast<double>().norm();
    // The two angular axes scale with range because a fixed angular error
    // subtends a wider arc further out. The range axis does not: an FMCW
    // radar's range precision comes from its bandwidth and is constant, about
    // 0.02 m here. The legacy form made it 0.00215 * dist -- roughly right at
    // 10 m, ten times too large at 100 m -- so distant points were discounted
    // along exactly the axis that observes forward motion.
    double s_x = range_sigma_ > 0.0 ? range_sigma_ : dist * distance_variance_ / 400; // 0.00215
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
    Eigen::Matrix4d cov_dist = Eigen::Matrix4d::Zero();
    cov_dist.block<3, 3>(0, 0) = cov_r;
    
    /*****  Equ. (2) in paper --- CiB + T*CiA*T(-1) *****/
    Eigen::Matrix4d RCR;
    RCR = (cov_B + cov_dist) + trans.matrix() * (cov_A + cov_dist) * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    mahalanobis_[i] = RCR.inverse(); // (CiB + T*CiA*T^(-1))^(−1)
    mahalanobis_[i](3, 3) = 0.0f;
  }
}

// Calculate H matrix and b matrix
template <typename PointSource, typename PointTarget>
double FastAPDGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {

  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }
#if debug_out
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hint(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bint(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hgeo(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bgeo(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hint[i].setZero();
    bint[i].setZero();
    Hgeo[i].setZero();
    bgeo[i].setZero();
  }
#endif
#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    /*****  ai ∼ N(a^i; CiA) and bi ∼ N(b^i; CiB)  *****/
    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    /*****  Eq. (2) in paper --- di(T)  *****/
    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A; // di(T)

    // ws
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_A.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d values;
    values = svd.singularValues()/svd.singularValues().maxCoeff();
    double geo_weight = values(2);

    double cl_weight = 0.0;
    const bool labels_agree = target_->at(target_index).normal_x == input_->at(i).normal_x;
    if(labels_agree)
      cl_weight = cluster_weight_ > 0.0 ? cluster_weight_ : 1.0/correspondences_.size();
    
    // sum_errors += error.transpose() * mahalanobis_[i] * error;
    sum_errors += (1.0 + geo_weight + cl_weight) * error.transpose() * mahalanobis_[i] * error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    /*****  chapter 4.3.5 in the book (page 85), Derivative of lie algebra  *****/
    // The derivative of di(T) = bi − T*ai, id est -ai
    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>()); // vector to Inverse Symetric Matrix
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0; // Jacobian matrix of loss in lie algebra form ?
    if(cluster_label_stats_) {
      // The label is a rank by centroid distance, so it only corresponds across
      // frames while the cluster count and ordering hold. Compared against the
      // chance rate below, this says whether the mechanism has any signal.
      cl_agree_ += labels_agree ? 1 : 0;
      cl_total_ += 1;
      cl_src_label_hist_[static_cast<int>(input_->at(i).normal_x)]++;
      cl_tgt_label_hist_[static_cast<int>(target_->at(target_index).normal_x)]++;
    }

    // H and b of Geometry
    Eigen::Matrix<double, 6, 6> H_geo = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> b_geo = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += H_geo;
    bs[omp_get_thread_num()] += b_geo;

  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double FastAPDGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    // ws
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_A.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d values;
    values = svd.singularValues()/svd.singularValues().maxCoeff();
    double geo_weight = values(2);

    double cl_weight = 0.0;
    const bool labels_agree = target_->at(target_index).normal_x == input_->at(i).normal_x;
    if(labels_agree)
      cl_weight = cluster_weight_ > 0.0 ? cluster_weight_ : 1.0/correspondences_.size();
    
    // std::cout << "weight: " << geo_weight << " " << cl_weight << std::endl;

    sum_errors += (1.0 + geo_weight + cl_weight) * error.transpose() * mahalanobis_[i] * error;
    // sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
// CiA and CiB is calculated here
bool FastAPDGICP<PointSource, PointTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::KdTree<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;

    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    }
    else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    }
    else {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;

      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE: // enter here
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
      // cout << covariances[i] << endl;
    }
  }

  return true;
}



}  // namespace fast_gicp

#endif
