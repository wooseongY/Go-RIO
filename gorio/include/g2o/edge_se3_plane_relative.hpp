// A ground plane constraint between consecutive keyframes.
//
// The existing floor constraint ties every keyframe to one fixed, global plane,
// which assumes the site is flat. That holds for the cp carpark, 1.8 m of
// vertical range, and not for the loop sequences at 22 to 73 m, so it is off by
// default and rejected as a method: a vehicle driving uphill violates it
// outright.
//
// Between two consecutive keyframes the assumption is much weaker. The ground is
// locally planar even on a slope, because the slope itself changes slowly, so
// the plane one keyframe sees and the plane the next sees are the same physical
// surface seen from two poses. Requiring them to agree constrains the relative
// pose without saying anything about the world being level.
//
// With the plane in keyframe i's body frame as (n_i, d_i), meaning n_i . p +
// d_i = 0, and T_ij = (R, t) taking a point from j's frame to i's, the same
// surface expressed in i is
//
//     n = R n_j,    d = d_j - n . t
//
// so the residual is the disagreement between that and (n_i, d_i). It has three
// degrees of freedom: two tilts of the normal, which are the relative roll and
// pitch against the ground, and one distance, which is the motion along the
// normal -- height above the surface. Yaw and in-plane translation are left
// free, correctly, since a plane is symmetric about its normal.
//
// This is the measurement the vertical error needs. Height is the one axis where
// yaw cannot rotate the error away, and it is the only component measured to
// accumulate coherently in both factors, 0.92-0.97 on the loops, where the
// horizontal components are 0.50-0.75 and near random.
#ifndef EDGE_SE3_PLANE_RELATIVE_HPP
#define EDGE_SE3_PLANE_RELATIVE_HPP

#include <Eigen/Dense>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace g2o {

// Measurement: the two planes, each in its own keyframe's body frame, packed as
// [n_i (3), d_i, n_j (3), d_j].
class EdgeSE3PlaneRelative
    : public BaseBinaryEdge<3, Eigen::Matrix<double, 8, 1>, VertexSE3, VertexSE3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSE3PlaneRelative()
      : BaseBinaryEdge<3, Eigen::Matrix<double, 8, 1>, VertexSE3, VertexSE3>() {
    _measurement.setZero();
  }

  void setPlanes(const Eigen::Vector4d& plane_i, const Eigen::Vector4d& plane_j) {
    _measurement.head<4>() = plane_i;
    _measurement.tail<4>() = plane_j;
  }

  void computeError() override {
    const VertexSE3* vi = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3* vj = static_cast<const VertexSE3*>(_vertices[1]);

    Eigen::Vector3d n_i = _measurement.head<3>();
    Eigen::Vector3d n_j = _measurement.segment<3>(4);
    const double d_i = _measurement(3);
    const double d_j = _measurement(7);
    const double li = n_i.norm();
    const double lj = n_j.norm();
    if(li < 1e-9 || lj < 1e-9) {
      _error.setZero();
      return;
    }
    n_i /= li;
    n_j /= lj;

    // Both normals are written pointing the same way, up out of the ground, by
    // the sign convention below; without that a flipped fit would look like a
    // 180 degree tilt.
    const Eigen::Isometry3d T_ij = vi->estimate().inverse() * vj->estimate();
    const Eigen::Vector3d n = T_ij.rotation() * n_j;
    const double d = d_j / lj - n.dot(T_ij.translation());

    // Two tangential components of the normal disagreement, in a basis
    // orthogonal to n_i. The component along n_i is second order in the tilt
    // and carries no information, so including it would add a row that is
    // always near zero and make the information matrix singular.
    Eigen::Vector3d a = n_i.cross(Eigen::Vector3d::UnitX());
    if(a.norm() < 0.1) {
      a = n_i.cross(Eigen::Vector3d::UnitY());
    }
    a.normalize();
    const Eigen::Vector3d b = n_i.cross(a);

    const Eigen::Vector3d dn = n - n_i;
    _error(0) = dn.dot(a);
    _error(1) = dn.dot(b);
    _error(2) = d - d_i / li;
  }

  bool read(std::istream& is) override {
    for(int i = 0; i < 8; ++i) is >> _measurement(i);
    for(int i = 0; i < 3; ++i)
      for(int j = i; j < 3; ++j) {
        is >> information()(i, j);
        if(i != j) information()(j, i) = information()(i, j);
      }
    return true;
  }

  bool write(std::ostream& os) const override {
    for(int i = 0; i < 8; ++i) os << _measurement(i) << " ";
    for(int i = 0; i < 3; ++i)
      for(int j = i; j < 3; ++j) os << " " << information()(i, j);
    return os.good();
  }
};

}  // namespace g2o

#endif
