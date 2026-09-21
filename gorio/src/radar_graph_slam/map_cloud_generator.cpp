// SPDX-License-Identifier: BSD-2-Clause

#include <radar_graph_slam/map_cloud_generator.hpp>

#include <pcl/octree/octree_search.h>

namespace radar_graph_slam {

MapCloudGenerator::MapCloudGenerator() {}

MapCloudGenerator::~MapCloudGenerator() {}

pcl::PointCloud<MapCloudGenerator::PointT>::Ptr MapCloudGenerator::generate(const std::vector<KeyFrameSnapshot::Ptr>& keyframes, double resolution) const {
  if(keyframes.empty()) {
    std::cerr << "warning: keyframes empty!!" << std::endl;
    return nullptr;
  }

  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  cloud->reserve(keyframes.front()->cloud->size() * keyframes.size());

  for(const auto& keyframe : keyframes) {
    Eigen::Matrix4f pose = keyframe->pose.matrix().cast<float>();
    for(const auto& src_pt : keyframe->cloud->points) {
      double d = src_pt.getVector3fMap().norm();
      if (d > 50) continue;
      PointT dst_pt;
      dst_pt.getVector4fMap() = pose * src_pt.getVector4fMap();
      dst_pt.intensity = src_pt.intensity;
      cloud->push_back(dst_pt);
    }
  }

  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = false;

  if (resolution <=0.0)
    return cloud; // To get unfiltered point cloud with intensity

  pcl::octree::OctreePointCloud<PointT> octree(resolution);
  octree.setInputCloud(cloud);
  octree.addPointsFromInputCloud();


  float search_radius = resolution * 1.5;


  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  octree.getOccupiedVoxelCenters(filtered->points);

  // for (auto& point : filtered->points) {
  //   // Convert the voxel center to a search point
  //   PointT searchPoint;
  //   searchPoint.x = point.x;
  //   searchPoint.y = point.y;
  //   searchPoint.z = point.z;

  //   // Initialize intensity sum and point count
  //   float intensity_sum = 0.0f;
  //   int point_count = 0;

  //   std::vector<int> point_indices;
  //   std::vector<float> point_distances;

  //   if (octree.radiusSearch(search_point, search_radius, point_indices, point_distances) > 0) {
  //     // Compute average intensity of the points found
  //     float intensity_sum = 0.0;
  //     for (int idx : point_indices) {
  //         intensity_sum += cloud->points[idx].intensity;
  //     }
  //     point.intensity = intensity_sum / static_cast<float>(point_indices.size());
  //   } else {
  //     point.intensity = 0.0; // Default intensity if no points are found
  //   }
  // }
  // filtered->width = filtered->size();
  // filtered->height = 1;
  // filtered->is_dense = false;

  filtered->width = filtered->size();
  filtered->height = 1;
  filtered->is_dense = false;
  // no intensity in pcl octree

  

  return filtered;
}

}  // namespace radar_graph_slam
