/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace
{
bool isPatchProjectionValid(const V3D &pf, const V2D &pc, int width, int height, int scale, int patch_size_half)
{
  if (!pf.allFinite() || !pc.allFinite() || pf[2] <= 1e-6 || scale <= 0) return false;

  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const int pixel_margin = (patch_size_half + 2) * scale;
  return u_ref_i >= pixel_margin &&
         v_ref_i >= pixel_margin &&
         u_ref_i < width - pixel_margin &&
         v_ref_i < height - pixel_margin;
}

bool isFiniteNormal(const V3D &normal)
{
  const double norm = normal.norm();
  return normal.allFinite() && std::isfinite(norm) && norm > 1e-6;
}

bool isFinitePointWithVarForVio(const pointWithVar &pv, bool require_normal)
{
  return pv.point_w.allFinite() &&
         pv.point_b.allFinite() &&
         pv.var.allFinite() &&
         (!require_normal || isFiniteNormal(pv.normal));
}

bool isFiniteVisualPointForVio(const VisualPoint *pt, bool require_normal)
{
  if (pt == nullptr || !pt->pos_.allFinite()) return false;
  if (require_normal && !isFiniteNormal(pt->normal_)) return false;
  return true;
}

bool isFiniteGSPointForVio(const GS_point *pt)
{
  return pt != nullptr &&
         std::isfinite(pt->_points.x) &&
         std::isfinite(pt->_points.y) &&
         std::isfinite(pt->_points.z) &&
         std::isfinite(pt->_normals.x) &&
         std::isfinite(pt->_normals.y) &&
         std::isfinite(pt->_normals.z);
}

bool isFiniteStateForVio(const StatesGroup *state)
{
  return state != nullptr &&
         state->rot_end.allFinite() &&
         state->pos_end.allFinite() &&
         state->vel_end.allFinite() &&
         state->bias_g.allFinite() &&
         state->bias_a.allFinite() &&
         state->gravity.allFinite() &&
         state->cov.allFinite() &&
         std::isfinite(state->inv_expo_time);
}

cv::Mat toGray8UContinuous(const cv::Mat &img)
{
  if (img.empty()) return cv::Mat();

  cv::Mat gray;
  if (img.channels() == 1)
  {
    gray = img;
  }
  else if (img.channels() == 3)
  {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  }
  else if (img.channels() == 4)
  {
    cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
  }
  else
  {
    return cv::Mat();
  }

  if (gray.depth() != CV_8U)
  {
    cv::Mat gray_u8;
    gray.convertTo(gray_u8, CV_8U);
    gray = gray_u8;
  }

  return gray.isContinuous() ? gray : gray.clone();
}
} // namespace

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;



  gsmap_manager.reset(new GSMapManager(gs_octree, gs_voxel_size, octree_max_level));
  cout << GREEN << "root_voxel_size: " << root_voxel_size
       << " gs_voxel_size: " << gs_voxel_size
       << " octree_max_level: " << octree_max_level
       << " border_gs: " << border_gs
       << " normal_rejecter: " << normal_rejecter
       << " scale_factor: " << scale_factor
       << " scale_factor2: " << scale_factor2 << endl << RESET;

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width) / grid_size);
    grid_n_height = ceil(static_cast<double>(height) / grid_size);
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height) / grid_size);
    grid_n_width = ceil(static_cast<double>(width) / grid_size);
  }
  length = grid_n_width * grid_n_height;

   sub_GSMap.reserve(1000);


    pointType= torch::TensorOptions().dtype(torch::kFloat32);

    background = torch::tensor(
        gs_white_background ? std::vector<float>{1.f, 1.f, 1.f} : std::vector<float>{0.f, 0.f, 0.f},
        pointType).to(torch::kCUDA);
    std::cout << "[GS] background="
              << (gs_white_background ? "white" : "black")
              << ", output_dir=" << resolveGSOutputDir()
              << ", save_results=" << (gs_save_results ? "true" : "false")
              << ", save_rendered_images=" << (gs_save_rendered_images ? "true" : "false")
              << ", save_gt_images=" << (gs_save_gt_images ? "true" : "false")
              << ", sparse_vio_fallback=" << (gs_sparse_vio_fallback_en ? "true" : "false")
              << ", pose_update=" << (gs_pose_update_en ? "true" : "false")
              << ", render_jacobian=" << (gs_render_jacobian_en ? "true" : "false")
              << ", pose_fd_jacobian=" << (gs_pose_finite_diff_jacobian_en ? "true" : "false")
              << ", pose_update_exposure=" << (gs_pose_update_exposure_en ? "true" : "false")
              << ", pose_fd_rot_eps=" << gs_pose_fd_rot_eps
              << ", pose_fd_trans_eps=" << gs_pose_fd_trans_eps
              << ", pose_fd_max_gaussians=" << gs_pose_fd_max_gaussians
              << ", pose_update_start_frame=" << gs_pose_update_start_frame
              << ", pose_update_min_gaussians=" << gs_pose_update_min_gaussians
              << ", pose_update_min_points=" << gs_pose_update_min_points
              << ", pose_update_min_measurements=" << gs_pose_update_min_measurements
              << ", pose_update_max_rmse=" << gs_pose_update_max_rmse
              << ", pose_update_step_damping=" << gs_pose_update_step_damping
              << ", pose_update_max_raw_rot_deg=" << gs_pose_update_max_raw_rot_deg
              << ", pose_update_max_raw_trans=" << gs_pose_update_max_raw_trans
              << ", max_pose_update_rot_deg=" << gs_max_pose_update_rot_deg
              << ", max_pose_update_trans=" << gs_max_pose_update_trans
              << ", active_voxel_radius=" << gs_active_voxel_radius
              << ", max_seed_voxels=" << gs_max_seed_voxels
              << ", max_active_voxels=" << gs_max_active_voxels
              << ", max_map_voxels=" << gs_max_map_voxels
              << ", max_total_gaussians=" << gs_max_total_gaussians
              << ", prune_interval_frames=" << gs_prune_interval_frames
              << ", max_insert_gaussians=" << gs_max_insert_gaussians
              << ", max_points_per_voxel=" << gs_max_points_per_voxel
              << ", save_map_interval=" << save_GS_iter
              << std::endl;
    gs::param::ModelParameters modelParams = gs::param::ModelParameters();
    modelParams.type=3;
    modelParams.width=width;
    modelParams.height=height;
    modelParams.scale=image_resize_factor;
    modelParams.output_path = resolveGSOutputDir();

    modelParams.channels=3;
    modelParams.fx=fx;
    modelParams.fy=fy;
    modelParams.cx=cx;
    modelParams.cy=cy;

    // gaussians = std::make_unique<GaussianModel>(2);
    scene = std::make_unique<Scene>(gaussians, modelParams);

    gaussians.Training_setup(gs_params);
    

    img_rendered = cv::Mat::zeros(cam->height(), cam->width(), CV_8UC3);



  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}


Quaternions computeQuaternionFromNormals(const Eigen::Vector3f& normal, const Eigen::Vector3f& ref_vector = Eigen::Vector3f(1, 0, 0)) {
    // if (normal.isZero(1e-6)) {
    //     throw std::invalid_argument("Normal vector must not be zero.");
    // }

    // Normalize the input vectors
    Eigen::Vector3f v1 = normal.normalized();
    Eigen::Vector3f v2 = ref_vector.normalized();

    // Check if the reference vector is almost parallel to the normal vector
    if (std::fabs(v1.dot(v2)) > 0.999) {
        // If almost parallel, choose a different reference vector
        v2 = Eigen::Vector3f(0, 1, 0);  // Use Y axis as default
        if (std::fabs(v1.dot(v2)) > 0.999) {
            v2 = Eigen::Vector3f(0, 0, 1);  // Use Z axis if Y is also parallel
        }
    }

    // Compute the quaternion using Eigen
    Eigen::Quaternionf eigenQuat;
    eigenQuat.setFromTwoVectors(v2, v1); 
    Quaternions result;
    result.qw = eigenQuat.w();
    result.qx = eigenQuat.x();
    result.qy = eigenQuat.y();
    result.qz = eigenQuat.z();

    return result;
}



void VIOManager::insertPointInto_GS_Map2(const std::vector<pointWithVar>& pg)
{
  size_t inserted = 0;
  size_t reject_behind = 0;
  size_t reject_frame = 0;
  size_t reject_normal = 0;
  size_t reject_invalid = 0;
  size_t reject_density = 0;
  size_t reject_voxel_capacity = 0;

  constexpr int kInsertGridCols = 64;
  constexpr int kInsertGridRows = 48;
  std::vector<uint8_t> occupied_insert_bins(kInsertGridCols * kInsertGridRows, 0);
  const bool limit_insert = gs_max_insert_gaussians > 0;
  const size_t max_insert = limit_insert ? static_cast<size_t>(gs_max_insert_gaussians)
                                         : std::numeric_limits<size_t>::max();
  const bool limit_voxel_capacity = gs_max_points_per_voxel > 0 && std::isfinite(gs_voxel_size) && gs_voxel_size > 0.0;
  std::unordered_map<VOXEL_LOCATION, size_t> voxel_point_counts;

  auto current_voxel_count = [&](const VOXEL_LOCATION &position) -> size_t {
    auto cached_count = voxel_point_counts.find(position);
    if (cached_count != voxel_point_counts.end()) return cached_count->second;

    size_t count = 0;
    if (gsmap_manager != nullptr)
    {
      auto voxel_iter = gsmap_manager->gs_map_.find(position);
      if (voxel_iter != gsmap_manager->gs_map_.end() && voxel_iter->second != nullptr)
      {
        count = voxel_iter->second->count_gs_points();
      }
    }
    voxel_point_counts[position] = count;
    return count;
  };

  for (const auto& pointVar : pg) 
  {
    if (!isFinitePointWithVarForVio(pointVar, true))
    {
      reject_invalid++;
      continue;
    }
    V3D p_w(pointVar.point_w.x(), pointVar.point_w.y(), pointVar.point_w.z());
    V3D p_b(pointVar.point_b.x(), pointVar.point_b.y(), pointVar.point_b.z());
    V3D pf = new_frame_->w2f(p_w);
    if (!pf.allFinite() || pf[2] < 0)
    {
      reject_behind++;
      continue;
    }
    V2D pc = new_frame_->w2c(p_w);
    if (!pc.allFinite() || !new_frame_->cam_->isInFrame(pc.cast<int>(), border_gs))
    {
      reject_frame++;
      continue;
    }

    Normal normal;

    Eigen::Vector3d normalizedNormal = pointVar.normal;



    normal.x = static_cast<float>(normalizedNormal.x());
    normal.y = static_cast<float>(normalizedNormal.y());
    normal.z = static_cast<float>(normalizedNormal.z());

    float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (!std::isfinite(length) || length < std::max(static_cast<float>(normal_rejecter), 1e-6f))
    {
      reject_normal++;
      continue;
    }


    Quaternions quaternion = computeQuaternionFromNormals(pointVar.normal.cast<float>(), Eigen::Vector3f(1, 0, 0));

    Point point;
    point.x = static_cast<float>(p_w.x());
    point.y = static_cast<float>(p_w.y());
    point.z = static_cast<float>(p_w.z());

    cv::Mat img_rgb = img_undistort; // Assuming this is how you access the image
    V3F pixel = getInterpolatedPixel(img_rgb, pc);

    Color color;
    color.r = pixel[2];
    color.g = pixel[1];
    color.b = pixel[0];

    if (limit_insert)
    {
      if (inserted >= max_insert)
      {
        reject_density++;
        continue;
      }

      const int bin_x = std::clamp(
          static_cast<int>(pc[0] * kInsertGridCols / std::max(1, width)),
          0, kInsertGridCols - 1);
      const int bin_y = std::clamp(
          static_cast<int>(pc[1] * kInsertGridRows / std::max(1, height)),
          0, kInsertGridRows - 1);
      const size_t bin_index = static_cast<size_t>(bin_y * kInsertGridCols + bin_x);
      if (occupied_insert_bins[bin_index] != 0)
      {
        reject_density++;
        continue;
      }
      occupied_insert_bins[bin_index] = 1;
    }

    if (limit_voxel_capacity)
    {
      VOXEL_LOCATION position(
          static_cast<int64_t>(std::floor(p_w[0] / gs_voxel_size)),
          static_cast<int64_t>(std::floor(p_w[1] / gs_voxel_size)),
          static_cast<int64_t>(std::floor(p_w[2] / gs_voxel_size)));
      const size_t count = current_voxel_count(position);
      if (count >= static_cast<size_t>(gs_max_points_per_voxel))
      {
        reject_voxel_capacity++;
        continue;
      }
      voxel_point_counts[position] = count + 1;
    }

    Distance distance;
    float dis=p_b.x()/(fx)*scale_factor;

    distance.r1= static_cast<float>(0.01);

    distance.r2 = static_cast<float>(scale_factor2);
    distance.r3 = static_cast<float>(scale_factor2);



    GS_point* gs_pt = new GS_point;

    gs_pt->_points =point;
    gs_pt->_colors = color;
    gs_pt->_normals = normal;
    gs_pt->_distance = distance;
    gs_pt->_quaternion = quaternion;
    gs_pt->_opacity = 1.0;
    gs_total++;
    inserted++;

          gsmap_manager->UpdateGSMap(gs_pt);
    
    
  }

  std::cout << "[GS] insert: input_pg=" << pg.size()
            << ", inserted=" << inserted
            << ", reject_behind=" << reject_behind
            << ", reject_frame=" << reject_frame
            << ", reject_normal=" << reject_normal
            << ", reject_invalid=" << reject_invalid
            << ", reject_density=" << reject_density
            << ", reject_voxel_capacity=" << reject_voxel_capacity
            << ", max_insert=" << (limit_insert ? std::to_string(max_insert) : std::string("unlimited"))
            << ", max_points_per_voxel="
            << (limit_voxel_capacity ? std::to_string(gs_max_points_per_voxel) : std::string("unlimited"))
            << ", gs_total=" << gs_total
            << ", map_voxels=" << gsmap_manager->gs_map_.size()
            << ", border_gs=" << border_gs
            << ", normal_rejecter=" << normal_rejecter
            << std::endl;
}

void VIOManager::pruneGSMapIfNeeded()
{
  if (gsmap_manager == nullptr || gsmap_manager->gs_map_.empty()) return;
  if (!std::isfinite(gs_voxel_size) || gs_voxel_size <= 0.0) return;

  const bool limit_voxels = gs_max_map_voxels > 0 &&
      gsmap_manager->gs_map_.size() > static_cast<size_t>(gs_max_map_voxels);
  const bool limit_points = gs_max_total_gaussians > 0 &&
      gs_total > static_cast<int64>(gs_max_total_gaussians);
  if (!limit_voxels && !limit_points) return;

  if (gs_prune_interval_frames > 1 &&
      gs_frame_count > 0 &&
      (gs_frame_count % gs_prune_interval_frames) != 0)
  {
    return;
  }

  V3D reference_pos = state != nullptr ? state->pos_end : V3D::Zero();
  if (Rcw.allFinite() && Pcw.allFinite())
  {
    reference_pos = -Rcw.transpose() * Pcw;
  }

  struct PruneCandidate
  {
    VOXEL_LOCATION position;
    double distance2 = 0.0;
    size_t point_count = 0;
  };

  std::vector<PruneCandidate> candidates;
  candidates.reserve(gsmap_manager->gs_map_.size());
  int64 counted_points = 0;
  for (const auto &entry : gsmap_manager->gs_map_)
  {
    if (entry.second == nullptr) continue;

    size_t point_count = 0;
    if (limit_points)
    {
      point_count = entry.second->count_gs_points();
      counted_points += static_cast<int64>(point_count);
    }

    const V3D voxel_center(
        (static_cast<double>(entry.first.x) + 0.5) * gs_voxel_size,
        (static_cast<double>(entry.first.y) + 0.5) * gs_voxel_size,
        (static_cast<double>(entry.first.z) + 0.5) * gs_voxel_size);
    candidates.push_back({entry.first, (voxel_center - reference_pos).squaredNorm(), point_count});
  }

  if (candidates.empty()) return;

  if (limit_points && counted_points >= 0) gs_total = counted_points;

  const auto farther_first = [](const PruneCandidate &lhs, const PruneCandidate &rhs) {
    return lhs.distance2 > rhs.distance2;
  };
  if (limit_voxels && !limit_points)
  {
    const size_t keep_remove_count = gsmap_manager->gs_map_.size() -
        static_cast<size_t>(gs_max_map_voxels);
    if (keep_remove_count < candidates.size())
    {
      std::nth_element(candidates.begin(), candidates.begin() + keep_remove_count,
                       candidates.end(), farther_first);
      candidates.resize(keep_remove_count);
    }
  }
  std::sort(candidates.begin(), candidates.end(), farther_first);

  const size_t voxel_count_before = gsmap_manager->gs_map_.size();
  const int64 gs_total_before = gs_total;
  size_t removed_voxels = 0;
  int64 removed_points = 0;

  for (const PruneCandidate &candidate : candidates)
  {
    const bool voxel_over_limit = gs_max_map_voxels > 0 &&
        gsmap_manager->gs_map_.size() > static_cast<size_t>(gs_max_map_voxels);
    const bool point_over_limit = gs_max_total_gaussians > 0 &&
        gs_total > static_cast<int64>(gs_max_total_gaussians);
    if (!voxel_over_limit && !point_over_limit) break;

    auto iter = gsmap_manager->gs_map_.find(candidate.position);
    if (iter == gsmap_manager->gs_map_.end() || iter->second == nullptr) continue;

    std::vector<GS_point *> voxel_points;
    iter->second->get_all_gs_points(voxel_points);
    std::unordered_set<GS_point *> unique_points(voxel_points.begin(), voxel_points.end());
    for (GS_point *point : unique_points)
    {
      delete point;
    }
    removed_points += static_cast<int64>(unique_points.size());
    delete iter->second;
    gsmap_manager->gs_map_.erase(iter);
    removed_voxels++;
    gs_total = std::max<int64>(0, gs_total - static_cast<int64>(unique_points.size()));
  }

  if (removed_voxels > 0)
  {
    std::cout << "[GS] prune map: removed_voxels=" << removed_voxels
              << ", removed_points=" << removed_points
              << ", voxels_before=" << voxel_count_before
              << ", voxels_after=" << gsmap_manager->gs_map_.size()
              << ", gs_total_before=" << gs_total_before
              << ", gs_total_after=" << gs_total
              << ", max_map_voxels=" << gs_max_map_voxels
              << ", max_total_gaussians=" << gs_max_total_gaussians
              << ", frame=" << gs_frame_count
              << std::endl;
  }
}


void VIOManager::retrieveFrom_GS_Map2(vector<pointWithVar> &pg)
{
  sub_GSMap.clear();
  sub_GSMap_ptrs.clear();
  sub_octree_ptr_list.clear();
  sub_gs_map.clear();

  if(gsmap_manager->gs_map_.empty()) return;

  const bool unlimited_render_gs = outlier_threshold3 <= 0.0;
  const size_t max_render_gs = !unlimited_render_gs
      ? static_cast<size_t>(outlier_threshold3)
      : std::numeric_limits<size_t>::max();
  std::unordered_set<VOXEL_LOCATION> seed_voxels;
  seed_voxels.reserve(pg.size());

  if (!std::isfinite(gs_voxel_size) || gs_voxel_size <= 0.0)
  {
    std::cout << "[GS] retrieve skip: invalid gs_voxel_size=" << gs_voxel_size << std::endl;
    return;
  }

  for (const auto &point : pg)
  {
    const V3D &pt_w = point.point_w;
    if (!pt_w.allFinite()) continue;
    VOXEL_LOCATION position(
        static_cast<int64_t>(std::floor(pt_w[0] / gs_voxel_size)),
        static_cast<int64_t>(std::floor(pt_w[1] / gs_voxel_size)),
        static_cast<int64_t>(std::floor(pt_w[2] / gs_voxel_size)));
    seed_voxels.insert(position);
  }

  V3D camera_center_w = state != nullptr ? state->pos_end : V3D::Zero();
  if (Rcw.allFinite() && Pcw.allFinite())
  {
    camera_center_w = -Rcw.transpose() * Pcw;
  }

  std::vector<VOXEL_LOCATION> seed_voxel_list;
  seed_voxel_list.reserve(seed_voxels.size());
  for (const VOXEL_LOCATION &seed : seed_voxels)
  {
    seed_voxel_list.push_back(seed);
  }

  bool seed_voxels_capped = false;
  if (gs_max_seed_voxels > 0 && seed_voxel_list.size() > static_cast<size_t>(gs_max_seed_voxels))
  {
    std::vector<std::pair<double, VOXEL_LOCATION>> ranked_seeds;
    ranked_seeds.reserve(seed_voxel_list.size());
    for (const VOXEL_LOCATION &position : seed_voxel_list)
    {
      const V3D voxel_center(
          (static_cast<double>(position.x) + 0.5) * gs_voxel_size,
          (static_cast<double>(position.y) + 0.5) * gs_voxel_size,
          (static_cast<double>(position.z) + 0.5) * gs_voxel_size);
      ranked_seeds.emplace_back((voxel_center - camera_center_w).squaredNorm(), position);
    }

    const size_t keep_count = static_cast<size_t>(gs_max_seed_voxels);
    std::nth_element(ranked_seeds.begin(), ranked_seeds.begin() + keep_count, ranked_seeds.end(),
                     [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    ranked_seeds.resize(keep_count);
    std::sort(ranked_seeds.begin(), ranked_seeds.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    seed_voxel_list.clear();
    seed_voxel_list.reserve(static_cast<size_t>(gs_max_seed_voxels));
    for (size_t i = 0; i < ranked_seeds.size(); ++i)
    {
      seed_voxel_list.push_back(ranked_seeds[i].second);
    }
    seed_voxels_capped = true;
  }

  const int active_radius = std::clamp(gs_active_voxel_radius, 0, 3);
  const size_t neighbor_span = static_cast<size_t>(2 * active_radius + 1);
  const size_t neighbor_count = neighbor_span * neighbor_span * neighbor_span;
  std::unordered_set<VOXEL_LOCATION> query_voxels;
  const size_t query_reserve_target =
      seed_voxel_list.size() * std::max<size_t>(neighbor_count, 1);
  query_voxels.reserve(gs_max_active_voxels > 0
      ? std::min(query_reserve_target, static_cast<size_t>(gs_max_active_voxels))
      : query_reserve_target);

  for (const VOXEL_LOCATION &seed : seed_voxel_list)
  {
    for (int dx = -active_radius; dx <= active_radius; ++dx)
    {
      for (int dy = -active_radius; dy <= active_radius; ++dy)
      {
        for (int dz = -active_radius; dz <= active_radius; ++dz)
        {
          query_voxels.emplace(seed.x + dx, seed.y + dy, seed.z + dz);
        }
      }
    }
  }

  const size_t expanded_query_voxels = query_voxels.size();
  bool query_voxels_capped = false;
  if (gs_max_active_voxels > 0 && query_voxels.size() > static_cast<size_t>(gs_max_active_voxels))
  {
    std::vector<std::pair<double, VOXEL_LOCATION>> ranked_voxels;
    ranked_voxels.reserve(query_voxels.size());
    for (const VOXEL_LOCATION &position : query_voxels)
    {
      const V3D voxel_center(
          (static_cast<double>(position.x) + 0.5) * gs_voxel_size,
          (static_cast<double>(position.y) + 0.5) * gs_voxel_size,
          (static_cast<double>(position.z) + 0.5) * gs_voxel_size);
      ranked_voxels.emplace_back((voxel_center - camera_center_w).squaredNorm(), position);
    }

    const size_t keep_count = static_cast<size_t>(gs_max_active_voxels);
    std::nth_element(ranked_voxels.begin(), ranked_voxels.begin() + keep_count, ranked_voxels.end(),
                     [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    ranked_voxels.resize(keep_count);
    std::sort(ranked_voxels.begin(), ranked_voxels.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    query_voxels.clear();
    query_voxels.reserve(static_cast<size_t>(gs_max_active_voxels));
    for (size_t i = 0; i < ranked_voxels.size(); ++i)
    {
      query_voxels.insert(ranked_voxels[i].second);
    }
    query_voxels_capped = true;
  }

  constexpr int kSampleGridCols = 32;
  constexpr int kSampleGridRows = 24;
  struct GSCandidate
  {
    GS_point *point = nullptr;
    double depth = 0.0;
  };
  std::vector<std::vector<GSCandidate>> image_bins(kSampleGridCols * kSampleGridRows);

  size_t matched_voxels = 0;
  size_t candidate_points = 0;
  size_t valid_candidates = 0;
  size_t reject_behind = 0;
  size_t reject_frame = 0;
  size_t reject_invalid = 0;
  size_t occupied_bins = 0;

  for (const auto &position : query_voxels)
  {
    auto corre_gs_voxel = gsmap_manager->gs_map_.find(position);
    if (corre_gs_voxel == gsmap_manager->gs_map_.end()) continue;
    matched_voxels++;

    std::vector<GS_point*> voxel_points;
    corre_gs_voxel->second->get_all_gs_points(voxel_points);
    candidate_points += voxel_points.size();

    for (GS_point *pt : voxel_points)
    {
      if (!isFiniteGSPointForVio(pt))
      {
        reject_invalid++;
        continue;
      }

      V3D p_w(pt->_points.x, pt->_points.y, pt->_points.z);
      V3D pf = new_frame_->w2f(p_w);
      if (!pf.allFinite() || pf[2] <= 0)
      {
        reject_behind++;
        continue;
      }

      V2D pc(new_frame_->w2c(p_w));
      if (!pc.allFinite() || !new_frame_->cam_->isInFrame(pc.cast<int>(), border_gs))
      {
        reject_frame++;
        continue;
      }

      const int bin_x = std::clamp(
          static_cast<int>(pc[0] * kSampleGridCols / std::max(1, width)),
          0, kSampleGridCols - 1);
      const int bin_y = std::clamp(
          static_cast<int>(pc[1] * kSampleGridRows / std::max(1, height)),
          0, kSampleGridRows - 1);
      auto &bin = image_bins[bin_y * kSampleGridCols + bin_x];
      if (bin.empty()) occupied_bins++;
      bin.push_back({pt, pf[2]});
      valid_candidates++;
    }
  }

  sub_GSMap.reserve(std::min(valid_candidates, max_render_gs));
  sub_GSMap_ptrs.reserve(std::min(valid_candidates, max_render_gs));
  if (valid_candidates <= max_render_gs)
  {
    for (const auto &bin : image_bins)
    {
      for (const GSCandidate &candidate : bin)
      {
        sub_GSMap.push_back(*candidate.point);
        sub_GSMap_ptrs.push_back(candidate.point);
      }
    }
  }
  else
  {
    for (auto &bin : image_bins)
    {
      std::sort(bin.begin(), bin.end(), [](const GSCandidate &lhs, const GSCandidate &rhs) {
        return lhs.depth < rhs.depth;
      });
    }

    std::vector<size_t> bin_offsets(image_bins.size(), 0);
    while (sub_GSMap.size() < max_render_gs)
    {
      bool made_progress = false;
      for (size_t i = 0; i < image_bins.size() && sub_GSMap.size() < max_render_gs; ++i)
      {
        if (bin_offsets[i] >= image_bins[i].size()) continue;
        GS_point *selected_point = image_bins[i][bin_offsets[i]++].point;
        sub_GSMap.push_back(*selected_point);
        sub_GSMap_ptrs.push_back(selected_point);
        made_progress = true;
      }
      if (!made_progress) break;
    }
  }

  std::cout << "[GS] retrieve: seed_voxels=" << seed_voxels.size()
            << ", seed_voxels_used=" << seed_voxel_list.size()
            << ", seed_voxels_capped=" << (seed_voxels_capped ? "true" : "false")
            << ", active_radius=" << active_radius
            << ", expanded_query_voxels=" << expanded_query_voxels
            << ", query_voxels=" << query_voxels.size()
            << ", query_voxels_capped=" << (query_voxels_capped ? "true" : "false")
            << ", matched_voxels=" << matched_voxels
            << ", candidate_points=" << candidate_points
            << ", valid_candidates=" << valid_candidates
            << ", selected=" << sub_GSMap.size()
            << ", reject_behind=" << reject_behind
            << ", reject_frame=" << reject_frame
            << ", reject_invalid=" << reject_invalid
            << ", occupied_bins=" << occupied_bins
            << ", max_render_gs=" << (unlimited_render_gs ? std::string("unlimited") : std::to_string(max_render_gs))
            << ", capped=" << (valid_candidates > sub_GSMap.size() ? "true" : "false")
            << std::endl;
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    V3D pt_w = pg[i].point_w;
    if (!pt_w.allFinite()) continue;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c.allFinite() && pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (px.allFinite() && new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map)
  {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (!isFiniteVisualPointForVio(pt, false)) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (!dir.allFinite() || dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (pc.allFinite() && new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i])
      {
        V3D sample_point_w = new_frame_->f2w(it);
        if (!sample_point_w.allFinite()) continue;
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (!isFiniteVisualPointForVio(pt, false)) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (!dir.allFinite() || dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (pc.allFinite() && new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_ &&
                current_octo->plane_ptr_->center_.allFinite() &&
                isFiniteNormal(current_octo->plane_ptr_->normal_))
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      if (!isFiniteVisualPointForVio(pt, true)) continue;
      V2D pc(new_frame_->w2c(pt->pos_));
      V3D pt_cam(new_frame_->w2f(pt->pos_));
      if (!isPatchProjectionValid(pt_cam, pc, width, height, 1, patch_size_half)) continue;

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
	      else
	      {
	        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
	      }
	      if (ref_ftr == nullptr || !ref_ftr->px_.allFinite()) continue;

	      if (normal_en)
	      {
	        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();

	        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
	        if (!norm_vec.allFinite() || !pf.allFinite()) continue;
	        // V3D pf_norm = pf.normalized();

	        // double cos_theta = norm_vec.dot(pf_norm);
	        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

	        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);
	        if (!A_cur_ref_zero.allFinite()) continue;

	        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
	      }
	      else
	      {
	        auto iter_warp = warp_map.find(ref_ftr->id_);
	        if (iter_warp != warp_map.end())
	        {
	          search_level = iter_warp->second->search_level;
	          A_cur_ref_zero = iter_warp->second->A_cur_ref;
	          if (!A_cur_ref_zero.allFinite()) continue;
	        }
	        else
	        {
	          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
	                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);
	          if (!A_cur_ref_zero.allFinite()) continue;

	          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

	      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
	      {
	        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
	      }
	      if (!std::all_of(patch_wrap.begin(), patch_wrap.end(), [](float value) { return std::isfinite(value); })) continue;

	      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
	        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
	                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
	      }
	      if (!std::isfinite(error)) continue;

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

bool VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  G.setZero();
  if (total_points == 0) return true;
  
  compute_jacobian_time = update_ekf_time = 0.0;
  const StatesGroup state_before = (*state);
  const MD(DIM_STATE, DIM_STATE) cov_before = state->cov;
  MD(DIM_STATE, DIM_STATE) last_valid_G;
  last_valid_G.setZero();
  bool has_valid_gain = false;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    G.setZero();
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);

    if (G.allFinite() && G.norm() > 0.0)
    {
      last_valid_G = G;
      has_valid_gain = true;
    }
  }

  const bool finite_state = state->rot_end.allFinite() &&
                            state->pos_end.allFinite() &&
                            state->vel_end.allFinite() &&
                            state->bias_g.allFinite() &&
                            state->bias_a.allFinite() &&
                            state->gravity.allFinite() &&
                            std::isfinite(state->inv_expo_time);
  if (!finite_state || !has_valid_gain)
  {
    (*state) = state_before;
    state->cov = cov_before;
    G.setZero();
    updateFrameState(*state);
    std::cout << "[VIO-EKF] update rejected: finite_state="
              << (finite_state ? "true" : "false")
              << ", valid_gain=" << (has_valid_gain ? "true" : "false")
              << std::endl;
    return false;
  }

  const VD(DIM_STATE) state_delta = (*state) - state_before;
  const double rot_delta_deg = state_delta.block<3, 1>(0, 0).norm() * 57.29577951308232;
  const double trans_delta = state_delta.block<3, 1>(3, 0).norm();
  const double vel_delta = state_delta.block<3, 1>(7, 0).norm();
  const bool rot_too_large = vio_max_update_rot_deg > 0.0 && rot_delta_deg > vio_max_update_rot_deg;
  const bool trans_too_large = vio_max_update_trans > 0.0 && trans_delta > vio_max_update_trans;
  const bool vel_too_large = vio_max_update_vel > 0.0 && vel_delta > vio_max_update_vel;
  if (!std::isfinite(rot_delta_deg) || !std::isfinite(trans_delta) ||
      !std::isfinite(vel_delta) || rot_too_large || trans_too_large || vel_too_large)
  {
    (*state) = state_before;
    state->cov = cov_before;
    G.setZero();
    updateFrameState(*state);
    std::cout << "[VIO-EKF] update rejected by pose gate: rot_deg=" << rot_delta_deg
              << " (max=" << vio_max_update_rot_deg
              << "), trans=" << trans_delta
              << " (max=" << vio_max_update_trans
              << "), vel=" << vel_delta
              << " (max=" << vio_max_update_vel << ")" << std::endl;
    return false;
  }

  const MD(DIM_STATE, DIM_STATE) cov_after = cov_before - last_valid_G * cov_before;
  if (!cov_after.allFinite())
  {
    (*state) = state_before;
    state->cov = cov_before;
    G.setZero();
    updateFrameState(*state);
    std::cout << "[VIO-EKF] update rejected: covariance is not finite" << std::endl;
    return false;
  }

  state->cov = cov_after;
  G = last_valid_G;
  updateFrameState(*state);
  return true;
}

void VIOManager::computeJacobianAndUpdateEKF_GS(cv::Mat img,cv::Mat img_rendered)
{
  G.setZero();

  if (!gs_pose_update_en)
  {
    std::cout << "[GS-EKF] pose update disabled; keep GS render/map only" << std::endl;
    return;
  }

  if (gs_frame_count < gs_pose_update_start_frame)
  {
    std::cout << "[GS-EKF] skip: waiting for mature GS map, frame=" << gs_frame_count
              << ", start_frame=" << gs_pose_update_start_frame << std::endl;
    return;
  }

  if (gs_pose_update_min_gaussians > 0 &&
      sub_GSMap.size() < static_cast<size_t>(gs_pose_update_min_gaussians))
  {
    std::cout << "[GS-EKF] skip: gaussian window too small, sub_GSMap=" << sub_GSMap.size()
              << ", min=" << gs_pose_update_min_gaussians << std::endl;
    return;
  }

  if (total_points == 0 || img_rendered.empty())
  {
    std::cout << "[GS-EKF] skip: total_points=" << total_points
              << ", img_rendered_empty="
              << (img_rendered.empty() ? "true" : "false") << std::endl;
    return;
  }

  if (gs_pose_update_min_points > 0 && total_points < gs_pose_update_min_points)
  {
    std::cout << "[GS-EKF] skip: not enough selected GS measurements, total_points="
              << total_points << ", min=" << gs_pose_update_min_points << std::endl;
    return;
  }

  const StatesGroup state_before = (*state);
  const MD(DIM_STATE, DIM_STATE) cov_before = state->cov;
  std::cout << "[GS-EKF] update start: total_points=" << total_points
            << ", img_rendered_empty=" << (img_rendered.empty() ? "true" : "false")
            << ", exposure_update=" << (gs_pose_update_exposure_en ? "true" : "false")
            << std::endl;

  cv::Mat img_rendered_gray = toGray8UContinuous(img_rendered);
  if (img_rendered_gray.empty())
  {
    std::cout << "[GS-EKF] skip: rendered image cannot be converted to gray8, channels="
              << img_rendered.channels() << ", depth=" << img_rendered.depth() << std::endl;
    return;
  }

  const bool update_success = updateState_gs(img, img_rendered_gray, 0);
  const bool finite_gain = G.allFinite();
  if (!update_success || !finite_gain)
  {
    (*state) = state_before;
    state->cov = cov_before;
    G.setZero();
    updateFrameState(*state);
    std::cout << "[GS-EKF] update rejected: success="
              << (update_success ? "true" : "false")
              << ", finite_G=" << (finite_gain ? "true" : "false")
              << std::endl;
    return;
  }

  const VD(DIM_STATE) state_delta = (*state) - state_before;
  const double rot_delta_deg = state_delta.block<3, 1>(0, 0).norm() * 57.29577951308232;
  const double trans_delta = state_delta.block<3, 1>(3, 0).norm();
  const bool rot_too_large = gs_max_pose_update_rot_deg > 0.0 && rot_delta_deg > gs_max_pose_update_rot_deg;
  const bool trans_too_large = gs_max_pose_update_trans > 0.0 && trans_delta > gs_max_pose_update_trans;
  if (!std::isfinite(rot_delta_deg) || !std::isfinite(trans_delta) || rot_too_large || trans_too_large)
  {
    (*state) = state_before;
    state->cov = cov_before;
    G.setZero();
    updateFrameState(*state);
    std::cout << "[GS-EKF] update rejected by pose gate: rot_deg=" << rot_delta_deg
              << " (max=" << gs_max_pose_update_rot_deg
              << "), trans=" << trans_delta
              << " (max=" << gs_max_pose_update_trans << ")" << std::endl;
    return;
  }

  const MD(DIM_STATE, DIM_STATE) cov_after = cov_before - G * cov_before;
  if (!cov_after.allFinite())
  {
    (*state) = state_before;
    state->cov = cov_before;
    G.setZero();
    updateFrameState(*state);
    std::cout << "[GS-EKF] update rejected: covariance is not finite" << std::endl;
    return;
  }
  state->cov = cov_after;

  updateFrameState(*state);

  std::cout << "[GS-EKF] update done: rot_deg=" << rot_delta_deg
            << ", trans=" << trans_delta
            << ", n_meas=" << gs_last_update_measurements
            << ", rmse=" << gs_last_update_rmse
            << std::endl;
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (!isFinitePointWithVarForVio(pg[i], true)) continue;

    V3D pt = pg[i].point_w;
    V3D pf = new_frame_->w2f(pt);
    if (!pf.allFinite() || pf[2] <= 1e-6) continue;
    V2D pc(new_frame_->w2c(pt));
    if (!pc.allFinite()) continue;

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    if (!isFinitePointWithVarForVio(visual_submap->add_from_voxel_map[j], true)) continue;
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V3D pf = new_frame_->w2f(pt);
    if (!pf.allFinite() || pf[2] <= 1e-6) continue;
    V2D pc(new_frame_->w2c(pt));
    if (!pc.allFinite()) continue;

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;
      if (!isFinitePointWithVarForVio(pt_var, true)) continue;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      if (!norm_vec.allFinite() || !dir.allFinite() || dir.norm() <= 1e-6) continue;
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      if (!std::isfinite(cos_theta)) continue;
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));
      V3D pf = new_frame_->w2f(pt);
      if (!isPatchProjectionValid(pf, pc, width, height, 1, patch_size_half)) continue;

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (!isFiniteVisualPointForVio(pt, false)) continue;
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    V3D pf(new_frame_->w2f(pt->pos_));
    if (!isPatchProjectionValid(pf, pc, width, height, 1, patch_size_half)) continue;
    if (pt->obs_.empty()) continue;
    bool add_flag = false;

    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    if (last_feature == nullptr || !last_feature->px_.allFinite()) continue;

    float *patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    const double cos_delta = std::clamp(0.5 * (delta_pose.rotation_matrix().trace() - 1), -1.0, 1.0);
    double delta_theta = std::acos(cos_delta);
    if (!std::isfinite(delta_p) || !std::isfinite(delta_theta))
    {
      delete[] patch_temp;
      continue;
    }
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
	    if (add_flag)
	    {
	      Vector3d f = cam->cam2world(pc);
	      if (!f.allFinite())
	      {
	        delete[] patch_temp;
	        continue;
	      }
	      update_num += 1;
	      update_flag[i] = 1;
	      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
	      ftr_new->img_ = img;
	      ftr_new->id_ = new_frame_->id_;
	      ftr_new->inv_expo_time_ = state->inv_expo_time;
	      pt->addFrameRef(ftr_new);
	    }
	    else
	    {
	      delete[] patch_temp;
	    }
	  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!isFiniteVisualPointForVio(pt, false)) continue;
    if (!pt->is_normal_initialized_) continue;
    if (!isFiniteNormal(pt->normal_) || !pt->covariance_.allFinite()) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    if (!p_w.allFinite()) continue;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        if (!plane.center_.allFinite() ||
            !isFiniteNormal(plane.normal_) ||
            !plane.plane_var_.allFinite() ||
            !std::isfinite(plane.radius_))
        {
          continue;
        }
        double dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        double dis_to_plane_abs = fabs(dis_to_plane);
        double dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                               (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                               (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        double range_sq = dis_to_center - dis_to_plane * dis_to_plane;
        if (!std::isfinite(dis_to_plane) ||
            !std::isfinite(dis_to_center) ||
            !std::isfinite(range_sq) ||
            range_sq < -1e-9)
        {
          continue;
        }
        double range_dis = sqrt(std::max(0.0, range_sq));
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (!J_nq.allFinite() || !std::isfinite(sigma_l) || sigma_l <= 1e-12) continue;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);
            
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0 || visual_submap == nullptr || new_frame_ == nullptr || new_frame_->img_.empty()) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  const size_t tracked_points = std::min(visual_submap->voxel_points.size(), visual_submap->warp_patch.size());
  for (size_t i = 0; i < tracked_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (isFiniteVisualPointForVio(pt, true) && pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      if (ref_ftr == nullptr || ref_ftr->img_.empty() || !ref_ftr->px_.allFinite() ||
          visual_submap->warp_patch[i].size() < static_cast<size_t>(patch_size_total))
      {
        continue;
      }
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));
      V3D pf_cur(new_frame_->w2f(pt->pos_));
      if (!isPatchProjectionValid(pf_cur, pc, width, height, 1, patch_size_half) || !pc_prior.allFinite())
      {
        continue;
      }

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);
      if (!norm_vec.allFinite() || !pf.allFinite()) continue;

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);
      if (!A_cur_ref.allFinite()) continue;

      double D = A_cur_ref.determinant();
      if (!std::isfinite(D) || std::abs(D) < 1e-9 || std::abs(D) > 3) continue;
      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            if (i >= 0 && j >= 0 && i < width && j < height) it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (size_t i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!isFiniteVisualPointForVio(pt, true) || !pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;
    if (ref_ftr == nullptr || ref_ftr->img_.empty() || !ref_ftr->px_.allFinite() || !ref_ftr->f_.allFinite() ||
        !pc.allFinite())
    {
      continue;
    }

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    if (!A_cur_ref.allFinite()) continue;
    double D = A_cur_ref.determinant();
    if (!std::isfinite(D) || std::abs(D) < 1e-9 || std::abs(D) > 3) continue;
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          if (col >= 0 && row >= 0 && col < width && row < height) it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr || pt->ref_patch == nullptr) continue;
    cv::Mat img = pt->ref_patch->img_;
    if (img.empty()) continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    if (!isPatchProjectionValid(pf, pc, width, height, scale, patch_size_half)) continue;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      if (!pf.allFinite() || pf[2] <= 1e-6) continue;
      pc = cam->world2cam(pf);
      if (!isPatchProjectionValid(pf, pc, width, height, scale, patch_size_half)) continue;

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    if (n_meas == 0)
    {
      std::cout << "[VIO-EKF] inverse update rejected: no valid patch pixels at level "
                << level << ", iteration=" << iteration << std::endl;
      (*state) = old_state;
      G.setZero();
      break;
    }
    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      if (!K_1.allFinite())
      {
        std::cout << "[VIO-EKF] inverse update rejected: non-finite Kalman matrix" << std::endl;
        (*state) = old_state;
        G.setZero();
        break;
      }
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      if (!G.allFinite() || !solution.allFinite())
      {
        std::cout << "[VIO-EKF] inverse update rejected: non-finite gain or solution" << std::endl;
        (*state) = old_state;
        G.setZero();
        break;
      }
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break; 
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();
    
    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      if (!pf.allFinite() || pf[2] <= 1e-6) continue;
      V2D pc = cam->world2cam(pf);
      if (!isPatchProjectionValid(pf, pc, width, height, scale, patch_size_half)) continue;

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;

          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    if (n_meas == 0)
    {
      std::cout << "[VIO-EKF] update rejected: no valid patch pixels at level "
                << level << ", iteration=" << iteration << std::endl;
      (*state) = old_state;
      G.setZero();
      break;
    }
    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      if (!K_1.allFinite())
      {
        std::cout << "[VIO-EKF] update rejected: non-finite Kalman matrix" << std::endl;
        (*state) = old_state;
        G.setZero();
        break;
      }
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);
      if (!G.allFinite() || !solution.allFinite())
      {
        std::cout << "[VIO-EKF] update rejected: non-finite gain or solution" << std::endl;
        (*state) = old_state;
        G.setZero();
        break;
      }

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

bool VIOManager::updateState_gs(cv::Mat img, cv::Mat img_render, int level)
{
if (total_points == 0) return false;
gs_last_update_measurements = 0;
gs_last_update_rmse = 0.0;
const bool gs_exposure_update_en = gs_pose_update_exposure_en && exposure_estimate_en;
img = toGray8UContinuous(img);
img_render = toGray8UContinuous(img_render);
if (img.empty() || img_render.empty() || img.size() != img_render.size())
{
std::cout << "[GS-EKF] update abort: invalid gray images, img="
          << img.cols << "x" << img.rows
          << ", render=" << img_render.cols << "x" << img_render.rows
          << std::endl;
return false;
}
if (img.cols != width || img.rows != height)
{
std::cout << "[GS-EKF] update abort: image size does not match camera, img="
          << img.cols << "x" << img.rows
          << ", camera=" << width << "x" << height
          << std::endl;
return false;
}
StatesGroup old_state = (*state);
VectorXd z;
MatrixXd H_sub;
bool EKF_end = false;
bool has_valid_update = false;
float last_error = std::numeric_limits<float>::max();
const int H_DIM = total_points * patch_size_total;
z.resize(H_DIM);
z.setZero();
H_sub.resize(H_DIM, 7);
H_sub.setZero();
for (int iteration = 0; iteration < max_iterations; iteration++)
{
double t1 = omp_get_wtime();
M3D Rwi(state->rot_end);
V3D Pwi(state->pos_end);
Rcw = Rci * Rwi.transpose();
Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
Jdp_dt = Rci * Rwi.transpose();
cv::Mat render_for_update = img_render;
std::array<cv::Mat, 6> pose_render_jac;
bool use_pose_fd_jacobian = false;
double pose_fd_time = 0.0;
const bool pose_fd_window_small_enough =
    gs_pose_fd_max_gaussians <= 0 || sub_GSMap.size() <= static_cast<size_t>(gs_pose_fd_max_gaussians);
if (gs_pose_finite_diff_jacobian_en && !pose_fd_window_small_enough && iteration == 0)
{
std::cout << "[GS-EKF] pose finite-diff disabled for speed: sub_GSMap="
          << sub_GSMap.size()
          << ", max=" << gs_pose_fd_max_gaussians
          << std::endl;
}
if (gs_pose_finite_diff_jacobian_en && pose_fd_window_small_enough && scene && !sub_GSMap.empty())
{
const double pose_fd_start = omp_get_wtime();
cv::Mat base_render = renderGSImageAtState(*state);
if (!base_render.empty())
{
  render_for_update = toGray8UContinuous(base_render);
  if (!render_for_update.empty())
  {
cv::Mat base_render_float;
render_for_update.convertTo(base_render_float, CV_32F);
use_pose_fd_jacobian = true;

for (int fd_idx = 0; fd_idx < 6; ++fd_idx)
{
const double eps = fd_idx < 3 ? gs_pose_fd_rot_eps : gs_pose_fd_trans_eps;
if (eps <= 0.0)
{
use_pose_fd_jacobian = false;
break;
}
MD(DIM_STATE, 1) state_add;
state_add.setZero();
state_add(fd_idx, 0) = eps;
StatesGroup perturbed_state = (*state) + state_add;
cv::Mat perturbed_render = renderGSImageAtState(perturbed_state);
if (perturbed_render.empty())
{
use_pose_fd_jacobian = false;
break;
}
cv::Mat perturbed_gray;
  perturbed_gray = toGray8UContinuous(perturbed_render);
  if (perturbed_gray.empty() || perturbed_gray.size() != render_for_update.size())
  {
  use_pose_fd_jacobian = false;
break;
}
cv::Mat perturbed_float;
perturbed_gray.convertTo(perturbed_float, CV_32F);
pose_render_jac[fd_idx] = (perturbed_float - base_render_float) * (1.0 / eps);
}

updateGSCameraPoseFromState(*state);
pose_fd_time = omp_get_wtime() - pose_fd_start;
}
}
}
float error = 0.0;
int n_meas = 0;
// int max_threads = omp_get_max_threads();
// int desired_threads = std::min(max_threads, total_points);
// omp_set_num_threads(desired_threads);
#ifdef MP_EN
omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for reduction(+:error, n_meas)
#endif
for (int i = 0; i < total_points; i++)
{
// printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
MD(1, 2) Jimg;
MD(2, 3) Jdpi;
MD(1, 3) Jdphi, Jdp, JdR, Jdt;
float patch_error = 0.0;
int search_level = visual_submap->search_levels[i];
int pyramid_level = level + search_level;
int scale = (1 << pyramid_level);
float inv_scale = 1.0f / scale;
  VisualPoint *pt = visual_submap->voxel_points[i];
  if (pt == nullptr) continue;
  V3D pf = Rcw * pt->pos_ + Pcw;
  if (!pf.allFinite() || pf[2] <= 1e-6) continue;
  V2D pc = cam->world2cam(pf);
  if (!pc.allFinite()) continue;
  computeProjectionJacobian(pf, Jdpi);
M3D p_hat;
p_hat << SKEW_SYM_MATRX(pf);
float u_ref = pc[0];
float v_ref = pc[1];
int u_ref_i = floorf(pc[0] / scale) * scale;
int v_ref_i = floorf(pc[1] / scale) * scale;
const int pixel_margin = (patch_size_half + 2) * scale;
if (u_ref_i < pixel_margin || v_ref_i < pixel_margin || u_ref_i >= width - pixel_margin || v_ref_i >= height - pixel_margin) continue;
float subpix_u_ref = (u_ref - u_ref_i) / scale;
float subpix_v_ref = (v_ref - v_ref_i) / scale;
float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
float w_ref_br = subpix_u_ref * subpix_v_ref;
double inv_ref_expo = 1.0;  // Current-frame GS rendering has no reference exposure gap.
double inv_cur_expo = 1.0;
// ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);
for (int x = 0; x < patch_size; x++)
{
const int py = v_ref_i + x * scale - patch_size_half * scale;
const int px_start = u_ref_i - patch_size_half * scale;
uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
uint8_t *render_ptr = (uint8_t *)render_for_update.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
for (int y = 0; y < patch_size; ++y, img_ptr += scale, render_ptr += scale)
{
const int px = px_start + y * scale;
float du =
0.5f *
((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
w_ref_br * img_ptr[scale * width + scale * 2]) -
(w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
float dv =
0.5f *
((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
w_ref_br * img_ptr[width * scale * 2 + scale]) -
(w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));
float du2 =
0.5f *
((w_ref_tl * render_ptr[scale] + w_ref_tr * render_ptr[scale * 2] + w_ref_bl * render_ptr[scale * width + scale] +
w_ref_br * render_ptr[scale * width + scale * 2]) -
(w_ref_tl * render_ptr[-scale] + w_ref_tr * render_ptr[0] + w_ref_bl * render_ptr[scale * width - scale] + w_ref_br * render_ptr[scale * width]));
float dv2 =
0.5f *
((w_ref_tl * render_ptr[scale * width] + w_ref_tr * render_ptr[scale + scale * width] + w_ref_bl * render_ptr[width * scale * 2] +
w_ref_br * render_ptr[width * scale * 2 + scale]) -
(w_ref_tl * render_ptr[-scale * width] + w_ref_tr * render_ptr[-scale * width + scale] + w_ref_bl * render_ptr[0] + w_ref_br * render_ptr[scale]));
if (gs_render_jacobian_en)
{
Jimg << -inv_ref_expo * du2, -inv_ref_expo * dv2;
}
else
{
Jimg << inv_cur_expo * du - inv_ref_expo * du2, inv_cur_expo * dv - inv_ref_expo * dv2;
}
Jimg = Jimg * inv_scale;
Jdphi = Jimg * Jdpi * p_hat;
Jdp = -Jimg * Jdpi;
JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
Jdt = Jdp * Jdp_dt;
double cur_value =
w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
double ref_value =
w_ref_tl * render_ptr[0] + w_ref_tr * render_ptr[scale] + w_ref_bl * render_ptr[scale * width] + w_ref_br * render_ptr[scale * width + scale];
const bool background_like = gs_white_background ? (ref_value > 250.0) : (ref_value < 5.0);
if (background_like) continue;  // No rendering, skip
double res = inv_cur_expo * cur_value - inv_ref_expo * ref_value;
z(i * patch_size_total + x * patch_size + y) = res;
patch_error += res * res;
n_meas += 1;
const int row_idx = i * patch_size_total + x * patch_size + y;
if (use_pose_fd_jacobian)
{
const double h0 = -inv_ref_expo * pose_render_jac[0].at<float>(py, px);
const double h1 = -inv_ref_expo * pose_render_jac[1].at<float>(py, px);
const double h2 = -inv_ref_expo * pose_render_jac[2].at<float>(py, px);
const double h3 = -inv_ref_expo * pose_render_jac[3].at<float>(py, px);
const double h4 = -inv_ref_expo * pose_render_jac[4].at<float>(py, px);
const double h5 = -inv_ref_expo * pose_render_jac[5].at<float>(py, px);
if (gs_exposure_update_en) { H_sub.block<1, 7>(row_idx, 0) << h0, h1, h2, h3, h4, h5, cur_value; }
else { H_sub.block<1, 6>(row_idx, 0) << h0, h1, h2, h3, h4, h5; }
}
else
{
if (gs_exposure_update_en) { H_sub.block<1, 7>(row_idx, 0) << JdR, Jdt, cur_value; }
else { H_sub.block<1, 6>(row_idx, 0) << JdR, Jdt; }
}
}
}
visual_submap->errors[i] = patch_error;
error += patch_error;
}
if (n_meas == 0)
{
std::cout << "[GS-EKF] update abort: no valid rendered patch pixels at iteration "
          << iteration << std::endl;
(*state) = old_state;
G.setZero();
return false;
}
if (gs_pose_update_min_measurements > 0 && n_meas < gs_pose_update_min_measurements)
{
std::cout << "[GS-EKF] update abort: not enough rendered patch pixels, n_meas="
          << n_meas << ", min=" << gs_pose_update_min_measurements
          << ", iteration=" << iteration << std::endl;
(*state) = old_state;
G.setZero();
return false;
}
error = error / n_meas;
const double rmse = std::sqrt(error);
gs_last_update_measurements = n_meas;
gs_last_update_rmse = rmse;
if ((gs_pose_update_max_rmse > 0.0 && rmse > gs_pose_update_max_rmse) || !std::isfinite(rmse))
{
std::cout << "[GS-EKF] update abort: photometric rmse gate, rmse=" << rmse
          << ", max=" << gs_pose_update_max_rmse
          << ", n_meas=" << n_meas
          << ", iteration=" << iteration << std::endl;
(*state) = old_state;
G.setZero();
return false;
}
if (pose_fd_time > 0.0)
{
std::cout << "[GS-EKF] pose finite-diff jacobian time=" << pose_fd_time
          << ", sub_GSMap=" << sub_GSMap.size()
          << ", iteration=" << iteration
          << ", active=" << (use_pose_fd_jacobian ? "true" : "false")
          << std::endl;
}
compute_jacobian_time += omp_get_wtime() - t1;
// printf("\nPYRAMID LEVEL %i\n---------------\n", level);
// std::cout << "It. " << iteration
// << "\t last_error = " << last_error
// << "\t new_error = " << error
// << std::endl;
double t3 = omp_get_wtime();
if (error <= last_error)
{
old_state = (*state);
last_error = error;
auto &&H_sub_T = H_sub.transpose();
H_T_H.setZero();
G.setZero();
H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
if (!K_1.allFinite())
{
std::cout << "[GS-EKF] update abort: non-finite gain matrix" << std::endl;
(*state) = old_state;
G.setZero();
return false;
}
auto &&HTz = H_sub_T * z;
auto vec = (*state_propagat) - (*state);
G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
MD(DIM_STATE, 1)
solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);
if (!gs_exposure_update_en) solution(6, 0) = 0.0;
if (!solution.allFinite())
{
std::cout << "[GS-EKF] update abort: non-finite solution" << std::endl;
(*state) = old_state;
G.setZero();
return false;
}
const double rot_step_deg = solution.block<3, 1>(0, 0).norm() * 57.29577951308232;
const double trans_step = solution.block<3, 1>(3, 0).norm();
const bool rot_step_too_large = gs_max_pose_update_rot_deg > 0.0 && rot_step_deg > gs_max_pose_update_rot_deg;
const bool trans_step_too_large = gs_max_pose_update_trans > 0.0 && trans_step > gs_max_pose_update_trans;
if (!std::isfinite(rot_step_deg) || !std::isfinite(trans_step) || rot_step_too_large || trans_step_too_large)
{
const bool raw_rot_too_large =
    gs_pose_update_max_raw_rot_deg > 0.0 && rot_step_deg > gs_pose_update_max_raw_rot_deg;
const bool raw_trans_too_large =
    gs_pose_update_max_raw_trans > 0.0 && trans_step > gs_pose_update_max_raw_trans;
if (!std::isfinite(rot_step_deg) || !std::isfinite(trans_step) ||
    raw_rot_too_large || raw_trans_too_large || gs_pose_update_step_damping <= 0.0)
{
std::cout << "[GS-EKF] update abort: step too large, rot_deg=" << rot_step_deg
          << " (max=" << gs_max_pose_update_rot_deg
          << ", raw_max=" << gs_pose_update_max_raw_rot_deg
          << "), trans=" << trans_step
          << " (max=" << gs_max_pose_update_trans
          << ", raw_max=" << gs_pose_update_max_raw_trans << ")" << std::endl;
(*state) = old_state;
G.setZero();
return false;
}

double step_scale = std::min(1.0, gs_pose_update_step_damping);
if (rot_step_too_large && rot_step_deg > 1e-12)
{
step_scale = std::min(step_scale, gs_max_pose_update_rot_deg / rot_step_deg);
}
if (trans_step_too_large && trans_step > 1e-12)
{
step_scale = std::min(step_scale, gs_max_pose_update_trans / trans_step);
}
step_scale *= std::min(1.0, gs_pose_update_step_damping);
if (!std::isfinite(step_scale) || step_scale <= 0.0)
{
std::cout << "[GS-EKF] update abort: invalid damping scale=" << step_scale << std::endl;
(*state) = old_state;
G.setZero();
return false;
}

solution *= step_scale;
G *= step_scale;
std::cout << "[GS-EKF] step damped: raw_rot_deg=" << rot_step_deg
          << ", raw_trans=" << trans_step
          << ", scale=" << step_scale
          << ", applied_rot_deg="
          << solution.block<3, 1>(0, 0).norm() * 57.29577951308232
          << ", applied_trans=" << solution.block<3, 1>(3, 0).norm()
          << std::endl;
EKF_end = true;
}
(*state) += solution;
has_valid_update = true;
// if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
const double applied_rot_step_deg = solution.block<3, 1>(0, 0).norm() * 57.29577951308232;
const double applied_trans_step = solution.block<3, 1>(3, 0).norm();
if ((applied_rot_step_deg < 0.001) && (applied_trans_step * 100.0 < 0.001)) EKF_end = true;
}
else
{
(*state) = old_state;
G.setZero();
return false;
}
update_ekf_time += omp_get_wtime() - t3;
if (iteration == max_iterations || EKF_end) break;
}
// if (state->inv_expo_time < 0.0) {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
if (!has_valid_update) G.setZero();
return has_valid_update;
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void VIOManager::plotTrackedPoints()
{
  if (visual_submap == nullptr || img_cp.empty()) return;
  const size_t total_points = std::min(
      visual_submap->voxel_points.size(),
      std::min(visual_submap->errors.size(), visual_submap->propa_errors.size()));
  if (total_points == 0) return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (size_t i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (!isFiniteVisualPointForVio(pt, false)) continue;
    V2D pc(new_frame_->w2c(pt->pos_));
    if (!pc.allFinite() || pc[0] < 0.0 || pc[1] < 0.0 ||
        pc[0] >= img_cp.cols || pc[1] >= img_cp.rows)
    {
      continue;
    }

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(const cv::Mat &img, V2D pc)
{
  if (img.empty() || img.depth() != CV_8U || !pc.allFinite()) return V3F::Zero();

  const int channels = img.channels();
  if (channels != 1 && channels != 3 && channels != 4) return V3F::Zero();

  const float u_ref = static_cast<float>(pc[0]);
  const float v_ref = static_cast<float>(pc[1]);
  const int u_ref_i = floorf(u_ref);
  const int v_ref_i = floorf(v_ref);
  if (u_ref_i < 0 || v_ref_i < 0 || u_ref_i >= img.cols - 1 || v_ref_i >= img.rows - 1)
  {
    return V3F::Zero();
  }

  const float subpix_u_ref = u_ref - u_ref_i;
  const float subpix_v_ref = v_ref - v_ref_i;
  const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
  const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;

  const uint8_t *row_top = img.ptr<uint8_t>(v_ref_i);
  const uint8_t *row_bottom = img.ptr<uint8_t>(v_ref_i + 1);
  const uint8_t *p_tl = row_top + u_ref_i * channels;
  const uint8_t *p_tr = p_tl + channels;
  const uint8_t *p_bl = row_bottom + u_ref_i * channels;
  const uint8_t *p_br = p_bl + channels;

  auto interpolate_channel = [&](int channel_idx) {
    return w_ref_tl * p_tl[channel_idx] +
           w_ref_tr * p_tr[channel_idx] +
           w_ref_bl * p_bl[channel_idx] +
           w_ref_br * p_br[channel_idx];
  };

  if (channels == 1)
  {
    const float gray = interpolate_channel(0);
    return V3F(gray, gray, gray);
  }

  return V3F(interpolate_channel(0), interpolate_channel(1), interpolate_channel(2));
}



cv::Mat VIOManager::tensor_to_mat4(torch::Tensor tensor) {
    // Convert from float [0, 1] to uint8 [0, 255]

    if (tensor.device().type() != torch::kCPU) {
        tensor = tensor.to(torch::kCPU);  // Move local copy to CPU
    }
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);

    // Permute dimensions back from [channels, height, width] to [height, width, channels]
    tensor = tensor.permute({1, 2, 0}).contiguous();

    // Get the height, width, and channels
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);
      // Debug output
    // std::cout << "Tensor dimensions - Height: " << height << ", Width: " << width << ", Channels: " << channels << std::endl;


    if (channels != 3) {
        throw std::runtime_error("Expected 3 channels for RGB image");
    }

    cv::Mat rgb(height, width, CV_8UC3, tensor.data_ptr());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    return bgr.clone();
}

void VIOManager::updateGSCameraPoseFromState(const StatesGroup &render_state)
{
  if (!scene) return;

  const M3D Rwi(render_state.rot_end);
  const V3D Pwi(render_state.pos_end);
  const M3D render_Rcw = Rci * Rwi.transpose();
  const V3D render_Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  scene->UpdateFirstCameraPose(render_Rcw.transpose().cast<float>(), render_Pcw.cast<float>());
}

cv::Mat VIOManager::renderGSImageAtState(const StatesGroup &render_state)
{
  if (!scene || sub_GSMap.empty()) return cv::Mat();

  updateGSCameraPoseFromState(render_state);
  auto &render_cam = scene->Get_training_camera(0);

  torch::NoGradGuard no_grad;
  auto [image, viewspace_point_tensor, visibility_filter, radii] = render(render_cam, gaussians, background);
  return tensor_to_mat4(image);
}

std::filesystem::path VIOManager::resolveGSOutputDir() const
{
  std::filesystem::path output_dir = gs_output_dir.empty()
      ? std::filesystem::path(ROOT_DIR) / "Log" / "GS"
      : std::filesystem::path(gs_output_dir);
  if (output_dir.is_relative()) output_dir = std::filesystem::path(ROOT_DIR) / output_dir;
  return output_dir;
}

std::vector<GS_point> VIOManager::collectGSMapSnapshot()
{
  std::vector<GS_point> snapshot;
  if (gsmap_manager)
  {
    for (auto &entry : gsmap_manager->gs_map_)
    {
      if (entry.second == nullptr) continue;
      std::vector<GS_point*> points;
      entry.second->get_all_gs_points(points);
      snapshot.reserve(snapshot.size() + points.size());
      for (GS_point *point : points)
      {
        if (point != nullptr) snapshot.push_back(*point);
      }
    }
  }

  if (snapshot.empty()) snapshot = sub_GSMap;
  return snapshot;
}

void VIOManager::writeBackOptimizedSubGSMap()
{
  const size_t update_size = std::min(sub_GSMap.size(), sub_GSMap_ptrs.size());
  size_t updated = 0;
  for (size_t i = 0; i < update_size; ++i)
  {
    if (sub_GSMap_ptrs[i] == nullptr) continue;
    sub_GSMap[i].index = -1;
    *sub_GSMap_ptrs[i] = sub_GSMap[i];
    sub_GSMap_ptrs[i]->index = -1;
    updated++;
  }

  std::cout << "[GS] write-back optimized submap: updated=" << updated
            << ", sub_GSMap=" << sub_GSMap.size()
            << ", ptrs=" << sub_GSMap_ptrs.size()
            << std::endl;
}

void VIOManager::prepareGSPhotometricMeasurements(const cv::Mat &img, const cv::Mat &rendered, const vector<pointWithVar> &pg)
{
  gs_measurement_points.clear();
  if (visual_submap != nullptr) visual_submap->reset();
  total_points = 0;

  const bool use_gaussian_window = !sub_GSMap.empty();
  if (visual_submap == nullptr || img.empty() || rendered.empty() || (!use_gaussian_window && pg.empty()))
  {
    std::cout << "[GS-EKF] prepare skip: visual_submap_null="
              << (visual_submap == nullptr ? "true" : "false")
              << ", img_empty=" << (img.empty() ? "true" : "false")
              << ", rendered_empty=" << (rendered.empty() ? "true" : "false")
              << ", sub_GSMap=" << sub_GSMap.size()
              << ", pg=" << pg.size()
              << std::endl;
    return;
  }

  cv::Mat img_gray = toGray8UContinuous(img);
  cv::Mat render_gray = toGray8UContinuous(rendered);
  if (img_gray.empty() || render_gray.empty())
  {
    std::cout << "[GS-EKF] prepare skip: unsupported image format, img_channels="
              << img.channels() << ", rendered_channels=" << rendered.channels()
              << std::endl;
    return;
  }

  if (img_gray.size() != render_gray.size())
  {
    std::cout << "[GS-EKF] prepare skip: image size mismatch img="
              << img_gray.cols << "x" << img_gray.rows
              << ", rendered=" << render_gray.cols << "x" << render_gray.rows
              << std::endl;
    return;
  }

  const int cell_size = std::max(1, grid_size);
  const int cols = std::max(1, static_cast<int>(std::ceil(static_cast<double>(width) / cell_size)));
  const int rows = std::max(1, static_cast<int>(std::ceil(static_cast<double>(height) / cell_size)));
  const int margin = std::max(3, patch_size_half + 2);

  struct MeasurementCandidate
  {
    bool valid = false;
    V3D point_w = V3D::Zero();
    V3D normal = V3D::Zero();
    double depth = std::numeric_limits<double>::max();
    double score = -1.0;
  };

  std::vector<MeasurementCandidate> best_candidates(cols * rows);
  size_t candidates = 0;
  size_t reject_behind = 0;
  size_t reject_frame = 0;
  size_t reject_background = 0;
  size_t reject_invalid = 0;

  auto add_candidate = [&](const V3D &point_w, const V3D &normal) {
    if (!point_w.allFinite() || !isFiniteNormal(normal))
    {
      reject_invalid++;
      return;
    }
    V3D pf = new_frame_->w2f(point_w);
    if (!pf.allFinite() || pf[2] <= 0)
    {
      reject_behind++;
      return;
    }

    V2D pc = new_frame_->w2c(point_w);
    if (!pc.allFinite())
    {
      reject_frame++;
      return;
    }
    const int u = static_cast<int>(std::lround(pc[0]));
    const int v = static_cast<int>(std::lround(pc[1]));
    if (u < margin || v < margin || u >= width - margin || v >= height - margin)
    {
      reject_frame++;
      return;
    }

    const uchar render_value = render_gray.at<uchar>(v, u);
    const bool background_like = gs_white_background ? (render_value > 250) : (render_value < 5);
    if (background_like)
    {
      reject_background++;
      return;
    }

    const uchar img_value = img_gray.at<uchar>(v, u);
    const double residual = std::abs(static_cast<double>(img_value) - static_cast<double>(render_value));
    const double grad_u = std::abs(static_cast<double>(render_gray.at<uchar>(v, u + 1)) - static_cast<double>(render_gray.at<uchar>(v, u - 1)));
    const double grad_v = std::abs(static_cast<double>(render_gray.at<uchar>(v + 1, u)) - static_cast<double>(render_gray.at<uchar>(v - 1, u)));
    const double score = residual + 0.5 * (grad_u + grad_v);
    if (!std::isfinite(score))
    {
      reject_invalid++;
      return;
    }

    const int col = std::clamp(u / cell_size, 0, cols - 1);
    const int row = std::clamp(v / cell_size, 0, rows - 1);
    MeasurementCandidate &candidate = best_candidates[row * cols + col];
    if (!candidate.valid || score > candidate.score || (score == candidate.score && pf[2] < candidate.depth))
    {
      candidate.valid = true;
      candidate.point_w = point_w;
      candidate.normal = normal;
      candidate.depth = pf[2];
      candidate.score = score;
    }
    candidates++;
  };

  if (use_gaussian_window)
  {
    for (const GS_point &point : sub_GSMap)
    {
      const V3D point_w(point._points.x, point._points.y, point._points.z);
      const V3D normal(point._normals.x, point._normals.y, point._normals.z);
      add_candidate(point_w, normal);
    }
  }
  else
  {
    for (const auto &point : pg)
    {
      add_candidate(point.point_w, point.normal);
    }
  }

  for (const MeasurementCandidate &candidate : best_candidates)
  {
    if (!candidate.valid) continue;
    auto visual_point = std::make_unique<VisualPoint>(candidate.point_w);
    visual_point->normal_ = candidate.normal;
    visual_point->is_normal_initialized_ = true;

    VisualPoint *visual_point_ptr = visual_point.get();
    gs_measurement_points.push_back(std::move(visual_point));
    visual_submap->voxel_points.push_back(visual_point_ptr);
    visual_submap->propa_errors.push_back(0.0f);
    visual_submap->errors.push_back(0.0f);
    visual_submap->search_levels.push_back(0);
    visual_submap->warp_patch.emplace_back();
    visual_submap->inv_expo_list.push_back(1.0);
  }

  total_points = static_cast<int>(visual_submap->voxel_points.size());
  std::cout << "[GS-EKF] prepare measurements: source="
            << (use_gaussian_window ? "gaussian_window" : "lidar_fallback")
            << ", sub_GSMap=" << sub_GSMap.size()
            << ", pg=" << pg.size()
            << ", candidates=" << candidates
            << ", selected=" << total_points
            << ", reject_behind=" << reject_behind
            << ", reject_frame=" << reject_frame
            << ", reject_background=" << reject_background
            << ", reject_invalid=" << reject_invalid
            << ", grid=" << cols << "x" << rows
            << ", margin=" << margin
            << std::endl;
}

void VIOManager::writeGSPointPly(const std::filesystem::path &file_path, const std::vector<GS_point> &points) const
{
  std::filesystem::create_directories(file_path.parent_path());
  std::ofstream out(file_path);
  if (!out.is_open())
  {
    std::cerr << "[GS] failed to open PLY for writing: " << file_path << std::endl;
    return;
  }

  out << "ply\n";
  out << "format ascii 1.0\n";
  out << "element vertex " << points.size() << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property float nx\n";
  out << "property float ny\n";
  out << "property float nz\n";
  out << "property float scale_0\n";
  out << "property float scale_1\n";
  out << "property float scale_2\n";
  out << "property float rot_0\n";
  out << "property float rot_1\n";
  out << "property float rot_2\n";
  out << "property float rot_3\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  out << "property float opacity\n";
  out << "end_header\n";

  out << std::fixed << std::setprecision(6);
  for (const GS_point &point : points)
  {
    const int red = std::clamp(static_cast<int>(std::lround(point._colors.r)), 0, 255);
    const int green = std::clamp(static_cast<int>(std::lround(point._colors.g)), 0, 255);
    const int blue = std::clamp(static_cast<int>(std::lround(point._colors.b)), 0, 255);
    out << point._points.x << " " << point._points.y << " " << point._points.z << " "
        << point._normals.x << " " << point._normals.y << " " << point._normals.z << " "
        << point._distance.r1 << " " << point._distance.r2 << " " << point._distance.r3 << " "
        << point._quaternion.qw << " " << point._quaternion.qx << " "
        << point._quaternion.qy << " " << point._quaternion.qz << " "
        << red << " " << green << " " << blue << " "
        << point._opacity << "\n";
  }
}

void VIOManager::saveGSResults(int frame_id, double img_time)
{
  if (!gs_save_results) return;

  const std::filesystem::path output_dir = resolveGSOutputDir();
  std::ostringstream frame_name;
  frame_name << "frame_" << std::setw(6) << std::setfill('0') << frame_id
             << "_t_" << std::fixed << std::setprecision(6) << img_time;

  if (gs_save_rendered_images && !img_rendered.empty())
  {
    const std::filesystem::path rendered_dir = output_dir / "rendered";
    std::filesystem::create_directories(rendered_dir);
    const std::filesystem::path image_path = rendered_dir / (frame_name.str() + ".png");
    cv::imwrite(image_path.string(), img_rendered);

    if (gs_save_gt_images && !img_undistort.empty())
    {
      const std::filesystem::path gt_dir = output_dir / "gt";
      std::filesystem::create_directories(gt_dir);
      cv::imwrite((gt_dir / (frame_name.str() + ".png")).string(), img_undistort);
    }

    cv::Mat gray;
    if (img_rendered.channels() == 3)
      cv::cvtColor(img_rendered, gray, cv::COLOR_BGR2GRAY);
    else
      gray = img_rendered;

    cv::Mat white_mask;
    cv::threshold(gray, white_mask, 245, 255, cv::THRESH_BINARY);
    const double white_ratio = gray.empty() ? 0.0 : static_cast<double>(cv::countNonZero(white_mask)) / static_cast<double>(gray.total());
    const double mean_intensity = gray.empty() ? 0.0 : cv::mean(gray)[0];
    double mean_absdiff = -1.0;
    if (!img_undistort.empty() && img_undistort.size() == img_rendered.size() && img_undistort.type() == img_rendered.type())
    {
      cv::Mat abs_diff;
      cv::absdiff(img_rendered, img_undistort, abs_diff);
      const cv::Scalar diff_mean = cv::mean(abs_diff);
      mean_absdiff = (diff_mean[0] + diff_mean[1] + diff_mean[2]) / 3.0;
    }
    std::cout << "[GS] saved rendered image: " << image_path
              << ", mean_intensity=" << mean_intensity
              << ", white_ratio=" << white_ratio
              << ", mean_absdiff_to_gt=" << mean_absdiff
              << std::endl;
  }

  if (save_GS_iter > 0 && frame_id % save_GS_iter == 0)
  {
    const std::vector<GS_point> snapshot = collectGSMapSnapshot();
    if (!snapshot.empty())
    {
      const std::filesystem::path map_path = output_dir / "map" / (frame_name.str() + ".ply");
      writeGSPointPly(map_path, snapshot);
      std::cout << "[GS] saved GS map snapshot: " << map_path
                << ", points=" << snapshot.size()
                << std::endl;
    }
  }
}






void VIOManager::dumpDataForColmap()
{
  if (pinhole_cam == nullptr || img_rgb.empty() || new_frame_ == nullptr)
  {
    std::cout << "[ VIO ] skip Colmap dump: invalid image or pinhole camera" << std::endl;
    return;
  }

  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}


void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  img_rendered.release();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();

  retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  const bool vio_update_accepted = computeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();
  double t4 = t3;
  double t5 = t3;
  double t6 = t3;
  double t7 = t3;

  if (vio_update_accepted)
  {
    generateVisualMapPoints(img, pg);
    t4 = omp_get_wtime();

    plotTrackedPoints();

    if (plot_flag) projectPatchFromRefToCur(feat_map);

    t5 = omp_get_wtime();

    updateVisualMapPoints(img);

    t6 = omp_get_wtime();

    updateReferencePatch(feat_map);

    t7 = omp_get_wtime();

    if(colmap_output_en)  dumpDataForColmap();
  }
  else
  {
    std::cout << "[VIO-EKF] skip visual map maintenance for rejected update" << std::endl;
  }

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
  // cv::imwrite("/home/chunran/Desktop/raycasting/" + std::to_string(new_frame_->id_) + ".png", img_cp);
}

void VIOManager::processFrameGS(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  static int gs_frame_id = 0;
  gs_frame_id++;
  gs_frame_count = gs_frame_id;

  if (img.empty())
  {
    img_rgb.release();
    img_cp.release();
    img_undistort.release();
    img_rendered.release();
    total_points = 0;
    sub_GSMap.clear();
    sub_GSMap_ptrs.clear();
    std::cout << "[GS] frame=" << gs_frame_id << " skip: empty image" << std::endl;
    return;
  }

  if (width <= 1 || height <= 1 || !std::isfinite(image_resize_factor) || image_resize_factor <= 0.0)
  {
    img_rgb.release();
    img_cp.release();
    img_undistort.release();
    img_rendered.release();
    new_frame_.reset();
    total_points = 0;
    std::cout << "[GS] frame=" << gs_frame_id
              << " skip: invalid image geometry, width=" << width
              << ", height=" << height
              << ", scale=" << image_resize_factor
              << std::endl;
    return;
  }

  if (width != img.cols || height != img.rows)
  {
    cv::resize(img, img, cv::Size(width, height), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  img_rendered.release();
  // img_test = img.clone();

  if (img.channels() == 4)
  {
    cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGRA2BGR);
    cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
    img_cp = img_rgb.clone();
  }
  else if (img.channels() == 3)
  {
    cv::cvtColor(img, img, CV_BGR2GRAY);
  }
  else if (img.channels() != 1)
  {
    img_rgb.release();
    img_cp.release();
    img_undistort.release();
    img_rendered.release();
    new_frame_.reset();
    total_points = 0;
    std::cout << "[GS] frame=" << gs_frame_id
              << " skip: unsupported image channels=" << img.channels()
              << std::endl;
    return;
  }

// sheng 去畸变
  pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
  if (pinhole_cam != nullptr)
  {
    pinhole_cam->undistortImage(img_rgb, img_undistort);
    img_undistort = img_undistort.clone();
  }
  else
  {
    img_undistort = img_rgb.clone();
    std::cout << "[GS] frame=" << gs_frame_id
              << " use raw image for GS color: camera is not PinholeCamera"
              << std::endl;
  }

  new_frame_.reset(new Frame(cam, img));
  if (!isFiniteStateForVio(state))
  {
    img_rgb.release();
    img_cp.release();
    img_undistort.release();
    img_rendered.release();
    new_frame_.reset();
    total_points = 0;
    std::cout << "[GS] frame=" << gs_frame_id
              << " skip: non-finite input state"
              << std::endl;
    return;
  }
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();
  double t_after_sparse_vio = t1;

  std::cout << "[GS] frame=" << gs_frame_id
            << " begin: pg=" << pg.size()
            << ", gs_total=" << gs_total
            << ", map_voxels=" << gsmap_manager->gs_map_.size()
            << ", sub_GSMap_before=" << sub_GSMap.size()
            << ", total_points=" << total_points
            << ", sparse_vio_fallback=" << (gs_sparse_vio_fallback_en ? "on" : "off")
            << std::endl;

  if (gs_sparse_vio_fallback_en)
  {
    const double t_sparse_start = omp_get_wtime();
    retrieveFromVisualSparseMap(img, pg, feat_map);
    const double t_sparse_retrieve = omp_get_wtime();
    const bool vio_update_accepted = computeJacobianAndUpdateEKF(img);
    const double t_sparse_ekf = omp_get_wtime();
    double t_sparse_generate = t_sparse_ekf;
    double t_sparse_plot = t_sparse_ekf;
    double t_sparse_update = t_sparse_ekf;
    if (vio_update_accepted)
    {
      generateVisualMapPoints(img, pg);
      t_sparse_generate = omp_get_wtime();
      plotTrackedPoints();
      if (plot_flag) projectPatchFromRefToCur(feat_map);
      t_sparse_plot = omp_get_wtime();
      updateVisualMapPoints(img);
      t_sparse_update = omp_get_wtime();
      updateReferencePatch(feat_map);
    }
    else
    {
      std::cout << "[VIO-EKF] skip sparse visual map maintenance for rejected update" << std::endl;
    }
    t_after_sparse_vio = omp_get_wtime();

    std::cout << "[GS] frame=" << gs_frame_id
              << " sparse VIO fallback: retrieve=" << (t_sparse_retrieve - t_sparse_start)
              << ", ekf=" << (t_sparse_ekf - t_sparse_retrieve)
              << ", generate=" << (t_sparse_generate - t_sparse_ekf)
              << ", plot=" << (t_sparse_plot - t_sparse_generate)
              << ", update=" << (t_sparse_update - t_sparse_plot)
              << ", reference=" << (t_after_sparse_vio - t_sparse_update)
              << ", total=" << (t_after_sparse_vio - t_sparse_start)
              << std::endl;
  }

  retrieveFrom_GS_Map2(pg);
  const double t_after_retrieve = omp_get_wtime();

  std::cout << "[GS] frame=" << gs_frame_id
            << " after retrieve: sub_GSMap=" << sub_GSMap.size()
            << ", map_voxels=" << gsmap_manager->gs_map_.size()
            << ", total_points=" << total_points
            << std::endl;

  const double t_render_start = omp_get_wtime();
  if(sub_GSMap.size()>0)
  {
    std::cout << "[GS] frame=" << gs_frame_id
              << " render start: gaussian_count=" << sub_GSMap.size()
              << ", iterations=" << gs_params.iterations
              << std::endl;

    gaussians.Create_from_our_format(sub_GSMap);
    gaussians.Training_setup(gs_params);
    cv::Mat gs_training_image;
    if (img_undistort.channels() == 3)
      cv::cvtColor(img_undistort, gs_training_image, cv::COLOR_BGR2RGB);
    else if (img_undistort.channels() == 1)
      cv::cvtColor(img_undistort, gs_training_image, cv::COLOR_GRAY2RGB);
    else
      gs_training_image = img_undistort;
    scene->_scene_infos->_cameras[0]._img_data= gs_training_image.data;
    scene->UpdateFirstCameraImage();
    scene->UpdateFirstCameraPose(Rcw.transpose().cast<float>(), (Pcw).cast<float>());
    auto& cam = (*scene).Get_training_camera(0);
    int optimizer_steps = 0;
    int skipped_backward = 0;
    for(int i=0;i<gs_params.iterations;i++)
    {
      
      auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam,  gaussians, background);
      auto gt_image = cam.Get_original_image().to(torch::kCUDA, true);
      auto l1l = gaussian_splatting::l1_loss(image, gt_image).to(torch::kCUDA, true);
      torch::Tensor loss = l1l;
      const int64_t visible_gaussians = visibility_filter.sum().item<int64_t>();
      const bool image_requires_grad = image.requires_grad();
      const bool loss_requires_grad = loss.requires_grad();

      if (visible_gaussians == 0 || !loss_requires_grad)
      {
        skipped_backward++;
        if (i == 0 || i == gs_params.iterations - 1)
        {
          std::cout << "[GS] frame=" << gs_frame_id
                    << " skip optimizer: iter=" << i
                    << ", visible_gaussians=" << visible_gaussians
                    << ", image_requires_grad=" << (image_requires_grad ? "true" : "false")
                    << ", loss_requires_grad=" << (loss_requires_grad ? "true" : "false")
                    << std::endl;
        }

        if(i==(gs_params.iterations-1))
        {
          float loss_value = loss.item<float>();
          cv::Mat image_mat = tensor_to_mat4(image);
          img_rendered=image_mat.clone();
          std::cout << "[GS] frame=" << gs_frame_id
                    << " render done: loss=" << loss_value
                    << ", rendered_size=" << img_rendered.cols << "x" << img_rendered.rows
                    << ", visible_gaussians=" << visible_gaussians
                    << ", optimizer_steps=" << optimizer_steps
                    << ", skipped_backward=" << skipped_backward
                    << ", loss_requires_grad=" << (loss_requires_grad ? "true" : "false")
                    << std::endl;
          if (plot_gs_render)
          {
            cv::imshow("Rendered Image", img_rendered);
            cv::imshow("img_undistort k", img_undistort);
            cv::waitKey(1);
          }
        }
        continue;
      }

      loss.backward();
      if (visible_gaussians > 0)
      {
        auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
        auto visible_radii = radii.masked_select(visibility_filter);
        auto max_radii = torch::max(visible_max_radii, visible_radii);
        gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);
      }
      gaussians._optimizer->step();
      gaussians._optimizer->zero_grad(true);
      optimizer_steps++;
      if(i==(gs_params.iterations-1))
      {
        float loss_value = loss.item<float>();
        cv::Mat image_mat = tensor_to_mat4(image);
        img_rendered=image_mat.clone();
        std::cout << "[GS] frame=" << gs_frame_id
                  << " render done: loss=" << loss_value
                  << ", rendered_size=" << img_rendered.cols << "x" << img_rendered.rows
                  << ", visible_gaussians=" << visible_gaussians
                  << ", optimizer_steps=" << optimizer_steps
                  << ", skipped_backward=" << skipped_backward
                  << ", loss_requires_grad=" << (loss_requires_grad ? "true" : "false")
                  << std::endl;
        if (plot_gs_render)
        {
          cv::imshow("Rendered Image", img_rendered);
          cv::imshow("img_undistort k", img_undistort);
          cv::waitKey(1);
        }
      }
    }
    gaussians.Dump_to_our_format(sub_GSMap, static_cast<int>(sub_GSMap.size()));
    writeBackOptimizedSubGSMap();
  }
  else
  {
    std::cout << "[GS] frame=" << gs_frame_id
              << " skip render: sub_GSMap is empty" << std::endl;
  }
  const double t_after_render = omp_get_wtime();



  prepareGSPhotometricMeasurements(img, img_rendered, pg);
  const double t_after_prepare = omp_get_wtime();
  std::cout << "[GS] frame=" << gs_frame_id
            << " before GS-EKF: total_points=" << total_points
            << ", img_rendered_empty=" << (img_rendered.empty() ? "true" : "false")
            << std::endl;
  computeJacobianAndUpdateEKF_GS(img,img_rendered);
  const double t_after_ekf = omp_get_wtime();

  insertPointInto_GS_Map2(pg);
  const double t_after_insert = omp_get_wtime();
  pruneGSMapIfNeeded();
  const double t_after_prune = omp_get_wtime();
  saveGSResults(gs_frame_id, img_time);
  const double t_after_save = omp_get_wtime();

  std::cout << "[GS-TIME] frame=" << gs_frame_id
            << ", retrieve=" << (t_after_retrieve - t_after_sparse_vio)
            << ", render_opt=" << (t_after_render - t_render_start)
            << ", prepare_meas=" << (t_after_prepare - t_after_render)
            << ", gs_ekf=" << (t_after_ekf - t_after_prepare)
            << ", insert=" << (t_after_insert - t_after_ekf)
            << ", prune=" << (t_after_prune - t_after_insert)
            << ", save=" << (t_after_save - t_after_prune)
            << ", total=" << (t_after_save - t1)
            << ", sparse_vio=" << (t_after_sparse_vio - t1)
            << ", gaussian_count=" << sub_GSMap.size()
            << ", gs_iterations=" << gs_params.iterations
            << ", sparse_vio_fallback=" << (gs_sparse_vio_fallback_en ? "on" : "off")
            << ", pose_update=" << (gs_pose_update_en ? "on" : "off")
            << ", pose_fd=" << (gs_pose_finite_diff_jacobian_en ? "on" : "off")
            << std::endl;
}
