#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "read_utils.cuh"
#include <exception>
#include <thread>

GaussianModel::GaussianModel(void){
    _max_sh_degree=0;
}
GaussianModel::GaussianModel(int sh_degree) : _max_sh_degree(sh_degree) {
}

torch::Tensor GaussianModel::Get_covariance(float scaling_modifier) {
    auto L = build_scaling_rotation(scaling_modifier * Get_scaling(), _rotation);
    auto actual_covariance = torch::mm(L, L.transpose(1, 2));
    auto symm = strip_symmetric(actual_covariance);
    return symm;
}

/**
 * @brief Fetches the features of the Gaussian model
 *
 * This function concatenates _features_dc and _features_rest along the second dimension.
 *
 * @return Tensor of the concatenated features
 */
torch::Tensor GaussianModel::Get_features() const {
    auto features_dc = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
}

/**
 * @brief Increment the SH degree by 1
 *
 * This function increments the active_sh_degree by 1, up to a maximum of max_sh_degree.
 */
void GaussianModel::One_up_sh_degree() {
    if (_active_sh_degree < _max_sh_degree) {
        _active_sh_degree++;
    }
}

/**
 * @brief Initialize Gaussian Model from a Point Cloud.
 *
 * This function creates a Gaussian model from a given PointCloud object. It also sets
 * the spatial learning rate scale. The model's features, scales, rotations, and opacities
 * are initialized based on the input point cloud.
 *
 * @param pcd The input point cloud
 * @param spatial_lr_scale The spatial learning rate scale
 */

void GaussianModel::Create_from_pcd(PointCloud& pcd, float spatial_lr_scale) {
    _spatial_lr_scale = spatial_lr_scale;

    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    _xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA).set_requires_grad(true);
    std::cout << "_xyz size: " << _xyz.sizes() << std::endl;

    // auto dist2 = torch::clamp_min(distCUDA2(_xyz), 0.0000001);
    auto dist2 = torch::full({_xyz.size(0)}, 0.001, _xyz.options()).to(torch::kCUDA);
    _scaling = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3}).to(torch::kCUDA, true).set_requires_grad(true);
    _rotation = torch::zeros({_xyz.size(0), 4}).index_put_({torch::indexing::Slice(), 0}, 1).to(torch::kCUDA, true).set_requires_grad(true);
    _opacity = inverse_sigmoid(0.5 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true).set_requires_grad(true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

    // colors
    auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f).to(torch::kCUDA);
    // auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) ).to(torch::kCUDA);
    // features
    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);
}

void GaussianModel::Export_to_pcd(PointCloud& pcd) {
    // Ensure that the _xyz tensor is on the CPU
    auto xyz_cpu = _xyz.to(torch::kCPU);
    auto scaling_cpu = _scaling.to(torch::kCPU);
    auto rotation_cpu = _rotation.to(torch::kCPU);
    auto color_cpu = _features_dc.transpose(1, 2).index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}).to(torch::kCPU) * 255.f;

    // Resize the PointCloud vectors to the appropriate size
    pcd._points.resize(xyz_cpu.size(0));
    pcd._normals.resize(xyz_cpu.size(0)); // Assuming normals are stored in _scaling (if not, modify accordingly)
    pcd._distance.resize(xyz_cpu.size(0)); // If the Distance struct maps to _scaling
    pcd._quaternion.resize(xyz_cpu.size(0)); // If the Quaternions struct maps to _rotation
    pcd._colors.resize(xyz_cpu.size(0));

    // Populate the PointCloud struct
    for (size_t i = 0; i < xyz_cpu.size(0); ++i) {
        // Point
        pcd._points[i].x = xyz_cpu[i][0].item<float>();
        pcd._points[i].y = xyz_cpu[i][1].item<float>();
        pcd._points[i].z = xyz_cpu[i][2].item<float>();

        // Color
        pcd._colors[i].r = static_cast<unsigned char>(color_cpu[i][0].item<float>());
        pcd._colors[i].g = static_cast<unsigned char>(color_cpu[i][1].item<float>());
        pcd._colors[i].b = static_cast<unsigned char>(color_cpu[i][2].item<float>());

        // Normals (assuming scaling corresponds to normals)
        pcd._normals[i].x = scaling_cpu[i][0].item<float>();
        pcd._normals[i].y = scaling_cpu[i][1].item<float>();
        pcd._normals[i].z = scaling_cpu[i][2].item<float>();

        // Quaternions
        pcd._quaternion[i].qw = rotation_cpu[i][0].item<float>();
        pcd._quaternion[i].qx = rotation_cpu[i][1].item<float>();
        pcd._quaternion[i].qy = rotation_cpu[i][2].item<float>();
        pcd._quaternion[i].qz = rotation_cpu[i][3].item<float>();

        // Distance (assuming it's directly stored or computed)
        pcd._distance[i].r1 = scaling_cpu[i][0].item<float>(); // Replace with actual mapping if different
        pcd._distance[i].r2 = scaling_cpu[i][1].item<float>(); // Replace with actual mapping if different
        pcd._distance[i].r3 = scaling_cpu[i][2].item<float>(); // Replace with actual mapping if different
    }
}

// __global__ void processPoints(GS_point **GaussianCloud, float *points, float *normals, float *distances, float *quaternions, uchar3 *colors, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n && GaussianCloud[idx] != nullptr) {
//         GS_point* gs_point = GaussianCloud[idx];
//         // Assume each attribute has an appropriate layout in memory
//         points[idx * 3 + 0] = gs_point->_points.x;
//         points[idx * 3 + 1] = gs_point->_points.y;
//         points[idx * 3 + 2] = gs_point->_points.z;
//         normals[idx * 3 + 0] = gs_point->_normals.x;
//         normals[idx * 3 + 1] = gs_point->_normals.y;
//         normals[idx * 3 + 2] = gs_point->_normals.z;
//         distances[idx] = gs_point->_distance.r1;
//         quaternions[idx * 4 + 0] = gs_point->_quaternion.qx;
//         quaternions[idx * 4 + 1] = gs_point->_quaternion.qy;
//         quaternions[idx * 4 + 2] = gs_point->_quaternion.qz;
//         quaternions[idx * 4 + 3] = gs_point->_quaternion.qw;
//         colors[idx] = make_uchar3(gs_point->_colors.r, gs_point->_colors.g, gs_point->_colors.b);
//     }
// }

__global__ void processPoints(GS_point *voxel_gs_points, int num_points,
                              Point *points, Normal *normals, Distance *distances,
                              Quaternions *quaternions, Color *colors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {

        points[idx] = voxel_gs_points[idx]._points;
        normals[idx] = voxel_gs_points[idx]._normals;
        distances[idx] = voxel_gs_points[idx]._distance;
        quaternions[idx] = voxel_gs_points[idx]._quaternion;
        colors[idx] = voxel_gs_points[idx]._colors;
    }
}

void cudaProcess(std::vector<GS_point>& voxel_gs_points, PointCloud& pc_his) {
    int num_points = voxel_gs_points.size();


    GS_point* d_voxel_gs_points;



    cudaMalloc((void**)&d_voxel_gs_points, num_points * sizeof(GS_point));
    cudaMemcpy(d_voxel_gs_points, voxel_gs_points.data(), num_points * sizeof(GS_point), cudaMemcpyHostToDevice);

    Point* d_points;
    Normal* d_normals;
    Distance* d_distances;
    Quaternions* d_quaternions;
    Color* d_colors;
    cudaMalloc((void**)&d_points, num_points * sizeof(Point));
    cudaMalloc((void**)&d_normals, num_points * sizeof(Normal));
    cudaMalloc((void**)&d_distances, num_points * sizeof(Distance));
    cudaMalloc((void**)&d_quaternions, num_points * sizeof(Quaternions));
    cudaMalloc((void**)&d_colors, num_points * sizeof(Color));


    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;




    processPoints<<<blocksPerGrid, threadsPerBlock>>>(d_voxel_gs_points, num_points,
                                                      d_points, d_normals, d_distances,
                                                      d_quaternions, d_colors);


    pc_his._points.resize(num_points);
    pc_his._normals.resize(num_points);
    pc_his._distance.resize(num_points);
    pc_his._quaternion.resize(num_points);
    pc_his._colors.resize(num_points);
    cudaMemcpy(pc_his._points.data(), d_points, num_points * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._normals.data(), d_normals, num_points * sizeof(Normal), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._distance.data(), d_distances, num_points * sizeof(Distance), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._quaternion.data(), d_quaternions, num_points * sizeof(Quaternions), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._colors.data(), d_colors, num_points * sizeof(Color), cudaMemcpyDeviceToHost);


    cudaFree(d_voxel_gs_points);
    cudaFree(d_points);
    cudaFree(d_normals);
    cudaFree(d_distances);
    cudaFree(d_quaternions);
    cudaFree(d_colors);
}


// void GaussianModel::Create_from_our_format(std::vector<GS_point>& GaussianCloud) {

//     // PointCloud pcd;
//     // static int cnt=0;
//     // cudaProcess(GaussianCloud, pcd);
    
//     const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
//     // 
    
//     // _xyz = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
//     //                         at::ArrayRef<int64_t>{3 * sizeof(float), sizeof(float)}, pointType)
//     //          .clone().to(torch::kCUDA).set_requires_grad(true);

//     int floats_per_gs_point = sizeof(GS_point) / sizeof(float);

//     size_t num_points = GaussianCloud.size();
//     auto gaussians = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), floats_per_gs_point}, options)


//     // if(cnt==20)
//     // {
//     // torch::save(gaussians, "debug/ALL.pt");
//     // }
    
//     // _xyz = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3},
//     //                         {sizeof(GS_point) / sizeof(float), 1}, pointType)
//     //          .clone().to(torch::kCUDA).set_requires_grad(true);

//     _xyz = gaussians.narrow(1, 0, 3).set_requires_grad(true);
//     // _xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA).set_requires_grad(true);
   
//     //xyz 012

//     //nxyz 345
//     // dxyz 678
//     auto _distance = gaussians.narrow(1, 6, 3);
//     _scaling =  torch::log(_distance).to(torch::kCUDA).set_requires_grad(true);
    
//     // _scaling =  torch::log((torch::from_blob(pcd._distance.data(), {static_cast<long>(pcd._distance.size()), 3}, pointType))).to(torch::kCUDA).set_requires_grad(true);

//     // qwxyz 9 10 11 12
//     _rotation = gaussians.narrow(1, 9, 4).set_requires_grad(true);

//     // _rotation = (torch::from_blob(pcd._quaternion.data(), {static_cast<long>(pcd._quaternion.size()), 4}, pointType)).to(torch::kCUDA).set_requires_grad(true);



//     _opacity = inverse_sigmoid(0.5 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true).set_requires_grad(true);
//     _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

//     // colors
//     // auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
//     auto colorType = torch::TensorOptions().dtype(torch::kFloat32);
//     // rgb 13 14 15
//     // _
//     // auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f).to(torch::kCUDA);
//     auto fused_color = RGB2SH(gaussians.narrow(1, 13, 3)/ 255.f);
//     // auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) ).to(torch::kCUDA);
//     // features
//     auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
//     features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
//     features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
//     _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);
//     // _features_dc = gaussians.narrow(1, 13, 3);
//     _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);
//     //   if(cnt==20)
//     // {
//     //     torch::save(_xyz, "debug/_xyz.pt");
//     //     torch::save(_scaling, "debug/_scaling.pt");
//     //     torch::save(_rotation, "debug/_rotation.pt");
//     //     torch::save(_features_dc, "debug/_features_dc.pt");

//     // }
//     // cnt++;

// }


void GaussianModel::Create_from_our_format(std::vector<GS_point>& GaussianCloud) {

    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);


    int floats_per_gs_point = sizeof(GS_point) / sizeof(float);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    size_t num_points = GaussianCloud.size();


    auto gaussians = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), floats_per_gs_point}, options)
                    .clone()
                    .to(torch::kCUDA);


    _xyz = gaussians.narrow(1, 0, 3).set_requires_grad(true);
// 345

    auto _distance = gaussians.narrow(1, 6, 3);
    _scaling =  torch::log(_distance).to(torch::kCUDA).set_requires_grad(true);
    // _scaling =  _distance.to(torch::kCUDA).set_requires_grad(true);

// 9 10 11 12
    _rotation = gaussians.narrow(1, 9, 4).set_requires_grad(true);

    // _opacity = inverse_sigmoid(1.0 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true).set_requires_grad(true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

    auto colorType = torch::TensorOptions().dtype(torch::kFloat32);


// 13 14 15
    auto fused_color = RGB2SH(gaussians.narrow(1, 13, 3)/ 255.f);

    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);

    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);

    //16 
    _opacity=gaussians.narrow(1, 16, 1).to(torch::kCUDA, true).set_requires_grad(true);

     _index = gaussians.narrow(1, 17, 1).to(torch::kCPU);
    
     _flag_in_fov= gaussians.narrow(1, 18, 1).to(torch::kCPU);


}


void GaussianModel::Dump_to_our_format(std::vector<GS_point>& GaussianCloud,int size) {



    auto _distance = torch::exp(_scaling).to(torch::kCPU);



    auto _fused_color = SH2RGB(_features_dc.to(torch::kCPU).squeeze(1)) * 255;
    auto _xyz2=_xyz.to(torch::kCPU); 
    auto _rotation2=_rotation.to(torch::kCPU); 
    auto _opacity2=_opacity.to(torch::kCPU);
//   const char* red_background = "\033[41m";
//   // ANSI escape code for green text
//   const char* green_text = "\033[32m";
//   // ANSI escape code to reset colors
//   const char* reset_colors = "\033[0m";
//   std::cout << "GaussianCloud.size() size: " <<green_text<<red_background<< GaussianCloud.size() <<reset_colors<< std::endl;


//     std::cout << "_scaling shape: " << _scaling.sizes() << std::endl;
//     std::cout << "_distance shape: " << _distance.sizes() << std::endl;
//     std::cout << "_features_dc shape: " << _features_dc.sizes() << std::endl;
//     std::cout << "_fused_color shape: " << _fused_color.sizes() << std::endl;
//     std::cout << "_xyz shape: " << _xyz.sizes() << std::endl;
//     std::cout << "_rotation shape: " << _rotation.sizes() << std::endl;
    // auto zeros_column = torch::zeros({_xyz2.size(0), 1}, torch::kInt32).to(torch::kCPU);

    torch::Tensor gaussians = torch::cat({_xyz2, _distance, _distance, _rotation2, _fused_color,_opacity2,_flag_in_fov,_index}, 1);


    // std::vector<GS_point> gs_points(GaussianCloud.size());
    // std::memcpy(gs_points.data(), gaussians.data_ptr<float>(), gaussians.numel() * sizeof(float));


    std::vector<GS_point> gs_points(size);
    // std::memcpy(gs_points.data(), gaussians.data_ptr<float>(), gaussians.numel() * sizeof(float));
    std::memcpy(gs_points.data(), gaussians.data_ptr<float>(), size * sizeof(GS_point));


    GaussianCloud = std::move(gs_points);
}


// void GaussianModel::Dump_to_our_format(std::vector<GS_point>& GaussianCloud) {


//     auto scaling_tensor = _scaling.cpu();
//     auto rotation_tensor = _rotation.cpu();
//     // auto color_tensor = _features_dc.cpu().to(torch::kFloat32);  // assuming color data is in _features_dc
//     auto color_tensor = _features_dc.cpu().to(torch::kFloat32).squeeze(1); 
//     std::cout << "color_tensor sizes: " << color_tensor.sizes() << std::endl;


//     GaussianCloud.resize(xyz_tensor.size(0));


//     for (int i = 0; i < xyz_tensor.size(0); ++i) {
//         GS_point gs_point;


//         gs_point._points.x = xyz_tensor[i][0].item<float>();
//         gs_point._points.y = xyz_tensor[i][1].item<float>();
//         gs_point._points.z = xyz_tensor[i][2].item<float>();


//         gs_point._normals.x = xyz_tensor[i][0].item<float>();  // Replace with actual normal extraction
//         gs_point._normals.y = xyz_tensor[i][1].item<float>();
//         gs_point._normals.z = xyz_tensor[i][2].item<float>();


//         gs_point._distance.r1 = scaling_tensor[i][0].item<float>();
//         gs_point._distance.r2 = scaling_tensor[i][1].item<float>();
//         gs_point._distance.r3 = scaling_tensor[i][2].item<float>();


//         gs_point._quaternion.qw = rotation_tensor[i][0].item<float>();
//         gs_point._quaternion.qx = rotation_tensor[i][1].item<float>();
//         gs_point._quaternion.qy = rotation_tensor[i][2].item<float>();
//         gs_point._quaternion.qz = rotation_tensor[i][3].item<float>();


//         gs_point._colors.r = static_cast<unsigned char>(color_tensor[i][0].item<float>() * 255);
//         gs_point._colors.g = static_cast<unsigned char>(color_tensor[i][1].item<float>() * 255);
//         gs_point._colors.b = static_cast<unsigned char>(color_tensor[i][2].item<float>() * 255);
        


//         GaussianCloud[i] = gs_point;
//     }
// }


// void GaussianModel::Dump_to_our_format(std::vector<GS_point>& GaussianCloud) {


//     auto scaling_tensor = _scaling.cpu();
//     auto rotation_tensor = _rotation.cpu();
//     // auto color_tensor = _features_dc.cpu().to(torch::kFloat32);  // assuming color data is in _features_dc
//     auto color_tensor = _features_dc.cpu().to(torch::kFloat32).squeeze(1); 
//     std::cout << "color_tensor sizes: " << color_tensor.sizes() << std::endl;


//     GaussianCloud.resize(xyz_tensor.size(0));


//     for (int i = 0; i < xyz_tensor.size(0); ++i) {
//         GS_point gs_point;


//         gs_point._points.x = xyz_tensor[i][0].item<float>();
//         gs_point._points.y = xyz_tensor[i][1].item<float>();
//         gs_point._points.z = xyz_tensor[i][2].item<float>();


//         gs_point._normals.x = xyz_tensor[i][0].item<float>();  // Replace with actual normal extraction
//         gs_point._normals.y = xyz_tensor[i][1].item<float>();
//         gs_point._normals.z = xyz_tensor[i][2].item<float>();


//         gs_point._distance.r1 = scaling_tensor[i][0].item<float>();
//         gs_point._distance.r2 = scaling_tensor[i][1].item<float>();
//         gs_point._distance.r3 = scaling_tensor[i][2].item<float>();


//         gs_point._quaternion.qw = rotation_tensor[i][0].item<float>();
//         gs_point._quaternion.qx = rotation_tensor[i][1].item<float>();
//         gs_point._quaternion.qy = rotation_tensor[i][2].item<float>();
//         gs_point._quaternion.qz = rotation_tensor[i][3].item<float>();


//         gs_point._colors.r = static_cast<unsigned char>(color_tensor[i][0].item<float>() * 255);
//         gs_point._colors.g = static_cast<unsigned char>(color_tensor[i][1].item<float>() * 255);
//         gs_point._colors.b = static_cast<unsigned char>(color_tensor[i][2].item<float>() * 255);
        


//         GaussianCloud[i] = gs_point;
//     }
// }


void GaussianModel::Create_from_our_format_optimized(std::vector<GS_point>& GaussianCloud) {
    size_t num_points = GaussianCloud.size();
    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    const auto colorType = torch::TensorOptions().dtype(torch::kUInt8);


    _xyz = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
                            at::ArrayRef<int64_t>{3 * sizeof(float), sizeof(float)}, pointType)
             .clone().to(torch::kCUDA).set_requires_grad(true);

    _scaling = torch::log(torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
                                           at::ArrayRef<int64_t>{sizeof(GS_point), sizeof(float)}, pointType)
                            .clone().to(torch::kCUDA)).set_requires_grad(true);

    _rotation = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 4}, 
                                 at::ArrayRef<int64_t>{sizeof(GS_point), sizeof(float)}, pointType)
                 .clone().to(torch::kCUDA).set_requires_grad(true);

    _opacity = inverse_sigmoid(0.5 * torch::ones({static_cast<long>(num_points), 1})).to(torch::kCUDA, true).set_requires_grad(true);
    _max_radii2D = torch::zeros({static_cast<long>(num_points)}).to(torch::kCUDA, true);

    // Colors
    auto fused_color = RGB2SH(torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
                                               at::ArrayRef<int64_t>{sizeof(GS_point), sizeof(uint8_t)}, colorType)
                              .clone().to(pointType) / 255.f).to(torch::kCUDA);

    // Features
    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);
}

__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}


__global__ void computeDepthImage(Point* points,
                                  const float* T_f_w,
                                  const float fx, const float fy, const float cx, const float cy,
                                  float* depth_img,
                                  int width, int height,
                                  int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

        // printf("Camera intrinsics: fx = %f, fy = %f, cx = %f, cy = %f\n", fx, fy, cx, cy);
        

        // printf("T_f_w matrix:\n");
        // for (int i = 0; i < 3; i++) {
        //     printf("%f %f %f %f\n", T_f_w[i * 4], T_f_w[i * 4 + 1], T_f_w[i * 4 + 2], T_f_w[i * 4 + 3]);
        // }


    // printf("Thread %d is running.\n", idx);


    float x = points[idx].x;
    float y = points[idx].y;
    float z = points[idx].z;


    float pt_c_x = T_f_w[0] * x + T_f_w[1] * y + T_f_w[2] * z + T_f_w[3];
    float pt_c_y = T_f_w[4] * x + T_f_w[5] * y + T_f_w[6] * z + T_f_w[7];
    float pt_c_z = T_f_w[8] * x + T_f_w[9] * y + T_f_w[10] * z + T_f_w[11];
// printf("x = %d, y = %d , z = %d\n", pt_c_x, pt_c_y,pt_c_z);
    if (pt_c_z <= 0) return;


    float u = fx * (pt_c_x / pt_c_z) + cx;

    float v = fy * (pt_c_y / pt_c_z) + cy;
// printf("u = %d, v = %d ", u, v);
    int col = static_cast<int>(u);
    int row = static_cast<int>(v);
// printf("col = %d, row = %d\n", col, row);

    if (col >= 0 && col < width && row >= 0 && row < height) {

        float depth = pt_c_z;
        int pixel_idx = row * width + col;
        if (atomicCAS((int*)&depth_img[pixel_idx], __float_as_int(0.0f), __float_as_int(depth)) == __float_as_int(0.0f)) {

            return;
        }

        atomicMinFloat(&depth_img[pixel_idx], depth);

    }
}

void cudaProcess_with_depth_making(std::vector<GS_point>& voxel_gs_points, PointCloud& pc_his
, const float* Rcw, const float* Pcw, float fx, float fy, float cx, float cy,int width,int height,
float* depth_img_cpu)  {
    int num_points = voxel_gs_points.size();


    GS_point* d_voxel_gs_points;

    
    float* d_depth_img;
    cudaMalloc((void**)&d_depth_img, width * height * sizeof(float));
    cudaMemset(d_depth_img, 0x00, width * height * sizeof(float)); 



    cudaMalloc((void**)&d_voxel_gs_points, num_points * sizeof(GS_point));
    cudaMemcpy(d_voxel_gs_points, voxel_gs_points.data(), num_points * sizeof(GS_point), cudaMemcpyHostToDevice);

    Point* d_points;
    Normal* d_normals;
    Distance* d_distances;
    Quaternions* d_quaternions;
    Color* d_colors;
    cudaMalloc((void**)&d_points, num_points * sizeof(Point));
    cudaMalloc((void**)&d_normals, num_points * sizeof(Normal));
    cudaMalloc((void**)&d_distances, num_points * sizeof(Distance));
    cudaMalloc((void**)&d_quaternions, num_points * sizeof(Quaternions));
    cudaMalloc((void**)&d_colors, num_points * sizeof(Color));


    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    processPoints<<<blocksPerGrid, threadsPerBlock>>>(d_voxel_gs_points, num_points,
                                                      d_points, d_normals, d_distances,
                                                      d_quaternions, d_colors);
 

    float T_f_w[12];

        
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T_f_w[i * 4 + j] = Rcw[i * 3 + j];
        }
        T_f_w[i * 4 + 3] = Pcw[i];
    }
    float* d_T_f_w;
    cudaMalloc((void**)&d_T_f_w, 12 * sizeof(float));
    cudaMemcpy(d_T_f_w, T_f_w, 12 * sizeof(float), cudaMemcpyHostToDevice);
    
    computeDepthImage<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_T_f_w, fx, fy, cx, cy, 
                                                          d_depth_img,
                                                          width, height, num_points);
    
    // float* depth_img_cpu = new float[width * height];
    cudaMemcpy(depth_img_cpu, d_depth_img, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_depth_img);
    
    





    pc_his._points.resize(num_points);
    pc_his._normals.resize(num_points);
    pc_his._distance.resize(num_points);
    pc_his._quaternion.resize(num_points);
    pc_his._colors.resize(num_points);
    cudaMemcpy(pc_his._points.data(), d_points, num_points * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._normals.data(), d_normals, num_points * sizeof(Normal), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._distance.data(), d_distances, num_points * sizeof(Distance), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._quaternion.data(), d_quaternions, num_points * sizeof(Quaternions), cudaMemcpyDeviceToHost);
    cudaMemcpy(pc_his._colors.data(), d_colors, num_points * sizeof(Color), cudaMemcpyDeviceToHost);


    cudaFree(d_voxel_gs_points);
    cudaFree(d_points);
    cudaFree(d_normals);
    cudaFree(d_distances);
    cudaFree(d_quaternions);
    cudaFree(d_colors);
}


void GaussianModel::Create_from_our_format_with_depth_making(std::vector<GS_point>& GaussianCloud,
                                                             float Rcw_f[9],
                                                             float Pcw_f[3],
                                                             float fx, float fy, float cx, float cy,
                                                             int width, int height,float* depth_img_cpu) 

{
    PointCloud pcd;


    // float* depth_img_cpu = new float[width * height]; 
    cudaProcess_with_depth_making(GaussianCloud, pcd, Rcw_f, Pcw_f, fx, fy, cx, cy, width, height, depth_img_cpu);
    // cudaProcess_with_depth_making(GaussianCloud, pcd, Rcw_f, Pcw_f, fx, fy, cx, cy,width,height);

    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    // 
    _xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA).set_requires_grad(true);
   
    //    
    _scaling =  torch::log((torch::from_blob(pcd._distance.data(), {static_cast<long>(pcd._distance.size()), 3}, pointType))).to(torch::kCUDA).set_requires_grad(true);

    // 
    _rotation = (torch::from_blob(pcd._quaternion.data(), {static_cast<long>(pcd._quaternion.size()), 4}, pointType)).to(torch::kCUDA).set_requires_grad(true);
  


    _opacity = inverse_sigmoid(0.5 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true).set_requires_grad(true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

    // colors
    auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f).to(torch::kCUDA);
    // auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) ).to(torch::kCUDA);
    // features
    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);
}


void GaussianModel::Create_from_pcd_our(PointCloud& pcd) {

    static bool is_dumped = false;
    static bool is_original = true;
    // static int count = 0;
    // count++;
    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    // 
    _xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA).set_requires_grad(true);
   
    //    
    _scaling =  torch::log((torch::from_blob(pcd._distance.data(), {static_cast<long>(pcd._distance.size()), 3}, pointType))).to(torch::kCUDA).set_requires_grad(true);

    // 
    _rotation = (torch::from_blob(pcd._quaternion.data(), {static_cast<long>(pcd._quaternion.size()), 4}, pointType)).to(torch::kCUDA).set_requires_grad(true);
  


    _opacity = inverse_sigmoid(0.5 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true).set_requires_grad(true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

    // colors
    auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f).to(torch::kCUDA);
    // auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) ).to(torch::kCUDA);
    // features
    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);
}
/**
 * @brief Setup the Gaussian Model for training
 *
 * This function sets up the Gaussian model for training by initializing several
 * parameters and settings based on the provided OptimizationParameters object.
 *
 * @param params The OptimizationParameters object providing the settings for training
 */


void GaussianModel::Training_setup(const gs::param::OptimizationParameters& params) {
    this->_percent_dense = params.percent_dense;
    this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                              params.position_lr_final * this->_spatial_lr_scale,
                                              params.position_lr_delay_mult,
                                              params.position_lr_max_steps);

    std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups;
    optimizer_params_groups.reserve(6);
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_xyz}, std::make_unique<torch::optim::AdamOptions>(params.position_lr_init * this->_spatial_lr_scale)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_features_dc}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_features_rest}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr / 20.)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_scaling}, std::make_unique<torch::optim::AdamOptions>(params.scaling_lr * this->_spatial_lr_scale)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_rotation}, std::make_unique<torch::optim::AdamOptions>(params.rotation_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_opacity}, std::make_unique<torch::optim::AdamOptions>(params.opacity_lr)));

    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[4].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[5].options()).eps(1e-15);

    _optimizer = std::make_unique<torch::optim::Adam>(optimizer_params_groups, torch::optim::AdamOptions(0.f).eps(1e-15));
}

void GaussianModel::Update_learning_rate(float iteration) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    auto lr = _xyz_scheduler_args(iteration);
    static_cast<torch::optim::AdamOptions&>(_optimizer->param_groups()[0].options()).set_lr(lr);
}

void GaussianModel::Reset_opacity() {
    // opacitiy activation
    auto new_opacity = inverse_sigmoid(torch::ones_like(_opacity, torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f);

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer->state()[_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl()]));

    _optimizer->state().erase(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl());

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    _optimizer->param_groups()[5].params()[0] = new_opacity.set_requires_grad(true);
    _opacity = _optimizer->param_groups()[5].params()[0];

    _optimizer->state()[_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl()] = std::move(adamParamStates);
}

void prune_optimizer(torch::optim::Adam* optimizer, const torch::Tensor& mask, torch::Tensor& old_tensor, int param_position) {
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()]));
    optimizer->state().erase(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl());

    adamParamStates->exp_avg(adamParamStates->exp_avg().index_select(0, mask));
    adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index_select(0, mask));

    optimizer->param_groups()[param_position].params()[0] = old_tensor.index_select(0, mask).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0]; // update old tensor
    optimizer->state()[optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()] = std::move(adamParamStates);
}

void GaussianModel::prune_points(torch::Tensor mask) {
    // reverse to keep points
    auto valid_point_mask = ~mask;
    int true_count = valid_point_mask.sum().item<int>();
    auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    prune_optimizer(_optimizer.get(), indices, _xyz, 0);
    prune_optimizer(_optimizer.get(), indices, _features_dc, 1);
    prune_optimizer(_optimizer.get(), indices, _features_rest, 2);
    prune_optimizer(_optimizer.get(), indices, _scaling, 3);
    prune_optimizer(_optimizer.get(), indices, _rotation, 4);
    prune_optimizer(_optimizer.get(), indices, _opacity, 5);

    _xyz_gradient_accum = _xyz_gradient_accum.index_select(0, indices);
    _denom = _denom.index_select(0, indices);
    _max_radii2D = _max_radii2D.index_select(0, indices);
}

void cat_tensors_to_optimizer(torch::optim::Adam* optimizer,
                              torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              int param_position) {
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()]));
    optimizer->state().erase(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl());

    adamParamStates->exp_avg(torch::cat({adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)}, 0));
    adamParamStates->exp_avg_sq(torch::cat({adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));

    optimizer->param_groups()[param_position].params()[0] = torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    optimizer->state()[optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()] = std::move(adamParamStates);
}

void GaussianModel::densification_postfix(torch::Tensor& new_xyz,
                                          torch::Tensor& new_features_dc,
                                          torch::Tensor& new_features_rest,
                                          torch::Tensor& new_scaling,
                                          torch::Tensor& new_rotation,
                                          torch::Tensor& new_opacity) {
    cat_tensors_to_optimizer(_optimizer.get(), new_xyz, _xyz, 0);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_dc, _features_dc, 1);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_rest, _features_rest, 2);
    cat_tensors_to_optimizer(_optimizer.get(), new_scaling, _scaling, 3);
    cat_tensors_to_optimizer(_optimizer.get(), new_rotation, _rotation, 4);
    cat_tensors_to_optimizer(_optimizer.get(), new_opacity, _opacity, 5);

    _xyz_gradient_accum = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _denom = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
}

void GaussianModel::densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, float min_opacity, float max_screen_size) {
    static const int N = 2;
    const int n_init_points = _xyz.size(0);
    // Extract points that satisfy the gradient condition
    torch::Tensor padded_grad = torch::zeros({n_init_points}).to(torch::kCUDA);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();
    torch::Tensor selected_pts_mask = torch::where(padded_grad >= grad_threshold, torch::ones_like(padded_grad).to(torch::kBool), torch::zeros_like(padded_grad).to(torch::kBool));
    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)) > _percent_dense * scene_extent);
    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);

    torch::Tensor stds = Get_scaling().index_select(0, indices).repeat({N, 1});
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots = build_rotation(_rotation.index_select(0, indices)).repeat({N, 1, 1});

    torch::Tensor new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _xyz.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_scaling = torch::log(Get_scaling().index_select(0, indices).repeat({N, 1}) / (0.8 * N));
    torch::Tensor new_rotation = _rotation.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_features_dc = _features_dc.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_opacity = _opacity.index_select(0, indices).repeat({N, 1});

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);

    torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(torch::kCUDA)});
    // torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()})}).to(torch::kBool).to(torch::kCUDA);
    prune_filter = torch::logical_or(prune_filter, (Get_opacity() < min_opacity).squeeze(-1));
    prune_points(prune_filter);
}

void GaussianModel::densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent) {
    // Extract points that satisfy the gradient condition
    auto grad_norm = torch::sqrt(torch::sum(grads * grads, 1, true));
    torch::Tensor selected_pts_mask = torch::where(grad_norm >= grad_threshold,
                                                   torch::ones_like(grads.index({torch::indexing::Slice()})).to(torch::kBool),
                                                   torch::zeros_like(grads.index({torch::indexing::Slice()})).to(torch::kBool))
                                          .to(torch::kLong);

    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)).unsqueeze(-1) <= _percent_dense * scene_extent);

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    torch::Tensor new_xyz = _xyz.index_select(0, indices);
    torch::Tensor new_features_dc = _features_dc.index_select(0, indices);
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices);
    torch::Tensor new_opacity = _opacity.index_select(0, indices);
    torch::Tensor new_scaling = _scaling.index_select(0, indices);
    torch::Tensor new_rotation = _rotation.index_select(0, indices);

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);
}

void GaussianModel::Densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size) {
    torch::Tensor grads = _xyz_gradient_accum / _denom;
    grads.index_put_({grads.isnan()}, 0.0);

    densify_and_clone(grads, max_grad, extent);
    densify_and_split(grads, max_grad, extent, min_opacity, max_screen_size);
}

void GaussianModel::Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter) {
    _xyz_gradient_accum.index_put_({update_filter}, _xyz_gradient_accum.index_select(0, update_filter.nonzero().squeeze()) + viewspace_point_tensor.grad().index_select(0, update_filter.nonzero().squeeze()).slice(1, 0, 2).norm(2, -1, true));
    _denom.index_put_({update_filter}, _denom.index_select(0, update_filter.nonzero().squeeze()) + 1);
}

std::vector<std::string> GaussianModel::construct_list_of_attributes() {
    std::vector<std::string> attributes = {"x", "y", "z", "nx", "ny", "nz"};

    for (int i = 0; i < _features_dc.size(1) * _features_dc.size(2); ++i)
        attributes.push_back("f_dc_" + std::to_string(i));

    for (int i = 0; i < _features_rest.size(1) * _features_rest.size(2); ++i)
        attributes.push_back("f_rest_" + std::to_string(i));

    attributes.emplace_back("opacity");

    for (int i = 0; i < _scaling.size(1); ++i)
        attributes.push_back("scale_" + std::to_string(i));

    for (int i = 0; i < 4; ++i)
        attributes.push_back("rot_" + std::to_string(i));

    return attributes;
}

void GaussianModel::Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration) {
    std::cout << "Saving at " << std::to_string(iteration) << " iterations\n";
    auto folder = file_path / ("point_cloud/iteration_" + std::to_string(iteration));
    std::filesystem::create_directories(folder);

    auto xyz = _xyz.cpu().contiguous();
    auto normals = torch::zeros_like(xyz);
    auto f_dc = _features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
    auto f_rest = _features_rest.transpose(1, 2).flatten(1).cpu().contiguous();
    auto opacities = _opacity.cpu();
    auto scale = _scaling.cpu();
    auto rotation = _rotation.cpu();

    std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
                                                    normals.clone(),
                                                    f_dc.clone(),
                                                    f_rest.clone(),
                                                    opacities.clone(),
                                                    scale.clone(),
                                                    rotation.clone()};
    auto attributes = construct_list_of_attributes();
    std::thread t = std::thread([folder, tensor_attributes, attributes]() {
        Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);
    });

    if (isLastIteration) {
        t.join();
    } else {
        t.detach();
    }
}


void GaussianModel::Save_ply_our(const std::filesystem::path& file_path,std::vector<GS_point>& GaussianCloud) {
    PointCloud pcd;

    // auto folder = file_path / ("point_cloud/iteration_999999");
    // auto folder =;
    // std::filesystem::create_directories(folder);
    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);

    int floats_per_gs_point = sizeof(GS_point) / sizeof(float);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    size_t num_points = GaussianCloud.size();
    auto gaussians = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), floats_per_gs_point}, options)
                .clone()
                .to(torch::kCUDA);
    auto opacities=gaussians.narrow(1, 16, 1).cpu();

    cudaProcess(GaussianCloud, pcd);
    std::cout<<"dumped out GS to pcd"<<std::endl;
    
    // const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    // 
    auto xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType);

    auto normals = torch::zeros_like(xyz);


    auto colorType = torch::TensorOptions().dtype(torch::kFloat32);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f);

    // features

    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))});



    features.index_put_({torch::indexing::Slice(), 
    torch::indexing::Slice(torch::indexing::None, 3),



     0
     }, fused_color);








    features.index_put_({torch::indexing::Slice(), 

    torch::indexing::Slice(3, torch::indexing::None),

     torch::indexing::Slice(1, torch::indexing::None)

     }, 0.0

     );



    auto features_dc = features.index(
    {torch::indexing::Slice(),
    torch::indexing::Slice(),
    torch::indexing::Slice(0, 1)

     })
    //  (batch_size, 3, 1)
     .transpose(1, 2)

     .contiguous();
    
    auto features_rest  =features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous();


    auto f_dc = features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
    f_dc = f_dc.index({torch::indexing::Slice(), torch::tensor({2, 1, 0})});





    auto f_rest = features_rest.transpose(1, 2).flatten(1).cpu().contiguous();
    // auto f_rest = _features_rest.transpose(1, 2).flatten(1).cpu().contiguous();

    // auto opacities = inverse_sigmoid(0.5 * torch::ones({xyz.size(0), 1}));
    auto scale =   torch::log((torch::from_blob(pcd._distance.data(), {static_cast<long>(pcd._distance.size()), 3}, pointType)));
    auto rotation = (torch::from_blob(pcd._quaternion.data(), {static_cast<long>(pcd._quaternion.size()), 4}, pointType));
  
    std::cout<<"inited GS to vector"<<std::endl;
    std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
                                                    normals.clone(),
                                                    f_dc.clone(),
                                                    f_rest.clone(),
                                                    opacities.clone(),
                                                    scale.clone(),
                                                    rotation.clone()};


    auto attributes = construct_list_of_attributes();




    // colors


std::cout<<"inited GS to attributes"<<std::endl;
    Write_output_ply(file_path, tensor_attributes, attributes);

std::cout<<" GS saved"<<std::endl;

}



void GaussianModel::Save_ply_our2(const std::filesystem::path& file_path,std::vector<GS_point>& GaussianCloud) {


const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);

int floats_per_gs_point = sizeof(GS_point) / sizeof(float);
auto options = torch::TensorOptions().dtype(torch::kFloat32);
size_t num_points = GaussianCloud.size();


auto gaussians = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), floats_per_gs_point}, options)
                .clone()
                .to(torch::kCUDA);


    auto xyz = gaussians.narrow(1, 0, 3).cpu().contiguous();
// 345

    auto distance = gaussians.narrow(1, 6, 3);
    auto scale =  torch::log(distance).to(torch::kCPU).cpu();
    // _scaling =  _distance.to(torch::kCUDA).set_requires_grad(true);

// 9 10 11 12
    auto rotation= gaussians.narrow(1, 9, 4).cpu();
    auto opacities=gaussians.narrow(1, 16, 1).to(torch::kCPU, true).cpu();
    // auto opacities = inverse_sigmoid(1.0 * torch::ones({_xyz.size(0), 1})).to(torch::kCPU, true).cpu();
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCPU, true).cpu();

    auto colorType = torch::TensorOptions().dtype(torch::kFloat32);


// 13 14 15
    std::cout << "gaussians.narrow(1, 13, 3) " << gaussians.narrow(1, 13, 3).sizes() << std::endl;
    auto fused_color = RGB2SH(gaussians.narrow(1, 13, 3)/ 255.f);

    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCPU);
    
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    
    auto features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();

    auto features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous();
    

    auto normals = torch::zeros_like(xyz);
    auto f_dc = features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
    auto f_rest = features_rest.transpose(1, 2).flatten(1).cpu().contiguous();


std::cout << "xyz dimensions: " << xyz.sizes() << std::endl;
std::cout << "normals dimensions: " << normals.sizes() << std::endl;
std::cout << "f_dc dimensions: " << f_dc.sizes() << std::endl;
std::cout << "f_rest dimensions: " << f_rest.sizes() << std::endl;
std::cout << "opacities dimensions: " << opacities.sizes() << std::endl;
std::cout << "scale dimensions: " << scale.sizes() << std::endl;
std::cout << "rotation dimensions: " << rotation.sizes() << std::endl;

    
std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
                                                normals.clone(),
                                                f_dc.clone(),
                                                f_rest.clone(),
                                                opacities.clone(),
                                                scale.clone(),
                                                rotation.clone()};



    auto attributes = construct_list_of_attributes();


    std::cout<<"inited GS to attributes"<<std::endl;
    Write_output_ply(file_path, tensor_attributes, attributes);
    std::cout<<" GS saved"<<std::endl;

                                                
}

// void GaussianModel::Save_ply_our2(const std::filesystem::path& file_path,std::vector<GS_point>& GaussianCloud) {


//     size_t num_points = GaussianCloud.size();

//     const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
//     const auto colorType = torch::TensorOptions().dtype(torch::kUInt8);


//     auto xyz = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
//                             at::ArrayRef<int64_t>{3 * sizeof(float), sizeof(float)}, pointType)
//              .clone().to(torch::kCPU);

//     auto scale = torch::log(torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
//                                            at::ArrayRef<int64_t>{sizeof(GS_point), sizeof(float)}, pointType)
//                             .clone().to(torch::kCPU));

//     auto rotation = torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 4}, 
//                                  at::ArrayRef<int64_t>{sizeof(GS_point), sizeof(float)}, pointType)
//                  .clone().to(torch::kCPU);

//     auto opacity = inverse_sigmoid(0.5 * torch::ones({static_cast<long>(num_points), 1})).to(torch::kCPU, true);

//     auto fused_color = RGB2SH(torch::from_blob(GaussianCloud.data(), {static_cast<long>(num_points), 3}, 
//                                                at::ArrayRef<int64_t>{sizeof(GS_point), sizeof(uint8_t)}, colorType)
//                               .clone().to(pointType) / 255.f).to(torch::kCPU);

//     // Features
//     auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
//     features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
//     features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
//     auto f_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
//     auto f_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous();


//     std::cout<<"inited GS to vector"<<std::endl;
//     std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
//                                                     xyz.clone(),
//                                                     f_dc.clone(),
//                                                     f_rest.clone(),
//                                                     opacity.clone(),
//                                                     scale.clone(),
//                                                     rotation.clone()};


//     auto attributes = construct_list_of_attributes();


//     std::cout<<"inited GS to attributes"<<std::endl;
//     auto folder = file_path / ("point_cloud/iteration_999999");
//     // Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);
//     Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);
//     std::cout<<" GS saved"<<std::endl;

// }


// void GaussianModel::Save_ply_our_backup(const std::filesystem::path& file_path,std::vector<GS_point>& GaussianCloud) {
//     PointCloud pcd;

//     auto folder = file_path / ("point_cloud/iteration_999999");
//     std::filesystem::create_directories(folder);

//     cudaProcess(GaussianCloud, pcd);
//     std::cout<<"dumped out GS to pcd"<<std::endl;
    
//     const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
//     // 
//     auto xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType);

//     auto normals = torch::zeros_like(xyz);


//     auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
//     auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f);

//     // features

//     auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))});



//     features.index_put_({torch::indexing::Slice(), 
//     torch::indexing::Slice(torch::indexing::None, 3),




//      }, fused_color);








//     features.index_put_({torch::indexing::Slice(), 

//     torch::indexing::Slice(3, torch::indexing::None),

//      torch::indexing::Slice(1, torch::indexing::None)

//      }, 0.0

//      );



//     auto features_dc = features.index(




//      })
//     //  (batch_size, 3, 1)
//      .transpose(1, 2)

//      .contiguous();
    
//     auto features_rest  =features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous();


//     auto f_dc = features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
//     f_dc = f_dc.index({torch::indexing::Slice(), torch::tensor({2, 1, 0})});





//     auto f_rest = features_rest.transpose(1, 2).flatten(1).cpu().contiguous();
//     // auto f_rest = _features_rest.transpose(1, 2).flatten(1).cpu().contiguous();

//     auto opacities = inverse_sigmoid(0.5 * torch::ones({xyz.size(0), 1}));
//     auto scale =   torch::log((torch::from_blob(pcd._distance.data(), {static_cast<long>(pcd._distance.size()), 3}, pointType)));
//     auto rotation = (torch::from_blob(pcd._quaternion.data(), {static_cast<long>(pcd._quaternion.size()), 4}, pointType));
  
//     std::cout<<"inited GS to vector"<<std::endl;
//     std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
//                                                     normals.clone(),
//                                                     f_dc.clone(),
//                                                     f_rest.clone(),
//                                                     opacities.clone(),
//                                                     scale.clone(),
//                                                     rotation.clone()};


//     auto attributes = construct_list_of_attributes();




//     // colors


// std::cout<<"inited GS to attributes"<<std::endl;

//         Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);

// std::cout<<" GS saved"<<std::endl;

// }
