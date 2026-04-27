#include "camera.cuh"
#include "camera_info.cuh"
#include "camera_utils.cuh"
#include "parameters.cuh"
#include <string>
#include <torch/torch.h>
#include <utility>

Camera::Camera(int imported_colmap_id,
               Eigen::Matrix3f R, Eigen::Vector3f T,Eigen::Matrix3f K,
               float FoVx, float FoVy,
               torch::Tensor image,
               std::string image_name,
               int uid,
               float scale) : _colmap_id(imported_colmap_id),
                              _R(R),
                              _T(T),
                              _K(K),
                              _FoVx(FoVx),
                              _FoVy(FoVy),
                              _image_name(std::move(std::move(image_name))),
                              _uid(uid),
                              _scale(scale) {

    this->_original_image = torch::clamp(image, 0.f, 1.f);
    this->_image_width = this->_original_image.size(2);
    this->_image_height = this->_original_image.size(1);
    using namespace std;
    cout<<"image_width: "<<this->_image_width<<endl;
    cout<<"image_height: "<<this->_image_height<<endl;

    this->_zfar = 100.f;
    this->_znear = 0.01f;

    this->_world_view_transform = getWorld2View2(R, T, Eigen::Vector3f::Zero(), _scale).to(torch::kCUDA, true);
    // cout<<"_world_view_transform cuda: "<<endl<<this->_world_view_transform<<endl;

    // std::cout << "World View Transform: " << this->_world_view_transform << std::endl;


    this->_image_width = this->_original_image.size(2);
    this->_image_height = this->_original_image.size(1);
    // torch::Tensor compute_perspective_matrix_from_intrinsics(const Eigen::Matrix3f& K, int width, int height, float near, float far) 
    // self.projection_matrix = compute_perspective_matrix_from_intrinsics(K, self.image_width, self.image_height, 0.001, 10).transpose(0,1).cuda()

    this->_projection_matrix = compute_perspective_matrix_from_intrinsics(this->_K,   this->_image_width,  this->_image_height , 0.001, 10).to(torch::kCUDA, true);
  

    // cout<<"projection_matrix cuda: "<<endl<<this->_projection_matrix<<endl;
    // this->_projection_matrix = getProjectionMatrix(this->_znear, this->_zfar, this->_FoVx, this->_FoVy).to(torch::kCUDA, true);
    
    this->_full_proj_transform = this->_world_view_transform.unsqueeze(0).bmm(this->_projection_matrix.unsqueeze(0)).squeeze(0);

    // cout<<"_full_proj_transform cuda: "<<endl<<this->_full_proj_transform<<endl;

    auto world_view_transform_cpu = this->_world_view_transform.cpu();


    auto A = world_view_transform_cpu.inverse();


    A = A.to(torch::kCUDA);
    this->_camera_center =A[3].slice(0, 0, 3).to(torch::kCUDA, true);
    

}


void Camera::ChangeImage(Eigen::Matrix3f R, Eigen::Vector3f T,
               torch::Tensor image) 
{

    this->_original_image = torch::clamp(image, 0.f, 1.f);
    this->_image_width = this->_original_image.size(2);
    this->_image_height = this->_original_image.size(1);
    using namespace std;
    
    this->_world_view_transform = getWorld2View2(R, T, Eigen::Vector3f::Zero(), _scale).to(torch::kCUDA, true);
    this->_image_width = this->_original_image.size(2);
    this->_image_height = this->_original_image.size(1);

    this->_full_proj_transform = this->_world_view_transform.unsqueeze(0).bmm(
    this->_projection_matrix.unsqueeze(0)).squeeze(0);
    this->_camera_center = this->_world_view_transform.inverse()[3].slice(0, 0, 3);
}


void Camera::ChangePose(Eigen::Matrix3f R, Eigen::Vector3f T)
{
    this->_world_view_transform = getWorld2View2(R, T, Eigen::Vector3f::Zero(), _scale).to(torch::kCUDA, true);
    this->_full_proj_transform = this->_world_view_transform.unsqueeze(0).bmm(this->_projection_matrix.unsqueeze(0)).squeeze(0);
    this->_camera_center = this->_world_view_transform.inverse()[3].slice(0, 0, 3);

}

#include <opencv2/opencv.hpp>
void display_image(const std::string& window_name, unsigned char* img_data, int width, int height, int channels) {
    cv::Mat img(height, width, CV_8UC3, img_data);
    cv::imshow(window_name, img);
    cv::waitKey(10);
}

// cv::Mat tensor_to_mat2(const torch::Tensor& tensor) {

//     if (tensor.dim() != 3) {
//         std::cerr << "Expected a 3-dimensional tensor" << std::endl;
//         throw std::runtime_error("Expected a 3-dimensional tensor");
//     }

//     torch::Tensor local_tensor = tensor;


//     if (local_tensor.device().type() != torch::kCPU) {
//         local_tensor = local_tensor.to(torch::kCPU);
//     }


//     auto tensor_cont = local_tensor.is_contiguous() ? local_tensor : local_tensor.contiguous();


//     if (tensor_cont.dtype() == torch::kFloat32) {
//         tensor_cont = tensor_cont.mul(255).clamp(0, 255).to(torch::kUInt8);
//     } else if (tensor_cont.dtype() != torch::kUInt8) {
//         std::cerr << "Unsupported tensor datatype" << std::endl;
//         throw std::runtime_error("Unsupported tensor datatype");
//     }


//     tensor_cont = tensor_cont.permute({1, 2, 0});


//     int height = tensor_cont.size(0);
//     int width = tensor_cont.size(1);
//     int channels = tensor_cont.size(2);


//     if (channels != 3) {
//         std::cerr << "Expected 3 channels for RGB image, got " << channels << std::endl;
//         throw std::runtime_error("Expected 3 channels for RGB image");
//     }


//     cv::Mat output_mat(height, width, CV_8UC3, tensor_cont.data_ptr<uchar>());


//     return output_mat.clone();
// }

cv::Mat tensor_to_mat2(torch::Tensor tensor) {
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);
    if (channels != 3) {
        throw std::runtime_error("Expected 3 channels for RGB image");
    }
    cv::Mat mat(height, width, CV_8UC3, tensor.data_ptr());
    return mat.clone(); // clone to ensure data ownership
}
cv::Mat tensor_to_mat3(torch::Tensor tensor) {
    // Convert from float [0, 1] to uint8 [0, 255]
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);

    // Permute dimensions back from [channels, height, width] to [height, width, channels]
    tensor = tensor.permute({1, 2, 0}).contiguous();

    // Get the height, width, and channels
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);

    if (channels != 3) {
        throw std::runtime_error("Expected 3 channels for RGB image");
    }

    // Create a cv::Mat from the tensor
    cv::Mat mat(height, width, CV_8UC3, tensor.data_ptr());

    return mat.clone(); // Clone to ensure data ownership
}

// TODO: I have skipped the resolution for now.
Camera loadCam(const gs::param::ModelParameters& params, int id, CameraInfo& cam_info) {
    // Create a torch::Tensor from the image data

    // display_image("1", cam_info._img_data , cam_info._img_w, cam_info._img_h,cam_info._channels) ;
    std::cout << "Image Height: " << cam_info._img_h << ", Width: " << cam_info._img_w << ", Channels: " << cam_info._channels << std::endl;

    torch::Tensor original_image_tensor = torch::from_blob(cam_info._img_data,
                                                           {cam_info._img_h, cam_info._img_w, cam_info._channels},        // img size
                                                           {cam_info._img_w * cam_info._channels, cam_info._channels, 1}, // stride
                                                           torch::kUInt8);


    
    original_image_tensor = original_image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;
    // torch::Tensor original_image_tensor2=torch::clamp(original_image_tensor, 0.f, 1.f);
    // cv::Mat image_mat = tensor_to_mat3(original_image_tensor2);

    // cv::imshow("Concatenated Image", image_mat);
                


    // free_image(cam_info._img_data); // we dont longer need the image here.
      // Assure that we dont use the image data anymore.

    // cam_info._img_data = nullptr; 
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0,0) = cam_info._params[0];
    K(1,1) = cam_info._params[1];
    K(0,2) = cam_info._params[2];
    K(1,2) = cam_info._params[3];

    return Camera(cam_info._camera_ID, cam_info._R, cam_info._T,K, cam_info._fov_x, cam_info._fov_y, original_image_tensor,cam_info._image_name, id);
}