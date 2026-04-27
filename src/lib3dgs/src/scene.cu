#include "camera.cuh"
#include "camera_utils.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"

// TODO: support start from later iterations. Compare original code
// We also have only training, no testing
// TODO: support also testing
#include <iostream>
using namespace std;

// std::unique_ptr<SceneInfo> read_realtime_scene_info() ;
// 0TODO read_realtime_scene_info
std::unique_ptr<SceneInfo> Scene::read_realtime_scene_info() 
{
    auto sceneInfos = std::make_unique<SceneInfo>();

    std::vector<CameraInfo> camera_infos;
    CameraInfo camera_info;

    std::cout <<"\033[41;32m "<< "_params Height: " <<_params.height << ", Width: " << _params.width<< ", Channels: " << _params.channels <<"\033[0m" << std::endl;    


    camera_info._camera_model = CAMERA_MODEL::PINHOLE;
    camera_info._params.resize(4);
    
    camera_info._params[0] = _params.fx;
    camera_info._params[1] = _params.fy;
    camera_info._params[2] = _params.cx;
    camera_info._params[3] = _params.cy;
    camera_info._width=_params.width;
    camera_info._height=_params.height;
    camera_info._img_w=_params.width;
    camera_info._img_h=_params.height;
    camera_info._fov_x= focal2fov(camera_info._params[0], _params.width);

    camera_info._fov_y=focal2fov(camera_info._params[1] , _params.height);
    

    camera_info._channels=_params.channels;
    std::cout << "camera_info Height: " <<camera_info._height << ", Width: " << camera_info._width<< ", Channels: " << camera_info._channels << std::endl;    
    // camera_info._params[0] = _params.fx*_params.scale;
    // camera_info._params[1] = _params.fy*_params.scale;
    // camera_info._params[2] = _params.cx*_params.scale;
    // camera_info._params[3] = _params.cy*_params.scale;

    // camera_info._fov_x= fx*scale;
    // camera_info._fov_y=fy*scale;

    // camera_info._width=_params.width*_params.scale;
    // camera_info._height=_params.height*_params.scale;
    // camera_info._img_w=_params.width*_params.scale;
    // camera_info._img_h=_params.height*_params.scale;
    // camera_info._channels=_params.channels;

    camera_info._img_data = static_cast<uint8_t*>(malloc(camera_info._width * camera_info._height * _params.channels));
    if (camera_info._img_data != nullptr) {
        for (int i = 0; i < camera_info._width  * camera_info._height * _params.channels; ++i) {
            camera_info._img_data[i] = 255;
        }
    }
    camera_info._R = Eigen::Matrix3f::Identity();
    camera_info._T = Eigen::Vector3f::Zero();


    camera_infos.push_back(camera_info);
    sceneInfos->_cameras =camera_infos;

    // sceneInfos->_cameras = init_first_camera();

    return sceneInfos;
}

void Scene::UpdateFirstCameraImage() {
    CameraInfo& cam_info = _scene_infos->_cameras[0];
    torch::Tensor new_image_tensor = torch::from_blob(cam_info._img_data,
                                                        {cam_info._img_h, cam_info._img_w, cam_info._channels},        // img size
                                                        {cam_info._img_w * cam_info._channels, cam_info._channels, 1}, // stride
                                                        torch::kUInt8);

 
    new_image_tensor = new_image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;

    if (!_cameras.empty()) {
        _cameras[0]._original_image = torch::clamp(new_image_tensor, 0.f, 1.f);
        _cameras[0]._image_width = _cameras[0]._original_image.size(2);
        _cameras[0]._image_height = _cameras[0]._original_image.size(1);
    } else {
        std::cerr << "No cameras available to update!" << std::endl;
    }


}

void Scene::UpdateFirstCameraPose(Eigen::Matrix3f R, Eigen::Vector3f T)
{
    if (!_cameras.empty()) 
    {
    _cameras[0]._world_view_transform = getWorld2View2(R, T, Eigen::Vector3f::Zero(), 1).to(torch::kCUDA, true);
    _cameras[0]._full_proj_transform =_cameras[0]._world_view_transform.unsqueeze(0).bmm(_cameras[0]._projection_matrix.unsqueeze(0)).squeeze(0);
    _cameras[0]._camera_center = _cameras[0]._world_view_transform.inverse()[3].slice(0, 0, 3);
    }

}

Scene::Scene(GaussianModel& gaussians, const gs::param::ModelParameters& params) : _gaussians(gaussians),
                                                                                   _params(params) {

    std::cout << "Scene Height: " << params.height << ", Width: " << params.width<< ", Channels: " << params.channels << std::endl;                       

    // Right now there is only support for colmap
    if(_params.type!=3)
    {
        if (std::filesystem::exists(_params.source_path)) 
        {

            if(_params.type==2)
            {
                cout << "\033[41;32m LIVO Mode模式读取场景数据\033[0m" << endl;
                _scene_infos = read_LIVO_scene_info(_params.source_path, _params.resolution);
            }
            else
            {
                cout << "\033[41;32m COLMAP Mode 模式读取场景数据\033[0m" << endl;
                _scene_infos = read_colmap_scene_info(_params.source_path, _params.resolution);
            }
        } else {
            std::cout << "Error: " << _params.source_path << " does not exist!" << std::endl;
            exit(-1);
        }
    }
    else{
        cout << "\033[41;32m Realtime 模式读取场景数据\033[0m" << endl;

        _scene_infos = read_realtime_scene_info();
    }

    _cameras.reserve(_scene_infos->_cameras.size());
    std::vector<nlohmann::json> json_cams;
    json_cams.reserve(_scene_infos->_cameras.size());
    int counter = 0;
    for (auto& cam_info : _scene_infos->_cameras) {

        _cameras.emplace_back(loadCam(_params, counter, cam_info));


        json_cams.push_back(Convert_camera_to_JSON(cam_info, counter, _cameras.back().Get_R(), _cameras.back().Get_T()));
        ++counter;
    }
    
    dump_JSON(params.output_path / "cameras.json", json_cams);
    // TODO: json camera dumping for debugging purpose at least

    // get the parameterr self.cameras.extent

    _gaussians.Create_from_pcd(_scene_infos->_point_cloud, _scene_infos->_nerf_norm_radius);
}