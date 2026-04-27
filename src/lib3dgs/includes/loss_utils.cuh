// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <cmath>
#include <torch/torch.h>

namespace gaussian_splatting {
    static const float C1 = 0.01 * 0.01;
    static const float C2 = 0.03 * 0.03;


    float psnr_metric_with_mask(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) 
    {

        auto white_mask = (rendered_img >= 0.9).to(torch::kFloat32);
        auto black_mask = (rendered_img <= 0.1).to(torch::kFloat32);
        auto mask = 1 - (white_mask + black_mask);


        torch::Tensor valid_count = mask.view({rendered_img.size(0), -1}).sum(1, true);


        valid_count = torch::clamp(valid_count, /*min=*/1.0);


        torch::Tensor squared_diff = (rendered_img - gt_img).pow(2) * mask;
        torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).sum(1, true) / valid_count;


        float max_pixel_value = 1.0;
        torch::Tensor psnr_val = 20 * torch::log10(max_pixel_value / mse_val.sqrt());

        return psnr_val.mean().item<float>();
    }

    float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) 
    {

        torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
        torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);

        return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
    }

    torch::Tensor l1_loss(const torch::Tensor& network_output, const torch::Tensor& gt) {
        auto white_mask = (network_output >= 0.99).to(torch::kFloat32);
        auto black_mask = (network_output<= 0.01).to(torch::kFloat32);
        auto valid_mask = 1 - (white_mask + black_mask);
        auto loss = torch::abs(network_output - gt) * valid_mask;
        auto valid_count = valid_mask.sum();
        if (valid_count.item<float>() == 0) {
        return torch::tensor(0.0, torch::kFloat32);
        }
        else
        {
            return torch::abs((network_output - gt)).mean();
        }

    }

    // 1D Gaussian kernel
    torch::Tensor gaussian(int window_size, float sigma) {
        torch::Tensor gauss = torch::empty(window_size);
        for (int x = 0; x < window_size; ++x) {
            gauss[x] = std::exp(-(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) / (2.f * sigma * sigma));
        }
        return gauss / gauss.sum();
    }

    torch::Tensor create_window(int window_size, int channel) {
        auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
        auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
        return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
    }

    // Image Quality Assessment: From Error Visibility to
    // Structural Similarity (SSIM), Wang et al. 2004
    // The SSIM value lies between -1 and 1, where 1 means perfect similarity.
    // It's considered a better metric than mean squared error for perceptual image quality as it considers changes in structural information,
    // luminance, and contrast.
    torch::Tensor ssim(const torch::Tensor& img1, const torch::Tensor& img2, const torch::Tensor& window, int window_size, int channel) {
        auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
        auto mu1_sq = mu1.pow(2);
        auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_sq;

        auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
        auto mu2_sq = mu2.pow(2);
        auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu2_sq;

        auto mu1_mu2 = mu1 * mu2;
        auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_mu2;
        auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

        return ssim_map.mean();
    }
} // namespace gaussian_splatting
