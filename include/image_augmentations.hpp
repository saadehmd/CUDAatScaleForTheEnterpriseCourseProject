/** Author: Saad Ahmad
 * Coursera Course: CUDA at Scale For the Enterprise, course project.
 * This file contains the implementation of various image augmentation functions using NVIDIA Performance Primitives
 * (NPP) library. The augmentations include random rotation, translation, cropping and resizing, brightness adjustment,
 * and blurring. Each function takes an input image and produces an output image with the specified augmentation
 * applied. The augmentations are designed to be applied to images in batches for efficient processing on CUDA-enabled
 * GPUs.
 */
#pragma once

#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>

#include <random>
#include <set>

const auto seed = std::random_device{}();
std::mt19937 rng(seed);

void randomRotate(npp::ImageNPP_8u_C3 &input_image, npp::ImageNPP_8u_C3 &output_image)
{
    int angle = std::uniform_int_distribution<int>(-5, 5)(rng);  // Random angle between -5 and 5 degrees
    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiRect img_roi = {0, 0, input_image.width(), input_image.height()};

    NPP_CHECK_NPP(nppiRotate_8u_C3R(input_image.data(), input_img_size, input_image.pitch(), img_roi,
                                    output_image.data(), output_image.pitch(), img_roi, angle, 0, 0,
                                    NPPI_INTER_LINEAR));
}

void randomTranslate(npp::ImageNPP_8u_C3 &input_image, npp::ImageNPP_8u_C3 &output_image)
{
    int x_offset = std::uniform_int_distribution<int>(-50, 50)(rng);  // Random x translation between -50 and 50 pixels
    int y_offset = std::uniform_int_distribution<int>(-10, 10)(rng);  // Random y translation between -10 and 10 pixels
    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiRect img_roi = {0, 0, input_image.width(), input_image.height()};

    // The name might seem misleading, but nppiRotate_8u_C3R can also be used for translation simply by setting the
    // rotation angle to 0 and providing x and y offsets.
    NPP_CHECK_NPP(nppiRotate_8u_C3R(input_image.data(), input_img_size, input_image.pitch(), img_roi,
                                    output_image.data(), output_image.pitch(), img_roi, 0, x_offset, y_offset,
                                    NPPI_INTER_LINEAR));
}

void randomCropResize(npp::ImageNPP_8u_C3 &input_image, npp::ImageNPP_8u_C3 &output_image)
{
    int img_width = input_image.width();
    int img_height = input_image.height();

    // Random crop width between 50% and 70% of original width
    int crop_width = std::uniform_int_distribution<int>(img_width / 2, (img_width * 7) / 10)(rng);
    // Random crop height between 50% and 70% of original height
    int crop_height = std::uniform_int_distribution<int>(img_height / 2, (img_height * 7) / 10)(rng);

    // Choose the crop-region origin so the entire crop stays within the source image.
    int x = std::uniform_int_distribution<int>(0, img_width - crop_width)(rng);
    int y = std::uniform_int_distribution<int>(0, img_height - crop_height)(rng);

    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiRect crop_roi = {x, y, crop_width, crop_height};
    NppiRect output_image_roi = {0, 0, output_image.width(), output_image.height()};
    NppiSize output_image_size = {output_image.width(), output_image.height()};
    NPP_CHECK_NPP(nppiResize_8u_C3R(input_image.data(), input_image.pitch(), input_img_size, crop_roi,
                                    output_image.data(), output_image.pitch(), output_image_size, output_image_roi,
                                    NPPI_INTER_LINEAR));
}

void randomBrightness(npp::ImageNPP_8u_C3 &input_image, npp::ImageNPP_8u_C3 &output_image)
{
    auto brightness_multipler = std::uniform_int_distribution<uint8_t>(1, 3)(rng);
    auto brightness_divider = std::uniform_int_distribution<uint8_t>(0, 2)(rng);
    NppiSize input_img_size = {input_image.width(), input_image.height()};
    const Npp8u constants[3] = {brightness_multipler, brightness_multipler, brightness_multipler};

    // Because NPP integer functions cannot accept floating-point constants, we use a combination of multiplication and
    // division i.e. integer fraction, to achieve scaling by a floating-point factor. The 'nConstant' argument in the
    // following function is the numerator of the scaling fraction and the 'nScaleFactor' argument is the denominator.
    NPP_CHECK_NPP(nppiMulC_8u_C3RSfs(input_image.data(), input_image.pitch(), constants, output_image.data(),
                                     output_image.pitch(), input_img_size, brightness_divider));
}

void randomBlur(npp::ImageNPP_8u_C3 &input_image, npp::ImageNPP_8u_C3 &output_image, int kernel_size)
{
    // Randomly select between 3x3 and 5x5 kernel size for blurring
    size_t kernel_size_idx = std::uniform_int_distribution<size_t>(0, 2)(rng);
    std::array<NppiMaskSize, 3> mask_sizes{NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_7_X_7};

    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiPoint src_offset = {0, 0};
    NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C3R(input_image.data(), input_image.pitch(), input_img_size, src_offset,
                                               output_image.data(), output_image.pitch(), input_img_size,
                                               mask_sizes[kernel_size_idx], NPP_BORDER_REPLICATE));
}
