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

void randomRotate(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    int angle = std::uniform_int_distribution<int>(-45, 45)(rng);  // Random angle between -45 and 45 degrees
    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiRect img_roi = {0, 0, input_image.width(), input_image.height()};
    nppiRotate_8u_C1R(input_image.data(), input_img_size, input_image.pitch(), img_roi, output_image.data(),
                      output_image.pitch(), img_roi, angle, 0, 0, NPPI_INTER_LINEAR);
}

void randomTranslate(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    int x_offset = std::uniform_int_distribution<int>(-50, 50)(rng);  // Random x translation between -50 and 50 pixels
    int y_offset = std::uniform_int_distribution<int>(-10, 10)(rng);  // Random y translation between -10 and 10 pixels
    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiRect img_roi = {0, 0, input_image.width(), input_image.height()};

    // The name might seem misleading, but nppiRotate_8u_C1R can also be used for translation simply by setting the
    // rotation angle to 0 and providing x and y offsets.
    nppiRotate_8u_C1R(input_image.data(), input_img_size, input_image.pitch(), img_roi, output_image.data(),
                      output_image.pitch(), img_roi, 0, x_offset, y_offset, NPPI_INTER_LINEAR);
}

void randomCropResize(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    int img_width = input_image.width();
    int img_height = input_image.height();

    // Random crop width between 25% and 50% of original width
    int crop_width = std::uniform_int_distribution<int>(img_width / 4, img_width / 2)(rng);
    // Random crop height between 25% and 50% of original height
    int crop_height = std::uniform_int_distribution<int>(img_height / 4, img_height / 2)(rng);

    // Random x,y coordinates for crop-region top-left corner, ensuring the crop region fits within the original image.
    int x = std::uniform_int_distribution<int>(crop_width, img_width - crop_width)(rng);
    int y = std::uniform_int_distribution<int>(crop_height, img_height - crop_height)(rng);

    NppiSize input_img_size = {input_image.width(), input_image.height()};
    NppiRect crop_roi = {x, y, crop_width, crop_height};
    NppiRect output_image_roi = {0, 0, output_image.width(), output_image.height()};
    NppiSize output_image_size = {output_image.width(), output_image.height()};
    nppiResize_8u_C1R(input_image.data(), input_image.pitch(), input_img_size, crop_roi, output_image.data(),
                      output_image.pitch(), output_image_size, output_image_roi, NPPI_INTER_LINEAR);
}

void randomBrightness(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    auto brightness_multipler = std::uniform_int_distribution<uint8_t>(1, 3)(rng);
    auto brightness_divider = std::uniform_int_distribution<uint8_t>(0, 2)(rng);
    NppiSize input_img_size = {input_image.width(), input_image.height()};

    // Because NPP integer functions cannot accept floating-point constants, we use a combination of multiplication and
    // division to achieve scaling by a floating-point factor. The 'nConstant' argument in the following function is the
    // numerator of the scaling fraction and the 'nScaleFactor' argument is the denominator.
    nppiMulC_8u_C1RSfs(input_image.data(), input_image.pitch(), brightness_multipler, output_image.data(),
                       output_image.pitch(), input_img_size, brightness_divider);
}

void randomBlur(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image, int kernel_size)
{
    // Randomly select between 3x3 and 5x5 kernel size for blurring
    size_t kernel_size_idx = std::uniform_int_distribution<size_t>(0, 2)(rng);
    std::array<NppiMaskSize, 3> mask_sizes{NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_7_X_7};

    NppiSize input_img_size = {input_image.width(), input_image.height()};
    nppiFilterGauss_8u_C1R(input_image.data(), input_image.pitch(), output_image.data(), output_image.pitch(),
                           input_img_size, mask_sizes[kernel_size_idx]);
}
