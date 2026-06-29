#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>

#include "image_augmentations.hpp"

void debugCout(std::ostream &os)
{
#ifdef DEBUG
    os << message;
#endif
}

const std::set<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"};
enum class AugmentationType
{
    RandomRotation,
    RandomTranslation,
    RandomCropResize,
    RandomBrightness,
    RandomBlurring,
};

std::ostream &operator<<(std::ostream &os, const AugmentationType &aug)
{
    switch (aug)
    {
        case AugmentationType::RandomRotation:
            os << "Random Rotation";
            break;
        case AugmentationType::RandomTranslation:
            os << "Random Translation";
            break;
        case AugmentationType::RandomCropResize:
            os << "Random Crop-Resize";
            break;
        case AugmentationType::RandomBrightness:
            os << "Random Brightness";
            break;
        case AugmentationType::RandomBlurring:
            os << "Random Blurring";
            break;
    }
    return os;
}

void applyAugmentation(const AugmentationType &augmentation, npp::ImageNPP_8u_C3 &input_image,
                       npp::ImageNPP_8u_C3 &output_image)
{
    switch (augmentation)
    {
        case AugmentationType::RandomRotation:
            randomRotate(input_image, output_image);
            break;
        case AugmentationType::RandomTranslation:
            randomTranslate(input_image, output_image);
            break;
        case AugmentationType::RandomCropResize:
            randomCropResize(input_image, output_image);
            break;
        case AugmentationType::RandomBrightness:
            randomBrightness(input_image, output_image);
            break;
        case AugmentationType::RandomBlurring:
            randomBlur(input_image, output_image, 0);  // kernel size will be randomly selected inside the blur function
            break;
    }
}

std::vector<AugmentationType> parseAugmentationTypes(int argc, char *argv[])
{
    std::vector<AugmentationType> augmentations;

    if (checkCmdLineFlag(argc, (const char **)argv, "rotate"))
    {
        augmentations.push_back(AugmentationType::RandomRotation);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "translate"))
    {
        augmentations.push_back(AugmentationType::RandomTranslation);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "crop_resize"))
    {
        augmentations.push_back(AugmentationType::RandomCropResize);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "brightness"))
    {
        augmentations.push_back(AugmentationType::RandomBrightness);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "blur"))
    {
        augmentations.push_back(AugmentationType::RandomBlurring);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "all_augmentations"))
    {
        augmentations = {AugmentationType::RandomRotation, AugmentationType::RandomTranslation,
                         AugmentationType::RandomCropResize, AugmentationType::RandomBrightness,
                         AugmentationType::RandomBlurring};
    }
    if (augmentations.empty())
    {
        std::cout << "No augmentations specified. Use command line flags to specify augmentations.\n";
        throw std::invalid_argument("No augmentations specified");
    }
    else
    {
        std::cout << "Using following user-specified augmentations: \n";
        for (const auto &aug : augmentations)
        {
            std::cout << aug << "\n";
        }
    }

    return augmentations;
}

std::pair<std::filesystem::path, std::filesystem::path> parseInputOutputDirectories(int argc, char *argv[])
{
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;

    if (checkCmdLineFlag(argc, (const char **)argv, "input_dir"))
    {
        char *input_dir_path;
        getCmdLineArgumentString(argc, (const char **)argv, "input_dir", &input_dir_path);
        input_dir = input_dir_path;
    }
    else
    {
        std::cout
            << "No input directory specified. Use --input_dir flag to specify the directory containing input images.\n";
        throw std::invalid_argument("No input directory specified");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "output_dir"))
    {
        char *output_dir_path;
        getCmdLineArgumentString(argc, (const char **)argv, "output_dir", &output_dir_path);
        output_dir = output_dir_path;
    }
    else
    {
        std::cout << "No output directory specified. Use --output_dir flag to specify the directory for saving "
                     "augmented images.\n";
        throw std::invalid_argument("No output directory specified");
    }

    if (output_dir == input_dir)
    {
        std::cout << "Output directory cannot be the same as input directory. Please specify different directories for "
                     "input and output.\n";
        throw std::invalid_argument("Output directory cannot be the same as input directory");
    }

    return {input_dir, output_dir};
}

size_t parseBatchSize(int argc, char *argv[])
{
    size_t batch_size = 4;  // Default batch size
    if (checkCmdLineFlag(argc, (const char **)argv, "batch_size"))
    {
        int parsed_batch_size = getCmdLineArgumentInt(argc, (const char **)argv, "batch_size");
        if (parsed_batch_size > 0)
        {
            batch_size = static_cast<size_t>(parsed_batch_size);
        }
        else
        {
            std::cout << "Invalid batch size specified. Using default batch size of 4.\n";
        }
    }
    return batch_size;
}


bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}


class ImageBatch : private std::vector<std::string>
{
  public:
    ImageBatch(size_t batch_size) : std::vector<std::string>(), batch_size_(batch_size) {}

    bool isFull() const { return size() >= batch_size_; }

    void addImage(const std::string &image_path)
    {
        if (isFull())
        {
            throw std::runtime_error(
                "You tried to add an image to a full batch. This should not happen. Check your code");
        }
        push_back(image_path);
    }

    size_t size() const { return std::vector<std::string>::size(); }
    bool empty() const { return std::vector<std::string>::empty(); }
    auto begin() const { return std::vector<std::string>::begin(); }
    auto end() const { return std::vector<std::string>::end(); }

  private:
    size_t batch_size_;
};

bool isAnImageFile(const std::filesystem::path file_path)
{
    std::string ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return valid_extensions.count(ext) > 0;
}

std::vector<ImageBatch> getImageBatchesFromDirectory(const std::filesystem::path images_dir, size_t batch_size)
{
    std::vector<ImageBatch> batches;
    ImageBatch temp_batch(batch_size);
    for (const auto &entry : std::filesystem::directory_iterator(images_dir))
    {
        if (entry.is_regular_file() and isAnImageFile(entry.path()))
        {
            temp_batch.addImage(entry.path().string());
            if (temp_batch.isFull())
            {
                batches.push_back(temp_batch);
                temp_batch = ImageBatch(batch_size);
            }
        }
    }

    // Add the last batch, which may not be full
    if (not temp_batch.empty())
    {
        batches.push_back(temp_batch);
    }

    size_t total_images = std::accumulate(batches.begin(), batches.end(), 0,
                                          [](size_t sum, const ImageBatch &batch) { return sum + batch.size(); }) +
                          temp_batch.size();

    std::cout << "Total images found in directory: " << total_images << "\n";
    std::cout << "Total batches created: " << batches.size() << "\n";

    return batches;
}

bool loadImageToDevice(const std::string &image_path, npp::ImageNPP_8u_C3 &device_image)
{
    cv::Mat host_image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (host_image.empty())
    {
        return false;
    }
    // std::cout << "Loaded image: " << image_path << " with size: " << host_image.cols << "x" << host_image.rows <<
    // "\n";
    device_image = npp::ImageNPP_8u_C3(host_image.cols, host_image.rows);
    device_image.copyFrom(host_image.data, host_image.step);

    return true;
}

void saveImageFromDevice(const std::string &output_path, const npp::ImageNPP_8u_C3 &device_image)
{
    cv::Mat host_image(device_image.height(), device_image.width(), CV_8UC3);
    device_image.copyTo(host_image.data, host_image.step);

    if (!cv::imwrite(output_path, host_image))
    {
        throw std::runtime_error("Failed to save image: " + output_path);
    }
}