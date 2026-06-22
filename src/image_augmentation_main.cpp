
/** This is a toy image augmentation application, created for the coursera assignment and for understanding basic image
 * processing concepts using NVIDIA's NPP library. */

#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>
#include <string.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

const std::set<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"};
const auto seed = std::random_device{}();
std::mt19937 rng(seed);

void rotate(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    double angle = std::uniform_real_distribution<double>(-45.0, 45.0)(rng);  // Random angle between -45 and 45 degrees
}

void scale(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    double scale_factor =
        std::uniform_real_distribution<double>(0.5, 1.5)(rng);  // Random scale factor between 0.5 and 1.5
}

void translate(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    int x_offset = std::uniform_int_distribution<int>(-50, 50)(rng);  // Random x translation between -50 and 50 pixels
    int y_offset = std::uniform_int_distribution<int>(-10, 10)(rng);  // Random y translation between -10 and 10 pixels
}

void crop(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    int img_width = input_image.width();
    int img_height = input_image.height();

    // Random crop width between 25% and 50% of original width
    int width = std::uniform_int_distribution<int>(img_width / 4, img_width / 2)(rng);
    // Random crop height between 25% and 50% of original height
    int height = std::uniform_int_distribution<int>(img_height / 4, img_height / 2)(rng);
    int x = std::uniform_int_distribution<int>(0, img_width - width)(rng);    // Random x coordinate for crop
    int y = std::uniform_int_distribution<int>(0, img_height - height)(rng);  // Random y coordinate for crop
}

void randomBrightness(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image)
{
    double brightness_factor =
        std::uniform_real_distribution<double>(0.5, 1.5)(rng);  // Random brightness factor between 0.5 and 1.5
}

void blur(npp::ImageNPP_8u_C1 &input_image, npp::ImageNPP_8u_C1 &output_image, int kernel_size)
{
    // Randomly select between 3x3 and 5x5 kernel size for blurring
    kernel_size = std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? 3 : 5;
}

enum class AugmentationType
{
    Rotation,
    Scaling,
    Translation,
    Cropping,
    RandomBrightness,
    Blurring,
};

std::ostream &operator<<(std::ostream &os, const AugmentationType &aug)
{
    switch (aug)
    {
        case AugmentationType::Rotation:
            os << "Rotation";
            break;
        case AugmentationType::Scaling:
            os << "Scaling";
            break;
        case AugmentationType::Translation:
            os << "Translation";
            break;
        case AugmentationType::Cropping:
            os << "Cropping";
            break;
        case AugmentationType::RandomBrightness:
            os << "Random Brightness";
            break;
        case AugmentationType::Blurring:
            os << "Blurring";
            break;
    }
    return os;
}

void applyAugmentation(const AugmentationType &augmentation, npp::ImageNPP_8u_C1 &input_image,
                       npp::ImageNPP_8u_C1 &output_image)
{
    switch (augmentation)
    {
        case AugmentationType::Rotation:
            rotate(input_image, output_image);
            break;
        case AugmentationType::Scaling:
            scale(input_image, output_image);
            break;
        case AugmentationType::Translation:
            translate(input_image, output_image);
            break;
        case AugmentationType::Cropping:
            crop(input_image, output_image);
            break;
        case AugmentationType::RandomBrightness:
            randomBrightness(input_image, output_image);
            break;
        case AugmentationType::Blurring:
            blur(input_image, output_image, 0);  // kernel size will be randomly selected inside the blur function
            break;
    }
}

std::vector<AugmentationType> parseAugmentationTypes(int argc, char *argv[])
{
    std::vector<AugmentationType> augmentations;

    if (checkCmdLineFlag(argc, (const char **)argv, "rotate"))
    {
        augmentations.push_back(AugmentationType::Rotation);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "scale"))
    {
        augmentations.push_back(AugmentationType::Scaling);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "translate"))
    {
        augmentations.push_back(AugmentationType::Translation);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "crop"))
    {
        augmentations.push_back(AugmentationType::Cropping);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "brightness"))
    {
        augmentations.push_back(AugmentationType::RandomBrightness);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "blur"))
    {
        augmentations.push_back(AugmentationType::Blurring);
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
    ImageBatch(size_t batch_size) : std::vector<std::string>(batch_size), batch_size_(batch_size) {}

    bool isFull() const { return size() >= batch_size_; }

    void addImage(const std::string &image_path)
    {
        if (isFull())
        {
            throw std::runtime_error("Batch is already full");
        }
        push_back(image_path);
    }

    auto begin() const { return std::vector<std::string>::begin(); }

    auto end() const { return std::vector<std::string>::end(); }

  private:
    size_t batch_size_;
};

bool isAnImageFile(const std::filesystem::path file_path)
{
    return valid_extensions.count(file_path.extension().string()) > 0;
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
    return batches;
}

int main(int argc, char *argv[])
{
    if (!printfNPPinfo(argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    try
    {
        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        const auto augmentations = parseAugmentationTypes(argc, argv);
        const auto [input_dir, output_dir] = parseInputOutputDirectories(argc, argv);
        size_t batch_size = parseBatchSize(argc, argv);
        auto image_batches = getImageBatchesFromDirectory(input_dir, batch_size);

        for (const auto &batch : image_batches)
        {
            for (const auto &image_path : batch)
            {
                npp::ImageCPU_8u_C1 host_image;
                npp::loadImage(image_path, host_image);
                npp::ImageNPP_8u_C1 device_image(host_image);
                npp::ImageNPP_8u_C1 augmented_image(device_image.size());

                // Randomly select an augmentation to apply to the image - With equal probability for each augmentation
                size_t aug_index = std::uniform_int_distribution<size_t>(0, augmentations.size() - 1)(rng);
                applyAugmentation(augmentations.at(aug_index), device_image, augmented_image);

                npp::ImageCPU_8u_C1 host_augmented_image(augmented_image.size());
                augmented_image.copyTo(host_augmented_image.data(), host_augmented_image.pitch());
                npp::saveImage(output_dir / (std::filesystem::path(image_path).filename().string()),
                               host_augmented_image);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
