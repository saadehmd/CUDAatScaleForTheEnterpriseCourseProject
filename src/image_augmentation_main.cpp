
/** This is a toy image augmentation application, created for the coursera assignment and for understanding basic image
 * processing concepts using NVIDIA's NPP library. */

#include <string.h>

#include <random>
#include <set>

#include "image_augmentation_utils.hpp"

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
            // Todo: The so-called "batched" processing here is not really batched processing since each image in
            // the batch is still being augmented with a separate call. So this is only placeholder logic for now.
            // A future-improvement could be to implment own batched augmentation kernels, that apply single
            // augmentation to the whole batch in one go.
            for (const auto &image_path : batch)
            {
                npp::ImageCPU_8u_C1 host_image;
                npp::loadImage(image_path, host_image);
                npp::ImageNPP_8u_C1 device_image(host_image);
                npp::ImageNPP_8u_C1 augmented_image(device_image.size());

                // Randomly select an augmentation to apply to the image - With equal probability for each augmentation
                size_t aug_index = std::uniform_int_distribution<size_t>(0, augmentations.size() - 1)(rng);
                const auto &augmentation = augmentations.at(aug_index);
                applyAugmentation(augmentation, device_image, augmented_image);

                npp::ImageCPU_8u_C1 host_augmented_image(augmented_image.size());
                augmented_image.copyTo(host_augmented_image.data(), host_augmented_image.pitch());
                std::ostringstream aug_name;
                aug_name << augmentation;
                npp::saveImage(
                    output_dir / (std::filesystem::path(image_path).filename().string() + "_" + aug_name.str()),
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
