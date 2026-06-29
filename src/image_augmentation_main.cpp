
/** This is a toy image augmentation application, created for the coursera assignment and for understanding basic image
 * processing concepts using NVIDIA's NPP library. */

#include <string.h>

#include <random>
#include <ranges>
#include <set>

#include "image_augmentation_utils.hpp"

int main(int argc, char *argv[])
{
    try
    {
        findCudaDevice(argc, (const char **)argv);

        if (!printfNPPinfo(argc, argv))
        {
            return EXIT_SUCCESS;
        }

        const auto augmentations = parseAugmentationTypes(argc, argv);
        const auto [input_dir, output_dir] = parseInputOutputDirectories(argc, argv);
        size_t batch_size = parseBatchSize(argc, argv);
        auto image_batches = getImageBatchesFromDirectory(input_dir, batch_size);


        npp::ImageNPP_8u_C3 device_image;
        npp::ImageNPP_8u_C3 augmented_image;

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t batch_index = 0; batch_index < image_batches.size(); ++batch_index)
        {
            const auto &batch = image_batches[batch_index];
            const double progress = (batch_index + 1) * 100.0 / image_batches.size();

            // std::cout << "\033[2J\033[H";  // Clear the console screen and move the cursor to the top-left corner
            std::cout << "Processing batch: " << batch_index + 1 << " of " << image_batches.size() << " (" << progress
                      << "%)\n";

            // Todo: The so-called "batched" processing here is not really batched processing since each image in
            // the batch is still being augmented with a separate call. So this is only placeholder logic for now.
            // A future-improvement could be to implment own batched augmentation kernels, that apply single
            // augmentation to the whole batch in one go.
            for (const auto &image_path : batch)
            {
                if (!loadImageToDevice(image_path, device_image))
                {
                    std::cout << "Failed to load image: " << image_path << ". Skipping...\n";
                    continue;  // Skip this image if it failed to load
                }

                // Since we're using images of different sizes, we need to reallocate the augmented_image if the size of
                // the input image changes.
                if (augmented_image.size() != device_image.size())
                {
                    augmented_image = npp::ImageNPP_8u_C3(device_image.size());
                    // Zero out device memory to avoid uninitialized data
                    // cudaMemset(augmented_image.data(), 0, augmented_image.pitch() * augmented_image.height());
                }

                std::cout << "Processing image: " << image_path << " with size: " << device_image.width() << "x"
                          << device_image.height() << "\n";
                // Randomly select an augmentation to apply to the image - With equal probability for each augmentation
                size_t aug_index = std::uniform_int_distribution<size_t>(0, augmentations.size() - 1)(rng);
                const auto &augmentation = augmentations.at(aug_index);
                applyAugmentation(augmentation, device_image, augmented_image);
                std::ostringstream aug_name;
                aug_name << augmentation;
                std::cout << "Applied augmentation: " << aug_name.str() << "\n";

                cudaDeviceSynchronize();  // Ensure all CUDA operations are complete before saving the image
                saveImageFromDevice(
                    output_dir / (std::filesystem::path(image_path).filename().string() + "_" + aug_name.str()),
                    augmented_image);
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        std::cout << "Total processing time: " << elapsed_time.count() << " seconds\n";
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "NPP error: " << e << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
