# Coursera: CUDA-at-Scale-For-The-Enterprise-Course-Project

## Overview
This is a toy image augmentation application, created for the coursera assignment and for understanding basic image
processing concepts using NVIDIA's NPP library. It is not exactly trying to be a practically useful image-augmentation 
pipeline at all. It's only purpose is to create an understandable context for the project as a toy application that uses
a couple of different NPP functions for a well-defined and commonly used purpose in ML pipelines. 

## Code Organization

The application main executeable is ```./src/image_augmentation_main.cpp```

The following headers are in the `include` folder:-
- `image_augmentation_utils.hpp`  (Arg Parsing, ImageIO, Augmentation-Enumerations and Miscallaneous Utils)
- `image_augmentations.hpp` (Actual Image augmentation functions.)

`dataset/` holds the dataset that the application runs on and also the `dataset/output` folder has the outputs produced 
by a single run of this application on the complete dataset.

## Building 
It's recommended (but not necessary) to build and compile the code inside docker-image provided in the `dockerfile`.
In the terminal run following from the root of this repo, to build the docker image for this project 

``` bash
# Note!! you should have atleast CUDA-drivers installed on your system for this. 
# You might also have to change the base image in this dockerfile depending on your
# GPU and it's compatible CUDA version.
docker build -f dockerfile -t coursera_project:latest . 
```
### Compiling the code:-
If you chose to compile and run everything inside the docker container, run the container first as follows:

```bash
docker run -it -v <path_to_this_repo>:/project --network=host --name=coursera_cuda_npp_project --ipc=host --gpus=all coursera_gpu_prog:latest
```

From inside the docker image you can run following to compile the code.:-
```bash
# from the repo root
sh build.sh
```
You can still use the above command to compile even outside the container, but you have to then install all dependencies from the `dockerfile` locally on your system.

### Download Dataset and results:
Download the toy dataset that I created from the following public google-drive link:-

``` https://drive.google.com/drive/folders/1dnyrth1-aLEJUjNY_QiF-TjLV3PAo3YC?usp=sharing ```

The dataset contains both the images that I worked on and the an `output` folder which shows the result from the application run. If you don't plan on 
building and running the application yourself, you can simply check the results in the `weather-dataset/output` folder.

### Run the application:
```bash
# from the repo root
sh run.sh
```

### Important command-line args
If you look inside `run.sh`, it simply runs the application in the following way:-
```bash
./build/image_augmentation_app --input_dir ./dataset/container_dataset/ --output_dir ./dataset/container_dataset/output/ --all_augmentations
```
Following are the important cmd-line args here:-

`--input_dir`: This is the path to where all your input images are. They can be either the dataset provided with this project or you can point it to your own path. (required arg)
`--output_dir`: The output path to save augmented images to. (required arg)

The other args are all for selecting the kind of image aumentations you want to apply. The application randomly applies (one augmentation per image) from your chosen set of augmentations with equal chance for each. The following options are valid augmentations:-
`--rotate` : Applies image rotation of a random angle between -5 and 5 degrees.

`--translate` : Applies image translation of random x an y offsets.

`--crop_resize`: Applies random-cropping of the image and resizes to original image size.

`--brightness`: Applies random brightness enhancement or reduction to the image.

`--blur`: Applies blur kernel choosing randomly between kernel sizes 3x3, 5x5, 7x7

`--all_augmentations`: Randomly Applies all of above (one augmentation per image) with equal chance for each.

If you don't provide the any of above augmentation args, the program would throw an exception.

