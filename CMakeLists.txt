cmake_minimum_required(VERSION 3.18)
project(MyNppProject LANGUAGES CXX)

# Find the toolkit and its libraries
find_package(CUDAToolkit REQUIRED)
#find_package(FreeImage REQUIRED)

# 1. Search for the FreeImage header file
find_path(FREEIMAGE_INCLUDE_DIR FreeImage.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

# 2. Search for the FreeImage binary library file
find_library(FREEIMAGE_LIBRARY NAMES freeimage freeimaged
    PATHS /usr/lib /usr/local/lib /opt/local/lib
)

# 3. Verify that both files were actually found
if(FREEIMAGE_INCLUDE_DIR AND FREEIMAGE_LIBRARY)
    message(STATUS "Found FreeImage: ${FREEIMAGE_LIBRARY}")
else()
    message(FATAL_ERROR "Could NOT find FreeImage library or header!")
endif()

#add_executable(imageRotationNPP src/imageRotationNPP.cpp)
add_executable(image_augmentation_app src/image_augmentation_main.cpp)
# Link the NPP targets created by find_package(CUDAToolkit)
# -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc

target_link_libraries(image_augmentation_app 
    PRIVATE 
        CUDA::cudart
        #CUDA::nppcore
        CUDA::nppig
        CUDA::nppim
        CUDA::nppitc
        CUDA::nppial
        CUDA::nppicc
        CUDA::nppidei
        CUDA::nppist
        CUDA::nppisu
        CUDA::nppc
        CUDA::nppif
        ${FREEIMAGE_LIBRARY}
)


set(npp_util_headers ${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common/UtilNPP)
set(cuda_util_headers ${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common)
target_include_directories(image_augmentation_app PRIVATE 
                           ${CUDAToolkit_INCLUDE_DIRS} 
                           ${CUDAToolkit_INCLUDE_DIRS}/npp 
                           ${npp_util_headers} 
                           ${cuda_util_headers} 
                           ${FREEIMAGE_INCLUDE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/include
                           )

#print the CUDA include directories for debugging
message(STATUS "CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")