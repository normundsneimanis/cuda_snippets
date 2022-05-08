/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#include "cuda_runtime.h"
#include "bitmap_image.h"

extern "C" cudaError_t medianFilterGrayscaleCUDA(const BitmapImage& inputImage, const BitmapImage& outputImage, int maxThreadPerBlock);