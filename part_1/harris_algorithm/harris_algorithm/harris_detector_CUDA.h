/*
 * Harris corners detector algorithm on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#include "cuda_runtime.h"
#include "bitmap_image.h"

cudaError_t harrisDetectorCUDA(const BitmapImage& inputImage, BYTE* cornersMap, int maxThreadPerBlock, float threshold);