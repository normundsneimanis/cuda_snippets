/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#include "bitmap_image.h"

bool medianFilterGrayscaleCPU(const BitmapImage& inputImage, const BitmapImage& outputImage);