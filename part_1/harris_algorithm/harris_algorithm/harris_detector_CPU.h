/*
 * Harris corners detector algorithm on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#include "bitmap_image.h"

void harrisDetectorCPU(const BitmapImage& inputImage, BYTE* cornersMap, float threshold);