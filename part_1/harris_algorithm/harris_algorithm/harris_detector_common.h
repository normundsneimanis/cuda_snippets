/*
 * Harris corners detector algorithm on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#include <string>
#include "bitmap_image.h"

#define PI	3.14159265358979323846f

// Harris parameter
#define K 	0.04f

// Non-maximum suppression parameters
#define R	3

// Gaussian smoothing parameter
#define KERNEL_RADIUS	3
#define KERNEL_LENGTH	(2 * KERNEL_RADIUS + 1)
#define SIGMA	1.4f

typedef std::pair<int, int> IntPoint;

void createGaussianKernel(float* kernel, int kernelSize, float sigma);

void drawRectangleMarker(BYTE* image, int x, int y, const RGBA& color, int rectSize, int width, int height);


