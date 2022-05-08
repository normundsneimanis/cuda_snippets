/*
 * Harris corners detector algorithm on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#include <utility>
#include <exception>

#include "harris_detector_common.h"

struct ArgumentException : public std::exception {
   std::string message;
   
   ArgumentException(const std::string& s) : message(s) {}
   ~ArgumentException() throw() {}
   const char* what() const throw() { return message.c_str(); }
};

float gaussianFunction1D(int x, float sigma) {
	return exp(x * x / (-2 * sigma * sigma)) / sqrt(2 * PI * sigma);
}

float gaussianFunction2D(int x, int y, float sigma) {
	return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
}

void createGaussianKernel(float *kernel, int kernelSize, float sigma) {
	if (sigma <= 0) {
		throw ArgumentException("Invalid sigma for Gaussian kernel.");
	}
	
	// check for even size and for out of range
    if (((kernelSize % 2) == 0) || (kernelSize < 3) || (kernelSize > 101)) {
		throw ArgumentException("Wrong Gaussian kernel size.");
    }

    // radius
    const int r = kernelSize / 2;

	for (int x = -r, i = 0; i < kernelSize; x++, i++) {
		kernel[i] = gaussianFunction1D(x, sigma);
    }
}

void drawRectangleMarker(BYTE* image, int x, int y, const RGBA& color, int rectSize, int width, int height) {
	int l = x - rectSize / 2; 
	if (l < 0) {
		l = 0;
	}
	int r = x + rectSize / 2;
	if (r >= width) {
		r = width - 1;
	}
	
	int t = y - rectSize / 2; 	
	int b = y + rectSize / 2;	

	if (t >= 0) {
		for (int xx = l; xx <= r; xx++) { 
			*((RGBA*)&image[4 * (xx + t * width)]) = color;
		}
	}
	if (b < height) {
		for (int xx = l; xx <= r; xx++) { 
			*((RGBA*)&image[4 * (xx + b * width)]) = color;
		}
	}

	l = x - rectSize / 2; 
	r = x + rectSize / 2;
	if (t < 0) {
		t = 0;
	}	
	if (b >= height) {
		b = height - 1;
	}

	if (l >= 0) {
		for (int yy = t; yy <= b; yy++) {
			*((RGBA*)&image[4 * (l + yy * width)]) = color;
		}
	}
	if (r < width) {
		for (int yy = t; yy <= b; yy++) {
			*((RGBA*)&image[4 * (r + yy * width)]) = color;
		}
	}
}
