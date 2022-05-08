/*
 * Harris corners detector algorithm on GPU 
 * Arnis Lektauers 2019 Riga Technical University
 */
#include "harris_detector_CPU.h"

#include <cmath>
#include <ctime>
#include <iostream>

#include "harris_detector_common.h"

void convolve(float* ptrImage, int width, int height, float* ptrTemp, float* kernel) {
    float* src = ptrImage + KERNEL_RADIUS;
    float* tmp = ptrTemp + KERNEL_RADIUS;

    for (int y = 0; y < height; y++) {
		for (int x = KERNEL_RADIUS; x < width - KERNEL_RADIUS; x++, src++, tmp++) {
			float v = 0;
            for (int k = 0; k < KERNEL_LENGTH; k++) {
				v += src[k - KERNEL_RADIUS] * kernel[k];
			}
            *tmp = v;
        }
        src += 2 * KERNEL_RADIUS;
        tmp += 2 * KERNEL_RADIUS;
    }

	for (int x = 0; x < width; x++) {
		for (int y = KERNEL_RADIUS; y < height - KERNEL_RADIUS; y++) {
			src = ptrImage + y * width + x;
            tmp = ptrTemp + y * width + x;

            float v = 0;
            for (int k = 0; k < KERNEL_LENGTH; k++) {
				v += tmp[width * (k - KERNEL_RADIUS)] * kernel[k];
            }
			*src = v;
        }
    }
}

void harrisCornersDetector(const BitmapImage& inputImage, BYTE* cornersMap, float threshold) {
	BitmapImage *grayImage = const_cast<BitmapImage*>(&inputImage);
		
	// get source image size
    const int width = grayImage->getWidth();
    const int height = grayImage->getHeight();

	memset(cornersMap, 0, height * width);
        	
	// 1. Calculate partial differences
	float *diffx = new float[height * width];
	memset(diffx, 0, height * width * sizeof(float));

    float *diffy = new float[height * width];
	memset(diffy, 0, height * width * sizeof(float));

    float *diffxy = new float[height * width];
	memset(diffxy, 0, height * width * sizeof(float));
	
    BYTE* src = grayImage->getRawData();
	
    // for each line
    for (int y = 1; y < height - 1; y++) {
		// for each pixel
        for (int x = 1; x < width - 1; x++) {
			int p = x + y * width;

			// Convolution with horizontal differentiation kernel mask
            float h = ((src[p - width + 1] + src[p + 1] + src[p + width + 1]) -
                       (src[p - width - 1] + src[p - 1] + src[p + width - 1])) * 0.166666667f;
			
            // Convolution vertical differentiation kernel mask
            float v = ((src[p + width - 1] + src[p + width] + src[p + width + 1]) -
                       (src[p - width - 1] + src[p - width] + src[p - width + 1])) * 0.166666667f;
			
            // Store squared differences directly
            diffx[p] = h * h;
            diffy[p] = v * v;
            diffxy[p] = h * v;
        }
    }

    // Free some resources which wont be needed anymore
	if (!inputImage.getPaletteColors()) {
		delete grayImage;
	}
	
	// 2. Smooth the diff images
    {
		float kernel[KERNEL_LENGTH];
		createGaussianKernel(kernel, KERNEL_LENGTH, SIGMA);

		float *temp = new float[height * width];
		memset(temp, 0, height * width * sizeof(float));

        // Convolve with Gaussian kernel
        convolve(diffx, width, height, temp, kernel);
        convolve(diffy, width, height, temp, kernel);
        convolve(diffxy, width, height, temp, kernel);

		delete[] temp;
    }
	
	// 3. Compute Harris Corner Response Map
    float *map = new float[height * width];
	memset(map, 0, height * width * sizeof(float));
	
    float M, A, B, C;

    for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int p = x + y * width;

            A = diffx[p];
            B = diffy[p];
            C = diffxy[p];

            // Original Harris corner measure
            M = (A * B - C * C) - (K * ((A + B) * (A + B)));      

            if (M > threshold) {
				map[p] = M; // insert value in the map
            } 
        }
    }
	
	delete[] diffx;
	delete[] diffy;
	delete[] diffxy;
    
    // 4. Suppress non-maximum points
   
	// for each row
    for (int y = R, maxY = height - R; y < maxY; y++) {
		// for each pixel
        for (int x = R, maxX = width - R; x < maxX; x++) {
            float currentValue = map[x + y * width];

            // for each windows' row
            for (int i = -R; (currentValue != 0) && (i <= R); i++) {
                // for each windows' pixel
                for (int j = -R; j <= R; j++) {
					if (map[x + j + (y + i) * width] > currentValue) {
                        currentValue = 0;
                        break;
                    }
                }
            }

            // check if this point is really interesting
            if (currentValue != 0) {
				cornersMap[x + y * width] = 1;
			}
        }
    }
	
	delete[] map;
}

void harrisDetectorCPU(const BitmapImage& inputImage, BYTE* cornersMap, float threshold) {
	using namespace std;

	clock_t begin = clock();
	
	harrisCornersDetector(inputImage, cornersMap, threshold);
	
	clock_t end = clock();
	float elapsedTime = float(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "---------------------" << endl;
	cout << "Elapsed Harris algorithm execution time on CPU: " << elapsedTime << " milliseconds" << endl;
	cout << "---------------------" << endl;
}
