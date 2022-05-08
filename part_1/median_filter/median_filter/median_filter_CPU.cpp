/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#include "median_filter_CPU.h"

#include <ctime>
#include "median_filter_common.h"

#include <iostream>

void medianFilter8BPPPalette(const BYTE* inputImage, const DWORD* paletteColors, BYTE* outputImage, int width, int height) {
	BYTE window[WINDOW_SIZE];

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int i = 0;
			for (int xx = x - EDGE_SIZE; xx <= x + EDGE_SIZE; xx++) {
				for (int yy = y - EDGE_SIZE; yy <= y + EDGE_SIZE; yy++) {
					if (0 <= xx && xx < width && 0 <= yy && yy < height) { // check boundaries
						window[i++] = inputImage[yy * width + xx];
					} else {
						window[i++] = 0;
					}
				}
			}

			// bubble-sort
			for (int i = 0; i < WINDOW_SIZE; i++) {
				for (int j = i + 1; j < WINDOW_SIZE; j++) {
					if (paletteColors[window[i]] > paletteColors[window[j]]) { 
						BYTE tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}

			// pick the middle one
			outputImage[y * width + x] = window[WINDOW_SIZE / 2];
		}
	}
}

void medianFilterRGBA(const DWORD* inputImage, DWORD* outputImage, int width, int height) {
	DWORD window[WINDOW_SIZE];

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int i = 0;
			for (int xx = x - EDGE_SIZE; xx <= x + EDGE_SIZE; xx++) {
				for (int yy = y - EDGE_SIZE; yy <= y + EDGE_SIZE; yy++) {
					if (0 <= xx && xx < width && 0 <= yy && yy < height) { // check boundaries
						window[i++] = inputImage[yy * width + xx];
					} else {
						window[i++] = 0;
					}
				}
			}

			// bubble-sort
			for (int i = 0; i < WINDOW_SIZE; i++) {
				for (int j = i + 1; j < WINDOW_SIZE; j++) {
					if (window[i] > window[j]) { 
						DWORD tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}

			// pick the middle one
			outputImage[y * width + x] = window[WINDOW_SIZE / 2];
		}
	}
}

bool medianFilterGrayscaleCPU(const BitmapImage& inputImage, const BitmapImage& outputImage) {
	using namespace std;

	clock_t begin = clock();
	
	if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
		medianFilter8BPPPalette(inputImage.getRawData(), (DWORD*)inputImage.getPaletteColors(), outputImage.getRawData(), inputImage.getWidth(), inputImage.getHeight());
	} else {
		medianFilterRGBA((DWORD*)inputImage.getRawData(), (DWORD*)outputImage.getRawData(), inputImage.getWidth(), inputImage.getHeight());
	}
		
	clock_t end = clock();
	float elapsedTime = float(end - begin) / CLOCKS_PER_SEC;
	cout << "---------------------" << endl;
	cout << "Elapsed image processing time on CPU: " << elapsedTime << " seconds" << endl;
	cout << "---------------------" << endl;
	
	return true;
}
