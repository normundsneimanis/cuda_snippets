/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#include "median_filter_CUDA.h"

#include <iostream>
#include <cmath>

#include "device_launch_parameters.h"
#include "median_filter_common.h"

//#define BLOCK_SIZE	32

__constant__ DWORD palette[MAX_PALETTE_SIZE];

texture<BYTE> byteTexture;
texture<DWORD> dwordTexture;

__global__ void medianFilter8BPPPaletteKernel(BYTE* output, const BYTE* input, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > width || y > height) {
		return;
	}
	
    BYTE window[WINDOW_SIZE];

	int i = 0;
    for (int xx = x - EDGE_SIZE; xx <= x + EDGE_SIZE; xx++) {
        for (int yy = y - EDGE_SIZE; yy <= y + EDGE_SIZE; yy++) {
            if (0 <= xx && xx < width && 0 <= yy && yy < height) { // check boundaries
                window[i++] = tex1Dfetch(byteTexture, yy * width + xx);
			} else {
				window[i++] = tex1Dfetch(byteTexture, y * width + x);
			}
        }
    }

    // bubble-sort
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = i + 1; j < WINDOW_SIZE; j++) {
            if (palette[window[i]] > palette[window[j]]) { 
                BYTE tmp = window[i];
                window[i] = window[j];
                window[j] = tmp;
            }
        }
    }

    // pick the middle one
	output[y * width + x] = window[WINDOW_SIZE / 2];
}

__global__ void medianFilterRGBAKernel(DWORD* output, const DWORD* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > width || y > height) {
		return;
	}
	
    DWORD window[WINDOW_SIZE];

	int i = 0;
    for (int xx = x - EDGE_SIZE; xx <= x + EDGE_SIZE; xx++) {
        for (int yy = y - EDGE_SIZE; yy <= y + EDGE_SIZE; yy++) {
            if (0 <= xx && xx < width && 0 <= yy && yy < height) { // check boundaries
                window[i++] = tex1Dfetch(dwordTexture, yy * width + xx);
				//window[i++] = input[yy * width + xx];
			} else {
				window[i++] = tex1Dfetch(dwordTexture, y * width + x);
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
	output[y * width + x] = window[WINDOW_SIZE / 2];
}

extern "C" cudaError_t medianFilterGrayscaleCUDA(const BitmapImage& inputImage, const BitmapImage& outputImage, int maxThreadPerBlock) {
	using namespace std;
	
	const size_t dataSize = inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE ? inputImage.getWidth() * inputImage.getHeight() : inputImage.getWidth() * inputImage.getHeight() * sizeof(RGBA); 
	
	const int BLOCK_SIZE = sqrt((double)maxThreadPerBlock);

	dim3 blocks(ceil((double)inputImage.getWidth() / BLOCK_SIZE), ceil((double)inputImage.getHeight()  / BLOCK_SIZE));
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    BYTE* dev_input = NULL;
    BYTE* dev_output = NULL;
	
	cudaEvent_t start, stop;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
        goto Error;
    }
	
    // Allocate GPU buffers for vectors (input, output)    .
    cudaStatus = cudaMalloc(&dev_output, dataSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_input, dataSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, inputImage.getRawData(), dataSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy for input failed!" << endl;
        goto Error;
    }
	
	if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
		cudaMemcpyToSymbol(palette, inputImage.getPaletteColors(), inputImage.getNumberOfColors() * sizeof(RGBA), 0, cudaMemcpyHostToDevice);
		
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<BYTE>();		
	    cudaBindTexture(NULL, byteTexture, dev_input, desc, dataSize);
	} else {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<DWORD>();		
		cudaBindTexture(NULL, dwordTexture, dev_input, desc, dataSize);
	}	

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);	
	
	if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {		
		medianFilter8BPPPaletteKernel<<<blocks, threads>>>(dev_output, dev_input, inputImage.getWidth(), inputImage.getHeight());
	} else {
		medianFilterRGBAKernel<<<blocks, threads>>>((DWORD*)dev_output, (DWORD*)dev_input, inputImage.getWidth(), inputImage.getHeight());
	}	

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);	

	// any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching kernel!" << endl;;
        goto Error;
    }

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		 cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << endl;
		 goto Error;
	}

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "---------------------" << endl;
	cout << "Elapsed image processing time on GPU: " << elapsedTime << " milliseconds" << endl;
	cout << "---------------------" << endl;	

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outputImage.getRawData(), dev_output, dataSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy for output failed!"  << endl;
        goto Error;
    }

	if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
		cudaUnbindTexture(byteTexture); 
	} else {
		cudaUnbindTexture(dwordTexture); 
	}

Error:
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return cudaStatus;
}