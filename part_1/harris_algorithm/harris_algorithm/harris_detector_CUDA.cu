/*
 * Harris corners detector algorithm on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#include "harris_detector_CUDA.h"

#include <iostream>
#include <cmath>

#include "device_launch_parameters.h"
#include "harris_detector_common.h"

__constant__ float gaussianKernel[KERNEL_LENGTH];

texture<BYTE> sourceTexture;
texture<float> tempTexture;

__global__ void partialDifferencesGPUKernel(int width, int height, float* diffx, float* diffy, float* diffxy) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int p = x + y * width;
	
	// Convolution with horizontal differentiation kernel mask
    float h = ((tex1Dfetch(sourceTexture, p - width + 1) + tex1Dfetch(sourceTexture, p + 1) + tex1Dfetch(sourceTexture, p + width + 1)) -
               (tex1Dfetch(sourceTexture, p - width - 1) + tex1Dfetch(sourceTexture, p - 1) + tex1Dfetch(sourceTexture, p + width - 1))) * 0.166666667f;
			
    // Convolution vertical differentiation kernel mask
    float v = ((tex1Dfetch(sourceTexture, p + width - 1) + tex1Dfetch(sourceTexture, p + width) + tex1Dfetch(sourceTexture, p + width + 1)) -
               (tex1Dfetch(sourceTexture, p - width - 1) + tex1Dfetch(sourceTexture, p - width) + tex1Dfetch(sourceTexture, p - width + 1))) * 0.166666667f;
			
    // Store squared differences directly
    diffx[p] = h * h;
    diffy[p] = v * v;
    diffxy[p] = h * v;
}

__global__ void convolutionColumnGPUKernel(float *dest, int width, int height) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
		
	float sum = 0;
	for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++){
		int dy = y + k;
		if (dy >= 0 && dy < height) {
			sum += tex1Dfetch(tempTexture, x + dy * width) * gaussianKernel[KERNEL_RADIUS - k];
		}
	}
	dest[x + y * width] = sum;
}

__global__ void convolutionRowGPUKernel(float *dest, int width, int height) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	float sum = 0;
	for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++){
		int dx = x + k;
		if (dx >= 0 && dx < width) {
			sum += tex1Dfetch(tempTexture, dx + y * width) * gaussianKernel[KERNEL_RADIUS - k];
		}
	}
	dest[x + y * width] = sum;
}

__global__ void harrisCornerResponseMapGPUKernel(const float *diffx, const float *diffy, const float *diffxy, float *map,
	                                             float threshold, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	const int p = x + y * width;

	const float A = diffx[p];
    const float B = diffy[p];
    const float C = diffxy[p];
	
    // Original Harris corner measure
    float M = (A * B - C * C) - (K * ((A + B) * (A + B)));      

    if (M > threshold) {
		map[p] = M; // insert value in the map
    } else {
		map[p] = 0;
	}
}

__global__ void suppressedNonMaximumPointOutputGPUKernel(BYTE *output, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int p = x + y * width;

	float currentValue = tex1Dfetch(tempTexture, p);

	// for each windows' row
    for (int i = -R; (currentValue != 0) && (i <= R); i++) {
		// for each windows' pixel
        for (int j = -R; j <= R; j++) {
			if (tex1Dfetch(tempTexture, p + j + i * width) > currentValue) {
				currentValue = 0;
                break;
            }
        }
    }
		
    // check if this point is really interesting
    output[p] = currentValue != 0 ? 1 : 0;
}

void convolve(const dim3& blocks, const dim3& threads, const cudaChannelFormatDesc& channelDesc, float *data, float *temp, int width, int height) {
	const size_t dataSize = width * height * sizeof(float);

	cudaBindTexture(NULL, tempTexture, data, channelDesc, dataSize);
	convolutionColumnGPUKernel<<<blocks, threads>>>(temp, width, height);
	cudaUnbindTexture(tempTexture);

	cudaBindTexture(NULL, tempTexture, temp, channelDesc, dataSize);
	convolutionRowGPUKernel<<<blocks, threads>>>(data, width, height);
	cudaUnbindTexture(tempTexture);
}

cudaError_t harrisDetectorCUDA(const BitmapImage& inputImage, BYTE* cornersMap, int maxThreadPerBlock, float threshold) {
	using namespace std;

	if (inputImage.getNumberOfColors() > MAX_PALETTE_SIZE) {
		cerr << "Unsupported number of colors: " << inputImage.getNumberOfColors();
		return (cudaError_t)0; 
	}

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	const int width = inputImage.getWidth();
	const int height = inputImage.getHeight();
	const size_t imageSize = width * height; 
	
	const int BLOCK_SIZE = sqrt((double)maxThreadPerBlock);

	dim3 blocks(ceil((double)width / BLOCK_SIZE), ceil((double)height  / BLOCK_SIZE));
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
        goto Error;
    }
		    
    // Allocate GPU buffers for vectors (input, output) 
	BYTE* dev_output;
    cudaStatus = cudaMalloc(&dev_output, imageSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

	BYTE* dev_input;
	cudaStatus = cudaMalloc(&dev_input, imageSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, inputImage.getRawData(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy for input failed!" << endl;
        goto Error;
    }

	// Create Gaussian kernek
	{
		float kernel[KERNEL_LENGTH];
		createGaussianKernel(kernel, KERNEL_LENGTH, SIGMA);		
		cudaMemcpyToSymbol(gaussianKernel, kernel, KERNEL_LENGTH * sizeof(float), 0, cudaMemcpyHostToDevice);
	}
	
	// Create texture
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<BYTE>();		
		cudaBindTexture(NULL, sourceTexture, dev_input, channelDesc, imageSize);
	}
		
	// 1. Calculate partial differences
	float* diffx;
	cudaStatus = cudaMalloc(&diffx, imageSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

	float* diffy;
	cudaStatus = cudaMalloc(&diffy, imageSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

	float* diffxy;
	cudaStatus = cudaMalloc(&diffxy, imageSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

	partialDifferencesGPUKernel<<<blocks, threads>>>(width, height, diffx, diffy, diffxy);
	
	cudaUnbindTexture(sourceTexture); 

	// 2. Smooth the diff images
	float *temp;
	cudaStatus = cudaMalloc(&temp, imageSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc failed!";
		goto Error;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	
	convolve(blocks, threads, channelDesc, diffx, temp, width, height);
	convolve(blocks, threads, channelDesc, diffy, temp, width, height);
	convolve(blocks, threads, channelDesc, diffxy, temp, width, height);
	
	// 3. Compute Harris Corner Response Map
	harrisCornerResponseMapGPUKernel<<<blocks, threads>>>(diffx, diffy, diffxy, temp, threshold, width, height); 
	
	cudaFree(diffx);
	cudaFree(diffy);
	cudaFree(diffxy);

	// 4. Suppression and output of non-maximum points
	cudaBindTexture(NULL, tempTexture, temp, channelDesc, imageSize * sizeof(float));

	suppressedNonMaximumPointOutputGPUKernel<<<blocks, threads>>>(dev_output, width, height);

	cudaUnbindTexture(tempTexture);
	
	cudaFree(temp);

	//----------------
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);	

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) { // any errors encountered during the launch.
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
	cout << "Elapsed Harris algorithm execution time on GPU: " << elapsedTime << " milliseconds" << endl;
	cout << "---------------------" << endl;	

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cornersMap, dev_output, imageSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy for output failed!"  << endl;
        goto Error;
    }
		
Error:
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return cudaStatus;
}