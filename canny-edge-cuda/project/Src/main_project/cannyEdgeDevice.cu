#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cannyEdgeDevice.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128 

CannyEdgeDevice::CannyEdgeDevice(float *h_image, int width, int height) {
	this->width = width;
	this->height = height;
	this->highThreshold = 0.0f;
	this->lowThreshold = 0.0f;
	this->totTime = 0.0f;

	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * width * height));
	checkCudaErrors(cudaMemcpy(this->d_image, h_image, sizeof(float) * width * height, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void**)&d_filterImage, sizeof(float) * width * height));
	checkCudaErrors(cudaMalloc((void**)&d_gradientX, sizeof(float) * width * height));
	checkCudaErrors(cudaMalloc((void**)&d_gradientY, sizeof(float) * width * height));
	checkCudaErrors(cudaMalloc((void**)&d_gradientMag, sizeof(float) * width * height));
	checkCudaErrors(cudaMalloc((void**)&d_nonMaxSup, sizeof(float) * width * height));
	checkCudaErrors(cudaMalloc((void**)&d_highThreshHyst, sizeof(float) * width * height));
	checkCudaErrors(cudaMalloc((void**)&d_lowThreshHyst, sizeof(float) * width * height));

	checkCudaErrors(cudaMalloc((void**)&d_gaussianKernel, sizeof(float) * GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_sobelKernelX, sizeof(float) * SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_sobelKernelY, sizeof(float) * SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE));

	initializeGaussianKernel();
	initializeSobelFilters();
}

CannyEdgeDevice::~CannyEdgeDevice() {
	if (d_image != NULL) checkCudaErrors(cudaFree(d_image));
	if (d_filterImage != NULL) checkCudaErrors(cudaFree(d_filterImage));
	if (d_gradientX != NULL) checkCudaErrors(cudaFree(d_gradientX));
	if (d_gradientY != NULL) checkCudaErrors(cudaFree(d_gradientY));
	if (d_gradientMag != NULL) checkCudaErrors(cudaFree(d_gradientMag));
	if (d_nonMaxSup != NULL) checkCudaErrors(cudaFree(d_nonMaxSup));
	if (d_highThreshHyst != NULL) checkCudaErrors(cudaFree(d_highThreshHyst));
	if (d_lowThreshHyst != NULL) checkCudaErrors(cudaFree(d_lowThreshHyst));
	if (d_gaussianKernel != NULL) checkCudaErrors(cudaFree(d_gaussianKernel));
	if (d_sobelKernelX != NULL) checkCudaErrors(cudaFree(d_sobelKernelX));
	if (d_sobelKernelY != NULL) checkCudaErrors(cudaFree(d_sobelKernelY));
}

__global__ void initializeGaussian(float *d_gaussianKernel) {
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	
	if (ix < GAUSSIAN_KERNEL_SIZE && iy < GAUSSIAN_KERNEL_SIZE) {
		int index = iy * GAUSSIAN_KERNEL_SIZE + ix;
		float stddev = powf((float)(GAUSSIAN_KERNEL_SIZE / 3), 2);
		
		float x = powf(fabsf((float)(ix - (GAUSSIAN_KERNEL_SIZE / 2))), 2.0f);
		float y = powf(fabsf((float)(iy - (GAUSSIAN_KERNEL_SIZE / 2))), 2.0f);
		d_gaussianKernel[index] = (1 / (2 * (22 / 7) * stddev)) * expf((-(x + y) / (2 * stddev)));
	}
}

__global__ void initializeSobel(float *d_sobelKernelX, float *d_sobelKernelY) {
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int weight = SOBEL_KERNEL_SIZE / 2;

	if (ix < SOBEL_KERNEL_SIZE && iy < SOBEL_KERNEL_SIZE) {
		int index = iy * SOBEL_KERNEL_SIZE + ix;
		float sx = ix - SOBEL_KERNEL_SIZE / 2;
		float sy = iy - SOBEL_KERNEL_SIZE / 2;
		float norm = sx * sx + sy *sy;

		if (norm == 0.0f) {
			d_sobelKernelX[index] = 0.0f;
			d_sobelKernelY[index] = 0.0f;
		}
		else {
			d_sobelKernelX[index] = sx * weight / norm;
			d_sobelKernelY[index] = sy * weight / norm;
		}
	}
}

__global__ void convolution(float *d_image, float *kernel, float *d_result, int width, int height, int kernelSize) {
	const int sharedMemWidth = TILE_WIDTH + MAX(SOBEL_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE) - 1;
	__shared__ float sharedMem[sharedMemWidth][sharedMemWidth];

	int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
	int destY = dest / sharedMemWidth;
	int destX = dest % sharedMemWidth;
	int srcY = blockIdx.y * TILE_WIDTH + destY - (kernelSize / 2);
	int srcX = blockIdx.x * TILE_WIDTH + destX - (kernelSize / 2);
	int src = (srcY * width + srcX);
	if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
		sharedMem[destY][destX] = d_image[src];
	else
		sharedMem[destY][destX] = 0;

	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
	destY = dest / sharedMemWidth;
	destX = dest % sharedMemWidth;
	srcY = blockIdx.y * TILE_WIDTH + destY - (kernelSize / 2);
	srcX = blockIdx.x * TILE_WIDTH + destX - (kernelSize / 2);
	src = (srcY * width + srcX);
	if (destY < sharedMemWidth) {
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			sharedMem[destY][destX] = d_image[src];
		else
			sharedMem[destY][destX] = 0;
	}
	__syncthreads();

	float accum = 0;
	for (int j = 0; j < kernelSize; j++)
		for (int i = 0; i < kernelSize; i++)
			accum += sharedMem[threadIdx.y + j][threadIdx.x + i] * kernel[j * kernelSize + i];
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
	if (x < width && y < height)
		d_result[y * width + x] = (fminf(fmaxf((accum), 0.0), 1.0));
	__syncthreads();
}

__global__ void magnitudeImage(float *d_gradientX, float *d_gradientY, float *d_gradientMag, int width, int height) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < width && iy < height) {
		int idx = iy * width + ix;
		d_gradientMag[idx] = sqrtf(powf(d_gradientX[idx], 2.0f) + powf(d_gradientY[idx], 2.0f));
	}
}

__global__ void nonMaxSuppressionDevice(int width, int height, float *d_gradientX, float *d_gradientY, float* d_gradientMag, float* d_nonMax) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < width && iy < height) {
		int tid = iy * width + ix;

		float d_gradientMag_tid = d_gradientMag[tid];
		float d_gradientMag_tid_next = d_gradientMag[tid + 1];
		float d_gradientMag_tid_prev = d_gradientMag[tid - 1];
		float d_gradientMag_tid_width_next = d_gradientMag[tid + width + 1];
		float d_gradientMag_tid_width_prev = d_gradientMag[tid - width - 1];
		float d_gradientMag_tid_width_plus = d_gradientMag[tid + width];
		float d_gradientMag_tid_width_minus = d_gradientMag[tid - width];
		float d_gradientMag_tid_width_minus_next = d_gradientMag[tid - width + 1];
		float d_gradientMag_tid_width_plus_prev = d_gradientMag[tid + width - 1];

		float d_gradientXT = d_gradientX[tid];
		float d_gradientYT = d_gradientY[tid];

		float tanYX;
		float magB, magA;

		if ((tid < width) || (tid >= ((height - 1) * width))) // Top and Bottom Edge
			d_nonMax[tid] = 0;
		else if ((tid % width == 0) || (tid % width == (width - 1))) // Left and Right Edge
			d_nonMax[tid] = 0;
		else {
			if (d_gradientMag_tid == 0)
				d_nonMax[tid] = 0;
			else if (d_gradientXT >= 0) { // Direction East
				if (d_gradientYT >= 0) { // Direction South-East
					if (d_gradientXT >= d_gradientYT) { // East of South-East direction
						tanYX = (float)(d_gradientYT / d_gradientXT);

						magA = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_next);
						magB = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_plus_prev);
					}
					else { // South of South-East direction
						tanYX = (float)(d_gradientXT / d_gradientYT);

						magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_next);
						magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_prev);
					}
				}
				else { // Direction North-East
					if (d_gradientXT >= (-1 * d_gradientYT)) { // East of North-East direction
						tanYX = (float)((-1 * d_gradientYT) / d_gradientXT);

						magA = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_minus_next);
						magB = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_plus_prev);
					}
					else { // North of North-East direction	
						tanYX = (float)(d_gradientXT / (-1 * d_gradientYT));

						magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_plus_prev);
						magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_minus_next);
					}
				}
			}
			else { // Direction West
				if (d_gradientYT >= 0) { // Direction South-West
					if (d_gradientYT >= (-1 * d_gradientXT)) { // South of South-West direction
						tanYX = (float)((-1 * d_gradientXT) / d_gradientYT);
						magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_plus_prev);
						magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_minus_next);
					}
					else { // West of South-West direction
						tanYX = (float)(d_gradientYT / (-1 * d_gradientXT));
						magA = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_plus_prev);
						magB = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_minus_next);
					}
				}
				else { // Direction North-West
					if (d_gradientYT >= d_gradientXT) { // West of North-West direction
						tanYX = (float)(d_gradientYT / d_gradientXT);
						magA = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_prev);
						magB = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_next);
					}
					else {// North of North-West direction
						tanYX = (float)(d_gradientXT / d_gradientYT);
						magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_next);
						magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_prev);
					}
				}
			}

			if ((d_gradientMag_tid < magA) || (d_gradientMag_tid < magB))
				d_nonMax[tid] = 0;
			else
				d_nonMax[tid] = d_gradientMag_tid;
		}
	}
}

__global__ void computeSum(float *d_filteredImage, float *d_imageSumGrid, unsigned int n)
{
	__shared__ float smem[SMEM_SIZE];
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	float localSum = 0;

	if (idx + 3 * blockDim.x < n)
	{
		float a1 = d_filteredImage[idx];
		float a2 = d_filteredImage[idx + blockDim.x];
		float a3 = d_filteredImage[idx + 2 * blockDim.x];
		float a4 = d_filteredImage[idx + 3 * blockDim.x];
		localSum = a1 + a2 + a3 + a4;
	}

	smem[tid] = localSum;
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512) 
		smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) 
		smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) 
		smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64) 
		smem[tid] += smem[tid + 64];
	__syncthreads();

	if (tid < 32)
	{
		volatile float *vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) d_imageSumGrid[blockIdx.x] = smem[0];
}

__global__ void highHysterisis(int width, int height, float* d_nonMax, float highThreshold, float *d_highThreshHyst) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < width && iy < height) {
		int tid = iy * width + ix;

		d_highThreshHyst[tid] = 0.0f;
		if(d_nonMax[tid] > highThreshold)
			d_highThreshHyst[tid] = 1.0f;
	}
}

__global__ void lowHysterisis(int width, int height, float *d_nonMax, float* d_highThreshHyst, float lowThreshold, float *d_lowThreshHyst) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((ix > 0) && (ix < (width - 1)) && (iy > 0) && iy < (height - 1)) {
		int tid = iy * width + ix;

		d_lowThreshHyst[tid] = d_highThreshHyst[tid];

		if (d_highThreshHyst[tid] == 1) {
			// Determine neighbour indices
			int eastN = tid + 1;
			int westN = tid - 1;
			int northN = tid - width;
			int southN = tid + width;

			int southEastN = southN + 1;
			int northEastN = northN	+ 1;
			int southWestN = southN - 1;
			int northWestN = northN	- 1;

			if (d_nonMax[eastN] > lowThreshold)
				d_lowThreshHyst[eastN] = 1.0f;

			if (d_nonMax[westN] > lowThreshold)
				d_lowThreshHyst[westN] = 1.0f;

			if (d_nonMax[northN] > lowThreshold)
				d_lowThreshHyst[northN] = 1.0f;

			if (d_nonMax[southN] > lowThreshold)
				d_lowThreshHyst[southN] = 1.0f;

			if (d_nonMax[southEastN] > lowThreshold)
				d_lowThreshHyst[southEastN] = 1.0f;

			if (d_nonMax[northEastN] > lowThreshold)
				d_lowThreshHyst[northEastN] = 1.0f;

			if (d_nonMax[southWestN] > lowThreshold)
				d_lowThreshHyst[southWestN] = 1.0f;

			if (d_nonMax[northWestN] > lowThreshold)
				d_lowThreshHyst[northWestN] = 1.0f;
		}
	}
}


void CannyEdgeDevice::initializeGaussianKernel() {
	dim3 block(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);
	initializeGaussian << <1, block >> > (d_gaussianKernel);
}

void CannyEdgeDevice::initializeSobelFilters() {
	dim3 block(SOBEL_KERNEL_SIZE, SOBEL_KERNEL_SIZE);
	initializeSobel << <1, block >> > (d_sobelKernelX, d_sobelKernelY);
}

void CannyEdgeDevice::performGaussianFiltering() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 grid(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH));
	dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
	convolution << <grid, block >> >(d_image, d_gaussianKernel, d_filterImage, width, height, GAUSSIAN_KERNEL_SIZE);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Gaussian Smoothening completed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::performImageGradientX() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 dimGrid(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	convolution << <dimGrid, dimBlock >> >(d_filterImage, d_sobelKernelX, d_gradientX, width, height, SOBEL_KERNEL_SIZE);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Image Grdient in X direction computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::performImageGradientY() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 dimGrid(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	convolution << <dimGrid, dimBlock >> >(d_filterImage, d_sobelKernelY, d_gradientY, width, height, SOBEL_KERNEL_SIZE);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Image Grdient in Y direction computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::computeMagnitude() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 block(32, 4);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	magnitudeImage << <grid, block >> > (d_gradientX, d_gradientY, d_gradientMag, width, height);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Image Grdient magnitude computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::nonMaxSuppression() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 block(32, 2);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	nonMaxSuppressionDevice << <grid, block >> > (width, height, d_gradientX, d_gradientY, d_gradientMag, d_nonMaxSup);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Non Max Suppression computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::computeCannyThresholds() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	float imageSum = 0.0f;
	float *d_imageSumGrid = NULL;
	dim3 block(SMEM_SIZE, 1);
	dim3 grid((width * height + block.x - 1) / block.x, 1);

	checkCudaErrors(cudaMalloc((void **)&d_imageSumGrid, grid.x * sizeof(float)));
	float *h_imageSumGrid = (float *)malloc(grid.x * sizeof(float));

	computeSum << <grid.x, block >> >(d_filterImage, d_imageSumGrid, width * height);
	checkCudaErrors(cudaMemcpy(h_imageSumGrid, d_imageSumGrid, grid.x * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < grid.x; i++) 
		imageSum += h_imageSumGrid[i];

	lowThreshold = 0.66f * (imageSum / (width * height));
	highThreshold = 1.33f * (imageSum / (width * height));

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Thresholds computed - (lowThreshold, highThreshold): (%f, %f) in %f ms\n", lowThreshold, highThreshold, msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::highHysterisisThresholding() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 block(32, 2);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	highHysterisis << <grid, block >> > (width, height, d_nonMaxSup, highThreshold, d_highThreshHyst);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device High Threshold Hysterisis computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeDevice::lowHysterisisThresholding() {
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	dim3 block(32, 2);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	lowHysterisis << <grid, block >> > (width, height, d_nonMaxSup, d_highThreshHyst, lowThreshold, d_lowThreshHyst);

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTime, start, stop));

	printf("Device Low Threshold Hysterisis computed in %f ms\n", msecTime);
	totTime += msecTime;
}

float *CannyEdgeDevice::getD_gaussianKernel() {
	return d_gaussianKernel;
}

float *CannyEdgeDevice::getD_sobelKernelX() {
	return d_sobelKernelX;
}

float *CannyEdgeDevice::getD_sobelKernelY() {
	return d_sobelKernelY;
}

float *CannyEdgeDevice::getD_FilterImage() {
	return d_filterImage;
}

float *CannyEdgeDevice::getD_GradientX() {
	return d_gradientX;
}

float *CannyEdgeDevice::getD_GradientY() {
	return d_gradientY;
}

float *CannyEdgeDevice::getD_gradientMag() {
	return d_gradientMag;
}

float *CannyEdgeDevice::getD_nonMaxSup() {
	return d_nonMaxSup;
}

float CannyEdgeDevice::getLowThreshold() {
	return lowThreshold;
}

float CannyEdgeDevice::getHighThreshold() {
	return highThreshold;
}

float *CannyEdgeDevice::getD_HighThreshold() {
	return d_highThreshHyst;
}

float *CannyEdgeDevice::getD_LowThreshold() {
	return d_lowThreshHyst;
}

float CannyEdgeDevice::getTotTime() {
	return totTime;
}

