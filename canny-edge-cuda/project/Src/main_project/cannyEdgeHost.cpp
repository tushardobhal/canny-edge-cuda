#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include<chrono>

#include "helper.h"
#include "cannyEdgeHost.h"

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5

# define START_TIMER auto start = std::chrono::high_resolution_clock::now()
# define STOP_TIMER auto finish = std::chrono::high_resolution_clock::now()
# define CALC_DURATION std::chrono::duration<double> elapsed = finish - start

CannyEdgeHost::CannyEdgeHost(float *h_image, int width, int height) {
	this->width = width;
	this->height = height;
	this->highThreshold = 0.0f;
	this->lowThreshold = 0.0f;
	this->totTime = 0.0f;

	this->h_image = (float *)malloc(width * height * sizeof(float));
	memcpy(this->h_image, h_image, width * height * sizeof(float));

	h_filterImage = (float *)malloc(width * height * sizeof(float));
	h_gradientX = (float *)malloc(width * height * sizeof(float));
	h_gradientY = (float *)malloc(width * height * sizeof(float));
	h_gradientMag = (float *)malloc(width * height * sizeof(float));
	h_nonMaxSup = (float *)malloc(width * height * sizeof(float));
	h_highThreshHyst = (float *)malloc(width * height * sizeof(float));
	h_lowThreshHyst = (float *)malloc(width * height * sizeof(float));

	h_gaussianKernel = (float *)malloc(GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE * sizeof(float));
	h_sobelKernelX = (float *)malloc(SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float));
	h_sobelKernelY = (float *)malloc(SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float));

	initializeGaussianKernel();
	initializeSobelFilters();
}

CannyEdgeHost::~CannyEdgeHost() {
	if (h_image != NULL) free(h_image);
	if (h_filterImage != NULL) free(h_filterImage);
	if (h_gradientX != NULL) free(h_gradientX);
	if (h_gradientY != NULL) free(h_gradientY);
	if (h_gradientMag != NULL) free(h_gradientMag);
	if (h_nonMaxSup != NULL) free(h_nonMaxSup);
	if (h_highThreshHyst != NULL) free(h_highThreshHyst);
	if (h_lowThreshHyst != NULL) free(h_lowThreshHyst);
	if (h_gaussianKernel != NULL) free(h_gaussianKernel);
	if (h_sobelKernelX != NULL) free(h_sobelKernelX);
	if (h_sobelKernelY != NULL) free(h_sobelKernelY);
}

void CannyEdgeHost::initializeGaussianKernel() {
	float stddev = pow((float)(GAUSSIAN_KERNEL_SIZE / 3), 2);
	for (int i = 0; i < GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE; i++) {
		int ix = i % GAUSSIAN_KERNEL_SIZE;
		int iy = i / GAUSSIAN_KERNEL_SIZE;
		int index = iy * GAUSSIAN_KERNEL_SIZE + ix;

		float x = pow(abs((float)(ix - (GAUSSIAN_KERNEL_SIZE / 2))), 2.0f);
		float y = pow(abs((float)(iy - (GAUSSIAN_KERNEL_SIZE / 2))), 2.0f);
		h_gaussianKernel[index] = (1 / (2 * (22 / 7) * stddev)) * exp((-(x + y) / (2 * stddev)));
	}
}

void CannyEdgeHost::initializeSobelFilters() {
	int weight = SOBEL_KERNEL_SIZE / 2;

	for (int i = 0; i < SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE; i++) {
		int ix = i % SOBEL_KERNEL_SIZE;
		int iy = i / SOBEL_KERNEL_SIZE;

		float sx = ix - SOBEL_KERNEL_SIZE / 2;
		float sy = iy - SOBEL_KERNEL_SIZE / 2;
		float norm = sx * sx + sy *sy;

		if (norm == 0.0f) {
			h_sobelKernelX[i] = 0.0f;
			h_sobelKernelY[i] = 0.0f;
		}
		else {
			h_sobelKernelX[i] = sx * weight / norm;
			h_sobelKernelY[i] = sy * weight / norm;
		}
	}
}


void CannyEdgeHost::convolution(float *h_image, float *kernel, float *h_result, int kernelSize) {
	for (int iy = 0; iy < height; iy++) {
		for (int ix = 0; ix < width; ix++) {
			float sum = 0.0f;
			for (int m = 0; m < kernelSize; m++) {
				int flipped_ky = kernelSize - 1 - m;

				for (int n = 0; n < kernelSize; n++) {
					int flipped_kx = kernelSize - 1 - n;
					int srcY = iy + ((kernelSize / 2) - flipped_ky);
					int srcX = ix + ((kernelSize / 2) - flipped_kx);

					if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height)
						sum += h_image[srcY * width + srcX] * kernel[flipped_ky * kernelSize + flipped_kx];
				}
			}
			h_result[iy * width + ix] = fminf(fmaxf(sum, 0.0f), 1.0f);
		}
	}
}

void CannyEdgeHost::magnitudeImage() {
	for (int iy = 0; iy < height; iy++) {
		for (int ix = 0; ix < width; ix++) {
			int idx = iy * width + ix;
			h_gradientMag[idx] = sqrt(pow(h_gradientX[idx], 2.0f) + pow(h_gradientY[idx], 2.0f));
		}
	}
}

void CannyEdgeHost::nonMaxSuppressionHost() {
	for (int iy = 0; iy < height; iy++) {
		for (int ix = 0; ix < width; ix++) {
			int tid = iy * width + ix;

			float h_gradientXT = h_gradientX[tid];
			float h_gradientYT = h_gradientY[tid];
			float diffh_gradientXY = h_gradientXT - h_gradientYT;

			float tanYX = 0.0f;
			float magB = 0.0f, magA = 0.0f;

			float diffImage = 0.0f;

			if ((tid < width) || (tid >= ((height - 1) * width))) // Top and Bottom Edge
				h_nonMaxSup[tid] = 0;
			else if ((tid % width == 0) || (tid % width == (width - 1))) // Left and Right Edge
				h_nonMaxSup[tid] = 0;
			else {
				if (h_gradientMag[tid] == 0)
					h_nonMaxSup[tid] = 0;
				else if (h_gradientXT >= 0) { // Direction East
					if (h_gradientYT >= 0) { // Direction South-East
						if (h_gradientXT >= h_gradientYT) { // East of South-East direction
							tanYX = (float)(h_gradientYT / h_gradientXT);

							magA = ((1 - tanYX) * h_gradientMag[tid + 1]) + (tanYX * h_gradientMag[tid + width + 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid - 1]) + (tanYX * h_gradientMag[tid + width - 1]);
						}
						else { // South of South-East direction
							tanYX = (float)(h_gradientXT / h_gradientYT);

							magA = ((1 - tanYX) * h_gradientMag[tid + width]) + (tanYX * h_gradientMag[tid + width + 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid - width]) + (tanYX * h_gradientMag[tid - width - 1]);
						}
					}
					else { // Direction North-East
						if (h_gradientXT >= (-1 * h_gradientYT)) { // East of North-East direction
							tanYX = (float)((-1 * h_gradientYT) / h_gradientXT);

							magA = ((1 - tanYX) * h_gradientMag[tid + 1]) + (tanYX * h_gradientMag[tid - width + 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid - 1]) + (tanYX * h_gradientMag[tid + width - 1]);
						}
						else { // North of North-East direction	
							tanYX = (float)(h_gradientXT / (-1 * h_gradientYT));

							magA = ((1 - tanYX) * h_gradientMag[tid + width]) + (tanYX * h_gradientMag[tid + width - 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid - width]) + (tanYX * h_gradientMag[tid - width + 1]);
						}
					}
				}
				else { // Direction West
					if (h_gradientYT >= 0) { // Direction South-West
						if (h_gradientYT >= (-1 * h_gradientXT)) { // South of South-West direction
							tanYX = (float)((-1 * h_gradientXT) / h_gradientYT);
							magA = ((1 - tanYX) * h_gradientMag[tid + width]) + (tanYX * h_gradientMag[tid + width - 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid - width]) + (tanYX * h_gradientMag[tid - width + 1]);
						}
						else { // West of South-West direction
							tanYX = (float)(h_gradientYT / (-1 * h_gradientXT));
							magA = ((1 - tanYX) * h_gradientMag[tid - 1]) + (tanYX * h_gradientMag[tid + width - 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid + 1]) + (tanYX * h_gradientMag[tid - width + 1]);
						}
					}
					else { // Direction North-West
						if (h_gradientYT >= h_gradientXT) { // West of North-West direction
							tanYX = (float)(h_gradientYT / h_gradientXT);
							magA = ((1 - tanYX) * h_gradientMag[tid - 1]) + (tanYX * h_gradientMag[tid - width - 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid + 1]) + (tanYX * h_gradientMag[tid + width + 1]);
						}
						else {// North of North-West direction
							tanYX = (float)(h_gradientXT / h_gradientYT);
							magA = ((1 - tanYX) * h_gradientMag[tid + width]) + (tanYX * h_gradientMag[tid + width + 1]);
							magB = ((1 - tanYX) * h_gradientMag[tid - width]) + (tanYX * h_gradientMag[tid - width - 1]);
						}
					}
				}

				if ((h_gradientMag[tid] < magA) || (h_gradientMag[tid] < magB))
					h_nonMaxSup[tid] = 0;
				else
					h_nonMaxSup[tid] = h_gradientMag[tid];
			}
		}
	}
}

void CannyEdgeHost::highHysterisisThresholdingHost(int width, int height, float *h_nonMaxSup, float highThreshold, float *h_highThreshHyst) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int addr = ((i * width) + j);

			if (h_nonMaxSup[addr] > highThreshold)
				h_highThreshHyst[addr] = 1.0f;
			else
				h_highThreshHyst[addr] = 0.0f;
		}
	}
}

void CannyEdgeHost::lowHysterisisThresholdingHost(int width, int height, float *h_nonMax, float lowThreshold, float *h_highThreshHyst, float *h_lowThreshHyst) {
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			int addr = ((i * width) + j);

			h_lowThreshHyst[addr] = h_highThreshHyst[addr];

			if (h_highThreshHyst[addr] = 1) {
				// Determine neighbour indices
				int northN = addr - width;
				int southN = addr + width;
				int eastN = addr + 1;
				int westN = addr - 1;

				int southEastN = southN + 1;
				int southWestN = southN - 1;
				int northEastN = northN + 1;
				int northWestN = northN - 1;

				if (h_nonMax[eastN] > lowThreshold)
					h_lowThreshHyst[eastN] = 1.0f;

				if (h_nonMax[westN] > lowThreshold)
					h_lowThreshHyst[westN] = 1.0f;

				if (h_nonMax[northN] > lowThreshold)
					h_lowThreshHyst[northN] = 1.0f;

				if (h_nonMax[southN] > lowThreshold)
					h_lowThreshHyst[southN] = 1.0f;

				if (h_nonMax[southEastN] > lowThreshold)
					h_lowThreshHyst[southEastN] = 1.0f;

				if (h_nonMax[northEastN] > lowThreshold)
					h_lowThreshHyst[northEastN] = 1.0f;

				if (h_nonMax[southWestN] > lowThreshold)
					h_lowThreshHyst[southWestN] = 1.0f;

				if (h_nonMax[northWestN] > lowThreshold)
					h_lowThreshHyst[northWestN] = 1.0f;
			}
		}
	}
}

void CannyEdgeHost::performGaussianFiltering() {
	START_TIMER;

	convolution(h_image, h_gaussianKernel, h_filterImage, GAUSSIAN_KERNEL_SIZE);

	STOP_TIMER;

	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Gaussian Smoothening completed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::performImageGradientX() {
	START_TIMER;

	convolution(h_filterImage, h_sobelKernelX, h_gradientX, SOBEL_KERNEL_SIZE);

	STOP_TIMER;

	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Image Grdient in X direction computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::performImageGradientY() {
	START_TIMER;

	convolution(h_filterImage, h_sobelKernelY, h_gradientY, SOBEL_KERNEL_SIZE);

	STOP_TIMER;

	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Image Grdient in Y direction computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::computeMagnitude() {
	START_TIMER;

	magnitudeImage();

	STOP_TIMER;

	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Image Grdient magnitude computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::nonMaxSuppression() {
	START_TIMER;

	nonMaxSuppressionHost();

	STOP_TIMER;

	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Non Max Suppression computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::computeCannyThresholds() {
	START_TIMER;

	float imageSum = 0.0f;
	for (int ix = 0; ix < width; ix++) {
		for (int iy = 0; iy < height; iy++) {
			imageSum += h_filterImage[iy * width + ix];
		}
	}

	lowThreshold = 0.66f * (imageSum / (width * height));
	highThreshold = 1.33f * (imageSum / (width * height));

	STOP_TIMER;
	
	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Thresholds computed - (lowThreshold, highThreshold): (%f, %f) in %f ms\n", lowThreshold, highThreshold, msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::highHysterisisThresholding() {
	START_TIMER;

	highHysterisisThresholdingHost(width, height, h_nonMaxSup, highThreshold, h_highThreshHyst);

	STOP_TIMER;

	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host High Hysterisis Thresholding computed in %f ms\n", msecTime);
	totTime += msecTime;
}

void CannyEdgeHost::lowHysterisisThresholding() {
	START_TIMER;

	lowHysterisisThresholdingHost(width, height, h_nonMaxSup, lowThreshold, h_highThreshHyst, h_lowThreshHyst);

	STOP_TIMER;
	
	CALC_DURATION;
	double msecTime = elapsed.count() * 1000;
	printf("Host Low Hysterisis Thresholding computed in %f ms\n", msecTime);
	totTime += msecTime;

}

float *CannyEdgeHost::getH_gaussianKernel() {
	return h_gaussianKernel;
}

float *CannyEdgeHost::getH_sobelKernelX() {
	return h_sobelKernelX;
}

float *CannyEdgeHost::getH_sobelKernelY() {
	return h_sobelKernelY;
}

float *CannyEdgeHost::getH_FilterImage() {
	return h_filterImage;
}

float *CannyEdgeHost::getH_GradientX() {
	return h_gradientX;
}

float *CannyEdgeHost::getH_GradientY() {
	return h_gradientY;
}

float *CannyEdgeHost::getH_gradientMag() {
	return h_gradientMag;
}

float *CannyEdgeHost::getH_nonMaxSup() {
	return h_nonMaxSup;
}

float CannyEdgeHost::getLowThreshold() {
	return lowThreshold;
}

float CannyEdgeHost::getHighThreshold() {
	return highThreshold;
}

float *CannyEdgeHost::getH_HighThreshold() {
	return h_highThreshHyst;
}

float *CannyEdgeHost::getH_LowThreshold() {
	return h_lowThreshHyst;
}

float CannyEdgeHost::getTotTime() {
	return totTime;
}