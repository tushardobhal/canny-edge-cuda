#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/common.h"
#include "intImage.h"

IntImage::IntImage(int *h_image, int width, int height) {
	this->h_image = h_image;

	CHECK(cudaMalloc((void**)&d_image, sizeof(int) * width * height));
	CHECK(cudaMemcpy(this->d_image, h_image, sizeof(int) * width * height, cudaMemcpyHostToDevice));

	this->width = width;
	this->height = height;
}

IntImage::~IntImage() {
	if (h_image != NULL) free(h_image);
	if (d_image != NULL) CHECK(cudaFree(d_image));
}

int *IntImage::hostImage() {
	return this->h_image;
}

int *IntImage::deviceImage() {
	return this->d_image;
}

int IntImage::imageWidth() {
	return this->width;
}

int IntImage::imageHeight() {
	return this->height;
}

int IntImage::pixel(int x, int y) {
	return this->h_image[x * this->width + y];
}