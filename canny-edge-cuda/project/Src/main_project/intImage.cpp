#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "intImage.h"

IntImage::IntImage(float *h_image, int width, int height) {
	this->h_image = (float *)malloc(width * height * sizeof(float));
	memcpy(this->h_image, h_image, width * height * sizeof(float));
	this->width = width;
	this->height = height;
}

IntImage::~IntImage() {
	if (this->h_image != NULL) free(h_image);
}

float *IntImage::hostImage() {
	return this->h_image;
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