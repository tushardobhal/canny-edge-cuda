#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "imageLib/Image.h"
#include "imageLib/ImageIO.h"
#include "imageLib/Convert.h"

#include "intImage.h"
#include "cannyEdgeDevice.h"
#include "cannyEdgeHost.h"

// Path to source image and computed images 
const char *SRC_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/tree.png";
const char *GAUSS_FILTERED_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/gauss_filtered.png";
const char *X_GRADIENT_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/x_gradient.png";
const char *Y_GRADIENT_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/y_gradient.png";
const char *MAGNITUDE_GRADIENT_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/magnitude_gradient.png";
const char *NON_MAX_SUPPR_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/non_max_suppression.png";
const char *FINAL_THRESHOLD_FILE_NAME = "C:/Users/tdobhal/Downloads/canny-edge-cuda/project/data/final_threshold.png";

float *convertCImageToCuda(CByteImage img) {
	CShape shape = img.Shape();
	float *src = (float *)malloc(shape.width * shape.height * sizeof(float));

	for (int x = 0; x < shape.width; x++) {
		for (int y = 0; y < shape.height; y++) {
			src[y  *shape.width + x] = (img.Pixel(x,y,0))/255.0f;
		}
	}
	return src;
}

CByteImage convertCudaToCImage(float *img, int width, int height) {
	CByteImage dst(width, height, 1);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			dst.Pixel(x, y, 0) = (int)(img[y * width + x] * 255);
		}
	}
	return dst;
}

void writeImageToFile(float *d_image, int width, int height, const char *fileName, bool isGpu) {
	float *h_image = (float *)malloc(sizeof(float) * width * height);;
	if (isGpu) {
		checkCudaErrors(cudaMemcpy(h_image, d_image, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	}
	else {
		h_image = d_image;
	}

	CByteImage constructedImage = convertCudaToCImage(h_image, width, height);
	WriteImage(constructedImage, fileName);
}

int main(int argc, char **argv)
{
	CByteImage img;
	ReadImage(img, SRC_FILE_NAME);
	
	CByteImage gray = ConvertToGray(img);
	CShape shape = gray.Shape();

	IntImage image = IntImage(convertCImageToCuda(gray), shape.width, shape.height);
	printf("Image from path %s loaded. Width - %d, Height - %d\n", SRC_FILE_NAME, image.imageWidth(), image.imageHeight());

	CannyEdgeDevice deviceCannyEdge = CannyEdgeDevice(image.hostImage(), image.imageWidth(), image.imageHeight());
	CannyEdgeHost hostCannyEdge = CannyEdgeHost(image.hostImage(), image.imageWidth(), image.imageHeight());

	printf("Host and Device Canny Edge Detectors Intialized\n\n");

	deviceCannyEdge.performGaussianFiltering();
	hostCannyEdge.performGaussianFiltering();

	deviceCannyEdge.performImageGradientX();
	hostCannyEdge.performImageGradientX();

	deviceCannyEdge.performImageGradientY();
	hostCannyEdge.performImageGradientY();

	deviceCannyEdge.computeMagnitude();
	hostCannyEdge.computeMagnitude();

	deviceCannyEdge.nonMaxSuppression();
	hostCannyEdge.nonMaxSuppression();

	deviceCannyEdge.computeCannyThresholds();
	hostCannyEdge.computeCannyThresholds();

	deviceCannyEdge.highHysterisisThresholding();
	hostCannyEdge.highHysterisisThresholding();

	deviceCannyEdge.lowHysterisisThresholding();
	hostCannyEdge.lowHysterisisThresholding();

	printf("\nDevice Canny Edge Detection completed in %f ms\n", deviceCannyEdge.getTotTime());
	printf("Host Canny Edge Detection completed in %f ms\n\n", hostCannyEdge.getTotTime());
	
	float *d_filter = deviceCannyEdge.getD_FilterImage();
	writeImageToFile(d_filter, image.imageWidth(), image.imageHeight(), GAUSS_FILTERED_FILE_NAME, true);
	float *dx = deviceCannyEdge.getD_GradientX();
	writeImageToFile(dx, image.imageWidth(), image.imageHeight(), X_GRADIENT_FILE_NAME, true);
	float *dy = deviceCannyEdge.getD_GradientY();
	writeImageToFile(dy, image.imageWidth(), image.imageHeight(), Y_GRADIENT_FILE_NAME, true);
	float *dmag = deviceCannyEdge.getD_gradientMag();
	writeImageToFile(dmag, image.imageWidth(), image.imageHeight(), MAGNITUDE_GRADIENT_FILE_NAME, true);
	float *dnm = deviceCannyEdge.getD_nonMaxSup();
	writeImageToFile(dnm, image.imageWidth(), image.imageHeight(), NON_MAX_SUPPR_FILE_NAME, true);
	float *dlowThresh = deviceCannyEdge.getD_LowThreshold();
	writeImageToFile(dlowThresh, image.imageWidth(), image.imageHeight(), FINAL_THRESHOLD_FILE_NAME, true);

	system(SRC_FILE_NAME);
	system(GAUSS_FILTERED_FILE_NAME);
	system(X_GRADIENT_FILE_NAME);
	system(Y_GRADIENT_FILE_NAME);
	system(MAGNITUDE_GRADIENT_FILE_NAME);
	system(NON_MAX_SUPPR_FILE_NAME);
	system(FINAL_THRESHOLD_FILE_NAME);
}