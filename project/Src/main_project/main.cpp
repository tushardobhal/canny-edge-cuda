#include <cuda_runtime.h>
#include <helper_functions.h>

#include "imageLib/Image.h"
#include "imageLib/ImageIO.h"
#include "imageLib/Convert.h"

#include "intImage.h"

const char *SRC_FILE_NAME = "C:/Users/tdobhal/Downloads/project/data/lena.png";
const char *DST_FILE_NAME = "C:/Users/tdobhal/Downloads/project/data/lena_reconstruct.png";

int *convertCImageToCuda(CByteImage img) {
	CShape shape = img.Shape();
	int *src = (int *)malloc(shape.width * shape.height * sizeof(int));

	for (int x = 0; x < shape.width; x++) {
		for (int y = 0; y < shape.height; y++) {
			src[x*shape.width + y] = img.Pixel(x,y,0);
		}
	}
	return src;
}

CByteImage convertCudaToCImage(int *img, int width, int height) {
	CByteImage dst(width, height, 1);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			dst.Pixel(x,y,0) = img[x*width + y];
		}
	}
	return dst;
}

int main(int argc, char **argv)
{
	CByteImage img;
	ReadImage(img, SRC_FILE_NAME);
	
	CByteImage gray = ConvertToGray(img);
	CShape shape = gray.Shape();

	IntImage image = IntImage(convertCImageToCuda(gray), shape.width, shape.height);
	printf("Image from path %s loaded. Width - %d, Height - %d\n", SRC_FILE_NAME, image.imageWidth(), image.imageHeight());

	CByteImage constructedImage = convertCudaToCImage(image.hostImage(), image.imageWidth(), image.imageHeight());
	WriteImage(constructedImage, DST_FILE_NAME);
}