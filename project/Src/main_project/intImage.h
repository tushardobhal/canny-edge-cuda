class IntImage {
	private:
		int *h_image;
		int *d_image;
		int width;
		int height;

public:
	IntImage(int *h_image, int width, int height);
	~IntImage();

	int *hostImage();

	int *deviceImage();

	int imageWidth();

	int imageHeight();

	int pixel(int x, int y);
};