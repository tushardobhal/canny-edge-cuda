class IntImage {
	private:
		float *h_image;
		int width;
		int height;

public:
	IntImage(float *h_image, int width, int height);
	~IntImage();

	float *hostImage();

	int imageWidth();

	int imageHeight();

	int pixel(int x, int y);
};