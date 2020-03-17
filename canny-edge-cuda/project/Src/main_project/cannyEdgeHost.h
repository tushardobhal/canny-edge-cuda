class CannyEdgeHost {
	float *h_image;

	float *h_filterImage;
	float *h_gradientX;
	float *h_gradientY;
	float *h_gradientMag;
	float *h_nonMaxSup;
	float *h_highThreshHyst;
	float *h_lowThreshHyst;

	float *h_gaussianKernel;
	float *h_sobelKernelX;
	float *h_sobelKernelY;

	int width;
	int height;

	float lowThreshold;
	float highThreshold;
	float totTime;

	void initializeGaussianKernel();
	void initializeSobelFilters();
	void convolution(float *h_image, float *kernel, float *h_result, int kernelSize);
	void magnitudeImage();
	void nonMaxSuppressionHost();
	void highHysterisisThresholdingHost(int width, int height, float *h_nonMaxSup, float highThreshold, float *h_highThreshHyst);
	void lowHysterisisThresholdingHost(int width, int height, float *h_nonMax, float lowThreshold, float *h_highThreshHyst, float *h_lowThreshHyst);

public:
	CannyEdgeHost(float *h_image, int width, int height);
	~CannyEdgeHost();

	void performGaussianFiltering();
	void performImageGradientX();
	void performImageGradientY();
	void computeMagnitude();
	void nonMaxSuppression();
	void computeCannyThresholds();
	void lowHysterisisThresholding();
	void highHysterisisThresholding();

	float *getH_gaussianKernel();
	float *getH_sobelKernelX();
	float *getH_sobelKernelY();
	float *getH_FilterImage();
	float *getH_GradientX();
	float *getH_GradientY();
	float *getH_gradientMag();
	float *getH_nonMaxSup();
	float getLowThreshold();
	float getHighThreshold();
	float *getH_HighThreshold();
	float *getH_LowThreshold();
	float getTotTime();
};