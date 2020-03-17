class CannyEdgeDevice {
	float *d_image;
	
	float *d_filterImage;
	float *d_gradientX;
	float *d_gradientY;
	float *d_gradientMag;
	float *d_nonMaxSup;
	float *d_highThreshHyst;
	float *d_lowThreshHyst;
	
	float *d_gaussianKernel;
	float *d_sobelKernelX;
	float *d_sobelKernelY;

	int width;
	int height;

	float lowThreshold;
	float highThreshold;
	float totTime;

	void initializeGaussianKernel();
	void initializeSobelFilters();

	public:
		CannyEdgeDevice(float *h_image, int width, int height);
		~CannyEdgeDevice();

		void performGaussianFiltering();
		void performImageGradientX();
		void performImageGradientY();
		void computeMagnitude();
		void nonMaxSuppression();
		void computeCannyThresholds();
		void lowHysterisisThresholding();
		void highHysterisisThresholding();

		float *getD_gaussianKernel();
		float *getD_sobelKernelX();
		float *getD_sobelKernelY();
		float *getD_FilterImage();
		float *getD_GradientX();
		float *getD_GradientY();
		float *getD_gradientMag();
		float *getD_nonMaxSup();
		float getLowThreshold();
		float getHighThreshold();
		float *getD_HighThreshold();
		float *getD_LowThreshold();
		float getTotTime();
};