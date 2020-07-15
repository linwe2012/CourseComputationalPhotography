#include "config.h"
#ifdef HW_5_2
#include <opencv2/opencv.hpp>
#include <iostream>
#include "main5.h"
#include "utils.h"

int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "GaussianFilter", "<input-image> <output-image> <sigma>", 3)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}

	double sigma = atof(argv[3]);
	int sz = 2 * static_cast<int>(std::floor(5 * sigma)) + 1;
	std::cout << "Kernel size: " << sz << std::endl;

	filter2D(src, src, -1, GenGaussKernel(Size{ sz , sz }, static_cast<float>(sigma)));
	if (!imwrite(argv[2], src)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif
