#include "config.h"

#ifdef HW_2

#define CVUI_IMPLEMENTATION
#include <opencv2/opencv.hpp>
#include "unitext/cvUniText.hpp"
#include "utils.h"
#include "cvui.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <filesystem>

namespace fs = std::filesystem;



/*
void Conv(const cv::Mat& src, cv::Mat& dst, cv::Mat& kernel, int border = cv::BORDER_REFLECT) {
	using namespace cv;

	const int base_h = kernel.rows / 2;
	const int base_w = kernel.cols / 2;
	const int channels = src.channels();

	if (dst.size() != src.size()) {
		resize(src, dst, src.size());
	}

	Mat base;
	// we are going to copy whole anyway, so 
	// copying with border won't have much perf penalties
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, border);

	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<uchar*> ring(kernel.rows, 10);
	for (int i = 0; i < kernel.rows - 1; ++i) {
		ring.push(base.ptr(i));
	}

	// we can access kernel elements quickly when doing conv
	std::vector<float> vkern;
	for (int i = 0; i < kernel.rows; ++i) {
		for (int j = 0; j < kernel.cols; ++j) {
			vkern.push_back(kernel.at<float>(i, j));
		}
	}

	// for each pixel
	for (int i = 0; i < src.rows; ++i) {
		ring.push(base.ptr(i + kernel.rows - 1));
		uchar* ptr = dst.ptr(i);
		for (int j = 0; j < src.cols; ++j) {

			// for each channels
			for (int ch = 0; ch < channels; ++ch) {
				double tmp = 0;
				int kern_idx = 0;

				// for each elem in kernel
				for (int u = 0; u < kernel.rows; ++u) {
					for (int v = 0; v < kernel.cols; ++v) {
						auto p = ring[u] + (j + v) * channels + ch;
						tmp += static_cast<double>(*p) * vkern[kern_idx];
						++kern_idx;
					}
				}

				
				// result for each channel
				if (tmp < 0) {
					*ptr++ = 0;
				}
				else if (tmp > 255) {
					*ptr++ = 255u;
				}
				else {
					*ptr++ = static_cast<uchar>(tmp);
				}
				
			}
		}
	}
}

*/


void nonMaxSupression(const cv::Mat&x, const cv::Mat& y, const cv::Mat& sum, cv::Mat& out, cv::Mat& dir) {
	// going to copy sum anyway
	cv::resize(sum, out, sum.size());
	
	for (int i = 2; i < x.rows; ++i) {
		auto usx = (float*)sum.ptr(i - 2);
		auto sx = (float*)sum.ptr(i - 1);
		auto dsx = (float*)sum.ptr(i);

		auto px = (float*)x.ptr(i - 1);
		auto py = (float*)y.ptr(i - 1);

		auto d = (float*)out.ptr(i - 1);

		++usx; ++sx; ++dsx;
		++px; ++py;
		++d;

		auto td = dir.ptr(i - 1) + 3;

		for (int j = 1; j < x.cols-1; ++j) {
			auto est = estimateWheelDirection(*px, *py);
			auto t = *sx;
			switch (est)
			{
			case WheelDirections::UpDown:
				if (std::max(*usx, *dsx) > t) {
					*d = 0;
				}
				else {
					// blue
					td[2] = 2;
					td[1] = 131;
					td[0] = 223;
					
				}
				break;
			case WheelDirections::LeftRight:
				if (std::max(*(sx - 1), *(sx + 1)) > t) {
					*d = 0;
				}
				else {
					// light purple
					td[2] = 181;
					td[1] = 143;
					td[0] = 243;
				}
				break;
			case WheelDirections::TopRight:
				if (std::max(*(usx + 1), *(dsx - 1)) > t) {
					*d = 0;
				}
				else {
					// light green
					td[2] = 141;
					td[1] = 247;
					td[0] = 185;
				}
				break;
			case WheelDirections::TopLeft:
				if (std::max(*(usx - 1), *(dsx + 1)) > t) {
					*d = 0;
				}
				else {
					// orange
					td[2] = 243;
					td[1] = 178;
					td[0] = 78;
				}
				break;
			default:
				break;
			}

			++usx; ++sx; ++dsx;
			++px; ++py;
			++d;
			td += 3;
		}
	}
}


void doubleThresholdFilter(const cv::Mat& src, cv::Mat& dst, double high_thres, double low_thres = -1) {
	using namespace cv;
	Mat low, high;
	cv::resize(src, dst, src.size());

	if (low_thres < 0) {
		low_thres = high_thres * 0.4;
	}

	bool col_max[3] = { false };

	auto usx = (float*)src.ptr(0);
	auto sx = (float*)src.ptr(1);
	auto dsx = (float*)src.ptr(2);

	auto test_max = [&col_max, &usx, &sx, &dsx, high_thres, low_thres](int idx) {
		col_max[idx] = false;
		if (*usx > high_thres || *sx > high_thres || *dsx > high_thres) {
			col_max[idx] = true;
		}
	};

	for (int i = 2; i < src.rows; ++i) {
		usx = (float*)src.ptr(i - 2);
		sx = (float*)src.ptr(i - 1);
		dsx = (float*)src.ptr(i);

		auto d = (float*)dst.ptr(i - 1);

		test_max(0);

		++usx; ++sx; ++dsx;
		test_max(1);
		++usx; ++sx; ++dsx;
		test_max(2);

		++d;
		for (int j = 1; j < src.cols - 1; ++j) {
			bool has_max = col_max[0] | col_max[1] | col_max[2];
			if (*sx < low_thres) {
				*d = 0;
			}
			else if(*sx > high_thres)  {
				*d = 1.f;
			}
			else if (has_max) {
				*d = 1.f;
			}
			else {
				*d = 0;
			}

			++usx; ++sx; ++dsx;
			++d;
			col_max[0] = col_max[1];
			col_max[1] = col_max[2];
			test_max(2);
		}
	}
}


int main() {
	using namespace cv;
	auto lena = imread("Lena.png");
	Mat lena_gray;

	{
		Mat lenas[3];
		Mat lena_yuv;
		rgb2yuv(lena, lena_yuv);
		split(lena_yuv, lenas);
		lena_gray = lenas[2];
	}
	

	auto kernel = GenGaussKernel({ 3, 3 }, 2);
	ConvT<uchar>(lena_gray, lena_gray, kernel);

	//Mat deri_x({ 3, 3 }, {
	//	-1.f, 0.f, 1.f,
	//	-2.f, 0.f, 2.f,
	//	-1.f, 0.f, 1.f,
	//	});
	//Mat deri_y({ 3, 3 }, {
	//	1.f, 2.f, 1.f,
	//	0.f, 0.f, 0.f,
	//	-1.f, -2.f, -1.f,
	//	});

	Mat deri_x({ 2, 2 }, {
		-1.f, 1.f,
		-1.f, 1.f
		});
	deri_x /= 2.f;
	
	Mat deri_y({ 2, 2 }, {
		1.f, 1.f,
		-1.f, -1.f
		});
	deri_x /= 2.f;

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	Mat lena_x, lena_y;
	ConvT<float>(lena_gray, lena_x, deri_x);
	ConvT<float>(lena_gray, lena_y, deri_y);
	
	const auto the_mul = 1.0;
	minMaxLoc(lena_x, &minVal, &maxVal, &minLoc, &maxLoc);
	lena_x /= maxVal * the_mul;

	minMaxLoc(lena_y, &minVal, &maxVal, &minLoc, &maxLoc);
	lena_y /= maxVal * the_mul;

	Mat sobel = lena_x.clone().mul(lena_x) + lena_y.clone().mul(lena_y);
	sqrt(sobel, sobel);

	Mat nms;
	Mat dir(sobel.size(), CV_8UC3, Scalar{ 22, 22, 22 });
	nonMaxSupression(lena_x, lena_y, sobel, nms, dir);


	double high_thres = 0.2;
	double low_thres = 0.08;

	Mat connected;
	doubleThresholdFilter(nms, connected, high_thres, low_thres);

	
	namedWindow("Connected");
	// imshow("Connected", connected);
	// Mat canvas = connected.clone();
	cvui::init("Connected");
	
	bool show_ctrl = true;
	
	constexpr int ctrl_w = 150;
	Mat canvas(Size(connected.cols + ctrl_w, connected.rows), connected.type());
	connected.copyTo(canvas(Rect{ ctrl_w, 0, connected.cols, connected.rows }));
	Mat background(Size(ctrl_w, connected.rows), connected.type(), Scalar(0.1));
	int pass = 3;
	Mat mixed;

	auto setCanvasImg = [ctrl_w, &canvas, &connected](Mat& m) {
		m.copyTo(canvas(Rect{ ctrl_w, 0, connected.cols, connected.rows }));
	};

	{
		Mat mask;
		connected.convertTo(mask, CV_8UC1, 255.0);
		lena.copyTo(mixed, mask);
	}

	constexpr int ctrl_h = 40;
	while (1)
	{
		background.copyTo(canvas(Rect{ 0, 0, ctrl_w, connected.rows }));

		int h = 1;
		
		if (cvui::button(canvas, 0, 0, "Gauss Pass")) {
			pass = 1;
			Mat m;
			lena_gray.convertTo(m, CV_32FC1, 1.0/255);
			setCanvasImg(m);
		}
		
		if (cvui::button(canvas, 0, ctrl_h * h++, "Derivitive X")) {
			pass = 2;
			setCanvasImg(lena_x);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "Derivitive Y")) {
			pass = 3;
			setCanvasImg(lena_y);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "Direction*")) {
			namedWindow("Direction");
			imshow("Direction", dir);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "Amplitude Pass")) {
			pass = 4;
			setCanvasImg(sobel);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "NMS Pass")) {
			pass = 4;
			setCanvasImg(nms);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "Final Result")) {
			pass = 4;
			setCanvasImg(connected);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "Mixed Result")) {
			pass = 4;
			Mat mask;
			mixed = Scalar::all(0);
			connected.convertTo(mask, CV_8UC1, 255.0);
			lena.copyTo(mixed, mask);
			namedWindow("Mixed");
			imshow("Mixed", mixed);
		}

		if (cvui::button(canvas, 0, ctrl_h * h++, "Save All")) {
			std::string d = "results/";
			if (!fs::exists(d)) {
				fs::create_directories(d);
			}
			Mat tmp;
			lena_x.convertTo(tmp, CV_8UC1, 255.0);
			imwrite(d + "lena-1-dx.png", tmp);

			lena_y.convertTo(tmp, CV_8UC1, 255.0);
			imwrite(d + "lena-1-dy.png", tmp);

			sobel.convertTo(tmp, CV_8UC1, 255.0);
			imwrite(d + "lena-2-amp.png", tmp);

			imwrite(d + "lena-2-dir.png", dir);

			nms.convertTo(tmp, CV_8UC1, 255.0);
			imwrite(d + "lena-3-nms.png", tmp);

			connected.convertTo(tmp, CV_8UC1, 255.0);
			imwrite(d + "lena-4-final.png", tmp);
			imwrite(d + "lena-4-mixed.png", mixed);
		}

		if (show_ctrl) {

			if (cvui::trackbar(canvas, 0, ctrl_h * h++, 120, &high_thres, low_thres, 0.5, 1, "%.2lf")) {
				doubleThresholdFilter(nms, connected, high_thres, low_thres);
				connected.copyTo(canvas(Rect{ ctrl_w, 0, connected.cols, connected.rows }));
			}
			if (cvui::trackbar(canvas, 0, ctrl_h * h++, 120, &low_thres, 0.001, high_thres, 1, "%.2lf")) {
				doubleThresholdFilter(nms, connected, high_thres, low_thres);
				connected.copyTo(canvas(Rect{ ctrl_w, 0, connected.cols, connected.rows }));
			}
		}
		


		cvui::imshow("Connected", canvas);
		
		int k = waitKey(30);
		if (k == 27) {
			break;
		}
	}


	// Mat masked(lena.size(), lena.type(), Scalar::all(0));
	// nms.convertTo(nms, CV_8UC1, 255);
	// threshold(nms, nms, 30, 255, THRESH_BINARY);
	// 
	// namedWindow("Mask");
	// imshow("Mask", nms);
	// 
	// lena.copyTo(masked, nms);
	// namedWindow("Colored");
	// imshow("Colored", masked);

	//std::cout << "lena: cols: " << lena.cols << " rows: " << lena.rows << std::endl;
	//waitKey(0);
}






#endif // HW_2
