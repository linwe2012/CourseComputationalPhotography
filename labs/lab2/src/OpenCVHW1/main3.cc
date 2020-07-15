#include "config.h"

#ifdef HW_3

#define CVUI_IMPLEMENTATION
#define CVUI_USE_CPP_FS

#include <opencv2/opencv.hpp>
#include "unitext/cvUniText.hpp"
#include "utils.h"
#include "cvui.h"

#include <filesystem>

namespace fs = std::filesystem;


cv::Mat heatMap(const cv::Mat& m) {
	using namespace cv;
	Mat heat(m.size(), CV_8UC3);

	const Scalar colors[] = {
		{220, 210, 0},
		{230, 60, 20},
		{0, 200, 120},
		{0, 255, 255},
		{0, 0, 255},
	};

	auto [max, min] = getMaxMin(m);
	auto gap = max - min;

	for (int i = 0; i < m.rows; ++i) {
		auto ptr = reinterpret_cast<const float*>(m.ptr(i));
		auto h = heat.ptr(i);
		constexpr double key[] = { 0.01, 0.012 };

		for (int j = 0; j < m.cols; ++j) {
			auto c = *ptr;
			Scalar r;
			if (c < 0) {
				r = colors[0];
			}
			else if (c < key[0]) {
				c *= 1/ key[0];
				r = colors[1] * (1-c) + colors[2] * (c);
			}
			else if (c < key[1]) {
				c -= key[0];
				c *= 1 / (1 - key[0]);
				r = colors[2] * (1 - c) + colors[3] * (c);
			}
			else if (c < 1) {
				c -= key[1];
				c *= 1/(1-key[1]);
				r = colors[3] * (1-c) + colors[4] * (c);
			}
			else {
				r = colors[3];
			}

			*h++ = saturate_cast<uchar>(r.val[0]);
			*h++ = saturate_cast<uchar>(r.val[1]);
			*h++ = saturate_cast<uchar>(r.val[2]);

			++ptr;
		}
	}

	return heat;
}



// 
//  a  b
//  c  d
//
auto eigen2x2(const cv::Mat& A, const cv::Mat& B, const cv::Mat& C, const cv::Mat& D) {
	using namespace cv;
	Mat A_D = A + D;
	Mat BC = B.clone().mul(C);
	Mat A_minus_D_2 = (A - D);
	A_minus_D_2 = A_minus_D_2.mul(A_minus_D_2);
	Mat Inside = A_minus_D_2 + 4 * BC;
	Mat Root;
	sqrt(Inside, Root);

	auto [max, min] = getMaxMin(Root);

	Mat lambda_max = (A_D + Root) / 2;
	Mat lambda_min = (A_D - Root) / 2;
	return std::make_tuple( lambda_max, lambda_min );
}

struct Params {
	double k = 0.05;
	double thres = 0.05;
	double thres2 = 0.05;
	double mul = 3;
};

auto harrisFullset(const cv::Mat& img, Params params) {
	using namespace cv;

	Mat gray = grayscale(img);
	gray.convertTo(gray, CV_32FC1, 1 / 255.0);

	Mat sobel_x({ 3, 3 }, {
		-1.f, 0.f, 1.f,
		-2.f, 0.f, 2.f,
		-1.f, 0.f, 1.f,
		});

	Mat sobel_y({ 3, 3 }, {
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
		});


	// Step 1: compute Gardient, and their square
	Mat I_x, I_y;
	ConvT<float, float>(gray, I_x, sobel_x);
	ConvT<float, float>(gray, I_y, sobel_y);

	Mat Ix_Iy = I_x.clone().mul(I_y);
	Mat Ix_2 = I_x.clone().mul(I_x);
	Mat Iy_2 = I_y.clone().mul(I_y);

	
	// Step 2: Gauss Conv on Gardient
	Mat gauss = GenGaussKernel({ 3, 3 }, 3);

	Mat Ix_2_Conv;
	Mat Iy_2_Conv;
	Mat Ix_Iy_Conv;

	ConvT<float, float>(Ix_2, Ix_2_Conv, gauss);
	ConvT<float, float>(Iy_2, Iy_2_Conv, gauss);
	ConvT<float, float>(Ix_Iy, Ix_Iy_Conv, gauss);


	// Step 3: exatrct eigen value, use it to compute Response
	auto [lambda_max, lambda_min] = eigen2x2(
		Ix_2_Conv, Ix_Iy_Conv,
		Ix_Iy_Conv, Iy_2_Conv
	);

	Mat detM = lambda_max.clone().mul(lambda_min);
	Mat lambda_sum2 = (lambda_max + lambda_min);
	lambda_sum2 = lambda_sum2.mul(lambda_sum2);
	Mat R = detM - params.k * lambda_sum2;
	R *= params.mul;

	// Step 4 (additional): compute heat map
	Mat heat = heatMap(R);
	heat /= 255;


	// Step 5: Thresholding Response & NMS
	threshold(R, R, params.thres, 1.0, THRESH_TOZERO);
	threshold(R, R, 1.0, 1.0, THRESH_TRUNC);

	Mat R_threshold;
	
	//Mat tmp = R_threshold.clone();
	nonMaxSupression(I_x, I_y, R, R_threshold);
	threshold(R_threshold, R_threshold, params.thres2, 1.0, THRESH_BINARY);


	// Step 5: Overlay response on image
	Mat overlayed;
	img.convertTo(overlayed, CV_32FC3, 1 / 255.0);

	Mat mask;
	R_threshold.convertTo(mask, CV_8UC1, 255.0);

	Mat red = overlayed.clone();
	red = Scalar{ 0.0, 0.0, 1.0 };
	red.copyTo(overlayed, mask);


	return std::make_tuple(gray, lambda_max, lambda_min, R, R_threshold, heat, overlayed);
}



int main()
{
	using namespace cv;

	Mat img = imread("test.png");
	cvui::Gallery gallery;
	auto images = gallery.scan(".");
	bool need_rerender = true;
	
	Params params;

	auto [gray, lambda_max, lambda_min, R, R_threshold, heat, overlayed] = harrisFullset(img, params);

	auto refresh = [&] {
		auto [gray_, lambda_max_, lambda_min_, R_, R_threshold_, heat_, overlayed_] = harrisFullset(img, params);
		gray = gray_;
		lambda_max = lambda_max_;
		lambda_min = lambda_min_;
		R = R_;
		R_threshold = R_threshold_;
		heat = heat_;
		overlayed = overlayed_;
		need_rerender = true;
	};


	int showing = 5;
	int height = 800;
	constexpr int ctrl_h = 40;
	constexpr int ctrl_w = 250;
	Mat canvas(Size{ 1000 + ctrl_w, 800 }, CV_32FC3, Scalar{ 22, 22, 22 } / 255.0);
	Mat ctrl_area(Size{ ctrl_w , height }, CV_32FC3, Scalar{22, 22, 22} / 255.0);
	int h = 0;
	
	cvui::init("Harris");

	auto process_pass = [&canvas, &h, ctrl_h, ctrl_w, &img, &showing, &need_rerender](const Mat& m, const char* name, double scale = 1) {
		int wh = h;

		if (cvui::button(canvas, 0, ctrl_h * h++, name) || (showing == wh && need_rerender)) {
			need_rerender = false;
			showing = wh;
			if (m.channels() == 1) {
				
				Mat res;
				merge(std::vector<Mat>{ m, m, m }, res);
				if (scale != 1) {
					res *= scale;
				}
				res.copyTo(canvas(Rect{ ctrl_w, 0, img.cols, img.rows }));
			}
			else {
				if (scale != 1) {
					Mat tmp = m * scale;
					tmp.copyTo(canvas(Rect{ ctrl_w, 0, img.cols, img.rows }));
				}
				else {
					m.copyTo(canvas(Rect{ ctrl_w, 0, img.cols, img.rows }));
				}
			}
		}
	};

	gallery.stuff(images, canvas, Rect{ ctrl_w, 0, img.cols, img.rows });
	
	bool isPicking = true;

	while (true) {
		if (isPicking) {
			int picked = gallery.pick();
			if (picked) {
				isPicking = false;
				img = images[picked-1].image;
				canvas = 0;
				refresh();
			}
			else {
				putText(canvas, "Select A Image", { 0, 30 }, HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.5, { 255, 255, 255 });
				cvui::imshow("Harris", canvas);
				waitKey(30);
			}
		}
		else {
			h = 0;
			ctrl_area.copyTo(canvas(Rect{ 0, 0, ctrl_w, height }));

			process_pass(lambda_max, "Lambda Max");
			process_pass(lambda_min, "Lambda Min");
			process_pass(R, "Response");
			process_pass(heat, "Heat Map");
			process_pass(R_threshold, "R threshold");
			process_pass(overlayed, "Overlayed");

			if (cvui::button(canvas, 0, ctrl_h * h++, "Gallery")) {
				isPicking = true;
				canvas = 0;
			}

			if (cvui::button(canvas, 0, ctrl_h * h++, "Save Images")) {
				if (!fs::exists("results")) {
					fs::create_directories("results");
				}
				int i = 0;
				auto save = [&i](const Mat& m, std::string name) {
					imwrite( "results/" + std::to_string(i) + "." + name + ".png", m*255);
					++i;
				};

				save(lambda_max, "lambda_max");
				save(lambda_min, "lambda_min");
				save(R, "R");
				save(R_threshold, "R_threshold");
				save(overlayed, "Overlayed");
			}

			bool aaa = false;

			constexpr double hscale = 1.2;
			constexpr int  pad = 15;

			aaa |= cvui::trackbar<double>(canvas, pad, hscale * ctrl_h * h++, ctrl_w-2* pad, &params.k, 0.01, 0.1, 1, "k=%.2lf");
			aaa |= cvui::trackbar<double>(canvas, pad, hscale * ctrl_h * h++, ctrl_w-2* pad, &params.thres, 0.01, 5.0, 1, "threshold 1: %.2lf");
			aaa |= cvui::trackbar<double>(canvas, pad, hscale * ctrl_h * h++, ctrl_w-2* pad, &params.thres2, 0.01, 2.0, 1, "threshold: %.2lf");
			aaa |= cvui::trackbar<double>(canvas, pad, hscale * ctrl_h * h++, ctrl_w - 2* pad, &params.mul, 1, 30, 1, "multiply: %.2lf");


			cvui::imshow("Harris", canvas);

			if (aaa) {
				refresh();
			}
			else {
				if (waitKey(30) == 27) {
					break;
				}
			}
		}
	}
}

#endif