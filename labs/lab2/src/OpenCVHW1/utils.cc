#include "utils.h"
using namespace cv;

void overlayImage(Mat& src, Mat& overlay, const Point& location)
{
	for (int y = max(location.y, 0); y < src.rows; ++y)
	{
		int fY = y - location.y;

		if (fY >= overlay.rows)
			break;

		for (int x = max(location.x, 0); x < src.cols; ++x)
		{
			int fX = x - location.x;

			if (fX >= overlay.cols)
				break;

			double opacity = ((double)overlay.data[fY * overlay.step + fX * overlay.channels() + 3]) / 255;

			for (int c = 0; opacity > 0 && c < src.channels(); ++c)
			{
				unsigned char overlayPx = overlay.data[fY * overlay.step + fX * overlay.channels() + c];
				unsigned char srcPx = src.data[y * src.step + x * src.channels() + c];
				src.data[y * src.step + src.channels() * x + c] = static_cast<uchar>(srcPx * (1. - opacity) + overlayPx * opacity);
			}
		}
	}
}

void rgb2yuv(const cv::Mat& src, cv::Mat& dst) {
	using namespace cv;

	static const double mat[3][3] = {
		0.299,   0.587, 0.114,
		-0.147, -0.289, 0.435,
		0.615,  -0.515,-0.1,
	};

	if (src.channels() != 3) {
		throw std::invalid_argument("rgb2yuv: Expected color depth of 3");
	}

	if (dst.size() != src.size()) {
		resize(src, dst, src.size());

	}

	for (int i = 0; i < src.rows; ++i) {
		auto ptr = src.ptr(i);
		auto dptr = dst.ptr(i);
		for (int j = 0; j < src.cols; ++j) {
			auto b = ptr[0];
			auto g = ptr[1];
			auto r = ptr[2];

			// Y
			dptr[2] = static_cast<uchar>(r * mat[0][0] + g * mat[0][1] + b * mat[0][2]);
			// U
			dptr[1] = static_cast<uchar>(r * mat[1][0] + g * mat[1][1] + b * mat[1][2]);
			// V
			dptr[0] = static_cast<uchar>(r * mat[2][0] + g * mat[2][1] + b * mat[2][2]);

			dptr += 3;
			ptr += 3;
		}
	}
}


cv::Mat GenGaussKernel(cv::Size size, float sigma) {
	using namespace cv;

	const int w = size.width / 2;
	const int h = size.height / 2;

	constexpr float pi = static_cast<float>(Math::PI);

	Mat kernel(size, CV_32F);
	const float sigma2 = sigma * sigma;
	const float coeff = 1.f / (2.f * pi * sigma2);
	const float inv_2sigma2 = 1.f / sigma2;

	double sum = 0;
	for (int i = 0; i < size.height; ++i) {
		for (int j = 0; j < size.width; ++j) {
			int dw = (i - (h));
			int dh = (j - (w));

			int nom = dw * dw + dh * dh;

			const float g = coeff * exp(-nom * inv_2sigma2);
			sum += g;
			kernel.at<float>(i, j) = g;
		}
	}

	kernel /= sum;
	return kernel;
}

void nonMaxSupression(const cv::Mat& x, const cv::Mat& y, const cv::Mat& sum, cv::Mat& out) {
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

		// auto td = dir.ptr(i - 1) + 3;

		for (int j = 1; j < x.cols - 1; ++j) {
			auto est = estimateWheelDirection(*px, *py);
			auto t = *sx;
			switch (est)
			{
			case WheelDirections::UpDown:
				if (std::max(*usx, *dsx) > t) {
					*d = 0;
				}
				break;
			case WheelDirections::LeftRight:
				if (std::max(*(sx - 1), *(sx + 1)) > t) {
					*d = 0;
				}
				break;
			case WheelDirections::TopRight:
				if (std::max(*(usx + 1), *(dsx - 1)) > t) {
					*d = 0;
				}
				break;
			case WheelDirections::TopLeft:
				if (std::max(*(usx - 1), *(dsx + 1)) > t) {
					*d = 0;
				}
				break;
			default:
				break;
			}

			++usx; ++sx; ++dsx;
			++px; ++py;
			++d;
		}
	}
}