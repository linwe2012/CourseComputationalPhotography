#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include "config.h"
#include "hw8_pa.h"
#include <tuple>
#include <algorithm>
#include <execution>
#include <Eigen/Sparse>
#include "sparse-matrix.h"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

void SolveChannel(
	int channel_idx, int constraint,
	const cv::Mat& color_gradient_x, const cv::Mat& color_gradient_y,
	cv::Mat& output, //const std::vector<cv::Mat>& Images, cv::Mat Labels,
	int iterations,
	std::vector<double> init,
	const cv::Mat& mask);

// a method to inject feature detector or matcher into main program
struct InjectAlgorithm {

    InjectAlgorithm() {
        using namespace cv;

        pdetector = AKAZE::create();

        //pmatcher = FlannBasedMatcher::create();

        pmatcher = BFMatcher::create();
    }

    void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) {
        pdetector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    }

    void match(cv::InputArray queryDescriptors, cv::InputArray trainDescriptors, std::vector<cv::DMatch>& matches) {
        try {
            pmatcher->match(queryDescriptors, trainDescriptors, matches);
        }
        catch (cv::Exception& e) {
            std::cout << e.what();
        }

    }

    cv::Ptr<cv::Feature2D> pdetector;
    cv::Ptr<cv::DescriptorMatcher> pmatcher;
};

class ImageToStitch {
public:

	cv::Mat im;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};



struct ImageToMatch {
	ImageToMatch(const ImageToStitch& im1, const ImageToStitch& im2, InjectAlgorithm algo)
		:m1(im1), m2(im2)
	{
		using namespace cv;
		algo.match(im1.descriptors, im2.descriptors, matched);

		// we only pick first 80 matches
		std::sort(matched.begin(), matched.end());
		matched.erase(matched.begin() + 80, matched.end());

		a.resize(matched.size());
		b.resize(matched.size());

		// make data of best matches
		for (int i = 0; i < matched.size(); ++i) {
			a[i] = im1.keypoints[matched[i].queryIdx].pt;
			b[i] = im2.keypoints[matched[i].trainIdx].pt;
		}

		m = findHomography(b, a, RANSAC);
	}

	std::vector<cv::DMatch> matched;
	cv::Mat m;
	const ImageToStitch& m1;
	const ImageToStitch& m2;
	std::vector<cv::Point2f> a;
	std::vector<cv::Point2f> b;
};
cv::Mat stitchImages(std::vector<ImageToMatch>& matches, std::vector<cv::Mat>& images, cv::Mat& originalImage, double f);

class Panorama5728 : public CylindricalPanorama {
public:
    bool makePanorama(
        std::vector<cv::Mat>& img_vec, cv::Mat& img_out, double f) override;

    static cv::Mat toCylinder(const cv::Mat& im, double f);
	static cv::Mat toPlanar(const cv::Mat& im, double f);

private:
	// emmm, 貌似这个公式用了会让图片集中在左上角orz
#if 0
     static auto GetCylinderCoords(int x, int y, double f, int width, int height) {
        const auto r = f;
        double x_prime = r * std::atan2(x, f);
        double y_prime = r * y / std::sqrt(x * x + f * f);

        return std::make_tuple(x_prime, y_prime);
     };

     static auto GetPlanarCoords(int x_prime, int y_prime, double f, int width, int height) {
         const auto r = f;

         double x = f * std::tan(x_prime / r);
         double y = y_prime / r * std::sqrt(x * x + f * f);

         return std::make_tuple(x, y);
     }

	 static auto GetPlanerWidthHeight(int x_prime, int y_prime, double f)
	 {
		 return GetPlanarCoords(x_prime, y_prime, f, 0, 0);
	 }

#endif
#if 1
	 static auto GetCylinderCoords(int x, int y, double f, int width, int height) {

		 double x05 = x - 0.5 * width;
		 double x_prime = f * std::atan2(x05, f) + f * std::atan2(0.5 * width, f);

		 
		 double y_prime = f * (y - 0.5 * height) / std::sqrt(x05 * x05 + f * f) + 0.5 * height;

		 return std::make_tuple(x_prime, y_prime);
	 };

	 static auto GetPlanarCoords(int x_prime, int y_prime, double f, int width, int height) {
		 const auto r = f;

		 double x = 0.5 * width + f * tan(x_prime / f - atan2(0.5 * width, f));
		 double x05 = x - 0.5 * width;
		 double y = (y_prime - 0.5 * height) * sqrt(x05 * x05 + f * f) / f + 0.5 * height;
		 return std::make_tuple(x, y);
	 }

	 static auto GetPlanerWidthHeight(int x_prime, int y_prime, double f)
	 {
		 double x = 2 * f * std::tan(x_prime / 2 / f);
		 double x05 = ((int)x) * 0.5;

		 double y = y_prime / ((0.5 * f) / std::sqrt(x05 * x05 + f * f) + 0.5);

		 return std::make_tuple(x, y);
	 }
#endif

};


bool Panorama5728::makePanorama(std::vector<cv::Mat>& img_vec, cv::Mat& img_out, double f)
{
    // Step1: 柱面坐标转换
    std::vector<cv::Mat> cylinder_imgs(img_vec.size());
    std::vector<ImageToStitch> match_info(img_vec.size());

    std::transform(img_vec.cbegin(), img_vec.cend(), cylinder_imgs.begin(),
		[f](const cv::Mat& im) { return toCylinder(im, f); });

	{
		int i = 0;
		for (auto& im : cylinder_imgs)
		{
			cv::imwrite("results/im" + std::to_string(i) + ".png", cylinder_imgs[i]);
			++i;
		}

	}
	
	
	
	// Step2: 特征检测 & 匹配
	InjectAlgorithm algo;

	for (int i = 0; i < cylinder_imgs.size(); ++i)
	{
		match_info[i].im = cylinder_imgs[i];
		algo.detect(cylinder_imgs[i], match_info[i].keypoints, match_info[i].descriptors);
	}


	std::vector<ImageToMatch> pairs;
	for (int i = 0; i < int(cylinder_imgs.size()) - 1; ++i) {
		std::cout << "Matching " << i << ", " << i + 1 << "...";
		pairs.emplace_back(match_info[i], match_info[i + 1], algo);
		std::cout << " [Done]" << std::endl;
		// pairs.push_back(ImageToMatch(match_info[i], match_info[i + 1], algo));
	}

	cv::Mat stitched = stitchImages(pairs, cylinder_imgs, img_vec[0], f);
	img_out = stitched;



	return true;

}

struct Quadrangle {
	enum {
		TopLeft,
		BottomLeft,
		BottomRight,
		TopRight
	};
	cv::Point3d box[4];

	Quadrangle() {
		box[0] = cv::Point3d(0, 0, 0);
		box[1] = box[0];
		box[2] = box[0];
		box[3] = box[0];
	}

	cv::Rect rect(Quadrangle b) {
		return cv::Rect(leftmost() - b.leftmost(), highest() - b.highest(), ceilWidth(), ceilHeight());
	}

	double leftmost() {
		return std::min(box[TopLeft].x, box[BottomLeft].x);
	}

	double rightmost() {
		return std::max(box[TopRight].x, box[BottomRight].x);
	}

	double width() {
		return rightmost() - leftmost();
	}

	int ceilWidth() {
		return static_cast<int>(std::ceil(width()));
	}

	double height() {
		return lowest() - highest();
	}

	int ceilHeight() {
		return static_cast<int>(std::ceil(height()));
	}

	double lowest() {
		return std::max(box[BottomLeft].y, box[BottomRight].y);
	}

	double highest() {
		return std::min(box[TopLeft].y, box[TopRight].y);
	}

	cv::Size size() {
		return cv::Size(ceilWidth(), ceilHeight());
	}

	void fromTransformedRect(cv::Mat t, cv::Rect r) {
		auto g = [&t](int i, int j) {
			return t.at<double>(i, j);
		};
		auto apply = [this, &t, &g](int idx, int x, int y) {
			box[idx].x = g(0, 0) * x + g(0, 1) * y + g(0, 2);
			box[idx].y = g(1, 0) * x + g(1, 1) * y + g(1, 2);
			box[idx].z = g(2, 0) * x + g(2, 1) * y + g(2, 2);
			box[idx].x /= box[idx].z;
			box[idx].y /= box[idx].z;
		};

		apply(TopLeft, r.tl().x, r.tl().y);
		apply(BottomLeft, r.tl().x, r.br().y);
		apply(BottomRight, r.br().x, r.br().y);
		apply(TopRight, r.br().x, r.tl().y);

		//std::vector<cv::Point2f> corners;
		//corners.emplace_back(r.tl().x, r.tl().y); // Topleft
		//corners.emplace_back(r.tl().x, r.br().y); // BottomLeft
		//corners.emplace_back(r.br().x, r.br().y); // BottomRight
		//corners.emplace_back(r.br().x, r.tl().y); // TopRight
		//std::vector<cv::Point2f> after(corners.size());
		//
		//cv::perspectiveTransform(corners, after, t);

	}

	void wrap(Quadrangle r) {
		using std::max;
		using std::min;

		box[TopLeft].x = min(box[TopLeft].x, r.box[TopLeft].x);
		box[TopLeft].y = min(box[TopLeft].y, r.box[TopLeft].y);

		box[BottomLeft].x = min(box[BottomLeft].x, r.box[BottomLeft].x);
		box[BottomLeft].y = max(box[BottomLeft].y, r.box[BottomLeft].y);

		box[BottomRight].x = max(box[BottomRight].x, r.box[BottomRight].x);
		box[BottomRight].y = max(box[BottomRight].y, r.box[BottomRight].y);

		box[TopRight].x = max(box[TopRight].x, r.box[TopRight].x);
		box[TopRight].y = min(box[TopRight].y, r.box[TopRight].y);
	}

};

void GradientAt(const cv::Mat& Image, int x, int y, cv::Vec3f& grad_x, cv::Vec3f& grad_y)
{
	using namespace cv;
	Vec3i color1 = Image.at<Vec3b>(y, x);
	Vec3i color2 = Image.at<Vec3b>(y, x + 1);
	Vec3i color3 = Image.at<Vec3b>(y + 1, x);
	grad_x = color2 - color1;
	grad_y = color3 - color1;

}

void ZeroGradientAt(const cv::Mat& Image, int x, int y, cv::Vec3f& grad_x, cv::Vec3f& grad_y)
{
	using namespace cv;
	Vec3i color1 = Image.at<Vec3b>(y, x);
	Vec3i color2 = Image.at<Vec3b>(y, x + 1);
	Vec3i color3 = Image.at<Vec3b>(y + 1, x);
	grad_x = color2;
	grad_y = color3;

}

template<typename T>
void MergeImage2(cv::Mat& target, cv::Mat& src, cv::Mat& target_mask, cv::Mat& src_outer_mask, cv::Mat& src_inner_mask, double SkipHowMany)
{
	for (int i = 0; i < target.rows; ++i)
	{
		T* dp = reinterpret_cast<T*>(target.ptr(i));
		T* sp = reinterpret_cast<T*>(src.ptr(i));
		uchar* tgp = reinterpret_cast<uchar*>(target_mask.ptr(i));
		uchar* sop = reinterpret_cast<uchar*>(src_outer_mask.ptr(i));
		uchar* sip = reinterpret_cast<uchar*>(src_inner_mask.ptr(i));
		int k = 0;

		// 这一行不属于 src image 部分, 我们不操作
		while (k < target.cols && *sop == 0)
		{
			sp += 3;
			dp += 3;
			++tgp;
			++k;
			++sop;
			++sip;
		}


		// 如果目标图像的像素点不为0，在原图里我们跳过这些点
		while (*sip == 0 && *tgp != 0)
		{
			sp += 3;
			dp += 3;
			++k;
			++sip;
			++tgp;
			++sop;
		}

		int c = 0;

		// 统计一下我们需要复制多少个点
		while (*sop && k < target.cols)
		{
			++sop;
			++c;
			++k;
		}

		memcpy(dp, sp, sizeof(T) * 3 * c);
	}
}

template<typename T, int channel = 3>
void MergeImage(cv::Mat& target, cv::Mat& src, cv::Mat& target_mask, cv::Mat& src_mask, double SkipHowMany)
{
	for (int i = 0; i < target.rows; ++i)
	{
		T* dp = reinterpret_cast<T*>(target.ptr(i));
		T* sp = reinterpret_cast<T*>(src.ptr(i));
		uchar* tgp = reinterpret_cast<uchar*>(target_mask.ptr(i));
		uchar* smp = reinterpret_cast<uchar*>(src_mask.ptr(i));
		int k = 0;

		// 这一行不属于 src image 部分, 我们不操作
		while (k < target.cols  && *smp == 0)
		{
			sp += channel;
			dp += channel;
			++tgp;
			++k;
			++smp;
		}

		int c = 0;

		// 如果目标图像的像素点不为0，在原图里我们跳过这些点
		if (*tgp != 0 && SkipHowMany > 0)
		{
			//while (*tgp != 0 && *smp && abs(sp[0] - dp[0]) + abs(sp[1] - dp[1]) + abs(sp[2] - dp[2]) > SkipHowMany) {
			//	sp += 3;
			//	dp += 3;
			//	++k;
			//	++c;
			//	++smp;
			//	++tgp;
			//}

			while (c < SkipHowMany)
			{
				sp += channel;
				dp += channel;
				++k;
				++c;
				++smp;
			}
		}
		c = 0;

		// 统计一下我们需要复制多少个点
		while (*smp && k < target.cols)
		{
			++smp;
			++c;
			++k;
		}

		memcpy(dp, sp, sizeof(T) * channel * c);
	}
}

cv::Mat MaskImage(cv::Mat& src, cv::Mat& mask)
{
	cv::Mat res = src.clone();
	for (int i = 0; i < src.rows; ++i)
	{
		uchar* mp = mask.ptr(i);
		uchar* dp = reinterpret_cast<uchar*>(res.ptr(i));

		for (int j = 0; j < src.cols; ++j)
		{
			if (*mp == 0) {
				dp[0] = 0;
				dp[1] = 0;
				dp[2] = 0;
			}

			++mp;
			dp += 3;
		}
	}
	return res;
}

void EnforceGradientBound(cv::Mat& dx, cv::Mat& dy, cv::Mat src, cv::Mat mask)
{
	using namespace cv;
	for (int i = 0; i < dx.rows; ++i) {
		int isflag = 0;

		uchar* ptr = mask.ptr(i);

		for (int j = 0; j < dx.cols; ++j, ++ptr) {
			if (*ptr) {
				// ZeroGradientAt(src, j, i, dx.at<Vec3f>(i, j), dy.at<Vec3f>(i, j));
				GradientAt(src, j, i, dx.at<Vec3f>(i, j), dy.at<Vec3f>(i, j));
				GradientAt(src, j, i - 1, dx.at<Vec3f>(i - 1, j), dy.at<Vec3f>(i - 1, j));
				GradientAt(src, j, i + 1, dx.at<Vec3f>(i + 1, j), dy.at<Vec3f>(i + 1, j));
			}
			//GradientAt(src, j, i, dx.at<Vec3f>(i, j), dy.at<Vec3f>(i, j));
			//GradientAt(src, j - 1, i, dx.at<Vec3f>(i, j - 1), dy.at<Vec3f>(i, j - 1));
			//GradientAt(src, j, i - 1, dx.at<Vec3f>(i - 1, j), dy.at<Vec3f>(i - 1, j));
			//GradientAt(src, j - 1, i - 1, dx.at<Vec3f>(i - 1, j - 1), dy.at<Vec3f>(i - 1, j - 1));
			//GradientAt(src, j - 2, i, dx.at<Vec3f>(i, j - 2), dy.at<Vec3f>(i, j - 2));
			//GradientAt(src, j, i - 2, dx.at<Vec3f>(i - 2, j), dy.at<Vec3f>(i - 2, j));
			//GradientAt(src, j - 2, i - 2, dx.at<Vec3f>(i - 2, j - 2), dy.at<Vec3f>(i - 2, j - 2));
			//GradientAt(src, j + 1, i, dx.at<Vec3f>(i, j + 1), dy.at<Vec3f>(i, j + 1));
			//GradientAt(src, j, i + 1, dx.at<Vec3f>(i + 1, j), dy.at<Vec3f>(i + 1, j));
			//GradientAt(src, j + 1, i + 1, dx.at<Vec3f>(i + 1, j + 1), dy.at<Vec3f>(i + 1, j + 1));
			//GradientAt(src, j + 2, i, dx.at<Vec3f>(i, j + 2), dy.at<Vec3f>(i, j + 2));
			//GradientAt(src, j, i + 2, dx.at<Vec3f>(i + 2, j), dy.at<Vec3f>(i + 2, j));
			//GradientAt(src, j + 2, i + 2, dx.at<Vec3f>(i + 2, j + 2), dy.at<Vec3f>(i + 2, j + 2));
		}
	}
}


cv::Mat stitchImages(std::vector<ImageToMatch>& matches, std::vector<cv::Mat>& images, cv::Mat& originalImage, double f) {
	using namespace cv;

	std::vector<Mat> trans;
	std::vector<Quadrangle> bounds;
	const Mat& ref = matches[0].m1.im;
	int cols = ref.cols;
	int rows = ref.rows;

	Mat identity({ 3, 3 }, {
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0
		});

	Mat trans_cul = identity.clone();

	int mid = matches.size() / 2;
	for (int i = mid - 1; i >= 0; --i) {
		auto& match = matches[i];
		auto& tran = match.m;
		Mat mytran = tran.inv() * trans_cul;
		trans_cul = mytran.clone();

		Quadrangle b;
		b.fromTransformedRect(mytran, Rect(0, 0, cols, rows));

		bounds.push_back(b);
		trans.push_back(mytran);
	}
	std::reverse(bounds.begin(), bounds.end());
	std::reverse(trans.begin(), trans.end());



	trans.push_back(identity.clone());
	{
		Quadrangle b;
		b.fromTransformedRect(trans.back(), Rect(0, 0, cols, rows));
		bounds.push_back(b);
	}

	trans_cul = identity.clone();
	for (int i = mid; i < matches.size(); ++i) {
		auto& match = matches[i];
		auto& tran = match.m;
		Mat mytran = trans_cul * tran;
		trans_cul = mytran.clone();

		Quadrangle b;
		b.fromTransformedRect(mytran, Rect(0, 0, cols, rows));

		bounds.push_back(b);
		trans.push_back(mytran);
	}

	Quadrangle bound;
	for (auto& quad : bounds) {
		bound.wrap(quad);
	}

	for (auto& tran : trans) {
		tran.at<double>(1, 2) -= bound.highest() / 2 - 50;
		tran.at<double>(0, 2) -= bound.leftmost() - 10;
	}

	Quadrangle x;
	{
		
		for (auto& tran : trans) {
			Quadrangle b;
			b.fromTransformedRect(tran, Rect(0, 0, cols, rows));
			x.wrap(b);
		}
		
	}


	auto rect = x.size();

	//auto rect = bound.size();
	//rect.width += 100;
	rect.height += 60;

	Mat stitched(rect, CV_8UC3);
	Mat raw(rect, CV_8UC3);
	Mat tmp(rect, CV_8UC3);

	//Mat tep_mask(rect, CV_8U);
	//tep_mask = 0;
	//tep_mask(Rect(0, 0, cols, rows)) = 255;
	//Mat old_mask(rect, CV_8U);
	stitched = 0;
	raw = 0;

	cv::Mat mask(raw.size(), CV_8UC1);
	mask = 0;
	
	cv::Mat erode_mask = mask.clone();
	cv::Mat erode_mask2 = erode_mask.clone();

	struct Gradients
	{
		cv::Mat x, y;

		Gradients(const cv::Mat& m) {
			Mat dx({ 2, 1 }, {
				1.f, -1.f
				});

			Mat dy({ 1, 2 }, {
				1.f,
				-1.f,
				});
			
			int width = m.cols;
			int height = m.rows;

			Mat color_gradient_x(height, width, CV_32FC3);
			Mat color_gradient_y(height, width, CV_32FC3);

			for (int y = 0; y < height - 1; y++)
			{
				for (int x = 0; x < width - 1; x++)
				{
					GradientAt(m, x, y, color_gradient_x.at<Vec3f>(y, x), color_gradient_y.at<Vec3f>(y, x));
				}
			}
			x = color_gradient_x;
			y = color_gradient_y;
		}

		Gradients(const cv::Mat& m, const cv::Mat& mask) {
			Mat dx({ 2, 1 }, {
				1.f, -1.f
				});

			Mat dy({ 1, 2 }, {
				1.f,
				-1.f,
				});

			int width = m.cols;
			int height = m.rows;

			Mat color_gradient_x(height, width, CV_32FC3, Scalar(0, 0, 0));
			Mat color_gradient_y(height, width, CV_32FC3, Scalar(0, 0, 0));

			
			for (int y = 0; y < height - 1; y++)
			{
				const uchar* ptr = mask.ptr(y);
				float* dx = reinterpret_cast<float*>(color_gradient_x.ptr(y));
				float* dy = reinterpret_cast<float*>(color_gradient_y.ptr(y));

				int x = 0;
				while (*ptr == 0)
				{
					++ptr;
					++x;
				}

				if (x < width - 2) {
					ZeroGradientAt(m, x-1, y, color_gradient_x.at<Vec3f>(y, x-1), color_gradient_y.at<Vec3f>(y, x-1));
				}

				for (; x < width - 1 && *ptr; x++)
				{
					GradientAt(m, x, y, color_gradient_x.at<Vec3f>(y, x), color_gradient_y.at<Vec3f>(y, x));
				}

			}
			x = color_gradient_x;
			y = color_gradient_y;
		}
	};

	cv::Mat dx(raw.size(), CV_32FC3);
	cv::Mat dy = dx.clone();
	dx = Mat::zeros(1, 1, dx.type());
	dy = Mat::zeros(1, 1, dy.type());

	cv::Mat littlemask;
	{
		Mat x = cv::Mat(originalImage.size(), x.type(), Scalar(255, 255, 255));
		Mat y = Panorama5728::toCylinder(x, f);
		
		Mat xs[3];

		split(y, xs);

		littlemask = xs[0];
	}

#ifdef DEBUG_WRITE
	imwrite("results/littlemask.png", littlemask);
#endif // DEBUG_WRITE


	cv::Mat erode_littlemask = littlemask.clone();
	{
		for (int i = 0; i < erode_littlemask.rows; ++i) {
			erode_littlemask.at<uchar>(i, 0) = 0;
			erode_littlemask.at<uchar>(i, erode_littlemask.cols - 1) = 0;
		}
	}
	cv::Mat erode_littlemask2 = erode_littlemask.clone();

	erode(erode_littlemask2, erode_littlemask2, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	erode(erode_littlemask2, erode_littlemask2, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	//erode(erode_littlemask2, erode_littlemask2, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	erode(erode_littlemask2, erode_littlemask, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	erode(erode_littlemask, erode_littlemask, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	//erode(erode_littlemask, erode_littlemask, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	threshold(erode_littlemask2, erode_littlemask2, 100, 255, THRESH_BINARY);
	threshold(erode_littlemask, erode_littlemask, 100, 255, THRESH_BINARY);
#ifdef DEBUG_WRITE
	imwrite("results/erode_littlemask.png", erode_littlemask);
	imwrite("results/littlemask_sub.png", erode_littlemask2 - erode_littlemask);
#endif // DEBUG_WRITE
	{
		tmp = Mat::zeros(1, 1, tmp.type());

		int i = 0;
		
		
		warpPerspective(matches[i].m1.im, tmp, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		warpPerspective(matches[i].m1.im, raw, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		warpPerspective(littlemask, mask, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);

		Gradients grad(tmp);
		
		dx = grad.x;
		dy = grad.y;

		//warpPerspective(grad.x, dx, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		//warpPerspective(grad.y, dy, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
	}

	for (int i = 0; i < matches.size(); ++i) {
		tmp = Mat::zeros(1, 1, tmp.type());
		// erode_mask = Mat::zeros(1, 1, erode_mask.type());
		memset(erode_mask.data, 0, erode_mask.cols* erode_mask.rows);
		memset(erode_mask2.data, 0, erode_mask.cols* erode_mask.rows);
		warpPerspective(erode_littlemask, erode_mask, trans[i + 1], stitched.size(), INTER_NEAREST, BORDER_TRANSPARENT);
		warpPerspective(erode_littlemask2, erode_mask2, trans[i + 1], stitched.size(), INTER_NEAREST, BORDER_TRANSPARENT);

		warpPerspective(matches[i].m2.im, tmp, trans[i + 1], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);

		Mat tmp_masked = MaskImage(tmp, erode_mask2);

		Gradients grad(tmp_masked);
		//MergeImage<float>(dx, grad.x, mask, erode_mask2, 18);
		//MergeImage<float>(dy, grad.y, mask, erode_mask2, 18);

		MergeImage2<float>(dx, grad.x, mask, erode_mask2, erode_mask, 10);
		MergeImage2<float>(dy, grad.y, mask, erode_mask2, erode_mask, 10);

		tmp = Mat::zeros(1, 1, tmp.type());
		warpPerspective(matches[i].m2.im, tmp, trans[i + 1], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		MergeImage<uchar>(raw, tmp, mask, erode_mask, 1);

		//tmp = 0;
		// warpPerspective(grad.x, tmp, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		
		MergeImage<uchar, 1>(mask, erode_mask, mask, erode_mask, 0);
		//tmp = 0;
		//warpPerspective(grad.y, tmp, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		
		//warpPerspective(erode_littlemask, mask, trans[i + 1], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		//warpPerspective(grad.x, dx, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		//warpPerspective(grad.y, dy, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);

#ifdef DEBUG_WRITE
		imwrite("results/grad.x."+ std::to_string(i) + ".png", grad.x * 10);
		imwrite("results/grad.y."+ std::to_string(i) + ".png", grad.y * 10);
		imwrite("results/emask." + std::to_string(i) + ".png", erode_mask);
		imwrite("results/mask." + std::to_string(i) + ".png", mask);
		imwrite("results/mask.sub." + std::to_string(i) + ".png", erode_mask2 - erode_mask);
#endif // DEBUG_WRITE

	}
	Mat res(raw.size(), CV_8UC3);
	Mat fill_erode_mask;
	erode(mask, fill_erode_mask, getStructuringElement(MORPH_CROSS, cv::Size(3, 3)));
	EnforceGradientBound(dx, dy, raw, mask - fill_erode_mask);
	imwrite("results/dx.png", dx * 10);
	imwrite("results/dy.png", dy * 10);

	// Gradient Domain Fushion
	{
		int width = raw.cols;
		int height = raw.rows;

		std::vector<double> r(width*height), g(width* height), b(width* height);
		Mat draw;
		raw.convertTo(draw, CV_64FC3);

		Mat channles[3];
		split(draw, channles);

		memcpy(b.data(), channles[2].data, sizeof(double)* b.size());
		memcpy(g.data(), channles[1].data, sizeof(double)* b.size());
		memcpy(r.data(), channles[0].data, sizeof(double)* b.size());


		Vec3b color0 = matches[0].m1.im.at<Vec3b>(0, 0);
		SolveChannel(0, color0[0], dx, dy, res, 50, r, mask);
		SolveChannel(1, color0[1], dx, dy, res, 50, g, mask);
		SolveChannel(2, color0[2], dx, dy, res, 50, b, mask);
	}
	

	//blender.blend(stitched, tep_mask);
	//stitched.convertTo(stitched, CV_8UC3);
	cv::imwrite("results/stitched_not_smoothed.jpg", raw);
	cv::imwrite("results/stitched.jpg", res);
	// imwrite("results/stitched_final.jpg", stitched);
	return res;
}


cv::Mat Panorama5728::toCylinder(const cv::Mat& im, double f)
{
	
    auto [cols, rows] = GetCylinderCoords(im.cols, im.rows, f, im.cols, im.rows);
	auto [cols1, rows1] = GetCylinderCoords(im.cols / 2, im.rows, f, im.cols, im.rows);
	cols = std::max(cols, cols1);
	rows = std::max(rows, rows1);

	auto [k, v] = GetPlanarCoords(cols, rows, f, im.cols, im.rows);

    cv::Mat res(rows, cols, CV_8UC3);

    // TODO: use bilinear sample
	
    res.forEach<cv::Vec3b>([&](cv::Vec3b& v, const int* pos) {
        auto [x, y] = GetPlanarCoords(pos[1], pos[0], f, im.cols, im.rows);
        if (x >= im.cols || y >= im.rows || x < 0 || y < 0) {
            v[0] = v[1] = v[2] = 0;
        }
        else {
            v = im.at<cv::Vec3b>(y, x);
        }
    });
	
	/*
	auto [cols, rows] = GetCylinderCoords(im.cols, im.rows, f, im.cols, im.rows);
	auto [cols1, rows1] = GetCylinderCoords(im.cols / 2, im.rows, f, im.cols, im.rows);
	cols = std::max(cols, cols1);
	rows = std::max(rows, rows1);

	cv::Mat res(rows, cols, CV_8UC3);
	im.forEach<cv::Vec3b>([&](cv::Vec3b& v, const int* pos) {
		auto [x, y] = GetCylinderCoords(pos[1], pos[0], f, im.cols, im.rows);
		if (x >= res.cols || y >= res.rows) {
			v[0] = v[1] = v[2] = 0;
		}
		else {
			res.at<cv::Vec3b>(y, x) = v;
		}
	});
	*/
    return res;
}

cv::Mat Panorama5728::toPlanar(const cv::Mat& im, double f)
{
	auto [cols, rows] = GetPlanerWidthHeight(im.cols, im.rows, f);
	double width = cols;
	double height = rows;
	cv::Mat res(rows, cols, CV_8UC3);

	// TODO: use bilinear sample
	res.forEach<cv::Vec3b>([&](cv::Vec3b& v, const int* pos) {
		auto [x, y] = GetCylinderCoords(pos[1], pos[0], f, width, height);
		if (x >= im.cols || y >= im.rows) {
			v[0] = v[1] = v[2] = 0;
		}
		else {
			v = im.at<cv::Vec3b>(y, x);
		}
	});


	return res;
}

SparseMatrix<double> ConvertFromEigen(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
{
	SparseMatrix<double> sp;
	sp.initializeFromEigenRowMajor(
		mat.valuePtr(), mat.data().allocatedSize(),
		mat.outerIndexPtr(), mat.outerSize(),
		mat.innerIndexPtr(), mat.innerSize(),
		mat.innerNonZeroPtr(), mat.outerSize()
	);
	return std::move(sp);
}


void SolveChannel(
	int channel_idx, int constraint, 
	const cv::Mat& color_gradient_x, const cv::Mat& color_gradient_y, 
	cv::Mat& output, //const std::vector<cv::Mat>& Images, cv::Mat Labels,
	 int iterations,
	std::vector<double> init,
	const cv::Mat& mask)
{

	int width = color_gradient_x.cols;
	int height = color_gradient_x.rows;


	int NumOfUnknownTerm = 2 * width * height + 1;
	std::vector<Eigen::Triplet<double>> NonZeroTerms;
	NonZeroTerms.resize((height - 1) * (width - 1) * 4);
	Eigen::VectorXd b(NumOfUnknownTerm);

	// 构建求解泊松方程的稀疏矩阵
	for (int y = 0; y < height - 1; y++)
	{
		int idx = y * (width - 1) * 4;
		for (int x = 0; x < width - 1; x++)
		{
			int col_xy = width * y + x;

			// 第 2k 行, 在矩阵里表示 v(x+1,y)-v(x,y)
			int row_xy = 2 * col_xy;
			int col_x1y = col_xy + 1;
			NonZeroTerms[idx++] = Eigen::Triplet<double>(row_xy, col_xy, -1); // -v(x, y)
			NonZeroTerms[idx++] = Eigen::Triplet<double>(row_xy, col_x1y, 1); // v(x+1, y)
			cv::Vec3f grads_x = color_gradient_x.at<cv::Vec3f>(y, x);
			b(row_xy) = grads_x[channel_idx];

			// 第 2k + 1 行, 在矩阵里表示 v(x,y+1)-v(x,y)
			int row_xy1 = row_xy + 1;
			int col_xy1 = col_xy + width;
			NonZeroTerms[idx++] = Eigen::Triplet<double>(row_xy1, col_xy, -1); // -v(x, y)
			NonZeroTerms[idx++] = Eigen::Triplet<double>(row_xy1, col_xy1, 1); // v(x, y+1)
			cv::Vec3f grads_y = color_gradient_y.at<cv::Vec3f>(y, x);
			b(row_xy1) = grads_y[channel_idx];
		}
	}



	///constraint
	int eq_idx = width * height * 2;
	NonZeroTerms.push_back(Eigen::Triplet<double>(eq_idx, 0, 1));
	b(eq_idx) = constraint;

	Eigen::SparseMatrix<double> A(NumOfUnknownTerm, width * height);
	A.setFromTriplets(NonZeroTerms.begin(), NonZeroTerms.end());

	// 释放内存
	NonZeroTerms.clear();
	NonZeroTerms.shrink_to_fit();

	Eigen::SparseMatrix<double, Eigen::RowMajor> ATA(width * height, width * height);
	ATA = A.transpose() * A;
	Eigen::VectorXd ATb = A.transpose() * b;


	auto MyMatrix = ConvertFromEigen(ATA);
	std::vector<double> myATb;
	myATb.insert(myATb.begin(), ATb.data(), ATb.data() + ATb.rows());

	//std::vector<double> init;

	printf("\nSolving...\n");
	auto mysolution = MyMatrix.conjugateGradient(myATb, 1e-10, iterations, init);
	printf("Solved!\n");
	// vecadd(mysolution, 90, mysolution);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec3b& temp = output.at<cv::Vec3b>(y, x);
			temp[channel_idx] = uchar(std::max(std::min(mysolution[y * width + x], 255.0), 0.0));
			//printf("%d,%d,  %f, %f\n",y,x, solution(y * width + x), ATb(y*width + x));
			//system("pause");
		}
	}
}

// https://stackoverflow.com/questions/13856975/how-to-sort-file-names-with-numbers-and-alphabets-in-order-in-c
int strcasecmp_withNumbers(const char* a, const char* b) {;

	if (!a || !b) { // if one doesn't exist, other wins by default
		return a ? 1 : b ? -1 : 0;
	}
	if (isdigit(*a) && isdigit(*b)) { // if both start with numbers
		char* remainderA;
		char* remainderB;
		long valA = strtol(a, &remainderA, 10);
		long valB = strtol(b, &remainderB, 10);
		if (valA != valB)
			return valA - valB;
		// if you wish 7 == 007, comment out the next two lines
		else if (remainderB - b != remainderA - a) // equal with diff lengths
			return (remainderB - b) - (remainderA - a); // set 007 before 7
		else // if numerical parts equal, recurse
			return strcasecmp_withNumbers(remainderA, remainderB);
	}
	if (isdigit(*a) || isdigit(*b)) { // if just one is a number
		return isdigit(*a) ? -1 : 1; // numbers always come first
	}
	while (*a && *b) { // non-numeric characters
		if (isdigit(*a) || isdigit(*b))
			return strcasecmp_withNumbers(a, b); // recurse
		if (tolower(*a) != tolower(*b))
			return tolower(*a) - tolower(*b);
		a++;
		b++;
	}
	return *a ? 1 : *b ? -1 : 0;
}

bool sort_file(const fs::path& a, const fs::path& b) {
	auto x = a.filename().string();
	auto y = b.filename().string();
	int res =  strcasecmp_withNumbers((const char*)x.c_str(), (const char*)y.c_str());
	return res < 0;
}



int main()
{
	double f = 512.89;
	std::vector<fs::path> files;
	std::vector<cv::Mat> mats;

	std::vector<fs::path> pathes;

	std::cout << "Input dir name:" << std::endl;
	for (auto& path : fs::directory_iterator("."))
	{
		if (fs::is_directory(path)) {
			std::cout << "[" << pathes.size() << "] " << fs::path(path).filename() << std::endl;
			pathes.push_back(path);
		}
	}

	int choice = -1;
	
	std::cin >> choice;
	std::string path = "panorama-data2/";
	if (choice >= 0 && choice < pathes.size()) {
		path = pathes[choice].string();
	}
	else {
		std::cout << "Trying to open dir: " << path << std::endl;
	}
	
	for (auto& file : fs::directory_iterator(path))
	{
		files.push_back(file);
	}

	auto K_txt = std::find(files.begin(), files.end(), path / fs::path("K.txt"));
	if (K_txt == files.end()) {
		std::cout << "K.txt not found, use default f instead" << std::endl;
	}
	else {
		std::ifstream ifs(*K_txt);
		ifs >> f;
		files.erase(K_txt);
		std::cout << "f is set to " << f << std::endl;
	}

	std::sort(files.begin(), files.end(), sort_file);
	for (auto& file : files) {
		auto mat = cv::imread(file.string());
		if (!mat.empty()) {
			mats.push_back(mat);
			std::cout << "Image: " << file << std::endl;
		}
		
	}


	Panorama5728 pano;
	cv::Mat result;
	pano.makePanorama(mats, result, f);



	//int n = 0;
	//for (int i = 1; i < 21; ++i)
	//{
	//	std::string file = path + std::to_string(i) + ".jpg";
	//	if (fs::exists(file))
	//	{
	//		mats.push_back(cv::imread(file));
	//	}
	//}
	//
	//Panorama5728 pano;
	//cv::Mat result;
	//pano.makePanorama(mats, result, 512.89);

	//cv::namedWindow("matched", cv::WINDOW_NORMAL);
	//cv::imshow("matched", result);
	//cv::imwrite("results/planar.png", Panorama5728::toPlanar(result, 512.89));

	// cv::waitKey(0);

}
