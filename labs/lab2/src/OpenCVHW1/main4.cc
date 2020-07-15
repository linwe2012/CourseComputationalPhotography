#include "config.h"

#ifdef HW_4

#define CVUI_IMPLEMENTATION
#define CVUI_USE_CPP_FS

#include <opencv2/opencv.hpp>
#include "unitext/cvUniText.hpp"
#include "utils.h"
#include "cvui.h"
#include <numeric>

#include <filesystem>



namespace fs = std::filesystem;

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
		catch (cv::Exception & e) {
			std::cout << e.what();
		}
		
	}

	cv::Ptr<cv::Feature2D> pdetector;
	cv::Ptr<cv::DescriptorMatcher> pmatcher;
};


class ImageToStitch {
public:
	ImageToStitch(std::string p, InjectAlgorithm algo) {
		using namespace cv;

		im = imread(p);

		algo.detect(im, keypoints, descriptors);
	}

	cv::Mat im;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

void expoureCorrect(std::vector<ImageToStitch>& imgs) {
	std::vector<double> estimate;
	double div = imgs[0].im.cols * imgs[0].im.rows;

	// compute estimate lightness for each image
	for (auto& img : imgs) {
		auto gray = grayscale(img.im);
		// thresholding out undesired pixels
		cv::threshold(gray, gray, 20, 255, cv::THRESH_TOZERO);
		cv::threshold(gray, gray, 250, 255, cv::THRESH_TOZERO_INV);

		// just sum them up
		auto mysum = sum(gray);
		estimate.push_back((mysum[0]) / div);
	}

	// find median
	auto sorted = estimate;
	std::sort(sorted.begin(), sorted.end());
	double median = sorted[sorted.size() / 2];
	int i = 0;

	// adjust lights
	for (auto& img : imgs) {
		img.im += (median- estimate[i]) / 100;
		++i;
	}

}

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

		m = findHomography(b,a, RANSAC);
	}

	std::vector<cv::DMatch> matched;
	cv::Mat m;
	const ImageToStitch& m1;
	const ImageToStitch& m2;
	std::vector<cv::Point2f> a;
	std::vector<cv::Point2f> b;
};

cv::Mat stitchImages(std::vector<ImageToMatch>& matches);

int main()
{
	using namespace cv;

	InjectAlgorithm algo;

	std::vector<std::string> selections = {
		"yosemite/",
		"caozhu_xs/",
	};
	
	std::cout << "Pick a number: " << std::endl;
	for (int i = 0; i < selections.size(); ++i) {
		std::cout << "[" << i << "]: " << selections[i] << std::endl;
	}

	std::string base_path = "yosemite/";
	std::string str;
	std::getline(std::cin, str);
	if (str.size() == 0) {
		// do nothing
	}
	else if(isdigit(str[0])) {
		base_path = selections[str[0] - '0'];
	}
	else {
		base_path = str;
	}
	

	std::vector<ImageToStitch> imgs;
	{
		int i = 1;
		while (true)
		{
			std::string possible = base_path + std::to_string(i) + ".jpg";
			if (fs::exists(possible)) {
				std::cout << "Detecting keypoints: " << possible << "...";
				imgs.emplace_back(possible, algo);
				Mat res;
				drawKeypoints(imgs.back().im, imgs.back().keypoints, res);
				imwrite("results/keypoints_" + std::to_string(i) + ".jpg", res);
				std::cout << " [Done]" << std::endl;
			}
			else {
				break;
			}
			++i;
		}
	}
	expoureCorrect(imgs);

	std::vector<ImageToMatch> pairs;
	for (int i = 0; i < int(imgs.size()) - 1; ++i) {
		std::cout << "Matching " << i << ", " << i + 1 <<  "...";
		pairs.emplace_back(imgs[i], imgs[i + 1], algo);

		auto& im1 = imgs[i];
		auto& im2 = imgs[i + 1];

		// write our results
		Mat disp;
		drawMatches(im1.im, im1.keypoints, im2.im, im2.keypoints, pairs.back().matched, disp);
		imwrite("results/match_" + std::to_string(i) + "_" + std::to_string(i+1) + ".jpg", disp);

		std::cout << " [Done]" << std::endl;
	}


	Mat stitched = stitchImages(pairs);

	namedWindow("matched", WINDOW_NORMAL);
	imshow("matched", stitched);


	waitKey(0);
}


struct Quadrangle {
	enum {
		TopLeft,
		BottomLeft,
		BottomRight,
		TopRight
	};
	cv::Point3d box[4];

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
			//box[idx].x /= box[idx].z;
			//box[idx].y /= box[idx].z;
		};

		apply(TopLeft,     r.tl().x, r.tl().y);
		apply(BottomLeft,  r.tl().x, r.br().y);
		apply(BottomRight, r.br().x, r.br().y);
		apply(TopRight,    r.br().x, r.tl().y);
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

cv::Point2f operator*(cv::Mat M, const cv::Point2f& p)
{
	cv::Mat_<double> src(3/*rows*/, 1 /* cols */);

	src(0, 0) = p.x;
	src(1, 0) = p.y;
	src(2, 0) = 1.0;

	cv::Mat_<double> dst = M * src; //USE MATRIX ALGEBRA 
	return cv::Point2f(dst(0, 0), dst(1, 0));
}

void maxLighting(cv::Mat& src, cv::Mat& dst, cv::Mat src_mask, cv::Mat dst_mask) {
	using namespace cv;

	//detail::MultiBandBlender blender(false);
	//blender.prepare(Rect(0, 0, src.cols, src.rows));
	//blender.feed(src, src_mask, Point(0, 0));
	//blender.feed(dst, dst_mask, Point(0, 0));
	//Mat  res_mask;
	//blender.blend(dst, res_mask);
	//dst.convertTo(dst, CV_8UC3);
	//return;

	auto is_zero = [](uchar* p) {
		return p[0] == 0 && p[1] == 0 && p[2] == 0;
	};
#define MAX_T 70
#define MAX_U 200
	auto is_over = [](uchar* p, uchar* q) {
		return (abs(p[0] - q[0]) > MAX_T) || (abs(p[1] - q[1]) > MAX_T) || (abs(p[2] - q[2]) > MAX_T) || 
			(abs(p[0]+p[1]+p[2]-q[0]-q[1]-q[2]) > MAX_U);
	};
#undef MAX_U
#undef MAX_T
	for (int i = 0; i < src.rows; ++i) {
		auto s = src.ptr(i);
		auto d = dst.ptr(i);
		int j = 0;
		int cnt = 0;
		constexpr int maxcnt = 120;

		for (; j < src.cols; ++j) {
			if (is_zero(s)) {
				s += 3;
				d += 3;
			}
			else {
				break;
			}
		}

		for (; j < src.cols && cnt < maxcnt; ++j, ++cnt) {
			double ratio = double(cnt) / maxcnt;
			if (is_zero(d) || is_over(s, d)) {
				//if (!is_zero(s)) {
					for (int k = 0; k < 3; ++k) {
						d[k] = s[k];
					}
				//}
			}
			else {
				for (int k = 0; k < 3; ++k) {
					d[k] = s[k] * ratio + d[k] * (1 - ratio);
				}
			}
			
			d += 3;
			s += 3;
		}
		memcpy(d, s, (src.cols - j) * 3);
	}
}


cv::Mat stitchImages(std::vector<ImageToMatch>& matches) {
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
	for (int i = mid-1; i >= 0; --i) {
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
		tran.at<double>(1, 2) -= bound.highest()/2;
		tran.at<double>(0, 2) -= bound.leftmost();
	}

	auto rect = bound.size();
	rect.width += 800;
	Mat stitched(rect, CV_8UC3);
	Mat raw(rect, CV_8UC3);
	Mat tmp (rect, CV_8UC3);
	//Mat tep_mask(rect, CV_8U);
	//tep_mask = 0;
	//tep_mask(Rect(0, 0, cols, rows)) = 255;
	//Mat old_mask(rect, CV_8U);
	stitched = 0;
	raw = 0;

	//detail::MultiBandBlender blender(false);
	//blender.prepare(Rect(0, 0, stitched.cols, stitched.rows));
	Mat erosion({ 3,3 }, {
		0,1,0,
		1,1,1,
		0,1,0
		});
	erosion.convertTo(erosion, CV_8U);
	{
		tmp = 0;

		int i = 0;
		warpPerspective(matches[i].m1.im, tmp, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		warpPerspective(matches[i].m1.im, raw, trans[i], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		// warpPerspective(old_mask, tep_mask, trans[i], stitched.size());
		maxLighting(tmp, stitched, (grayscale(tmp) > 0) * 255, (grayscale(stitched) > 0)*255);
		//Mat mask = (grayscale(tmp) > 0);
		//erode(mask, mask, erosion);
		//blender.feed(tmp.clone(), grayscale(tmp) > 0, Point(0, 0));
	}
	
	
	for (int i = 0; i < matches.size(); ++i) {
		tmp = 0;
		warpPerspective(matches[i].m2.im, tmp, trans[i+1], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		warpPerspective(matches[i].m2.im, raw, trans[i+1], stitched.size(), INTER_LINEAR, BORDER_TRANSPARENT);
		// warpPerspective(old_mask, tep_mask, trans[i+1], stitched.size());
		maxLighting(tmp, stitched, (grayscale(tmp) > 0) * 255, (grayscale(stitched) > 0)*255);
		//Mat mask = (grayscale(tmp) > 0);
		//erode(mask, mask, erosion);
		
		//blender.feed(tmp.clone(), mask, Point(0, 0));
	}
	//blender.blend(stitched, tep_mask);
	//stitched.convertTo(stitched, CV_8UC3);
	imwrite("results/stitched_not_smoothed.jpg", raw);
	imwrite("results/stitched_final.jpg", stitched);
	return stitched;
}


#endif