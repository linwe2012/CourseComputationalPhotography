#pragma once
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/flann/flann.hpp"
#include <tuple>

class PhotoMontage
{
public:
	enum Objectives {
		kMatchColors,
		kMatchGradients,
		kMatchColorAndGradients,
		kMatchColorsAndEdges,
		kMatchLaplacian
	};
private:
	void BuildSolveMRF(const std::vector<cv::Mat> & Images, const cv::Mat & Label, enum Objectives objectives, bool preview, int iterations);
	void VisResultLabelMap(const cv::Mat & ResultLabel, int n_label);
	void VisCompositeImage(const cv::Mat & ResultLabel, const std::vector<cv::Mat> & Images);
	void BuildSolveGradientFusion(const std::vector<cv::Mat> & Images, const cv::Mat & ResultLabel, int preview);

	void SolveChannel( int channel_idx, int constraint, const cv::Mat &color_gradient_x, const cv::Mat &color_gradient_y , cv::Mat & output , const std::vector<cv::Mat>& Images);

	void GradientAt(const cv::Mat & Image, int x, int y, cv::Vec3f & grad_x, cv::Vec3f & grad_y);

	

	
public:
	enum Stage {
		kDonePrepareData,
		kOptimizeDone,
		kDoneSolveChannel1,
		kDoneSolveChannel2,
		kDoneSolveChannel3,
	};

	int fast_init_value = 0;
	
	std::function<void(double, enum Stage)>progress;

	void Run(const std::vector<cv::Mat> & Images, const cv::Mat & Label, enum Objectives objectives = kMatchColors,
		bool preview = true, int iterations = 5000);
	
	auto GetResultAndLabel()
	{
		return std::make_tuple(result_, result_label_);
	}

private:
	cv::flann::Index * AddInertiaConstraint( const cv::Mat & Label );

	void Progress(double _0_1_progress, enum Stage stage);
	
	cv::Mat result_;
	cv::Mat result_label_;
	int iterations_;
public:
	enum
	{
		undefined = -1
	};
};