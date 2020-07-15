#include "stdafx.h"
#include "PhotoMontage.h"
#include "GCoptimization.h"
#include <Eigen/Sparse>

#define USE_DEMO 1

#include "utils.h"



using namespace cv;

constexpr double penalty_scale = 1;

const double large_penalty = 1000.0;
Mat _data;

struct __ExtraData
{
	std::vector<cv::Mat> Images;

	std::vector<cv::Mat> ImagesDx;
	std::vector<cv::Mat> ImagesDy;
	std::vector<cv::Mat> ImagesEdge;

	// enum PhotoMontage::Objectives objectives;

	cv::Mat Label;
	cv::flann::Index * kdtree;
};

double dataFn(int p, int l, void *data)
{
	///请同学们填写这里的代码，这里就是实验中所说的数据项
	auto ptr_extra_data = static_cast<__ExtraData*>(data);
	const auto& Label = ptr_extra_data->Label;

	auto width = Label.cols;

	int x = p % width;
	int y = p / width;

	auto label = Label.at<char>(y, x);

	// 用户指定的像素点
	if (label != PhotoMontage::undefined)
	{
		if (label == l) {
			return 0.0;
		}

		// 这个像素点和用户指定的 label 不一样
		return large_penalty;
	}

	return large_penalty;
}

double euc_dist(const Vec3b & a, const Vec3b & b)
{
	Vec3d double_diff = a - b;
	return sqrt( double_diff[0] * double_diff[0] + double_diff[1] * double_diff[1] + double_diff[2] * double_diff[2]);
}

template<enum PhotoMontage::Objectives obj>
double smoothFnT(int p, int q, int lp, int lq, void* data)
{
	///请同学们填写这里的代码，这里就是实验中所说的平滑项 

	// Interactive Panelty (P)

	// P : X     if match colors
	//   | Y     if match gradients
	//   | X+Y   if match color & gradients
	//   | X/Z   if match color & edges

	// p, q 是同一个 label
	if (lp == lq)
	{
		return 0.0;
	}

	auto ptr_extra_data = static_cast<__ExtraData*>(data);
	const auto& Label = ptr_extra_data->Label;
	const auto& Images = ptr_extra_data->Images;

	auto width = Label.cols;
	// 图像的坐标
	const int coords[] = {
		p / width, // yp
		p % width, // xp
		q / width, // yq
		q % width, // xq
	};

	double res = 0;
	double X = 0;
	double Y = 0;
	

	if constexpr (obj == PhotoMontage::kMatchColors ||
		obj == PhotoMontage::kMatchColorsAndEdges ||
		obj == PhotoMontage::kMatchColorsAndEdges)
	{
		// 两个像素在两张图片上的欧氏距离
		auto dist = [&Images, &coords, lp, lq](int idx1, int idx2)
		{
			auto& a = Images[lp].at<Vec3b>(coords[idx1], coords[idx1 + 1]);
			auto& b = Images[lq].at<Vec3b>(coords[idx2], coords[idx2 + 1]);

			Vec3d c = a - b;
			auto res = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
			return res;
		};
		X = dist(0, 2) + dist(2, 0);
	}

	if constexpr (obj == PhotoMontage::kMatchGradients ||
		obj == PhotoMontage::kMatchColorAndGradients)
	{
		auto& ImDx = ptr_extra_data->ImagesDx;
		auto& ImDy = ptr_extra_data->ImagesDy;

		auto dist = [&ImDx, &ImDy, &coords, lp, lq](int idx1, int idx2)
		{
			auto& ax = ImDx[lp].at<Vec3f>(coords[idx1], coords[idx1 + 1]);
			auto& ay = ImDy[lp].at<Vec3f>(coords[idx1], coords[idx1 + 1]);

			auto& bx = ImDx[lq].at<Vec3f>(coords[idx2], coords[idx2 + 1]);
			auto& by = ImDy[lq].at<Vec3f>(coords[idx2], coords[idx2 + 1]);


			Vec3d cx = ax - bx;
			Vec3d cy = ay - by;

			auto res = sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]+
							cy[0] * cy[0] + cy[1] * cy[1] + cy[2] * cy[2]);
			return res;
		};

		Y = dist(0, 2) + dist(2, 0);
	}

	if constexpr (obj == PhotoMontage::kMatchColorsAndEdges)
	{
		auto& ImEdge = ptr_extra_data->ImagesEdge;
		auto dist = [&ImEdge, &coords](int l)
		{
			auto& a = ImEdge[l].at<uchar>(coords[0], coords[1]);
			auto& b = ImEdge[l].at<uchar>(coords[2], coords[3]);

			return abs(a - b);
		};

		double Z = dist(lp) + dist(lq) + 1;
		res = X / Z;
	}
	
	if constexpr (obj == PhotoMontage::kMatchColors)
	{
		res = X;
	}

	if constexpr (obj == PhotoMontage::kMatchGradients)
	{
		res = X + Y;
	}

	if constexpr (obj == PhotoMontage::kMatchColorAndGradients)
	{
		res = X + Y;
	}

	if constexpr (obj == PhotoMontage::kMatchLaplacian)
	{
		auto& ImLap = ptr_extra_data->ImagesDx;

		auto dist = [&ImLap, &coords, lp, lq](int idx1, int idx2)
		{
			auto& ax = ImLap[lp].at<Vec3f>(coords[idx1], coords[idx1 + 1]);
			auto& bx = ImLap[lq].at<Vec3f>(coords[idx2], coords[idx2 + 1]);


			Vec3d cx = ax - bx;

			auto res = sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]);
			return res;
		};

		res = dist(0, 2) + dist(2, 0);
	}

	return res;
}

/*
double smoothFn(int p, int q, int lp, int lq, void * data)
{
	///请同学们填写这里的代码，这里就是实验中所说的平滑项 

	// Interactive Panelty (P)

	// P : X     if match colors
	//   | Y     if match gradients
	//   | X+Y   if match color & gradients
	//   | X/Z   if match color & edges

	// p, q 是同一个 label
	if (lp == lq)
	{
		return 0.0;
	}

	auto ptr_extra_data = static_cast<__ExtraData*>(data);
	const auto& Label = ptr_extra_data->Label;
	const auto& Images = ptr_extra_data->Images;

	auto width = Label.cols;
	const int coords[] = {
		p / width, // yp
		p % width, // xp
		q / width, // yq
		q % width, // xq
	};

	auto dist = [&Images, &coords, lp, lq](int idx1, int idx2)
	{
		auto& a = Images[lp].at<Vec3b>(coords[idx1], coords[idx1 + 1]);
		auto& b = Images[lq].at<Vec3b>(coords[idx2], coords[idx2 + 1]);

		Vec3d c = a - b;
		auto res =  sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
		return res;
	};

	double X = dist(0, 2) + dist(2, 0);

	return X;
}
*/


void PhotoMontage::Run( const std::vector<cv::Mat> & Images, const cv::Mat & Label, enum Objectives objectives, bool preview, int iterations)
{	
	assert(Images[0].rows == Label.rows);
	assert(Images[0].cols == Label.cols);
	iterations_ = iterations;

	BuildSolveMRF( Images, Label , objectives, preview, iterations);
	
}

void PhotoMontage::BuildSolveMRF( const std::vector<cv::Mat> & Images, const cv::Mat & Label, enum Objectives objectives, bool preview, int iterations)
{
	const auto n_imgs = Images.size();
	__ExtraData extra_data;
	extra_data.Images.resize(n_imgs);
	
	for (int i = 0 ; i < n_imgs; i++)
	{
		extra_data.Images[i] = Images[i];
	}

	if (objectives == PhotoMontage::kMatchColorAndGradients || objectives == PhotoMontage::kMatchGradients)
	{
		extra_data.ImagesDx.resize(n_imgs);
		extra_data.ImagesDy.resize(n_imgs);

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

		for (int i = 0; i < n_imgs; i++)
		{
			cv::Mat dx, dy;
			cv::filter2D(Images[i], dx, CV_32FC3, sobel_x);
			cv::filter2D(Images[i], dy, CV_32FC3, sobel_y);
			extra_data.ImagesDx[i] = dx;
			extra_data.ImagesDy[i] = dy;
		}
	}

	if (objectives == PhotoMontage::kMatchColorsAndEdges)
	{
		extra_data.ImagesEdge.resize(n_imgs);
		for (int i = 0; i < n_imgs; i++)
		{
			cv::Mat edge;
			cv::cvtColor(Images[i], edge, cv::COLOR_BGR2GRAY);

			cv::Canny(edge, edge, 200 / 3, 200);
			extra_data.ImagesEdge[i] = edge;
		}
	}

	if (objectives == PhotoMontage::kMatchLaplacian)
	{
		Mat laplacian({ 3, 3 }, {
		1.0, 1.0, 1.0,
		1.0, -8.0, 1.0,
		1.0, 1.0, 1.0
			});

		extra_data.ImagesDx.resize(n_imgs);
		for (int i = 0; i < n_imgs; i++)
		{
			cv::filter2D(Images[i], extra_data.ImagesDx[i], CV_32FC3, laplacian);
		}
	}

	extra_data.Label = Label;
	//extra_data.kdtree = AddInertiaConstraint( Label );
	int width = Label.cols;
	int height = Label.rows;
	int n_label = n_imgs;

	Progress(0.1, kDonePrepareData);

	try
	{
		//VisResultLabelMap(Label,n_label);

		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,n_imgs);

		// set up the needed data to pass to function for the data costs
		gc->setDataCost(&dataFn,&extra_data);

		if (objectives == PhotoMontage::kMatchColors)
		{
			// smoothness comes from function pointer
			gc->setSmoothCost(&smoothFnT<kMatchColors>, &extra_data);
		}

		if (objectives == PhotoMontage::kMatchGradients)
		{
			// smoothness comes from function pointer
			gc->setSmoothCost(&smoothFnT<kMatchGradients>, &extra_data);
		}

		if (objectives == PhotoMontage::kMatchColorAndGradients)
		{
			// smoothness comes from function pointer
			gc->setSmoothCost(&smoothFnT<kMatchColorAndGradients>, &extra_data);
		}

		if (objectives == PhotoMontage::kMatchColorsAndEdges)
		{
			// smoothness comes from function pointer
			gc->setSmoothCost(&smoothFnT<kMatchColorsAndEdges>, &extra_data);
		}
		

		printf("\nBefore optimization energy is %f",gc->compute_energy());
		// gc->swap(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		gc->swap(-1);
		printf("\nAfter optimization energy is %f",gc->compute_energy());

		Progress(0.3, kOptimizeDone);

		Mat result_label(height, width, CV_8UC1);
		
		for (int y = 0 ; y < height ; y++)
		{
			for (int x = 0 ; x < width; x++)
			{
				int idx = y * width + x;

				result_label.at<uchar>(y,x) = gc->whatLabel(idx);
			}
		}		
		delete gc;

		result_label_ = result_label;

		if (preview)
		{
			VisResultLabelMap(result_label, n_label);
		}
		
		//VisCompositeImage( result_label, Images );

		BuildSolveGradientFusion(Images, result_label, preview);
	}
	catch (GCException e)
	{
		e.Report();
	}
}

void PhotoMontage::GradientAt( const cv::Mat & Image, int x, int y, cv::Vec3f & grad_x, cv::Vec3f & grad_y )
{
	
	Vec3i color1 = Image.at<Vec3b>(y, x);
	Vec3i color2 = Image.at<Vec3b>(y, x + 1);
	Vec3i color3 = Image.at<Vec3b>(y + 1, x);
	grad_x = color2 - color1;
	grad_y = color3 - color1;
	
}

void PhotoMontage::BuildSolveGradientFusion( const std::vector<cv::Mat> & Images, const cv::Mat & ResultLabel , int preview)
{

	int width = ResultLabel.cols;
	int height = ResultLabel.rows;
	Mat color_result(height, width, CV_8UC3);
	Mat color_gradient_x(height, width, CV_32FC3);
	Mat color_gradient_y(height, width, CV_32FC3);
	
	for (int y = 0 ; y < height - 1; y++)
	{
		for (int x = 0 ;x < width - 1; x++)
		{
			GradientAt( Images[ResultLabel.at<uchar>(y,x)], x, y, color_gradient_x.at<Vec3f>(y,x), color_gradient_y.at<Vec3f>(y,x) );
		}
	}	


	Vec3b color0 = Images[0].at<Vec3b>(0,0);
	SolveChannel( 0, color0[0], color_gradient_x, color_gradient_y , color_result , Images);
	Progress(0.5, kDoneSolveChannel1);
	SolveChannel( 1, color0[1], color_gradient_x, color_gradient_y , color_result, Images);
	Progress(0.8, kDoneSolveChannel2);
	SolveChannel( 2, color0[2], color_gradient_x, color_gradient_y , color_result, Images);
	Progress(1, kDoneSolveChannel3);

	result_ = color_result;
	if (preview)
	{
		imshow("color result", color_result);
		while (waitKey(27) != 27)
		{

		}
	}
}

void PhotoMontage::VisResultLabelMap( const cv::Mat & ResultLabel, int n_label )
{
	int width = ResultLabel.cols;
	int height = ResultLabel.rows;
	Mat color_result_map(height, width, CV_8UC3);
	std::vector<Vec3b> label_colors;
	for (int i = 0 ; i < n_label; i++)
	{
		label_colors.push_back(Vec3b(rand()%256,rand()%256,rand()%256));
	}
	//label_colors.push_back(Vec3b(255,0,0));
	//label_colors.push_back(Vec3b(0,255,0));
	//label_colors.push_back(Vec3b(0,0,255));
	//label_colors.push_back(Vec3b(255,255,0));
	//label_colors.push_back(Vec3b(0,255,255));


	for (int y = 0 ; y < height ; y++)
	{
		for (int x = 0 ; x < width ; x++)
		{
			color_result_map.at<Vec3b>(y,x) = label_colors[ResultLabel.at<uchar>(y,x)];
		}
	}

	imshow("result labels",color_result_map);
	waitKey(27);
}

void PhotoMontage::VisCompositeImage( const cv::Mat & ResultLabel, const std::vector<cv::Mat> & Images )
{
	int width = ResultLabel.cols;
	int height = ResultLabel.rows;
	Mat composite_image(height, width, CV_8UC3);

	for (int y = 0 ; y < height ; y++)
	{
		for (int x = 0 ; x < width ; x++)
		{
			composite_image.at<Vec3b>(y,x) = Images[ResultLabel.at<uchar>(y,x)].at<Vec3b>(y,x);
		}
	}

	imshow("composite image",composite_image);
	waitKey(27);
}

cv::flann::Index *  PhotoMontage::AddInertiaConstraint( const cv::Mat & Label )
{
	int height = Label.rows;
	int width = Label.cols;
	std::vector<Point2f> _data_vec;
	for (int y = 0 ; y < height; y++)
	{
		for (int x = 0 ; x < width; x++)
		{
			if (Label.at<char>(y,x) > 0)
			{
				_data_vec.push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));
			}
		}
	}

	_data.create(_data_vec.size(),2,CV_32FC1);	
	for (int i = 0 ; i < _data_vec.size(); i++)
	{
		_data.at<float>(i,0) = _data_vec[i].x;
		_data.at<float>(i,1) = _data_vec[i].y;
	}
	cv::flann::KDTreeIndexParams indexParams; 
	return new cv::flann::Index(_data, indexParams); 

	//std::vector<int> indices(1); 
	//std::vector<float> dists(1); 
	//Mat query(1,2,CV_32FC1);
	//query.at<float>(0,0) = 522;
	//query.at<float>(0,1) = 57;
	//kdtree->knnSearch(query, indices, dists, 1,cv::flann::SearchParams(64)); 
}

void PhotoMontage::Progress(double _0_1_progress, Stage stage)
{
	if (progress)
	{
		progress(_0_1_progress, stage);
	}
}

void PhotoMontage::SolveChannel( int channel_idx, int constraint, const cv::Mat &color_gradient_x, const cv::Mat &color_gradient_y , cv::Mat & output, const std::vector<cv::Mat>& Images)
{
	///请同学们填写这里的代码，这里就是实验中所说的单颜色通道的Gradient Fusion

#if 1

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
			Vec3f grads_x = color_gradient_x.at<Vec3f>(y, x);
			b(row_xy) = grads_x[channel_idx];

			// 第 2k + 1 行, 在矩阵里表示 v(x,y+1)-v(x,y)
			int row_xy1 = row_xy + 1;
			int col_xy1 = col_xy + width;
			NonZeroTerms[idx++] = Eigen::Triplet<double>(row_xy1, col_xy, -1); // -v(x, y)
			NonZeroTerms[idx++] = Eigen::Triplet<double>(row_xy1, col_xy1, 1); // v(x, y+1)
			Vec3f grads_y = color_gradient_y.at<Vec3f>(y, x);
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
	
	std::vector<double> init;
	if (fast_init_value)
	{
		init.resize(width * height);
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				init[y * width + x] = Images[result_label_.at<uchar>(y, x)].at<Vec3b>(y, x)[channel_idx];
			}
		}
	}

	printf("\nSolving...\n");
	auto mysolution = MyMatrix.conjugateGradient(myATb, 1e-10, iterations_, init);
	printf("Solved!\n");
	// vecadd(mysolution, 90, mysolution);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Vec3b& temp = output.at<Vec3b>(y, x);
			temp[channel_idx] = uchar(std::max(std::min(mysolution[y * width + x], 255.0), 0.0));
			//printf("%d,%d,  %f, %f\n",y,x, solution(y * width + x), ATb(y*width + x));
			//system("pause");
		}
	}
#endif
}
