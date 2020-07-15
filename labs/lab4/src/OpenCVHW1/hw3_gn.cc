#include "config.h"

#ifdef HW3_GN
#include "hw3_gn.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include <execution>
#include <map>
#include <fstream>
#include "utils.h"
#define NUM_FMT "%.2e\t"

void FPrint(double x)
{
	printf("%.2e\t", x);
}

void FPrint(const char* x)
{
	printf("%s", x);
}


template<int N>
void FPrint(const char x[N])
{
	FPrint((const char*)x);
}

void FPrint(const cv::Mat1d& x)
{
	for (int i = 0; i < x.rows; ++i)
	{
		for (int j = 0; j < x.cols; ++j)
		{
			FPrint(x.at<double>(i, j));
		}
		printf("\n");
	}
	printf("\n");
}


template<typename T>
void FPrints(T val)
{
	FPrint(val);
}

template<typename T, typename... Ts>
void FPrints(T&& val, Ts&&... vals)
{
	FPrints(std::forward<T>(val));
	FPrints(std::forward<Ts>(vals)...);
}



using D2 = double[4];

class Solver5728 : public GaussNewtonSolver
{
public:
	double solve(
		ResidualFunction* pf, // Ŀ�꺯��
		double* pX,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport* report = nullptr // �Ż��������
	) override
	{
		auto& f = *pf;
		int n = 0;

		int nX = f.nX();
		int nR = f.nR();

		cv::Mat1d R(nR, 1); // Residual
		cv::Mat1d J(nR, nX); // Jacobi
		cv::Mat1d JT(nX, nR); // transpose of Jacobi
		cv::Mat1d JT_J(nX, nX); // JT * J
		cv::Mat1d Dx(nX, 1); // \Delta x, direction of descent
		cv::Mat1d X(nX, 1, pX); // paramters to be optimized
		cv::Mat1d b(nX, 1); // bias

		using StopType = GaussNewtonReport::StopType;
		StopType stop_type = StopType::STOP_NO_CONVERGE;

		if (param.verbose)
		{
			printf("iter\tcost\t\tcost_change\tgradient\tstep\titer_time\ttotal_time\n");
		}

		constexpr auto inf = std::numeric_limits<double>::max();
		auto last_res = inf;
		Timer t;
		t.Start();
		double last_time = 0;
		while (n < param.max_iter)
		{
			f.eval(data(R), data(J), data(X));

			cv::transpose(J, JT);
			JT_J = JT * J;
			b = -JT * R;
			if (!cv::solve(JT_J, b, Dx))
			{
				stop_type = StopType::STOP_NUMERIC_FAILURE;
				if (param.verbose)
				{
					printf("Numerica Failure, cannot solve linear system\n");
					//FPrints("JT_T\n", JT_J, "J\n", J, "R\n", R, "b\n", b);
				}
				break;
			}
			
			auto max_res = maxabs(R);
			auto max_grad = maxabs(Dx);

			if (max_res < param.residual_tolerance ||
				max_grad < param.gradient_tolerance )
			{
				
				if (max_res < param.residual_tolerance)
				{
					stop_type = StopType::STOP_RESIDUAL_TOL;
				}
				else if (max_grad < param.gradient_tolerance)
				{
					stop_type = StopType::STOP_GRAD_TOL;
				}

				break;
				
			}
			if (n == param.max_iter)
			{
				stop_type = StopType::STOP_NO_CONVERGE;
				printf("Failed to converge\n");
				break;
			}

			// damped newton parameter, see: 
			double alpha = param.exact_line_search ?
				linear_search_algo(f, R, J, X, Dx, max_res) 
				: backtracking_algo(f, R, J, JT, X, Dx);
			X += alpha * Dx;

			if (alpha == 0)
			{
				stop_type = StopType::STOP_NO_CONVERGE;
				if (param.verbose) {
					printf("Search stopped at local minimum\n");
				}
				break;
			}

			if (param.verbose)
			{
				double total_time = t.elapse_ms();

				printf("%d\t", n);
				FPrints(max_res, last_res == inf? 0.0 : max_res - last_res, max_grad, alpha, 
					total_time - last_time, total_time);
				printf("\n");
				last_res = max_res;
				last_time = total_time;
			}

			++n;
		}

		memcpy(pX, X.data, nX * sizeof(double));

		if (report != nullptr)
		{
			report->n_iter = n;
			report->stop_type = stop_type;
		}
		

		return sum(R)[0];
	}

	static double linear_search_algo(ResidualFunction& f, cv::Mat1d& R, cv::Mat1d& J, const cv::Mat1d& X, const cv::Mat& Dx, double min)
	{
		cv::Mat1d mX = X.clone();
		double min_alpha = 0;
		int i = 1; // step
		double denominator = 1; // step size
		double alpha = 0; // result
		int n = 0; // iteration counter
		bool reached = false; // if reach a place smaller than original

		// If we completed 10 iterations, and at least one iteration
		// found a suitable step size alpha
		// Otherwise we will keep iterate for 1000 times waiting for the 
		// step that can minimize Residual
		while (n < 10 || (!reached && n < 1000))
		{
			// step forward, compute new residual
			alpha += 1 / denominator;
			mX = X + alpha * Dx;
			f.eval(data(R), data(J), data(mX));

			// if the next step makes residual lesser , that is good
			// we record the alpha(step size) as min_alpha
			if (double eps = maxabs(R); eps < min)
			{
				min = eps;
				min_alpha = alpha;
				reached = true;
			}
			// The reisudal is larger than min, that means
			// we may overstepped, therefore we withdraw the
			// step, shrink our step size by factor of 10
			else
			{
				alpha -= 1 / denominator;
				denominator *= 10;
				i = 0;
			}
			++n;
			++i;
		}
		return min_alpha;
	}

	static double backtracking_algo(ResidualFunction& f, const cv::Mat1d& R0, cv::Mat1d& J, cv::Mat1d& JT, 
		const cv::Mat1d& X, const cv::Mat& Dx)
	{
		// paramters
		constexpr double beta = 0.9;
		constexpr double gamma = 0.9;

		double JT_Dx_gamma = Dx.dot(Dx) * gamma;
		cv::Mat1d R = R0.clone();
		double alpha = 1;
		cv::Mat1d mX = X.clone();

		// check all elements in R is larger or equal to L
		auto one_greater = [](const double* L, const double* R, int count)
		{
			for (int i = 0; i < count; ++i)
			{
				if (*L > *R) {
					return true;
				}
				++L;
				++R;
			}

			return false;
		};

		// do we still have to loop
		auto loop = [&] {
			mX = X + alpha * Dx;
			f.eval(data(R), data(J), data(mX));
			const cv::Mat1d Rhs = (R0 + alpha * JT_Dx_gamma);
			assert(Rhs.size() == R.size());
			return one_greater(data(R0), data(Rhs), R0.rows);  //sum(R)[0] > sum(Rhs)[0]; // one_greater(data(R0), data(Rhs), R0.rows);  //sum(R)[0] > sum(Rhs)[0]; //one_greater(data(R0), data(Rhs));///sum(R)[0] < sum(Rhs)[0];
		};
		
		while (loop())
		{
			alpha *= beta;
		}
		return alpha;
	}

	static const double* data(const cv::Mat1d& x) {
		return (const double*)x.data;
	};

	static double* data(cv::Mat1d& x) {
		return (double*)x.data;
	};

	static double maxabs(const cv::Mat1d& x) {
		double mx, mi;
		cv::minMaxLoc(cv::abs(x), &mi, &mx);
		return mx;
	};

	
	static double max(const cv::Mat1d& x) {
		double mx, mi;
		cv::minMaxLoc(x, &mi, &mx);
		return mx;
	};

	static double normal_inf(const double* begin, int count)
	{
		return std::reduce(std::execution::par, begin, begin + count, std::numeric_limits<double>::max(), 
			[](double a, double b) {
			return std::max(std::abs(a), std::abs(b));
		});
	}

	// x = x + alpha * delta_x;
	static void step(double* x, double alpha, const double* delta_x, int count)
	{
		std::transform(std::execution::par, 
			x, x + count, // first1, last1
			delta_x, x,  // first2, dest
			[alpha](double mx, double m_Dx) {
			return mx + alpha * m_Dx;
		});
	}
};

#include "utils.h"
int benchmark()
{
	constexpr int cnt = 100000;
	using namespace cv;

	Mat1d m(cnt, 1);
	std::vector<double> x(cnt);

	m.forEach([&](double& pixel, const int* position) {
		double u = rand() / 37483.0;
		x[position[0]] = u;
		pixel = u;
	});
	Timer t;
	double a, c, b;
	t.Start();
	
	cv::minMaxLoc(cv::abs(m), &c, &a );
	t.End().Print("OpenCV");

	t.Start();
	b = Solver5728::normal_inf((double*)m.data, cnt);
	t.End().Print("STL");

	if (a != b)
	{
		printf("Result is wrong!!!");
	}
	return 0;
}

class LineResidual : public ResidualFunction
{
public:
	
	std::vector<double> mX;
	std::vector<double> mY;
	std::vector<double> param;
	double k;
	double b;
	LineResidual(double k, double b, int num_dots, bool perturb_x = false, bool perturb_y = false)
	{
		for (int i = 0; i < num_dots; ++i)
		{
			double x = i;
			if (perturb_x)
			{
				i += (rand() % 512 - 256) / 512.0 / 2.0;
			}
			double y = k * x + b;
			if (perturb_y)
			{
				y += (rand() % 512 - 256) / 512.0 / 2.0;
			}
			
			mX.emplace_back(x);
			mY.emplace_back(y);
		}
		//param.push_back(rand());
		//param.push_back(rand());
		param.push_back(1);
		param.push_back(1);
		this->k = k;
		this->b = b;
	}

	int nR() const override
	{
		return mX.size();
	}

	int nX() const override
	{
		return param.size();
	}

	void eval(double* R, double* J, double* X) override
	{
		double a = X[0];
		double b = X[1];


		auto mnX = nR();
		R[0] = 0;
		for (int i = 0; i < mnX; ++i)
		{
			double x = mX[i];
			double y = mY[i];

			J[i * 2 + 0] = x;
			J[i * 2 + 1] = 1;
			auto r = x * a + b - y;
			R[i] = r;
		}
	}
};

template<int n>
void RandInit(std::array<double, n>& arr)
{
	for (auto& r : arr)
	{
		r = rand() / 121763.0;
	}
}


void ExportToStream(std::ostream& os, const double& items)
{
	os << items << "  ";
}

void ExportToStream(std::ostream& os, const cv::Point2d& items)
{
	ExportToStream(os, items.x);
	ExportToStream(os, items.y);
	os << "\n";
}

void ExportToStream(std::ostream& os, const cv::Point3d& items)
{
	ExportToStream(os, items.x);
	ExportToStream(os, items.y);
	ExportToStream(os, items.z);
	os << "\n";
}

template<typename Iteratable>
void ExportToStream(std::ostream& os,  const Iteratable& items)
{
	for (auto& item : items)
	{
		ExportToStream(os, item);
	}
	os << "\n";
}


double get_noise(double scale) 
{
	return ((rand() % 512 - 256) / 512.0 * scale);
}

class CircleResidual : public ResidualFunction
{
public:
	std::array<double, 3> param;
	std::vector<cv::Point2d> points;
	double a, b, r;
	CircleResidual(double A, double B, double R)
	{
		a = A;
		b = B;
		r = R;

		for (int i = 0; i < 120; ++i)
		{
			double theta = i / 20.0;
			double x = a + r * cos(theta) + ( (rand() % 512 - 256) / 512.0 * 2);
			double y = b + r * sin(theta) + ((rand() % 512 - 256) / 512.0 * 2);
			points.emplace_back(x, y);
		}
		RandInit(param);
		// param = { a, b ,r };
	}

	int nR() const override
	{
		return points.size();
	}

	int nX() const override
	{
		return param.size();
	}

	void eval(double* R, double* J, double* X) override
	{
		double A = X[0];
		double B = X[1];
		double tR = X[2];

		int mnR = nR();
		for (int i = 0; i < mnR; ++i)
		{
			auto& p = points[i];
			double XA = p.x - A;
			double YB = p.y - B;

			R[i] = XA * XA + YB * YB - tR * tR;

			J[i * 3] = -2 * XA;
			J[i * 3 + 1] = -2 * YB;
			J[i * 3 + 2] = -2 * tR;
		}
	}
};

class EclipseResidual : public ResidualFunction
{
public:
	std::array<double, 3> param;
	std::vector<cv::Point3d> points;
	double mA, mB, mC;

	EclipseResidual(double A, double B, double C)
	{
		for (int i = 0; i < 20; ++i) {
			for (int j = 0; j < 20; ++j) {
				double t = i / 20.0;
				double p = j / 20.0;
				double x = A * sin(t) * cos(p) + get_noise(2);
				double y = B * sin(t) * sin(p) + get_noise(2);
				double z = C * cos(t) + get_noise(2);

				points.emplace_back(x, y, z);
			}
		}
		RandInit(param);

		mA = A;
		mB = B;
		mC = C;
	}

	EclipseResidual(std::string filename)
	{
		std::ifstream ifs(filename);
		if (!ifs.good()) {
			return;
		}

		while (!ifs.eof())
		{
			double x, y, z;
			ifs >> x >> y;
			if (ifs.eof()) {
				break;
			}
			ifs >> z;
			

			points.emplace_back(x, y, z);
		}
		RandInit(param);
		mA = mB = mC = std::numeric_limits<double>::quiet_NaN();
	}


	int nR() const override
	{
		return points.size();
	}

	int nX() const override
	{
		return param.size();
	}

	void eval(double* R, double* J, double* X) override
	{
		double A = X[0];
		double B = X[1];
		double C = X[2];

		auto mnX = nR();
		for (int i = 0; i < mnX; ++i)
		{
			auto& p = points[i];
			auto v = [](double U, double t) {
				return -2 * t * t / U / U / U;
			};
			auto e = [](double U, double t) {
				return t * t / U / U;
			};

			J[i * 3 + 0] = v(A, p.x);
			J[i * 3 + 1] = v(B, p.y);
			J[i * 3 + 2] = v(C, p.z);
			auto r = e(A, p.x) + e(B, p.y) + e(C, p.z) - 1;
			R[i] = r;
		}
	}

};

int main()
{
	// benchmark();
	LineResidual line(2.1, -3.1, 980, true, true);
	Solver5728 mysol;
	GaussNewtonParams param;
	param.verbose = true;
	param.max_iter = 150;
	param.exact_line_search = true;

	mysol.solve(&line, line.param.data(), param);
	printf("truth: k %lf, b %lf\n", line.k, line.b);
	printf("solved: k %lf, b %lf\n\n", line.param[0], line.param[1]);

	CircleResidual circle(8, 9, 3);
	mysol.solve(&circle, circle.param.data(), param);
	printf("circle truth: A %lf, B %lf, R %lf\n", circle.a, circle.b, circle.r);
	printf("circle solved: A %lf, B %lf, R %lf\n\n", circle.param[0], circle.param[1], circle.param[2]);
	std::ofstream ofs("circle_result.txt");
	ExportToStream(ofs, circle.param);
	ExportToStream(ofs, circle.points);
	ofs.close();

	// EclipseResidual eclipse(10, 12, 13);
	EclipseResidual eclipse("test/ellipse753.txt");
	// assert(eclipse.points.size() == 753);
	
	mysol.solve(&eclipse, eclipse.param.data(), param);
	printf("truth: A %lf, B %lf, C %lf\n", eclipse.mA, eclipse.mB, eclipse.mC);
	printf("solved: A %lf, B %lf, C %lf\n", eclipse.param[0], eclipse.param[1], eclipse.param[2]);
	ofs.open("ecclipse753_result.txt");
	ExportToStream(ofs, eclipse.param);
	ExportToStream(ofs, eclipse.points);
	ofs.close();
}

#endif // HW3_GN

