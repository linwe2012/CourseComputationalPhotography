#pragma once
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#ifdef _DEBUG
#define ASSERT(what) assert(what)
#else
#define ASSERT(what) (void) 0;
#endif

void overlayImage(cv::Mat& src, cv::Mat& overlay, const cv::Point& location);

// if area is emty, then use target's size
// if area overflows, will fallback to target's size
// returns a tuple with 4 ints
inline auto normalizeRect(const cv::Mat& target, const cv::Rect& area) {
	int tlx = 0;
	int tly = 0;
	int brx = target.size().width;
	int bry = target.size().height;

	if (area.tl().x > 0) {
		tlx = area.tl().x;
	}

	if (area.tl().y > 0) {
		tly = area.tl().y;
	}

	{
		int x = area.br().x;
		if (x > tlx && x < brx) {
			brx = x;
		}
	}
	
	{
		int y = area.br().y;
		if (y > tly && y < bry) {
			bry = y;
		}
	}

	return std::make_tuple( tlx, tly, brx, bry );
}

// helper class for manipulating pixels
struct Pixel3i {
	Pixel3i() :
		x(0), y(0), z(0) {}

	void operator /=(int div) {
		x /= div;
		y /= div;
		z /= div;
	}
	union
	{
		struct {
			int x;
			int y;
			int z;
		};
		struct {
			int r;
			int g;
			int b;
		};
	};

	void exp_avg(double cur, const Pixel3i& r) {
		double hist = 1 - cur;
		x = static_cast<int>(cur * x + r.x * hist);
		y = static_cast<int>(cur * y + r.y * hist);
		z = static_cast<int>(cur * z + r.z * hist);
	}

	void from_ptr(uchar* uc) {
		x = *uc++;
		y = *uc++;
		z = *uc;
	}

	void sub_prt(uchar* uc) {
		x -= *uc++;
		y -= *uc++;
		z -= *uc;
		x = x < 0 ? 0 : x;
		y = y < 0 ? 0 : y;
		z = z < 0 ? 0 : z;
	}
};

inline void centralGradient(cv::Mat& canvas, int r, int g, int b) {
	auto div = canvas.cols / 2 + canvas.rows / 2;
	for (int i = 0; i < canvas.rows; ++i) {
		auto ptr = canvas.ptr(i);
		for (int j = 0; j < canvas.cols; ++j) {
			double dist = sqrt(pow(i - canvas.rows / 2, 2) + pow(j - canvas.cols / 2, 2));
			auto decay = exp(-dist / div);

			*ptr++ = static_cast<uchar>(r * decay);
			*ptr++ = static_cast<uchar>(g * decay);
			*ptr++ = static_cast<uchar>(b * decay );
		}
	}
}


template<typename T>
class Ring {
	int sz_ = 0;
	T* beg_ = nullptr;
	T* end_ = nullptr;
	std::vector<T> buf_;
	int N = 0;
	int cap = 0;
public:

	void reset() {
		beg_ = end_ = buf_.data();
		sz_ = 0;
	}

	// static_assert(N >= 1, );

	Ring(int _N, int inflate_ratio = 2) {
		assert(_N >= 1 && " Ring buffer size must be psoitive");
		N = _N;
		cap = N * inflate_ratio;
		buf_.resize(cap);
		beg_ = buf_.data();
		end_ = beg_;
	}


	void push(const T& val) {
		++end_;
		if (end_ == data() + cap) {
			memcpy(data(), beg_ + 1, (N - 1)*sizeof(T));
			beg_ = data();
			end_ = beg_ + N;
			*(end_ - 1) = val;
			return;
		}
		if (sz_ == N) {
			++beg_;
		}
		else {
			++sz_;
		}

		*(end_ - 1) = val;
	}


	T* begin() {
		return beg_;
	}

	T* end() {
		return end_;
	}

	T* data() {
		return buf_.data();
	}



	T& operator[](int idx) {
		ASSERT(idx < sz_);
		return *(beg_ + idx);
	}


};



namespace Math {
	constexpr double PI = 3.14159265358979323846;
	constexpr double PI_2 = PI / 2.0;

	namespace detail {
		template<typename T>
		constexpr T simple_pow(T base, int p) {
			T res = 1;
			for (int i = 0; i < p; ++i) {
				res *= base;
			}
			return res;
		}

		constexpr double taylor_term(double x, int i) {
			return 1.0 / i * simple_pow(x, i);
		}
	}

	template<typename T>
	constexpr T abs(T x) {
		if (x >= 0) {
			return x;
		}
		return -x;
	}
	
	
	template<typename T>
	constexpr T atan(T x) {
		if (x < 0) {
			return -atan(-x);
		}
		if (x > 1) {
			return PI_2 - T(atan(1.0 / x));
		}
		T res = 0;
		for (int i = 0; i < 40; ++i) {
			if (i % 2) {
				res -= T(detail::taylor_term(x, 2 * i + 1));
			}
			else {
				res += T(detail::taylor_term(x, 2 * i + 1));
			}
		}
		return res;
	}

	// compile time sqrt
	template<typename T>
	constexpr T sqrt(T m, double accuracy = 1e-10) {
		double x = 0.5;
		double fx = x * x - m;
		while (abs(fx) > accuracy)
		{
			x = x - fx / (2 * x);
			fx = abs(x * x - m);
		}
		return static_cast<T>(x);
	}

	template <typename T> 
	constexpr int sign(T val) {
		return (T(0) < val) - (val < T(0));
	}

	static_assert(sign(10) == 1, "Sign");
	static_assert(sign(-5) == -1, "Sign");
	static_assert(sign(0) == 0, "Sign");
}

template<typename T>
struct GaussFunc {
	constexpr GaussFunc(T sigma) {
		constexpr T sqrt_2pi = static_cast<T>(Math::sqrt(2.0 * Math::PI));
		a = 1.0f / (sigma * sqrt_2pi);
		b = 2 * sigma * sigma;
		b = -1 / b;
	}

	T calc(T x) {
		return a * exp(x * x * b);
	}

	T calc2(T x2) {
		return a * exp(x2 * b);
	}

	T a;
	T b;
};


class Timer {
	std::chrono::steady_clock::time_point begin_;
	std::chrono::steady_clock::time_point end_;
public:
	void Start() {
		begin_ = std::chrono::steady_clock::now();
	}

	Timer& End() {
		end_ = std::chrono::steady_clock::now();
		return *this;
	}

	Timer& Print(std::string beg = "\rTime used") {
		auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count();
		std::cout << beg  << ": " << elapse << "ms       " << std::endl;
		return *this;
	}
};

enum struct WheelDirections {
	UpDown,
	LeftRight,
	TopRight, // To Bottom Left
	TopLeft, // To Bottom Right
};

inline WheelDirections estimateWheelDirection(double x, double y) {
	float abs_x = static_cast<float>(std::abs(x));

	// prevent arithmetic overflow
	if (abs_x < 1e-6) {
		return WheelDirections::UpDown;
	}

	double t = y / x;

	constexpr double pi_8 = Math::PI / 8.0;
	constexpr double atan_pi_8 = Math::atan(pi_8);

	constexpr double pi3_8 = Math::PI / 8.0 * 3.0;
	constexpr double atan_pi3_8 = Math::atan(pi3_8);

	if (t <= -atan_pi3_8 || t >= atan_pi3_8) {
		return WheelDirections::UpDown;
	}

	if (t >= atan_pi_8 && t <= atan_pi3_8) {
		return WheelDirections::TopRight;
	}

	if (t >= -atan_pi_8 && t <= atan_pi_8) {
		return WheelDirections::LeftRight;
	}

	return WheelDirections::TopLeft;
}



template<typename T>
void ConvT(const cv::Mat& src, cv::Mat& dst, cv::Mat& kernel, int border = cv::BORDER_REFLECT) {
	using namespace cv;

	const int base_h = kernel.rows / 2;
	const int base_w = kernel.cols / 2;
	const int channels = src.channels();

	if (dst.size() != src.size()) {
		resize(src, dst, src.size());
	}

	dst.convertTo(dst, CV_MAKE_TYPE(DataType<T>::type, channels));

	Mat base;
	// we are going to copy whole anyway, so 
	// copying with border won't have much perf penalties
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, border);

	T max = T(0);
	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<uchar*> ring(kernel.rows, 2);
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
		T* ptr = reinterpret_cast<T*> (dst.ptr(i));
		for (int j = 0; j < src.cols; ++j) {

			// for each channels
			for (int ch = 0; ch < channels; ++ch) {
				double tmp = 0;
				int kern_idx = 0;

				// for each elem in kernel
				for (int u = 0; u < kernel.rows; ++u) {
					for (int v = 0; v < kernel.cols; ++v) {
						auto p = ring[u] + (j + v) * channels + ch;
						tmp += static_cast<double>(*p)* vkern[kern_idx];
						++kern_idx;
					}
				}

				*ptr++ = saturate_cast<T>(tmp);
			}
		}
	}
}

template<typename FromT, typename T>
void ConvT(const cv::Mat& src, cv::Mat& dst, cv::Mat& kernel, int border = cv::BORDER_REFLECT) {
	using namespace cv;

	const int base_h = kernel.rows / 2;
	const int base_w = kernel.cols / 2;
	const int channels = src.channels();

	if (dst.size() != src.size()) {
		resize(src, dst, src.size());
	}

	dst.convertTo(dst, CV_MAKE_TYPE(DataType<T>::type, channels));

	Mat base;
	// we are going to copy whole anyway, so 
	// copying with border won't have much perf penalties
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, border);

	T max = T(0);
	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<FromT*> ring(kernel.rows, 10);
	for (int i = 0; i < kernel.rows - 1; ++i) {
		ring.push(reinterpret_cast<FromT*>(base.ptr(i)));
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
		ring.push(reinterpret_cast<FromT*>(base.ptr(i + kernel.rows - 1)));
		T* ptr = reinterpret_cast<T*> (dst.ptr(i));
		for (int j = 0; j < src.cols; ++j) {

			// for each channels
			for (int ch = 0; ch < channels; ++ch) {
				double tmp = 0;
				int kern_idx = 0;

				// for each elem in kernel
				for (int u = 0; u < kernel.rows; ++u) {
					for (int v = 0; v < kernel.cols; ++v) {
						auto p = ring[u] + (j + v) * channels + ch;
						tmp += static_cast<double>(*p)* vkern[kern_idx];
						++kern_idx;
					}
				}

				*ptr++ = saturate_cast<T>(tmp);
			}
		}
	}
}


template<typename FromT, typename T>
void SlidingWindowT(const cv::Mat& src, cv::Mat& dst, cv::Size window_size, std::function<void(Ring<FromT*>&,int,int,T*)> func, int border = cv::BORDER_REFLECT) {
	using namespace cv;

	const int base_h = window_size.height / 2;
	const int base_w = window_size.width  / 2;
	const int channels = src.channels();

	if (dst.size() != src.size()) {
		resize(src, dst, src.size());
	}

	dst.convertTo(dst, CV_MAKE_TYPE(DataType<T>::type, channels));

	Mat base;
	// we are going to copy whole anyway, so 
	// copying with border won't have much perf penalties
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, border);

	T max = T(0);
	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<FromT*> ring(window_size.height, 2);
	for (int i = 0; i < window_size.height - 1; ++i) {
		ring.push(reinterpret_cast<FromT*>(base.ptr(i)));
	}

	// for each pixel
	for (int i = 0; i < src.rows; ++i) {
		ring.push(reinterpret_cast<FromT*>(base.ptr(i + window_size.height - 1)));
		T* ptr = reinterpret_cast<T*> (dst.ptr(i));
#pragma loop(hint_parallel(8))
		for (int j = 0; j < src.cols; ++j) {
			func(ring, i, j, ptr);
			ptr += channels;
		}
	}
}

void rgb2yuv(const cv::Mat& src, cv::Mat& dst);

inline cv::Mat grayscale(cv::Mat original) {
	using namespace cv;
	Mat mats[3];
	Mat yuv;
	rgb2yuv(original, yuv);
	split(yuv, mats);
	return mats[2];
}

inline auto getMaxMin(const cv::Mat& m) {
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;

	cv::minMaxLoc(m, &minVal, &maxVal, &minLoc, &maxLoc);

	return std::make_tuple(maxVal, minVal);
}

cv::Mat GenGaussKernel(cv::Size size, float sigma);

void nonMaxSupression(const cv::Mat& x, const cv::Mat& y, const cv::Mat& sum, cv::Mat& out);
