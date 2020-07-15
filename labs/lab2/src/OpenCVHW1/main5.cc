#include "config.h"

#include <opencv2/opencv.hpp>
#include <iostream>

#include "main5.h"
#include "utils.h"
#include <chrono>
#include <array>
#include <future>

#ifdef HW_5_1
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "BoxFilter", "<input-image> <output-image> <w> <h>", 4)) {
		return -1;
	}
	
	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}

	auto [w, h, float_w, float_h] = getWH(argv, 3, 4);

	Mat kernel(Size{ w, h }, CV_32F);
	kernel = 1.0f / float_w / float_h; // normalize

	filter2D(src, src, -1, kernel);
	if (!imwrite(argv[2], src)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}
	
	return 0;
}
#endif

#ifdef HW_5_2
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


#ifdef HW_5_3_NAVIVE
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "MedianFilterNaive", "<input-image> <output-image> <w> <h>", 4)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}

	auto [w, h, float_w, float_h] = getWH(argv, 3, 4);
	int mw = w, mh = h;
	std::cout << "Kernel size: (" << w << ", " << h << ")" << std::endl;
	timer.Start();

	const int mid = w * h / 2;
	std::vector<uchar> nums(w*h);

	Mat dst = src.clone();
	int imh = src.rows;

	const int base_h = h / 2;
	const int base_w = w / 2;
	const int channels = src.channels();

	Mat base;
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, BORDER_REFLECT);

	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<uchar*> ring(h, 2);
	for (int i = 0; i < h - 1; ++i) {
		ring.push(reinterpret_cast<uchar*>(base.ptr(i)));
	}

	// for each pixel
	for (int i = 0; i < src.rows; ++i) {
		ring.push(reinterpret_cast<uchar*>(base.ptr(i + h - 1)));
		uchar* ptr = reinterpret_cast<uchar*> (dst.ptr(i));
		for (int j = 0; j < src.cols; ++j) {
			for (int ch = 0; ch < channels; ++ch) {
				int n = 0;
				for (int u = 0; u < mh; ++u) {
					auto p = ring[u] + j * channels + ch;
					for (int v = 0; v < mw; ++v) {
						p += channels;
						nums[n++] = *p;
					}
				}
				std::sort(nums.begin(), nums.end());
				*ptr++ = nums[mid];
			}
			
		}
	}

	timer.End().Print();

	
	if (!imwrite(argv[2], dst)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif
int med(int a[], int c, int mid) {
	int sum = 0;
	int i = 0;
	mid /= 2;
	for (; sum < mid;) {
		sum += a[i];
		++i;
	}
	return --i;
}

#ifdef HW_5_3
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "MedianFilter", "<input-image> <output-image> <w> <h>", 4)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}

	auto [w, h, float_w, float_h] = getWH(argv, 3, 4);
	std::cout << "Kernel size: (" << w << ", " << h << ")" << std::endl;
	Timer timer;
	timer.Start();

	const int mid = w * h / 2;
	const int wh = w * h;
	std::vector<uchar> nums(wh);

	Mat dst(src.size(), src.type());
	int imh = src.rows;

	const int base_h = h / 2;
	const int base_w = w / 2;
	const int channels = src.channels();

	Mat base;
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, BORDER_REFLECT);
	enum {
		kBinA = 16,
		kBinB = 256,
		kNumA = kBinB / kBinA
	};

	std::array<int, kBinA>bin_a;
	std::array<int, kBinB>bin_b;

	uchar a_idx;
	uchar b_idx;
	uchar& median = b_idx;
	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<uchar*> ring(h, 2);

	for (int ch = 0; ch < channels; ++ch) {	
		ring.reset();
		for (int i = 0; i < h - 1; ++i) {
			ring.push(reinterpret_cast<uchar*>(base.ptr(i)) + ch);
		}
		// for each pixel
		for (int i = 0; i < src.rows; ++i) {
			ring.push(reinterpret_cast<uchar*>(base.ptr(i + h - 1)) + ch);
			uchar* ptr = reinterpret_cast<uchar*> (dst.ptr(i)+ch);

			memset(bin_a.data(), 0, kBinA * sizeof(int));
			memset(bin_b.data(), 0, kBinB * sizeof(int));

			for (int u = 0; u < h; ++u) {
				auto p = ring[u];
				for (int v = 0; v < w; ++v) {
					++bin_a[*p / kNumA];
					++bin_b[*p];
					p += channels;
				}
			}
			int before = 0;
			int before_a = 0;
			{
				for (a_idx = 0; before_a < mid; ++a_idx) {
					before_a += bin_a[a_idx];
				}
				--a_idx;
				before_a = before_a - bin_a[a_idx];
				int sum = before_a;
				for (b_idx = a_idx * kNumA; sum < mid; ) {
					sum += bin_b[b_idx];
					++b_idx;
					before += bin_b[b_idx];
				}
				//median = static_cast<uchar>(--b_idx);
				--b_idx;
				before -= bin_b[b_idx];
			}
			
			*ptr = median;
			ptr += channels;

			for (int j = 1; j < src.cols; ++j) {
				for (int u = 0; u < h; ++u) {
					{
						auto p = *(ring[u] + (j - 1) * channels);
						register auto dp = p / kNumA;
						--bin_a[dp];
						--bin_b[p];
						if (p < median) {
							if (dp == a_idx)
								--before;
							else --before_a;
						}
					}

					{
						auto xp = *(ring[u] + (j + w - 1) * channels);
						register auto dp = xp / kNumA;
						++bin_a[dp];
						++bin_b[xp];
						if (xp < median) {
							if (dp == a_idx)
								++before;
							else ++before_a;
						}
					}
					
				}

				int left = before + before_a;
				int right = left + bin_b[b_idx];

				if (left > mid) {
					while (before_a > mid)
					{
						before_a -= bin_a[--a_idx];
					}
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					--b_idx;
					before = sum - before_a - bin_b[b_idx];
				}
				else if (right < mid) {
					do {
						before_a += bin_a[a_idx++];
					} while (before_a < mid);
					before_a -= bin_a[--a_idx];
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					--b_idx;
					before = sum - before_a - bin_b[b_idx];
				}
				
				*ptr = median;
				ptr += channels;
			}
		}
	}

	timer.End().Print();

	timer.Start();
	cv::medianBlur(src, src, w);
	timer.End().Print("[OpenCV] used");

	if (!imwrite(argv[2], dst)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif


#ifdef HW_5_3_Stock
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "MedianFilter", "<input-image> <output-image> <w> <h>", 4)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}

	auto [w, h, float_w, float_h] = getWH(argv, 3, 4);
	std::cout << "Kernel size: (" << w << ", " << h << ")" << std::endl;
	Timer timer;
	timer.Start();

	const int mid = w * h / 2;
	const int wh = w * h;
	std::vector<uchar> nums(wh);

	Mat dst = src.clone();
	int imh = src.rows;

	const int base_h = h / 2;
	const int base_w = w / 2;
	const int channels = src.channels();

	Mat base;
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, BORDER_REFLECT);
	enum {
		kBinA = 16,
		kBinB = 256,
		kNumA = kBinB / kBinA
	};

	std::array<int, kBinA>bin_a;
	std::array<int, kBinB>bin_b;

	std::array<int, kBinA>last_bin_a;
	std::array<int, kBinB>last_bin_b;

	uchar a_idx;
	uchar b_idx;
	uchar& median = b_idx;
	// we keep a **ring** buffer to store [pointers to image rows]
	Ring<uchar*> ring(h, 2);

	int last_before;
	int last_before_a;
	uchar last_a_idx;
	uchar last_b_idx;

	for (int ch = 0; ch < channels; ++ch) {
		ring.reset();
		for (int i = 0; i < h; ++i) {
			ring.push(reinterpret_cast<uchar*>(base.ptr(i)) + ch);
		}

		memset(bin_a.data(), 0, kBinA * sizeof(int));
		memset(bin_b.data(), 0, kBinB * sizeof(int));

		for (int u = 0; u < h; ++u) {
			auto p = ring[u];
			for (int v = 0; v < w; ++v) {
				++bin_a[*p / kNumA];
				++bin_b[*p];
				p += channels;
			}
		}
		int before = 0;
		int before_a = 0;
		{
			for (a_idx = 0; before_a < mid; ++a_idx) {
				before_a += bin_a[a_idx];
			}
			--a_idx;
			before_a = before_a - bin_a[a_idx];
			int sum = before_a;
			for (b_idx = a_idx * kNumA; sum < mid; ) {
				sum += bin_b[b_idx];
				++b_idx;
				before += bin_b[b_idx];
			}
			--b_idx;
			before -= bin_b[b_idx];
		}

		// for each pixel
		for (int i = 0; i < src.rows; ++i) {
			uchar* ptr = reinterpret_cast<uchar*> (dst.ptr(i) + ch);

			last_bin_a = bin_a;
			last_bin_b = bin_b;
			last_before = before;
			last_before_a = before_a;
			last_a_idx = a_idx;
			last_b_idx = b_idx;

			*ptr = median;
			ptr += channels;

			for (int j = 1; j < src.cols; ++j) {
				for (int u = 0; u < h; ++u) {
					{
						auto p = *(ring[u] + (j - 1) * channels);
						--bin_a[p / kNumA];
						--bin_b[p];
						if (p < median) {
							if ((p / kNumA) == a_idx)
								--before;
							else --before_a;
						}
					}

					{
						auto xp = *(ring[u] + (j + w - 1) * channels);
						++bin_a[xp / kNumA];
						++bin_b[xp];
						if (xp < median) {
							if ((xp / kNumA) == a_idx)
								++before;
							else ++before_a;
						}
					}

				}

				int left = before + before_a;
				int right = left + bin_b[b_idx];

				if (left > mid) {
					while (before_a > mid)
					{
						before_a -= bin_a[--a_idx];
					}
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					--b_idx;
					before = sum - before_a - bin_b[b_idx];
				}
				else if (right < mid) {
					do {
						before_a += bin_a[a_idx++];
					} while (before_a < mid);
					before_a -= bin_a[--a_idx];
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					--b_idx;
					before = sum - before_a - bin_b[b_idx];
				}

				*ptr = median;
				ptr += channels;
			}

			//int c = med(bin_b.data(), 256, wh);
			//assert(median == c);
			before_a = last_before_a;
			before = last_before;
			a_idx = last_a_idx;
			b_idx = last_b_idx;
			if (i != src.rows-1) {
				using std::swap;
				swap(last_bin_a, bin_a);
				swap(last_bin_b, bin_b);

				auto nxt = base.ptr(i + h) + ch;

				for (int v = 0; v < w; ++v) {
					{
						auto p = *(ring[0] + v * channels);
						--bin_a[p / kNumA];
						--bin_b[p];
						if (p < median) {
							if ((p / kNumA) == a_idx)
								--before;
							else --before_a;
						}
					}

					{
						auto xp = *(nxt   + v * channels);
						++bin_a[xp / kNumA];
						++bin_b[xp];
						if (xp < median) {
							if ((xp / kNumA) == a_idx)
								++before;
							else ++before_a;
						}
					}
				}

				int left = before + before_a;
				int right = left + bin_b[b_idx];

				if (left > mid) {
					while (before_a > mid)
					{
						before_a -= bin_a[--a_idx];
					}
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					--b_idx;
					before = sum - before_a - bin_b[b_idx];
				}
				else if (right < mid) {
					do {
						before_a += bin_a[a_idx++];
					} while (before_a < mid);
					before_a -= bin_a[--a_idx];
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					--b_idx;
					before = sum - before_a - bin_b[b_idx];
				}
				//c = med(bin_b.data(), 256, wh);
				//assert(median == c);
				ring.push(reinterpret_cast<uchar*>(nxt));
			}
			
		}
	}

	timer.End().Print();

	timer.Start();
	cv::medianBlur(src, src, w);
	timer.End().Print("[OpenCV] used");

	if (!imwrite(argv[2], dst)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif


#ifdef HW_5_3_THREAD_POOL
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "MedianFilterMulti", "<input-image> <output-image> <w> <h>", 4)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}
	auto [w, h, float_w, float_h] = getWH(argv, 3, 4);
	std::cout << "Kernel size: (" << w << ", " << h << ")" << std::endl;

	Timer timer;
	timer.Start();

	const int mid = w * h / 2;
	const int wh = w * h;

	Mat dst(src.size(), src.type()); //= src.clone();

	const int base_h = h / 2;
	const int base_w = w / 2;
	const int channels = src.channels();

	Mat base;
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, BORDER_REFLECT);
	enum {
		kBinA = 16,
		kBinB = 256,
		kNumA = kBinB / kBinA,
		kNumThreads = 8
	};

	auto func = [&dst, &base, &src, channels, base_h, base_w, mid, w, h, wh](int from_row, int to_row) {
		using BinType = int;

		std::array<BinType, kBinA>bin_a;
		std::array<BinType, kBinB>bin_b;

		std::array<BinType, kBinA>last_bin_a;
		std::array<BinType, kBinB>last_bin_b;


		uchar a_idx;
		uchar b_idx;
		uchar& median = b_idx;

		int cols = src.cols;

		// we keep a **ring** buffer to store [pointers to image rows]
		Ring<uchar*> ring(h, 2);
		

		for (int ch = 0; ch < channels; ++ch) {
			ring.reset();
			memset(bin_a.data(), 0, kBinA * sizeof(BinType));
			memset(bin_b.data(), 0, kBinB * sizeof(BinType));
			for (int i = 0; i < h - 1; ++i) {
				auto p = reinterpret_cast<uchar*>(base.ptr(i + from_row)) + ch;
				// auto p = ring[u];
				ring.push(p);
				for (int v = 0; v < w; ++v) {
					++bin_a[*p / kNumA];
					++bin_b[*p];
					p += channels;
				}
			}
			// for each pixel
			for (int i = from_row; i < to_row; i ++) {
				uchar* ptr = reinterpret_cast<uchar*> (dst.ptr(i) + ch);

				{
					auto p = reinterpret_cast<uchar*>(base.ptr(i + h - 1)) + ch;
					ring.push(p);

					for (int v = 0; v < w; ++v) {
						++bin_a[*p / kNumA];
						++bin_b[*p];
						p += channels;
					}
				}

				last_bin_a = bin_a;
				last_bin_b = bin_b;
				//memcpy(last_bin_a, bin_a, sizeof(BinType)* kBinA);
				//memcpy(last_bin_b, bin_b, sizeof(BinType)* kBinB);

				int before = 0;
				int before_a = 0;
				{
					for (a_idx = 0; before_a < mid; ++a_idx) {
						before_a += bin_a[a_idx];
					}
					--a_idx;
					before_a = before_a - bin_a[a_idx];
					int sum = before_a;
					for (b_idx = a_idx * kNumA; sum < mid; ) {
						sum += bin_b[b_idx];
						++b_idx;
						before += bin_b[b_idx];
					}
					median = static_cast<uchar>(--b_idx);
					before -= bin_b[b_idx];
				}

				*ptr = median;
				ptr += channels;

				for (int j = 1; j < cols; ++j) {
					for (int u = 0; u < h; ++u) {
						{
							auto p = *(ring[u] + (j - 1) * channels);
							auto dp = p / kNumA;
							--bin_a[dp];
							--bin_b[p];
							if (p < median) {
								if (dp == a_idx)
									--before;
								else --before_a;
							}
						}

						{
							auto xp = *(ring[u] + (j + w - 1) * channels);
							auto dp = xp / kNumA;
							++bin_a[dp];
							++bin_b[xp];
							if (xp < median) {
								if (dp == a_idx)
									++before;
								else ++before_a;
							}
						}

					}

					int left = before + before_a;
					int right = left + bin_b[b_idx];

					if (left > mid) {
						while (before_a > mid)
						{
							before_a -= bin_a[--a_idx];
						}
						int sum = before_a;
						b_idx = a_idx * kNumA;
						while (sum < mid) {
							sum += bin_b[b_idx];
							++b_idx;
						}
						median = --b_idx;
						before = sum - before_a - bin_b[b_idx];
					}
					else if (right < mid) {
						do {
							before_a += bin_a[a_idx++];
						} while (before_a < mid);
						before_a -= bin_a[--a_idx];
						int sum = before_a;
						b_idx = a_idx * kNumA;
						while (sum < mid) {
							sum += bin_b[b_idx];
							++b_idx;
						}
						median = --b_idx;
						before = sum - before_a - bin_b[b_idx];
					}

					*ptr = median;
					ptr += channels;
				}

				using std::swap;
				swap(last_bin_a, bin_a);
				swap(last_bin_b, bin_b);

				auto p = ring[0];
				for (int v = 0; v < w; ++v) {
					--bin_a[*p / kNumA];
					--bin_b[*p];
					p += channels;
				}
			}
		}
	};

	std::vector<std::future<void>> vec;
	vec.reserve(kNumThreads);
	int per = src.rows / kNumThreads;
	int sum = 0;
	for (int t = 0; t < kNumThreads-1; ++t) {
		vec.push_back(std::async(func, sum, sum+per));
		sum += per;
	}
	vec.push_back(std::async(func, sum, src.rows));
	
	for (auto& f : vec) {
		f.wait();
	}

	timer.End().Print();


	Mat dst2;
	timer.Start();
	cv::medianBlur(src, dst2, w);
	timer.End().Print("[OpenCV] used");

	if (!imwrite(argv[2], dst)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif

#ifdef HW_5_3_MULTI_THREADS
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "MedianFilterMulti", "<input-image> <output-image> <w> <h>", 4)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}
	auto [w, h, float_w, float_h] = getWH(argv, 3, 4);
	std::cout << "Kernel size: (" << w << ", " << h << ")" << std::endl;

	Timer timer;
	timer.Start();

	const int mid = w * h / 2;
	const int wh = w * h;
	std::vector<uchar> nums(wh);

	Mat dst(src.size(), src.type()); //= src.clone();

	const int base_h = h / 2;
	const int base_w = w / 2;
	const int channels = src.channels();

	Mat base;
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, BORDER_REFLECT);
	enum {
		kBinA = 16,
		kBinB = 256,
		kNumA = kBinB / kBinA
	};

	auto func = [&dst, &base, &src, channels, base_h, base_w, mid, w, h, wh](int ch) {
		using BinType = int;
		
		std::array<BinType, kBinA>bin_a;
		std::array<BinType, kBinB>bin_b;

		std::array<BinType, kBinA>last_bin_a;
		std::array<BinType, kBinB>last_bin_b;
		

		uchar a_idx;
		uchar b_idx;
		uchar& median = b_idx;

		// we keep a **ring** buffer to store [pointers to image rows]
		Ring<uchar*> ring(h, 2);
		memset(bin_a.data(), 0, kBinA * sizeof(BinType));
		memset(bin_b.data(), 0, kBinB * sizeof(BinType));
		for (int i = 0; i < h - 1; ++i) {
			auto p = reinterpret_cast<uchar*>(base.ptr(i)) + ch;
			// auto p = ring[u];
			ring.push(p);
			for (int v = 0; v < w; ++v) {
				++bin_a[*p / kNumA];
				++bin_b[*p];
				p += channels;
			}
		}
		// for each pixel
		for (int i = 0; i < src.rows; ++i) {
			uchar* ptr = reinterpret_cast<uchar*> (dst.ptr(i) + ch);

			{
				auto p = reinterpret_cast<uchar*>(base.ptr(i + h - 1)) + ch;
				ring.push(p);

				for (int v = 0; v < w; ++v) {
					++bin_a[*p / kNumA];
					++bin_b[*p];
					p += channels;
				}
			}
			
			last_bin_a = bin_a;
			last_bin_b = bin_b;
			//memcpy(last_bin_a, bin_a, sizeof(BinType)* kBinA);
			//memcpy(last_bin_b, bin_b, sizeof(BinType)* kBinB);

			int before = 0;
			int before_a = 0;
			{
				for (a_idx = 0; before_a < mid; ++a_idx) {
					before_a += bin_a[a_idx];
				}
				--a_idx;
				before_a = before_a - bin_a[a_idx];
				int sum = before_a;
				for (b_idx = a_idx * kNumA; sum < mid; ) {
					sum += bin_b[b_idx];
					++b_idx;
					before += bin_b[b_idx];
				}
				median = static_cast<uchar>(--b_idx);
				before -= bin_b[b_idx];
			}

			*ptr = median;
			ptr += channels;

			for (int j = 1; j < src.cols; ++j) {
				for (int u = 0; u < h; ++u) {
					{
						auto p = *(ring[u] + (j - 1) * channels);
						auto dp = p / kNumA;
						--bin_a[dp];
						--bin_b[p];
						if (p < median) {
							if (dp == a_idx)
								--before;
							else --before_a;
						}
					}

					{
						auto xp = *(ring[u] + (j + w - 1) * channels);
						auto dp = xp / kNumA;
						++bin_a[dp];
						++bin_b[xp];
						if (xp < median) {
							if (dp == a_idx)
								++before;
							else ++before_a;
						}
					}

				}

				int left = before + before_a;
				int right = left + bin_b[b_idx];

				if (left > mid) {
					while (before_a > mid)
					{
						before_a -= bin_a[--a_idx];
					}
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					median = --b_idx;
					before = sum - before_a - bin_b[b_idx];
				}
				else if (right < mid) {
					do {
						before_a += bin_a[a_idx++];
					} while (before_a < mid);
					before_a -= bin_a[--a_idx];
					int sum = before_a;
					b_idx = a_idx * kNumA;
					while (sum < mid) {
						sum += bin_b[b_idx];
						++b_idx;
					}
					median = --b_idx;
					before = sum - before_a - bin_b[b_idx];
				}

				*ptr = median;
				ptr += channels;
			}

			using std::swap;
			swap(last_bin_a, bin_a);
			swap(last_bin_b, bin_b);

			auto p = ring[0];
			for (int v = 0; v < w; ++v) {
				--bin_a[*p / kNumA];
				--bin_b[*p];
				p += channels;
			}
		}
	};

	std::vector<std::future<void>> vec;
	vec.reserve(channels);
	for (int ch = 0; ch < channels; ++ch) {
		vec.push_back(std::async(func, int(ch)));
	}

	for (int ch = 0; ch < channels; ++ch) {
		vec[ch].wait();
	}

	timer.End().Print();

	
	Mat dst2;
	timer.Start();
	cv::medianBlur(src, dst2, w);
	timer.End().Print("[OpenCV] used");

	if (!imwrite(argv[2], dst)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif

#ifdef HW_5_4
int main(int argc, char** argv) {
	if (initArgs(argc, &argv, "BilateralFilter", "<input-image> <output-image> <sigma-s> <sigma-r> -w <width> -h <height>", 4)) {
		return -1;
	}

	using namespace cv;
	Mat src = imread(argv[1]);
	if (src.empty()) {
		std::cout << "Unable to open file: " << argv[1] << std::endl;
		return -1;
	}

	auto [_1, _2, sigma_s, sigma_r] = getWH(argv, 3, 4);
	GaussFunc<double> gauss_s(sigma_s), gauss_r(sigma_r);

	int w, h;
	w = h = floor(2 * sigma_s) + 1;
	

	for (int i = 0; i < argc-1; ++i) {
		if (argv[i] == std::string("-w")) {
			w = atoi(argv[i + 1]);
		}
		else if (argv[i] == std::string("-h")) {
			h = atoi(argv[i + 1]);
		}
	}
	int mw = w / 2, mh = h / 2;
	
	std::cout << "Kernel size: (" << w << ", " << h << ")" << std::endl;

	const int imh = src.rows;
	std::vector<double> dists(w*h);
	{
		int cnt = 0;
		
		for (int u = 0; u < h; ++u) {
			for (int v = 0; v < w; ++v) {
				int dw = u - mh;
				int dh = v - mw;
				int dist2 = dw * dw + dh * dh;

				dists[cnt++] = gauss_s.calc2(dist2);
			}
		}
	}
	
	const int base_h = h / 2;
	const int base_w = w / 2;
	const int channels = src.channels();

	Mat dst(src.size(), src.type());
	Mat base;
	copyMakeBorder(src, base, base_h, base_h, base_w, base_w, cv::BORDER_REFLECT);

	Ring<uchar*> ring(h, 2);
	for (int i = 0; i < h - 1; ++i) {
		ring.push(reinterpret_cast<uchar*>(base.ptr(i)));
	}

	// for each pixel
	for (int i = 0; i < src.rows; ++i) {
		ring.push(reinterpret_cast<uchar*>(base.ptr(i + h - 1)));
		uchar* ptr = reinterpret_cast<uchar*> (dst.ptr(i));

		// for each channel
		for (int j = 0; j < src.cols; ++j) {
			for (int ch = 0; ch < channels; ++ch) {
				int n = 0;
				double sum = 0;
				double norm = 0;
				auto center = *(ring[mh] + (j + mw) * channels + ch);
				for (int u = 0; u < h; ++u) {
					auto p = ring[u] + (j + 0) * channels + ch;
					for (int v = 0; v < w; ++v) {
						p += channels;
					
						int delta_I = *p - center;

						double tmp = 100 * dists[n++] * gauss_r.calc(delta_I);

						sum += tmp * *p;
						norm += tmp;
					}
				}
				*ptr++ = saturate_cast<uchar>(sum / norm);
			}
		}
		if (i % 20 == 0) {
			std::cout << "\rProgress: " << double(i) / imh * 100 << "%";
		}
	}

	std::cout << "\n";


	if (!imwrite(argv[2], dst)) {
		std::cout << "Failed to write to: " << argv[2] << std::endl;
	}

	return 0;
}
#endif


#ifdef HW_5_5

// rearranges the outputs of dft by moving the zero-frequency component to the center of the array.
void fftshift(const cv::Mat& src, cv::Mat& dst) {
	using namespace cv;

	dst.create(src.size(), src.type());
	int rows = src.rows, cols = src.cols;
	Rect roiTopBand, roiBottomBand, roiLeftBand, roiRightBand;
	if (rows % 2 == 0) {
		roiTopBand = Rect(0, 0, cols, rows / 2);
		roiBottomBand = Rect(0, rows / 2, cols, rows / 2);
	}
	else {
		roiTopBand = Rect(0, 0, cols, rows / 2 + 1);
		roiBottomBand = Rect(0, rows / 2 + 1, cols, rows / 2);
	}
	if (cols % 2 == 0) {
		roiLeftBand = Rect(0, 0, cols / 2, rows);
		roiRightBand = Rect(cols / 2, 0, cols / 2, rows);
	}
	else {
		roiLeftBand = Rect(0, 0, cols / 2 + 1, rows);
		roiRightBand = Rect(cols / 2 + 1, 0, cols / 2, rows);
	}
	Mat srcTopBand = src(roiTopBand);
	Mat dstTopBand = dst(roiTopBand);
	Mat srcBottomBand = src(roiBottomBand);
	Mat dstBottomBand = dst(roiBottomBand);
	Mat srcLeftBand = src(roiLeftBand);
	Mat dstLeftBand = dst(roiLeftBand);
	Mat srcRightBand = src(roiRightBand);
	Mat dstRightBand = dst(roiRightBand);
	flip(srcTopBand, dstTopBand, 0);
	flip(srcBottomBand, dstBottomBand, 0);
	flip(dst, dst, 0);
	flip(srcLeftBand, dstLeftBand, 1);
	flip(srcRightBand, dstRightBand, 1);
	flip(dst, dst, 1);
}

cv::Mat freqImg(cv::Mat& src) {
	using namespace cv;
	Mat dst(src.size(), CV_32FC2);
	dft(src, dst, DFT_COMPLEX_OUTPUT);
	fftshift(dst, dst);

	Mat mag;

	std::vector<Mat> K;
	split(dst, K);
	pow(K[0], 2, K[0]);
	pow(K[1], 2, K[1]);
	mag = K[0] + K[1];

	Mat logMag;
	log(mag + 1, logMag);
	normalize(logMag, logMag, 1.0, 0.0, NORM_MINMAX);
	return logMag;
}

int main(int argc, char** argv)
{
	using namespace cv;

	Mat src(512, 512, CV_32FC1);
	src = 0;
	src(Rect(256 - 10, 256 - 30, 20, 60)) = 1.0;

	Mat logMag = freqImg(src);
	imshow("FFT", logMag);
	imwrite("results/FFT.png", logMag*255);

	Mat rot;
	rotate(src, rot, ROTATE_90_CLOCKWISE);
	logMag = freqImg(rot);
	imshow("FFT ROTATE_90_CLOCKWISE", logMag);
	imwrite("results/FFT ROTATE_90_CLOCKWISE.png", logMag*255);

	Mat trans(512, 512, CV_32FC1);
	int num_trans = 10;
	trans = 0;
	trans(Rect(256 - 10 + num_trans, 256 - 30, 20, 60)) = 1.0;
	logMag = freqImg(trans);
	imshow("FFT Translate", logMag);
	imwrite("results/FFT Translate.png", logMag*255);

	Mat enlarge(512, 512, CV_32FC1);
	int num_enls = 2;
	enlarge = 0;
	enlarge(Rect(256 - 10*num_enls, (256 - num_enls*3*10), 20* num_enls, 60* num_enls)) = 1.0;
	logMag = freqImg(enlarge);
	imshow("FFT Enlarged", logMag);
	imwrite("results/FFT Enlarged.png", logMag*255);

	waitKey(0);
	//imwrite(argv[2], logMag * 200);
}

#endif