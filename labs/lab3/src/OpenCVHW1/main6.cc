#include "config.h"

// 1>C:\Users\leon\source\repos\OpenCVHW1\packages\Eigen.3.3.3\build\native\include\Eigen\src\Core\functors\StlFunctors.h(87,1): error C4996: 'std::unary_negate<_Fn>': 
// warning STL4008: std::not1(), std::not2(), std::unary_negate, and std::binary_negate are deprecated in C++17. They are superseded by std::not_fn(). You can define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to acknowledge that you have received this warning.
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#ifdef HW_6
#include <iostream>

#include "utils.h"
#include "sparse-matrix.h"
#include "cxxopts.hpp"

#include <Eigen/SparseCore>



template<typename T>
bool CheckEqual(SparseMatrix<T> mat, std::vector<std::vector<T>> v, Eigen::Triplet<T>* t = nullptr) {
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			if (mat.at(i, j) != v[i][j]) {
				if (t) {
					*t = Eigen::Triplet<T>(i, j, v[i][j]);
				}
				
				return false;
			}
		}
	}
	return true;
}

template<typename T>
void PrettyPrint(const SparseMatrix<T>& mat) {
	std::cout << "Matrix (" << mat.rows() << " * " << mat.cols() << ")" << std::endl;
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			std::cout << mat.at(i, j) << "  ";
		}
		std::cout << std::endl;
	}
}

template<typename T>
void PrettyPrint(const std::vector<std::vector<T>>& mat) {
	std::cout << "[std::vector] Matrix (" << mat.size() << " * " << mat[0].size() << ")" << std::endl;
	for (auto& m : mat) {
		for (auto& x : m) {
			std::cout << x  << "  ";
		}
		std::cout << std::endl;
	}
}

template<typename T>
void PrettyPrint(const std::vector<T>& mat) {
	std::cout << "[std::vector] (" << mat.size() <<  ")" << std::endl;
	for (auto& m : mat) {
		std::cout << m << "  ";
	}
	std::cout << std::endl;
}

int test();

int main(int argc, char** argv) {
	cxxopts::Options options("lab3", "Sparse Matrix & Iterative Linear System Solving");
	options.add_options()
		("s,size",    "size of the benchmarking matrix", cxxopts::value<int>()->default_value("1000"))
		("d,density", "density percentage of non zero elements in benchmarking matrix", cxxopts::value<int>()->default_value("80"))
		("m,modify", "modify percentage in benchmarking matrix", cxxopts::value<int>()->default_value("20"))
		("b,benchmark-only", "used for python script for benchmarking", cxxopts::value<bool>()->default_value("false"))
		("h,help", "Print usage")
		;
	options.allow_unrecognised_options();

	auto result = options.parse(argc, argv);
	if (result.count("help"))
	{
		std::cout << options.help() << std::endl;
		exit(0);
	}

	bool only_benchmark = result["benchmark-only"].as<bool>();
	if (!only_benchmark) {
		test();
	}

	// benchmark
	int size = result["size"].as<int>();
	int density = result["density"].as<int>(); // % of elements are non zero
	int modify_rate = result["modify"].as<int>();


	Eigen::SparseMatrix<int> esp(size, size);
	SparseMatrix<int> sp;
	
	std::vector<std::vector<int>> mat(size, std::vector<int>(size, 0));
	std::vector<Eigen::Triplet<int>> trip;
	std::vector<int> rows;
	std::vector<int> cols;
	std::vector<int> vals;

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (rand() % 100 < density) {
				int val = rand() % 10000;
				trip.push_back({ i, j, val });
				vals.push_back(val);
				rows.push_back(i);
				cols.push_back(j);
				mat[i][j] = val;
			}
		}
	}

	if (rows.back() != size - 1 || cols.back() != size - 1) {
		rows.push_back(size - 1);
		cols.push_back(size - 1);
		vals.push_back(0);
	}

	Timer timer;

	timer.Start();
	esp.setFromTriplets(trip.begin(), trip.end());
	timer.End().Print("[Initialize] Eigen");
	timer.Start();
	sp.initializeFromVector(rows, std::move(cols), std::move(vals));
	timer.End().Print("[Initialize] Mine");

	if (!CheckEqual(sp, mat)) {
		std::cout << "[!!Error] My implementation is wrong" << std::endl;
	}

	// prevent compiler optimize away access call
	volatile int c = 0;

	//esp.makeCompressed();

	timer.Start();
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			c ^= esp.coeff(i, j);
		}
	}
	timer.End().Print("[Access all elements] Eigen");

	timer.Start();
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			c ^= sp.at(i, j);
		}
	}
	timer.End().Print("[Access all elements] Mine");

	
	std::vector<Eigen::Triplet<int>> mods;
	
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (rand() % 100 < modify_rate) {
				auto val = rand() % 10000;
				mods.push_back({ i, j, 0 });
				mat[i][j] = 0;
			}
		}
	}

	timer.Start();
	for (auto& m : mods) {
		esp.coeffRef(m.row(), m.col()) = m.value();
	}
	timer.End().Print("[Modify elements] Eigen");

	timer.Start();
	for (auto& m : mods) {
		sp.insert(m.value(), m.row(), m.col());
	}
	timer.End().Print("[Modify elements] Mine");

	
	if (!CheckEqual(sp, mat)) {
		std::cout << "[!!Error] My implementation is wrong" << std::endl;
	}

	return 0;
}

int test() {
	SparseMatrix<int> spi;
	std::vector<std::vector<int>> mat = {
		{1, 0, 0, 1, 0},
		{0, 0, 0, 0, 0},
		{8, 0, 1, 0, 0},
	};

	std::vector<int> vals = { 1, 1, 0, 8, 1 };
	std::vector<int> cols = { 0, 3, 4, 0, 2 };
	std::vector<int> rows = { 0, 0, 0, 2, 2 };
	spi.initializeFromVector(rows, std::move(cols), std::move(vals));

	auto prt = [&](std::string what) {
		std::cout << what << std::endl;
		PrettyPrint(spi);
		PrettyPrint(mat);
		std::cout << std::endl;
	};

	prt("Initial stage");
	if (!CheckEqual(spi, mat)) {
		throw;
	}

	
	auto modify = [&](int x, int row, int col, std::string label) {
		spi.insert(x, row, col);
		mat[row][col] = x;
		prt(label);
		if (!CheckEqual(spi, mat)) {	
			throw;
		}
	};

	modify(0, 1, 0, "Test1 - make zero val 0");
	modify(0, 0, 0, "Test2 - make non zero val 0");
	modify(1, 2, 2, "Test3 - the matrix is not modified");
	modify(8, 0, 0, "Test4 - make non zero val on row with extra space left");
	modify(9, 1, 1, "Test5 - mak non zero val on row with NO extra space left");

	std::vector<double> v1 = { 1.0, 2.0, 3.0, 10.0 };
	std::vector<double> v2 = { 2.0, 1.0, 3.0, 8.0 };
	int c = manhattonDist(v1, v2);


	SparseMatrix<int> sp2;
	sp2.initialize(4, 4, {
		10, -1,  2,  0,
		-1, 11, -1,  3,
		 2, -1, 10, -1,
		 0,  3, -1,  8
		});
	PrettyPrint(sp2);
	std::vector<double> b = { 6, 25, -11, 15 };
	auto vec = sp2.gaussSeidel(b);
	std::cout << "By Guass-Seidel ";
	PrettyPrint(vec);

	vec = sp2.conjugateGradient(b);
	std::cout << "By Conjugate Gradient ";
	PrettyPrint(vec);

	return 0;
}
#endif // HW_6