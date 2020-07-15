#pragma once

#include "sparse-matrix.h"
#include <Eigen/Sparse>

SparseMatrix<double> ConvertFromEigen(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);

template<typename Q, typename T>
bool CheckEqual(Q& mat, std::vector<std::vector<T>>& v, Eigen::Triplet<T>* t = nullptr) {
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			if (mat.coeff(i, j) != v[i][j]) {
				if (t) {
					*t = Eigen::Triplet<T>(i, j, v[i][j]);
				}

				return false;
			}
		}
	}
	return true;
}

template<typename Q, typename T>
bool CheckEqual(Q& mat, Eigen::SparseMatrix<T, Eigen::RowMajor>& v, Eigen::Triplet<T>* t = nullptr) {
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			if (mat.coeff(i, j) != v.coeff(i, j)) {
				if (t) {
					*t = Eigen::Triplet<T>(i, j, v.coeff(i, j));
				}

				return false;
			}
		}
	}
	return true;
}

template<typename T>
bool CheckEqual(Eigen::VectorXd& vx, std::vector<T>& v) {
	for (int i = 0; i < v.size(); ++i)
	{
		if (vx[i] != v[i])
		{
			return false;
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
void PrettyPrint(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat) {
	std::cout << "Matrix (" << mat.rows() << " * " << mat.cols() << ")" << std::endl;
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			printf("%.3f  ", mat.coeff(i, j));
		}
		std::cout << std::endl;
	}
}

template<typename T>
void PrettyPrint(const std::vector<T>& mat) {
	std::cout << "[std::vector] (" << mat.size() << ")" << std::endl;
	for (auto& m : mat) {
		std::cout << m << "  ";
	}
	std::cout << std::endl;
}

inline void PrettyPrint(Eigen::VectorXd v)
{
	std::cout << "[Eigen::VectorXd] (" << v.rows() << ")" << std::endl;
	for (int i = 0; i < v.rows(); ++i)
	{
		std::cout << v.coeff(i) << "  ";
	}
	std::cout << std::endl;
}

// return 0: test success
// otherwise: test failed
int RunTest();