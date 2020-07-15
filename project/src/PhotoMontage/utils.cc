#include <iostream>
#include "utils.h"


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


int RunTest()
{
	constexpr int sizex1 = 40;
	constexpr int sizey1 = 40;
	constexpr int num_non_zeros = 112;
	std::vector<std::vector<double>> v(sizex1, std::vector<double>(sizey1, 0));
	Eigen::SparseMatrix<double, Eigen::RowMajor> m(sizex1, sizey1);

	std::vector<double> b(sizey1, 0);
	Eigen::VectorXd  xb(sizey1);

	int failed = 0;

	srand(1);

	for (int i = 0; i < num_non_zeros; ++i)
	{
		int x = rand() % sizex1;
		int y = rand() % sizey1;
		double val = rand() / 32784.0;
		
		m.coeffRef(y, x) = val;
	}
	m = m.transpose() * m;


	for (int i = 0; i < sizex1; ++i)
	{
		for (int j = 0; j < sizey1; ++j)
		{
			v[i][j] = m.coeff(i, j);
		}
	}

	{
		int i = 0;
		for (auto& x : b)
		{
			x = rand() / 32784.0;
			xb[i] = x;
			++i;
		}
	}
	
	//PrettyPrint(m);
	m.coeff(5, 5);
	auto sp = ConvertFromEigen(m);

	std::vector<double> aa(b.size());
	sp.applyToVector(b, aa);
	Eigen::VectorXd qq = m * xb;

	if (!CheckEqual(qq, aa))
	{
		std::cout << "[!!ERROR]: My implementation of martix * vector is wrong!!" << std::endl;
	}

	std::vector<double> va = { 1, 2, 3, 4 };
	std::vector<double> vb = va;
	double mma = veclen2(va);
	vecadd(va, vb, 0.1, va);
	
	
	//PrettyPrint(sp);

	if (!CheckEqual(sp, v))
	{
		std::cout << "[!!ERROR]: My implementation of sparse matrix is wrong!!" << std::endl;
		failed++;
		PrettyPrint(m);
		PrettyPrint(sp);
	}

	if (!CheckEqual(m, v))
	{
		std::cout << "[!!ERROR]: My implementation of checking algo is wrong!!" << std::endl;
		failed++;
	}

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> CGSolver(m);
	Eigen::VectorXd eigen_res = CGSolver.solve(xb);
	auto my_res = sp.conjugateGradientEigen(b);
	if (!CheckEqual(eigen_res, my_res))
	{
		std::cout << "[!!ERROR]: My implementation of conjugate grad is wrong!!" << std::endl;
		failed++;
	}
	std::vector<double> resb(b.size(), 0);
	sp.applyToVector(my_res, resb);
	Eigen::VectorXd ss = m * eigen_res;

	using VectorType = Eigen::Matrix<double, Eigen::Dynamic, 1> ;
	constexpr int vtsz = 1000;
	VectorType vt(vtsz);
	VectorType vt1(vtsz);
	std::vector<double> vvt(vtsz);
	std::vector<double> vvt1(vtsz);
	for (int i = 0; i < vtsz; ++i)
	{
		auto t1 = rand() / 746734.0;
		auto t2 = rand() / 746734.0;

		vt(i) = t1;
		vt1(i) = t2;

		vvt[i] = t1;
		vvt1[i] = t2;
	}

	double vr = dotProd(vvt, vvt1);
	double vr2 = vt.dot(vt1);

	if (!failed)
	{
		std::cout << "All Test passed" << std::endl;
	}
	else {
		std::cout << failed << " Tests failed" << std::endl;
	}

	return failed;
}