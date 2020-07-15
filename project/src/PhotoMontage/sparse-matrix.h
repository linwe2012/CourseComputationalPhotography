#pragma once

// Must be compiled under c++17 standard

#include <vector>
#include <memory>
#include <numeric>
#include <execution>
#include <algorithm>

#ifdef _DEBUG

#include <exception>
#define M_ASSERT(cond, msg) if(!(cond)) { throw std::exception(msg); }

#else 

#define M_ASSERT(cond, msg) (void) 0

#endif // _DEBUG

#ifdef USE_NAME_SPACE
namespace USE_NAME_SPACE {
#endif

template< class T = void >
struct abs_diff {
	constexpr T operator()(const T& lhs, const T& rhs) const {
		return std::abs(lhs - rhs);
	}
};

template <>
struct abs_diff<void> {
	using is_transparent = int;

	template <class _Ty1, class _Ty2>
	constexpr auto operator()(_Ty1&& _Left, _Ty2&& _Right) const
		noexcept(noexcept(static_cast<_Ty1&&>(_Left) + static_cast<_Ty2&&>(_Right))) // strengthened
		-> decltype(static_cast<_Ty1&&>(_Left) + static_cast<_Ty2&&>(_Right)) {
		return std::abs(static_cast<_Ty1&&>(_Left) - static_cast<_Ty2&&>(_Right));
	}
};

inline double manhattonDist(const std::vector<double>& a, const std::vector<double>& b) {
	M_ASSERT(a.size() == b.size(), "dims of 2 vecs mismatch");
	return std::transform_reduce(std::execution::par,
		a.begin(), a.end(), b.begin(), 0.0, std::plus<>(), abs_diff<>());
}

template<typename T>
inline double veclen2(const std::vector<T>& a) {
	return std::transform_reduce(std::execution::par,
		a.begin(), a.end(), a.begin(), 0.0, std::plus<>(), std::multiplies<>());
}


template<typename T>
inline double dotProd(const std::vector<T>& a, const std::vector<T>& b) {
	return std::transform_reduce(std::execution::par,
		a.begin(), a.end(), b.begin(), 0.0, std::plus<>(), std::multiplies<>());
	///return std::inner_product(a.begin(), a.end(), b.begin(), 0);
}
/*
template<typename T>
inline double dotProd(const std::vector<T>& a, const std::vector<T>& b) {
	double res = 0;
	for (int i = 0; i < a.size(); ++i)
	{
		res += a[i] * b[i];
	}
	return res;
}*/

template<typename T>
inline void vecsub(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out) {
	std::transform(std::execution::par,
		a.begin(), a.end(), b.begin(), out.begin(), std::minus<>());
}

template<typename T>
inline void vecadd(const std::vector<T>& a, const std::vector<T>& b, T scale_b, std::vector<T>& out) {
	std::transform(std::execution::par,
		a.begin(), a.end(), b.begin(), out.begin(), [scale_b](auto a, auto b) { return a + scale_b * b; });
}

template<typename T>
inline void vecadd(const std::vector<T>& src, const double inc, std::vector<T>& out)
{
	std::transform(std::execution::par,
		src.begin(), src.end(), out.begin(), [inc](double a) { return a + inc; });
}

template<typename T>
inline void vecmul(const std::vector<T>& src, const double scale, std::vector<T>& out)
{
	std::transform(std::execution::par,
		src.begin(), src.end(), out.begin(), [scale](double a) { return a * scale; });
}

template<typename T>
inline void vecmul(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out) {
	std::transform(std::execution::par,
		a.begin(), a.end(), b.begin(), out.begin(), std::multiplies<>());
}

template<typename T, typename IndexType = int>
class SparseMatrix {

	using Vector = std::vector<T>;
	
	
	// using Bitmap = std::vector<bool>;

	

public:

	using Index = IndexType;
	using IndexVector = std::vector<Index>;

	//TODO
	struct RowMajorIterator {
		SparseMatrix& sp_;
		Index row_;


		struct ColumnInEachRow {
			SparseMatrix& sp_;
			Index col_;
			Index row_;

			bool operator==(const ColumnInEachRow& rhs) const {
				M_ASSERT(&sp_ == &rhs.sp_, "Must be iterators from the same matrix");
				M_ASSERT(row_ == rhs.row_, "Must be iterators from the same row");
				return rhs.col_ == rhs.col_;
			}

			ColumnInEachRow(SparseMatrix& sp, Index row) :sp_(sp), row_(row) {

			}
		};

		ColumnInEachRow operator*() {
			return ColumnInEachRow(sp_, row_);
		}

		void operator++() {
			++row_;
		}

		bool operator==(const RowMajorIterator& rhs) const {
			M_ASSERT(&sp_ == &rhs.sp_, "Must be iterators from the same matrix");
			return rhs.row_ == rhs.row_;
		}

	};

	Index cols() const { return n_cols_; }
	Index rows() const { return n_rows_; }

	T at(Index row, Index col) const {
		if (!row_num_nze_[row])
			return T(0);

		Index idx = getNearestIndex(row, col);

		if (col_offset_[idx] == col) {
			return values_[idx];
		}

		return T(0);
	}

	// Suuport Eigen-ish API
	T coeff(Index row, Index col) const {
		return at(row, col);
	}

	SparseMatrix() = default;
	SparseMatrix(SparseMatrix&&) = default;

	void insertZero(Index row, Index col) {
		// all elements in this row is already zero, so we dont have to do anything
		if (row_num_nze_[row] == 0) {
			return;
		}

		Index idx = getNearestIndex(row, col);

		// make an non zero element to zero, we need to adjust this row
		if (col_offset_[idx] == col) {
			--row_num_nze_[row];
			++row_space_left_[row];
			auto cnt = idx - row_begin_[row];
			auto all = row_num_nze_[row] - cnt;
			memmove(values_.data() + idx, values_.data() + idx + 1, all * sizeof(T));
			memmove(col_offset_.data() + idx, col_offset_.data() + idx + 1, all * sizeof(T));
		}
		// else make an zero element to zero, we don't have to do anything
	}

	void insertNoneZero(T&& val, Index row, Index col) {
		Index idx = row_begin_[row];

		if (row_num_nze_[row]) {
			idx = getNearestIndex(row, col);
			// we modify an existing non-zero element, simply change it and indices are left untouched
			if (col_offset_[idx] == col) {
				values_[idx] = std::forward<T>(val);
				return;
			}
		}

		// there is still spaces in this row, so we use it
		if (row_space_left_[row]) {
			--row_space_left_[row];
			Index beg = idx;
			Index cnt = row_begin_[row] + row_num_nze_[row] - idx - 1;
			memmove(values_.data() + beg + 1, values_.data() + beg, cnt * sizeof(T));
			memmove(col_offset_.data() + beg + 1, col_offset_.data() + beg, cnt * sizeof(Index));
			values_[idx] = std::forward<T>(val);
			col_offset_[idx] = col;
		}
		// no space left, we insert it into the right position, and update indices for rows
		else {
			values_.insert(values_.begin() + idx, val);
			col_offset_.insert(col_offset_.begin() + idx, col);
			auto sz = static_cast<Index>(row_begin_.size());
			for (Index i = row + 1; i < sz; ++i) {
				++row_begin_[i];
			}
		}


		++row_num_nze_[row];
	}

	void insert(const T& val, Index row, Index col) {
		insert(T(val), row, col);
	}

	void insert(T&& val, Index row, Index col) {
		return val == T(0) ?
			insertZero(row, col) :
			insertNoneZero(std::forward<T>(val), row, col);
	}

	struct Triplet
	{
		Index row;
		Index col;
		T val;
	};

	void initializeFromTriplets(Triplet* a, Index cnt)
	{
		for (Index i = 0; i < cnt; ++i)
		{
			auto& r = a[i];
			insert(r.val, r.row, r.col);
		}
	}

	void initializeFromVector(const IndexVector& rows, IndexVector&& cols, Vector&& vals) {
		values_ = std::forward<Vector>(vals);
		col_offset_ = std::forward<IndexVector>(cols);

		// estimating the size of matrix
		n_rows_ = rows.back() + 1;
		n_cols_ = 0;
		for (auto x : col_offset_) {
			n_cols_ = std::max(n_cols_, x);
		}
		n_cols_++;

		// reseting buffers
		row_begin_.clear();
		row_begin_.resize(n_rows_, 0);
		row_space_left_.clear();
		row_space_left_.resize(n_rows_, 0);

		// reorganize & trim out zeros from vector
		Index idx = 0;
		Index ridx = 0;
		Index last_row = -1;
		for (auto& r : rows) {
			// update estimation of row count
			if (r != last_row) {
				last_row = r;
				ridx = idx;
			}

			// inserting 0
			if (values_[idx] == 0) {
				++row_space_left_[r];
			}
			// inserting non zero
			else {
				++row_begin_[r];  // actually this is the non zero counter
				values_[ridx] = values_[idx];
				col_offset_[ridx] = col_offset_[idx];
				++ridx;
			}
			++idx;
		}

		row_num_nze_ = row_begin_;

		// recompute rows
		Index sum = 0;
		Index num = 0;
		for (auto& r : row_begin_) {
			Index last_sum = sum;
			sum += r + row_space_left_[num];
			r = last_sum;
			++num;
		}
	}

	void initialize(int row, int col)
	{
		n_rows_ = row;
		n_cols_ = col;

		row_begin_.clear();
		row_begin_.resize(n_rows_, 0);
		row_space_left_.clear();
		row_space_left_.resize(n_rows_, 0);
	}

	void initialize(int row, int col, std::initializer_list<T> x) {
		std::vector<T> v(std::move(x));
		std::vector<Index> rows(x.size());
		std::vector<Index> cols(x.size());

		int cnt = 0;
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				rows[cnt] = i;
				cols[cnt] = j;
				++cnt;
			}
		}

		initializeFromVector(rows, std::move(cols), std::move(v));
	}


	std::vector<double> gaussSeidel(const std::vector<double>& b, double epsilon = 1e-6, int max_iteration = 1000) {
		M_ASSERT(b.size() == n_cols_, "len(b) must match matrix's column");
		std::vector<double> x(b.size(), 1.0f);
		std::vector<double> prev(b.size(), 1.0f);
		double eps = 10;
		int cnt = 0;
		while (eps > epsilon&& cnt < max_iteration)
		{
			prev = x;
			for (Index i = 0; i < n_rows_; ++i) {
				auto a_ii = at(i, i);
				if (a_ii == 0) {
					continue;
				}
				double sigma = 0;
				Index idx = row_begin_[i];
				for (Index j = 0; j < row_num_nze_[i]; ++j) {
					auto col = col_offset_[idx];
					if (col != i) {
						sigma += values_[idx] * x[col];
					}
					++idx;
				}
				x[i] = (b[i] - sigma) / a_ii;
			}

			eps = manhattonDist(x, prev);
			++cnt;
		}
		return x;
	}

	void applyToVector(const std::vector<double>& in, std::vector<double>& out) {
		for (Index i = 0; i < n_rows_; ++i) {
			Index idx = row_begin_[i];
			double sum = 0;
			for (Index j = 0; j < row_num_nze_[i]; ++j) {
				auto col = col_offset_[idx];
				sum += values_[idx] * in[col];
				++idx;
			}
			out[i] = sum;
		}
	}


	std::vector<double> conjugateGradient(const std::vector<double>& b, double epsilon = 1e-16, int max_iteration = 1000, const std::vector<double>& initialize = std::vector<double>()) {
		std::vector<double> x(b.size(), 0);
		std::vector<double> r(b.size());
		std::vector<double> r1(b.size());

		if (initialize.size()) {
			x = initialize;
		}
		
		// r(0) = b - Ax
		applyToVector(x, r);
		vecsub(b, r, r);
		
		
		std::vector<double> p = r; // p(0) = r(0)
		int k = 0;

		std::vector<double> Ap(b.size());
		int cnt = 0;
		double error = 0;
		while (cnt < max_iteration)
		{
			double rlen = veclen2(r);
			applyToVector(p, Ap); // Ap = A * p(k)
			double alpha = rlen / dotProd(p, Ap); // alpha = (r' * r) / (p' * Ap)
			vecadd(x, p, alpha, x);     // x(k+1) = x(k) + alpha * p(k)
			vecadd(r, Ap, -alpha, r1);  // r(k+1) = r(k) - alpha * A * p(k)
			auto r1len = veclen2(r1);   // r(k+1) sufficiently small:
			error = r1len;
			if (sqrt(r1len) < epsilon) break; //     break;
			double beta = r1len / rlen; // beta = len2(r(k+1)) / len2(r(k))
			vecadd(r1, p, beta, p);     // p(k+1) = r(k+1) + beta * p(k)
			using std::swap;
			swap(r1, r);                // 
			++cnt;
		}

		return x;
	}

	std::vector<double> conjugateGradientPaper(const std::vector<double>& b, double epsilon = 1e-16, int max_iteration = 1000) {
		std::vector<double> x(b.size(), 0);
		std::vector<double> r(b.size());
		std::vector<double> r1(b.size());

		// r(0) = b - Ax
		applyToVector(x, r);
		vecsub(b, r, r);
		std::vector<double> p = r; // p(0) = r(0)
		int k = 0;

		std::vector<double> Ap(b.size());
		int cnt = 0;
		double error = 0;
		while (cnt < max_iteration)
		{
			double rlen = veclen2(r);
			applyToVector(p, Ap); // Ap = A * p(k)
			double alpha = rlen / dotProd(p, Ap); // alpha = (r' * r) / (p' * Ap)
			vecadd(x, p, alpha, x);     // x(k+1) = x(k) + alpha * p(k)
			vecadd(r, Ap, -alpha, r1);  // r(k+1) = r(k) - alpha * A * p(k)
			auto r1len = veclen2(r1);   // r(k+1) sufficiently small:
			error = r1len;
			if (sqrt(r1len) < epsilon) break; //     break;
			double beta = r1len / rlen; // beta = len2(r(k+1)) / len2(r(k))
			vecadd(r1, p, beta, p);     // p(k+1) = r(k+1) + beta * p(k)
			using std::swap;
			swap(r1, r);                // 
			++cnt;
		}

		return x;
	}

	// extract diagnal sized column from matrix 
	// then inv the result
	std::vector<T> extractDiagnolColInv()
	{
		std::vector<T> res(cols(), T(1));

		for (Index i = 0; i < n_rows_; ++i) {
			Index idx = row_begin_[i];
			for (Index j = 0; j < row_num_nze_[i]; ++j) {
				auto col = col_offset_[idx];
				if (col == i) {
					if (values_[idx] != 0)
					{
						res[col] = T(1) / values_[idx];
					}
					break;
				}
				++idx;
			}
		}
		return res;
	}


	std::vector<double> conjugateGradientEigen(const std::vector<double>& b, double epsilon = 1e-16, int max_iteration = 180) {
		std::vector<double> x(b.size(), 0);
		std::vector<double> r(b.size());

		auto invdiag = extractDiagnolColInv();
		std::vector<double> z(invdiag.size());

		// r(0) = b - Ax
		applyToVector(x, r);
		vecsub(b, r, r);
		std::vector<double> p(r.size()); // p(0) = r(0)
		vecmul(r, invdiag, p);
		int k = 0;

		std::vector<double> Ap(b.size());
		int cnt = 0;
		double olddist = dotProd(p, r);
		double error = 0;
		for (auto& t : r)
		{
			error += t;
		}
		while (cnt < max_iteration)
		{
			// double rlen = veclen2(r);
			applyToVector(p, Ap); // Ap = A * p(k)
			double alpha = olddist / dotProd(p, Ap); // alpha = (r' * r) / (p' * Ap)
			vecadd(x, p, alpha, x);     // x(k+1) = x(k) + alpha * p(k)
			vecadd(r, Ap, -alpha, r);  // r(k+1) = r(k) - alpha * A * p(k)
			error = veclen2(r);   // r(k+1) sufficiently small:
			if (sqrt(error) < epsilon) break; //     break;
			vecmul(r, invdiag, z);
			double newdist = dotProd(z, r);
			double beta = newdist / olddist; // beta = len2(r(k+1)) / len2(r(k))
			olddist = newdist;
			vecadd(z, p, beta, p);     // p(k+1) = r(k+1) + beta * p(k)

			++cnt;
		}

		return x;
	}

	void initializeFromEigenRowMajor(const T* values, Index n_values,    // valuePtr()
		                     const Index* row_offset, Index n_row_offset, // innderIndex()
	                         const Index* col_offset, Index n_col_offset, // outerIndex()
		                     const Index* non_zeros,  Index n_non_zeros  // non zeros
	                         )
	{
		values_.clear();
		row_begin_.clear();
		col_offset_.clear();
		row_num_nze_.clear();
		row_space_left_.clear();

		n_rows_ = n_row_offset;
		n_cols_ = n_col_offset;

		values_.insert(values_.begin(), values, values + n_values);
		row_begin_.insert(row_begin_.begin(), row_offset, row_offset + n_row_offset);
		col_offset_.insert(col_offset_.begin(), col_offset, col_offset + n_values);
		
		row_space_left_.resize(n_rows_, 0);

		// there are 'holes' in data, we need to compute the size of the  'holes'
		// i.e. the space left in the row
		if(non_zeros != nullptr) {
			row_num_nze_.insert(row_num_nze_.begin(), non_zeros, non_zeros + n_non_zeros);
			{
				Index last = 0;
				Index i = 0;
				for (; i < n_rows_; ++i)
				{
					if (row_begin_[i] == n_values) {
						break;
					}
				}

				if (i > 0)
				{
					last = row_begin_[i - 1];
					last += row_num_nze_[i - 1];
				}

				for (; i < n_rows_; ++i)
				{
					row_begin_[i] = last;
				}
			}
			
			for (Index i = 0; i < n_rows_-1; ++i)
			{
				row_space_left_[i] = row_begin_[i + 1] - row_begin_[i] - row_num_nze_[i];
			}
			row_space_left_[n_rows_ - 1] = n_values - row_begin_[n_rows_ - 2] - row_num_nze_[n_rows_-1];
		}
		// there are no holes, we still need to adjust `row_begin_`, and compute 
		// `row_num_nze_` ourself
		else {
			row_num_nze_.resize(n_rows_, 0);
			Index last = 0;
			Index i = 0;
			for (; i < n_rows_-1; ++i)
			{
				row_num_nze_[i] = row_begin_[i + 1] - row_begin_[i];
				if (row_begin_[i] == n_values) {
					break;
				}
			}
			

			// the rest of rows for the matrix are all zeros
			// so `row_num_nze_` stays zero, but `row_begin_` needs 
			// to be adjusted
			if (row_begin_[i] == n_values)
			{
				for (; i < n_rows_; ++i)
				{
					--row_begin_[i];
				}
			}
			else {
				row_num_nze_[i] = n_values - row_begin_[i];
			}

		}
	}

private:

	// It is up to caller to ensure that: row_num_nze_[row] > 0,
	// that is, there is at least ONE element in this row,
	// otherwise the result is NOT valid
	inline Index getNearestIndex(Index row, Index col) const {
		Index idx = row_begin_[row];
		Index end = row_begin_[row] + row_num_nze_[row] - 1;
		if (col_offset_[idx] == col) {
			return idx;
		}

		while (end > idx)
		{
			Index mid = (end + idx) / 2;
			if (col_offset_[mid] < col) {
				idx = mid + 1;
			}
			else {
				end = mid;
			}
		}
		return idx;
	}

	/*
	Index getNearestIndex(Index row, Index col) const {
		Index idx = row_begin_[row];
		Index cnt = 0;
		Index end = row_num_nze_[row];
		while (cnt < end && col_offset_[idx++] < col)
		{
			++cnt;
		}
		return --idx;
	}

	auto getNearestIndexWithCount(Index row, Index col) const {
		Index idx = row_begin_[row];
		Index cnt = 0;
		while (cnt < row_num_nze_[row] && col_offset_[idx++] < col)
		{
			++cnt;
		}
		return std::make_tuple(--idx, cnt);
	}
	*/

	Vector values_;
	IndexVector col_offset_;
	IndexVector row_begin_;
	IndexVector row_num_nze_; // number of non zero elements in row
	IndexVector row_space_left_;
	Index n_rows_ = 0;
	Index n_cols_ = 0;
};

#ifdef USE_NAME_SPACE
}
#endif