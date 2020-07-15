#pragma once


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
}
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
class SparseMatrix {

	using Vector = std::vector<T>;
	using Index = int;
	using IndexVector = std::vector<Index>;
	// using Bitmap = std::vector<bool>;

	Vector values_;
	IndexVector col_offset_;
	IndexVector row_begin_;
	IndexVector row_num_nze_; // number of non zero elements in row
	IndexVector row_space_left_;
	Index n_rows_;
	Index n_cols_;

public:

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
			memmove(values_.data() + idx, values_.data() + idx+1, all * sizeof(T));
			memmove(col_offset_.data() + idx, col_offset_.data() + idx+1, all * sizeof(T));
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

	
	void initializeFromVector(const IndexVector& rows, IndexVector&& cols, Vector&& vals) {
		values_ = std::forward<Vector>(vals);
		col_offset_ = std::forward<IndexVector>(cols);

		n_rows_ = rows.back() + 1;
		n_cols_ = 0;
		for (auto x : col_offset_) {
			n_cols_ = std::max(n_cols_, x);
		}
		n_cols_++;
		
		row_begin_.clear();
		row_begin_.resize(n_rows_, 0);
		row_space_left_.clear();
		row_space_left_.resize(n_rows_, 0);

		Index idx = 0;
		Index ridx = 0;
		Index last_row = -1;
		for (auto& r : rows) {
			if (r != last_row) {
				last_row = r;
				ridx = idx;
			}
			if (values_[idx] == 0) {
				++row_space_left_[r];
			}
			else {
				++row_begin_[r];
				values_[ridx] = values_[idx];
				col_offset_[ridx] = col_offset_[idx];
				++ridx;
			}
			++idx;
		}

		row_num_nze_ = row_begin_;

		Index sum = 0;
		Index num = 0;
		for (auto& r : row_begin_) {
			Index last_sum = sum;
			sum += r + row_space_left_[num];
			r = last_sum;
			++num;
		}
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
	
	
	std::vector<double> gaussSeidel(std::vector<double> b, double epsilon = 1e-6, int max_iteration = 1000) {
		M_ASSERT(b.size() == n_cols_, "len(b) must match matrix's column");
		std::vector<double> x(b.size(), 1.0f);
		std::vector<double> prev(b.size(), 1.0f);
		double eps = 10;
		int cnt = 0;
		while (eps > epsilon && cnt < max_iteration)
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

	void applyToVector(std::vector<double>& in, std::vector<double>& out) {
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


	std::vector<double> conjugateGradient(const std::vector<double>& b, double epsilon = 1e-6, int max_iteration = 1000) {
		std::vector<double> x(b.size(), 0);
		std::vector<double> r(b.size());
		std::vector<double> r1(b.size());
		applyToVector(x, r);
		vecsub(b, r, r);
		std::vector<double> p = r;
		int k = 0;

		std::vector<double> Ap(b.size());
		int cnt = 0;
		while (cnt < max_iteration)
		{
			applyToVector(p, Ap);
			double alpha = veclen2(r) / dotProd(p, Ap);
			vecadd(x, p, alpha, x);
			vecadd(r, Ap, -alpha, r1);
			auto r1len = veclen2(r1);
			if (r1len < epsilon) break;
			double beta = r1len / veclen2(r);
			vecadd(r1, p, beta, p);
			using std::swap;
			swap(r1, r);
			++cnt;
		}

		return x;
	}

private:
	
	// row_num_nze_[row] > 0, that is , there is at least an element in this row,
	// otherwise the result is NOT valid
	inline Index getNearestIndex(Index row, Index col) const {
		Index idx = row_begin_[row];
		Index end = row_begin_[row] + row_num_nze_[row]-1;
		if (col_offset_[idx] == col) {
			return idx;
		}

		while (end > idx)
		{
			Index mid = (end + idx) / 2;
			if (col_offset_[mid] < col) {
				idx = mid+1;
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
};