#ifndef HW3_GN_34804D67
#define HW3_GN_34804D67

struct GaussNewtonParams {
	GaussNewtonParams() :
		exact_line_search(false),
		gradient_tolerance(1e-5),
		residual_tolerance(1e-5),
		max_iter(1000),
		verbose(false)
	{}
	bool exact_line_search; // ʹ�þ�ȷ�����������ǽ�����������
	double gradient_tolerance; // �ݶ���ֵ����ǰ�ݶ�С�������ֵʱֹͣ����
	double residual_tolerance; // ������ֵ����ǰ����С�������ֵʱֹͣ����
	int max_iter; // ����������
	bool verbose; // �Ƿ��ӡÿ����������Ϣ
};

struct GaussNewtonReport {
	enum StopType {
		STOP_GRAD_TOL,       // �ݶȴﵽ��ֵ
		STOP_RESIDUAL_TOL,   // ����ﵽ��ֵ
		STOP_NO_CONVERGE,    // ������
		STOP_NUMERIC_FAILURE // ������ֵ����
	};
	StopType stop_type; // �Ż���ֹ��ԭ��
	double n_iter;      // ��������
};

class ResidualFunction {
public:
	virtual int nR() const = 0;
	virtual int nX() const = 0;
	virtual void eval(double* R, double* J, double* X) = 0;
};

class GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction* f, // Ŀ�꺯��
		double* X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport* report = nullptr // �Ż��������
	) = 0;
};

#endif /* HW3_GN_34804D67 */