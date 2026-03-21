#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <omp.h>

double u_test(double x, double y) {
	return std::exp(x * x - y * y);
}

double f_main(double x, double y) {
	return std::atan(x / y);
}

double f_test(double x, double y) {
	return 4.0 * (x * x + y * y) * std::exp(x * x - y * y);
}

double mu1(double y) {
	return 0.0;
}

double mu2(double y) {
	return 0.0;
}

double mu3(double x) {
	const double PI = 3.14159265358979323846;
	return std::sin(PI * x) * std::sin(PI * x);
}

double mu4(double x) {
	return std::cosh((x - 1.0) * (x - 2.0)) - 1.0;
}

double mu1_test(double y) {
	return std::exp(1.0 - y * y);
}

double mu2_test(double y) {
	return std::exp(4.0 - y * y);
}

double mu3_test(double x) {
	return std::exp(x * x - 1.0);
}

double mu4_test(double x) {
	return std::exp(x * x - 4.0);
}

class Matrix {
public:
	double A = 0.0;
	double h2 = 0.0;
	double k2 = 0.0;
	int64_t n = 0;
	int64_t m = 0;
	std::vector<double> data;
	Matrix() = default;
	Matrix(int64_t _n, int64_t _m, double h, double k) : n(_n), m(_m) {
		// A is a -A matrix exactly
		h2 = -1.0 / (h * h);
		k2 = -1.0 / (k * k);
		A = -2.0 * (h2 + k2);
	}
	Matrix(const Matrix& mat) = default;
	Matrix(Matrix&& mat) = default;
	Matrix& operator=(const Matrix& mat) = default;
	Matrix& operator=(Matrix&& mat) = default;
};

class Vector {
public:
	std::vector<double> data;
	Vector() = default;
	Vector(const Vector& mat) = default;
	Vector(Vector&& mat) = default;
	Vector& operator=(const Vector& mat) = default;
	Vector& operator=(Vector&& mat) = default;
};

Vector operator* (double a, const Vector& v) {
	Vector res = v;
	for (double& x : res.data) x *= a;
	return res;
}

Vector operator* (const Vector& v, double a) {
	Vector res = v;
	for (double& x : res.data) x *= a;
	return res;
}

Vector& operator*= (Vector& v, double a) {
	for (double& x : v.data) x *= a;
	return v;
}

Vector operator/ (const Vector& v, double a) {
	Vector res = v;
	for (double& x : res.data) x /= a;
	return res;
}

Vector operator+ (const Vector& v1, const Vector& v2) {
	Vector res = v1;
	for (size_t i = 0; i < v1.data.size(); i++) res.data[i] += v2.data[i];
	return res;
}

Vector& operator+= (Vector& v1, const Vector& v2) {
	for (size_t i = 0; i < v1.data.size(); i++) v1.data[i] += v2.data[i];
	return v1;
}

Vector& operator-= (Vector& v1, const Vector& v2) {
	for (size_t i = 0; i < v1.data.size(); i++) v1.data[i] -= v2.data[i];
	return v1;
}

Vector operator- (const Vector& v1, const Vector& v2) {
	Vector res = v1;
	for (size_t i = 0; i < v1.data.size(); i++) res.data[i] -= v2.data[i];
	return res;
}

double norm2(const Vector& v) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int i = 0; i < v.data.size(); ++i) {
		res += v.data[i] * v.data[i];
	}
	res = std::sqrt(res);
	return res;
}

double norminf(const Vector& v, double& x_max, double& y_max) {
	double res = 0.0;
	for (double x : v.data)
		res = std::max(res, std::abs(x));
	return res;
}

double norm_inverse_matrix(const Matrix& A) {
	double res = 0.0;
	const double PI = 3.14159265358979323846;
	double n = static_cast<double>(A.n);
	double m = static_cast<double>(A.m);

	res = 4.0 * A.h2 * std::sin(PI / (2.0 * n)) * std::sin(PI / (2.0 * n)) + 4.0 * A.k2 * std::sin(PI / (2.0 * m)) * std::sin(PI / (2.0 * m));
	res = std::abs(res);
	res = 1.0 / res;

	return res;
}

double prod(const Vector& v1, const Vector& v2) {
	double res = 0.0;

#pragma omp parallel for reduction(+:res)
	for (int i = 0; i < v1.data.size(); ++i) {
		res += v1.data[i] * v2.data[i];
	}
	return res;
}

void init(Vector& v, size_t n) {
	v.data.assign(n, 0.0);
}

void print_grid(const Vector& v, int64_t n, int64_t m) {
	for (int64_t j = m; j >= 0; --j) {
		for (int64_t i = 0; i <= n; ++i) {
			std::cout << std::setw(10) << std::fixed << std::setprecision(4) << v.data[(j * (n + 1)) + i];
		}
		std::cout << std::endl;
	}
}

Vector gemv(const Matrix& A, const Vector& x) {
	Vector y;
	init(y, x.data.size());
	const std::vector<double>& v = x.data;
#pragma omp parallel for
	for (int j = 1; j < A.m; ++j) {

		int row = j * (A.n + 1);
		int row_up = (j + 1) * (A.n + 1);
		int row_down = (j - 1) * (A.n + 1);

		const double* v_row = v.data() + row;
		const double* v_row_up = v.data() + row_up;
		const double* v_row_down = v.data() + row_down;
		double* y_row = y.data.data() + row;

		for (int i = 1; i < A.n; ++i) {
			y_row[i] = A.A * v_row[i] + A.h2 * (v_row[i - 1] + v_row[i + 1]) + A.k2 * (v_row_down[i] + v_row_up[i]);
		}
	}

	return y;
}

void gemv_inplace(const Matrix& A, const Vector& x, Vector& y) {
	const std::vector<double>& v = x.data;
#pragma omp parallel for
	for (int j = 1; j < A.m; ++j) {

		int row = j * (A.n + 1);
		int row_up = (j + 1) * (A.n + 1);
		int row_down = (j - 1) * (A.n + 1);

		const double* v_row = v.data() + row;
		const double* v_row_up = v.data() + row_up;
		const double* v_row_down = v.data() + row_down;
		double* y_row = y.data.data() + row;

		for (int i = 1; i < A.n; ++i) {
			y_row[i] = A.A * v_row[i] + A.h2 * (v_row[i - 1] + v_row[i + 1]) + A.k2 * (v_row_down[i] + v_row_up[i]);
		}
	}
}

void VectorFMA(double alpha, Vector& v1, double beta, const Vector& v2) {
#pragma omp parallel for
	for (int i = 0; i < v1.data.size(); ++i) {
		v1.data[i] *= alpha;
		v1.data[i] += beta * v2.data[i];
	}
}

int MCG(const Matrix& A, const Vector& b, Vector& x, double& eps, int N, double& eps_r, int& currentN) {
	int steps = currentN;
	int mode = 0;
	size_t n = x.data.size();
	Vector h;
	Vector Ah;
	Vector r;
	init(h, n);
	init(Ah, n);
	r = gemv(A, x) - b;
	double beta = 0.0;
	double alpha = 0.0;
	double cureps = 0.0;
	double cureps_r = 0.0;
	double prodAhh = 0.0;

	for (;;) {
		// can I put all of this into one cycle? - exactly, I can, but for what reason?
		VectorFMA(beta, h, -1.0, r); // h = beta * h + (-1.0) * r; 
		gemv_inplace(A, h, Ah);
		prodAhh = prod(Ah, h);
		alpha = -prod(r, h) / prodAhh;
		VectorFMA(1.0, x, alpha, h); // x = 1.0 * x + alpha * h;

		steps++;
		cureps = alpha * norm2(h);
		VectorFMA(1.0, r, alpha, Ah); // r = 1.0 * r + alpha * Ah; r = Ax - b, x = x_prev + alpha*h -> r = r_prev + alpha * Ah
		cureps_r = norm2(r);
		beta = prod(Ah, r) / prodAhh;

		if (cureps_r <= eps_r) {
			mode = 1;
			break;
		}
		if (cureps <= eps) {
			mode = 2;
			break;
		}
		if (steps >= N) {
			mode = 0;
			break;
		}
	}

	eps = cureps;
	eps_r = cureps_r;
	currentN = steps;
	return mode;
}

class Solver {
public:
	// parameters
	const double a;
	const double b;
	const double c;
	const double d;
	int64_t n;
	int64_t m;
	int first_x_mode;
	int task_number;
	Matrix A;
	Vector f;

	// stats
	Vector x;
	Vector x_example; // more accurate x
	Vector x_interp; // x0
	Vector x_example_interp; // x0 for accurate method
	double cureps;
	double cureps2;
	double cureps_r;
	double cureps_r2;
	int curN;
	int curN2;
	double time_grid;
	double time_grid2;

	void init_x() { // mode, task
		init(x, (n + 1) * (m + 1));
		double h = (b - a) / static_cast<double>(n);
		double k = (d - c) / static_cast<double>(m);
		double y = 0.0;
		double x = 0.0;
		size_t offset = 0;
		if (task_number == 0) {
			for (size_t j = 0; j <= m; ++j) {
				y = c + static_cast<double>(j) * k;
				offset = j * (n + 1);
				this->x.data[offset + 0] = mu1_test(y);
				this->x.data[offset + n] = mu2_test(y);
			}
			for (size_t i = 0; i <= n; ++i) {
				x = a + static_cast<double>(i) * h;
				this->x.data[0 * (n + 1) + i] = mu3_test(x);
				this->x.data[m * (n + 1) + i] = mu4_test(x);
			}
		}
		else if (task_number == 1) {
			for (size_t j = 0; j <= m; ++j) {
				y = c + static_cast<double>(j) * k;
				offset = j * (n + 1);
				this->x.data[offset + 0] = mu1(y);
				this->x.data[offset + n] = mu2(y);
			}
			for (size_t i = 0; i <= n; ++i) {
				x = a + static_cast<double>(i) * h;
				this->x.data[0 * (n + 1) + i] = mu3(x);
				this->x.data[m * (n + 1) + i] = mu4(x);
			}
		}
		else {
			throw std::runtime_error("Bad task choice");
		}

		if (first_x_mode == 0) {
			for (size_t j = 1; j < A.m; ++j) {
				for (size_t i = 1; i < A.n; ++i) {
					size_t central = (j * (A.n + 1)) + i;
					this->x.data[central] = 0.0;
				}
			}
		}
		else if (first_x_mode == 1) { // x mode
			for (size_t j = 1; j < A.m; ++j) {
				y = c + static_cast<double>(j) * k;
				for (size_t i = 1; i < A.n; ++i) {
					x = a + static_cast<double>(i) * h;
					size_t central = (j * (A.n + 1)) + i;
					if (task_number == 0) {
						this->x.data[central] = mu1_test(y) * (b - x) / (b - a) + mu2_test(y) * (x - a) / (b - a);
					}
					else if (task_number == 1) {
						this->x.data[central] = mu1(y) * (b - x) / (b - a) + mu2(y) * (x - a) / (b - a);
					}
					else {
						throw std::runtime_error("Bad task choice");
					}
				}
			}
		}
		else if (first_x_mode == 2) { // y mode
			for (size_t j = 1; j < A.m; ++j) {
				y = c + static_cast<double>(j) * k;
				for (size_t i = 1; i < A.n; ++i) {
					x = a + static_cast<double>(i) * h;
					size_t central = (j * (A.n + 1)) + i;
					if (task_number == 0) {
						this->x.data[central] = mu3_test(x) * (d - y) / (d - c) + mu4_test(x) * (y - c) / (d - c);
					}
					else if (task_number == 1) {
						this->x.data[central] = mu3(x) * (d - y) / (d - c) + mu4(x) * (y - c) / (d - c);
					}
					else {
						throw std::runtime_error("Bad task choice");
					}
				}
			}
		}
		else if (first_x_mode == 3) { // avg mode
			for (size_t j = 1; j < A.m; ++j) {
				y = c + static_cast<double>(j) * k;
				for (size_t i = 1; i < A.n; ++i) {
					x = a + static_cast<double>(i) * h;
					size_t central = (j * (A.n + 1)) + i;
					if (task_number == 0) {
						this->x.data[central] = 0.5 * (mu1_test(y) * (b - x) / (b - a) + mu2_test(y) * (x - a) / (b - a) + mu3_test(x) * (d - y) / (d - c) + mu4_test(x) * (y - c) / (d - c));
					}
					else if (task_number == 1) {
						this->x.data[central] = 0.5 * (mu1(y) * (b - x) / (b - a) + mu2(y) * (x - a) / (b - a) + mu3(x) * (d - y) / (d - c) + mu4(x) * (y - c) / (d - c));
					}
					else {
						throw std::runtime_error("Bad task choice");
					}
				}
			}
		}
		else {
			throw std::runtime_error("Bad first x mode choice");
		}

		this->x_interp = this->x;
	}

	void init_mat() {
		double h = (b - a) / static_cast<double>(n);
		double k = (d - c) / static_cast<double>(m);
		A = Matrix(n, m, h, k);
	}

	void init_f() { // task
		init(f, (n + 1) * (m + 1));
		double h = (b - a) / static_cast<double>(n);
		double k = (d - c) / static_cast<double>(m);
		double y = 0.0;
		double x = 0.0;
		size_t offset = 0;
		if (task_number == 0) {
			for (size_t j = 1; j < m; ++j) {
				y = c + static_cast<double>(j) * k;
				offset = j * (n + 1);
				for (size_t i = 1; i < n; ++i) {
					x = a + static_cast<double>(i) * h;
					f.data[offset + i] = -f_test(x, y);
				}
			}
		}
		else if (task_number == 1) {
			for (size_t j = 1; j < m; ++j) {
				y = c + static_cast<double>(j) * k;
				offset = j * (n + 1);
				for (size_t i = 1; i < n; ++i) {
					x = a + static_cast<double>(i) * h;
					f.data[offset + i] = -f_main(x, y);
				}
			}
		}
		else {
			throw std::runtime_error("Bad task choice");
		}
	}

	Solver(int64_t _n, int64_t _m, int _first_x_mode, int _task_number) :
		a(1.0), b(2.0), c(1.0), d(2.0), n(_n), m(_m), first_x_mode(_first_x_mode), task_number(_task_number) {
		init_mat();
		init_x();
		init_f();
	}

	void solve(double eps, int N, double eps_r) {
		int mode = -1;
		this->cureps = eps;
		this->cureps_r = eps_r;
		this->curN = 0;
		auto t1 = std::chrono::steady_clock::now();
		mode = MCG(this->A, this->f, this->x, cureps, N, cureps_r, curN);
		auto t2 = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed = t2 - t1;
		time_grid = elapsed.count();
	}

	std::tuple<double, double, double> finderr(int norm_type, double eps, double eps_r, int N) {
		Vector res = this->x;
		double h = (b - a) / static_cast<double>(n);
		double k = (d - c) / static_cast<double>(m);
		double y = 0.0;
		double x = 0.0;
		double x_err = 0.0;
		double y_err = 0.0;
		size_t offset = 0;
		init(x_example, (n + 1) * (m + 1));
		init(x_example_interp, (n + 1) * (m + 1));

		if (task_number == 0) {
			for (size_t j = 0; j <= m; ++j) {
				y = c + static_cast<double>(j) * k;
				offset = j * (n + 1);
				for (size_t i = 0; i <= n; ++i) {
					x = a + static_cast<double>(i) * h;
					res.data[offset + i] -= u_test(x, y);
					this->x_example.data[offset + i] = u_test(x, y);
					this->x_example_interp.data[offset + i] = u_test(x, y);
				}
			}			
			cureps2 = 0.0;
			cureps_r2 = 0.0;
			curN2 = 0;
			time_grid2 = 0.0;
		}
		else if (task_number == 1) {
			Solver s(n * 2, m * 2, first_x_mode, task_number);
			s.solve(eps, N, eps_r);
			const Vector& x2 = s.x;

			for (size_t j = 0; j <= m; ++j) {
				offset = j * (n + 1);
				for (size_t i = 0; i <= n; ++i) {
					res.data[(j * (n + 1)) + i] -= x2.data[(j * 2) * (2 * n + 1) + i * 2];
					this->x_example.data[offset + i] = x2.data[(j * 2) * (2 * n + 1) + i * 2];
					this->x_example_interp.data[offset + i] = s.x_interp.data[(j * 2) * (2 * n + 1) + i * 2];
				}
			}
			cureps2 = s.cureps;
			cureps_r2 = s.cureps_r;
			curN2 = s.curN;
			time_grid2 = s.time_grid;
		}
		else {
			throw std::runtime_error("Invalid task number");
		}

		double resnorm = 0.0;
		if (norm_type == 0) {
			resnorm = norm2(res);
		}
		else if (norm_type == 1) {
			resnorm = 0.0;
			for (size_t j = 0; j <= m; ++j) {
				y = c + static_cast<double>(j) * k;
				offset = j * (n + 1);
				for (size_t i = 0; i <= n; ++i) {
					x = a + static_cast<double>(i) * h;
					if (std::abs(res.data[offset + i]) > resnorm) {
						resnorm = std::abs(res.data[offset + i]);
						x_err = x;
						y_err = y;
					}
				}
			}
		}
		else {
			throw std::runtime_error("Invalid norm type");
		}

		return std::make_tuple(resnorm, x_err, y_err);
	}

	double get_r0() {
		Vector r;
		r = gemv(A, x_interp) - f;
		return norm2(r);
	}

	void print_grid(const Vector& v) {
		for (int64_t j = m; j >= 0; --j) {
			for (int64_t i = 0; i <= n; ++i) {
				std::cout << std::setw(10) << std::fixed << std::setprecision(4) << v.data[(j * (n + 1)) + i];
			}
			std::cout << std::endl;
		}
	}

	void print_test_func() {
		double h = (b - a) / static_cast<double>(n);
		double k = (d - c) / static_cast<double>(m);
		double y = 0.0;
		double x = 0.0;
		for (int64_t j = m; j >= 0; --j) {
			y = c + static_cast<double>(j) * k;
			for (int64_t i = 0; i <= n; ++i) {
				x = a + static_cast<double>(i) * h;
				std::cout << std::setw(10) << std::fixed << std::setprecision(4) << u_test(x, y);
			}
			std::cout << std::endl;
		}
	}

	void print_f_func() {
		double h = (b - a) / static_cast<double>(n);
		double k = (d - c) / static_cast<double>(m);
		double y = 0.0;
		double x = 0.0;
		for (int64_t j = m; j >= 0; --j) {
			y = c + static_cast<double>(j) * k;
			for (int64_t i = 0; i <= n; ++i) {
				x = a + static_cast<double>(i) * h;
				std::cout << std::setw(10) << std::fixed << std::setprecision(4) << f_test(x, y);
			}
			std::cout << std::endl;
		}
	}

	void write_to_file(std::string file) {
		Vector diff;
		init(diff, (n + 1) * (m + 1));
		diff = this->x - this->x_example;

		std::string path_to_dir = "C:/<path>/data/";
		std::string file_x = path_to_dir + file + "_x.bin";
		std::string file_example = path_to_dir + file + "_example.bin";
		std::string file_diff = path_to_dir + file + "_diff.bin";
		std::string file_x_interp = path_to_dir + file + "_x_interp.bin";
		std::string file_example_interp = path_to_dir + file + "_example_interp.bin";

		std::ofstream out_x(file_x, std::ios::out | std::ios::binary);
		std::ofstream out_example(file_example, std::ios::out | std::ios::binary);
		std::ofstream out_diff(file_diff, std::ios::out | std::ios::binary);
		std::ofstream out_x_interp(file_x_interp, std::ios::out | std::ios::binary);
		std::ofstream out_example_interp(file_example_interp, std::ios::out | std::ios::binary);

		out_x.write(reinterpret_cast<const char*>(x.data.data()), x.data.size() * sizeof(double));
		out_example.write(reinterpret_cast<const char*>(x_example.data.data()), x_example.data.size() * sizeof(double));
		out_diff.write(reinterpret_cast<const char*>(diff.data.data()), diff.data.size() * sizeof(double));
		out_x_interp.write(reinterpret_cast<const char*>(x_interp.data.data()), x_interp.data.size() * sizeof(double));
		out_example_interp.write(reinterpret_cast<const char*>(x_example_interp.data.data()), x_example_interp.data.size() * sizeof(double));

		out_x.close();
		out_example.close();
		out_diff.close();
		out_x_interp.close();
		out_example_interp.close();
	}
};

int main(int argc, char* argv[]) { // n, m, task, first_x, eps, eps_r, maxN, eps2, eps_r2, maxN2
	std::string path_to_dir = "C:/<path>/data/";
	const bool console_mode = true;
	const bool file_mode = true;
	int64_t n = 0;
	int64_t m = 0;
	int task = 0;
	int first_x = 0;
	double eps = 0.0;
	double eps_r = 0.0;
	int maxN = 0;
	double eps2 = 0.0;
	double eps_r2 = 0.0;
	int maxN2 = 0;

	if (console_mode) {
		if (argc != 11) {
			std::cerr << "Usage: n, m, task, first_x, eps, eps_r, maxN, eps2, eps_r2, maxN2" << std::endl;
			throw std::runtime_error("Invalid usage\n");
		}

		n = atoll(argv[1]);
		m = atoll(argv[2]);
		task = atoi(argv[3]);
		first_x = atoi(argv[4]);
		eps = atof(argv[5]);
		eps_r = atof(argv[6]);
		maxN = atoi(argv[7]);
		eps2 = atof(argv[8]);
		eps_r2 = atof(argv[9]);
		maxN2 = atoi(argv[10]);
	}
	else {
		n = 4;
		m = 4;
		task = 0;
		first_x = 1;
		eps = 1e-13;
		eps_r = 1e-12;
	    maxN = 100000;
		eps2 = 1e-13;
		eps_r2 = 1e-12;
		maxN2 = 400000;
	}

	if (!file_mode) {
		std::cout << n << " " << m << " " << task << " " << first_x << " " << eps << " " << eps_r << " " << maxN << " " << eps2 << " " << eps_r2 << " " << maxN2 << std::endl;
	}
	
	try {
		double err, x_err, y_err;
		int norm = 1;
		Solver s(n, m, first_x, task);
		std::string file = "data";
		double norm_r0;

		s.solve(eps, maxN, eps_r);
		auto restuple = s.finderr(norm, eps2, eps_r2, maxN2);
		err = std::get<0>(restuple);
		x_err = std::get<1>(restuple);
		y_err = std::get<2>(restuple);
		norm_r0 = s.get_r0();

		if (!file_mode) {
			std::cout << "Error is: " << err << "; x: " << x_err << "; y: " << y_err << std::endl;
			std::cout << "eps: " << s.cureps << ", eps_r: " << s.cureps_r << ", N: " << s.curN << " " << ", time: " << s.time_grid << std::endl;
			std::cout << "eps2: " << s.cureps2 << ", eps_r2: " << s.cureps_r2 << ", N2: " << s.curN2 << " " << ", time2: " << s.time_grid2 << std::endl;
			std::cout << "norm_r0: " << norm_r0 << std::endl;
		}
		else {
			std::ofstream out(path_to_dir + file + ".txt"); // dim of x first (only linear, mxm)
			out << (n + 1) << std::endl << (m + 1) << std::endl << err << std::endl << x_err << std::endl << y_err << std::endl
				<< s.cureps << std::endl << s.cureps_r << std::endl << s.curN << std::endl << s.time_grid << std::endl
				<< s.cureps2 << std::endl << s.cureps_r2 << std::endl << s.curN2 << std::endl << s.time_grid2 << std::endl
				<< norm_r0 << std::endl;
			out.close();
			s.write_to_file(file);
		}
	}
	catch (std::exception& e) {
		std::cerr << "Exception was caught " << e.what() << std::endl;
		throw std::runtime_error("Exception happened\n");
	}

	return 0;
}
