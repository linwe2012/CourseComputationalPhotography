#include "config.h"

#ifdef HW_1



#include <opencv2/opencv.hpp>
#include "unitext/cvUniText.hpp"
#include <vector>
#include <map>
#include <algorithm>

#include <thread>
#include <chrono>

#include <filesystem>


#include "utils.h"
#define CVUI_IMPLEMENTATION
#include "cvui.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;
using namespace cv;
namespace fs = std::filesystem;

#define MainWindow "HW1"

class Chalkboard {
public:
	Chalkboard(Mat& canvas)
		:can_(canvas)
	{
		eraser_ = make_eraser();
		set_raw(canvas);
	}

	Mat make_eraser() {
		const int ext = eraser_sz_.height / 6;
		const int brush = ext / 3;
		const int ew = eraser_sz_.width;
		const int eh = eraser_sz_.height;
		const int w = ext + ew;
		const int h = ext + eh;

		Mat eraser(h, w, CV_8UC4);
		eraser = Scalar(220, 240, 250, 255); // flush w/ white

		std::vector<Point> points= {
			// inner
		{ ext, 0 },
		{ w, 0 },
		{ w, eh},
		{ ext, eh},
		// wires
		{ ext, 0 },
		{ 0, ext },
		{ 0, h},
		{ ext, eh},
		{ ew, h},
		{ w, eh},
		};

		std::vector<Point> vbrush = {
			// left
		{brush, brush * 2},
		{ 0, ext},
		{ 0, h },
		{brush, h - brush },
		// bottom
		{ ew + brush, h - brush },
		{ ew, h },
		};

		const int grace = 2;
		std::vector<Point> trim = {
			{ 0, 0 },
		{ 0, ext-grace },
		{ ext-grace, 0 },
		{w, eh + grace},
		{w, h},
		{ew + grace, h},
		{0, h-2},
		{0, h},
		{2, h}
		};

		for (int i = 0; i < 4; ++i) {
			line(eraser, points[i], points[(i + 1) % 4], { 0 , 0, 0, 255 }, 2);
		}
		for (int i = 4; i < 10; i += 2) {
			line(eraser, points[i], points[i+1], { 0 , 0, 0, 255 }, 2);
		}

		fillConvexPoly(eraser, vbrush.data(), 4, { 22 , 22, 22, 255 });
		fillConvexPoly(eraser, vbrush.data() + 2, 4, { 22, 22, 22, 255 });
		
		fillConvexPoly(eraser, trim.data(), 3, { 0 , 0, 0, 0 });
		fillConvexPoly(eraser, trim.data() + 3, 3, { 0 , 0, 0, 0 });
		fillConvexPoly(eraser, trim.data() + 6, 3, { 0 , 0, 0, 0 });

		return eraser;
	}

	void erase(std::function<void(const Mat&, int)> rerender, Rect area = Rect(0, 0, 0, 0)) {
		const auto [tlx, tly, brx, bry] = normalizeRect(can_, area);

		std::vector<Pixel3i> hist(eraser_sz_.height);
		for (int k = tly; k < bry - eraser_sz_.height; k += 80) {
			for (int j = tlx; j < brx; ++j) {
				Pixel3i point;
				
#define UC(what) static_cast<uchar>(what)

				int cnt = 0;
				for (int i = k; i < k + eraser_sz_.height; ++i) {
					auto raw = raw_.ptr(i) + j * 3;
					auto p = can_.ptr(i) + j * 3;
					point.from_ptr(p);
					point.sub_prt(raw);
					hist[cnt].exp_avg(0.98, point);
					
					*p = UC((*raw) + hist[cnt].x * 0.15);
					++p; ++raw;
					*p = UC((*raw) + hist[cnt].y * 0.15);
					++p; ++raw;
					*p = UC((*raw) + hist[cnt].z * 0.15);
					++cnt;
				}
#undef UC
				if (j % 30 == 0) {
					auto base = can_.clone();
					overlayImage(base, eraser_, { j, k });
					int k = waitKey(20);
					rerender(base, k);
				}
			}
		}
		// rerender(base);
		// waitKey(1000);
		// imshow(MainWindow, can_);
	}

	void set_raw(const Mat& m) {
		raw_ = m.clone();
	}

	static void green(Mat& canvas) {
		Mat noise(600, 900, CV_8UC(1));

		randn(noise, -8, 7);
		for (int i = 0; i < noise.rows; ++i) {
			auto ptr = noise.ptr(i);
			for (int j = 0; j < canvas.cols; ++j) {
				if (*ptr < 80) {
					*ptr = 0;
				}
			}
		}

		int kernel_data[] = {
			1, 1, 0,
			0, 1, 0,
			0, 1, 1
		};
		Mat n_kernel(Size{ 3, 3 }, CV_8UC1, kernel_data);

		dilate(noise, noise, n_kernel);
		dilate(noise, noise, n_kernel);
		dilate(noise, noise, n_kernel);

		auto div = canvas.cols / 2 + canvas.rows / 2;
		for (int i = 0; i < canvas.rows; ++i) {
			auto ptr = canvas.ptr(i);
			for (int j = 0; j < canvas.cols; ++j) {
				double dist = sqrt(pow(i - canvas.rows / 2, 2) + pow(j - canvas.cols / 2, 2));
				auto decay = exp(-dist / div - 0.1);
				int n = noise.at<uchar>(i, j);

				*ptr++ = static_cast<uchar>(51 * decay + n);
				*ptr++ = static_cast<uchar>(96 * decay + n);
				*ptr++ = static_cast<uchar>(57 * decay + n);
			}
		}
	}

private:
	Size eraser_sz_ = {45, 80};
	Mat& can_;
	Mat eraser_;
	Mat raw_;
};

class MousePainter {
public:
	MousePainter (Mat& canvas) 
		:can_(canvas) 
	{ }

	int quick_dist(int x, int y) {
		return abs(last_x_ - x) + abs(last_y_ - y);
	}

	void handle(int ev, int x, int y, [[maybe_unused]] int flags) {
		if (!enable_) return;

		cvui::handleMouse(ev, x, y, flags, &cvui::internal::gContexts[MainWindow]);
		if (ev == EVENT_LBUTTONDOWN) {
			left_down_ = true;
			last_x_ = x;
			last_y_ = y;
			
			return make_dot(x, y);
		}
		else if(ev == EVENT_LBUTTONUP || x < 80)
		{
			if (left_down_) {
				left_down_ = false;
				make_dot(x, y);
				n_move_ = 0;
			}
			return ;
		}

		if (!left_down_) return;
		++n_move_;
		if (n_move_ % 3 == 0) {
			decay_ = rand() % 100 / 1000. + 0.9;
		}
		
		if (ev == EVENT_MOUSEMOVE) {
			
			line(can_, { x, y }, { last_x_, last_y_ }, color_* decay_, 3);
			last_x_ = x;
			last_y_ = y;
			rerender();
		}
	}

	static void dispatch(int ev, int x, int y, int flags, void* userdata) {
		static_cast<MousePainter*>(userdata)->handle(ev, x, y, flags);
	}

	void enable(bool should) {
		enable_ = should;
	}

	void set_color(Scalar color) {
		color_ = color;
	}
private:

	void make_dot(int x, int y) {
		circle(can_, { x, y }, 2, color_, 2);
		rerender();
	}

	void rerender() {
		// imshow(MainWindow, can_);
	}

	Mat& can_;
	bool left_down_ = false;
	int last_x_ = 0;
	int last_y_ = 0;
	double decay_ = 1.;
	int n_move_ = 0;
	bool enable_ = true;

	Scalar color_ = { 242, 252, 248 };
};

class ClosingTheme {
public:
	ClosingTheme(Mat& canvas, int h, int w) {
		Mat circ(h, w, CV_8UC(4));
		circ = Scalar(0, 0, 0, 0);
		circle(circ, { w / 2, h / 2 }, 40, { 0, 0, 0, 255 }, 80);
		circle(circ, { w / 2, h / 2 }, 140, { 0, 0, 0, 255 }, 60);
		circle(circ, { w / 2, h / 2 }, 220, { 0, 0, 0, 255 }, 60);
		circle(circ, { w / 2, h / 2 }, 330, { 0, 0, 0, 255 }, 80);
		circle(circ, { w / 2, h / 2 }, 440, { 0, 0, 0, 255 }, 80);

		overlayImage(canvas, circ, { 0, 0 });
		imshow(MainWindow, canvas);
		waitKey();
		
	}
};

void CharacterCloud(Mat& canvas, int h, int w, std::map<int, uni_text::UniText>& text, std::function<void(const Mat&, int)> rerender) {
	const char* messages[] = {
		"Love OpenCV",
		"Made by Leon Lin",
		"OpenCV Rocks",
		"Dilation",
		"HDR",
		"wsh tql",
		"All hail computer vision",
		"Computer Vision",
		"Visual Computing",
		"Open source computer vision",
		"real-time computer vision",
		"C++"
	};

	const Scalar colors[] = {
		{ 242, 252, 248 , 255},
		{ 208, 175, 94  , 255},
		{ 131, 128, 223 , 255},
		{ 112, 198, 222 , 255},
		{ 174, 202, 248 , 255},
		{ 215, 128, 168 , 255},
		{ 178, 80, 74   , 255},
	};

	constexpr size_t n_msg = sizeof(messages) / sizeof(char*);
	constexpr size_t n_color = sizeof(colors) / sizeof(Scalar);
	std::vector<int> keys;
	std::vector<int> lens;
	for (const auto& i : text) {
		keys.push_back(i.first);
	}
	for (int i = 0; i < n_msg; ++i) {
		lens.push_back(strlen(messages[i]));
	}

	std::vector<Rect> rects;
	Mat typo(canvas.size(), CV_8UC(4));
	typo = Scalar(0, 0, 0, 0);
	auto old_can = canvas.clone();
	auto sm_r = canvas.rows / 50;
	auto sm_c = canvas.cols / 50;
	Mat small(canvas.rows - sm_r*2, canvas.cols - sm_c*2, CV_8UC(4));

	for (int j = 0; j < 300; ++j) {
		resize(typo, small, small.size());
		typo = Scalar(0, 0, 0, 0);
		small.copyTo(typo(Rect{ sm_r, sm_c , small.cols, small.rows }));
		for (int i = 0; i < j / 8 + 1; ++i) {
			auto im = rand() % n_msg;
			auto ic = rand() % n_color;
			auto scale = rand() % 100 * 0.01;
			auto x = rand() % w;
			auto y = rand() % h;
			auto ifz = rand() % keys.size();
			auto fz = keys[ifz];

			text[fz].PutText(typo, messages[im], { x, y }, colors[ic]);
		}

		canvas = old_can.clone() * (100-j) / 100.0;
		overlayImage(canvas, typo, { 0, 0 });
		int k = waitKey(16);
		rerender(canvas, k);
		
	}
	for (int i = 0; i < 80; ++i) {
		canvas *= 0.98;
		text[keys.back()].PutText(canvas, "Thank You!", { 240, 260 }, colors[0]);
		text[keys.back() / 2].PutText(canvas, "Presented by Leonlin", { 240, 350 }, colors[0]);
		
		
		rerender(canvas, waitKey(30));
		
	}
	text[keys.front()].PutText(canvas, "Press any key to close window", { 360, 520 }, colors[0]);
	rerender(canvas, -1);
}


struct WarperMatrix {
	operator Matx33f& () {
		return mat;
	}

	WarperMatrix(int _cols) {
		mat = Matx33f({
		1.f, 0.f, 0.f,
		0.025f, 1.f, 0.f,
		0.0001f, 0.f, 1.f
		});
		cols = _cols;
	}

	void set_warp(float warp) {
		mat(1, 0) = 0.025f * warp;
		mat(2, 0) = 0.0001f * warp;
		if (warp < 0) {
			mat(2, 2) = 1.f - 0.0001f * warp * cols;
			mat(1, 2) = -0.025f * warp * cols;
		}
	}

	int estimate_cols() {
		return (mat(0, 0) * cols + mat(0, 2)) / (mat(2, 0) * cols + mat(2, 2));
 	}
	

	Matx33f mat;
	int cols;
};


struct WarpSlice {
	Mat mat;
	Mat buf;
	Rect rect;

	WarpSlice(const Mat& m) {
		mat = m.clone();
		buf = mat.clone();
	}
	WarpSlice(const Mat&m, Rect r) {
		rect = r;
		mat = m(r).clone();
		buf = mat.clone();
	}

};

void horizontal_shadow(Mat m, float range_start, float range_end, bool inv, float decay) {
	std::vector<float> buf(m.cols);
	if (inv) {
		for (int i = 0; i < m.cols; ++i) {
			float linear = -((double)(m.cols - i)) / m.cols;
			linear *= (range_end - range_start);
			linear += range_start;
			float x = 1 - exp(linear);
			buf[i] = x;
		}
	}
	else {
		for (int i = m.cols-1; i >= 0; --i) {
			float linear = -((double)(m.cols - i)) / m.cols;
			linear *= (range_end - range_start);
			linear += range_start;
			float x = 1 - exp(linear);
			buf[i] = x;
		}
	}


	for (int i = 0; i < m.rows; ++i) {
		auto ptr = m.ptr(i);
		for (int j = 0; j < m.cols; ++j) {
			*ptr = static_cast<uchar>(*ptr * buf[j] * decay);
			++ptr;
			*ptr = static_cast<uchar>(*ptr * buf[j] * decay);
			++ptr;
			*ptr = static_cast<uchar>(*ptr * buf[j] * decay);
			++ptr;
		}
	}
}

void Fold(Mat& canvas, std::function<void(const Mat&, int)> rerender) {
	int num_slices = 4;
	int slice_w = canvas.cols / num_slices;

	WarperMatrix goin(slice_w);
	WarperMatrix goout(slice_w);
	
	
	std::vector<WarpSlice> slices;
	{
		
		for (int i = 0; i < num_slices; ++i) {
			slices.emplace_back(
				canvas, Rect{ slice_w * (num_slices - i - 1), 0, slice_w, canvas.rows }
			);
		}
	}
	
	
	for (int u = 0; u < 300; ++u) {
		canvas = Scalar(0, 0, 0);
		goout.set_warp(-u / 5.f);
		goin.set_warp(u / 5.f);
		float resp = u / 200.f;
		int cnt = 0;
		int cols = goin.estimate_cols();
		for (auto& slice : slices) {
			int pos = (num_slices - cnt - 1) * cols;
			if (cnt % 2 == 0) {
				warpPerspective(slice.mat, slice.buf, goin.mat, slice.mat.size());
				Mat valid = slice.buf(Rect{ 0, 0, cols, slice.buf.rows });

				horizontal_shadow(valid, -3 + resp, -1 + resp, true, 0.91);
				valid.copyTo(
					canvas(Rect{ pos, 0, cols, slice.buf.rows })
				);
			}
			else {
				warpPerspective(slice.mat, slice.buf, goout.mat, slice.mat.size());
				Mat cl;
				resize(slice.buf, cl, Size(cols, slice.buf.rows));
				horizontal_shadow(cl, -9, -3, false, 1.f);
				cl.copyTo(
					canvas(Rect{ pos, 0, cols, slice.buf.rows })
				);
			}
			++cnt;
		}
		rerender(canvas, waitKey(5));
	}
	// warpPerspective(canvas, canvas, warper, canvas.size());
	
}

void FallDown(Mat& canvas, Mat& falling, std::function<void(const Mat&, int)> rerender) {
	Mat base = canvas.clone();
	Mat ori_falling = falling.clone();
	WarperMatrix warp(10);
	float warp_rate = 1.f;
	

}



int main(int argc, char** argv) {
	constexpr bool enable_capture = true;
	std::cout << argv[0];
	auto font_sizes = { 80, 60, 50, 40, 30, 20, 10 };
	std::map<int, uni_text::UniText> yahei;
	for (auto sz : font_sizes) {
		yahei.emplace(sz, uni_text::UniText("Microsoft YaHei Mono.ttf", sz));
	}
	
	fs::path capture_dir = "captures";
	if (!fs::exists(capture_dir)) {
		fs::create_directories(capture_dir);
	}
	int filename_idx = 0;
	std::string filename;
	while (true)
	{
		filename = std::to_string(filename_idx) + ".mp4";
		if (fs::exists(capture_dir / filename)) {
			++filename_idx;
		}
		else {
			break;
		}
	}
	
	int playback_speed = 1;
	int play_back_count = 0;
	VideoWriter cap((capture_dir / filename).string(), VideoWriter::fourcc('M', 'P', 'E', 'G'), 60, {900, 600});
	auto render_frame = [&enable_capture , &cap, &play_back_count, &playback_speed](const Mat& frame, int key) {
		cvui::imshow(MainWindow, frame);
		++play_back_count;
		if (play_back_count % playback_speed == 0) {
			if (enable_capture) {
				cap.write(frame);
			}
			
			play_back_count = 0;
		}
		if (key == ' ') {
			auto copy = frame.clone();
			copy *= 0.7;
			rectangle(copy, Rect{ 0, 0, 400, 50 }, { 10, 10, 10 }, -1);
			putText(copy, "Paused, Press Space to continue", { 15, 25 }, FONT_HERSHEY_DUPLEX, 0.65, { 255, 244, 222 });
			cvui::imshow(MainWindow, copy);
			while (waitKey(30) != ' ')
			{
				++play_back_count;
				if (play_back_count % playback_speed == 0) {
					if (enable_capture) {
						cap.write(frame);
					}
					play_back_count = 0;
				}
			}
		}
	};

	Mat canvas(600, 900, CV_8UC(3));
	centralGradient(canvas, 100, 40, 20);

	namedWindow(MainWindow);
	cvui::init(MainWindow);

	yahei[40].PutText(canvas, u8"¡÷’—Ïø", { 180, 300 }, { 255, 255, 255 });
	yahei[20].PutText(canvas, u8"3170105728", { 186, 335 }, { 255, 255, 255 });

	auto avatar = imread("000.jpg");
	resize(avatar, avatar, {}, 0.6, 0.6);
	auto padding = Size{ 250, 200 };
	auto base = canvas.size() - avatar.size();
	base -= padding;
	avatar.copyTo(canvas({ base.width, base.height, avatar.cols, avatar.rows }));
	

	for (int i = 0; i < 100; ++i) {
		int k = waitKey(20);
		if (k == 27) {
			break;
		}
		render_frame(canvas, k);
	}
	
	for (int i = 0; i < 50; ++i) {
		auto u = i > 25 ? 9 : 5;
		GaussianBlur(canvas, canvas, { u, u }, i * 40);
		int k = waitKey(20);
		if (k == 27) {
			break;
		}
		render_frame(canvas, k);
	}
	
	{
		Mat target(600, 900, CV_8UC(3));
		Chalkboard::green(target);
		
		for (int i = 0; i < 20; ++i) {
			double p = i / 50.0;
			canvas = canvas * (1-p) + target * p;
			int k = waitKey(30);
			if (k == 27) {
				break;
			}
			render_frame(canvas, k);
		}
	}


	Chalkboard::green(canvas);
	MousePainter painter(canvas);
	Chalkboard chalkboard(canvas);
	
	painter.enable(false);

	setMouseCallback(MainWindow, &MousePainter::dispatch, &painter);

	painter.enable(true);
	while (true)
	{
		constexpr int beg = 80;
		constexpr int inc = 40;
		if (cvui::button(canvas, 10, beg - inc, playback_speed == 1 ? "1x speed" : "3x speed")) {
			if (playback_speed == 1) {
				playback_speed = 3;
			}
			else {
				playback_speed = 1;
			}
		}
		if (cvui::button(canvas, 10, beg, "Erase!")) {
			chalkboard.erase(render_frame, Rect(Point{ 80, 100 }, Point{ 880, 550 }));
        }
		if (cvui::button(canvas, 10, beg + inc, "White")) {
			painter.set_color({ 242, 252, 248 });
		}
		if (cvui::button(canvas, 10, beg + 2*inc, "Blue")) {
			painter.set_color({ 208, 175, 94 });
		}
		if (cvui::button(canvas, 10, beg + 3 * inc, "Red")) {
			painter.set_color({ 131, 128, 223 });
		}
		if (cvui::button(canvas, 10, beg + 4 * inc, "Yellow")) {
			painter.set_color({ 112, 198, 222 });
		}
		if (cvui::button(canvas, 10, beg + 5 * inc, "Orange")) {
			painter.set_color({ 174, 202, 248 });
		}
		render_frame(canvas, -1);
		auto key = waitKey(20);
		if (key == 27) { // esc
			break;
		}
		else if (key == ' ') {
			painter.enable(false);
			render_frame(canvas, key);
			painter.enable(true);
		}
	}

	Fold(canvas, render_frame);
	CharacterCloud(canvas, 600, 900, yahei, render_frame);
	cap.release();
	waitKey(0);
	return 0;
}

#endif // HW_1
#if 0
using u32 = int;
using u64 = int;
using pid_t = int;
using kuid_t = int;
using kgid_t = int;
using uid_t = int;
using gid_t = int;
using mqd_t = int;
using umode_t = int;

/* The per-task audit context. */
struct audit_context {
	int		    dummy;	/* must be the first element */
	int		    in_syscall;	/* 1 if task is in a syscall */
	enum audit_state    state, current_state;
	unsigned int	    serial;     /* serial number for record */
	int		    major;      /* syscall number */
	struct timespec	    ctime;      /* time of syscall entry */
	unsigned long	    argv[4];    /* syscall arguments */
	long		    return_code;/* syscall return code */
	u64		    prio;
	int		    return_valid; /* return code is valid */
	/*
	 * The names_list is the list of all audit_names collected during this
	 * syscall.  The first AUDIT_NAMES entries in the names_list will
	 * actually be from the preallocated_names array for performance
	 * reasons.  Except during allocation they should never be referenced
	 * through the preallocated_names array and should only be found/used
	 * by running the names_list.
	 */
	struct audit_names  preallocated_names[AUDIT_NAMES];
	int		    name_count; /* total records in names_list */
	struct list_head    names_list;	/* struct audit_names->list anchor */
	char* filterkey;	/* key for rule that triggered record */
	struct path	    pwd;
	struct audit_aux_data* aux;
	struct audit_aux_data* aux_pids;
	struct sockaddr_storage* sockaddr;
	size_t sockaddr_len;
	/* Save things to print about task_struct */
	pid_t		    pid, ppid;
	kuid_t		    uid, euid, suid, fsuid;
	kgid_t		    gid, egid, sgid, fsgid;
	unsigned long	    personality;
	int		    arch;

	pid_t		    target_pid;
	kuid_t		    target_auid;
	kuid_t		    target_uid;
	unsigned int	    target_sessionid;
	u32		    target_sid;
	char		    target_comm[TASK_COMM_LEN];

	struct audit_tree_refs* trees, * first_trees;
	struct list_head killed_trees;
	int tree_count;

	int type;
	union {
		struct {
			int nargs;
			long args[6];
		} socketcall;
		struct {
			kuid_t			uid;
			kgid_t			gid;
			umode_t			mode;
			u32			osid;
			int			has_perm;
			uid_t			perm_uid;
			gid_t			perm_gid;
			umode_t			perm_mode;
			unsigned long		qbytes;
		} ipc;
		struct {
			mqd_t			mqdes;
			struct mq_attr		mqstat;
		} mq_getsetattr;
		struct {
			mqd_t			mqdes;
			int			sigev_signo;
		} mq_notify;
		struct {
			mqd_t			mqdes;
			size_t			msg_len;
			unsigned int		msg_prio;
			struct timespec		abs_timeout;
		} mq_sendrecv;
		struct {
			int			oflag;
			umode_t			mode;
			struct mq_attr		attr;
		} mq_open;
		struct {
			pid_t			pid;
			struct audit_cap_data	cap;
		} capset;
		struct {
			int			fd;
			int			flags;
		} mmap;
		struct {
			int			argc;
		} execve;
	};
	int fds[2];
	struct audit_proctitle proctitle;
};

using sigset_t = int;

/* Signal handlers: */
struct signal_struct* signal;
struct sighand_struct* sighand;
sigset_t			blocked;
sigset_t			real_blocked;
/* Restored if set_restore_sigmask() was used: */
sigset_t			saved_sigmask;
struct sigpending		pending;
unsigned long			sas_ss_sp;
size_t				sas_ss_size;
unsigned int			sas_ss_flags;

#ifdef CONFIG_CPUSETS
/* Protected by ->alloc_lock: */
nodemask_t			mems_allowed;
/* Seqence number to catch updates: */
seqcount_t			mems_allowed_seq;
int				cpuset_mem_spread_rotor;
int				cpuset_slab_spread_rotor;
#endif

#endif