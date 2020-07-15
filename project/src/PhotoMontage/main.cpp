// PhotoMontage.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include "PhotoMontage.h"

#include "utils.h"

#define CVUI_IMPLEMENTATION
#include "cvgui.h"
#include "cvui.h"

#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;
/*///请在此处读入N 张照片,例子如下:
	Images.push_back(cv::imread("0.JPG"));
	Images.push_back(cv::imread("1.JPG"));
	Images.push_back(cv::imread("2.JPG"));
	Images.push_back(cv::imread("3.JPG"));
	//Images.push_back(cv::imread("4.JPG"));
	
	///请在此处读入N 张Label,注意这些label里面,白色的代表用户的笔触,黑色的代表背景,例子如下:
	std::vector<cv::Mat> Labels;
	Labels.push_back(cv::imread("0.BMP"));
	Labels.push_back(cv::imread("1.BMP"));
	Labels.push_back(cv::imread("2.BMP"));
	Labels.push_back(cv::imread("3.BMP"));
	//Labels.push_back(cv::imread("4.BMP"));*/

struct ThumbInfo
{
	

};


cv::Scalar hsv2rgb(cv::Scalar src) {
#define FUNC_TUPLE_ASSIGN_3(h, s, v) { A=h; B=s; C=v;}
	double H, S, V, A, B, C;
	int Hi;
	double f, p, q, t;

		H = src[0];
		S = src[1];
		V = src[2];
		Hi = (int)(H / 60) % 6;
		f = H / 60 - Hi;
		p = V * (1 - S);
		q = V * (1 - f * S);
		t = V * (1 - (1 - f) * S);
		if (Hi == 0)
			FUNC_TUPLE_ASSIGN_3(V, t, p) //its important that NEVER add colon!!!!!!
		else if (Hi == 1)
			FUNC_TUPLE_ASSIGN_3(q, V, p)
		else if (Hi == 2)
			FUNC_TUPLE_ASSIGN_3(p, V, t)
		else if (Hi == 3)
			FUNC_TUPLE_ASSIGN_3(p, q, V)
		else if (Hi == 4)
			FUNC_TUPLE_ASSIGN_3(t, p, V)
		else
			FUNC_TUPLE_ASSIGN_3(V, p, q)
		
		return cv::Scalar(A, B, C);
}

void CompositeLabel(cv::Mat a, cv::Mat label)
{
	a.forEach<cv::Vec3b>([&label](cv::Vec3b& nv, const int* idxs) {
		auto v = label.at<cv::Vec3b>(idxs[0], idxs[1]);

		if (v[0] != 0 || v[1] != 0 || v[2] != 0) {
			nv = v;
		}
	});
}

void CompositeLabel(cv::Mat a, cv::Mat label, cv::Scalar color)
{
	a.forEach<cv::Vec3b>([&label, color](cv::Vec3b& nv, const int* idxs) {
		auto v = label.at<cv::Vec3b>(idxs[0], idxs[1]);

		if (v[0] != 0 || v[1] != 0 || v[2] != 0) {
			nv[0] = color[0];
			nv[1] = color[1];
			nv[2] = color[2];
		}
	});
}

void ConstructLabelForPhotoMonatage(cv::Mat& res, cv::Mat label, int id)
{
	label.forEach<cv::Vec3b>([&res, id](cv::Vec3b& v, const int* idxs) {
		if (v[0] != 0 || v[1] != 0 || v[2] != 0) {
			res.at<uchar>(idxs[0], idxs[1]) = id;
		}
	});
}


int _tmain(int argc, _TCHAR* argv[])
{
	// auto img = cv::imread("3.BMP");
	// int q = img.type();
	// constexpr int qq = CV_8UC3;
	//RunTest();

	std::string window_name = "Photomontage";

	cv::namedWindow(window_name, cv::WindowFlags::WINDOW_AUTOSIZE);
	cvui::init(window_name);

	cvgui::Input ipt;
	ipt.Init(window_name);

	auto cvui_context = &cvui::internal::gContexts[window_name];
	ipt.RegisterMouseCallback([cvui_context](auto e, int x, int y, int flags) {
		cvui::handleMouse(e, x, y, flags, cvui_context);
	});

	std::vector<cv::Mat> Images;
	std::vector<cv::Mat> Labels;
	std::vector<cv::Mat> thumbs;
	std::vector<cv::Mat> thumbs_labels;
	std::vector<ThumbInfo> thumb_infos;
	std::vector<int> labeled;
	std::vector<cv::Scalar> label_color;
	int gallery_selected = -1;
	int gallery_hovered = -1;
	int gallery_total_height = 0;
	
	int gallery_padding = 5;
	bool preview_mode = false;
	bool displaying_result = false;

	cv::Rect canvas_size(0, 0, 1200, 800);
	cv::Mat canvas(canvas_size.size(), CV_8UC3);

	cv::Mat result;
	cv::Mat result_lables;
	cv::Mat resul_labels_colored;
	cv::Mat result_brush;

	auto objectives = PhotoMontage::kMatchColors;

	

	bool showing_result_image = true;
	bool showing_result_label = false;
	bool showing_result_brush = false;

	bool last_showing_result_image = showing_result_image;
	bool last_showing_result_label = showing_result_label;
	bool last_showing_result_brush = showing_result_brush;
	bool show_popup = false;
	int num_iterations = 50;
	bool use_fast_method = true;

	cv::Rect area_gallery(
		canvas_size.width / 5 * 4,
		0,
		canvas_size.width / 5 * 1,
		canvas_size.height
	);

	cv::Rect area_button(
		0,
		0,
		canvas_size.width / 7 * 1,
		canvas_size.height
	);
	
	int padding = 10;
	cv::Rect area_paint(
		area_button.width + padding,
		padding,
		area_gallery.x  - padding - area_button.width,
		canvas_size.height - padding
	);

	cv::Rect area_iamge;

	int selected = 0;

	std::string dir = "./";
#define rgb(r, g, b) { b, g, r },
	std::vector<cv::Scalar> presets = {
			{ 92, 79, 242 },
			{ 94, 164, 251 },
			{ 139, 83, 123 },
			{ 226, 200, 2 },
			{ 217, 107, 1 },
			{ 170, 210, 164 },
			{ 211, 199, 119 },
			{ 172, 85, 181 },
			{ 39, 195, 134 },
			{ 246, 116, 200 },
			{ 7, 113, 230 },
			{ 60, 149, 0 },
			rgb(255, 187, 221)
			rgb(170, 221, 187)
			rgb(170, 170, 255)
			rgb(125, 140, 142)
			rgb(255, 235, 59)
			rgb(0, 188, 212)
			rgb(233, 30, 99)
			rgb(0, 150, 136)
			rgb(46, 161, 226)
			rgb(211, 236, 129)
			rgb(254, 187, 119)
			rgb(205, 236, 133)
			rgb(139, 215, 194)
			rgb(55, 201, 148)
			rgb(179, 222, 56)
			rgb(93, 242, 251)
			rgb(229, 103, 74)
	};
#undef rgb
	auto GetColorFromId = [&presets](int id) {
		
		if (id < presets.size()) return presets[id];
		auto res =  hsv2rgb(cv::Scalar((id * 30) % 360, 255 - 60 * (id / 5), 255 / (id / 17 + 1)));
		int b = res[0];
		int g = res[1];
		b <<= 4;
		g <<= 4;
		b &= 0x7;
		g &= 0x7;
		b |= id & 0x7;
		g |= (id >> 4) & 0x7;

		res[0] = b;
		res[1] = g;
		res /= 255;
		presets.push_back(res);

		// return res;
		return cv::Scalar(
			rand() % 255,
			rand() % 255,
			rand() % 255
		);
	};

	// 初始化 Gallery
	// ----------------------------------------

	// 扫描所有文件
	for (auto& file : fs::directory_iterator(dir))
	{
		auto mat = cv::imread(file.path().string());
		if (!mat.empty())
		{
			int need_resize = 0;
			int width = mat.cols;
			int height = mat.rows;

			if (width > area_paint.width - 10)
			{
				need_resize = 1;
				height = double(area_paint.width - 10) / width * height;
				width = area_paint.width - 10;
			}

			if (height > area_paint.height - 10)
			{
				need_resize = 1;
				width = double(area_paint.height - 10) / height * width;
				height = area_paint.height - 10;
			}

			if (need_resize)
			{
				cv::resize(mat, mat, cv::Size(width, height));
			}

			Images.push_back(mat);
			labeled.push_back(0);
			Labels.push_back(cv::Mat(mat.size(), CV_8UC3));
			Labels.back() = 0;
			label_color.push_back(GetColorFromId(Images.size() - 1));
		}
	}

	// 创建缩略图
	for (auto& img : Images)
	{
		cv::Mat thumb;
		int width = area_gallery.size().width;
		int height = double(img.rows) / img.cols * width;
		cv::resize(img, thumb, cv::Size(width, height));
		thumb.convertTo(thumb, CV_8UC3);
		thumbs.push_back(thumb);
		auto l = thumb.clone();
		l = 0;
		thumbs_labels.push_back(l);
		gallery_total_height += height;
	}

	auto RenderGallery = [&] (int& y) {
		int ny = 0;
		if (y < 0) {
			y = 0;
		}
		if (gallery_total_height - y - area_gallery.height < 0)
		{
			y = gallery_total_height - area_gallery.height-1;
		}
		auto& g = area_gallery;
		canvas(area_gallery) = 0;
		int cnt = 0;


		for (auto& thumb : thumbs)
		{
			++cnt;
			if (ny + thumb.rows < y) {
				ny += gallery_padding;
				ny += thumb.rows;
				continue;
			}
			else if (ny - y > g.height) {
				break;
			}

			cv::Mat t;
			if (gallery_selected == cnt - 1) {
				t = thumb.clone();
			}
			else if (gallery_hovered == cnt - 1)
			{
				t = thumb.clone() * 0.8 + cv::Scalar(255, 0, 0) * 0.3;
			}
			else {
				t = thumb.clone() * 0.7;
			}

			if (labeled[cnt-1])
			{
				CompositeLabel(t, thumbs_labels[cnt-1]);
			}

			if (ny < y)
			{
				auto height = thumb.rows - (y - ny);
				if (height > 0) {
					t(cv::Rect(0, (y - ny), thumb.cols, height)).copyTo(
						canvas(cv::Rect(g.x, g.y, thumb.cols, height))
					);
				}
			}
			else if ((ny - y) + thumb.rows > g.height)
			{
				auto height = g.height - (ny-y) - 1;
				if (height <= 0) break;
				t(cv::Rect(0, 0, thumb.cols, height)).copyTo(
					canvas(cv::Rect(g.x, g.y + ny - y, thumb.cols, height))
				);
			}
			else {
				t.copyTo(canvas(cv::Rect(g.x, g.y + ny-y, thumb.cols, thumb.rows)));
			}

			ny += gallery_padding;
			ny += thumb.rows;
		}

	};

	// 计算选中的图片
	auto GalleryHandleClick = [&](int y) -> int
	{
		int last_select = gallery_selected >= 0 || gallery_hovered >= 0;
		gallery_hovered = -1;
		gallery_selected = -1;

		if (!ipt.MouseOver(area_gallery))
		{
			return last_select;
		}
		
		int selected = -1;
		int posy = y + ipt.mouse.y;
		int h = 0;
		int i = 0;
		for (auto& thumb : thumbs)
		{
			if (h > posy) break;
			h += thumb.rows;
			if (h > posy) {
				selected = i;
				break;
			}
			h += gallery_padding;
			++i;
		}

		if (selected < 0) {
			return last_select;
		}
		
		// click
		if (ipt.IsMouseDown(cvgui::Input::MouseLeft))
		{
			gallery_selected = selected;
		}
		// hovering
		else if(ipt.IsAllMouseUp())
		{
			gallery_hovered = selected;
		}

		
		return 1;
	};

	
	auto paint_thickness = 3;

	// 画笔工具:
	auto DoPaint = [&](int id, cv::Mat Label)
	{
		auto mpos = cv::Point(ipt.mouse.x, ipt.mouse.y);
		auto lastmpos = cv::Point(ipt.last_mouse.x, ipt.last_mouse.y);
		auto pos = cv::Point(ipt.mouse.x - area_paint.x, ipt.mouse.y - area_paint.y);
		auto lastpos = cv::Point(ipt.last_mouse.x - area_paint.x, ipt.last_mouse.y - area_paint.y);
		//if (Labels[id].cols <= pos.x || Labels[id].rows <= pos.y)
		//{
		//	return;
		//}
		if (!area_iamge.contains(lastmpos))
		{
			return;
		}
		

		//if (pos.y > Labels[id].rows || pos.x > Labels[id].cols) return;
		//if (lastmpos.x > Labels[id].rows || lastmpos.y > Labels[id].cols) return;

		cv::line(Labels[id], lastpos, pos, label_color[id], paint_thickness);
		cv::line(Label, lastpos, pos, label_color[id], paint_thickness);
		cv::line(canvas, lastmpos, mpos, label_color[id], paint_thickness);
		labeled[id] = 1;
		
		/*
		cv::circle(Labels[id], pos, 3, label_color[id]);
		cv::circle(Label, pos, 3, label_color[id]);
		cv::circle(canvas, pos, 3, label_color[id]);
		*/
	};

	// 用户选择新的页面
	auto UpdateSelect = [&](int last_id, int id, cv::Mat Label)
	{
		displaying_result = false;

		canvas(area_paint) = 0;
		auto& p = area_paint;
		auto& im = Images[id];

		area_iamge = cv::Rect(p.x, p.y, im.cols, im.rows);
		im.copyTo(canvas(cv::Rect(p.x, p.y, im.cols, im.rows)));

		CompositeLabel(canvas(cv::Rect(p.x, p.y, im.cols, im.rows)), Labels[id]);
		if (last_id > 0)
		{
			cv::resize(Labels[last_id], thumbs_labels[last_id], thumbs_labels[last_id].size());
		}
	};

	// 刷新渲染结果
	auto RenderResults = [&]
	{
		if (!displaying_result) return;

		
		cv::Mat composite;
		if (showing_result_image)
		{
			composite = result.clone();
		}

		if (showing_result_label)
		{
			if (composite.empty()) {
				composite = 0.9 * (resul_labels_colored - 100);
			}
			else
			{
				composite = composite * 0.9 - 30 + 0.8 * (resul_labels_colored - 80);
			}
		}

		if (showing_result_brush)
		{
			if (composite.empty()) {
				composite = result_brush.clone();
			}
			else
			{
				CompositeLabel(composite, result_brush);
			}
		}

		if (!composite.empty())
		{
			auto& p = area_paint;
			composite.copyTo(canvas(cv::Rect(p.x, p.y, composite.cols, composite.rows)));
		}
	};

	// Run 按钮
	auto RunPhotoMontage = [&](enum PhotoMontage::Objectives obj)
	{
		int div = preview_mode ? 2 : 1;

		std::vector<cv::Mat> mats;
		cv::Size original_size;
		cv::Size size;

		cv::Mat flabel;
		int k = 0;
		

		std::vector<int> ids;

		for (int i = 0; i < labeled.size(); ++i)
		{
			if (labeled[i])
			{
				if (flabel.empty())
				{
					original_size = cv::Size(Images[i].cols, Images[i].rows);
					size = original_size / div;
					flabel = cv::Mat(size, CV_8SC1);
					flabel = PhotoMontage::undefined;
				}
				else if (original_size != Images[i].size())
				{
					std::cout << "Detected umatched dimension, skipping" << std::endl;
					continue;
				}
				if (preview_mode)
				{
					cv::Mat shrinked;
					cv::resize(Images[i], shrinked, size);
					mats.push_back(shrinked);

					cv::Mat small_label;
					cv::resize(Labels[i], small_label, size);
					ConstructLabelForPhotoMonatage(flabel, small_label, k);
				}
				else {
					mats.push_back(Images[i]);
					ConstructLabelForPhotoMonatage(flabel, Labels[i], k);
				}
				
				ids.push_back(i);
				++k;
			}
		}
		if (ids.size() == 0) {
			return;
		}
		std::cout << ids.size() << " image will be fusioned" << std::endl;
		displaying_result = true;


		PhotoMontage PM;
		PM.fast_init_value = use_fast_method;
		PM.progress = [](double progress, PhotoMontage::Stage stage)
		{

		};

		auto tik = std::chrono::steady_clock::now();
		PM.Run(mats, flabel, obj, show_popup, preview_mode ? (use_fast_method ? 5 : 300) : num_iterations);
		auto tok = std::chrono::steady_clock::now();

		std::cout << "Time used: " << (tok - tik).count() / 1000000 << "ms\n";
		auto [res, res_label] = PM.GetResultAndLabel();
		if (preview_mode)
		{
			if (!use_fast_method)
			{
				res += cv::Scalar(100, 100, 100);
			}
			
			cv::resize(res, res, original_size);
			cv::resize(res_label, res_label, original_size);
		}
		result = res;
		result_lables = res_label;

		cv::Mat res_brush(res_label.size(), CV_8UC3);
		res_brush = 0;

		cv::Mat res_label_color = res_brush.clone();

		for (auto id : ids)
		{
			CompositeLabel(res_brush, Labels[id], presets[id]);
		}
		result_brush = res_brush;

		res_label.forEach<signed char>([&ids, &res_label_color, &presets](signed char& v, const int* pos) {
			auto& q = res_label_color.at<cv::Vec3b>(pos[0], pos[1]);
			int id = ids[v];
			auto& color = presets[id];
			q[0] = color[0];
			q[1] = color[1];
			q[2] = color[2];
		});
		resul_labels_colored = res_label_color;

		RenderResults();
	};

	auto RunCleanLabel = [&]()
	{
		for (auto& label : labeled) {
			label = 0;
		}

		for (auto& lebel : Labels)
		{
			lebel = 0;
		}

		for (auto& thumbs_label : thumbs_labels)
		{
			thumbs_label = 0;
		}
	};

	int gallery_scroll = 0;
	int paint_select = -1;

	

	cv::Mat OverlayLabel(Images[0].rows, Images[0].cols,  CV_8UC3);
	
	canvas = 0;
	RenderGallery(gallery_scroll);
	while (true)
	{
		int last_select = gallery_selected;
		if (ipt.Scroll(area_gallery))
		{
			gallery_scroll += -ipt.mouse.scroll_y;
			RenderGallery(gallery_scroll);
		}
		else if (GalleryHandleClick(gallery_scroll))
		{
			RenderGallery(gallery_scroll);
			if (paint_select != gallery_selected && gallery_selected >= 0)
			{
				UpdateSelect(paint_select, gallery_selected, OverlayLabel);
				paint_select = gallery_selected;
				RenderGallery(gallery_scroll);
			}
		}
		
		if (ipt.LeftDrag(area_iamge) && paint_select >= 0)
		{
			labeled[paint_select] = 1;
			DoPaint(paint_select, OverlayLabel);
		}
		
#define RUN(name) { constexpr int id = (int)PhotoMontage::k##name;\
		if (cvui::button(canvas, 5, 10 + 50 * id, 170, 40, #name, 0.4))\
		{\
			printf("Running");\
			RunPhotoMontage(PhotoMontage::k##name);\
		}}
		RUN(MatchColors);
		RUN(MatchGradients);
		RUN(MatchColorAndGradients);
		RUN(MatchColorsAndEdges);
		RUN(MatchLaplacian);


		int y_start = 50;

		if (cvui::button(canvas, 5, y_start + 240, 170, 40, "Reset Label", 0.4))
		{
			printf("Cleaning");
			RunCleanLabel();
			RenderGallery(gallery_scroll);
			UpdateSelect(-1, paint_select, OverlayLabel);
		}
		
		if (cvui::button(canvas, 5, y_start + 290, 170, 40, "Save res.png", 0.4))
		{
			if (!result.empty())
			{
				cv::imwrite("res.png", result);
				cv::imwrite("res_label.png", result_lables);
				cv::imwrite("res_brush.png", resul_labels_colored);
			}
		}

		if (cvui::button(canvas, 5, y_start + 340, 170, 40, "Last Result", 0.4))
		{
			if (!result_brush.empty()) {
				displaying_result = true;
				RenderResults();
			}
		}

		cvui::counter(canvas, 5, y_start + 390, &num_iterations, use_fast_method ? 50 : 500);

		y_start += 130;



		// 是否采用预览模式
		cvui::checkbox(
			canvas,
			5,
			y_start + 290,
			"Preview Mode",
			&preview_mode
			// unsigned int theColor = 0xCECECE
		);

		cvui::checkbox(
			canvas,
			5,
			y_start + 325,
			"Show PopUp",
			&show_popup
			// unsigned int theColor = 0xCECECE
		);

		cvui::checkbox(
			canvas,
			5,
			y_start + 350,
			"Fast init",
			&use_fast_method
			// unsigned int theColor = 0xCECECE
		);

		y_start += 50;

		// 渲染结果的合成选项
		if (displaying_result)
		{
			int deltay = 30;
			bool need_change = false;
			need_change |= cvui::checkbox(
				canvas,
				5,
				y_start + 360 + deltay * 0,
				"Show Result",
				&showing_result_image
				// unsigned int theColor = 0xCECECE
			);

			need_change |= cvui::checkbox(
				canvas,
				5,
				y_start + 360 + deltay * 1,
				"Show Label",
				&showing_result_label
				// unsigned int theColor = 0xCECECE
			);


			need_change |= cvui::checkbox(
				canvas,
				5,
				y_start + 360 + deltay * 2,
				"Show Brush",
				&showing_result_brush
				// unsigned int theColor = 0xCECECE
			);

			if (need_change)
			{
				RenderResults();
			}
		}
		ipt.NextFrame();
		cvui::imshow(window_name, canvas);
		auto key = cv::waitKey(20);
		if (key == 27)
		{
			break;
		}
	}


	/*
	///set all the labels to undefined
	cv::Mat Label(Images[0].rows, Images[0].cols, CV_8SC1);
	Label.setTo(PhotoMontage::undefined);
	///set the labels according to the Labels read
	int height = Images[0].rows;
	int width = Images[0].cols;
	for (int label_idx = 0 ; label_idx < Images.size(); label_idx++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0 ; x < width; x++)
			{
				if (Labels[label_idx].at<cv::Vec3b>(y,x)(0) > 0)
				{
					Label.at<uchar>(y,x) = label_idx;
				}
			}
		}
	}

	///Run photomontage
	PhotoMontage PM;
	PM.Run(Images,Label);
	*/
	return 0;
}

