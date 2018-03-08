// Image_Resizing.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <ilcplex/ilocplex.h>
#include <math.h>
#include <conio.h>
#include <fstream>
#include <algorithm>
#include <queue>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string>
// #include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

// #include <boost/process.hpp>
// #include <process.h>
#include <stdlib.h>
#include <map>

#include <cstdio>
#include <cstdlib>
#include "Segmentation/image.h"
#include "Segmentation/misc.h"
#include "Segmentation/pnmfile.h"
#include "Segmentation/segment-image.h"

using namespace std;
using namespace cv;

class Patch;
class Quad;

int* string2color(string color_string) {
	istringstream f(color_string);
	string bgr[3];

	for (int i = 0; i < 3; ++i)
		getline(f, bgr[i], ' ');

	// return Scalar(stod(bgr[0]), stod(bgr[1]), stod(bgr[2]));
	int color_array[3]{ stoi(bgr[0]), stoi(bgr[1]), stoi(bgr[2]) };
	return color_array;
}

class Quad {
public:
	// Quad(int x, int y, const Mat &segmentation_img, map<string, Patch> &patchs, int w = quad_width, int h = quad_height) {
	Quad(int x, int y, int w = quad_width, int h = quad_height) {
		this->width = w;
		this->height = h;
		this->size = (w + 1) * (h + 1);
		
		for (int i = 0; i < 4; ++i) {
			vertexes[i] = Mat(2, 1, CV_32FC1);
			transformations[i] = Mat(2, 1, CV_32FC1);
			edges[i] = Mat(2, 1, CV_32FC1);
		}

		int step = vertexes[0].step1();
		
		// top left
		float *data = (float *)vertexes[0].data;
		data[0 * step + 0] = x;
		data[1 * step + 0] = y;

		// top right
		data = (float *)vertexes[1].data;
		data[0 * step + 0] = x + w;
		data[1 * step + 0] = y;

		// bottom right
		data = (float *)vertexes[2].data;
		data[0 * step + 0] = x + w;
		data[1 * step + 0] = y + h;

		// bottom left
		data = (float *)vertexes[3].data;
		data[0 * step + 0] = x;
		data[1 * step + 0] = y + h;
		
		/*
		Mat top_left = Mat(2, 1, CV_32FC1), top_right = Mat(2, 1, CV_32FC1), bottom_right = Mat(2, 1, CV_32FC1), bottom_left = Mat(2, 1, CV_32FC1);
		int step = top_left.step;

		float *data = (float *)top_left.data;
		data[0 * step + 0] = x;
		data[1 * step + 0] = y;

		data = (float *)top_right.data;
		data[0 * step + 0] = x + w;
		data[1 * step + 0] = y;

		data = (float *)bottom_right.data;
		data[0 * step + 0] = x + w;
		data[1 * step + 0] = y + h;

		data = (float *)bottom_left.data;
		data[0 * step + 0] = x;
		data[1 * step + 0] = y + h;

		this->vertexes[0] = top_left;
		this->vertexes[1] = top_right;
		this->vertexes[2] = bottom_right;
		this->vertexes[3] = bottom_left;
		*/

		center_vertex = Mat(2, 1, CV_32FC1);
		data = (float *)center_vertex.data;
		data[0 * step + 0] = x + w / 2;
		data[1 * step + 0] = y + h / 2;

		set_boundaries(x, y, w, h);

		// set_patch_color(segmentation_img, patchs);

		// set_id(id);
	}

	/*
	~Quad() {
		// cout << "YO" << endl;
		for (int i = 0; i < 4; ++i) {
			vertexes[i].u->refcount = 0;
			vertexes[i].release();
			transformations[i].u->refcount = 0;
			transformations[i].release();
			edges[i].u->refcount = 0;
			edges[i].release();
		}
	}
	*/

	Point2f get_vertex_point(int index) {
		float *data = (float *)vertexes[index].data;
		int step = vertexes[index].step1();

		return Point2f(data[0 * step + 0], data[1 * step + 0]);
	}

	Point2f get_center_vetex_point() {
		float *data = (float *)center_vertex.data;
		int step = center_vertex.step1();

		return Point2f(data[0 * step + 0], data[1 * step + 0]);
	}

	Mat get_vertex(int index) {
		return vertexes[index];
	}

	Scalar get_patch_color() {
		// cout << patch_color_string << endl;
		int *color = string2color(patch_color_string);
		return Scalar(color[0], color[1], color[2]);
	}

	void set_patch_color_string(string s) {
		patch_color_string = s;
	}

	int* get_boundaries() {
		return boundaries;
	}

	map<string, int> get_color_counter() {
		return color_counter;
	}

	int get_size() {
		return size;
	}

	void set_transformations(const Mat &center_edge22) {
		for (int i = 0; i < 3; ++i) {
			edges[i] = vertexes[i + 1] - vertexes[i];
			transformations[i] = center_edge22 * edges[i];
		}

		edges[3] = vertexes[0] - vertexes[3];
		transformations[3] = center_edge22 * edges[3];
	}

	Mat* get_transformations() {
		return transformations;
	}

	Mat* get_edges() {
		return edges;
	}

	/*
	int get_id() {
	return id;
	}
	*/

	static int quad_width, quad_height;
private:
	void set_boundaries(int x, int y, int w, int h) {
		boundaries[0] = y;
		boundaries[1] = x + w;
		boundaries[2] = y + h;
		boundaries[3] = x;
	}

	/*
	void set_id(int id) {
	this->id = id;
	}
	*/

	map<string, int> color_counter;
	// top-left, top-right, bottom-right, botton-left
	Mat vertexes[4];
	Mat transformations[4];
	Mat edges[4];
	// top, right, bottom, left
	int boundaries[4];

	Mat center_vertex;
	string patch_color_string = "?";
	// Scalar patch_color;
	int width, height, size;
};

class Patch {
public:
	Patch() {
	}

	Patch(Quad &q) {
		add_quad(q);
	}

	/*
	Patch(Quad *q) {
	add_quad(q);
	}
	*/

	void add_quad(Quad &q) {
		quads.push_back(q);
	}


	vector<Quad> get_quads() {
		return quads;
	}

	void calculate_transformations() {
		int quads_size = quads.size();
		center_index = rand() % quads_size;

		// the center edge would be the top edge of quad
		center_edge = quads[center_index].get_vertex(1) - quads[center_index].get_vertex(0);

		center_edge22 = Mat(2, 2, CV_32FC1);
		float *center_edge_data = (float *)center_edge.data;
		float *center_edge22_data = (float *)center_edge22.data;
		int step12 = center_edge.step1(), step22 = center_edge22.step1();
		float cx = center_edge_data[0 * step12 + 0], cy = center_edge_data[1 * step12 + 0];

		center_edge22_data[0 * step22 + 0] = cx;
		center_edge22_data[1 * step22 + 1] = -cx;
		center_edge22_data[0 * step22 + 1] = cy;
		center_edge22_data[1 * step22 + 0] = cy;
		center_edge22 = center_edge22.inv();

		for (int i = 0; i < quads_size; ++i) {
			quads[i].set_transformations(center_edge22);
		}
	}

	void set_significance(float s) {
		significance = s / 255;
	}

	float get_significance() {
		return significance;
	}

	int get_center_index() {
		return center_index;
	}

private:
	vector<Quad> quads;
	Mat center_edge, center_edge22;
	float significance;
	int center_index;
};

void set_quad_patch_color(Quad &q, const Mat &segmentation_img, map<string, Patch> &patchs) {
	uint8_t *data = segmentation_img.data;
	int *boundaries = q.get_boundaries();
	map<string, int> color_counter;
	int step = segmentation_img.step1();

	// to count the number of colors
	for (int y = boundaries[0]; y <= boundaries[2]; ++y) {
		for (int x = boundaries[3]; x <= boundaries[1]; ++x) {
			int index = y * step + x * 3;

			string color = to_string(data[index]) + " ";
			color += to_string(data[index + 1]) + " ";
			color += to_string(data[index + 2]);

			// cout << "color = " << color << " x, y = " << x << ", " << y << endl;

			map<string, int>::iterator color_counter_find_iter = color_counter.find(color);
			if (color_counter_find_iter != color_counter.end()) {
				// found
				++color_counter[color];
			}
			else {
				// create a new color counter
				// cout << "create a counter indexed with color " << color << endl;
				color_counter[color] = 1;
			}
		}
	}

	// to select the color which has the max number
	string patch_color_string = "";
	int max_number = -1, color_counter_sum = 0;
	for (map<string, int>::iterator color_counter_iter = color_counter.begin(); color_counter_iter != color_counter.end(); ++color_counter_iter) {
		// max_number = counter_iter->second > max_number ? counter_iter->second : max_number;

		// cout << counter_iter->first << " => " << counter_iter->second << endl;
		// counter_sum += counter_iter->second;

		if (color_counter_iter->second > max_number) {
			max_number = color_counter_iter->second;
			patch_color_string = color_counter_iter->first;
		}
	}

	// cout << "the sum of counter is " << counter_sum << endl;
	// cout << "this quad belongs to color " << patch_color_string << endl;

	q.set_patch_color_string(patch_color_string);

	map<string, Patch>::iterator patchs_find_iter = patchs.find(patch_color_string);
	if (patchs_find_iter != patchs.end()) {
		patchs[patch_color_string].add_quad(q);
	}
	else {
		patchs[patch_color_string] = Patch(q);
	}

	// string color_list = patch_color_string;
	//patch_color = Scalar()
}

int Quad::quad_width = 0;
int Quad::quad_height = 0;

int main()
{
	int terminal_variable;

	/*
	float tes = (float)(6) / 4 + 1;
	cout << tes << endl;
	cin >> terminal_variable;
	return 0;
	*/

	string original_img_name, original_img_type;

	cout << "please input the image name" << endl;
	cin >> original_img_name;

	cout << "please input the image type(.jpg)" << endl;
	cin >> original_img_type;

	cout << "original_img_name = " << original_img_name << endl;
	cout << "original_img_type = " << original_img_type << endl;

	// in the "Image_Resizing\Image_Resizing"
	Mat img = imread(original_img_name + original_img_type, CV_LOAD_IMAGE_COLOR);

	if (!img.data) {
		cout << "image GG" << endl;
		cin >> terminal_variable;
		return -1;
	}

	// namedWindow("Display", CV_WINDOW_NORMAL);
	// imshow("Original Image", img);

	int img_rows = img.rows, img_cols = img.cols, new_img_rows, new_img_cols;
	cout << "original img rows = " << img_rows << " and cols = " << img_cols << endl;
	cout << "please input the new rows and cols" << endl;
	cin >> new_img_rows >> new_img_cols;
	cout << "new_img_rows = " << new_img_rows << " and new_img_cols = " << new_img_cols << endl;

	/* Felzenszwalb-Segmentation start */
	string ppm = ".ppm", segmentation_img_name = "segmentation_" + original_img_name;
	string img_ppm_path = original_img_name + ppm, segmentation_img_ppm_path = segmentation_img_name + ppm;

	/*
	imwrite(img_ppm_path, img);

	float sigma = 0.5;
	float k = 500.;
	int min_size = 20;
	int num_ccs;

	printf("loading input image.\n");
	image<rgb> *input = loadPPM(img_ppm_path.c_str());

	printf("processing\n");
	image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs);
	savePPM(seg, segmentation_img_ppm_path.c_str());
	*/

	Mat segmentation_img = imread(segmentation_img_ppm_path, CV_LOAD_IMAGE_COLOR);
	imwrite(segmentation_img_name + original_img_type, segmentation_img);
	// segmentation_img = imread(segmentation_img_name + original_img_type, CV_LOAD_IMAGE_COLOR);
	// imshow("Segmentation image", segmentation_img);
	// waitKey(0);

	/* Felzenszwalb-Segmentation end */

	/* Saliency start */
	/*
	cout << "saliency start" << endl;
	// string cmd_instruction = "cd Saliency\nmatlab -nodisplay -nojvm -nodesktop -nosplash -r -wait \"run_saliency ../" + original_img_name + original_img_type + "\"";
	string cmd_instruction = "matlab -nodisplay -nojvm -nodesktop -nosplash -r -wait \"run_saliency " + original_img_name + original_img_type + "\"";
	// cout << system(cmd_instruction.c_str()) << endl;
	// cout << system("dir") << endl;

	system(cmd_instruction.c_str());
	//system("dir");
	cout << "saliency end" << endl;
	*/

	string saliency_img_path = "Output/" + original_img_name + "_SaliencyMap.jpg";
	// Mat small_saliency_img = imread(saliency_img_path, CV_LOAD_IMAGE_COLOR), saliency_img;
	// Mat saliency_img = imread("saliency_" + original_img_name + ".jpg", CV_LOAD_IMAGE_COLOR);
	Mat saliency_img = imread("saliency_" + original_img_name + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
	// resize(small_saliency_img, saliency_img, Size(img.cols, img.rows));
	// imshow("Saliency", saliency_img);
	// imwrite("saliency_" + original_img_name + ".jpg", saliency_img);
	// waitKey(0);

	/* Saliency end */

	/*  create the quads start */
	map<string, Patch> patchs;
	vector<Quad> all_quads;
	uint8_t *segmentation_img_data = segmentation_img.data;
	int quad_width = 30, quad_height = 30;
	// "rows - quad_height" to prevent from index out of range
	int rows = segmentation_img.rows - quad_height, cols = segmentation_img.cols - quad_width, step = segmentation_img.step1();
	int x, y;

	Quad::quad_width = quad_width;
	Quad::quad_height = quad_height;

	cout << "create the quads" << endl;
	for (y = 0; y < rows; y += quad_height) {
		for (x = 0; x < cols; x += quad_width) {
			// top-left, top-right, bottom-right, botton-left

			// quads.push_back(Quad(top_left, top_right, bottom_right, bottom_left, segmentation_img, patchs));
			Quad q = Quad(x, y);
			set_quad_patch_color(q, segmentation_img, patchs);
			all_quads.push_back(q);
		}

		// the last x quad of this y
		int w = cols - 1 + quad_width - x;
		if (w) {
			Quad q = Quad(x, y, w, quad_height);
			set_quad_patch_color(q, segmentation_img, patchs);
			all_quads.push_back(q);
		}

		// cout << "line " << y << " is completed" << endl;
	}

	// the last y quad
	int h = rows - 1 + quad_height - y;
	for (x = 0; x < cols; x += quad_width) {
		// top-left, top-right, bottom-right, botton-left
		Quad q = Quad(x, y, quad_width, h);
		set_quad_patch_color(q, segmentation_img, patchs);
		all_quads.push_back(q);
	}

	// the last x quad of this y
	int w = cols - 1 + quad_width - x;
	if (w) {
		Quad q = Quad(x, y, w, h);
		set_quad_patch_color(q, segmentation_img, patchs);
		all_quads.push_back(q);
	}

	// cout << "line " << y << " is completed" << endl;

	// cout << "the last id = " << id << endl;

	cout << "creating the quads is completed" << endl;

	// to draw the quad in img
	Mat quad_img = Mat(img_rows, img_cols, CV_8UC3);
	Scalar white_color(255, 255, 255);
	int radius = 1;

	img.copyTo(quad_img);

	// cout << "start to draw" << endl;
	for each(Quad q in all_quads) {
		for (int i = 0; i < 4; ++i)
			circle(quad_img, q.get_vertex_point(i), radius, white_color, 1, 8, 0);
		circle(quad_img, q.get_center_vetex_point(), radius, q.get_patch_color(), 1, 8, 0);
	}

	// cout << "end to draw" << endl;

	// imshow("Quad Image", quad_img);
	imwrite("quad_" + original_img_name + ".jpg", quad_img);
	// waitKey(0);

	/* create the quads end*/

	/* create the transformations for every quad */

	srand(time(NULL));
	for (map<string, Patch>::iterator all_iter = patchs.begin(); all_iter != patchs.end(); ++all_iter) {
		all_iter->second.calculate_transformations();
	}

	/* create the transformations for every quad end */

	/*
	for (map<string, Patch>::iterator all_iter = patchs.begin(); all_iter != patchs.end(); ++all_iter) {
	vector<Quad> quads = all_iter->second.get_quads();
	int quads_size = quads.size();
	int test = quads_size > 10 ? 10 : quads_size;

	for (int i = 0; i < test; ++i) {
	Mat *edges = quads[i].get_edges(), *transformations = quads[i].get_transformations();

	cout << "edges: ";
	for (int j = 0; j < 4; ++j) {
	cout << edges[j];
	}
	cout << endl;

	cout << "transformations: ";
	for (int j = 0; j < 4; ++j) {
	cout << transformations[j];
	}
	cout << endl;
	}
	}
	*/

	/* create the significant image */

	// cout << "num_ccs" << num_ccs << endl;
	cout << "patchs size = " << patchs.size() << endl;

	// Mat significant_img = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat significant_img = Mat::zeros(img_rows, img_cols, CV_8UC1);
	uint8_t *saliency_img_data = saliency_img.data, *significant_img_data = significant_img.data;
	step = saliency_img.step1();

	for (map<string, Patch>::iterator all_iter = patchs.begin(); all_iter != patchs.end(); ++all_iter) {
		// it -> first => index it -> second => value
		vector<Quad> quads = all_iter->second.get_quads();
		int quads_size = quads.size();
		// unsigned int sum_bgr[3]{ 0, 0, 0 }, total_color_number = 0;
		unsigned int sum = 0, total_color_number = 0;
		map<string, int> patch_color_counter;

		// the problem of duplicate boundaries??
		for (int i = 0; i < quads_size; ++i) {
			int *boundaries = quads[i].get_boundaries();

			// cout << boundaries[0] << " " << boundaries[1] << " " << boundaries[2] << " " << boundaries[3] << " " << endl;
			// cout << quads[i].get_center_vetex();

			for (int y = boundaries[0]; y <= boundaries[2]; ++y) {
				for (int x = boundaries[3]; x <= boundaries[1]; ++x) {
					sum += saliency_img_data[y * step + x];

					//int index = y * step + x * 3;
					//for (int j = 0; j < 3; ++j) {
					// cout << unsigned(saliency_img_data[index + j]) << " ";
					//sum_bgr[j] += saliency_img_data[index + j];
					//}

					// cout << endl;
				}
			}

			total_color_number += quads[i].get_size();
		}


		//for (int i = 0; i < 3; ++i) {
		//sum_bgr[i] /= total_color_number;
		//cout << sum_bgr[i] << " ";
		//}

		//cout << endl;

		sum /= total_color_number;
		// cout << "sum = " << sum << endl;

		// cout << "total_color_number = " << total_color_number << endl;

		for (int i = 0; i < quads_size; ++i) {
			int *boundaries = quads[i].get_boundaries();

			for (int y = boundaries[0]; y <= boundaries[2]; ++y) {
				for (int x = boundaries[3]; x <= boundaries[1]; ++x) {
					significant_img_data[y * step + x] = sum;

					//int index = y * step + x * 3;
					//for (int j = 0; j < 3; ++j)
					//significant_img_data[index + j] = sum_bgr[j];
				}
			}
		}
	}

	// cout << "output significant image" << endl;

	// imshow("Significant Image", significant_img);
	imwrite("significant_" + original_img_name + original_img_type, significant_img);
	// waitKey(0);

	// Mat normalized_significant_img = Mat::zeros(img.rows, img.cols, CV_32FC1);
	Mat normalized_significant_img = Mat::zeros(img_rows, img_cols, CV_8UC1);
	normalize(significant_img, normalized_significant_img, 25, 255, NORM_MINMAX);
	uint8_t *normalized_significant_img_data = normalized_significant_img.data;
	step = normalized_significant_img.step1();

	/*
	for (int i = 0; i < 10; ++i) {
	cout << unsigned(normalized_significant_img_data[i]) << " ";
	}

	cout << endl;
	*/

	// imshow("Normailized significant image", normalized_significant_img);
	imwrite("normalized_significant_" + original_img_name + original_img_type, normalized_significant_img);
	// waitKey(0);

	// to give the significance value
	for (map<string, Patch>::iterator all_iter = patchs.begin(); all_iter != patchs.end(); ++all_iter) {
		Point2f p = all_iter->second.get_quads()[0].get_vertex_point(0);
		// cout << p << endl;
		try {
			all_iter->second.set_significance(normalized_significant_img_data[(int)(p.y) * step + (int)(p.x)]);
		}
		catch (...) {
			cout << "???" << endl;
			return 0;
		}
	}

	/* create the significant image end */

	/* to set the equations */

	IloEnv env;
	IloModel model(env);
	IloNumVarArray vars(env);

	rows = img.rows; cols = img.cols;

	/*
	for (int y = 0; y < rows; ++y) {
	for (int x = 0; x < cols; ++x) {
	// x and y
	vars.add(IloNumVar(env, 0.0, cols));
	vars.add(IloNumVar(env, 0.0, rows));
	}
	}
	*/

	int total_quad_y_size = ceil(float(img_rows - 1) / quad_height) + 1, total_quad_x_size = ceil(float(img_cols - 1) / quad_width) + 1;

	cout << "total_quad_y_size = " << total_quad_y_size << " and total_quad_x_size = " << total_quad_x_size << endl;
	cout << "all_quads size = " << all_quads.size() << endl;

	for (int y = 0; y < total_quad_y_size; ++y)
		for (int x = 0; x < total_quad_x_size; ++x) {
			// x and y
			vars.add(IloNumVar(env, 0.0, new_img_cols));
			vars.add(IloNumVar(env, 0.0, new_img_rows));
		}

	cout << "set the equations" << endl;

	IloExpr expr(env);
	float rows_ratio = (float)(new_img_rows) / img_rows, cols_ratio = (float)(new_img_cols) / img_cols;
	float ST_weight = 5.5, LT_weight = .8, OR_weight = 20.;


	for (map<string, Patch>::iterator all_iter = patchs.begin(); all_iter != patchs.end(); ++all_iter) {
		vector<Quad> quads = all_iter->second.get_quads();
		int quads_size = quads.size(), center_index = all_iter->second.get_center_index();
		Point2f center_point1 = quads[center_index].get_vertex_point(1), center_point0 = quads[center_index].get_vertex_point(0);
		// int vars_center_point1_index = center_point1.y * cols * 2 + center_point1.x * 2, vars_center_point0_index = center_point0.y * cols * 2 + center_point0.x * 2;
		// int vars_center_point1_index = center_quad_id * 2 + center_point1.x * 2, vars_center_point0_index = center_point0.y * cols * 2 + center_point0.x * 2;
		int vars_center_point1_index = ((int)ceil(center_point1.y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(center_point1.x / quad_width)) * 2, vars_center_point0_index = ((int)ceil(center_point0.y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(center_point0.x / quad_width)) * 2;
		float significance = all_iter->second.get_significance();


		for (int i = 0; i < quads_size; ++i) {
			// Mat *edges = quads[i].get_edges(), *transformations = quads[i].get_transformations();
			Mat *transformations = quads[i].get_transformations();
			// int *edge_data, *transformation_data;
			float *transformation_data;
			step = transformations[0].step1();

			int vars_indexes[4];
			// Point2f p0, p1;
			Point2f p0;
			float s, r;

			for (int j = 0; j < 4; ++j) {
				p0 = quads[i].get_vertex_point(j);
				vars_indexes[j] = ((int)ceil(p0.y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(p0.x / quad_width)) * 2;
			}

			for (int j = 0; j < 3; ++j) {
				transformation_data = (float *)transformations[j].data;
				s = transformation_data[0 * step + 0]; r = transformation_data[1 * step + 0];

				// p0 = quads[i].get_vertex_point(j);
				// p1 = quads[i].get_vertex_point(j + 1);
				// vars_index[j] = p0.y * cols * 2 + p0.x * 2;
				// vars_index[j + 1] = p1.y * cols * 2 + p1.x * 2;

				// DST
				expr += ST_weight * significance * IloPower((vars[vars_indexes[j + 1]] - vars[vars_indexes[j]]) - s * (vars[vars_center_point1_index] - vars[vars_center_point0_index]) - r * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]), 2);
				expr += ST_weight * significance * IloPower((vars[vars_indexes[j + 1] + 1] - vars[vars_indexes[j] + 1]) - s * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]) + r * (vars[vars_center_point1_index] - vars[vars_center_point0_index]), 2);

				// DLT
				expr += LT_weight * (1. - significance) * IloPower((vars[vars_indexes[j + 1]] - vars[vars_indexes[j]]) - s * rows_ratio * (vars[vars_center_point1_index] - vars[vars_center_point0_index]) - r * rows_ratio * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]), 2);
				expr += LT_weight * (1. - significance) * IloPower((vars[vars_indexes[j + 1] + 1] - vars[vars_indexes[j] + 1]) - s * cols_ratio * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]) + r * cols_ratio * (vars[vars_center_point1_index] - vars[vars_center_point0_index]), 2);
			}

			transformation_data = (float *)transformations[3].data;

			// p0 = quads[i].get_vertex_point(3);
			// p1 = quads[i].get_vertex_point(0);
			// vars_index0 = p0.y * cols * 2 + p0.x * 2;
			// vars_index1 = p1.y * cols * 2 + p1.x * 2;
			s = transformation_data[0 * step + 0], r = transformation_data[1 * step + 0];

			// DST
			expr += ST_weight * significance * IloPower((vars[vars_indexes[0]] - vars[vars_indexes[3]]) - s * (vars[vars_center_point1_index] - vars[vars_center_point0_index]) - r * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]), 2);
			expr += ST_weight * significance * IloPower((vars[vars_indexes[0] + 1] - vars[vars_indexes[3] + 1]) - s * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]) + r * (vars[vars_center_point1_index] - vars[vars_center_point0_index]), 2);

			// DLT
			expr += LT_weight * (1. - significance) * IloPower((vars[vars_indexes[0]] - vars[vars_indexes[3]]) - s * rows_ratio * (vars[vars_center_point1_index] - vars[vars_center_point0_index]) - r * rows_ratio * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]), 2);
			expr += LT_weight * (1. - significance) * IloPower((vars[vars_indexes[0] + 1] - vars[vars_indexes[3] + 1]) - s * cols_ratio * (vars[vars_center_point1_index + 1] - vars[vars_center_point0_index + 1]) + r * cols_ratio * (vars[vars_center_point1_index] - vars[vars_center_point0_index]), 2);


			// DOR
			expr += OR_weight * (IloPower(vars[vars_indexes[0] + 1] - vars[vars_indexes[1] + 1], 2) + IloPower(vars[vars_indexes[3] + 1] - vars[vars_indexes[2] + 1], 2) + IloPower(vars[vars_indexes[0]] - vars[vars_indexes[3]], 2) + IloPower(vars[vars_indexes[1]] - vars[vars_indexes[2]], 2));
		}
	}

	// boundary constraint
	int last_quad_y = total_quad_y_size - 1, last_quad_x = total_quad_x_size - 1;
	int last_y = new_img_rows - 1, last_x = new_img_cols - 1;
	for (int x = 0; x < total_quad_x_size; ++x) {
		// ((int)ceil(p0.y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(p0.x / quad_width)) * 2
		model.add(vars[0 * total_quad_x_size * 2 + x * 2 + 1] == 0);
		model.add(vars[last_quad_y * total_quad_x_size * 2 + x * 2 + 1] == last_y);
	}

	for (int y = 0; y < total_quad_y_size; ++y) {
		model.add(vars[y * total_quad_x_size * 2 + 0 * 2] == 0);
		model.add(vars[y * total_quad_x_size * 2 + last_quad_x * 2] == last_x);
	}

	model.add(IloMinimize(env, expr));

	cout << "start to solve" << endl;

	IloCplex solver(model);
	if (!solver.solve()) {
		env.error() << "Failed to optimize" << endl;
		return -1;
	}

	IloNumArray answers(env);

	// env.out() << "solution status = " << solver.getStatus() << endl;
	// env.out() << "Solution value = " << solver.getObjective() << endl;
	solver.getValues(answers, vars);

	// env.out() << "Values = " << answers << endl;

	Mat resized_img = Mat::zeros(new_img_rows, new_img_cols, CV_8UC3);
	uint8_t *resized_img_data = resized_img.data, *img_data = img.data;
	int resized_img_step1 = resized_img.step1(), img_step1 = img.step1();

	for (map<string, Patch>::iterator all_iter = patchs.begin(); all_iter != patchs.end(); ++all_iter) {
		vector<Quad> quads = all_iter->second.get_quads();
		int quads_size = quads.size();

		for (int i = 0; i < quads_size; ++i) {
			// cout << "quad " << i << endl;
			
			Point2f src_points[3]{ quads[i].get_vertex_point(2), quads[i].get_vertex_point(3), quads[i].get_vertex_point(0) }, dst_points[3];
			int answers_indexes[3];

			for (int j = 0; j < 3; ++j) {
				answers_indexes[j] = ((int)ceil(src_points[j].y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(src_points[j].x / quad_width)) * 2;
				dst_points[j] = Point2f(answers[answers_indexes[j]], answers[answers_indexes[j] + 1]);
			}

			// Mat transformation = getAffineTransform(src_points, dst_points);
			Mat transformation = getAffineTransform(dst_points, src_points);
			double *data = (double *)transformation.data;
			int temp_quad_width = 0, step1 = transformation.step1();

			double transformation_values[2][3];

			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 3; ++x) {
					transformation_values[y][x] = data[y * step1 + x];
				}
			}

			// for (int y = (int)src_points[2].y; y <= (int)src_points[0].y; ++y) {
			temp_quad_width = 0;
			for(int y = (int)dst_points[2].y; y <= (int)dst_points[0].y; ++y) {

				// for (int x = (int)src_points[2].x, temp = 0; temp <= temp_quad_width; ++temp, ++x) {
				for (int x = (int)dst_points[2].x, temp = 0; temp <= temp_quad_width; ++temp, ++x) {
					float new_point[2]{0., 0.};

					for (int j = 0; j < 2; ++j) {
						new_point[j] += transformation_values[j][0] * x;
						new_point[j] += transformation_values[j][1] * y;
						new_point[j] += transformation_values[j][2];
					}

					// int resized_index = new_point[1] * resized_img_step1 + new_point[0] * 3, img_index = y * img_step1 + x * 3;
					int resized_index = y * resized_img_step1 + x * 3, img_index = new_point[1] * img_step1 + new_point[0] * 3;
					// cout << "resized_index = " << resized_index << endl;

					for (int j = 0; j < 3; ++j) {
						// cout << "resized_index = " << resized_index << " img_index = " << img_index << endl;
						resized_img_data[resized_index + j] = img_data[img_index + j];
						int test = resized_img_data[resized_index + j];
					}
				}

				++temp_quad_width;

				/*
				for(int temp_quad_width = 0; i < )
				for (int x = (int)src_points[2].x; x <= (int)src_points[0].x; ++x) {
					++temp_quad_width;
				}
				*/
			}

			// cout << "transformation type is " << transformation.type() << endl;
			
			// cout << "down triangle" << endl;

			for (int j = 0; j < 3; ++j) {
				src_points[j] = quads[i].get_vertex_point(j);
				answers_indexes[j] = ((int)ceil(src_points[j].y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(src_points[j].x / quad_width)) * 2;
				dst_points[j] = Point2f(answers[answers_indexes[j]], answers[answers_indexes[j] + 1]);
			}

			// transformation = getAffineTransform(src_points, dst_points);
			transformation = getAffineTransform(dst_points, src_points);
			data = (double *)transformation.data;
			temp_quad_width = 0, step1 = transformation.step1();

			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 3; ++x) {
					transformation_values[y][x] = data[y * step1 + x];
				}
			}

			temp_quad_width = quad_width;
			// for (int y = (int)src_points[2].y; y <= (int)src_points[0].y; ++y) {
			for(int y = (int)dst_points[0].y; y <= (int)dst_points[2].y; ++y) {

				// for (int x = (int)src_points[2].x, temp = 0; temp <= temp_quad_width; ++temp, ++x) {
				for(int x = (int)dst_points[0].x, temp = temp_quad_width; temp >= 0; --temp, ++x) {
					int new_point[2]{ 0, 0 };

					for (int j = 0; j < 2; ++j) {
						new_point[j] += transformation_values[j][0] * x;
						new_point[j] += transformation_values[j][1] * y;
						new_point[j] += transformation_values[j][2];
					}

					// int resized_index = new_point[1] * resized_img_step1 + new_point[0] * 3, img_index = y * img_step1 + x * 3;
					int resized_index = y * resized_img_step1 + x * 3, img_index = new_point[1] * img_step1 + new_point[0] * 3;
					// cout << "resized_index = " << resized_index << endl;

					for (int j = 0; j < 3; ++j) {
						// cout << "resized_index = " << resized_index << " img_index = " << img_index << endl;
						resized_img_data[resized_index + j] = img_data[img_index + j];
						int test = resized_img_data[resized_index + j];
					}
				}

				--temp_quad_width;

				/*
				for(int temp_quad_width = 0; i < )
				for (int x = (int)src_points[2].x; x <= (int)src_points[0].x; ++x) {
				++temp_quad_width;
				}
				*/
			}


			/*
			// perspective transformation
			Point2f src_points[4], dst_points[4];
			int answers_indexes[4];

			for (int j = 0; j < 4; ++j) {
				src_points[j] = quads[i].get_vertex_point(j);
				answers_indexes[j] = ((int)ceil(src_points[j].y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(src_points[j].x / quad_width)) * 2;
				dst_points[j] = Point2f(answers[answers_indexes[j]], answers[answers_indexes[j] + 1]);
			}

			Mat transformation = getPerspectiveTransform(src_points, dst_points);
			*/
			


			/*
			// cout << "quad " << i << endl;
			Point2f src_points[3]{ quads[i].get_vertex_point(2), quads[i].get_vertex_point(3), quads[i].get_vertex_point(0) }, dst_points[3];

			// int vars_center_point1_index = center_point1.y * cols * 2 + center_point1.x * 2, vars_center_point0_index = center_point0.y * cols * 2 + center_point0.x * 2;
			// int vars_center_point1_index = center_quad_id * 2 + center_point1.x * 2, vars_center_point0_index = center_point0.y * cols * 2 + center_point0.x * 2;
			int answers_indexes[3];

			for (int j = 0; j < 3; ++j) {
				answers_indexes[j] = ((int)ceil(src_points[j].y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(src_points[j].x / quad_width)) * 2;
				dst_points[j] = Point2f(answers[answers_indexes[j]], answers[answers_indexes[j] + 1]);
			}

			warpAffine(img, resizing_img, getAffineTransform(src_points, dst_points), resizing_img.size());

			for (int j = 0; j < 3; ++j) {
				src_points[j] = quads[i].get_vertex_point(j);
				answers_indexes[j] = ((int)ceil(src_points[j].y / quad_height)) * total_quad_x_size * 2 + ((int)ceil(src_points[j].x / quad_width)) * 2;
				dst_points[j] = Point2f(answers[answers_indexes[j]], answers[answers_indexes[j] + 1]);
			}

			warpAffine(img, resizing_img, getAffineTransform(src_points, dst_points), resizing_img.size());
			*/
		}

		cout << "a patch has been completed" << endl;
	}

	namedWindow("Resized Image", CV_WINDOW_AUTOSIZE);
	imshow("Resized Image", resized_img);
	imwrite("Resized_" + original_img_name + original_img_type, resized_img);

	waitKey(0);

	return 0;
}

