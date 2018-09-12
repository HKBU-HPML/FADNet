/*
Copyright 2016 fixstars

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libsgm.h>
using namespace cv;
int main(int argc, char* argv[]) {
	if (argc < 4) {
		std::cerr << "usage: stereosgm left_img right_img out_disp [disp_size]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// cv::Mat left = cv::imread(argv[1], -1);
	// cv::Mat right = cv::imread(argv[2], -1);
	cv::Mat left = cv::imread(argv[1], IMREAD_GRAYSCALE); // CV_LOAD_IMAGE_COLOR
	cv::Mat right = cv::imread(argv[2], IMREAD_GRAYSCALE);
	std::string out_disp = argv[3];

	// cv::Mat left1 = cv::imread("/tmp/left1.png", IMREAD_GRAYSCALE); // CV_LOAD_IMAGE_COLOR
	// cv::Mat right1 = cv::imread("/tmp/right1.png", IMREAD_GRAYSCALE); // CV_LOAD_IMAGE_COLOR
	// cv::Mat left2 = cv::imread("/tmp/left3.png", IMREAD_GRAYSCALE); // CV_LOAD_IMAGE_COLOR
	// cv::Mat right2 = cv::imread("/tmp/right3.png", IMREAD_GRAYSCALE); // CV_LOAD_IMAGE_COLOR
	int disp_size = 64;
	if (argc >= 5) {
		disp_size = atoi(argv[4]);
	}

	Size newSize = Size(left.cols / 2, left.rows / 2);
	resize(left, left, newSize, 0, 0, CV_INTER_LINEAR);
	resize(right, right, newSize, 0, 0, CV_INTER_LINEAR);

    // if (argc >=6)
	// {
	// 	int resizew = atoi(argv[5]);
	// 	int resizeh = atoi(argv[6]);
	//     Size newsize = Size(resizew, resizeh);
	//     resize(left, left, newsize, 0, 0, CV_INTER_LINEAR);
	//     resize(right, right, newsize, 0, 0, CV_INTER_LINEAR);
	// }

	// resize(left2, left2, newsize, 0, 0, CV_INTER_LINEAR);
	// resize(right2, right2, newsize, 0, 0, CV_INTER_LINEAR);
	

	if (left.size() != right.size() || left.type() != right.type()) {
		std::cerr << "mismatch input image size" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int bits = 0;

	switch (left.type()) {
		case CV_8UC1: bits = 8; printf("bits=%d\n", bits); break;
		case CV_16UC1: bits = 16; printf("bits=%d\n", bits); break;
		default:
					   std::cerr << "invalid input image color format" << left.type() << std::endl;
					   std::exit(EXIT_FAILURE);
	}

	sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, bits, 8, sgm::EXECUTE_INOUT_HOST2HOST);
	printf("sgm::StereoSGM\n");

	cv::Mat output(cv::Size(left.cols, left.rows), CV_8UC1);

	ssgm.execute(left.data, right.data, (void**)&output.data);

	// cv::imshow("image", output * 256 / disp_size);

	// write disp map
	Size dispSize = Size(left.cols * 2, left.rows * 2);
	resize(output, output, dispSize, 0, 0, CV_INTER_LINEAR);
	output = output * 2;
	cv::imwrite(out_disp, output);
}
