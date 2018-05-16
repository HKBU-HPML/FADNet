#include "DepthCameraGrid.h"
#include "PointCloudViewer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "FileUtil.h"
#include <stdio.h>

#include "glm/ext.hpp" 
#include "glm/vec3.hpp" 
#include "glm/vec4.hpp" 
#include "glm/mat4x4.hpp" 
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/constants.hpp"

using namespace cv;


DepthCameraGrid::DepthCameraGrid() {
	
	Size size(IMAGE_WIDTH, IMAGE_HEIGHT);

	//data for test
	char rgb_filename[CAM_NUM][FRAME];
	char disp_filename[CAM_NUM][FRAME];

	float camData[8][6]  = {
		{ 0.123,8.618,35.329,		0,		0,			0 },
		{ 33.646,8.618,0.307,		0,		-90,		0 },
		{-1.377,8.618,- 33.216,		0,		180,		0 },
		{ -34.899,8.618,1.807,		0,		90,			0 },

		{ 21.855,21.637,22.477,		21.722,		-45,		0 },
		{ 20.794,21.637,- 21.425,	21.722,		-135,		0 },
		{ -23.108,21.637,- 20.364,	21.722,		135,		0 },
		{ -22.048,21.637,23.538,	21.722,		45,			0 }
	};
	float angToRad = 3.1415926f/180.0f;
	for (int i = 0; i < 8; i++) {
		cams[i].Init(size);
		for (int j = 0; j < FRAME; j++) {
			char rgbfileName[256];
			char depthfileName[256];
			sprintf(rgbfileName,   "./data/%i/%i.png", i,j);
			sprintf(depthfileName, "./data/%i/p%i.exr", i,j);
			printf(rgbfileName); printf("\n");
			printf(depthfileName); printf("\n");
			Mat rgb = imread(rgbfileName, IMREAD_COLOR);
			Mat depth = imread(depthfileName, IMREAD_UNCHANGED);
			
			
			
			// convert disparity data to point cloud
			vertexData[j][i] = (float*)malloc(sizeof(float) * 6 * IMAGE_WIDTH * IMAGE_HEIGHT / SAMPLE_STEP / SAMPLE_STEP);
			
			cams[i].ProjectFloatDepth(depth, rgb, vertexData[j][i]);
			
			glm::mat4x4 m0 = glm::mat4x4(1.0);
			glm::mat4x4 m1 = glm::translate(m0,glm::vec3(camData[i][0], camData[i][1], -camData[i][2]));

			glm::mat4x4 m2 = glm::rotate(m1, camData[i][4] * angToRad, glm::vec3(0, 1, 0));
			glm::mat4x4 m3 = glm::rotate(m2, camData[i][3] * angToRad, glm::vec3(1, 0, 0));
			glm::mat4x4 m4 = glm::rotate(m3, camData[i][5] * angToRad, glm::vec3(0, 0, 1));
			cams[i].transform = m4;
			rgb.release();
			depth.release();
		}
	}
	
	viewer = new PointCloudViewer();

	viewer->RegistGrid(this);
	//viewer->UpdatePointData(cams,vertexData[0], IMAGE_WIDTH * IMAGE_HEIGHT/SAMPLE_STEP/SAMPLE_STEP);
	viewer->LoopGlut();
}

/*void DepthCameraGrid::Start() {
	viewer->LoopGlut();
}*/
