#pragma once
#include "DepthCamera.h"
#include "PointCloudViewer.h"
#include "Constants.h"

class DepthCameraGrid
{
public:


	DepthCameraGrid();
	DepthCamera cams[CAM_NUM];
	float *vertexData[FRAME][CAM_NUM];
	char* inFilePath[CAM_NUM];
	char* exFilePath[CAM_NUM];
	PointCloudViewer *viewer;
	
	//void DepthCameraGrid::Start();
private:
	

};