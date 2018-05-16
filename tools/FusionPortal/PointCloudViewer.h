#pragma once
#include "DepthCamera.h"
class PointCloudViewer
{
public:
	
	PointCloudViewer();
	void UpdatePointData(DepthCamera *cameras, float **data, int vertNum);
	void LoopGlut();
	void *grid;
	void RegistGrid(void * grid);
	//void RegistCamera(DepthCamera* cams);
private:
};