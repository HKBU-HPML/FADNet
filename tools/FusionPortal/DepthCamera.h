#pragma once
#include "opencv2/core.hpp"
#include "glm/ext.hpp" 
#include "glm/vec3.hpp" 
#include "glm/vec4.hpp" 
#include "glm/mat4x4.hpp" 
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/constants.hpp"
class DepthCamera
{
public:
	DepthCamera();

	cv::Size size;

	glm::mat4x4 transform;

	// void DepthCamera::Init(cv::Size imgSize);

	// void DepthCamera::ProjectFloatDepth(cv::Mat &depth, cv::Mat &color, float *output);

	// void DepthCamera::ProjectEncodedDepthRGB(cv::Mat &depth, cv::Mat &color, float *output);

	// void DepthCamera::SavePLY(const char* filename, const cv::Mat& mat, const cv::Mat& color);

	void Init(cv::Size imgSize);

	void ProjectFloatDepth(cv::Mat &depth, cv::Mat &color, float *output);

	void ProjectEncodedDepthRGB(cv::Mat &depth, cv::Mat &color, float *output);

	void SavePLY(const char* filename, const cv::Mat& mat, const cv::Mat& color);
private:

};
