#include "DepthCamera.h"
#include "Constants.h"
#include "opencv2/calib3d/calib3d.hpp"

#include <stdio.h>
using namespace cv;

DepthCamera::DepthCamera() {

}

void DepthCamera::Init(Size imgSize) {
	size = imgSize;
}
void DepthCamera::ProjectFloatDepth(Mat &depthMap, Mat &colorMap, float *output) {

	int id = 0;
	float ox = IMAGE_WIDTH / 2 - 1;
	float oy = IMAGE_HEIGHT / 2 - 1;
	float fov = 37.849197*3.14159265/180.0;
	float projZ = 1.0 / tan(fov/2);
	float screenRatio = 1.0f * IMAGE_WIDTH / IMAGE_HEIGHT;

	for (int y = 0; y<size.height / SAMPLE_STEP; y++) {
		for (int x = 0; x<size.width / SAMPLE_STEP; x++) {
			float &depth = depthMap.at<Vec3f>(y*SAMPLE_STEP, x*SAMPLE_STEP)[0];
			Vec3b &rgb = colorMap.at<Vec3b>(y*SAMPLE_STEP, x*SAMPLE_STEP);

			float projx = (x*SAMPLE_STEP - ox) / ox;
			float projy = (y*SAMPLE_STEP - oy) / oy;

			float z = depth;

			output[id] = z / projZ * projx * screenRatio;
			output[id + 1] = -z / projZ * projy;
			output[id + 2] = z;
			
			output[id + 3] = rgb[2] / 255.0f;
			output[id + 4] = rgb[1] / 255.0f;
			output[id + 5] = rgb[0] / 255.0f;
			id += 6;
		}
	}
}

void DepthCamera::ProjectEncodedDepthRGB(Mat &depth, Mat &color, float *output) {
	/*uchar r; uchar g; uchar b;
	int px; int py;
	int id = 0;
	for (int y = 0; y<size.height / SAMPLE_STEP; y++) {
		for (int x = 0; x<size.width / SAMPLE_STEP; x++) {
			Vec3b &depth = color.at<Vec3b>(y*SAMPLE_STEP, x*SAMPLE_STEP);
			Vec3b &rgb = color.at<Vec3b>(y*SAMPLE_STEP, x*SAMPLE_STEP);
			output[id] = 0;// x*.01f;
			output[id + 1] = 0;// y*.01f;
			output[id + 2] = 0;
			output[id + 3] = rgb[0];
			output[id + 1] = rgb[1];
			output[id + 2] = rgb[2];
			id+=6;
		}
	}*/
	int id = 0;
	float ox = IMAGE_WIDTH / 2 -1;
	float oy = IMAGE_HEIGHT / 2 -1;

	float projZ = 1.0 / 0.53170943166f;
	float screenRatio = 1.0f * IMAGE_WIDTH / IMAGE_HEIGHT;

	for (int y = 0; y<size.height / SAMPLE_STEP; y++) {
		for (int x = 0; x<size.width / SAMPLE_STEP; x++) {
			Vec3b &depthrgb = depth.at<Vec3b>(y*SAMPLE_STEP, x*SAMPLE_STEP);
			Vec3b &rgb = color.at<Vec3b>(y*SAMPLE_STEP, x*SAMPLE_STEP);

			float projx = (x*SAMPLE_STEP - ox)/ox;
			float projy = (y*SAMPLE_STEP - oy)/oy;

			float z = 0.00001f * ((depthrgb[2] << 16) + (depthrgb[1] << 8) + depthrgb[0]) ;

			//output[id] = (x - .5*IMAGE_WIDTH/SAMPLE_STEP)*.1f;// z / projZ * projx * screenRatio;
			//output[id + 1] = (y - .5*IMAGE_HEIGHT / SAMPLE_STEP)*.1f;// z / projZ * projy;
			if (z > 10) {
				output[id]	   = 0;
				output[id + 1] = 0;
				output[id + 2] = 0;
			}
			else {
				output[id]	   = z / projZ * projx * screenRatio;
				output[id + 1] = -z / projZ * projy;
				output[id + 2] = z;
			}
			output[id + 3] =  rgb[2] / 255.0f;
			output[id + 4] =  rgb[1] / 255.0f;
			output[id + 5] =  rgb[0] / 255.0f;
			id += 6;
		}
	}
}

void DepthCamera::SavePLY(const char* filename, const Mat& mat, const Mat& color)
{
	int infc = 0;

	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (isinf(point[0])) {
				infc++;
			}
			else if (isinf(point[1])) {
				infc++;
			}
			else if (isinf(point[2]) || abs(point[2]) >= 1000) { infc++; }
		}
	}
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %i\n", mat.rows*mat.cols - infc);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);

			//if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			if (!isinf(point[0]) && !isinf(point[1]) && !isinf(point[2])) {
				//fprintf(fp, "%f %f %f %d %d %d", point[0], point[1], point[2], c, c, c);
				if (abs(point[2]) < 1000) {
					//point[2] = 0;
					Vec3b c = color.at<Vec3b>(y, x);
					fprintf(fp, "%f %f %f %d %d %d ", point[0], point[1], point[2], c[0], c[1], c[2]);
				}
			}
		}
	}
	fprintf(fp, "\n");
	fclose(fp);
}
