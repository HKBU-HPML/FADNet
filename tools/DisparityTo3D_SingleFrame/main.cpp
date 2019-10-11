#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
struct uchar3{
  uchar x, y, z;
};
int main(int argc, char **argv) {
  if (argc < 4 || argc > 8) {
    std::cout << "Project disparity map to 3D point cloud" << std::endl;
    std::cout << "Usage:  DisparityTo3D /path/to/disparity /path/to/out.obj /path/to/rgb inv_baseline focal max_disparity min_disparity" << std::endl;
    std::cout << "Latest three parameter can be ignored" << std::endl;
    return 1;
  }

  cv::Mat disp = cv::imread(argv[1], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  if (disp.empty()) {
    std::cout << "Cannot open file: " << argv[1] << std::endl;
    return 1;
  }

  int max_disp = 200, min_disp = 100;
  float inv_baseline = 0.0038f, focal = 1800.0f;
  inv_baseline = atof(argv[4]);
  focal = atof(argv[5]);

  if (argc > 6) {
    max_disp = atoi(argv[6]);
  }
  if (argc > 7) {
    min_disp = atoi(argv[7]);
  }

  int width = disp.cols, height = disp.rows;
  cv::Mat dis_flaot = cv::Mat(width,height,CV_32FC1);
  disp.convertTo(dis_flaot,CV_32FC1);
  int cx = width / 2, cy = height / 2;
  // float inv_baseline = 0.0019f, focal = 900.0f;
  float x,y,z, temp_dis;
  uchar3 color;

  std::ofstream f_mesh(argv[2]);
  cv::Mat rgb = cv::imread(argv[3], cv::IMREAD_COLOR);
  if (argc > 3) {
    if (rgb.empty()) {
      std::cout << "Cannot open file: " << argv[3] << std::endl;
      rgb = cv::Mat(width,height,CV_8UC3);
      memset(rgb.data, 255, height*width*3);
    }
  }

  for (int iter_i = 0; iter_i < height; ++iter_i) {
    for (int iter_j = 0; iter_j < width; ++iter_j) {
      temp_dis = dis_flaot.at<float>(iter_i,iter_j);
      // std::cout << temp_dis << std::endl;
      if (temp_dis < min_disp || temp_dis > max_disp) {
        // std::cout << temp_dis << std::endl;
        continue;
      }
      z = inv_baseline * temp_dis;
      x = -(iter_i - cx) / z;
      y = (iter_j - cy) / z;
      z = focal / z;
      if (argc <= 3) {
        f_mesh << "v " << x << " " << y << " " << z << std::endl;
      } else{
        color = rgb.at<uchar3>(iter_i,iter_j);
        // std::cout << "v " << x << " " << y << " " << z << " "
        //        << int(color.z) << " " << int(color.y) << " " << int(color.x) << std::endl;
        f_mesh << "v " << x << " " << y << " " << z << " "
               << int(color.z) << " " << int(color.y) << " " << int(color.x) << std::endl;
      }
    }
  }
  return 0;
}
