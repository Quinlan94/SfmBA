#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.h"
using namespace std;
using namespace cv;

bool FindCameraMatrices(const Mat& K,
						const Mat& Kinv,
						Mat ProjMat_1,
						Mat ProjMat_2,
						const Mat& distcoeff,
						std::vector<cv::DMatch>& matches,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						vector<CloudPoint>& outCloud,
						vector<double>& mean_reproj_err,
						bool initial);














