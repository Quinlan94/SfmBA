#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.h"
using namespace std;
using namespace cv;

bool FindCameraMatrices(const Mat& K, 
						const Mat& Kinv,
						const Mat& F,
						Mat ProjMat,
						Mat TransMat,
						const Mat& distcoeff,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud,
                        vector<double>& mean_reproj_err);














