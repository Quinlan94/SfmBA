#pragma once

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>



#include "Common.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;







bool FeatureMatching(const Mat& img_1,
				   const Mat& img_2, 
				   vector<KeyPoint>& keypts1,
				   vector<KeyPoint>& keypts2,
                     vector<KeyPoint>& orb_keypts1,
                     vector<KeyPoint>& orb_keypts2,
				   	vector<DMatch>& fliter_matches,
					 vector<DMatch>& best_matches,
					FeatureExtract method,
                    bool initial,
                   double& mindist);