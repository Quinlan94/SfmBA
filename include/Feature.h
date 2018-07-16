//
// Created by quinlan on 18-7-6.
//

#ifndef SFM_FEATURE_H
#define SFM_FEATURE_H

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Common.h"

#include <set>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void FeatureExtractor(vector<Mat> images,vector<v_keypoint>& keypoints,vector<v_match>& v_matches,FeatureExtract method);

bool getGoodMatches(vector<DMatch> &v_matches, vector<DMatch> &good_matches,double& midist,int orb );
bool RansacGoodMatches(vector<DMatch> &good_matches,vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2,
                       vector<KeyPoint>& imgpts_good_1, vector<KeyPoint>& imgpts_good_2,v_pair& kp_good_depth_idx,
                       unordered_set<int> &kp_depth_idx);
bool RansacGoodMatches(vector<DMatch> &good_matches,vector<KeyPoint>& orb_keypoints_1, vector<KeyPoint>& orb_keypoints_2,
                       vector<Point2f>& pts_1, vector<Point2f>& pts_2);
void WriteFileMatches(Mat images_1,Mat images_2,vector<DMatch> draw_matches,
                      vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2,string name);
#endif //SFM_FEATURE_H
