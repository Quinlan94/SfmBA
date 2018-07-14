#pragma once

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/core.hpp>

#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <iostream>

#include <set>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include <omp.h>

#ifndef _COMMON_H
#define _COMMON_H


using namespace std;
using namespace cv;
using namespace Eigen;
enum FeatureExtract{
	Surf= 0,Orb ,Sift
};
namespace Eigen {

	typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
	typedef Eigen::Matrix<double,9,1> Vector9d;
}
typedef pcl::PointXYZRGB Point_PCL;
typedef pcl::PointCloud<Point_PCL> PointCloud;

typedef vector<KeyPoint> v_keypoint;
typedef vector<DMatch> v_match;
typedef vector<Point2f> v_point;
typedef vector<pair<int,int>> v_pair;
typedef pair<int,int> double_pair;

# define M_PI		3.14159265358979323846
const double min_initial_angle = 6;
const double min_fliter_angle = 1.5;
const double max_reproj_err = 4;

struct CloudPoint {
    cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
	std::pair<double_pair,double_pair> track;
};


std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);
void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);
void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2,
							   std::vector<std::pair<int,int>>& kp_idx);
void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
                               const std::vector<cv::KeyPoint>& imgpts2,
                               const std::vector<cv::DMatch>& matches,
                               std::vector<cv::KeyPoint>& pt_set1,
                               std::vector<cv::KeyPoint>& pt_set2);
cv::Point3d FirstFrame2Second(cv::Point3d,cv::Mat P);
cv::Point3d CurrentPt2World(cv::Point3d point,std::vector<cv::Mat> P1_trans,int count);
bool cl_greater(const DMatch& a,const DMatch& b);
Scalar ReprojErrorAndPointCloud(vector<KeyPoint> &pt_set2, const Mat &K, const Matx34d &P1,
								vector<CloudPoint> &pointcloud, const vector<Point3d> &points_3d);
Scalar ReprojErrorAndPointCloud(vector<KeyPoint> &pt_set2, const Mat &K, const Matx34d &P1,
								vector<CloudPoint> &pointcloud, const vector<Point3d> &points_3d,std::vector<cv::DMatch>& good_matches);


double* CvMatrix2ArrayCamera( Mat R,Mat K,Mat t);
double* CvPoint3d2ArrayPoint( Point3d p);
void BundleAdjustment(vector<KeyPoint> keypoints_2_depth,
                      Mat &R, Mat& K,Mat& t,vector<CloudPoint> &pointcloud);



Point_PCL DisplayCamera(const Mat P);


#endif


