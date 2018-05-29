

#include <opencv2/core/core.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <iostream>
#include <list>
#include <set>

#ifndef _COMMON_H
#define _COMMON_H
using namespace std;
using namespace cv;
using namespace Eigen;
namespace Eigen {

	typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
	typedef Eigen::Matrix<double,9,1> Vector9d;
}
typedef vector<KeyPoint> v_keypoint;
typedef vector<DMatch> v_match;
typedef vector<Point2f> v_point;
typedef vector<pair<int,int>> v_pair;

struct CloudPoint {
    cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
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
cv::Point3d FirstFrame2Second(cv::Point3d,cv::Mat P);
cv::Point3d CurrentPt2World(cv::Point3d point,std::vector<cv::Mat> P1_trans,int count);

Scalar ReprojErrorAndPointCloud(const vector<KeyPoint> &pt_set2, const Mat &K, const Matx34d &P1,
								vector<CloudPoint> &pointcloud, const vector<Point3d> &points_3d);


double* CvMatrix2ArrayCamera( Mat R,Mat K,Mat t);
double* CvPoint3d2ArrayPoint( Point3d p);
void BundleAdjustment(const vector<KeyPoint> keypoints_2_depth,
                      Mat &R, Mat& K,Mat& t,vector<CloudPoint> &pointcloud);






#endif


