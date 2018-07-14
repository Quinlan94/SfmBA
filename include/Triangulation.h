
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "Common.h"
using namespace std;
using namespace cv;




double DegToRad(const double deg);
double RadToDeg(const double rad);

Eigen::Vector3d TriangulateMultiViewPoint(
        const std::vector<Eigen::Matrix3x4d>& proj_matrices,
        const std::vector<Eigen::Vector2d>& points);

double TriangulatePoints(std::vector<cv::DMatch>& matches,
		                 vector<KeyPoint>& pt_set1,
						 vector<KeyPoint>& pt_set2,
						const Mat& K,
						 Eigen::Vector3d proj_center_1,
						 Eigen::Vector3d proj_center_2,
						const Mat& P,
						const Mat& P1,
						vector<CloudPoint>& pointcloud,
						vector<KeyPoint>& correspImg1Pt,
						const Mat& distcoeff,
                        bool initial = false);


Eigen::Vector3d TriangulatePointDLT(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2);
double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
								   const Eigen::Vector3d& proj_center2,
								   const Eigen::Vector3d& point3D);
bool HasPointPositiveDepth(const Eigen::Matrix3x4d& proj_matrix,
						   const Eigen::Vector3d& point3D);

Eigen::Matrix3x4d CvMatToEigen34(Mat cv_matrix);
Eigen::Vector3d CvPointToVector3d(Point3d p);





