


#include "Common.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


using namespace std;
using namespace cv;

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches) {
	std::vector<cv::DMatch> flip;
	for(int i=0;i<matches.size();i++) {
		flip.push_back(matches[i]);
		swap(flip.back().queryIdx,flip.back().trainIdx);
	}
	return flip;
}

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) 
{
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) 
	{
		out.push_back(cpts[i].pt);
	}
	return out;
}

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2,
                               std::vector<int>& kp_idx)
{
	for (unsigned int i=0; i<matches.size(); i++) 
	{

		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
        kp_idx.push_back(matches[i].trainIdx);
	}	
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) 
{
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) 
{
	kps.clear();
	for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}
cv::Point3d FirstFrame2Second(cv::Point3d point,Mat P)
{
	Mat_<double> X = Mat_<double>(4,1);
	X << point.x,point.y,point.z,1;
	X = P * X;
	point.x = X(0);
	point.y = X(1);
	point.z = X(2);

	//cout << "Testing P1 is same as trans: " << P << endl;

	return  point;

}
cv::Point3d CurrentPt2World(cv::Point3d point,std::vector<cv::Mat> P1_trans,int count)
{
    Mat Pw = (Mat_<double>(4,4)<<1,0,0,0,
                                 0,1,0,0,
                                 0,0,1,0,
                                 0,0,0,1);
    //cout << "Testing Pw " << Pw << endl;

    for (int i = 0; i < count; ++i) {

        Pw = Pw *P1_trans[i];
        //cout << "Testing Pw " << Pw << endl;

    }

    return  FirstFrame2Second(point,Pw.inv());



}











