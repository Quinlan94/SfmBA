

#include "CalculateCameraMatrix.h"
#include "Triangulation.h"
#include "Common.h"



#include <vector>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>


using namespace cv;
using namespace std;




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
                        bool initial)
{
    /*
	Mat_<double> R1(3,3);
            R1= R_cv;
	Mat_<double> t1(1,3);
           t1 = t_cv;

    *
    //================================================================
//	Mat_<double> E = K.t() * F * K; // Essential Matrix
//	if(fabsf(determinant(E)) > 1e-05) {
//		cout << "det(E) != 0 : " << determinant(E) << "\n";
//		return false;
//	}
//	cout << "E method : "<<E<<endl;
	SVD svd(E);
	Matx33d W(0,-1,0,	
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd.u * Mat(W) * svd.vt; //Rotation solution 1
	R2 = svd.u * Mat(Wt) * svd.vt; //Rotation solution 2
	t1 = svd.u.col(2); //Translatiion solution 1
	t2 = -svd.u.col(2); //Translation solution 2
    //=========================================================================
	vector<Point2f> points1;
	vector<Point2f> points2;


	P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0),	R1(2,1),	R1(2,2),	t1(2)); // Camara Matrix
	ProjMat = Matx34d(1,0,0,0,
				0,1,0,0,
				0,0,1,0);*/
    Mat P,P1,temp,temp_1;
	Eigen::Vector3d proj_center_1;
	Eigen::Vector3d proj_center_2;


	P = ProjMat_1(Range(0,3),Range::all());
	P1 = ProjMat_2(Range(0,3),Range::all());

	temp = ProjMat_1.inv();
	temp_1 = ProjMat_2.inv();
	for (int j = 0; j < 3; ++j) {
		proj_center_1(j) = temp.at<double>(j,3);
		proj_center_2(j) = temp_1.at<double>(j,3);
	}



	vector<CloudPoint> pcloud,pcloud1; 
	vector<KeyPoint> corresp;

	double reproj_error1 = TriangulatePoints(matches,imgpts1_good,imgpts2_good,K,proj_center_1,proj_center_2,P,P1,pcloud,corresp,distcoeff,initial);
	//double reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,P,pcloud1,corresp,discoeff);
    mean_reproj_err.push_back(reproj_error1);
	vector<uchar> tmp_status;
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
                //kp_idx.insert(i);

			}

}





















