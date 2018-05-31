

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
						const Mat& F,
						Mat ProjMat,
                        Mat TransMat,
						const Mat& discoeff,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud,
                        vector<double>& mean_reproj_err)
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
    Matx34d P,P1;
    Mat temp;
    temp = TransMat * ProjMat;
	cout<<" ProjMat[0] :"<<ProjMat<<endl;
    cout << "Testing temp " << endl << temp << endl;


    P = Matx34d(ProjMat.at<double>(0,0),ProjMat.at<double>(0,1),ProjMat.at<double>(0,2),
                ProjMat.at<double>(1,0),ProjMat.at<double>(1,1),ProjMat.at<double>(1,2),
                ProjMat.at<double>(2,0),ProjMat.at<double>(2,1),ProjMat.at<double>(2,2));
    P1 = Matx34d(temp.at<double>(0,0),temp.at<double>(0,1),temp.at<double>(0,2),
                 temp.at<double>(1,0),temp.at<double>(1,1),temp.at<double>(1,2),
                 temp.at<double>(2,0),temp.at<double>(2,1),temp.at<double>(2,2));

    cout << "Testing P " << endl << P << endl;
	cout << "Testing P1 " << endl << P1 << endl;

	vector<CloudPoint> pcloud,pcloud1; 
	vector<KeyPoint> corresp;

	double reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,P,P1,pcloud,corresp,discoeff);
	double reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,P,pcloud1,corresp,discoeff);
    mean_reproj_err.push_back((reproj_error1+reproj_error2)/2);
	vector<uchar> tmp_status;
/*
	if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,ProjMat,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0)
	{
				P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); pcloud1.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,ProjMat,P1,pcloud,corresp,discoeff);
				reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,ProjMat,pcloud1,corresp,discoeff);
				
				if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,ProjMat,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {

					
					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					//cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); pcloud1.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,ProjMat,P1,pcloud,corresp,discoeff);
					reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,ProjMat,pcloud1,corresp,discoeff);
					
					if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,ProjMat,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); pcloud1.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,ProjMat,P1,pcloud,corresp,discoeff);
						reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,ProjMat,pcloud1,corresp,discoeff);
						
						if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,ProjMat,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
							cout << "Err is too big." << endl; 
							return false;
						}
					}				
				}			
    }*/
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
                //kp_idx.insert(i);

			}

}





















