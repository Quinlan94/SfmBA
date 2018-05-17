

#include "Triangulation.h"

#include <iostream>

using namespace std;
using namespace cv;

#undef __SFM__DEBUG__

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
            (
                    ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0),
                    ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1)
            );
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
								   Matx34d P,		//camera 1 matrix
								   Point3d u1,		//homogenous image point in 2nd camera
								   Matx34d P1		//camera 2 matrix
								   ) 
{
	
	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	//	cout << "u " << u <<", u1 " << u1 << endl;
	//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u.x*P(1)-u.y*P(0);
	//	A(3) = u1.x*P1(2)-P1(0);
	//	A(4) = u1.y*P1(2)-P1(1);
	//	A(5) = u1.x*P(1)-u1.y*P1(0);
	//	Matx43d A; //not working for some reason...
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u1.x*P1(2)-P1(0);
	//	A(3) = u1.y*P1(2)-P1(1);
	Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),		
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),		
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),	
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
			  );
	Matx41d B(-(u.x*P(2,3)	-P(0,3)),//这段什么意思
			  -(u.y*P(2,3)	-P(1,3)),
			  -(u1.x*P1(2,3)	-P1(0,3)),
			  -(u1.y*P1(2,3)	-P1(1,3)));
	
	Mat_<double> X;
	solve(A,B,X,DECOMP_SVD);
	
	return X;
}




Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
											Matx34d P,			//camera 1 matrix
											Point3d u1,			//homogenous image point in 2nd camera
											Matx34d P1			//camera 2 matrix
											) 
{
	double wi = 1, wi1 = 1;
	Mat_<double> X(4,1); 
	for (int i=0; i<10; i++) 
	{ 
		Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
		
		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
		
		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;
		
		wi = p2x;
		wi1 = p2x1;
		
		//reweight equations and solve
		Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,		
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,		
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,	
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		Mat_<double> B = (Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
												  -(u.y*P(2,3)	-P(1,3))/wi,
												  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );
		
		solve(A,B,X_,DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}



Eigen::Vector3d TriangulatePointDLT(const Eigen::Matrix3x4d& proj_matrix1,
                                    const Eigen::Matrix3x4d& proj_matrix2,
                                    const Eigen::Vector2d& point1,
                                    const Eigen::Vector2d& point2)
{
    Eigen::Matrix4d A;

    A.row(0) = point1(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
    A.row(1) = point1(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
    A.row(2) = point2(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
    A.row(3) = point2(1) * proj_matrix2.row(2) - proj_matrix2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

    return svd.matrixV().col(3).hnormalized();
}



//Triagulate points
double TriangulatePoints(const vector<KeyPoint>& pt_set1, 
						const vector<KeyPoint>& pt_set2, 
						const Mat& K,
						const Mat& Kinv,
						const Matx34d& P,
						const Matx34d& P1,
						vector<CloudPoint>& pointcloud,
						vector<KeyPoint>& correspImg1Pt,
						const Mat& distcoeff)
{

	vector<double> depths;

	
//	pointcloud.clear();
	correspImg1Pt.clear();
	
	Matx44d P1_(P1(0,0),P1(0,1),P1(0,2),P1(0,3),
				P1(1,0),P1(1,1),P1(1,2),P1(1,3),
				P1(2,0),P1(2,1),P1(2,2),P1(2,3),
				0,		0,		0,		1);
	Matx44d P1inv(P1_.inv());

    Eigen::Matrix3x4d P_matrix,P1_matrix;
	P_matrix<<P(0,0),P(0,1),P(0,2),P(0,3),
			P(1,0),P(1,1),P(1,2),P(1,3),
			P(2,0),P(2,1),P(2,2),P(2,3);

    P1_matrix<<P1(0,0),P1(0,1),P1(0,2),P1(0,3),
			P1(1,0),P1(1,1),P1(1,2),P1(1,3),
			P1(2,0),P1(2,1),P1(2,2),P1(2,3);

	Mat T1 = (Mat_<double>(3,4)<<1,0,0,0,
	                             0,1,0,0,
	                             0,0,1,0);
	//cout<< "T1: "<<T1<<endl;
	Mat T2 = (Mat_<double>(3,4)<<P1(0,0),P1(0,1),P1(0,2),P1(0,3),
			                     P1(1,0),P1(1,1),P1(1,2),P1(1,3),
			                     P1(2,0),P1(2,1),P1(2,2),P1(2,3));

	Mat R = (Mat_<double>(3,3)<<P1(0,0),P1(0,1),P1(0,2),
			                    P1(1,0),P1(1,1),P1(1,2),
			                    P1(2,0),P1(2,1),P1(2,2));

	Mat t0 = (Mat_<double>(3,1)<<P1(0,3),P1(1,3),P1(2,3));


	//cout<< "T2: "<<T2<<endl;




	cout << "Triangulating Now . . ."<<endl;
	double t = getTickCount();
	vector<double> reproj_error;
	unsigned int pts_size = pt_set1.size();
	
#if 1

	vector<Point2f> _pt_set1_pt,_pt_set2_pt;

    for (int j=0; j<pt_set1.size(); j++)
    {
		//要转化为相机的归一化坐标
        _pt_set1_pt.push_back(pixel2cam(pt_set1[j].pt,K));
        _pt_set2_pt.push_back(pixel2cam(pt_set2[j].pt,K));
    }

    Mat pts_4d;
    vector<Point3f> points_3d;
	cv::triangulatePoints(P,P1,_pt_set1_pt,_pt_set2_pt,pts_4d);
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
                x.at<float>(0,0),
                x.at<float>(1,0),
                x.at<float>(2,0)
        );
        points_3d.push_back( p );
    }

    Mat_<double> KP1 = K * Mat(P1);
    Mat_<double> X = Mat_<double>(4,1);
    for (int i=0; i<points_3d.size(); i++)
    {
        X << points_3d[i].x,points_3d[i].y,points_3d[i].z,1;
        Mat_<double> xPt_img = KP1 * X;
        Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

        double reprj_err = norm(xPt_img_ - pt_set2[i].pt);
        reproj_error.push_back(reprj_err);

        CloudPoint cp;
        cp.pt = Point3d(X(0), X(1), X(2));
        cp.reprojection_error = reprj_err;

        pointcloud.push_back(cp);
        correspImg1Pt.push_back(pt_set1[i]);

        depths.push_back(X(2));
    }


#else
	Mat_<double> KP1 = K * Mat(P1);
#pragma omp parallel for num_threads(1)
	for (int i=0; i<pts_size; i++) 
	{
		Point2f kp = pt_set1[i].pt; 
		Point3d u(kp.x,kp.y,1.0);
		Mat_<double> um = Kinv * Mat_<double>(u); 
		u.x = um(0); u.y = um(1); u.z = um(2);

		Point2f kp1 = pt_set2[i].pt; 
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1); 
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Eigen::Vector2d u_dlt,u1_dlt ;
		u_dlt<< um(0), um(1);
		u1_dlt<<  um1(0), um1(1);
		Mat pts_4d;



		Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);

        Eigen::Vector3d X_test,X_cv;
        X_test << X(0),X(1),X(2);


		Eigen::Vector3d X_dlt = TriangulatePointDLT(P_matrix,P1_matrix,u_dlt,u1_dlt);

		vector<Point2d> pt1,pt2;
		Point2d pt_1,pt_2;
		pt_1.x=um(0);
		pt_1.y=um(1);
		pt1.push_back(pt_1);
		pt_2.x=um1(0);
		pt_2.y=um1(1);
        pt2.push_back(pt_2);

		cv::triangulatePoints(T1,T2,pt1,pt2,pts_4d);
		Mat x = pts_4d.col(0);
		cout<<"X_origin: "<<X_test<<endl;
		cout<<"X_dlt: "<<X_dlt<<endl;

		x /= x.at<double>(3,0);
		cout<<"x_cv: "<<x<<endl;
		Point3d p (x.at<double>(0,0),
				   x.at<double>(1,0),
		           x.at<double>(2,0));
		Mat pt2_trans = R * (Mat_<double>(3,1)<<p.x,p.y,p.z)+t0;
		pt2_trans /= pt2_trans.at<double>(2,0);
		cout <<"pt2_trans: "<< pt2_trans<<endl;

		
//		cout << "3D Point: " << X << endl;
//		Mat_<double> x = Mat(P1) * X;
//		cout <<	"P1 * Point: " << x << endl;
//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
//		cout <<	"Point: " << xPt << endl;
		Mat_<double> xPt_img = KP1 * X;				//reproject
//		cout <<	"Point * K: " << xPt_img << endl;
		Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
				
#pragma omp critical
		{
			double reprj_err = norm(xPt_img_-kp1);
			reproj_error.push_back(reprj_err);

			CloudPoint cp; 
			cp.pt = Point3d(X(0),X(1),X(2));
			cp.reprojection_error = reprj_err;
			
			pointcloud.push_back(cp);
			correspImg1Pt.push_back(pt_set1[i]);

			depths.push_back(X(2));

		}
	}
#endif
	
	Scalar mse = mean(reproj_error);
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. \n\r"<<pointcloud.size()<<"points, " <<"mean square reprojetion err = " << mse[0] <<  endl;
	
	//show "range image"

//	{
//		double minVal,maxVal;
//		minMaxLoc(depths, &minVal, &maxVal);
//		Mat tmp(1224,1632,CV_8UC3,Scalar(0,0,0)); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
//		for (unsigned int i=0; i<pointcloud.size(); i++) {
//			double _d = MAX(MIN((pointcloud[i].pt.z-minVal)/(maxVal-minVal),1.0),0.0);
//			circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
//		}
//		cvtColor(tmp, tmp, CV_HSV2BGR);
////		imshow("Depth Map", tmp);
////		waitKey(0);
////		destroyWindow("Depth Map");
//	}

	
	return mse[0];
}



bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status) {
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye(); 
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.8)
		return false; //less than 80% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i=0; i<pcloud.size(); i++) {
			Vec3d w = Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}

	return true;
}
