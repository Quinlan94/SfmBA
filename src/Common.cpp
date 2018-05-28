


#include "Common.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ceres/ceres.h"
#include "SnavelyReprojectionError.h"

#include <iostream>



using namespace std;
using namespace cv;
using namespace ceres;

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
                               std::vector<pair<int,int>>& kp_idx)
{
	for (unsigned int i=0; i<matches.size(); i++) 
	{

		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
        kp_idx.push_back(make_pair(matches[i].queryIdx,matches[i].trainIdx));
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

	return  point;

}
cv::Point3d CurrentPt2World(cv::Point3d point,std::vector<cv::Mat> P1_trans,int count)
{
    Mat Pw = (Mat_<double>(4,4)<<1,0,0,0,
                                 0,1,0,0,
                                 0,0,1,0,
                                 0,0,0,1);

    for (int i = 0; i < count; ++i) {

        Pw = Pw *P1_trans[i];

    }
	return  FirstFrame2Second(point,Pw.inv());//注意 此时第一视图为当前坐标，第二视图为世界坐标
}
Scalar ReprojErrorAndPointCloud(const vector<KeyPoint> &pt_set2, const Mat &K, const Matx34d &P1,
                                vector<CloudPoint> &pointcloud, const vector<Point3f> &points_3d) {
    vector<double> reproj_error;
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
        //correspImg1Pt.push_back(pt_set1[i]);

        //depths.push_back(X(2));
    }
    Scalar mse = mean(reproj_error);
    cout << "Done. \n\r"<<pointcloud.size()<<"points, " <<"mean square reprojetion err = " << mse[0] <<  endl;
    return mse;
}
double* CvMatrix2ArrayCamera( Mat R,Mat K,Mat t)
{
    double* camera = new double[9];
    Mat_<double> r(3,1);
    if(R.cols==3&&R.rows==3)
        Rodrigues(R,r);
     else
        r = R;
    camera[0] = r.at<double>(0,0);
    camera[1] = r.at<double>(1,0);
    camera[2] = r.at<double>(2,0);
    camera[3] = t.at<double>(0,0);
    camera[4] = t.at<double>(1,0);
    camera[5] = t.at<double>(2,0);
    camera[6] = K.at<double>(0,0);
    camera[7] = K.at<double>(0,2);
    camera[8] = K.at<double>(1,2);

    cout<<"相机参数： "<<camera<<endl;
    return camera;
}

double* CvPoint3f2ArrayPoint( Point3d p)
{
    double* point;
    point[0] = p.x;
    point[1] = p.y;
    point[2] = p.z;

    return point;
}

void SetOrdering( double* cameras ,double * points,const int num_cameras,const int num_points,ceres::Solver::Options* options)
{

    const int point_block_size = 3;
    const int camera_block_size = 9;

    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras
    for(int i = 0; i < num_points; ++i)
        ordering->AddElementToGroup(points + point_block_size * i, 0);


    for(int i = 0; i < num_cameras; ++i)
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);

    options->linear_solver_ordering.reset(ordering);

}
void SetMinimizerOptions(Solver::Options* options){
    options->max_num_iterations = 100;
    options->minimizer_progress_to_stdout = true;
    options->num_threads = 1;
    // options->eta = params.eta;
    // options->max_solver_time_in_seconds = params.max_solver_time;

    options->trust_region_strategy_type = LEVENBERG_MARQUARDT;


}
void SetLinearSolver(ceres::Solver::Options* options)
{

    options->linear_solver_type = DENSE_SCHUR;//不能是dense_qr  不然卡出翔，还要花一天的时间去找错误，解大型矩阵，最好稀疏。
    options->sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options->dense_linear_algebra_library_type = ceres::EIGEN;


    options->num_linear_solver_threads =1;

}
void BundleAdjustment(const vector<KeyPoint> keypoints_2,const v_pair kp_depth_idx,
                      const Mat R,const Mat K,const Mat t,vector<CloudPoint> &pointcloud)
{
    double* camera = CvMatrix2ArrayCamera(R,K,t);
    cout<<*camera<<endl;
    double* points = new double[3*pointcloud.size()];//考虑使用智能指针，待优化
    double* points_temp = points;
    for (int j = 0; j < pointcloud.size(); ++j)
    {
        *points_temp = pointcloud[j].pt.x;
        points_temp++;
        *points_temp = pointcloud[j].pt.y;
        points_temp++;
        *points_temp = pointcloud[j].pt.z;
        points_temp++;

    }
    double *observe = new double[kp_depth_idx.size()*2];
    cout<<" kp_depth_idx.size() :"<<kp_depth_idx.size()<<endl;
    double *observe_temp = observe;
    for (int k = 0; k < kp_depth_idx.size(); ++k) {
        *observe_temp = keypoints_2[kp_depth_idx[k].second].pt.x;
        observe_temp++;
        *observe_temp = keypoints_2[kp_depth_idx[k].second].pt.y;
        observe_temp++;

    }
    Problem problem;
    for (int i = 0; i < kp_depth_idx.size(); ++i)
    {
        CostFunction* cost_function;
        cost_function = SnavelyReprojectionError::Create(observe[2*i+0],observe[2*i+1]);
        LossFunction* loss_function =  new HuberLoss(1.0);

        double *point =points+3*i;
        std::cout<<"point adress: "<<point<<endl;
        std::cout<<"point value: "<<*point<<" "<<*(point+1)<<" "<<*(point+2)<<endl;

        problem.AddResidualBlock(cost_function, NULL, camera, point);




    }
    Solver::Options options;
    SetMinimizerOptions(&options);
    SetLinearSolver(&options);
   // SetOrdering(camera,points,1,pointcloud.size(),&options);

    options.gradient_tolerance = 1e-15;
    options.function_tolerance = 1e-15;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    delete points;
    delete camera;
}













