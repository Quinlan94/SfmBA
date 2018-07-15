


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
void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
                               const std::vector<cv::KeyPoint>& imgpts2,
                               const std::vector<cv::DMatch>& matches,
                               std::vector<cv::KeyPoint>& pt_set1,
                               std::vector<cv::KeyPoint>& pt_set2)
{
    for (unsigned int i=0; i<matches.size(); i++)
    {

        pt_set1.push_back(imgpts1[matches[i].queryIdx]);
        pt_set2.push_back(imgpts2[matches[i].trainIdx]);
    }
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) 
{
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++)
        ps.push_back(kps[i].pt);
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

        Pw =  P1_trans[i]*Pw;

    }
	return  FirstFrame2Second(point,Pw);//注意 此时第一视图为当前坐标，第二视图为世界坐标
}
Scalar ReprojErrorAndPointCloud(vector<KeyPoint> &pt_set2, const Mat &K, const Matx34d &P1,
                                vector<CloudPoint> &pointcloud, const vector<Point3d> &points_3d,std::vector<cv::DMatch>& good_matches) {


    std::vector<cv::DMatch> temp_matches;
    vector<KeyPoint> temp_pt_set2;
    vector<double> reproj_error;
    Mat_<double> KP1 = K * Mat(P1);
    Mat_<double> X = Mat_<double>(4,1);
    for (int i=0; i<points_3d.size(); i++)
    {

            X << points_3d[i].x, points_3d[i].y, points_3d[i].z, 1;
            Mat_<double> xPt_img = KP1 * X;
            Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

            double reprj_err = norm(xPt_img_ - pt_set2[i].pt);


            if(reprj_err<max_reproj_err)
            {
                reproj_error.push_back(reprj_err);
                CloudPoint cp;
                cp.pt = Point3d(X(0), X(1), X(2));
                cp.reprojection_error = reprj_err;
                pointcloud.push_back(cp);
                temp_matches.push_back(good_matches[i]);
                temp_pt_set2.push_back(pt_set2[i]);
            }






    }
    good_matches = temp_matches;
    pt_set2 = temp_pt_set2;
    Scalar mse = mean(reproj_error);
    cout << "Done. \n\r"<<pointcloud.size()<<"points, " <<"mean square reprojetion err = " << mse[0] <<  endl;
    return mse;
}
Scalar ReprojErrorAndPointCloud(vector<KeyPoint> &pt_set2, const Mat &K, const Matx34d &P1,
                                vector<CloudPoint> &pointcloud, const vector<Point3d> &points_3d) {

    vector<CloudPoint> max_err_pointcloud;
    vector<double> reproj_error;
    Mat_<double> KP1 = K * Mat(P1);
    Mat_<double> X = Mat_<double>(4,1);
    for (int i=0; i<points_3d.size(); i++)
    {

        X << points_3d[i].x, points_3d[i].y, points_3d[i].z, 1;
        Mat_<double> xPt_img = KP1 * X;
        Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

        double reprj_err = norm(xPt_img_ - pt_set2[i].pt);

        reproj_error.push_back(reprj_err);

        CloudPoint cp;
        cp.pt = Point3d(X(0), X(1), X(2));
        cp.reprojection_error = reprj_err;

        pointcloud.push_back(cp);

    }
    Scalar mse = mean(reproj_error);
    cout << "Done. \n\r"<<pointcloud.size()<<"points, " <<"mean square reprojetion err = " << mse[0] <<  endl;
    return mse;
}
double* CvMatrix2ArrayCamera( Mat R,Mat t)
{
    double* camera = new double[6];
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


    cout<<"位姿参数： "<<camera<<endl;
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
Point_PCL DisplayCamera(const Mat temp_center)
{
    Point_PCL center;
    Point3d p_cv;
    center.x = temp_center.at<double>(0, 3);
    center.y = temp_center.at<double>(1, 3);
    center.z = temp_center.at<double>(2, 3);
    center.r= 55;
    center.g=255;
    center.b= 55;
    return center;
}
bool cl_greater(const DMatch& a,const DMatch& b)
{
    return  a.distance < b.distance;
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


    options->trust_region_strategy_type = LEVENBERG_MARQUARDT;


}
void SetLinearSolver(ceres::Solver::Options* options)
{

    options->linear_solver_type = DENSE_SCHUR;//不能是dense_qr  不然卡出翔，还要花一天的时间去找错误，解大型矩阵，最好稀疏。
    options->sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options->dense_linear_algebra_library_type = ceres::EIGEN;


    options->num_linear_solver_threads =1;

}
void BundleAdjustment(vector<KeyPoint> keypoints_1_depth,vector<KeyPoint> keypoints_2_depth,
                      Mat &Proj_1,Mat &Proj_2,Mat& K,vector<CloudPoint> &pointcloud)
{
    //double* pose_2 = CvMatrix2ArrayCamera(R,t);
    Mat r_mat_1 = Proj_1(Range(0,3),Range(0,3));
    Mat r_vec_1;
    Rodrigues(r_mat_1,r_vec_1);
    const Eigen::Vector3d r_vec_const(r_vec_1.at<double>(0),r_vec_1.at<double>(1),r_vec_1.at<double>(2));
    const Eigen::Vector3d t_vec_const(Proj_1.at<double>(0,3),Proj_1.at<double>(1,3),Proj_1.at<double>(2,3));
    const Eigen::Vector3d camera_param(K.at<double>(0,0),K.at<double>(0,2),K.at<double>(1,2));

    Mat r_mat_2 = Proj_2(Range(0,3),Range(0,3));
    Mat t_vec_2;
    t_vec_2 = Proj_2.col(3);
    double* pose_2 = CvMatrix2ArrayCamera(r_mat_2,t_vec_2);





    //std::cout<<"camera value: "<<*camera<<" "<<*(camera+1)<<" "<<*(camera+2)<<endl;
    double* points = new double[3*pointcloud.size()];
    double* points_temp = points;
    for (int j = 0; j < pointcloud.size(); ++j)
    {
        points[3*j+0] = pointcloud[j].pt.x;
        points[3*j+1] = pointcloud[j].pt.y;
        points[3*j+2] = pointcloud[j].pt.z;

    }
    double *observe_2 = new double[keypoints_2_depth.size()*2];
   /* double *observe_temp = observe;
    for (int k = 0; k < keypoints_2_depth.size(); ++k) {
        *observe_temp = keypoints_2_depth[k].pt.x;
        observe_temp++;
        *observe_temp = keypoints_2_depth[k].pt.y;
        observe_temp++;

    }*/
    Problem problem;

    for (int i = 0; i < keypoints_2_depth.size(); ++i)
    {
        Eigen::Vector2d observe_point_1(keypoints_1_depth[i].pt.x,keypoints_1_depth[i].pt.y);
        Eigen::Vector2d observe_point_2(keypoints_2_depth[i].pt.x,keypoints_2_depth[i].pt.y);
        CostFunction* cost_function;
        cost_function = SnavelyReprojectionError::Create(observe_point_1,observe_point_2,
                                                         r_vec_const,t_vec_const,camera_param);
        LossFunction* loss_function =  new HuberLoss(1.0);

        double *point =points+3*i;

        problem.AddResidualBlock(cost_function, loss_function, pose_2, point);




    }
    Solver::Options options;
    SetMinimizerOptions(&options);
    SetLinearSolver(&options);
   // SetOrdering(camera,points,1,pointcloud.size(),&options);

    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    Mat_<double> r_temp(3,1),t_temp(3,1);
    r_temp<<pose_2[0],pose_2[1],pose_2[2];
    Rodrigues(r_temp,r_mat_2);
    cout<<"R : "<<r_mat_2<<endl;
    t_temp<<pose_2[3],pose_2[4],pose_2[5];

    cout<<"t : "<<t_temp<<endl;
    Mat P1;

    Proj_2 = (Mat_<double>(4,4)<<r_mat_2.at<double>(0,0),	r_mat_2.at<double>(0,1),	r_mat_2.at<double>(0,2),	t_temp.at<double>(0),
            r_mat_2.at<double>(1,0),	r_mat_2.at<double>(1,1),	r_mat_2.at<double>(1,2),	t_temp.at<double>(1),
            r_mat_2.at<double>(2,0),	r_mat_2.at<double>(2,1),	r_mat_2.at<double>(2,2),	t_temp.at<double>(2),
            0,0,0,1);
    P1 = Proj_2(Range(0,3),Range::all());


    vector<Point3d> points_3d;
    for (int m = 0; m <pointcloud.size(); ++m) {
        Point3d p;
        p.x = points[3*m+0];
        p.y = points[3*m+1];
        p.z = points[3*m+2];

        points_3d.push_back(p);
    }
    cout<<"points_3d.size()"<<points_3d.size()<<endl;
    pointcloud.clear();
    Scalar mse = ReprojErrorAndPointCloud(keypoints_2_depth, K, P1, pointcloud, points_3d);
    cout<<"重投影误差： "<<mse[0]<<endl;



    std::cout<<"point value: "<<*points<<" "<<*(points+1)<<" "<<*(points+2)<<endl;




    delete points;
    delete pose_2;

}













