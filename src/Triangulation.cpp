

#include "Triangulation.h"

#include <iostream>

using namespace std;
using namespace cv;

#undef __SFM__DEBUG__



Point2f pixel2cam (const Point2d& p, const Mat& K )
{
    return Point2f
            (
                    ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0),
                    ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1)
            );
}
double DegToRad(const double deg) {
return deg * 0.0174532925199432954743716805978692718781530857086181640625;
}
double RadToDeg(const double rad){
return rad * 57.29577951308232286464772187173366546630859375;
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
Eigen::Vector3d TriangulateMultiViewPoint(
        const std::vector<Eigen::Matrix3x4d>& proj_matrices,
        const std::vector<Eigen::Vector2d>& points) {


    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

    for (size_t i = 0; i < points.size(); i++) {
        const Eigen::Vector3d point = points[i].homogeneous().normalized();
        const Eigen::Matrix3x4d term =
                proj_matrices[i] - point * point.transpose() * proj_matrices[i];
        A += term.transpose() * term;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

    return eigen_solver.eigenvectors().col(0).hnormalized();
}


//Triagulate points
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
                         bool initial) {

    /*vector<double> depths;


    pointcloud.clear();
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
    cout<< "T1: "<<T1<<endl;
    Mat T2 = (Mat_<double>(3,4)<<P1(0,0),P1(0,1),P1(0,2),P1(0,3),
                                 P1(1,0),P1(1,1),P1(1,2),P1(1,3),
                                 P1(2,0),P1(2,1),P1(2,2),P1(2,3));

    Mat R = (Mat_<double>(3,3)<<P1(0,0),P1(0,1),P1(0,2),
                                P1(1,0),P1(1,1),P1(1,2),
                                P1(2,0),P1(2,1),P1(2,2));

    Mat t0 = (Mat_<double>(3,1)<<P1(0,3),P1(1,3),P1(2,3));


    cout<< "T2: "<<T2<<endl;*/


    double min_angle;
    if(initial)
        min_angle = min_initial_angle;
    else
        min_angle = min_fliter_angle;

    Eigen::Matrix3x4d proj_matrix_1 = CvMatToEigen34(P);
    Eigen::Matrix3x4d proj_matrix_2 = CvMatToEigen34(P1);
    std::vector<cv::DMatch> temp_matches;
    vector<KeyPoint> temp_pt_set1;
    vector<KeyPoint> temp_pt_set2;

    cout << "Triangulating Now . . ." << endl;
    double t = getTickCount();

    unsigned int pts_size = pt_set1.size();


    vector<Point2f> _pt_set1_pt, _pt_set2_pt;

    for (int j = 0; j < pt_set1.size(); j++) {
        //要转化为相机的归一化坐标
        _pt_set1_pt.push_back(pixel2cam(pt_set1[j].pt, K));
        _pt_set2_pt.push_back(pixel2cam(pt_set2[j].pt, K));
    }

    Mat pts_4d;
    vector<Point3d> points_3d;
    cv::triangulatePoints(P, P1, _pt_set1_pt, _pt_set2_pt, pts_4d);
    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
        );
        Eigen::Vector3d xyz = CvPointToVector3d(p);
        double angle = CalculateTriangulationAngle(proj_center_1, proj_center_2, xyz);
        if (RadToDeg(angle) >= min_angle &&
                HasPointPositiveDepth(proj_matrix_1, xyz) &&
                HasPointPositiveDepth(proj_matrix_2, xyz))
        {
            points_3d.push_back(p);
            temp_matches.push_back(matches[i]);
            temp_pt_set1.push_back(pt_set1[i]);
            temp_pt_set2.push_back(pt_set2[i]);
        }

    }
        matches = temp_matches;
        pt_set1 = temp_pt_set1;
        pt_set2 = temp_pt_set2;
        cout << "三角测量点： " << points_3d.size() << endl;

        Scalar mse = ReprojErrorAndPointCloud(pt_set2, K, P1, pointcloud, points_3d,matches);
        return mse[0];



}



double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                       const Eigen::Vector3d& proj_center2,
                                       const Eigen::Vector3d& point3D)
{

const double baseline2 = (proj_center1 - proj_center2).squaredNorm();

const double ray1 = (point3D - proj_center1).norm();
const double ray2 = (point3D - proj_center2).norm();


  const double angle = std::abs(
		std::acos((ray1 * ray1 + ray2 * ray2 - baseline2) / (2 * ray1 * ray2)));

  if (angle!=angle) {
     return 0;
    } else {
      return std::min(angle, M_PI - angle);
  }
}


bool HasPointPositiveDepth(const Eigen::Matrix3x4d& proj_matrix,
                           const Eigen::Vector3d& point3D) {

    return (proj_matrix(2, 0) * point3D(0) + proj_matrix(2, 1) * point3D(1) +
            proj_matrix(2, 2) * point3D(2) + proj_matrix(2, 3)) >
           std::numeric_limits<double>::epsilon();//什么道理,z的位置。
}
Eigen::Matrix3x4d CvMatToEigen34(Mat cv_matrix)
{
    if(cv_matrix.cols>=3 && cv_matrix.rows>=3)
    {
        Eigen::Matrix3x4d eigen;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                eigen(i,j) = cv_matrix.at<double>(i,j);
            }
        }
        return eigen;
    }
}
Eigen::Vector3d CvPointToVector3d(Point3d p)
{
    Eigen::Vector3d vec_3;
    vec_3(0) = p.x;
    vec_3(1) = p.y;
    vec_3(2) = p.z;
    return vec_3;
}
