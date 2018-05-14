/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */


//pcl的库头文件放在最上面，不然会出现莫名奇妙的错误
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/core/core.hpp>

#include <boost/filesystem.hpp>


#include "FeatureMatching.h"
#include "CalculateCameraMatrix.h"
#include "Triangulation.h"
#include "Common.h"

#include <ArcBall.h>

#include <string>
#include <stdlib.h>

#include <GL/glut.h>
#include <GL/gl.h>
#include <math.h>




using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace boost::filesystem;




int rx = 0, ry = 0;
int eyex = 30, eyez = 20, atx = 100, atz = 50; 
int eyey = -15;
float scalar = 0.1;        //scalar of converting pixel color to float coordinates 
vector<CloudPoint> pointcloud;
float allx = 0.0;
float ally = 0.0;
float allz = 0.0;

typedef pcl::PointXYZ Point_PCL;
typedef pcl::PointCloud<Point_PCL> PointCloud;

vector<Mat> images;
typedef vector<KeyPoint> v_keypoint;
typedef vector<DMatch> v_match;
typedef vector<Point2f> v_point;

bool images_pair_is_initial = false;



/*
void special(int key, int x, int y)  
{  
    switch(key)  
    {  
    case GLUT_KEY_LEFT:  
        ry-=5;  
        glutPostRedisplay();  
        break;  
    case GLUT_KEY_RIGHT:  
        ry+=5;  
        glutPostRedisplay();  
        break;  
    case GLUT_KEY_UP:  
        rx+=5;  
        glutPostRedisplay();  
        break;  
    case GLUT_KEY_DOWN:  
        rx-=5;  
        glutPostRedisplay();  
        break;  
    }  
}  
  
//////////////////////////////////////////////////////////////////////////  

void renderScene(void) {  
   cout<<"renderScene "<<endl;
    glClear (GL_COLOR_BUFFER_BIT);  
    glLoadIdentity();// 将当前的用户坐标系的原点移到了屏幕中心：类似于一个复位操作
    gluLookAt (eyex, eyey, eyez, allx, ally, allz, 0.0, 1.0, 0.0);    //  眼睛所在的位置，和所看方向，以及指定朝上方向
    glRotatef(ry, 0.0, 1.0, 0.0);           //

	glRotatef(rx, 1.0, 0.0, 0.0);

    glPopMatrix();
    //gluLookAt (eyex, eyey, eyez, allx, ally, allz, 0.0, 1.0, 0.0);
    glRotatef(angle,axis(0),axis(1),axis(2));//奇怪 每次鼠标必须接在上一个点的位置旋转，才能正常，需要再研究一下
    glPushMatrix();
    float x,y,z;
  
    glPointSize(1.0);   
    glBegin(GL_POINTS);
	for(int i=0;i<pointcloud.size();i++)
	{
		glColor3f(255,255,255);
		x = -(pointcloud[i].pt.x - allx)/scalar;        // 
		y = -(pointcloud[i].pt.y - ally)/scalar;     
		z = (pointcloud[i].pt.z - allz)/scalar;  
		glVertex3f(x,y,z); 
	}
	/*
    for (int i=0;i<height;i++){   
        for (int j=0;j<width;j++){  
            glColor3f(texture[i][j][0]/255, texture[i][j][1]/255, texture[i][j][2]/255);    //  
            x=-imgdata[i][j][0]/scalar;        // 
            y=-imgdata[i][j][1]/scalar;   
            z=imgdata[i][j][2]/scalar;   
            glVertex3f(x,y,z);   
        }  
    }  

    glEnd();  
    glFlush();  
}  
  
//////////////////////////////////////////////////////////////////////////  

void reshape (int w, int h) {
    cout<<"reshape "<<endl;
    glViewport (0, 0, (GLsizei)w, (GLsizei)h);  
    glMatrixMode (GL_PROJECTION);  //表明接下来要做透视投影操作
    glLoadIdentity ();  //然后把矩阵设为单位矩阵：
    gluPerspective (60, (GLfloat)w / (GLfloat)h, 1.0, 5000.0);// 它们生成的矩阵会与当前的矩阵相乘,生成透视的效果
    glMatrixMode (GL_MODELVIEW);  
}
*/


int main( int argc, char** argv )
{

    PointCloud::Ptr pointCloud_PCL( new PointCloud );
    boost::filesystem::path images_dir("../Images");
    if(!exists(images_dir))
    {
        cout<< " 该目录不存在！！！ " << std::endl;
        return -1;
    }
	vector<Mat> images;
    vector<string> paths;
	boost::filesystem::directory_iterator  iters(images_dir);
    boost::filesystem::directory_iterator  end;
    while(iters != end)
    {
		boost::filesystem::path p = *iters;
        paths.push_back(path(*iters).string());

        iters++;

    }
    std::sort(paths.begin(),paths.end(),std::less<string>());
    for (int i = 0; i<paths.size();i++)
	{
		Mat img = imread(paths[i]);
		images.push_back(img);
	}
#if 1
    int n = images.size();
    vector<v_keypoint> keypoints(n),keypoints_good(n);//嵌套容器要初始化，不然内存出错
    vector<v_match> v_matches(n);
    vector<DMatch> v_new_matches;

    Mat P1_trans;
    Matx34d P;
    P1_trans = (Mat_<double>(4,4)<<1,0,0,0,
                            0,1,0,0,
                            0,0,1,0,
                            0,0,0,1);
    cout << "Testing P1_trans " << endl << P1_trans << endl;

    P= Matx34d(1,0,0,0,
               0,1,0,0,
               0,0,1,0);

    Mat v_K,v_Kinv,v_discoeff;
    v_K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    v_discoeff = ( Mat_<double> ( 5,1 ) << 0, 0, 0, 0, 0 );


    for(int i = 0;i<images.size()-1;i++)
    {

        FeatureMatching(images[i],images[i+1],keypoints[i],keypoints[i+1],
                        keypoints_good[i],keypoints_good[i+1],&v_matches[i],1);
        vector<v_point> pts(n);
        vector<int> kp_idx__temp;
        set<int> kp_depth_idx,kp_depth_good_idx;

        vector<uchar> status;
        vector<v_keypoint> imgpts_tmp(n),imgpts_good(n);
        GetAlignedPointsFromMatch(keypoints[i], keypoints[i+1], v_matches[i],
                                  imgpts_tmp[i], imgpts_tmp[i+1],kp_idx__temp);
        KeyPointsToPoints(imgpts_tmp[i], pts[i]);
        KeyPointsToPoints(imgpts_tmp[i+1], pts[i+1]);
        double minVal,maxVal;
        cv::minMaxIdx(pts[i],&minVal,&maxVal);

        Mat F = findFundamentalMat(pts[i], pts[i+1], FM_RANSAC, 0.006*maxVal, 0.99, status);

        for (unsigned int j=0; j<status.size(); j++)
        {
            if (status[j])
            {
                imgpts_good[i].push_back(imgpts_tmp[i][j]);
                imgpts_good[i+1].push_back(imgpts_tmp[i+1][j]);

                kp_depth_idx.insert(kp_idx__temp[j]);

                v_new_matches.push_back(v_matches[i][j]);


            }
        }
		v_matches[i] = v_new_matches;
        if(!images_pair_is_initial)
        {
            Point2d principal_point(325.1,249.7);
            int focal_length = 521;
            Mat essential_matrix;
            essential_matrix = findEssentialMat(pts[i],pts[i+1],focal_length,principal_point,RANSAC);
            cout << "E: "<<essential_matrix<<endl;

            //Mat R_cv,t_cv;
            Mat_<double> R(3,3);
            Mat_<double> t(1,3);
            recoverPose(essential_matrix,pts[i],pts[i+1],R,t,focal_length,principal_point);
            cout <<"R: "<<R<<endl;
            cout <<"t: "<<t<<endl;


            Mat P1_temp;
            Matx34d P1;

            v_Kinv = v_K.inv();
            P1_temp = Mat_<double>(4,4)<<(R(0,0),	R(0,1),	R(0,2),	t(0),
                                          R(1,0),	R(1,1),	R(1,2),	t(1),
                                          R(2,0),	R(2,1),	R(2,2),	t(2),
                                          0 ,         0,     0,      1);


            P1_trans = P1_trans * P1_temp;

            cout << "Testing P1_trans " << endl << P1_trans << endl;
            P1 = Matx34d (P1_trans.at<double>(0,0),	P1_trans.at<double>(0,1),	P1_trans.at<double>(0,2),	P1_trans.at<double>(0,3),
                         P1_trans.at<double>(1,0),	P1_trans.at<double>(1,1),	P1_trans.at<double>(1,2),	P1_trans.at<double>(1,3),
                         P1_trans.at<double>(2,0),	P1_trans.at<double>(2,1),	P1_trans.at<double>(2,2),	P1_trans.at<double>(2,3));
            cout << "Testing P " << endl << P << endl;
            cout << "Testing P1 " << endl << P1 << endl;
            FindCameraMatrices(v_K,v_Kinv,F,P,P1,R,t,v_discoeff,
                                         imgpts_good[i],imgpts_good[i+1],v_matches[i],pointcloud,kp_depth_good_idx);
            for(int i=0;i<pointcloud.size();i++ )
            {
                Point_PCL p;
                p.x = pointcloud[i].pt.x;
                p.y = pointcloud[i].pt.y;
                p.z = pointcloud[i].pt.z;

                pointCloud_PCL->points.push_back( p );
            }
            pointcloud.clear();
            images_pair_is_initial = true;

        } else{

            vector<Point3d> pts_3d;
            vector<Point2d> pts_2d;
            for (DMatch m:v_matches[i])
            {
                if(kp_depth_idx.find(m.queryIdx)==kp_depth_idx.end())
                {
                    pts_3d.push_back(pointcloud[i].pt);
                    pts_2d.push_back(keypoints[i+1][m.trainIdx].pt);



                }
            }
            cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;
            Mat r;
            Mat_<double> R(3,3);
            Mat_<double> t(1,3);
            solvePnP ( pts_3d, pts_2d, v_K, Mat(), r, t, false );

            cv::Rodrigues ( r, R );

            cout<<"R="<<endl<<R<<endl;
            cout<<"t="<<endl<<t<<endl;
            v_Kinv = v_K.inv();


            Mat P1_temp;
            Matx34d P1;

            v_Kinv = v_K.inv();
            P1_temp = Mat_<double>(4,4)<<(R(0,0),	R(0,1),	R(0,2),	t(0),
                                          R(1,0),	R(1,1),	R(1,2),	t(1),
                                          R(2,0),	R(2,1),	R(2,2),	t(2),
                                          0 ,         0,     0,      1);
            P1_trans = P1_trans * P1_temp;
            P1 = Matx34d (P1_trans.at<double>(0,0),	P1_trans.at<double>(0,1),	P1_trans.at<double>(0,2),	P1_trans.at<double>(0,3),
                          P1_trans.at<double>(1,0),	P1_trans.at<double>(1,1),	P1_trans.at<double>(1,2),	P1_trans.at<double>(1,3),
                          P1_trans.at<double>(2,0),	P1_trans.at<double>(2,1),	P1_trans.at<double>(2,2),	P1_trans.at<double>(2,3));


            FindCameraMatrices(v_K,v_Kinv,F,P,P1,R,t,v_discoeff,
                               imgpts_good[i],imgpts_good[i+1],v_matches[i],pointcloud,kp_depth_good_idx);
            for(int i=0;i<pointcloud.size();i++ )
            {
                Point_PCL p;
                p.x = pointcloud[i].pt.x;
                p.y = pointcloud[i].pt.y;
                p.z = pointcloud[i].pt.z;

                pointCloud_PCL->points.push_back( p );
            }
            pointcloud.clear();
        }





    }

#else

	Mat img_1 = imread("../1.png");
	Mat img_2 = imread("../2.png");
    std::vector< DMatch > matches;
	std::vector<KeyPoint> keypoints_1, keypoints_2,keypts1_good,keypts2_good;

	if( !img_1.data || !img_2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; } /// Read in Images

	// 匹配
	int Method = 1;
	FeatureMatching(img_1,img_2,keypoints_1,keypoints_2,keypts1_good,keypts2_good,&matches,Method); // matched featurepoints
	 // Calculate Matrices
	vector<Point2f> pts1,pts2;
	vector<uchar> status;
    vector<int> shit;


	vector<KeyPoint> imgpts1_tmp,imgpts2_tmp,imgpts1_good,imgpts2_good;
	GetAlignedPointsFromMatch(keypoints_1, keypoints_2, matches, imgpts1_tmp, imgpts2_tmp,shit);
	KeyPointsToPoints(imgpts1_tmp, pts1);//点按顺序排列，并且已对齐
	KeyPointsToPoints(imgpts2_tmp, pts2);
	double minVal,maxVal;
	cv::minMaxIdx(pts1,&minVal,&maxVal);

	Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006*maxVal, 0.99, status);//maxVal 过滤掉误差较大的匹配点
	/*
	 *  status
    具有N个元素的输出数组，在计算过程中没有被舍弃的点，元素被置为1；否则置为0。这个数组只可以在方法RANSAC and LMedS 情况下使用；
    在其它方法的情况下，status一律被置为1。这个参数是可选参数。
	 * */

	double status_nz = countNonZero(status);
	double status_sz = status.size();
	double kept_ratio = status_nz / status_sz;

	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;
	//剩余的点
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			new_matches.push_back(matches[i]);

			//good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
		}
	}	
	
	cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches; //keep only those points who survived the fundamental matrix

	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );		
	//-- Show detected matches
	imshow( "Feature Matches", img_matches );
	waitKey(30);
	destroyWindow("Feature Matches");
	imwrite("Image_Matches.jpg",img_matches);



	/////////////////////
	Mat K,Kinv,discoeff; // Read from calibration file
    K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    discoeff = ( Mat_<double> ( 5,1 ) << 0, 0, 0, 0, 0 );
    Point2d principal_point(325.1,249.7);
    int focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(pts1,pts2,focal_length,principal_point,RANSAC);
    cout << "E: "<<essential_matrix<<endl;

    Mat R_cv,t_cv;
    recoverPose(essential_matrix,pts1,pts2,R_cv,t_cv,focal_length,principal_point);
    cout <<"R: "<<R_cv<<endl;
    cout <<"t: "<<t_cv<<endl;


	Kinv = K.inv();

	Matx34d P, P1;
    set<int> kp_idx_hell;
	

	bool CM = FindCameraMatrices(K,Kinv,F,P,P1,R_cv,t_cv,discoeff,imgpts1_good,imgpts2_good,matches,pointcloud,kp_idx_hell);



//	Mat X(img_1.rows,img_1.cols,CV_32FC1);
//	Mat Y(img_1.rows,img_1.cols,CV_32FC1);
//	Mat Z(img_1.rows,img_1.cols,CV_32FC1);
//	string filepath = "./save/";
// saveXYZimages(img_1,pointcloud,imgpts1_good,filepath,X,Y,Z);







    for(int i=0;i<pointcloud.size();i++ )
    {
        Point_PCL p;
        p.x = pointcloud[i].pt.x;
        p.y= pointcloud[i].pt.y;
        p.z= pointcloud[i].pt.z;

        pointCloud_PCL->points.push_back( p );
    }
#endif
    pointCloud_PCL->is_dense = false;
    cout<<"点云共有"<<pointCloud_PCL->size()<<"个点."<<endl;
    pcl::io::savePCDFileBinary("../pointCloud_PCL.pcd", *pointCloud_PCL );


   /*
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(WIDTH,HEIGHT);
	glutCreateWindow("3D Reconstruct Model");
	glutReshapeFunc (reshape);//自适应屏幕窗口大小的改变，图形比例不改变
	glutDisplayFunc(renderScene);//这个函数告诉 GLUT 当窗口内容必须被绘制时,那个函数将被调用
	glutSpecialFunc(special);
    glutMouseFunc(mousePressEvent);
    glutMotionFunc(mouseMoveEvent);
	glutPostRedisplay();    
	glutMainLoop();  
*/


	return 0;
}

