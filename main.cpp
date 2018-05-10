/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */
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
//#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/core/core.hpp>


#include "FeatureMatching.h"
#include "CalculateCameraMatrix.h"
#include "Triangulation.h"
#include "Common.h"
#include "SaveXYZimages.h"
#include <ArcBall.h>


#include <stdlib.h>
//#include <windows.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <math.h>




using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void readme();
//
//float imgdata[2448][3264][3];
//float texture[2448][3264][3];
int width=0, height=0, rx = 0, ry = 0;  
int eyex = 30, eyez = 20, atx = 100, atz = 50; 
int eyey = -15;
float scalar = 0.1;        //scalar of converting pixel color to float coordinates 
vector<CloudPoint> pointcloud;
float allx = 0.0;
float ally = 0.0;
float allz = 0.0;



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
	*/
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





////Function Main
int main( int argc, char** argv )
{

	Mat img_1 = imread("../1.png");
	Mat img_2 = imread("../2.png");
    std::vector< DMatch > matches;
	std::vector<KeyPoint> keypoints_1, keypoints_2,keypts1_good,keypts2_good, corr;
	width = img_1.cols;
	height = img_1.rows;

	if( !img_1.data || !img_2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; } /// Read in Images

	// Start Feature Matching
	int Method = 1;
	FeatureMatching(img_1,img_2,keypoints_1,keypoints_2,keypts1_good,keypts2_good,&matches,Method); // matched featurepoints
	 // Calculate Matrices
	vector<Point2f> pts1,pts2;
	vector<uchar> status;


	vector<KeyPoint> imgpts1_tmp,imgpts2_tmp,imgpts1_good,imgpts2_good;
	GetAlignedPointsFromMatch(keypoints_1, keypoints_2, matches, imgpts1_tmp, imgpts2_tmp);
	KeyPointsToPoints(imgpts1_tmp, pts1);
	KeyPointsToPoints(imgpts2_tmp, pts2);
	double minVal,maxVal;
	cv::minMaxIdx(pts1,&minVal,&maxVal);

	Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006*maxVal, 0.99, status);//maxVal 过滤掉误差较大的匹配点
	/*
	 *  status
    具有N个元素的输出数组，在计算过程中没有被舍弃的点，元素被被置为1；否则置为0。这个数组只可以在方法RANSAC and LMedS 情况下使用；
    在其它方法的情况下，status一律被置为1。这个参数是可选参数。
	 * */

	double status_nz = countNonZero(status);
	double status_sz = status.size();
	double kept_ratio = status_nz / status_sz;

	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;
	//给大佬递茶
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

    /*
	string filename = "C:\\OpenCV_Project\\camera_calibration\\result.xml";
	FileStorage fs(filename, FileStorage::READ);
	FileNode n = fs.getFirstTopLevelNode();
	fs["Camera_Matrix"] >> K;
	fs["Distortion_Coefficients"] >> discoeff;
	cout << "K " << endl << Mat(K) << endl;
     */
	Kinv = K.inv();

	Matx34d P, P1;
	
	

	bool CM = FindCameraMatrices(K,Kinv,F,P,P1,R_cv,t_cv,discoeff,imgpts1_tmp,imgpts2_tmp,imgpts1_good,imgpts2_good,matches,pointcloud);
	
	// Reconstruct 3D
	//double mse = TriangulatePoints(keypts1_good,keypts2_good,K,Kinv,P,P1,pointcloud,keypts1_good,discoeff);

	// Write points to file
	Mat X(img_1.rows,img_1.cols,CV_32FC1);
	Mat Y(img_1.rows,img_1.cols,CV_32FC1);
	Mat Z(img_1.rows,img_1.cols,CV_32FC1);
	string filepath = "./save/";
	//saveXYZimages(img_1,pointcloud,imgpts1_good,filepath,X,Y,Z);

    typedef pcl::PointXYZ Point_PCL;
    typedef pcl::PointCloud<Point_PCL> PointCloud;

    PointCloud::Ptr pointCloud( new PointCloud );



    for(int i=0;i<pointcloud.size();i++ )
    {
        Point_PCL p;
        p.x = pointcloud[i].pt.x;
        p.y= pointcloud[i].pt.y;
        p.z= pointcloud[i].pt.z;

        pointCloud->points.push_back( p );
    }
    pointCloud->is_dense = false;
    cout<<"点云共有"<<pointCloud->size()<<"个点."<<endl;
    pcl::io::savePCDFileBinary("pointCloud.pcd", *pointCloud );



	double Nindex = X.rows * X.cols;


	for(int i=0;i<pointcloud.size();i++ )
	{
		allx += pointcloud[i].pt.x;
		ally += pointcloud[i].pt.y;
		allz += pointcloud[i].pt.z;
	}
	allx = 1.0 * allx/(float)pointcloud.size();//相机所看向的位置，点太多，全面考虑所有点
	ally = 1.0 * ally/(float)pointcloud.size();
	allz = 1.0 * allz/(float)pointcloud.size();



	/*
	for(int i=0;i<X.rows;i++)
	{
		for(int j=0;j<X.cols;j++)
		{
			float* Xr =X.ptr<float>(i);
			imgdata[i][j][0] = Xr[j];
			float* TXr = img_1.ptr<float>(i);
			texture[i][j][0] = TXr[j];
		}
	}


	for(int i=0;i<Y.rows;i++)
	{
		for(int j=0;j<Y.cols;j++)
		{
			float* Yr =Y.ptr<float>(i);
			imgdata[i][j][1] = Yr[j];
			float* TYr = img_1.ptr<float>(i);
			texture[i][j][1] = TYr[j];
		}
	}


	for(int i=0;i<Z.rows;i++)
	{
		for(int j=0;j<Z.cols;j++)
		{
			float* Zr =Z.ptr<float>(i);
			imgdata[i][j][2] = Zr[j];
			float* TZr = img_1.ptr<float>(i);
			texture[i][j][2] = TZr[j];
		}
	}
	*/

	//////// OpenGL Draw

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

	//cvWaitKey(0);

	return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }