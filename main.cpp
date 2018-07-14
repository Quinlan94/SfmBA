


//pcl的库头文件放在最上面，不然会出现莫名奇妙的错误

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


#include <stdio.h>


#include "opencv2/features2d/features2d.hpp"

#include <opencv2/calib3d/calib3d.hpp>



#include <boost/filesystem.hpp>
#include <boost/format.hpp>


#include "FeatureMatching.h"
#include "CalculateCameraMatrix.h"
#include "Triangulation.h"
#include "Common.h"

#include <ArcBall.h>

#include <string>
#include <stdlib.h>

//#include <GL/glut.h>
//#include <GL/gl.h>
#include <math.h>
#include <Feature.h>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace boost::filesystem;




/*
int rx = 0, ry = 0;
int eyex = 30, eyez = 20, atx = 100, atz = 50; 
int eyey = -15;
*/





vector<Mat> images;


bool images_pair_is_initial = true;



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

    vector<CloudPoint> pointcloud;

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
    vector<v_keypoint> keypoints(n),orb_keypoints(n);//嵌套容器要初始化，不然内存出错
    vector<v_match> v_matches(n-1),best_matches(n-1),good_matches(n-1);
    vector<DMatch> v_new_matches;

    vector<Mat> TransMat(n-1);
    vector<Mat> ProjMat(n);

    Mat v_K,v_Kinv,v_discoeff;
    v_K = ( Mat_<double> ( 3,3 ) << 2905.88, 0, 1416, 0, 2905.88, 1064, 0, 0, 1  );
    v_discoeff = ( Mat_<double> ( 5,1 ) << 0, 0, 0, 0, 0 );



    vector<v_point> pts(n),tem_best_pts(n),best_pts(n);
    ;
    vector<v_pair> kp_good_depth_idx(n-1);
    vector<double> each_mean_reproj_error;
    unordered_set<int> kp_depth_idx;

    FeatureExtractor(images,keypoints,v_matches,Sift);

    for(int i = 0;i<images.size()-1;i++)
    {

        vector<v_keypoint> imgpts_good(2);
       // WriteFileMatches(images[i],images[i+1],v_matches[i],keypoints[i],keypoints[i+1],"../youwenti.jpg");
        double mid_dist;

        getGoodMatches(v_matches[i],good_matches[i],mid_dist,0);

        RansacGoodMatches(good_matches[i],keypoints[i],keypoints[i+1],imgpts_good[0],imgpts_good[1],
                          kp_good_depth_idx[i],kp_depth_idx);

        boost::format fmt = boost::format("../ImagesMatches_%1%—%2%.jpg")%i %(i+1);

        std::string name = fmt.str();

        WriteFileMatches(images[i],images[i+1],good_matches[i],keypoints[i],keypoints[i+1],name);
        /*FeatureMatching(images[i],images[i+1],keypoints[i],keypoints[i+1],orb_keypoints[i],orb_keypoints[i+1],
                       v_matches[i],best_matches[i],Orb,images_pair_is_initial,mid_dist);//重复累赘，待优化
        vector<pair<int,int>> kp_idx__temp;
        vector<uchar> status;
        vector<v_keypoint> imgpts_tmp(n),imgpts_good(n),keypoints_best(n);
        GetAlignedPointsFromMatch(keypoints[i], keypoints[i+1], v_matches[i],
                                  imgpts_tmp[i], imgpts_tmp[i+1],kp_idx__temp);
        KeyPointsToPoints(imgpts_tmp[i], pts[i]);
        KeyPointsToPoints(imgpts_tmp[i+1], pts[i+1]);

        double minVal,maxVal;
        cv::minMaxIdx(pts[i],&minVal,&maxVal);

        Mat F = findFundamentalMat(pts[i], pts[i+1], FM_RANSAC, 0.006*maxVal, 0.99, status);
        cout << "筛选后匹配点" << countNonZero(status) << " / " << status.size() << endl;

        for (unsigned int j=0; j<status.size(); j++)
        {
            if (status[j])
            {
                imgpts_good[i].push_back(imgpts_tmp[i][j]);
                imgpts_good[i+1].push_back(imgpts_tmp[i+1][j]);

                kp_depth_idx.insert(kp_idx__temp[j].second);
                kp_good_depth_idx[i].push_back(kp_idx__temp[j]);//成双成对

                v_new_matches.push_back(v_matches[i][j]);

            }
        }


        v_matches[i] = v_new_matches;
        Mat img_matches_surf;

        drawMatches( images[i], keypoints[i], images[i+1], keypoints[i+1],
                     v_matches[i], img_matches_surf, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        imwrite("../Image_surf_Matches.jpg",img_matches_surf);
        v_new_matches.clear();
        status.clear();

        //******************************************************************
        GetAlignedPointsFromMatch(orb_keypoints[i], orb_keypoints[i+1], best_matches[i],
                                  keypoints_best[i], keypoints_best[i+1]);

        KeyPointsToPoints(keypoints_best[i], tem_best_pts[i]);
        KeyPointsToPoints(keypoints_best[i+1], tem_best_pts[i+1]);
        double t_minVal,t_maxVal;
        cv::minMaxIdx(pts[i],&t_minVal,&t_maxVal);
        Mat f = findFundamentalMat(tem_best_pts[i], tem_best_pts[i+1], FM_RANSAC, 0.006*t_maxVal, 0.99, status);
        cout << "筛选后匹配点" << countNonZero(status) << " / " << status.size() << endl;

        for (unsigned int j=0; j<status.size(); j++)
        {
            if (status[j])
            {
                best_pts[i].push_back(tem_best_pts[i][j]) ;
                best_pts[i+1].push_back(tem_best_pts[i+1][j]) ;
                v_new_matches.push_back(best_matches[i][j]);


            }
        }
        best_matches[i] = v_new_matches;
        v_new_matches.clear();
        status.clear();
        Mat img_matches;
        vector<DMatch> samp;
        for (int l = 0; l < 10; ++l) {
            samp.push_back(v_matches[i][l]);
        }
        drawMatches( images[i], orb_keypoints[i], images[i+1], orb_keypoints[i+1],
                     best_matches[i], img_matches, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        imwrite("../Image_Matches_orb.jpg",img_matches);*/

//**************************************************************************************
        if(images_pair_is_initial)
        {

            vector<v_keypoint> orb_keypoints(n);
            vector<v_match> orb_matches(n-1),orb_good_matches(n-1);
            vector<Point2f> best_pts_1,best_pts_2;

            double mid_dist_orb;
            FeatureExtractor(images,orb_keypoints,orb_matches,Orb);

            getGoodMatches(orb_matches[i],orb_good_matches[i],mid_dist_orb,1);

            RansacGoodMatches(orb_good_matches[i],orb_keypoints[i],orb_keypoints[i+1],best_pts_1,best_pts_2);

            Point2d principal_point(1416, 1064);
            double focal_length = 2905.88;
            Mat essential_matrix;
            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            essential_matrix = findEssentialMat(best_pts_1,best_pts_2,focal_length,principal_point,RANSAC);
            cout << "E: "<<essential_matrix<<endl;
            WriteFileMatches(images[i],images[i+1],orb_good_matches[i],orb_keypoints[i],orb_keypoints[i+1],"../orb_best_matches.jpg");

            //Mat R_cv,t_cv;
            Mat_<double> R(3,3);
            Mat_<double> t(3,1);
            recoverPose(essential_matrix,best_pts_1,best_pts_2,R,t,focal_length,principal_point);

            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
            cout<<"对极约束算法用时："<<time_used.count()<<" 秒。"<<endl;
            cout <<"R: "<<R<<endl;
            cout <<"t: "<<t<<endl;

            v_Kinv = v_K.inv();

            ProjMat[i] = (Mat_<double>(4,4)<<1,0,0,0,
                                             0,1,0,0,
                                             0,0,1,0,
                                             0,0,0,1);
            ProjMat[i+1] = (Mat_<double>(4,4)<<R(0,0),	R(0,1),	R(0,2),	t(0),
                    R(1,0),	R(1,1),	R(1,2),	t(1),
                    R(2,0),	R(2,1),	R(2,2),	t(2),
                    0 ,         0,     0,      1);





            FindCameraMatrices(v_K,v_Kinv,ProjMat[i],ProjMat[i+1],v_discoeff,good_matches[i],
                               imgpts_good[0],imgpts_good[1],pointcloud,each_mean_reproj_error,images_pair_is_initial);
           // BundleAdjustment(imgpts_good[1],R,v_K,t,pointcloud);


            int k = 0;
            for(int j=0;j<pointcloud.size();j++ )
            {
                if(pointcloud[j].reprojection_error<max_reproj_err)
                {
                    Point_PCL p;
                    p.x = pointcloud[j].pt.x;
                    p.y = pointcloud[j].pt.y;
                    p.z = pointcloud[j].pt.z;
                    p.r= 255;
                    p.g=255;
                    p.b= 255;

                   pointCloud_PCL->points.push_back( p );
                    k++;
                }
            }

            pointCloud_PCL->points.push_back( DisplayCamera(ProjMat[i].inv()) );//绿

            cout<<i<<"次点云共有"<<k<<"个点."<<endl;
            k= 0;
            images_pair_is_initial = false;

        }
        else
        {


            vector<Point3d> pts_3d;
            vector<Point2d> pts_2d;
            int idx;
            vector<DMatch> sample,sample_1;
            vector<double > reproj;

            for (DMatch m:good_matches[i])
            {
                idx = 0;
                for (DMatch b:good_matches[i - 1])
                {
                    if (m.queryIdx == b.trainIdx)
                    {
                        if (
                             m.distance<mid_dist/4  //很有必要，如果去掉，重投影误差虽然小，但形状变了
                            &&pointcloud[idx].reprojection_error<max_reproj_err/2&&
                            pointcloud[idx].pt.z>0

                                )
                        {
                            pts_3d.push_back(
                                    //FirstFrame2Second(pointcloud[idx].pt, ProjMat[i]
                                    pointcloud[idx].pt
                            );//可以考虑筛选一些重投影误差较大点
                            pts_2d.push_back(
                                    keypoints[i + 1][m.trainIdx].pt);
                            reproj.push_back(pointcloud[idx].reprojection_error);
                            sample_1.push_back(b);
                            sample.push_back(m);
                            idx++;
                            break;
                        }
                    }
                    idx++;

                }


            }
            /*for (DMatch m:v_matches[i])
            {
                vector<pair<int,int>>::iterator it;
                if(!(kp_depth_idx.find(m.queryIdx) == kp_depth_idx.end()))//避免每次从头到尾的遍历
                {
                    for (it = kp_good_depth_idx[i-1].begin();it!= kp_good_depth_idx[i-1].end();it++)
                    {
                        if(it->second == m.queryIdx)//当前匹配对的queryIdx是否在上一匹配对中计算过深度
                        {

                            int index = distance(kp_good_depth_idx[i-1].begin(),it);//索引位置对应点云位置，即对应的深度
                            //对应的点是否算放错了，而导致pnp算法不准确的
                            if(pointcloud[index].pt.z>=0 && pointcloud[index].reprojection_error<=each_mean_reproj_error[i-1])
                            {
                                pts_3d.push_back(FirstFrame2Second(pointcloud[index].pt,TransMat[i-1]));//可以考虑筛选一些重投影误差较大点
                               pts_2d.push_back(keypoints[i+1][m.trainIdx].pt);//然而并没有什么屁用，反而更差。而且上一次优化的结果，对本次并没有什么影响。。什么鬼
                               break;
                            }
                        }
                    }
                }


            }*/
            WriteFileMatches(images[i-1],images[i],sample_1,keypoints[i-1],keypoints[i],"../pre_sample_Matches.jpg");
            WriteFileMatches(images[i],images[i+1],sample,keypoints[i],keypoints[i+1],"../curr_sample_Matches.jpg");

            pointcloud.clear();
            cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;
            Mat r;
            Mat_<double> R(3,3);
            Mat_<double> t(3,1);
            solvePnP( pts_3d, pts_2d, v_K, Mat(), r, t, false);//筛选后的点是否效果明显,很奇怪。

            cv::Rodrigues ( r, R );//旋转向量是个3维，那旋转角度呢，模是弧度

            cout<<"R="<<endl<<R<<endl;
            cout<<"t="<<endl<<t<<endl;
            v_Kinv = v_K.inv();
            Mat temp;

            v_Kinv = v_K.inv();

            ProjMat[i+1] = (Mat_<double>(4,4)<<R(0,0),	R(0,1),	R(0,2),	t(0),
                    R(1,0),	R(1,1),	R(1,2),	t(1),
                    R(2,0),	R(2,1),	R(2,2),	t(2),
                    0,        0,      0,      1);


            Mat_<double> Pose_R,Pose_t;
            Pose_R = (Mat_<double>(3,3)<<ProjMat[i+1].at<double>(0,0),ProjMat[i+1].at<double>(0,1),ProjMat[i+1].at<double>(0,2),
                    ProjMat[i+1].at<double>(1,0),ProjMat[i+1].at<double>(1,1),ProjMat[i+1].at<double>(1,2),
                    ProjMat[i+1].at<double>(2,0),ProjMat[i+1].at<double>(2,1),ProjMat[i+1].at<double>(2,2));
            Pose_t = (Mat_<double>(3,1)<<ProjMat[i+1].at<double>(0,3),
                    ProjMat[i+1].at<double>(1,3),
                    ProjMat[i+1].at<double>(2,3));


            FindCameraMatrices(v_K,v_Kinv,ProjMat[i],ProjMat[i+1],v_discoeff,good_matches[i],
                               imgpts_good[0],imgpts_good[1],pointcloud,each_mean_reproj_error,images_pair_is_initial);
           /* BundleAdjustment(imgpts_good[1],Pose_R,v_K,Pose_t,pointcloud);
            ProjMat[i+1] = (Mat_<double>(4,4)<<Pose_R(0,0),	Pose_R(0,1),	Pose_R(0,2),	Pose_t(0),
                    Pose_R(1,0),	Pose_R(1,1),	Pose_R(1,2),	Pose_t(1),
                    Pose_R(2,0),	Pose_R(2,1),	Pose_R(2,2),	Pose_t(2),
                    0,        0,      0,      1);
*/
            Mat rotation = Pose_R.t()*Pose_R;

            double n = determinant(rotation);
            cout<<"单位矩阵: "<<rotation<<"行列式："<< n<<endl;
            int k = 0;
           /* vector<int> pre,current,differ;
            vector<int>::iterator f;
            for(pair<int,int> q : kp_good_depth_idx[i])
                current.push_back(q.first);

            for(pair<int,int> t : kp_good_depth_idx[i-1])
                pre.push_back(t.second);
            std::sort(pre.begin(),pre.end(),less<int>());

            std::set_difference(current.begin(), current.end(), pre.begin(), pre.end(),
                                std::inserter(differ, differ.begin()));//之前最好排序，不然差集可能出错*/
            //添加之前未算过深度的点
/*
            for(auto v:differ)
            {
                for (f = current.begin();f!= current.end();f++)
                {
                    if(v == *f )
                    {
                        int index = distance(current.begin(),f);
                        if(pointcloud[index].reprojection_error <= each_mean_reproj_error[i])
                        {
                            Point_PCL p;
                            p.x = CurrentPt2World(pointcloud[index].pt, TransMat, i).x;
                            p.y = CurrentPt2World(pointcloud[index].pt, TransMat, i).y;
                            p.z = CurrentPt2World(pointcloud[index].pt, TransMat, i).z;

                            p.r= 50;
                            p.g=255;
                            p.b= 60;

                            pointCloud_PCL->points.push_back(p);

                            k++;
                        }


                    }

                }

            }

/*
            for(int j=0;j<pointcloud.size();j++ )
            {

                //if(pointcloud[j].reprojection_error < each_mean_reproj_error[i])
                {
                    Point_PCL p;
                    p.x = CurrentPt2World(pointcloud[j].pt,TransMat,i).x;
                    p.y = CurrentPt2World(pointcloud[j].pt,TransMat,i).y;
                    p.z = CurrentPt2World(pointcloud[j].pt,TransMat,i).z;
//                p.x = pointcloud[i].pt.x;
//                p.y = pointcloud[i].pt.y;
//                p.z = pointcloud[i].pt.z;
                    p.r= 50;
                    p.g=255;
                    p.b= 60;


                    pointCloud_PCL->points.push_back( p );
                    k++;
                }

            }*/

            bool trans = false;
            Mat temp_center;
            for(int j=0;j<pointcloud.size();j++ )
            {
                //if(pointcloud[j].pt.z>=0&&pointcloud[j].pt.z<=100)
                {
                    Point_PCL p;
                    Point3d p_cv;
                    if(!trans) {
                        p.x = pointcloud[j].pt.x;
                        p.y = pointcloud[j].pt.y;
                        p.z = pointcloud[j].pt.z;
                    }else {
                        p_cv = CurrentPt2World(pointcloud[j].pt, TransMat, i);
                        p.x = p_cv.x;
                        p.y = p_cv.y;
                        p.z = p_cv.z;
                    }
                   if(i==1)
                   {
                       p.r= 255;
                       p.g= 0;
                       p.b= 255;
                   }
                   else if(i==2)
                   {
                       p.r= 0;
                       p.g= 255;
                       p.b= 255;
                   } else if(i==3)
                   {
                       p.r= 0;
                       p.g= 255;
                       p.b= 127;
                   }
                   else if(i==4)
                   {
                       p.r= 255;
                       p.g= 215;
                       p.b= 0;
                   }


                    pointCloud_PCL->points.push_back( p );
                    k++;
                }
            }

            pointCloud_PCL->points.push_back( DisplayCamera(ProjMat[i].inv()) );
            if(i==(images.size()-2))
                pointCloud_PCL->points.push_back( DisplayCamera(ProjMat[i+1].inv()) );

            //kp_idx__temp.clear();
           // pointcloud.clear();
            cout<<i<<"次点云共有新添加"<<k<<"个点."<<endl;

        }
        imgpts_good.clear();

    }

#else

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

