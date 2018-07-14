//
// Created by quinlan on 18-7-6.
//

#include "Feature.h"

void FeatureExtractor(vector<Mat> images,vector<v_keypoint>& keypoints,vector<v_match>& v_matches,FeatureExtract method)
{
    vector<Mat> descriptors(images.size());
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cout<<"################....特征提取中.....##################"<<endl;
    switch (method)
    {
        case Surf:
        {
            double minHessian = 400;
            int nOctaves = 4;
            int nOctaveLayers = 3;
            Ptr<SurfFeatureDetector> detector = SURF::create(minHessian, nOctaves, nOctaveLayers);
            Ptr<SurfDescriptorExtractor> extractor = SURF::create(minHessian, nOctaves, nOctaveLayers);
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
            for (int i = 0; i < images.size(); ++i) {

                detector->detect(images.at(i), keypoints.at(i));
                extractor->compute(images.at(i), keypoints.at(i), descriptors.at(i));
            }
            for (int j = 0; j < images.size()-1; ++j) {

                matcher->match(descriptors.at(j), descriptors.at(j + 1), v_matches.at(j));

            }
            break;
        }
        case Orb:
        {
            Ptr<FeatureDetector> detector = ORB::create();
            Ptr<DescriptorExtractor> extractor = ORB::create();

            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

            for (int i = 0; i < 2; ++i) {

                detector->detect(images.at(i), keypoints.at(i));
                extractor->compute(images.at(i), keypoints.at(i), descriptors.at(i));
            }
            for (int j = 0; j < 1; ++j) {

                matcher->match(descriptors.at(j), descriptors.at(j + 1), v_matches.at(j));

            }

            break;
        }
        case Sift:
        {
            int nfeatures = 0;
            int nOctaveLayers = 3;
            double contrastThreshold = 0.04;
            double edgeThreshold = 10;
            double sigma = 1.5;
                Ptr<SiftFeatureDetector> detector = SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);

                Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);

                Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
//#pragma omp parallel for
                for (int i = 0; i < images.size(); ++i) {

                    detector->detect(images.at(i), keypoints.at(i));
                    extractor->compute(images.at(i), keypoints.at(i), descriptors.at(i));
                }

                for (int j = 0; j < images.size() - 1; ++j) {

                    matcher->match(descriptors.at(j), descriptors.at(j + 1), v_matches.at(j));
                }

                break;

        }


    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"特征提取用时："<<time_used.count()<<" 秒。"<<endl;
}
bool getGoodMatches(vector<DMatch> &v_matches, vector<DMatch> &good_matches,double& midist,int orb=0 )
{

    double max_dist = 0;
    double min_dist = 1000.0;

    for(unsigned int i = 0; i < v_matches.size(); i++ )
    {
        double dist = v_matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }


    if (min_dist <= 0) {
        min_dist = 10.0;
    }
    double  cutoff;
    if(orb == 0)
    {
        cutoff = 15*min_dist;
    } else
    {
        cutoff = 2.5*min_dist;
    }


    set<int> existing_trainIdx;
    for(unsigned int i = 0; i < v_matches.size(); i++ )
    {

        if(v_matches[i].distance > 0.0&&v_matches[i].distance < 15*min_dist)
        {
            //如果一个描述符在train中只出现一次，说明它没有其他的重匹配，
            if (existing_trainIdx.find(v_matches[i].trainIdx) == existing_trainIdx.end() &&
                    v_matches[i].distance < cutoff) {
                good_matches.push_back(v_matches[i]);
                existing_trainIdx.insert(v_matches[i].trainIdx);
            }

        }
    }

    sort(good_matches.begin(),good_matches.end(),cl_greater);
    midist = good_matches[(int)good_matches.size()/2].distance;
}
bool RansacGoodMatches(vector<DMatch> &good_matches,vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2,
                       vector<KeyPoint>& imgpts_good_1, vector<KeyPoint>& imgpts_good_2,v_pair& kp_good_depth_idx,
                       unordered_set<int> &kp_depth_idx )
{
    vector<v_keypoint> imgpts_tmp(2);
    vector<DMatch> v_new_matches;
    vector<v_point> pts(2),tem_best_pts(2),best_pts(2);
    vector<pair<int,int>> kp_idx__temp;
    vector<uchar> status;
    GetAlignedPointsFromMatch(keypoint_1, keypoint_2, good_matches,
                              imgpts_tmp[0], imgpts_tmp[1],kp_idx__temp);
    KeyPointsToPoints(imgpts_tmp[0], pts[0]);
    KeyPointsToPoints(imgpts_tmp[1], pts[1]);

    double minVal,maxVal;
    cv::minMaxIdx(pts[0],&minVal,&maxVal);

    Mat F = findFundamentalMat(pts[0], pts[1], FM_RANSAC, 0.006*maxVal, 0.99, status);
    cout << "筛选后匹配点" << countNonZero(status) << " / " << status.size() << endl;

    for (unsigned int j=0; j<status.size(); j++)
    {
        if (status[j])
        {
            imgpts_good_1.push_back(imgpts_tmp[0][j]);
            imgpts_good_2.push_back(imgpts_tmp[1][j]);

            kp_depth_idx.insert(kp_idx__temp[j].second);
            kp_good_depth_idx.push_back(kp_idx__temp[j]);//成双成对

            v_new_matches.push_back(good_matches[j]);

        }
    }

    good_matches = v_new_matches;

}
bool RansacGoodMatches(vector<DMatch> &good_matches,vector<KeyPoint>& orb_keypoints_1, vector<KeyPoint>& orb_keypoints_2,
                       vector<Point2f>& pts_1, vector<Point2f>& pts_2)
{
    vector<v_keypoint> keypoints_best(2);
    vector<v_point> pts(2),tem_best_pts(2),best_pts(2);
    vector<DMatch> v_new_matches;
    vector<uchar> status;
    GetAlignedPointsFromMatch(orb_keypoints_1, orb_keypoints_2, good_matches,
                              keypoints_best[0], keypoints_best[1]);

    KeyPointsToPoints(keypoints_best[0], tem_best_pts[0]);
    KeyPointsToPoints(keypoints_best[1], tem_best_pts[1]);
    double t_minVal,t_maxVal;
    cv::minMaxIdx(tem_best_pts[0],&t_minVal,&t_maxVal);
    Mat f = findFundamentalMat(tem_best_pts[0], tem_best_pts[1], FM_RANSAC, 0.006*t_maxVal, 0.99, status);
    cout << "筛选后匹配点" << countNonZero(status) << " / " << status.size() << endl;

    for (unsigned int j=0; j<status.size(); j++)
    {
        if (status[j])
        {
            best_pts[0].push_back(tem_best_pts[0][j]) ;
            best_pts[1].push_back(tem_best_pts[1][j]) ;
            v_new_matches.push_back(good_matches[j]);
        }
    }
    pts_1 = best_pts[0];
    pts_2 = best_pts[1];
    good_matches = v_new_matches;
}
void WriteFileMatches(Mat images_1,Mat images_2,vector<DMatch> draw_matches,
                      vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2,string name)
{
    Mat img_matches;

    drawMatches( images_1,keypoint_1, images_2, keypoint_2,
                 draw_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite(name,img_matches);
}
