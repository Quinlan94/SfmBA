#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "FeatureMatching.h"
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include "opencv2/imgproc/imgproc.hpp"

#include "Common.h"
//#include "KAZE_match.h"

#include <iostream>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;



void FeatureMatching(const Mat& img_1, 
				   const Mat& img_2, 
				   vector<KeyPoint>& keypts1,
				   vector<KeyPoint>& keypts2,
				   vector<KeyPoint>& keypts1_good,
				   vector<KeyPoint>& keypts2_good,
				   	vector<DMatch>* matches,
					int method)
{
	
	Mat descriptors_1, descriptors_2;
	

	if(method == 1) // SURF descriptor
	{
		double minHessian = 400;
		Ptr<SurfFeatureDetector> detector = SURF::create( minHessian);

		detector->detect( img_1,keypts1);
		detector->detect( img_2, keypts2);

		//-- Step 2: Calculate descriptors (feature vectors)
		 Ptr<SurfDescriptorExtractor> extractor = SURF::create();


		extractor->compute( img_1,keypts1, descriptors_1 );
		extractor->compute( img_2, keypts2, descriptors_2 );


		


		//-- Draw only "good" matches
		/*Mat img_matches;
		drawMatches( img_1, keypts1, img_2, keypts2,
						good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );		
			//-- Show detected matches
		imshow( "Feature Matches", img_matches );
		waitKey(0);
		destroyWindow("Feature Matches");*/

	}
	if(method == 2) // BRIEF descriptor
	{
		Ptr<FeatureDetector> detector = ORB::create(); //"BRISK"
		detector->detect(img_1,keypts1);
		detector->detect(img_2,keypts2);

		Ptr<DescriptorExtractor> extractor = ORB::create();
		//extractor->create("ORB");
		extractor->compute(img_1,keypts1, descriptors_1);
		extractor->compute(img_2,keypts2, descriptors_2);


	}
	if(method == 3) // SIFT descriptor
	{
        Ptr<SiftFeatureDetector> detector = SIFT::create();
		detector->detect( img_1,keypts1);
		detector->detect( img_2, keypts2);

		//-- Step 2: Calculate descriptors (feature vectors)
		Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();

		
		extractor->compute( img_1,keypts1, descriptors_1 );
		extractor->compute( img_2, keypts2, descriptors_2 );

	}



	BFMatcher matcher(NORM_L2,true);//不建议构造函数初始化，使用 BFMatcher.create()
	std::vector< DMatch > matches_;
	if (matches == NULL) {
		matches = &matches_;
	}
	matcher.match( descriptors_1, descriptors_2, *matches ); // Match the feature points

	double max_dist = 0; double min_dist = 1000.0;
	//-- Quick calculation of max and min distances between keypoints
	for(unsigned int i = 0; i < matches->size(); i++ )
	{ 
		double dist = (*matches)[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	std::vector< DMatch > good_matches;
	vector<KeyPoint> imgpts1_good,imgpts2_good;

	if (min_dist <= 0) {//最小距离还能小于零不成？
		min_dist = 10.0;
	}

	double cutoff = 4.0*min_dist;//4.0*min_dist;
	std::set<int> existing_trainIdx;
	for(unsigned int i = 0; i < matches->size(); i++ )
	{ 
		if ((*matches)[i].trainIdx <= 0) {//？？？？会有这种情况？
			(*matches)[i].trainIdx = (*matches)[i].imgIdx;
		}
        //如果一个描述符在train中只出现一次，说明它没有其他的重匹配，
		if( existing_trainIdx.find((*matches)[i].trainIdx) == existing_trainIdx.end() && 
			(*matches)[i].trainIdx >= 0 && (*matches)[i].trainIdx < (int)(keypts2.size()) &&
			(*matches)[i].distance > 0.0 && (*matches)[i].distance < cutoff ) 
		{
			good_matches.push_back( (*matches)[i]);
			keypts1_good.push_back(keypts1[(*matches)[i].queryIdx]);
			keypts2_good.push_back(keypts2[(*matches)[i].trainIdx]);
			existing_trainIdx.insert((*matches)[i].trainIdx);
		}
	}
    matches = &good_matches;


		
}







































