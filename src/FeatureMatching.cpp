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

#include <iostream>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


bool getGoodMatches(vector<DMatch> &fliter_matches, vector<DMatch> &best_matches, bool initial,
                    const vector<DMatch> &tri_matches) {
	double max_dist = 0;
	double min_dist = 1000.0;

	for(unsigned int i = 0; i < tri_matches.size(); i++ )
	{
		double dist = tri_matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	vector< DMatch > good_matches,temp_matches;
	vector<KeyPoint> imgpts1_good,imgpts2_good;

	if (min_dist <= 0) {
		min_dist = 10.0;
	}


	double  cutoff = 2*min_dist;

	set<int> existing_trainIdx;
	for(unsigned int i = 0; i < tri_matches.size(); i++ )
	{

		if(tri_matches[i].distance > 0.0 &&tri_matches[i].distance < 10*min_dist)
		{
			//如果一个描述符在train中只出现一次，说明它没有其他的重匹配，
			if (existing_trainIdx.find(tri_matches[i].trainIdx) == existing_trainIdx.end() &&
				 tri_matches[i].distance < cutoff) {
				good_matches.push_back(tri_matches[i]);
				existing_trainIdx.insert(tri_matches[i].trainIdx);
			}
			temp_matches.push_back(tri_matches[i]);
		}
	}
	best_matches = good_matches;
	fliter_matches = temp_matches;
	sort(best_matches.begin(),best_matches.end(),cl_greater);
	sort(fliter_matches.begin(),fliter_matches.end(),cl_greater);
    cout<<"best_matches 存在： "<<best_matches.size()<<endl;
	return best_matches.size() > 4;
}

bool FeatureMatching(const Mat& img_1,
					 const Mat& img_2,
					 vector<KeyPoint>& keypts1,
					 vector<KeyPoint>& keypts2,
					 vector<KeyPoint>& orb_keypts1,
					 vector<KeyPoint>& orb_keypts2,
					 vector<DMatch>& fliter_matches,
					 vector<DMatch>& best_matches,
					 FeatureExtract method,
					 bool initial,
					 int& mindist)
{
	
	Mat descriptors_1, descriptors_2,descriptors_3, descriptors_4;
	std::vector< DMatch > tri_matches,b_matches;
	

	switch (method)
	{
		case Surf:
		{
			double minHessian = 400;
			Ptr<SurfFeatureDetector> detector = SURF::create(minHessian);
			detector->detect(img_1, keypts1);
			detector->detect(img_2, keypts2);
			Ptr<SurfDescriptorExtractor> extractor = SURF::create();
			extractor->compute(img_1, keypts1, descriptors_1);
			extractor->compute(img_2, keypts2, descriptors_2);

			Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
			matcher->match ( descriptors_1, descriptors_2, tri_matches );
			break;
		}
	    case Orb:
		{
			Ptr<FeatureDetector> detector = ORB::create();
			detector->detect(img_1,orb_keypts1);
			detector->detect(img_2,orb_keypts2);

			Ptr<DescriptorExtractor> extractor = ORB::create();
			extractor->compute(img_1,orb_keypts1, descriptors_1);
			extractor->compute(img_2,orb_keypts2, descriptors_2);
			Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
			matcher->match ( descriptors_1, descriptors_2, tri_matches );
			break;
		}
		case Sift: {
			Ptr<SiftFeatureDetector> detector = SIFT::create();
			detector->detect(img_1, keypts1);
			detector->detect(img_2, keypts2);

			Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();


			extractor->compute(img_1, keypts1, descriptors_1);
			extractor->compute(img_2, keypts2, descriptors_2);
			Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
			matcher->match ( descriptors_1, descriptors_2, tri_matches );

			break;
		}



	}


	bool status = getGoodMatches(fliter_matches, best_matches, initial, tri_matches);

	{
		double minHessian = 400;
		Ptr<SurfFeatureDetector> detector = SURF::create(minHessian);
		detector->detect(img_1, keypts1);
		detector->detect(img_2, keypts2);
		Ptr<SurfDescriptorExtractor> extractor = SURF::create();
		extractor->compute(img_1, keypts1, descriptors_3);
		extractor->compute(img_2, keypts2, descriptors_4);

		Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
		matcher->match ( descriptors_3, descriptors_4, b_matches );
		fliter_matches.clear();
		fliter_matches = b_matches;
		sort(fliter_matches.begin(),fliter_matches.end(),cl_greater);
		mindist = fliter_matches[(int)fliter_matches.size()/2-1].distance;
	}

	//-- Draw only "good" matches
    if(status) {
        Mat img_matches;
        drawMatches(img_1, keypts1, img_2, keypts2,
                    fliter_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imwrite("../fliter_matches.jpg", img_matches);

        Mat img_matches_1;
        drawMatches(img_1, orb_keypts1, img_2, orb_keypts2,
                    best_matches, img_matches_1, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imwrite("../best_matches.jpg", img_matches_1);
    } else
        cout<< "===================最佳点太少!!!!======================="<<endl;

		
}







































