#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

class ObjCat{
public:

	ObjCat();
	~ObjCat();
	
	void train();
	void test(std::string imgName);

private:
	bool debugging;
	cv::Mat mVocabulary;
	cv::Mat mLabels;
	int mK;//k-means' parameter
	cv::CvSVMParams mSVMParams;//SVM's parameters
	
	std::vector<cv::Mat> readImage();
	
	std::vector< std::vector< cv::KeyPoint > >
		detect(const std::vector<cv::Mat> &images);
	
	std::vector< cv::Mat >
		compute(const std::vector<cv::Mat> &images,
				std::vector<std::vector<cv::KeyPoint> > &kps);

	void cluster(const std::vector<cv::Mat>);
};
