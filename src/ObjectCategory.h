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

	cv::SVM mSVM;//the svm trainer and predictor
	
	cv::Mat mVocabulary;

	int mK;//k-means' parameter
	cv::SVMParams mSVMParams;//SVM's parameters
	
	std::vector<cv::Mat> readImage();
	
	std::vector< std::vector< cv::KeyPoint > >
		detect(const std::vector<cv::Mat> &images);
	
	std::vector< cv::Mat >
		compute(const std::vector<cv::Mat> &images,
				std::vector<std::vector<cv::KeyPoint> > &kps);

	void cluster(const std::vector<cv::Mat> &);

	void trainClassifier(const std::vector<cv::Mat> &images,
						 const std::vector<cv::Mat> &descriptors);
	cv::Mat getImageVector(const cv::Mat &image,
						   const cv::Mat &descriptor);
};
