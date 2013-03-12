#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

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

	std::vector<cv::Mat> readImage();
	
	std::vector< std::vector< cv::KeyPoint > >
		detect(const std::vector<cv::Mat> &images);
	
	std::vector< cv::Mat >
		compute(const std::vector<cv::Mat> &images,
				std::vector<std::vector<cv::KeyPoint> > &kps);

	void cluster(const std::vector<cv::Mat>);
};
