
#include "ObjectCategory.h"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;
using namespace cv;

ObjCat::ObjCat()
{
	debugging = false;
	mK = 30;
	mSVMParams.svm_type = CvSVM::C_SVC;
	mSVMParams.kernel_type = CvSVM::LINEAR;
	mSVMParams.term_crit = cvTermCriteria(CV_TERMCRIT_EPS, 100, 1e-6);
}

ObjCat::~ObjCat()
{}


/*
 * read images and train the clustered vocabulary
 *
 */
void ObjCat::train()
{
	//read images and compute key point descriptors
	vector<Mat> images = readImage();
	vector< vector<KeyPoint> > keyPoints = detect(images);
	vector<Mat> descriptors = compute(images, keyPoints);

	if (debugging)
	{
		int total = 0;
		for (int i = 0; i < descriptors.size(); ++i)
		{
			cout<<i<<": "<<descriptors[i].rows<<endl;
			total += descriptors[i].rows;
			cout<<total<<endl;
		}
	}

	//cluster the vocabulary
	cluster(descriptors);
	
	//train the classifier
	trainClassifier(images, descriptors);
}

void ObjCat::test(string imgName)
{
	Mat img = imread(imgName, CV_LOAD_IMAGE_COLOR);
	vector<Mat> imgVec;
	imgVec.push_back(img);
	vector<vector<KeyPoint> > keyps = detect(imgVec);
	vector<Mat> desps = compute(imgVec, keyps);

	Mat imageVector = getImageVector(img, desps[0]);
	float label = mSVM.predict(imageVector, false);
	cout<<"predict result: "<<label<<endl;
}

/*
 *use descriptor matcher to get image vector
 */
Mat ObjCat::getImageVector(const Mat &image,
						   const Mat &descriptor){
	int clusterCount = mVocabulary.rows;

	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match( descriptor, mVocabulary, matches );

	Mat imageVectorMat = Mat(1, clusterCount, CV_32FC1, Scalar::all(0.0));
	float *imageVector = (float*)imageVectorMat.data;
	for (int i = 0; i < matches.size(); ++i)
	{
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx;

		CV_Assert( queryIdx == i);

		imageVector[trainIdx] = imageVector[trainIdx] + 1.f;
	}
	
	//Normalize the vector
	imageVectorMat /= descriptor.rows;
	return imageVectorMat;
}

/*
 * use image labels to train the SVM classifier
 *
 */
void ObjCat::trainClassifier(const vector<Mat> &images,
							 const vector<Mat> &descriptors)
{
	float *labels = new float[images.size()];
	Mat imageVectors;
	for (int i = 0; i < images.size(); ++i)
	{
		//10 images of each category and are sorted in order
		labels[i] = 1.0* (i / 10);

		Mat imageVector = getImageVector(images[i], descriptors[i]);
		if (i == 0)
		{
			imageVectors = imageVector;
		} else {
			vconcat(imageVectors, imageVector, imageVectors);
		}
	}

	Mat labelsMat = Mat(images.size(), 1, CV_32FC1, labels);

	if (debugging)
	{
		cout<<labelsMat<<endl;

		cout<<imageVectors<<endl;
	}
	
	mSVM.train(imageVectors, labelsMat, Mat(), Mat(), mSVMParams);
	delete labels;

	cout<<"original labels: "<<labelsMat<<endl;
	cout<<"test on training samples: ";
	for (int i = 0; i < images.size(); ++i)
	{
		cout<<mSVM.predict(getImageVector(images[i], descriptors[i]))<<", ";
	}
	cout<<endl;
}

/*
 * use kmeans algorithm to cluster key point descriptors
 * first merge discriptors to one Mat
 * then clustering the centers as vocabulary
 */
void ObjCat::cluster(const vector<Mat> &descriptors)
{
	if (descriptors.empty())
	{
		return;
	}

	int descCount = 0;
    for( size_t i = 0; i < descriptors.size(); i++ )
        descCount += descriptors[i].rows;

    Mat mergedDescriptors( descCount,
						   descriptors[0].cols,
						   descriptors[0].type() );
    for( size_t i = 0, start = 0; i < descriptors.size(); i++ )
    {
        Mat submut = mergedDescriptors.rowRange((int)start,
												(int)(start +
													  descriptors[i].rows));
        descriptors[i].copyTo(submut);
        start += descriptors[i].rows;
    }

	Mat labels;//output labels
	Mat centers;//output centers
	// TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
	// 					  1000, 1e-7);//compute criteria to stop
	TermCriteria criteria(TermCriteria::EPS,
						  100000, 1e-7);//compute criteria to stop
	int K = mK;// K clusters
	int attempts = 15;//try attempts times to choose the best
	kmeans(mergedDescriptors, K, labels, criteria, attempts,
		   KMEANS_RANDOM_CENTERS, centers);
	mVocabulary = centers;
}


vector< vector< KeyPoint > > ObjCat::detect(const vector<Mat> &images)
{
	vector< vector< KeyPoint > > keyPoints;

	int minHessian = 400;
	SiftFeatureDetector detector(minHessian);
	for (int i = 0; i < images.size(); ++i)
	{	
		vector<KeyPoint> keyPoint;
		detector.detect( images[i], keyPoint );
		keyPoints.push_back(keyPoint);
	}

	return keyPoints;
}

vector< Mat >  ObjCat::compute(const vector<Mat> &images,
							   vector<vector<KeyPoint> > &kps)
{
	vector <Mat> descriptors;
	SiftDescriptorExtractor extractor;
	for (int i = 0; i < images.size(); ++i)
	{
		Mat desp;
		extractor.compute( images[i], kps[i], desp );
		descriptors.push_back(desp);
	}
	return descriptors;
}

/*
 * read images in default folder[../resource/train]
 * and put images into vector
 *
 */
vector<Mat> ObjCat::readImage()
{
	vector<Mat> images;

	string defaultDirPath = "../resource/train";

	path dir(defaultDirPath);
	if (exists(dir))
	{
		vector<path> vec;
		copy(directory_iterator(dir),
			 directory_iterator(), back_inserter(vec));
		for (vector<path>::const_iterator it (vec.begin());
			 it != vec.end(); ++it)
		{
			if (extension(*it) != ".jpg")
			{
				continue;
			}
			cout << "reading image: " << *it << endl;
			Mat img = imread( it->c_str(),
							  CV_LOAD_IMAGE_COLOR);
			if (debugging)
			{
				// imshow("training image", img);
			}
			images.push_back(img);
			
		}
	} else {
		cout<<dir<<" is not exist"<<endl;
	}
		
	return images;
}
