
#include "ObjectCategory.h"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;
using namespace cv;

ObjCat::ObjCat()
{
	debugging = true;
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

	//train the vocabulary
	cluster(descriptors);
	
	if (debugging)
	{
		int cols = mLabels.cols, rows = mLabels.rows;
		for(int i = 0; i < rows; i++)
		{
			const int* Mi = mLabels.ptr<int>(i);
			for(int j = 0; j < cols; j++)
				cout<<Mi[j]<<endl;
		}
	}

	//show the label distributation for each image
	int start = 0;
	for (int i = 0; i < images.size(); ++i)
	{
		int l0 = 0, l1= 0, l2 = 0, l3 = 0, l4 = 0;
		for (int j = 0; j < descriptors[i].rows; ++j)
		{
			switch (mLabels.at<int>(0, start+j))
			{
			case 0:
				l0 ++;
				break;
			case 1:
				l1 ++;
				break;
			case 2:
				l2 ++;
				break;
			case 3:
				l3 ++;
				break;
			case 4:
				l4 ++;
				break;
			default:
				cout<<"error label: "<<start+j<<endl;
				break;
			}
		}
		start += descriptors[i].rows;
		
		cout<<"image "<<i<<": "
			<<l0<<", "<<l1<<", "<<l2
			<<", "<<l3<<", "<<l4
			<<endl;
	}
}

void ObjCat::test(string imgName)
{

}

/*
 * use kmeans algorithm to cluster key point descriptors
 * first merge discriptors to one Mat
 * then clustering the centers as vocabulary
 */
void ObjCat::cluster(const vector<Mat> descriptors)
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
	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
						  1000, 1e-7);//compute criteria to stop
	int K = 5;// K clusters
	int attempts = 15;//try attempts times to choose the best
	kmeans(mergedDescriptors, K, labels, criteria, attempts,
		   KMEANS_RANDOM_CENTERS, centers);

	mLabels = labels;
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
				imshow("training image", img);
			}
			images.push_back(img);
			
		}
	} else {
		cout<<dir<<" is not exist"<<endl;
	}
		
	return images;
}
