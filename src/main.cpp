#include <iostream>
#include "ObjectCategory.h"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;

int main(int argc, char *argv[])
{
	ObjCat objCat;

	cout<<"start training"<<endl;
	objCat.train();

	cout<<"training finished."<<endl;
	cout<<"input the test file name[default folder: ../resource/test/]"
		<<endl<<"(type 'exit' to exit)"<<endl;
	string testFilePath;
	cin>>testFilePath;
	while (testFilePath != "exit")
	{

		string defaultPath = "../resource/test/"+testFilePath;
		cout<<"using file: "<<defaultPath<<endl;
		path filePath(defaultPath);
		if (!exists(filePath))
		{
			cout<<defaultPath<<" not found, using: "<<testFilePath<<endl;
			path absPath(testFilePath);
			if (!exists(absPath))
			{
				cout<<testFilePath<<" not found, failed to test this image."
					<<endl;
			} else {

				objCat.test(testFilePath);
			}
		} else {

			objCat.test(defaultPath);
		}
		
		cin>>testFilePath;
	}
	return 0;
}
