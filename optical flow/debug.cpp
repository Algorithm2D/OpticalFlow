#include"debug.h"

void _print_mat(Mat& m,string f_name)
	{
	ofstream f(f_name.c_str ());
	cv::Size size=m.size ();
	int type=m.type ();
	int depth=m.depth ();
	int channel=m.channels ();
	/*f<<"size:  ("<<size.width <<" , "<<size.height <<" )"<<endl;
	f<<"type: "<<type<<endl;
	f<<"depth: "<<depth<<endl;
	f<<"channel: "<<channel<<endl;*/
	int y,x;
	if(depth==0)
		{
		f<<"[ ";
		for(y=0;y<size.height ;++y)
			{
			uchar* ptr=m.ptr (y);
			for(x=0;x<size.width ;++x)
				{
				f<<unsigned int(ptr[x])<<" ";
				}
			if((y+1)==size.height ) f<<" ];"<<endl;else f<<";"<<endl;
			}
		}
	else if(depth==5)
		{
		f<<"[ ";
		for(y=0;y<size.height ;++y)
			{
			float* ptr=m.ptr<float> (y);
			for(x=0;x<size.width ;++x)
				{
				f<<float(ptr[x])<<" ";
				}
			if((y+1)==size.height ) f<<" ];"<<endl;else f<<";"<<endl;
			}
		
		}

	}
