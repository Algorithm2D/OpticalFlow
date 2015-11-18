

#include <highgui.h>
#include <iostream>
#include<fstream>
#include<string>
//#include "VarFlow.h" 
#include "ProfTimer.h"
#include"iOpticalFlow.h"

using namespace std;
//例子-计算光流矢量，输入2张图片
void caculateOpticalFlow_images(Mat& c,Mat& n,int wk=0)
	{
	Mat gc,gn;
	cvtColor(c,gc,CV_RGB2GRAY);
	cvtColor(n,gn,CV_RGB2GRAY);


	Mat u,v;
	Mat colorImage=Mat::zeros (c.size (),CV_8UC3);
	Mat motionImage=Mat::zeros (c.size (),CV_8UC3);





	int width,height;
	cv::Size size=c.size ();
	width=size.width ;
	height=size.height ;

	int max_level = 4;
	int start_level = 0;

	//Two pre and post smoothing steps, should be greater than zero
	int n1 = 2;
	int n2 = 2;

	//Smoothing and regularization parameters, experiment but keep them above zero
	float rho = 2.8;
	float alpha = 1400;
	float sigma = 1.5;


	ProfTimer t;
	OpticalFlow of(cv::Size (width, height), max_level, start_level, n1, n2, rho, alpha, sigma);


	t.Start ();
	caculateOpticalFlow(of,gc,gn,u,v); 
	t.Stop();
	drawMotionField(u, v, motionImage, 10, 10, 1, 5,cv::Scalar (0,255,0));


	cout<<"time=: "<<t.GetDurationInSecs ()*1000<<"(ms)"<<endl;
	imshow("motionField",motionImage);

	cv::waitKey (wk);

	}
//例子-计算光流矢量，输入一段视频
 void caculateOpticalFlow_video(string vfile_name)
	 {
	 VideoCapture video(vfile_name);
	 if(!video.isOpened ())
		 {
		 cout<<"file :"<<vfile_name<<" is not availible!"<<endl;
		 return;
		 }
	 Mat c(480,640,CV_8UC3),n(480,640,CV_8UC3),s;
	 video>>s;
	 s.copyTo (c);
	 while(true&&(!c.empty ()))
		 {
		  video>>s;
		  s.copyTo (n);

		  if(n.empty ()) break;
		  imshow("c",c);
		  imshow("n",n);
		  caculateOpticalFlow_images(c,n,5);
		  //n=c;
		  n.copyTo (c);
		 }
	 }
int main(int argc, char *argv[])
	{


	Mat c=imread("Data/yos_img_08.jpg");
	Mat n=imread("Data/yos_img_09.jpg");

	caculateOpticalFlow_images(c,n,0);
	string fn;
	cout<<"input video file:";
	cin>>fn;
    //caculateOpticalFlow_video(fn);
   
	}
