#ifndef  _DEBUG_H_
#define _DEBUG_H_
#include<cv.h>
#include<cxcore.h>
using namespace cv;
#include <iostream>
#include<fstream>
#include<string>
using namespace std;

 
void _print_mat(Mat& m,string f_name);
inline void print_mat(IplImage* m,string f_name){
	Mat _m(m);
	_print_mat(_m,f_name);
	}
	 
inline void print_mat(Mat& m,string f_name)
	{
	_print_mat(m,f_name);
	}
#endif