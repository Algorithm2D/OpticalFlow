#pragma once
#include<cv.h>
#include<cxcore.h>
using namespace cv;
#include <iostream>
#include<fstream>
#include<string>
using namespace std;

#ifdef _DEBUG
#include"debug.h"
#endif
 //计算光流矢量
class OpticalFlow
	{
	public:
		/*构造函数，设置参数
		size  :  输入算法图片的尺寸，若输入图片与size不符合，将采用线性插值的方法调整到size大小.
		max_level :  网格最大层次
		start_levl : 网格初始层次
		n1 : Gauss-Seidel迭代次数
		n2 : Gauss-Seidel迭代次数
		rho : 高斯平滑算子的参数，即高斯平滑函数的标准差
		sigma :同上
		*/
		OpticalFlow(cv::Size size, int max_level, int start_level, int n1, int n2,float rho, float alpha, float sigma);
		//析构函数，释放资源
		~OpticalFlow(void);
	public:
		/*计算光流矢量
		     current : 当前帧图像数据
		    next : current紧接着下一帧图像数据
			u : 光流矢量u分量
			v : 光流矢量v分量
			saved_data : 是否保存数据，true,保存；否则，不保存。
		*/
        int CalcFlow(Mat& current, Mat& next,Mat& u, Mat& v, bool saved_data);
	private:
		 
		//将src尺寸调整为dst的尺寸，采用线性插值的算法
		inline void _resize(Mat& src,Mat& dst);
		//采用高斯加权的方法平滑图像src,结果存放至dst
		inline void _smooth(Mat& src,Mat& dst,float sigma);
		//图像类型转换，从8位图像src,转换为32位的浮点型图像
		inline void _convert(Mat& src,Mat& dst);
    private:
		/*	执行一次网格迭代
		current_level : 当前网格层数
		max_level : 网格最大层数
		first-level :网格初始层数
		h : 网格尺寸
		J13v : 结构张量第(1,3)元素
		J23v: 结构张量第(2,3)元素
		*/
        void gaussSeidelRecursive (int current_level, int max_level, int first_level, float h, vector<Mat>& J13v,vector<Mat>& J23v);
		 /*	执行Gauss-Seidel迭代
		current_level : 当前网格层数
		h : 网格尺寸
		num_iter:迭代次数
		J13v : 结构张量第(1,3)元素
		J23v: 结构张量第(2,3)元素
		*/
		void gaussSeidelIteration (int current_level, float h, int num_iter,vector<Mat>& J13v,vector<Mat>& J23v);
		/* Gauss-seidel迭代算法过程中更新未知量
		 X : 未知量
		 x :欲更新未知量的坐标，x-坐标
		 y :欲更新未知量的坐标，y-坐标
		 h : 网格尺寸
		 J11 : 坐标(x,y)处未知量的结构张量的(1,1)元素
		 J12 : 坐标(x,y)处未知量的结构张量的(1,1)元素
		 J13 : 坐标(x,y)处未知量的结构张量的(1,1)元素
		*/
		float gaussSeidelStep (Mat& X, int x, int y, float h, float J11, float J12, float J13, float vi);
		/*	执行限制算子
		 X : 未知量
		 x :欲更新未知量的坐标，x-坐标
		 y :欲更新未知量的坐标，y-坐标
		 h : 网格尺寸
		 J11 : 坐标(x,y)处未知量的结构张量的(1,1)元素
		 J12 : 坐标(x,y)处未知量的结构张量的(1,1)元素
		 vi : 未知量第i个元素第k次取值
		*/
        float residualPartStep (Mat& X, int x, int y, float h, float J11, float J12, float vi);
		 /*计算当前网格的余量
		current_level : 当前网格层数
		h : 网格尺寸
		J13v : 结构张量第(1,3)元素
		J23v: 结构张量第(2,3)元素
		*/
        void calculateResidual (int current_level, float h, vector<Mat>& J13v,vector<Mat>& J23v);

	private:
		//求图像数据I导数的卷积模板
		Mat_<float> maskx,masky;
        
		//分别存放归一化的图像
		Mat  IcSmall,InSmall;   
		//存放归一化图像的浮点型数据
		Mat  Icfloat,Infloat;

        //存放dI/dx,dI/dy,dI/dt(或dI/dz)的数据
		Mat Ifx,Ify,Ift;
		//存放d^2I/dx^2,d(dI/dx)/dy,d(dI/dx)/dt,d^2I/dy^2,d(dI/dy)/dt;
		vector<Mat>  Ifxfx,Ifxfy,Ifxft,Ifyfy,Ifyft;
		//存放u,v矢量临时数据
		vector<Mat>  Iu,Iv,Iue,Ive;     
    
		//网格最大层数
        int max_level;
		//网格初始层数
        int start_level;      
		//平滑迭代次数
        int n1;
        int n2;  

		//高斯平滑参数
        float rho;
        float alpha;
		//高斯平滑参数
        float sigma;
	};
