#ifndef _I_OPTICAL_FLOW_H_
#define _I_OPTICAL_FLOW_H_

 
 
#include"OpticalFlow.h"
/*
	File Description: 本文件为光流算法对外接口，其他模块只能调用本								                                                   
	接口中提供的函数计算光流矢量。														                                                  
	其他计算细节本封装在OpticalFlow.h和OpticalFlow.cpp中，						 
	其中，OpticalFlow.h描述算法中需要使用变量的声明，函数的声明；	  
	OpticalFlow.cpp实现具体函数。																    
 																																			       
 																																			       
	Auther: PJ																																  
	Timer:  2010/12/03																													   
	Reference(s):																															  
		B. Andres,W. Joachim,F. Christian,K. Timo,S. Christoph ,															   
		Real-Time Optic Flow Computation with Variational Methods,													  
		Computer Science, Volume 2756/2003, pp 222-229.																	  
*/
/*	Function:       caculateOpticalFlow
	Description:   计算光流矢量 
	Calls:           
	Input:          
	变量名称		类型		说明
	pf		OpticalFlow&		计算光流矢量的对象
	current		Mat&		当前帧图像数据
	next		Mat&		下一帧图像数据
	Output:   
	变量名称		类型		说明
	u		Mat&，32位浮点矩阵		矢量(u,v)的u分量
	v		Mat&，32位浮点矩阵		矢量(u,v)的v分量
	Return:      无
	Others:      该函数将调用OpticalFlow类对象提供的算法计算矢量(u,v),
                    算法设计的参数由该对象负责；

*/
inline void caculateOpticalFlow(OpticalFlow& pf,Mat& current,Mat& next,Mat& u,Mat& v)
{
	//检查输出变量是否有效，无效则重建
	if(u.empty ()||v.empty ()||current.size ()!=u.size ()||current.size ()!=v.size ()){
		u.create (next.size (),CV_32FC1);
		v.create (next.size (),CV_32FC1);
		}
	//调用光流算法
	pf.CalcFlow (current,next,u,v,false);
	}

	
//	Function:       drawMotionField
//	Description:   绘制光流矢量，用带箭头的有限线段来表示。 
//	Calls:           
//	Input:          
//	变量名称		类型		说明
//	pf		OpticalFlow&		计算光流矢量的对象
//	u		Mat&		光流矢量,u分量
//	v		Mat&		光流矢量,v分量
//	xSpace		int			起始点x坐标
//	ySpace        in		起始点y坐标
//	cutoff		float		阈值
//	multiplier		int		矢量模长系数
//	color		Scalar		颜色向量
//	Output:   
//	变量名称		类型		说明
//	motion		Mat&，                 绘制有光流矢量的图片
//	Return:      无
//	Others:      无

void drawMotionField(Mat& u,Mat& v,Mat& motion, int xSpace, int ySpace, float cutoff, int multiplier, cv::Scalar  color);
 class OpticalFlow;
#endif
