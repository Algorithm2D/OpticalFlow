#ifndef _I_OPTICAL_FLOW_H_
#define _I_OPTICAL_FLOW_H_

 
 
#include"OpticalFlow.h"
/*
	File Description: ���ļ�Ϊ�����㷨����ӿڣ�����ģ��ֻ�ܵ��ñ�								                                                   
	�ӿ����ṩ�ĺ����������ʸ����														                                                  
	��������ϸ�ڱ���װ��OpticalFlow.h��OpticalFlow.cpp�У�						 
	���У�OpticalFlow.h�����㷨����Ҫʹ�ñ�����������������������	  
	OpticalFlow.cppʵ�־��庯����																    
 																																			       
 																																			       
	Auther: PJ																																  
	Timer:  2010/12/03																													   
	Reference(s):																															  
		B. Andres,W. Joachim,F. Christian,K. Timo,S. Christoph ,															   
		Real-Time Optic Flow Computation with Variational Methods,													  
		Computer Science, Volume 2756/2003, pp 222-229.																	  
*/
/*	Function:       caculateOpticalFlow
	Description:   �������ʸ�� 
	Calls:           
	Input:          
	��������		����		˵��
	pf		OpticalFlow&		�������ʸ���Ķ���
	current		Mat&		��ǰ֡ͼ������
	next		Mat&		��һ֡ͼ������
	Output:   
	��������		����		˵��
	u		Mat&��32λ�������		ʸ��(u,v)��u����
	v		Mat&��32λ�������		ʸ��(u,v)��v����
	Return:      ��
	Others:      �ú���������OpticalFlow������ṩ���㷨����ʸ��(u,v),
                    �㷨��ƵĲ����ɸö�����

*/
inline void caculateOpticalFlow(OpticalFlow& pf,Mat& current,Mat& next,Mat& u,Mat& v)
{
	//�����������Ƿ���Ч����Ч���ؽ�
	if(u.empty ()||v.empty ()||current.size ()!=u.size ()||current.size ()!=v.size ()){
		u.create (next.size (),CV_32FC1);
		v.create (next.size (),CV_32FC1);
		}
	//���ù����㷨
	pf.CalcFlow (current,next,u,v,false);
	}

	
//	Function:       drawMotionField
//	Description:   ���ƹ���ʸ�����ô���ͷ�������߶�����ʾ�� 
//	Calls:           
//	Input:          
//	��������		����		˵��
//	pf		OpticalFlow&		�������ʸ���Ķ���
//	u		Mat&		����ʸ��,u����
//	v		Mat&		����ʸ��,v����
//	xSpace		int			��ʼ��x����
//	ySpace        in		��ʼ��y����
//	cutoff		float		��ֵ
//	multiplier		int		ʸ��ģ��ϵ��
//	color		Scalar		��ɫ����
//	Output:   
//	��������		����		˵��
//	motion		Mat&��                 �����й���ʸ����ͼƬ
//	Return:      ��
//	Others:      ��

void drawMotionField(Mat& u,Mat& v,Mat& motion, int xSpace, int ySpace, float cutoff, int multiplier, cv::Scalar  color);
 class OpticalFlow;
#endif
