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
 //�������ʸ��
class OpticalFlow
	{
	public:
		/*���캯�������ò���
		size  :  �����㷨ͼƬ�ĳߴ磬������ͼƬ��size�����ϣ����������Բ�ֵ�ķ���������size��С.
		max_level :  ���������
		start_levl : �����ʼ���
		n1 : Gauss-Seidel��������
		n2 : Gauss-Seidel��������
		rho : ��˹ƽ�����ӵĲ���������˹ƽ�������ı�׼��
		sigma :ͬ��
		*/
		OpticalFlow(cv::Size size, int max_level, int start_level, int n1, int n2,float rho, float alpha, float sigma);
		//�����������ͷ���Դ
		~OpticalFlow(void);
	public:
		/*�������ʸ��
		     current : ��ǰ֡ͼ������
		    next : current��������һ֡ͼ������
			u : ����ʸ��u����
			v : ����ʸ��v����
			saved_data : �Ƿ񱣴����ݣ�true,���棻���򣬲����档
		*/
        int CalcFlow(Mat& current, Mat& next,Mat& u, Mat& v, bool saved_data);
	private:
		 
		//��src�ߴ����Ϊdst�ĳߴ磬�������Բ�ֵ���㷨
		inline void _resize(Mat& src,Mat& dst);
		//���ø�˹��Ȩ�ķ���ƽ��ͼ��src,��������dst
		inline void _smooth(Mat& src,Mat& dst,float sigma);
		//ͼ������ת������8λͼ��src,ת��Ϊ32λ�ĸ�����ͼ��
		inline void _convert(Mat& src,Mat& dst);
    private:
		/*	ִ��һ���������
		current_level : ��ǰ�������
		max_level : ����������
		first-level :�����ʼ����
		h : ����ߴ�
		J13v : �ṹ������(1,3)Ԫ��
		J23v: �ṹ������(2,3)Ԫ��
		*/
        void gaussSeidelRecursive (int current_level, int max_level, int first_level, float h, vector<Mat>& J13v,vector<Mat>& J23v);
		 /*	ִ��Gauss-Seidel����
		current_level : ��ǰ�������
		h : ����ߴ�
		num_iter:��������
		J13v : �ṹ������(1,3)Ԫ��
		J23v: �ṹ������(2,3)Ԫ��
		*/
		void gaussSeidelIteration (int current_level, float h, int num_iter,vector<Mat>& J13v,vector<Mat>& J23v);
		/* Gauss-seidel�����㷨�����и���δ֪��
		 X : δ֪��
		 x :������δ֪�������꣬x-����
		 y :������δ֪�������꣬y-����
		 h : ����ߴ�
		 J11 : ����(x,y)��δ֪���Ľṹ������(1,1)Ԫ��
		 J12 : ����(x,y)��δ֪���Ľṹ������(1,1)Ԫ��
		 J13 : ����(x,y)��δ֪���Ľṹ������(1,1)Ԫ��
		*/
		float gaussSeidelStep (Mat& X, int x, int y, float h, float J11, float J12, float J13, float vi);
		/*	ִ����������
		 X : δ֪��
		 x :������δ֪�������꣬x-����
		 y :������δ֪�������꣬y-����
		 h : ����ߴ�
		 J11 : ����(x,y)��δ֪���Ľṹ������(1,1)Ԫ��
		 J12 : ����(x,y)��δ֪���Ľṹ������(1,1)Ԫ��
		 vi : δ֪����i��Ԫ�ص�k��ȡֵ
		*/
        float residualPartStep (Mat& X, int x, int y, float h, float J11, float J12, float vi);
		 /*���㵱ǰ���������
		current_level : ��ǰ�������
		h : ����ߴ�
		J13v : �ṹ������(1,3)Ԫ��
		J23v: �ṹ������(2,3)Ԫ��
		*/
        void calculateResidual (int current_level, float h, vector<Mat>& J13v,vector<Mat>& J23v);

	private:
		//��ͼ������I�����ľ��ģ��
		Mat_<float> maskx,masky;
        
		//�ֱ��Ź�һ����ͼ��
		Mat  IcSmall,InSmall;   
		//��Ź�һ��ͼ��ĸ���������
		Mat  Icfloat,Infloat;

        //���dI/dx,dI/dy,dI/dt(��dI/dz)������
		Mat Ifx,Ify,Ift;
		//���d^2I/dx^2,d(dI/dx)/dy,d(dI/dx)/dt,d^2I/dy^2,d(dI/dy)/dt;
		vector<Mat>  Ifxfx,Ifxfy,Ifxft,Ifyfy,Ifyft;
		//���u,vʸ����ʱ����
		vector<Mat>  Iu,Iv,Iue,Ive;     
    
		//����������
        int max_level;
		//�����ʼ����
        int start_level;      
		//ƽ����������
        int n1;
        int n2;  

		//��˹ƽ������
        float rho;
        float alpha;
		//��˹ƽ������
        float sigma;
	};
