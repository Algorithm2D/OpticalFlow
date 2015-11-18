#include "OpticalFlow.h"
 
#include<iostream>
using namespace std;

 
 /*
    Uses the functions in the OpneCV with the special parameters.
 */
void inline OpticalFlow::_resize (Mat& src,Mat& dst){
	cv::resize (src, dst, dst.size(), (double)dst.cols/src.cols, (double)dst.rows/src.rows,  INTER_LINEAR);
	}
void inline OpticalFlow::_smooth (Mat& src,Mat& dst,float sigma){
	   cv::GaussianBlur( src, dst, cv::Size(0, 0), sigma, 0, cv::BORDER_REPLICATE );
	}
void inline OpticalFlow::_convert (Mat& src,Mat& dst){
    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );
    src.convertTo(dst, dst.type(), 1, 0);
	}
 /*
   Initializes all variables that don't need to get updated for each flow calculation. 
*/
OpticalFlow::OpticalFlow(cv::Size size, int max_level_ , int start_level_ , int n1_ , int n2_ ,
                float rho_ , float alpha_ , float sigma_ ){
					
	max_level = max_level_ ;
    start_level = start_level_ ;
    
   
	int width = (int)floor(size.width /powf(2.0,(float)(start_level)));
    int height = (int)floor(size.height /powf(2.0,(float)(start_level)));
    
    // start_level too large, correct it
    if(width < 1 || height < 1)
    {
        if(width < 1)
        {
              start_level = (int)floor(logf(size.width )/logf(2));
              width = (int)floor(size.width /powf(2.0,(float)(start_level)));
              height = (int)floor(size.height /powf(2.0,(float)(start_level)));
        }
        
        if(height < 1)
        {
              start_level = (int)floor(logf(size.height )/logf(2));
              width = (int)floor(size.width /powf(2.0,(float)(start_level)));
              height = (int)floor(size.height /powf(2.0,(float)(start_level)));
        }
    
        // Correct max_level as well
        max_level = start_level;
        cout<<"Warning: start_level too large, correcting start_level and max_level (new value = "<<start_level<<")"<<endl;
        
    }
    
    int width_end = (int)floor(size.width /powf(2.0,(float)(max_level)));
    int height_end = (int)floor(size.height /powf(2.0,(float)(max_level)));
    
    // max_level too large, correct it
    if(width_end < 1 || height_end < 1)
    {
        if(width_end < 1)
        {
              max_level = (int)floor(logf(size.width )/logf(2));
              height_end = (int)floor(size.height /powf(2.0,(float)(max_level)));
        }
        
        if(height_end < 1)
        {
              max_level = (int)floor(logf(size.height )/logf(2));
        }
        
        cout<<"Warning: max_level too large, correcting (new value = "<<max_level<<")"<<endl;
        
    }
          
             
    n1 = n1_ ;
    n2 = n2_ ;
    
    rho = rho_ ;
    alpha = alpha_ ;
    sigma = sigma_ ;
    
    // Spacial derivative masks
   
    
	maskx=(Mat_<float>(1,5)<<0.08333,-0.66666,0,0.66666,-0.08333);
	masky=(Mat_<float>(5,1)<<-0.08333,0.66666,0,-0.66666,0.08333);
  
	IcSmall.create (cv::Size (width,height),CV_8UC1);
	InSmall.create (cv::Size (width,height),CV_8UC1);
  
   
	Icfloat.create (cv::Size (width,height),CV_32FC1);
	Infloat.create (cv::Size (width,height),CV_32FC1);
  

	Ifx.create (cv::Size (width,height),CV_32FC1);
	Ify.create (cv::Size (width,height),CV_32FC1);
	Ift.create (cv::Size (width,height),CV_32FC1);
 
	Ifxfx.insert (Ifxfx.begin (),max_level-start_level+1,Mat());
	Ifxfy.insert (Ifxfy.begin (),max_level-start_level+1,Mat());
	Ifxft.insert (Ifxft.begin (),max_level-start_level+1,Mat());
	Ifyfy.insert (Ifyfy.begin (),max_level-start_level+1,Mat());
	Ifyft.insert (Ifyft.begin (),max_level-start_level+1,Mat());
 
	Iu.insert(Iu.begin (),max_level-start_level+1,Mat()); 
	Iv.insert(Iv.begin (),max_level-start_level+1,Mat()); 
	Iue.insert(Iue.begin (),max_level-start_level+1,Mat()); 
	Ive.insert(Ive.begin (),max_level-start_level+1,Mat()); 

 

    int i;
    
    //Allocate memory for image arrays
	int isize=max_level-start_level+1;
	int _width,_height=0;
    for(i = 0; i < isize; i++){
		_width=(int)floor(float(width)/powf(2.0,float(i)));
		_height=(int)floor(float(height)/powf(2.0,float(i)));

		Ifxfx[i].create (cv::Size (_width,_height),CV_32FC1);
		Ifxfy[i].create (cv::Size (_width,_height),CV_32FC1);
		Ifxft[i].create (cv::Size (_width,_height),CV_32FC1);
		Ifyfy[i].create (cv::Size (_width,_height),CV_32FC1);
		Ifyft[i].create (cv::Size (_width,_height),CV_32FC1);

		Iu[i]  =Mat::zeros  (cv::Size (_width,_height),CV_32FC1);
		Iv[i]  =Mat::zeros  (cv::Size (_width,_height),CV_32FC1);
		Iue[i]=Mat::zeros  (cv::Size (_width,_height),CV_32FC1);
		Ive[i]=Mat::zeros  (cv::Size (_width,_height),CV_32FC1);  
    }    
}

 /*
   Releases all memory uesed.
 */
OpticalFlow::~OpticalFlow(){
  
}

/*
   Implements the Gauss-Seidel method to upate the value of a flow field at a single pixel, equations 6 and 7 in Bruhn et al.
*/
float OpticalFlow::gaussSeidelStep(Mat& u, int x, int y, float h, float J11, float J12, float J13, float vi){
    
    int start_y, end_y, start_x, end_x;
    int N_num = 0;
                
    start_y = y - 1;
    end_y = y + 1;
    start_x = x - 1;
    end_x = x+1;         
                  
    float temp_u = 0;
            
    // Sum top neighbor    
    if(start_y > -1){              
        
        temp_u += *((float*)(u.data +start_y*u.step )/*imageData + start_y*u->widthStep) */+ x);
    
        N_num++;
     
    }
    
    // Sum bottom neighbor            
    if(end_y < u.size ().height ){   

        temp_u += *((float*)(/*u->imageData + end_y*u->widthStep*/u.data +end_y*u.step ) + x);
    
        N_num++;
              
    }
 
    // Sum left neighbor
    if(start_x > -1){              
                    
        temp_u += *((float*)(/*u->imageData + y*u->widthStep*/u.data +y*u.step ) + start_x);
    
        N_num++;
        
    }
    
    // Sum right neighbor
    if(end_x < u.size ().width ){              
                    
        temp_u += *((float*)(/*u->imageData + y*u->widthStep*/u.data +y*u.step ) + end_x);
    
        N_num++;
        
    }
    
    temp_u = temp_u - (h*h/alpha)*(J12*vi + J13);
    temp_u = temp_u / (N_num + (h*h/alpha)*J11);
                
                
    return temp_u;
    
}

/*
   Uses the Gauss-Seidel method to calculate the horizontal and vertical flow fields at a certain level in the multigrid
   process.
*/
void OpticalFlow::gaussSeidelIteration(int current_level, float h, int num_iter, vector<Mat>&  J13, vector<Mat>&J23){
                                
       
	Mat& u=Iu[current_level];
	Mat& v=Iv[current_level];
	Mat& fxfx=Ifxfx[current_level];
	Mat& fxfy=Ifxfy[current_level];
	Mat& fxft=J13[current_level];
	Mat& fyfy=Ifyfy[current_level];
	Mat& fyft=J23[current_level];

 
	int i, k;
	int x;
	int y;
	float *u_ptr, *v_ptr, *fxfx_ptr, *fxfy_ptr, *fxft_ptr, *fyfy_ptr, *fyft_ptr;
	
	int max_i =u.size ().area ();
	
	u_ptr = (float*)(u.data );
	v_ptr = (float*)(v.data );
	        
	fxfx_ptr = (float*)(fxfx.data );
	fxfy_ptr = (float*)(fxfy.data );
	fxft_ptr = (float*)(fxft.data  );
	fyfy_ptr = (float*)(fyfy.data );
	fyft_ptr = (float*)(fyft.data  );
	 
	
	for(k = 0; k < num_iter; k++){
        
        x = 0;
        y = 0;
        
        for(i = 0; i < max_i; i++){
                  
               // Update flow fields
                u_ptr[i] = gaussSeidelStep(u, x, y, h, fxfx_ptr[i], fxfy_ptr[i], fxft_ptr[i], v_ptr[i]);              
                v_ptr[i] = gaussSeidelStep(v, x, y, h, fyfy_ptr[i], fxfy_ptr[i], fyft_ptr[i], u_ptr[i]);
               
                x++;
                if(x == u.cols ){
                  x = 0;
                  y++;
                }
        
              
        }  // End for loop, image traversal
    
    }  // End for loop, number of iterations
    
}


/*
   Calculates part of the value of a residual field at a single pixel.
*/

float OpticalFlow::residualPartStep(Mat& u, int x, int y, float h, float J11, float J12, float vi){
    
    int start_y, end_y, start_x, end_x;
        
    start_y = y - 1;
    end_y = y + 1;
    start_x = x - 1;
    end_x = x+1;
                
    float ih2 = 1 / (h*h);
                  
    float temp_res = 0;
    int N_num = 0;
    float curr_u = *((float*)(/*u->imageData + y*u->widthStep*/u.data +y*u.step ) + x);
                
    // Sum top neighbor
      
    if(start_y > -1){              
        
        temp_res += *((float*)(/*u->imageData + start_y*u->widthStep*/u.data +start_y*u.step ) + x);
        N_num++;
     
    }
                
    if(end_y < u.size().height ){   // Sum bottom neighbor
                
        temp_res += *((float*)(/*u->imageData + end_y*u->widthStep*/u.data +end_y*u.step ) + x);
        N_num++;
              
    }
    
    // Sum left neighbor
      
    if(start_x > -1){              
        
        temp_res += *((float*)(/*u->imageData + y*u->widthStep*/u.data +y*u.step ) + start_x);
        N_num++;
        
    }
    
    // Sum right neighbor
    
    if(end_x < u.size ().width ){              
        
        temp_res += *((float*)(/*u->imageData + y*u->widthStep*/u.data +y*u.step ) + end_x);
        N_num++;
       
    }
    
    temp_res = N_num*curr_u - temp_res;
    
    temp_res *= ih2;
    
    temp_res -= (1/alpha)*(J11*curr_u + J12*vi);
    
    return temp_res;
}

/*
   Calculates the full residual of the current flow field based on equation 10 in Bruhn et al.
*/

void OpticalFlow::calculateResidual(int current_level, float h,vector<Mat>& J13,vector<Mat>& J23){

	Mat& u=Iu[current_level];
	Mat& v=Iv[current_level];
	Mat& fxfx=Ifxfx[current_level];
	Mat& fxfy=Ifxfy[current_level];
	Mat& fxft=J13[current_level];
	Mat& fyfy=Ifyfy[current_level];
	Mat& fyft=J23[current_level];
	Mat& ue=Iue[current_level];
	Mat& ve=Ive[current_level];

	int fxftd_=fxft.type  ();
	int ued_=ue.type  ();

    int i;
    float *u_ptr, *v_ptr, *fxfx_ptr, *fxfy_ptr, *fyfy_ptr, *u_res_err_ptr, *v_res_err_ptr;
    
    int max_i = u.size ().area ();
    int x, y;
    
    u_res_err_ptr = (float*)(ue.data );
    v_res_err_ptr = (float*)(ue.data );
        
    u_ptr = (float*)(u.data );
    v_ptr = (float*)(v.data );
            
    fxfx_ptr = (float*)(fxfx.data );
    fxfy_ptr = (float*)(fxfy.data );
    fyfy_ptr = (float*)(fyfy.data );
    
    x = 0;
    y = 0;
    
    for(i = 0; i < max_i; i++){
            
            // Get A^h * x_tilde^h (equation 10)
            u_res_err_ptr[i] = residualPartStep(u, x, y, h, fxfx_ptr[i], fxfy_ptr[i], v_ptr[i] );
            v_res_err_ptr[i] = residualPartStep(v, x, y, h, fyfy_ptr[i], fxfy_ptr[i], u_ptr[i] );
            
            x++;
            if(x == u.cols ){
                  x = 0;
                  y++;
            }
        
    }
    
    // Get full residual
	//int fxftd=fxft.type  ();
	//int ued=ue.type  ();
	 cv::addWeighted (fxft,(1/alpha), ue, -1, 0, ue);
	cv::addWeighted (fyft, (1/alpha),ve, -1, 0, ve);
   
}


/*
   This recursive function implements two V cycles of the Gauss-Seidel algorithm to calculate the flow field at a given level.
*/

void OpticalFlow::gaussSeidelRecursive(int current_level, int max_level, int first_level, float h, 
										vector<Mat>& J13, vector<Mat>& J23){
                                
    if(current_level == max_level){
         
		//求得初始解
        gaussSeidelIteration(current_level, h,  n1, J13, J23);
                     
    }
    
    else{
        
        //---------------------------- Start 1st V cycle -------------------------------------
     
		//Gauss-Seidel迭代，平滑迭代
        gaussSeidelIteration(current_level, h, n1, J13, J23); 
          
		//计算余量
        calculateResidual(current_level, h, J13, J23);
                               
   
		//余量限制
		_resize (Iue[current_level],Iue[current_level+1]);
		_resize (Ive[current_level],Ive[current_level+1]);
       
		//初始为0
		Iue[current_level+1].zeros (Iue[current_level+1].size (),CV_32FC1);
		Ive[current_level+1].zeros (Ive[current_level+1].size (),CV_32FC1);
       
        //在粗网格上求解，结果存放到Iue,Ive
		gaussSeidelRecursive(current_level+1, max_level, first_level, 2*h, Iue, Ive);
                       
		//更新本次误差
		_resize(Iue[current_level+1], Iue[current_level] );
		_resize(Ive[current_level+1], Ive[current_level] );
       
		//更新解
		cv::add(Iu[current_level],Iue[current_level],Iu[current_level]);
		cv::add(Iv[current_level],Ive[current_level],Iv[current_level]);
  
		//光滑迭代
        gaussSeidelIteration(current_level, h, n1+n2, J13, J23); 
                                     
       //---------------------------- End 1st V cycle, Start 2nd V cycle -------------------------------------                        
                          
       //计算余量
        calculateResidual(current_level,h, J13, J23);
                               
        // 执行限制操作算子
		_resize (Iue[current_level],Iue[current_level+1] );
		_resize (Ive[current_level],Ive[current_level+1] );

		//初始为0
		Iue[current_level+1].zeros (Iue[current_level+1].size (),CV_32FC1);
		Ive[current_level+1].zeros (Ive[current_level+1].size (),CV_32FC1);
   
        //Gauss-seidel求解      
        gaussSeidelRecursive(current_level+1, max_level, first_level, 2*h,Iue, Ive);
               
        //执行延拓算子                            
		_resize (Iue[current_level+1], Iue[current_level] );
        _resize (Ive[current_level+1], Ive[current_level] );
        
        //更新解
		cv::add (Iu[current_level],Iue[current_level],Iu[current_level]);
		cv::add (Iv[current_level],Ive[current_level],Iv[current_level]);
    
        // 执行平滑迭代以消除延拓算子引入的误差
        gaussSeidelIteration(current_level, h, n2, J13, J23); 
        
                               
        //---------------------------- End 2nd V cycle -------------------------------------
              
    }
                                
}



/*
   Calculates the optical flow between two images.
*/
int OpticalFlow::CalcFlow(Mat& current , Mat& next, Mat& u , Mat& v, bool saved_data = false){
    
   /* if(!initialized)
      return 0;*/
 
    //Don't recalculate imgAfloat, just swap with imgBfloat
    if(saved_data){
		 cv::swap (current,next);
	   _resize (next,InSmall );
	   _convert(InSmall,Infloat);
	   _smooth(Infloat,Infloat,sigma);   
    }
    
    //Calculate imgAfloat as well as imgBfloat
    else{
		 
		_resize (current,IcSmall );
		_resize (next,InSmall  );
    	 
		_convert(IcSmall,Icfloat);
		_convert(InSmall,Infloat);
	//	 print_mat(Icfloat,"convert_Icfloat.txt");
	//	  print_mat(Infloat,"convert_Infloat.txt");
		 
		_smooth(Icfloat,Icfloat,sigma);
		_smooth(Infloat,Infloat,sigma);
	//	 print_mat(Icfloat,"smooth_Icfloat.txt");
	//	  print_mat(Infloat,"smooth_Infloat.txt");
    }
    
   
	cv::filter2D (Icfloat,Ifx,Ifx.depth (),maskx,cv::Point(-1,-1),0,cv::BORDER_REPLICATE );//X spacial derivative
	cv::filter2D (Icfloat,Ify,Ify.depth (),masky,cv::Point (-1,-1),0,cv::BORDER_REPLICATE );// Y spacial derivative

 
	cv::subtract (Infloat,Icfloat,Ift);// Temporal derivative
    
 
 
	cv::multiply (Ifx,Ifx,Ifxfx[0],1);
	cv::multiply (Ifx,Ify,Ifxfy[0],1);
	cv::multiply (Ifx,Ift,Ifxft[0],1);
	cv::multiply (Ify,Ify,Ifyfy[0],1);
	cv::multiply (Ify,Ift,Ifyft[0],1);
    
	//print_mat(Ifxfx[0],"Ifxfx[0].txt");
   
	_smooth(Ifxfx[0],Ifxfx[0] ,rho);
	_smooth(Ifxfy[0],Ifxfy[0] ,rho);
	_smooth (Ifxft[0],Ifxft[0]  ,rho);
	_smooth (Ifyfy[0],Ifyfy[0] ,rho);
	_smooth (Ifyft[0],Ifyft[0],rho);
   
	//print_mat(Ifxfx[0],"sIfxfx[0].txt"); //ok
    int i;
    
    //Fill all the levels of the multigrid algorithm with resized images
    for(i = 1; i < (max_level - start_level + 1); i++){

		_resize (Ifxfx[i-1],Ifxfx[i] );
		_resize (Ifxfy[i-1],Ifxfy[i] );
		_resize (Ifxft[i-1],Ifxft[i] );
		_resize (Ifyfy[i-1],Ifyfy[i] );
		_resize (Ifyft[i-1],Ifyft[i] );

		/*print_mat(Ifxfx[i],"fxfx.txt");
		print_mat(Ifxfy[i],"fxfy.txt");
		print_mat(Ifxft[i],"fxft.txt");
		print_mat(Ifyfy[i],"fyfy.txt");
		print_mat(Ifyft[i],"fyft.txt");*/
        
    }
    
    int k = (max_level - start_level);

    while(1){
    
        gaussSeidelRecursive(k, (max_level - start_level), k, powf(2.0,(float)(k)), Ifxft, Ifyft);
    
		  //print_mat(u,"u.txt");
          //print_mat(v,"v.txt");
        
        if(k > 0){
            
            // Transfer velocity from coarse to fine                           
			_resize (Iu[k],Iu[k-1] );
			_resize (Iv[k],Iv[k-1] );
			
            
            k--;
            
        }
        else
          break;
    
        
    }
    
    // Transfer to output image, resize if necessary
	_resize (Iu[0],u );
	_resize (Iv[0],v );
//	print_mat(u,"u.txt");
//	print_mat(v,"v.txt");
    
    // If started algorithm with smaller image than original, scale the flow field
    if(start_level > 0){
		u.convertTo (u,u.type (),powf(2.0, start_level));
		v.convertTo (v,v.type (),powf(2.0, start_level));
		
	}
	
return 1;
}

/*
  Draws the Motion Field
*/
void drawMotionField(Mat& u,Mat& v,Mat& motion, int xSpace, int ySpace, float cutoff, int multiplier, cv::Scalar  color)
{
     int x, y;
    
	 cv::Point  p0 = cvPoint(0,0);
	 cv::Point  p1 = cvPoint(0,0);
     
     float deltaX, deltaY, angle, hyp;
	 cv::Size usize=u.size ();
     for(y = ySpace; y <usize.height ; y+= ySpace ) {
        for(x = xSpace; x <usize.width ; x+= xSpace ){
         
            p0.x = x;
            p0.y = y;
            
            deltaX = *((float*)(u.data +y*u.step )+x);
            deltaY = -(*((float*)(v.data +y*v.step )+x));
            
            angle = atan2(deltaY, deltaX);
            hyp = sqrt(deltaX*deltaX + deltaY*deltaY);
   
            if(hyp > cutoff){
                   
                p1.x = p0.x + cvRound(multiplier*hyp*cos(angle));
                p1.y = p0.y + cvRound(multiplier*hyp*sin(angle));
                   
               
				cv::line (motion,p0,p1,color,1,8,0);
                
                p0.x = p1.x + cvRound(3*cos(angle-CV_PI + CV_PI/4));
                p0.y = p1.y + cvRound(3*sin(angle-CV_PI + CV_PI/4));
               
				cv::line (motion,p0,p1,color,1,8,0);
                
                p0.x = p1.x + cvRound(3*cos(angle-CV_PI - CV_PI/4));
                p0.y = p1.y + cvRound(3*sin(angle-CV_PI - CV_PI/4));
          
				cv::line (motion,p0,p1,color,1,8,0);
            }
      
        }
    }
    
}

