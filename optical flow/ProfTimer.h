
#ifndef __TIME_H__
#define PROFTIMER

#ifdef _WIN32

#include <time.h>
#include <Windows.h>

class ProfTimer {
	
	public:
	
	    ProfTimer(){
        };   
	
    	void Start(void) {
        	QueryPerformanceCounter(&mTimeStart);
    	};

    	void Stop(void) {
        	QueryPerformanceCounter(&mTimeStop);
    	};

    	float GetDurationInSecs(void)
    	{
        	LARGE_INTEGER freq;
        	QueryPerformanceFrequency(&freq);
        	float duration = (float)(mTimeStop.QuadPart-mTimeStart.QuadPart)/(float)freq.QuadPart;
        	return duration;
    	}

    private:

    	LARGE_INTEGER mTimeStart;
    	LARGE_INTEGER mTimeStop;

};

#else

#include <sys/time.h>

class ProfTimer {

	public:

	    ProfTimer(){
        };   

    	void Start(void) {
        	gettimeofday(&timeStart, NULL);
    	};

    	void Stop(void) {
        	gettimeofday(&timeStop, NULL);
    	};

    	float GetDurationInSecs(void)
    	{
            float dur = (timeStop.tv_sec - timeStart.tv_sec) + (timeStop.tv_usec - timeStart.tv_usec)/1000000.0;

        	return dur;
    	};

    private:

        timeval timeStart;
	    timeval timeStop;

};

#endif   


#endif  
