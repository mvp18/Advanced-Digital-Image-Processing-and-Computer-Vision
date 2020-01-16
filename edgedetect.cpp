#include "opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include<iostream>
#include<math.h>
using namespace cv ;
using namespace std ;

int isvalid(int i, int j, Mat a)
{

	if (i < 0 || j < 0 || i >= a.rows || j >= a.cols)
	  return 0;
	else
	  return 1;
}

int main() 
{
  int i,j,gx4,l;
  Mat a=imread("cavepainting1.JPG",0);
  Mat b(a.rows,a.cols,CV_8UC1,Scalar(0));
  namedWindow("threshold",WINDOW_AUTOSIZE);
  createTrackbar("trackthreshold","threshold",&l,255);
  while(1)
    {
   for(i=1;i<a.rows-2;i++)
    {
      for(j=1;j<a.cols-2;j++)
	{
	  gx4=(abs((-2)*a.at<uchar>(i-1,j)+2*a.at<uchar>(i+1,j)+a.at<uchar>(i+1,j+1)+a.at<uchar>(i+1,j-1)-a.at<uchar>(i-1,j+1)-a.at<uchar>(i-1,j-1)))/4;
	  if(gx4>l)
	    b.at<uchar>(i,j)=255;
	  else
	    b.at<uchar>(i,j)=0;
	}
    }
  imshow("Imagea",a);
  imshow("threshold",b);
  waitKey(50);
    }
  return 0;
  
  
}
