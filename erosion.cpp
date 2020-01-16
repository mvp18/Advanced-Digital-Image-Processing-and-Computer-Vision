#include "opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include<iostream>
#include<algorithm>
using namespace cv ;
using namespace std ;
int main() 
{
  int i,j;
  Mat a=imread("erosion.png",0);
  Mat b=imread("erosion.png",0);
  Mat c(a.rows,a.cols,CV_8UC1,Scalar(0));
  for(i=1;i<a.rows-2;i++)
    {
      for(j=1;j<a.cols-2;j++)
	{
	  if(a.at<uchar>(i,j)==255 && ( a.at<uchar>(i-1,j)==0 || a.at<uchar>(i-1,j-1)==0 || a.at<uchar>(i-1,j+1)==0 ||a.at<uchar>(i+1,j)==0 || a.at<uchar>(i+1,j-1)==0 || a.at<uchar>(i+1,j+1)==0 ||  a.at<uchar>(i,j-1)==0 || a.at<uchar>(i,j+1)==0) )
	    b.at<uchar>(i,j)=0;
	}
    }
for(i=1;i<a.rows-2;i++)
    {
      for(j=1;j<a.cols-2;j++)
  	{
	  c.at<uchar>(i,j)=b.at<uchar>(i,j);
	}
    }
  
  for(i=1;i<a.rows-2;i++)
    {
      for(j=1;j<a.cols-2;j++)
  	{
  	  if(b.at<uchar>(i,j)==0 && ( b.at<uchar>(i-1,j)==255 || b.at<uchar>(i-1,j-1)==255 || b.at<uchar>(i-1,j+1)==255 ||b.at<uchar>(i+1,j)==255 || b.at<uchar>(i+1,j-1)==255 || b.at<uchar>(i+1,j+1)==255 ||  b.at<uchar>(i,j-1)==255 || b.at<uchar>(i,j+1)==255) )
    c.at<uchar>(i,j)=255;
	    	}
    }
  imshow("Imagea",a);
  imshow("imageb",c);
  waitKey(0);
  return 0;
}
