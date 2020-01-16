#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
using namespace cv ;
using namespace std ;

void make_binary(Mat a)
{
  int i,j;
  for(i=0;i<a.rows;i++)
    {
      for(j=0;j<a.cols;j++)
	{
	  if(a.at<uchar>(i,j)>127)
	    a.at<uchar>(i,j)=255;
	  else
	    a.at<uchar>(i,j)=0;
	}
    }
}

int isvalid(int i, int j, Mat a)
{

	if (i < 0 || j < 0 || i >= a.rows || j >= a.cols)
	  return 0;
	else
	  return 1;
}

void dfs_visit(Mat a, Mat visited, int i, int j, int count)
{
  int k,l;
  visited.at<uchar>(i,j)=255/count;
  for(k=i-1;k<i+2;k++)
    {
      for(l=j-1;l<j+2;l++)
	{
	  if(isvalid(k,l,a))
	    {
	      if(visited.at<uchar>(k,l)==0 && a.at<uchar>(k,l)==255)
		{
		  imshow("New",visited);
		  waitKey(1);
		  dfs_visit(a,visited,k,l,count);
		}
	    }
	}
    }
}

int main() 
{
  int i,j;
  int count=1;
  Mat a=imread("Binary1.png",0);
  make_binary(a);
  Mat visited(a.rows,a.cols,CV_8UC1,Scalar(0));
  for(i=0;i<a.rows;i++)
    {
      for(j=0;j<a.cols;j++)
	{
	  if(isvalid(i,j,a))
	    {
	      if(visited.at<uchar>(i,j)==0 && a.at<uchar>(i,j)==255)
		{
		  dfs_visit(a,visited,i,j,count);
	          count++;
		}
	    }
	    
	}
    }
  imshow("Original",a);
  imshow("Visited",visited);
  waitKey(0);
  return(0);
}
		  
		 
		 
