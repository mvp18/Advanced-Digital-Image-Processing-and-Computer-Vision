#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

typedef struct
{
  int x;
  int y;
} point;

int isvalid(int i, int j, Mat a)
{

	if (i < 0 || j < 0 || i >= a.rows || j >= a.cols)
	  return 0;
	else
	  return 1;
}

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

void bfs_visit(Mat a, Mat visited, queue<point> q,int count)
{
  int i,j,l,m;
  for(i=0;i<a.rows;i++)
    {
      for(j=0;j<a.cols;j++)
	{
	  if(a.at<uchar>(i,j)>220 && visited.at<uchar>(i,j)==0)
	    {
	      count++;
	      //visited.at<uchar>(i,j)=255/count;
	      point temp;
	      temp.x=i;
	      temp.y=j;
	      q.push(temp);
	      while(!q.empty())
		{
		  point u;
		  u.x=q.front().x;
		  u.y=q.front().y;
		  q.pop();
		  for(l=u.x-1;l<u.x+2;l++)
		    {
		      for(m=u.y-1;m<u.y+2;m++)
			{
			  if(isvalid(l,m,a))
			    {
			      if(a.at<uchar>(l,m)>220 && visited.at<uchar>(l,m)==0)
				{
				  
				  visited.at<uchar>(l,m)=255/count;
				  point v;
				  v.x=l;
				  v.y=m;
				  q.push(v);
				  imshow("ImageB",visited);
				  waitKey(1);
				}
			    }
			}
		    }
		}
	    }
	}
    }
}

int main()
{
  int count=0;
  Mat a=imread("Binary1.png",0);
  Mat visited(a.rows,a.cols,CV_8UC1,Scalar(0));
  make_binary(a);
  queue<point> q;
  bfs_visit(a,visited,q,count);
  imshow("Final",visited);
  waitKey(0);
  return 0;
}
  
  
  
  
