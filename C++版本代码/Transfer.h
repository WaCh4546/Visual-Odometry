
#include<opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/core/utils/logger.hpp>
#include<vector>
#include<map>
using namespace std;
using namespace cv;



class Frame
{
public:
	vector<KeyPoint> keypoints; 
	Mat descriptor,Pos2D,Pos3D;

public:
	Frame(vector<KeyPoint> keypoints, Mat descriptor) //之后传入角点然后 
	{
		this->keypoints = keypoints;
		this->descriptor = descriptor;
	}
	Frame(const Frame& p)
	{
		this->keypoints = p.keypoints;
		this->descriptor = p.descriptor;
		this->Pos2D = p.Pos2D;
		this->Pos3D = p.Pos3D;
	}
	Frame(Mat Pos2D)
	{
		this->Pos2D = Pos2D;
	}
	Frame()
	{
	}
};
class Transfer
{
public:
	Frame frame_last, frame_current;
	Mat  Pos2D_L, Pos2D_C, Pos3D_L, Pos3D_C, RT;
public:
	Transfer(Frame frame_last, Frame frame_current)
	{
		this->frame_last = frame_last;
		this->frame_current = frame_current;
	}
	~Transfer()
	{
	}
	bool GetRT(); //由前后帧计算两帧间的旋转平移量
	bool CheckRT();
	Mat EMatrix(const Mat &Pos2D_L, const Mat& Pos2D_C);
	void EliminatingOutliers();

private:
	Mat IntrinsicMatrix;

};