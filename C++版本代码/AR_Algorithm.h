#pragma once
#define _USE_MATH_DEFINES
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;
/** @brief 增强现实相关计算函数类，包括本质矩阵、PNP、三角化等方法.
* Triangulate_Get3DPoints() 三角化从2D坐标恢复3D坐标;
* EMatrix_GetCameraAttitude() 本质矩阵从两帧角点信息计算相机旋转平移矩阵;
* PNP_GetCameraAttitude() PNP方法从像素坐标系下的2D点及对应世界坐标系下的3D点计算相机当前姿态;
//@param IntrinsicMatrix：3X3相机内参 Mat1d类型，fx，fy，u0，v0.
*/
class AR_Algorithm
{
	
public:
	Mat1d IntrinsicMatrix;
	//@param IntrinsicMatrix：3X3相机内参 Mat1d类型，fx，fy，u0，v0.
	AR_Algorithm(Mat1d IntrinsicMatrix)
	{
		this->IntrinsicMatrix = IntrinsicMatrix;
	}
	//三角化恢复空间点3D坐标
	Mat1d Triangulate_Get3DPoints( Mat1d RT_last, Mat1d RT_current, Mat1d Pos2D_last, Mat1d Pos2D_current);
	Mat1d Triangulate_Get3DPoints( Mat1d RT_last, Mat1d RT_current, vector<Point2d> Pos2D_last, vector<Point2d> Pos2D_current);
	//从旋转平移量计算旋转矩阵
	Mat1d GetRotationMatrix(double rx, double ry, double rz, double dx, double dy, double dz);
	//从旋转矩阵恢复旋转平移量
	double* RestoreRTValue(Mat1d RT);
	//本质矩阵恢复旋转平移量
	Mat1d EMatrix_GetCameraAttitude( Mat1d Pos2D_last, Mat1d Pos2D_current);
	Mat1d EMatrix_GetCameraAttitude( vector<Point2d> Pos2D_last, vector<Point2d> Pos2D_current);
	//PNP求取相机当前姿态
	Mat1d PNP_GetCameraAttitude(Mat1d Pos3D, Mat1d Pos2D);
	Mat1d PNP_GetCameraAttitude( vector<Point3d> Pos3D, vector<Point2d> Pos2D);
};
