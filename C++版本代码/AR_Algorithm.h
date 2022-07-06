#pragma once
#define _USE_MATH_DEFINES
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;
/** @brief ��ǿ��ʵ��ؼ��㺯���࣬�������ʾ���PNP�����ǻ��ȷ���.
* Triangulate_Get3DPoints() ���ǻ���2D����ָ�3D����;
* EMatrix_GetCameraAttitude() ���ʾ������֡�ǵ���Ϣ���������תƽ�ƾ���;
* PNP_GetCameraAttitude() PNP��������������ϵ�µ�2D�㼰��Ӧ��������ϵ�µ�3D����������ǰ��̬;
//@param IntrinsicMatrix��3X3����ڲ� Mat1d���ͣ�fx��fy��u0��v0.
*/
class AR_Algorithm
{
	
public:
	Mat1d IntrinsicMatrix;
	//@param IntrinsicMatrix��3X3����ڲ� Mat1d���ͣ�fx��fy��u0��v0.
	AR_Algorithm(Mat1d IntrinsicMatrix)
	{
		this->IntrinsicMatrix = IntrinsicMatrix;
	}
	//���ǻ��ָ��ռ��3D����
	Mat1d Triangulate_Get3DPoints( Mat1d RT_last, Mat1d RT_current, Mat1d Pos2D_last, Mat1d Pos2D_current);
	Mat1d Triangulate_Get3DPoints( Mat1d RT_last, Mat1d RT_current, vector<Point2d> Pos2D_last, vector<Point2d> Pos2D_current);
	//����תƽ����������ת����
	Mat1d GetRotationMatrix(double rx, double ry, double rz, double dx, double dy, double dz);
	//����ת����ָ���תƽ����
	double* RestoreRTValue(Mat1d RT);
	//���ʾ���ָ���תƽ����
	Mat1d EMatrix_GetCameraAttitude( Mat1d Pos2D_last, Mat1d Pos2D_current);
	Mat1d EMatrix_GetCameraAttitude( vector<Point2d> Pos2D_last, vector<Point2d> Pos2D_current);
	//PNP��ȡ�����ǰ��̬
	Mat1d PNP_GetCameraAttitude(Mat1d Pos3D, Mat1d Pos2D);
	Mat1d PNP_GetCameraAttitude( vector<Point3d> Pos3D, vector<Point2d> Pos2D);
};
