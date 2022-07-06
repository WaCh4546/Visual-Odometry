#include "AR_Algorithm.h"
/** @brief 此函数用于将6自由度的旋转平移量转化为4*4的旋转平移矩阵.

@param rx： 单位为度，double类型，描述目标绕X轴做滚转运动，右手拇指指向X轴正方向,四指握向为滚转正方向.
@param ry： 单位为度，double类型，描述目标绕Y轴做俯仰运动，右手拇指指向Y轴正方向,四指握向为俯仰正方向.
@param rz： 单位为度，double类型，描述目标绕Z轴做偏航运动，右手拇指指向Z轴正方向,四指握向为偏航正方向.
@param dx： 单位为mm，double类型，描述目标沿X轴做平移运动.
@param dy： 单位为mm，double类型，描述目标沿Y轴做平移运动.
@param dz： 单位为mm，double类型，描述目标沿Z轴做平移运动.
@return 1维度4*4的旋转平移矩阵,元素单位为double.
 */
Mat1d AR_Algorithm::GetRotationMatrix(double rx, double ry, double rz, double dx, double dy, double dz)
{
	rx = rx * M_PI / 180;
	ry = ry * M_PI / 180;
	rz = rz * M_PI / 180;
	double R[4][4] = { cos(rz) * cos(ry), cos(rz) * sin(ry) * sin(rx) - sin(rz) * cos(rx), cos(rz) * sin(ry) * cos(rx) + sin(rz) * sin(rx), dx,
		sin(rz) * cos(ry), sin(rz) * sin(ry) * sin(rx) + cos(rz) * cos(rx), sin(rz) * sin(ry) * cos(rx) - cos(rz) * sin(rx), dy,
		-sin(ry), cos(ry) * sin(rx), cos(ry) * cos(rx), dz,
		0, 0, 0, 1 };
	Mat R_(4, 4, CV_64F, &R[0][0]);
	return R_.clone();
}
/** @brief 已知旋转平移矩阵恢复旋转角度与平移量.

@param RT：4*4或者3*4的旋转平移矩阵 Mat1d类型.


@return 1*6的数组地址，{ rx,ry,rz,dx,dy,dz };.
 */
double* AR_Algorithm::RestoreRTValue(Mat1d RT)
{
	double ry = asin(-RT[2][0]);
	double rx = asin(RT[2][1] / cos(ry));
	double rz = asin(RT[1][0] / cos(ry));
	double dx = RT[0][3];
	double dy = RT[1][3];
	double dz = RT[2][3];
	double a[6] = { rx,ry,rz,dx,dy,dz };
	return a;
}
/** @brief 此函数用于已知两帧图片中相互匹配的两组角点信息，以及两帧图片拍摄时相机的内参，来求取本质矩阵以恢复从第一帧到当前帧相机的旋转平移矩阵，由于缺失原图中的深度信息，平移量仅能体现比例，无尺度信息。



@param Pos2D_last： Mat1d类型，2XN，上一帧角点在像素坐标系下的坐标
@param Pos2D_current： Mat1d类型，2XN，当前帧角点在像素坐标系下的坐标

@return RT: 4X4，上一帧到本帧相机的旋转平移量.
 */
Mat1d AR_Algorithm::EMatrix_GetCameraAttitude( Mat1d Pos2D_last, Mat1d Pos2D_current)
{
	vector<Point2d> rpoints;
	vector<Point2d> lpoints;
	rpoints = Mat_<Point2d>(Pos2D_last);
	lpoints = Mat_<Point2d>(Pos2D_current);
	Mat1d E_mat = findEssentialMat(rpoints, lpoints, IntrinsicMatrix[0][0], Point2d(IntrinsicMatrix[0][2], IntrinsicMatrix[1][2]), RANSAC, 0.999, 1.f);
	Mat1d  RT,temp;
	Mat R, t;
	recoverPose(E_mat, rpoints, lpoints, R, t, IntrinsicMatrix[0][0], Point2d(IntrinsicMatrix[0][2], IntrinsicMatrix[1][2]));
	

	hconcat(R, t, RT);
	temp = (Mat_<double>(1, 4) << 0., 0., 0., 1.);
	vconcat(RT, temp, RT);
	return Mat1d(RT);
}
/** @brief 此函数用于已知两帧图片中相互匹配的两组角点信息，以及两帧图片拍摄时相机的内参，来求取本质矩阵以恢复从第一帧到当前帧相机的旋转平移矩阵，由于缺失原图中的深度信息，平移量仅能体现比例，无尺度信息。


@param lpoints： vector<Point2d>类型，上一帧角点在像素坐标系下的坐标
@param cpoints： vector<Point2d>类型，当前帧角点在像素坐标系下的坐标

@return RT: 4X4，上一帧到本帧相机的旋转平移量.
 */
Mat1d AR_Algorithm::EMatrix_GetCameraAttitude( vector<Point2d> lpoints, vector<Point2d> cpoints)
{

	Mat1d E_mat = findEssentialMat(lpoints, cpoints, IntrinsicMatrix[0][0], Point2d(IntrinsicMatrix[0][2], IntrinsicMatrix[1][2]), RANSAC, 0.999, 1.f);
	Mat1d R, t, RT, temp;

	recoverPose(E_mat, lpoints, cpoints, R, t);
	hconcat(R, t, RT);
	temp = (Mat_<double>(1, 4) << 0., 0., 0., 1.);
	vconcat(RT, temp, RT);
	return Mat1d(RT);
}
/** @brief 此函数用于已知图片中像素坐标系下的角点信息，角点对应的世界坐标系下的空间坐标，以及图片拍摄时相机的内参，以PNP的方法求取相机当前在世界坐标系下的位置姿态。

@param Pos3D： Mat1d类型，3XN,角点在像素坐标系下的坐标
@param Pos2D： Mat1d类型，2XN,角点在世界坐标系下的空间坐标

@return RT: 4X4，相机的旋转平移姿态.
 */
Mat1d AR_Algorithm::PNP_GetCameraAttitude( Mat1d Pos3D, Mat1d Pos2D)
{
	Mat1d r, t,RT, temp;
	solvePnP(Pos3D, Pos2D, IntrinsicMatrix, Mat1d::zeros(5, 1), r, t);
	Rodrigues(r, RT);
	hconcat(RT, t, RT);
	temp = (Mat_<double>(1, 4) << 0., 0., 0., 1.);
	vconcat(RT, temp, RT);
	return Mat1d(RT);
}
/** @brief 此函数用于已知图片中像素坐标系下的角点信息，角点对应的世界坐标系下的空间坐标，以及图片拍摄时相机的内参，以PNP的方法求取相机当前在世界坐标系下的位置姿态。

@param Pos3D： vector<Point3f>类型，角点在像素坐标系下的坐标
@param Pos2D： vector<Point2f>类型，角点在世界坐标系下的空间坐标

@return RT: 4X4，相机的旋转平移姿态.
 */
Mat1d AR_Algorithm::PNP_GetCameraAttitude( vector<Point3d> Pos3D, vector<Point2d> Pos2D)
{
	Mat1d r, t, RT, temp;
	Mat1d Pos3D1 = Mat(Pos3D);
	Mat1d Pos2D1 = Mat(Pos2D);
	solvePnP(Pos3D1, Pos2D1, IntrinsicMatrix, Mat1d::zeros(5, 1), r, t);
	Rodrigues(r, RT);
	hconcat(RT, t, RT);
	temp = (Mat_<float>(1, 4) << 0., 0., 0., 1.);
	vconcat(RT, temp, RT);
	return Mat1d(RT);
}
/** @brief 此函数用于已知两帧图片中相互匹配的两组角点信息，以及两帧图片拍摄时相机在世界坐标系下的位置姿态来解算两组角点在世界坐标系下对应的空间坐标.

@param RT_last： Mat1d类型，4X4，上一帧相机在世界坐标系下的姿态，若以此帧相机坐标系为世界坐标系，输入4X4的单位阵即可.
@param RT_current： Mat1d类型，4X4，当前帧相机在世界坐标系下的姿态.
@param Pos2D_last： Mat1d类型，2XN，上一帧角点在像素坐标系下的坐标
@param Pos2D_current： Mat1d类型，2XN，当前帧角点在像素坐标系下的坐标

@return Pos3D: 3XN，角点对应世界坐标系下的空间坐标.
 */
Mat1d AR_Algorithm::Triangulate_Get3DPoints(Mat1d RTlast, Mat1d RTcurrent, Mat1d Pos2D_last, Mat1d Pos2D_current)
{
	Mat1d POS3D,RT_last, temp, RT_current;
	RT_last = RTlast.clone();
	RT_current = RTcurrent.clone();
	//去除RT最后一行[0,0,0,1]
	RT_last.pop_back();
	RT_current.pop_back();
	//将POS2D转为2维点向量
	vector<Point2d> rpoints;
	vector<Point2d> lpoints;
	rpoints = Mat_<Point2d>(Pos2D_last);
	lpoints = Mat_<Point2d>(Pos2D_current);
	triangulatePoints(IntrinsicMatrix * RT_last, IntrinsicMatrix * RT_current, rpoints, lpoints, POS3D);
	hconcat(Mat1d::zeros(4, 3), Mat1d::ones(4, 1), temp);
	cout << POS3D.size << endl;
	cout << temp.size << endl;
	Mat1d aw = temp * POS3D;
	cout << aw << endl;
	POS3D = POS3D / (temp * POS3D);
	POS3D.pop_back();
	return POS3D.clone();
}
/** @brief 此函数用于已知两帧图片中相互匹配的两组角点信息，以及两帧图片拍摄时相机在世界坐标系下的位置姿态来解算两组角点在世界坐标系下对应的空间坐标.

@param RT_last： Mat1d类型，4X4，上一帧相机在世界坐标系下的姿态，若以此帧相机坐标系为世界坐标系，输入4X4的单位阵即可.
@param RT_current： Mat1d类型，4X4，当前帧相机在世界坐标系下的姿态.
@param lpoints： vector<Point2d>类型，上一帧角点在像素坐标系下的坐标
@param cpoints： vector<Point2d>类型，当前帧角点在像素坐标系下的坐标

@return Pos3D: 3XN，角点对应世界坐标系下的空间坐标.
 */
Mat1d AR_Algorithm::Triangulate_Get3DPoints(Mat1d RTlast, Mat1d RTcurrent, vector<Point2d> lpoints, vector<Point2d> cpoints)
{
	Mat1d POS3D, RT_last, RT_current;
	RT_last = RTlast.clone();
	RT_current = RTcurrent.clone();
	//去除RT最后一行[0,0,0,1]
	RT_last.pop_back();
	RT_current.pop_back();
	triangulatePoints(IntrinsicMatrix * RT_last, IntrinsicMatrix * RT_current, lpoints, cpoints, POS3D);
	return POS3D.clone();
}
