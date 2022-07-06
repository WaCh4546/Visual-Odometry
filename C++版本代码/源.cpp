#define _USE_MATH_DEFINES
#include<opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/core/utils/logger.hpp>
#include<vector>
#include<map>
#include "Transfer.h"
#include <cmath>
#include "AR_Test.h"

using namespace std;
using namespace cv;

Mat IntrinsicMatrix = (Mat_<double>(3, 3) << 667.2211303710938, 0, 606.8449096679688, 0, 667.2211303710938, 358.5794372558594, 0, 0, 1);
Mat F(double rx, double ry, double rz, double dx, double dy, double dz)
{
	rx = rx * M_PI / 180;
	ry = ry * M_PI / 180;
	rz = rz * M_PI / 180;
	double RZ[3][3] = {  cos(rz),-sin(rz),    0,
						 sin(rz), cos(rz),    0,
						    0   ,    0   ,    1    };					 
	Mat RZ1(3, 3, CV_64F, &RZ[0][0]);
	double RY[3][3] = {  cos(ry),    0   ,   sin(ry),
						 0      ,    1   ,     0    ,
						-sin(ry),    0   ,   cos(ry)};
	Mat RY1(3, 3, CV_64F, &RY[0][0]);
	double RX[3][3] = {    1    ,    0    ,    0     ,
						   0    ,  cos(rx),  -sin(rx), 
						   0    ,  sin(rx),   cos(rx)};
	Mat RX1(3, 3, CV_64F, &RX[0][0]);
	Mat1d RT= RZ1*  RY1 * RX1;
	vconcat(RT, Mat1d::zeros(1, 3), RT);
	Mat T = (Mat_<double>(4, 1) << dx, dy, dz,1);
	hconcat(RT,T , RT);
	
	//cout << RT << endl;




	double R[4][4] = { cos(rz) * cos(ry), cos(rz) * sin(ry) * sin(rx) - sin(rz) * cos(rx), cos(rz) * sin(ry) * cos(rx) + sin(rz) * sin(rx), dx,
		sin(rz) * cos(ry), sin(rz) * sin(ry) * sin(rx) + cos(rz) * cos(rx), sin(rz) * sin(ry) * cos(rx) - cos(rz) * sin(rx), dy,
		-sin(ry), cos(ry) * sin(rx), cos(ry) * cos(rx), dz,
		0, 0, 0, 1 };
	Mat R_(4, 4, CV_64F, &R[0][0]);

	//cout << R_ << endl;
	return R_.clone();
}
void test1()
{
	setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	Mat sc = imread("318042-106.jpg");
	Mat grayimg;

	Point3f  A = Point3f(2, 3, 4);
	Point2f  B = Point2f(2, 3);
	cout << B.dot(B) << endl;



	cvtColor(sc, grayimg, COLOR_BGR2GRAY);
	//Mat D(sc, Rect(10, 10, 10, 10)); // 用矩形界定
	int minHessian = 100;
	Ptr<SIFT> detector = SIFT::create(minHessian);//和surf的区别：只是SURF→SIFT
	vector<KeyPoint> keypoints;
	detector->detect(grayimg, keypoints, Mat());//找出关键点
	cout << Mat(keypoints[0].pt).t() << endl;
		// 绘制关键点
	Mat keypoint_img;
	drawKeypoints(grayimg, keypoints, keypoint_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("KeyPoints Image", keypoint_img);
	//Mat M4 = Mat_<double>(3, 3);
	//cout << M4 << endl;
	/*namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", grayimg);*/
	waitKey(0);
}
void test2()
{
	vector<Point3f> p;
	Point3f a = Point3f(1., 2., 3.);
	p.push_back(a);
	p.push_back(a);
	p.push_back(a);
	cout << p << endl;
	Mat m = Mat(p);
	m=m.reshape(1,9);

	p = Mat_<Point3f>(m);
	cout << p << endl;
}
void test3()
{
	Mat1d Pos2D(2, 5);
	Mat1d d(1, 5);
	RNG rng; // 实例化一个随机数发生器
	rng.fill(Pos2D, RNG::UNIFORM, 0.f, 1.f);
	rng.fill(d, RNG::UNIFORM, 200.f, 600.f);
	Pos2D = ((Mat_<double>(2, 1) << 1280, 720) * Mat1d::ones(1, 5)).mul(Pos2D);//(Mat_<double>(2, 1) << 1280, 720)初始化小点的矩阵直接赋值
	Pos2D.push_back(Mat1d::ones(1, 5));
	Mat Pos3D = (IntrinsicMatrix.inv() * Pos2D).mul(Mat1d::ones(3, 1)*d);
	
	Pos3D.at<double>(2, 0) = -10;
	Pos3D.at<double>(2, 1) = -10;
	cout<<Pos3D<<endl;
	/*Transfer t1(IntrinsicMatrix, Pos2D, Pos2D, Pos3D);
	t1.EliminatingOutliers();*/
	//print(t1.Pos3D_W);
}
void test4()
{
	Mat1d mat,mat1(1, 10), mat2(1, 10), mat3(1, 10),D(1,10);
	RNG rng; // 实例化一个随机数发生器
	rng.fill(mat1, RNG::UNIFORM, 0.f, 1280.f);
	rng.fill(mat2, RNG::UNIFORM, 0.f, 720.f);
	rng.fill(mat3, RNG::UNIFORM, 1.f, 1.f);
	rng.fill(D, RNG::UNIFORM, 200.f, 800.f);
	mat.push_back(mat1);
	mat.push_back(mat2);
	mat.push_back(mat3);
	//hconcat(mat2, mat3, mat1); //按列合并
	vconcat(mat2, mat3, mat1); //按行合并
	cout << mat1 << endl;

	double v1[3][3] = { 667.2211303710938, 0, 606.8449096679688,0,  667.2211303710938, 358.5794372558594,0,0,1 };
	Mat R_0(3, 3, CV_64F, &v1[0][0]);
	

	Mat pos3d = R_0.inv() * mat;
	//cout << R_0_ << endl;
	D = Mat::ones(3, 1, CV_64F)*D;
	//cout << D << endl;
	cout << pos3d.mul(D) << endl;
}
void test5() //验证由本质矩阵恢复出来的旋转平移量是前一帧相机坐标系下的3d坐标到下一帧相机坐标系下的3d坐标的变换关系，即前一帧
             //相机坐标系到下一帧相机坐标系的逆变换。
{
	int N = 10;
	//1 给出第一帧的2d，3d点坐标，并计算出下一帧的2d，3d点坐标
	
	Mat1d Pos2D(2, N),RT1;
	Mat1d d(1, N);
	RNG rng; // 实例化一个随机数发生器
	rng.fill(Pos2D, RNG::UNIFORM, 0.f, 1.f);
	rng.fill(d, RNG::UNIFORM, 200.f, 600.f);
	Pos2D = ((Mat_<double>(2, 1) << 1280, 720) * Mat1d::ones(1, N)).mul(Pos2D);//(Mat_<double>(2, 1) << 1280, 720)初始化小点的矩阵直接赋值
	Pos2D.push_back(Mat1d::ones(1, N));
	Mat Pos3D = (IntrinsicMatrix.inv() * Pos2D).mul(Mat1d::ones(3, 1) * d); 
	Pos3D.push_back(Mat1d::ones(1, N));
	Mat RT = F(2, 2, 2, 20, 30, 40);

	Mat Pos3D_c = RT* Pos3D;

	Pos3D_c.pop_back();
	d = Pos3D_c.row(2);
	Mat Pos2D_c = (IntrinsicMatrix * Pos3D_c)/ (Mat1d::ones(3, 1) * d);

	Pos2D.pop_back();
	Pos2D_c.pop_back();
	Pos2D = Pos2D.t();
	Pos2D_c = Pos2D_c.t();
	vector<Point2d> rpoints;
	vector<Point2d> lpoints;

	rpoints = Mat_<Point2d>(Pos2D);
	lpoints = Mat_<Point2d>(Pos2D_c);

	Mat E_mat = findEssentialMat(rpoints, lpoints, 667.2211303710938, Point2d(606.8449096679688, 358.5794372558594), RANSAC, 0.999, 1.f);
	Mat R, t;
	recoverPose(E_mat, rpoints, lpoints, R, t);
	R.push_back(Mat1d::zeros(1, 3));
	t.push_back(Mat1d::ones(1, 1));
	hconcat(R, t, RT1);
	//RT1 = RT1.inv();
	cout << "RT1:" << RT1 << endl;
	double ry = asin(-RT1[2][0]);
	double rx = asin(RT1[2][1] / cos(ry));
	double rz = asin(RT1[1][0] / cos(ry));
	cout << 180*rx/M_PI << " " << 180 * ry / M_PI << " " << 180 * rz / M_PI << endl;

	

}
void test6()//本质据很、三角化、PNP方法的验证。
{
	int N = 10;
	//1 给出第一帧的2d，3d点坐标，并计算出下一帧的2d，3d点坐标

	Mat1d Pos2D(2, N), RT1;
	Mat1d d(1, N);
	RNG rng; // 实例化一个随机数发生器
	rng.fill(Pos2D.row(0), RNG::UNIFORM, 0.f, 1.f);
	rng.fill(Pos2D.row(1), RNG::UNIFORM, 0.f, 1.f);
	//cout << Pos2D << endl;
	rng.fill(d, RNG::UNIFORM, 200.f, 600.f);
	Pos2D = ((Mat_<double>(2, 1) << 1280, 720) * Mat1d::ones(1, N)).mul(Pos2D);//(Mat_<double>(2, 1) << 1280, 720)初始化小点的矩阵直接赋值
	Pos2D.push_back(Mat1d::ones(1, N));
	Mat Pos3D = (IntrinsicMatrix.inv() * Pos2D).mul(Mat1d::ones(3, 1) * d);
	Pos3D.push_back(Mat1d::ones(1, N));
	Mat RT = F(32, 81, 24, 200, 300, 400);
	cout << "给定旋转量" << RT << endl;
	Mat Pos3D_c = RT * Pos3D;
	Pos3D.pop_back();
	Pos3D_c.pop_back();
	d = Pos3D_c.row(2);
	Mat Pos2D_c = (IntrinsicMatrix * Pos3D_c) / (Mat1d::ones(3, 1) * d);

	Pos2D.pop_back();
	Pos2D_c.pop_back();
	Pos2D = Pos2D.t();
	Pos2D_c = Pos2D_c.t();
	vector<Point2f> rpoints;
	vector<Point2f> lpoints;

	rpoints = Mat_<Point2d>(Pos2D);
	lpoints = Mat_<Point2d>(Pos2D_c);

	//2 本质矩阵验证，得到的本质矩阵即为前一帧相机坐标系下的3d点Pos3D到后一帧Pos3D_c的变换关系。Pos3D_c = RT * Pos3D;
	Mat E_mat = findEssentialMat(rpoints, lpoints, 667.2211303710938, Point2d(606.8449096679688, 358.5794372558594), RANSAC, 0.999, 1.f);
	Mat R, t;
	recoverPose(E_mat, rpoints, lpoints, R, t);
	R.push_back(Mat1d::zeros(1, 3));
	t.push_back(Mat1d::ones(1, 1));
	hconcat(R, t, RT1);
	cout << "本质矩阵验证:" << RT1 << endl;


	//3 三角化验证，得到的是两个相机所处在的公共世界坐标系下的角点的3D坐标，此时一第一个相机坐标系为世界坐标系，则可得到Pos3D
	Mat Rt_last,POS3D,temp;
	hconcat(Mat1d::eye(3, 3), Mat1d::zeros(3,1), Rt_last);
	RT.pop_back();
	triangulatePoints(IntrinsicMatrix *Rt_last, IntrinsicMatrix* RT, rpoints, lpoints, POS3D);

	
	
	
	hconcat(Mat::zeros(4, 3, CV_32F), Mat::ones(4, 1, CV_32F), temp);
	Mat aw = temp * POS3D;
	POS3D = POS3D / (temp * POS3D);
	POS3D.pop_back();
	cout << "第一帧实际3D" << Pos3D.t() << endl;
	cout << "转移后的3D" << Pos3D_c.t() << endl;
	cout << "三角化的3D" << POS3D.t() << endl;


	//4 PNP验证 以世界坐标系下的三维点以及对应相机坐标系下的二维点得到三维点从世界坐标系到相机坐标系的坐标变换关系，以第一帧的相机坐标系为世界坐标系得到的旋转平移量正是第一帧相机坐标系到第二帧相机坐标系三维点的变换关系即刚开始给出的RT
	Mat rr, tt;
	solvePnP(POS3D.t(), Pos2D_c, IntrinsicMatrix, Mat1d::zeros(5, 1), rr, tt);
	Rodrigues(rr, RT1);
	double ry = asin(-RT1[2][0]);
	double rx = asin(RT1[2][1] / cos(ry));
	double rz = asin(RT1[1][0] / cos(ry));
	cout << 180 * rx / M_PI << " " << 180 * ry / M_PI << " " << 180 * rz / M_PI << endl;
	cout << tt << endl;
}


void test7()
{
	Mat1d RT1(6, 2), Rt_last;
	RNG rng; // 实例化一个随机数发生器
	rng.fill(RT1.row(0), RNG::UNIFORM, 0.f, 8.f);
	rng.fill(RT1.row(1), RNG::UNIFORM, 0.f, 8.f);
	rng.fill(RT1.row(2), RNG::UNIFORM, 0.f, 8.f);
	rng.fill(RT1.row(3), RNG::UNIFORM, 0.f, 300.f);
	rng.fill(RT1.row(4), RNG::UNIFORM, 0.f, 100.f);
	rng.fill(RT1.row(5), RNG::UNIFORM, 0.f, 222.f);
	RT1 = RT1.t();
	cout << "设定的旋转平移量为： " << RT1 << endl;
	Mat1d IntrinsicMatrix = (Mat_<double>(3, 3) << 667.2211303710938, 0, 606.8449096679688, 0, 667.2211303710938, 358.5794372558594, 0., 0., 1.);
	AR_Test a(IntrinsicMatrix, 10, RT1);
	a.INIT();
	/// <summary>
	/// ////////////////////////////////////一帧
	/// </summary>
	Mat1d Pos3d_l=a.Triangulate_Get3DPoints(Mat1d::eye(4, 4), a.RT_result[a.f],a.Pos2d_l.t(),a.Pos2d_c.t());
	cout << Pos3d_l << endl;
	Pos3d_l.push_back(Mat::ones(1, a.N, CV_64F));
	a.Pos3d_w = a.RT_result[a.f] * Pos3d_l;
	a.f++;
	a.KeyPointsMatch(a.Pos2d_c, a.Pos3d_w);
	a.Pos3d_w.pop_back();
	a.RT_result.push_back(a.PNP_GetCameraAttitude(a.Pos3d_w.t(), a.Pos2d_c.t()));
	cout << a.RT_result[a.f] << endl;
	////////////////////////////////////////////

}
void test8()
{
	Mat1d rpoints , RT1;
	Mat1d lpoints ;

	double v1[10][2] = { 53.24459457397461, 200.5326080322266,
						141.1921234130859, 577.880859375,
						144.4255065917969, 571.1651611328125,
						145.0847778320313, 218.920166015625,
						146.2872772216797, 545.3117065429688,
						147.0303955078125, 566.847900390625,
						152.9879913330078, 555.9938354492188,
						153.0044097900391, 563.5020141601563,
						153.0570678710938, 702.9419555664063,
						155.8597869873047, 694.55029296875 };
	Mat R_0(10, 2, CV_64F, &v1[0][0]);
	double v2[10][2] = { 604.79345703125, 54.34686660766602,
						 447.5341186523438, 446.2546691894531,
						 39.17953491210938, 649.9733276367188,
						 89.80891418457031, 68.77220916748047,
						 413.4325561523438, 345.1985168457031,
						 448.9931640625, 450.4324340820313,
						 32.33852767944336, 659.2374877929688,
						 26.85315895080566, 674.1199340820313,
						 982.3158569335938, 454.0386352539063,
						 173.6414947509766, 446.4427490234375 };
	Mat R_1(10, 2, CV_64F, &v2[0][0]);

	Mat E_mat = findEssentialMat(R_0, R_1, 667.2211303710938, Point2d(606.8449096679688, 358.5794372558594), RANSAC, 0.999, 1.f);
	Mat R, t;
	recoverPose(E_mat, R_0, R_1, R, t);
	R.push_back(Mat1d::zeros(1, 3));
	t.push_back(Mat1d::ones(1, 1));
	hconcat(R, t, RT1);
	//RT1 = RT1.inv();
	cout << "RT1:" << RT1 << endl;
	double ry = asin(-RT1[2][0]);
	double rx = asin(RT1[2][1] / cos(ry));
	double rz = asin(RT1[1][0] / cos(ry));
	cout << 180 * rx / M_PI << " " << 180 * ry / M_PI << " " << 180 * rz / M_PI << endl;
}
int main()
{
	test6();

	return 0;
}