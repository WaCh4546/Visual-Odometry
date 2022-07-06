#include "AR_Algorithm.h"
/** @brief �˺������ڽ�6���ɶȵ���תƽ����ת��Ϊ4*4����תƽ�ƾ���.

@param rx�� ��λΪ�ȣ�double���ͣ�����Ŀ����X������ת�˶�������Ĵָָ��X��������,��ָ����Ϊ��ת������.
@param ry�� ��λΪ�ȣ�double���ͣ�����Ŀ����Y���������˶�������Ĵָָ��Y��������,��ָ����Ϊ����������.
@param rz�� ��λΪ�ȣ�double���ͣ�����Ŀ����Z����ƫ���˶�������Ĵָָ��Z��������,��ָ����Ϊƫ��������.
@param dx�� ��λΪmm��double���ͣ�����Ŀ����X����ƽ���˶�.
@param dy�� ��λΪmm��double���ͣ�����Ŀ����Y����ƽ���˶�.
@param dz�� ��λΪmm��double���ͣ�����Ŀ����Z����ƽ���˶�.
@return 1ά��4*4����תƽ�ƾ���,Ԫ�ص�λΪdouble.
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
/** @brief ��֪��תƽ�ƾ���ָ���ת�Ƕ���ƽ����.

@param RT��4*4����3*4����תƽ�ƾ��� Mat1d����.


@return 1*6�������ַ��{ rx,ry,rz,dx,dy,dz };.
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
/** @brief �˺���������֪��֡ͼƬ���໥ƥ�������ǵ���Ϣ���Լ���֡ͼƬ����ʱ������ڲΣ�����ȡ���ʾ����Իָ��ӵ�һ֡����ǰ֡�������תƽ�ƾ�������ȱʧԭͼ�е������Ϣ��ƽ�����������ֱ������޳߶���Ϣ��



@param Pos2D_last�� Mat1d���ͣ�2XN����һ֡�ǵ�����������ϵ�µ�����
@param Pos2D_current�� Mat1d���ͣ�2XN����ǰ֡�ǵ�����������ϵ�µ�����

@return RT: 4X4����һ֡����֡�������תƽ����.
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
/** @brief �˺���������֪��֡ͼƬ���໥ƥ�������ǵ���Ϣ���Լ���֡ͼƬ����ʱ������ڲΣ�����ȡ���ʾ����Իָ��ӵ�һ֡����ǰ֡�������תƽ�ƾ�������ȱʧԭͼ�е������Ϣ��ƽ�����������ֱ������޳߶���Ϣ��


@param lpoints�� vector<Point2d>���ͣ���һ֡�ǵ�����������ϵ�µ�����
@param cpoints�� vector<Point2d>���ͣ���ǰ֡�ǵ�����������ϵ�µ�����

@return RT: 4X4����һ֡����֡�������תƽ����.
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
/** @brief �˺���������֪ͼƬ����������ϵ�µĽǵ���Ϣ���ǵ��Ӧ����������ϵ�µĿռ����꣬�Լ�ͼƬ����ʱ������ڲΣ���PNP�ķ�����ȡ�����ǰ����������ϵ�µ�λ����̬��

@param Pos3D�� Mat1d���ͣ�3XN,�ǵ�����������ϵ�µ�����
@param Pos2D�� Mat1d���ͣ�2XN,�ǵ�����������ϵ�µĿռ�����

@return RT: 4X4���������תƽ����̬.
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
/** @brief �˺���������֪ͼƬ����������ϵ�µĽǵ���Ϣ���ǵ��Ӧ����������ϵ�µĿռ����꣬�Լ�ͼƬ����ʱ������ڲΣ���PNP�ķ�����ȡ�����ǰ����������ϵ�µ�λ����̬��

@param Pos3D�� vector<Point3f>���ͣ��ǵ�����������ϵ�µ�����
@param Pos2D�� vector<Point2f>���ͣ��ǵ�����������ϵ�µĿռ�����

@return RT: 4X4���������תƽ����̬.
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
/** @brief �˺���������֪��֡ͼƬ���໥ƥ�������ǵ���Ϣ���Լ���֡ͼƬ����ʱ�������������ϵ�µ�λ����̬����������ǵ�����������ϵ�¶�Ӧ�Ŀռ�����.

@param RT_last�� Mat1d���ͣ�4X4����һ֡�������������ϵ�µ���̬�����Դ�֡�������ϵΪ��������ϵ������4X4�ĵ�λ�󼴿�.
@param RT_current�� Mat1d���ͣ�4X4����ǰ֡�������������ϵ�µ���̬.
@param Pos2D_last�� Mat1d���ͣ�2XN����һ֡�ǵ�����������ϵ�µ�����
@param Pos2D_current�� Mat1d���ͣ�2XN����ǰ֡�ǵ�����������ϵ�µ�����

@return Pos3D: 3XN���ǵ��Ӧ��������ϵ�µĿռ�����.
 */
Mat1d AR_Algorithm::Triangulate_Get3DPoints(Mat1d RTlast, Mat1d RTcurrent, Mat1d Pos2D_last, Mat1d Pos2D_current)
{
	Mat1d POS3D,RT_last, temp, RT_current;
	RT_last = RTlast.clone();
	RT_current = RTcurrent.clone();
	//ȥ��RT���һ��[0,0,0,1]
	RT_last.pop_back();
	RT_current.pop_back();
	//��POS2DתΪ2ά������
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
/** @brief �˺���������֪��֡ͼƬ���໥ƥ�������ǵ���Ϣ���Լ���֡ͼƬ����ʱ�������������ϵ�µ�λ����̬����������ǵ�����������ϵ�¶�Ӧ�Ŀռ�����.

@param RT_last�� Mat1d���ͣ�4X4����һ֡�������������ϵ�µ���̬�����Դ�֡�������ϵΪ��������ϵ������4X4�ĵ�λ�󼴿�.
@param RT_current�� Mat1d���ͣ�4X4����ǰ֡�������������ϵ�µ���̬.
@param lpoints�� vector<Point2d>���ͣ���һ֡�ǵ�����������ϵ�µ�����
@param cpoints�� vector<Point2d>���ͣ���ǰ֡�ǵ�����������ϵ�µ�����

@return Pos3D: 3XN���ǵ��Ӧ��������ϵ�µĿռ�����.
 */
Mat1d AR_Algorithm::Triangulate_Get3DPoints(Mat1d RTlast, Mat1d RTcurrent, vector<Point2d> lpoints, vector<Point2d> cpoints)
{
	Mat1d POS3D, RT_last, RT_current;
	RT_last = RTlast.clone();
	RT_current = RTcurrent.clone();
	//ȥ��RT���һ��[0,0,0,1]
	RT_last.pop_back();
	RT_current.pop_back();
	triangulatePoints(IntrinsicMatrix * RT_last, IntrinsicMatrix * RT_current, lpoints, cpoints, POS3D);
	return POS3D.clone();
}
