#include "AR_Test.h"

void AR_Test::GeneratedData(Mat1d& P2C, Mat1d& P3C, Mat1d& RT_C)
{
	int n;
	RNG rng; // 实例化一个随机数发生器
	Pos2d_l = P2C.clone();
	Mat1d RT = GetRotationMatrix(RTs[f][0], RTs[f][1], RTs[f][2], RTs[f][3], RTs[f][4], RTs[f][5]);
	if (Pos2d_l.rows < N / 2)
	{
		n = N - Pos2d_l.rows;
		Mat1d D(1, n),pos2d(3,n);
		rng.fill(D, RNG::UNIFORM, 400.f, 900.f);
		rng.fill(pos2d.row(0), RNG::UNIFORM, 0.f, 1280.f);
		rng.fill(pos2d.row(1), RNG::UNIFORM, 0.f, 720.f);
		rng.fill(pos2d.row(2), RNG::UNIFORM, 1.f, 1.f);
		Mat1d Pos3Dl = (IntrinsicMatrix.inv() * Pos2d_l).mul(Mat1d::ones(3, 1) * D);
		Mat1d RT = GetRotationMatrix(RTs[f][0], RTs[f][1], RTs[f][2], RTs[f][3], RTs[f][4], RTs[f][5]);
		Mat1d Pos3Dc = RT * Pos3Dl;
		Pos3Dc.pop_back();
		D = Pos3Dc.row(2);
		Mat1d Pos2dc = (IntrinsicMatrix * Pos3Dc) / (Mat1d::ones(3, 1) * D);
		pos2d.pop_back();
		Pos2dc.pop_back();
	}
	else {}
	

}
void AR_Test::KeyPointsMatch(Mat1d& P2C, Mat1d& P3C)
{
	Pos2d_l = P2C.clone();
	Mat1d RT = GetRotationMatrix(RTs[f][0], RTs[f][1], RTs[f][2], RTs[f][3], RTs[f][4], RTs[f][5]);
	Mat1d Pos3Dc = RT * P3C;
	Pos3Dc.pop_back();
	Mat1d D = Pos3Dc.row(2);
	Mat1d Pos2dc = (IntrinsicMatrix * Pos3Dc) / (Mat1d::ones(3, 1) * D);
	Pos2dc.pop_back();
	Pos2d_c = Pos2dc.clone();
}
void AR_Test::INIT()
{
	GeneratedData();

	Mat1d RT=EMatrix_GetCameraAttitude(Pos2d_l.t(), Pos2d_c.t());
	RT[0][3] = RTs[f][3];
	RT[1][3] = (RTs[f][3] / RT[0][3]) * RTs[f][4];
	RT[2][3] = (RTs[f][3] / RT[0][3]) * RTs[f][5];
	RT_result.push_back(RT);
}
void AR_Test::GeneratedData()
{
	Mat1d D(1,N);
	RNG rng; // 实例化一个随机数发生器
	Pos2d_l= Mat1d(3, N);
	rng.fill(Pos2d_l.row(0), RNG::UNIFORM, 0.f, 1280.f);
	rng.fill(Pos2d_l.row(1), RNG::UNIFORM, 0.f, 720.f);
	rng.fill(Pos2d_l.row(2), RNG::UNIFORM, 1.f, 1.f);
	rng.fill(D, RNG::UNIFORM, 400.f, 900.f);
	Mat1d Pos3D_l = (IntrinsicMatrix.inv() * Pos2d_l).mul(Mat1d::ones(3, 1) * D);
	Pos3D_l.push_back(Mat1d::ones(1, N));
	Mat1d RT = GetRotationMatrix(RTs[f][0], RTs[f][1], RTs[f][2], RTs[f][3], RTs[f][4], RTs[f][5]);
	Mat1d Pos3D_c = RT * Pos3D_l;

	Pos3D_c.pop_back();
	D = Pos3D_c.row(2);
	Pos2d_c = (IntrinsicMatrix * Pos3D_c) / (Mat1d::ones(3, 1) * D);
	Pos2d_l.pop_back();
	Pos2d_c.pop_back();

}

void AR_Test::GetRT()
{

}

