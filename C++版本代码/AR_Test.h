#pragma once
#include "AR_Algorithm.h"
class AR_Test :
    public AR_Algorithm
{public:
    int N,f;
    vector<Mat1d> RT_result;
    Mat1d RTs,Pos2d_l, Pos2d_c, Pos3d_w;
    AR_Test(Mat1d IntrinsicMatrix,int N,Mat1d RTs) :AR_Algorithm(IntrinsicMatrix)
    {
        this->RTs = RTs;
        this->N = N;
        this->f = 0;
    }
    void GeneratedData();//��ʼ֡��������
    void GeneratedData(Mat1d& P2C, Mat1d& P3C, Mat1d& RT_C); //����֡�������ݣ����ǵ���С��N�򲹵�N
    void EliminatingOutliers(Mat1d& P2L, Mat1d& P2C, Mat1d& P3L, Mat1d& P3C);
    void GetRT();
    void INIT();
    void KeyPointsMatch(Mat1d& P2C, Mat1d& P3C);
};

