
#include "Transfer.h"


void Transfer::EliminatingOutliers() //Òì³£ÖµÌÞ³ý
{
	Mat1d temp2D_L, temp2D_C, temp3D_L;
	Pos2D_L=Pos2D_L.t();
	Pos2D_C=Pos2D_C.t();
	Pos3D_L=Pos3D_L.t();
	for (int i = 0; i < Pos3D_L.rows;i++)
	{
		if (Pos3D_L.at<double>(i, 2) > 0.0)
		{
			temp2D_L.push_back(Pos2D_L.row(i));
			temp2D_C.push_back(Pos2D_C.row(i));
			temp3D_L.push_back(Pos3D_L.row(i));
		}		
	}
	Pos2D_L = temp2D_L.clone().t();
	Pos2D_C = temp2D_C.clone().t();
	Pos3D_L = temp3D_L.clone().t();
}

