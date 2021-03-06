# 单目视觉体感增强系统

​	这个项目主要是通过分析航拍的视频前后帧信息，对图像中的有明显特征的**点**进行**三维重建**，后续通过跟踪特征点来还原相机的运动信息。说白了就是一个单目视觉SLAM的复现，在实现的过程中也是参考了网上一个很知名的开源SLAM框架ORB-SLAM2（实时计算相机的位姿并同时对周围环境稀疏三维重建）现在做的这个东西就是这个框架下的粗糙版。
![image](https://user-images.githubusercontent.com/79780137/178089616-1a0d455e-8c1f-4bb4-b736-a3fef1b47209.png)


## 实现步骤：

### 1 单目初始化

​	首先使用对极几何原理(本质矩阵计算)，对视频中两张有**明显位移**的帧进行分析，得出相机在两帧间的姿态变化和位移量(*对极几何已经达到目的了？没有！*)。虽然对极几何原理可以直接求出相机的运动信息，但由于从二维信息求三维运动，缺少一维相机的深度信息，所以得到的实际位移量**只有X/Y/Z比例而无单位**(*一帧一帧只有比例的数据缺乏基准没办法进行比较*)。如果每帧都用对极几何求取的话，只能获得相机的姿态变化，位移量由于没有基准无法获得实际值，所以需要以初始帧对极几何原理求得的位移比例，将其归一化后作为**基准**来进行**三维重建**，之后X、Y、Z坐标的单位值即为该基准。上述操作为单目初始化操作。

### 2 三维空间点还原

​	根据对极几何已知了两帧间相机的位置姿态变化，根据三角化方法便可将两帧间对应的特征点还原到三维空间，即三维重建，而还原出的三维点也可作为“路标”供后续帧推算出当前相机相对于路标所在的位置。

### 3 下一帧姿态计算

​	根据第二帧图像中“路标”的特征信息对第三帧进行匹配，得到第二帧的“路标”特征点在第三帧帧图像中的二维信息。根据PNP方法，利用若干“路标”的三维信息及对应第三帧图像中的二维信息便可计算出第三帧帧图像当前相对于“路标”所在坐标系的位置姿态。

### 4 路标补充

​	获得新的位置姿态信息后，对第二帧和第三帧进行特征点检测和匹配，得到更多对应的点，利用新的位置姿态信息将这些点还原到三维空间作为更多的“路标”。

​	如此反复便可以持续下去，直到视频断帧、切换画面导致上下帧画面完全不连贯。

## 难点

​	实现主体框架并不困难，困难点在于实际的特征匹配、三维重建、位置姿态计算都有误差，而由于后续帧都是在前一帧的基础上计算的，所以误差会累积，不处理这些误差会导致程序计算不了几十帧后数据就会明显发散偏离实际。所以主要的难点在于对计算中误差的处理，项目的主要工作量也都在误差的处理上。

### 1 误匹配剔除

  在进行特征检测时，利用欧氏距离判别、本质矩阵去剔除误匹配。

​	欧式距离判别：筛选出与图1特征点匹配度最高的图2中的两个特征点分别计算欧氏距离L1、L2，若不能满足L1>>L2则认为图1中的特征点与图2中两个特征点相似度太高，出现误匹配的几率过大，从而剔除这个匹配。

​	本质矩阵剔除，通过两幅图的对应的特征点，计算其本质矩阵，剔除运动趋势与其余特征点差别过大的特征点。

### 2 重投影

​	在三维重建阶段，对还原出的“路标”点在对应二维相机平面重新投影，计算重投影后的点，与之前点的误差，剔除误差过大的点。

需要不断的去调参，筛选条件太严格会导致路标点减少，后续帧观察不到路标信息，从而导致计算中断；筛选条件太松计算误差会增大，从而经过累积导致数据发散。 

