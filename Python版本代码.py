import cv2
import glob
import numpy as np
import math
from math import degrees as dg
import matplotlib.pyplot as plt
import pickle
#imgsize=(1920,1080)
#imgsize=(1280,720)
 
#imgsize=(3840,2160)
#fx=2303.05762010251
#fy=2332.34225886492
#u0=1878.01107990967
#v0=1509.84342575888

imgsize=(3840/2,2160/2)
fx=2303.05762010251/2
fy=2332.34225886492/2
u0=1878.01107990967/2
v0=1509.84342575888/2

#imgsize=(1280,720)
#fx=667.221
#fy=667.221
#u0=606.844
#v0=358.5794
init_f1=230
init_f2=80
step_f=20
cap = cv2.VideoCapture('DJI_0011.MP4')
save_data_dir=r"C:\Users\wang0\Desktop\VisualOdometry"
def save_pickle_data(save_data_dir, save_data, dataname):
    # 保存数据集
    with open(save_data_dir + "/" + dataname + ".pickle", "wb") as handle:
        # pickle.dump(save_data, handle)
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
 
 
def read_pickle_data(save_data_dir, dataname):
    with open(save_data_dir + "/" + dataname + ".pickle", 'rb') as handle:
        read_data = pickle.load(handle)
    handle.close()
    return read_data
class KEYPOINT(object):
    def __init__(self,pt,response ):
        self.pt=pt
        self.response=response
class KAD(object):
    """description of class"""
    def __init__(self,K=None,D=None,leftup=(),rightdown=(),P=None):
        self.leftup=leftup
        self.rightdown=rightdown
        self.KeyPoint=K
        self.Descriptor=D
        self.Pos3D=P
        self.preditxy=None



class KalmanFilter(object):
    def __init__(self):
        self.x = np.matrix('0. 0. 0. 0.').T 
        self.P = np.matrix(np.eye(4))*1000 # initial uncertainty
        self.result=[]
        self.observed_xy=[]
    def kalman_xy(self,x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
        """
        Parameters:    
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise 
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """
        return self.kalman(x, P, measurement, R, motion, Q,
                        F = np.matrix('''
                            1. 0. 1. 0.;
                            0. 1. 0. 1.;
                            0. 0. 1. 0.;
                            0. 0. 0. 1.
                            '''),
                        H = np.matrix('''
                            1. 0. 0. 0.;
                            0. 1. 0. 0.'''))

    def kalman(self,x, P, measurement, R, motion, Q, F, H):
        '''
        Parameters:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

        Return: the updated and predicted new values for (x, P)

        See also http://en.wikipedia.org/wiki/Kalman_filter

        This version of kalman can be applied to many different situations by
        appropriately defining F and H 
        '''
        # UPDATE x, P based on measurement m    
        # distance between measured and current position-belief
        y = np.matrix(measurement).T - H * x
        S = H * P * H.T + R  # residual convariance
        K = P * H.T * S.I    # Kalman gain
        x = x + K*y
        I = np.matrix(np.eye(F.shape[0])) # identity matrix
        P = (I - K*H)*P

        # PREDICT x, P based on motion
        x = F*x + motion
        P = F*P*F.T + Q

        return x, P
    def predit(self,observed_xy):
        self.observed_xy.append(observed_xy)
        R = 0.01**2
        self.x, self.P = self.kalman_xy(self.x, self.P, observed_xy, R)
        self.result.append((self.x[:2]).tolist())
        return (self.result[-1][0][0],self.result[-1][1][0])
    def draw(self):
        ob_x, ob_y = zip(*self.observed_xy)
        plt.plot(ob_x, ob_y, 'r-')
        kalman_x, kalman_y = zip(*self.result)
        plt.plot(kalman_x, kalman_y, 'g-')
        plt.show()
class FeatureDetection(object):
    def __init__(self,img,KP_num_min,quadripartion=False):
        self.img=img
        self.KP_num_min=KP_num_min
        #self.KeyPoints=[]
        #self.Descriptors=None
        self.quadripartion=quadripartion
        self.kad=[]
        KeyPoints=self.DetectKeyPoints()
        if quadripartion:
            self.Quadripartion(KeyPoints)
        else:
            self.kad=KeyPoints[0]
    def Quadripartion(self,KP=[]):#kp=[[],[],[]]
        KP1=[]
        for kp in KP:
            if len(kp)==0:
                continue
            xmin=kp[0].leftup[0]
            ymin=kp[0].leftup[1]
            xmax=kp[0].rightdown[0]
            ymax=kp[0].rightdown[1]
            xmean=int((xmin+xmax)/2)
            ymean=int((ymin+ymax)/2)
            area=[]
            for i in range(4):
                area.append([])
            for i in kp:
                if i.KeyPoint.pt[0]>xmin and i.KeyPoint.pt[0]<xmean:
                    if i.KeyPoint.pt[1]>ymin and i.KeyPoint.pt[1]<ymean:
                        i.leftup=(xmin,ymin)
                        i.rightdown=(xmean,ymean)
                        area[0].append(i)
                    else:
                        i.leftup=(xmin,ymean)
                        i.rightdown=(xmean,ymax)
                        area[2].append(i)

                elif i.KeyPoint.pt[0]>xmean and i.KeyPoint.pt[0]<xmax:
                    if i.KeyPoint.pt[1]>ymin and i.KeyPoint.pt[1]<ymean:
                        i.leftup=(xmean,ymin)
                        i.rightdown=(xmax,ymean)
                        area[1].append(i)
                    else:
                        i.leftup=(xmean,ymean)
                        i.rightdown=(xmax,ymax)
                        area[3].append(i)
            
            for areai in area:
                if len(areai)==1:
                    self.kad.append(areai[0])
                elif len(areai)>1:
                    KP1.append(areai)


        if len(self.kad)+len(KP1)>=self.KP_num_min :
            for kp in KP1:
                maxdistance=0
                maxkad=None
                for kad in kp:
                    if kad.KeyPoint.response>maxdistance:
                        maxdistance=kad.KeyPoint.response
                        maxkad=kad
                self.kad.append(maxkad)
            return True
        elif len(KP)==0:
            return False
        else:
            self.Quadripartion(KP1)

    def DetectKeyPoints(self,minThreashold=10000):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d_SIFT.create(nfeatures=minThreashold)
        keypoints, descriptor = sift.detectAndCompute(gray,None)
        KP=[]
        for i in range(len(keypoints)):
            KP.append(KAD(KEYPOINT(keypoints[i].pt,keypoints[i].response),descriptor[i],(0,0),(gray.shape[1],gray.shape[0]),None))
        KP1=[]
        KP1.append(KP)
        return KP1

    def DrawPoints(self,image,KDA=None):
        if KDA==None:
            KDA=self.kad
        for kad in KDA:
            image=cv2.circle(image, (int(kad.KeyPoint.pt[0]),int(kad.KeyPoint.pt[1])), 5, (0,0,255), -1)


class FeatureMatch(object):
    def __init__(self,KAD1=None,KAD2=None,radius=500):
        self.KAD1=[]
        self.KAD2=[]
        self.pos2d1=[]
        self.pos2d2=[]
        self.imgsize=KAD2[0].rightdown
        self.Match(KAD1,KAD2,radius)

    def LocatingPoints(self,currentxy,preditxy,radius):
        if preditxy is None:
            xy=currentxy
        else:
            xy=preditxy
        if xy[0]-radius<0:
            lp_x=0
        else:
            lp_x=xy[0]-radius

        if xy[1]-radius<0:
            lp_y=0
        else:
            lp_y=xy[1]-radius

        if xy[0]+radius>self.imgsize[0]:
            rd_x=self.imgsize[0]
        else:
            rd_x=xy[0]+radius

        if xy[1]+radius>self.imgsize[1]:
            rd_y=self.imgsize[1]
        else:
            rd_y=xy[1]+radius
        return (lp_x,lp_y),(rd_x,rd_y)

    def Match(self,KAD1,KAD2,r):
        KAD2_area=[]
        bfMatcher = cv2.FlannBasedMatcher()
        for k1 in KAD1:
            kad2_area=[]
            kad2_descriptor=[]
            
            leftup,rightdown =self.LocatingPoints(k1.KeyPoint.pt,k1.preditxy,r)
            for k2 in KAD2:
                if k2.KeyPoint.pt[0]>=leftup[0] and k2.KeyPoint.pt[0]<=rightdown[0] \
                and k2.KeyPoint.pt[1]>=leftup[1] and k2.KeyPoint.pt[1]<=rightdown[1]:
                    kad2_area.append(k2)
                    kad2_descriptor.append(k2.Descriptor)
            if len(kad2_area)!=0:
                kad1_descriptor=[]
                kad1_descriptor.append(k1.Descriptor)
                kad1_descriptor=np.array(kad1_descriptor)
                kad2_descriptor=np.array(kad2_descriptor)
                if kad2_descriptor.shape[0]>=2:
                    matches = bfMatcher.knnMatch(kad1_descriptor,kad2_descriptor, k=2)
                    good = [[m] for m, n in matches if m.distance < 0.4 * n.distance]
                    
                    if len(good)!=0:
                        self.pos2d1.append(k1.KeyPoint.pt)
                        self.pos2d2.append(kad2_area[good[0][0].trainIdx].KeyPoint.pt)
                        self.KAD1.append(k1)
                        self.KAD2.append(kad2_area[good[0][0].trainIdx])
        if len(self.pos2d1)!=0:
            self.pos2d1=np.array(self.pos2d1)
            self.pos2d2=np.array(self.pos2d2)
            return True
        return False

class SLAMInit(object):
    def __init__(self,KAD_last=None):
        self.initRT=np.eye(4)
        self.KAD_last=KAD_last
        self.K = np.array( \
        [[fx, 0.0000000e+00, u0], \
         [0.0000000e+00,  fy, v0], \
         [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
        self.d = np.array([[0],[0],[0],[0],[0]], dtype=np.float32)
        self.RT=None

    def EMatrix_GetCameraAttitude(self,KAD1,KAD2):
        pos2d1=self.KAD2Pos2d(KAD1)
        pos2d2=self.KAD2Pos2d(KAD2)    
        E, mask_match = cv2.findEssentialMat(pos2d1, pos2d2, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.999, threshold=2)                         
        i=0
        cou=len(KAD1)
        for j in range(cou):
            if mask_match[j][0]==0:
                del KAD1[i]
                del KAD2[i]
            else:
                i+=1
        pos2d1=self.KAD2Pos2d(KAD1)
        pos2d2=self.KAD2Pos2d(KAD2)
        E, mask_match = cv2.findEssentialMat(pos2d1, pos2d2, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.999, threshold=2)    
        if min(mask_match)==0:
            return self.EMatrix_GetCameraAttitude(KAD1,KAD2)
        else:
            pos, R, t, mask = cv2.recoverPose(E,pos2d1, pos2d2, self.K)
            cou=len(KAD1)
            i=0
            for j in range(cou):
                if mask[j][0]==0:
                    del KAD1[i]
                    del KAD2[i]
                else:
                    i+=1
            RT1=np.hstack((R,t))
            RT1=np.vstack((RT1,np.array([0,0,0,1])))
            return RT1
    def Triangulate_Get3DPoints(self,RT_last,RT_current,KAD1,KAD2):
        pos2d1=self.KAD2Pos2d(KAD1)
        pos2d2=self.KAD2Pos2d(KAD2)
        Rtold = np.matmul(self.K, RT_last) # 相机内参 相机外参
        Rtnew = np.matmul(self.K, RT_current) #

        points4D = cv2.triangulatePoints(Rtold, Rtnew, pos2d1.T, pos2d2.T)
        points4D /= points4D[3]       # 归一化
        Pos3D=points4D.T[:,:3]
        for i in range(len(KAD1)):
            KAD1[i].Pos3D=Pos3D[i]
            KAD2[i].Pos3D=Pos3D[i]
        return Pos3D
    def PNP_GetCameraAttitude(self,KAD3d,KAD2d):
        pos3D=self.KAD2Pos3d(KAD3d)
        pos2D=self.KAD2Pos2d(KAD2d)
        (success, rvec1, tvec1) =cv2.solvePnP(pos3D,pos2D,self.K,self.d)
        r_current=cv2.Rodrigues(rvec1)[0]
        RT=np.hstack((r_current,tvec1))[:]
        RT=np.vstack((RT,np.array([0,0,0,1])))
        return RT
    def KAD2Pos2d(self,KAD):
        pos2d=[]
        for k in KAD:
            pos2d.append(k.KeyPoint.pt)
        return np.array(pos2d)
    def KAD2Pos3d(self,KAD):
        pos3d=[]
        for k in KAD:
            #if abs(k.Pos3D[0])<100 and abs(k.Pos3D[1])<100 and abs(k.Pos3D[2])<100:
                pos3d.append(k.Pos3D)
        return np.array(pos3d)
    def RT2Value(self,R):
        ry = math.asin(-R[2][0])
        rx=math.asin(R[2][1]/math.cos(ry))
        rz=math.asin(R[1][0]/math.cos(ry))
        dx=R[0][3]
        dy=R[1][3]
        dz=R[2][3]
        return [dg(rx),dg(ry),dg(rz),dx,dy,dz]
    def Value2RT(self,rx,ry,rz,dx,dy,dz):
        rx=rx*math.pi/180
        ry=ry*math.pi/180
        rz=rz*math.pi/180
        R_=np.array(([ math.cos(rz)*math.cos(ry), math.cos(rz)*math.sin(ry)*math.sin(rx)-math.sin(rz)*math.cos(rx), math.cos(rz)*math.sin(ry)*math.cos(rx)+math.sin(rz)*math.sin(rx),dx],
                     [ math.sin(rz)*math.cos(ry), math.sin(rz)*math.sin(ry)*math.sin(rx)+math.cos(rz)*math.cos(rx), math.sin(rz)*math.sin(ry)*math.cos(rx)-math.cos(rz)*math.sin(rx),dy],
                     [-math.sin(ry),              math.cos(ry)*math.sin(rx),                                        math.cos(ry)*math.cos(rx),                                       dz],
                     [0,                          0,                                                                0,                                                               1 ]), dtype=np.double)
        return R_
    def GetRTInv(self,RT):
        R=np.linalg.inv(RT[:3,:3])
        T=-1*R@RT[:3,3:]
        rt=np.hstack((R,T))
        rt=np.vstack((rt,np.array([0,0,0,1])))
        return rt
    def Init(self,KAD_current):
        self.KAD_current=KAD_current
        RT=self.EMatrix_GetCameraAttitude(self.KAD_last, self.KAD_current)
        pos2d=self.KAD2Pos2d( self.KAD_current)
        RT_last=self.initRT[:3,:]
        RT_current=RT[:3,:]
        pos3d = self.Triangulate_Get3DPoints(RT_last, RT_current,self.KAD_last, self.KAD_current)
        
        pos3d=np.vstack((pos3d.T,np.ones((1,pos3d.shape[0]))))
        POS3D2=RT@pos3d
        Pos2D_2=((self.K@POS3D2[:3,:])/(POS3D2[2,:]))[:2,:]
        error=abs(Pos2D_2-pos2d.T)
        error=error[0]+error[1]

        cou=len(self.KAD_current)
        i=0
        err=error.tolist()
        for j in range(cou):
            if err[i]>=1:
                del err[i]
                del self.KAD_last[i]
                del self.KAD_current[i]
            else:
                i+=1
        error=np.array(err)
        error=error@np.ones((error.shape[0],1))/error.shape[0]
        self.RT=RT
        ###误差处理
        return RT,error

class Tracking(SLAMInit):
    def __init__(self,img_last,KAD,RT_init):#带有3维点的关键点、描述子作为初始点云
        self.img_last=img_last
        self.PointsCloud=[]
        self.PointsCloud.append(KAD)
        self.K = np.array( \
        [[fx, 0.0000000e+00, u0], \
         [0.0000000e+00,  fy, v0], \
         [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
        #self.K = np.array( \
        #[[960, 0.0000000e+00, 960], \
        # [0.0000000e+00,  540, 540], \
        # [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
        self.d = np.array([[0],[0],[0],[0],[0]], dtype=np.float32)
        self.RT_Points=[]
        self.RT_Points.append(RT_init)
        self.f=0
    def Track(self,KAD2):
        KAD1=self.PointsCloud[-1][:]
        match=FeatureMatch(KAD1,KAD2)
        print("")
        print("第"+str(len(self.RT_Points))+"帧")
        print("匹配到了"+str(len(match.KAD1))+"个点")
        if len(match.KAD1)<5:
            return False
        self.KAD_last=match.KAD1
        self.KAD_current=match.KAD2
        RT=self.PNP_GetCameraAttitude(match.KAD1,match.KAD2)
        self.RT_Points.append(RT)

        ERR=self.RebuildError(self.KAD_last,self.KAD_current,self.RT_Points[-1])
        print("重投影剔除剩下"+str(len(self.KAD_last))+"个点")
        if len(self.KAD_last)<5:
            self.RT_Points.pop()
            return False
        #计算预测点
        RTcurrent=self.RT_Points[-1]@np.linalg.inv(self.RT_Points[-2])
        pos3d=self.KAD2Pos3d(self.KAD_last)
        pos2d=self.KAD2Pos2d(self.KAD_current)
        pos3d=np.vstack((pos3d.T,np.ones((1,pos3d.shape[0]))))
        POS3D2=RTcurrent@pos3d
        Pos2D_2=((self.K@POS3D2[:3,:])/(POS3D2[2,:]))[:2,:].T
        for i in range(len(match.KAD1)):
            self.KAD_current[i].Pos3D=self.KAD_last[i].Pos3D
            self.KAD_current[i].preditxy=Pos2D_2[i]
        print("第"+str(len(self.RT_Points))+"帧位姿")  
        print(self.RT2Value(RT))
        return True
    def RebuildError(self,KAD3d,KAD2d,RT):
        pos3d=self.KAD2Pos3d(KAD3d)
        pos2d=self.KAD2Pos2d(KAD2d)
        pos3d=np.vstack((pos3d.T,np.ones((1,pos3d.shape[0]))))
        POS3D2=RT@pos3d
        Pos2D_2=((self.K@POS3D2[:3,:])/(POS3D2[2,:]))[:2,:]
        error=abs(Pos2D_2-pos2d.T)
        error=error[0]+error[1]
        print("重投影误差最大值"+str(max(error))+",最小值"+str(min(error)))

        cou=len(KAD3d)
        i=0
        err=error.tolist()
        print(err)
        for j in range(cou):
            if err[i]>=1:
                del err[i]
                del KAD3d[i]
                del KAD2d[i]
            else:
                i+=1
        error=np.array(err)
        error=error@np.ones((error.shape[0],1))/error.shape[0]
        return error
    def Refresh(self,f,img_current):
        self.f=f
        self.img_current=img_current
        f_track=FeatureDetection(img_current,100,False)
        if self.Track(f_track.kad[:])==False:
            return False
        
        if len(self.KAD_current)<=200:
            print("路标点："+str(len(self.KAD_current))+",进行补充")
            f_rebuild=FeatureDetection(self.img_last,300,True)
            fm_rebuild=FeatureMatch(f_rebuild.kad,f_track.kad)
            self.ReplenishPointsCloud(fm_rebuild.KAD1,fm_rebuild.KAD2)
            print("路标补充后，"+str(len(self.KAD_current))+"个")
        return True
    def ReplenishPointsCloud(self,KAD1,KAD2):
        RT_last=self.RT_Points[-2][:3,:]
        RT_current=self.RT_Points[-1][:3,:]
        pos3d = self.Triangulate_Get3DPoints(RT_last, RT_current,KAD1, KAD2)
        RTcurrent=self.RT_Points[-1]@np.linalg.inv(self.RT_Points[-2])
        pos3d1=np.vstack((pos3d.T,np.ones((1,pos3d.shape[0]))))
        POS3D2=RTcurrent@pos3d1
        Pos2D_2=((self.K@POS3D2[:3,:])/(POS3D2[2,:]))[:2,:].T
        cou=len(KAD1)
        i=0
        for j in range(cou):
            if pos3d[i][2]<0:
                pos3d= np.delete(pos3d, i,0)
                Pos2D_2= np.delete(Pos2D_2, i,0)
                del KAD1[i]
                del KAD2[i]
            else:
                KAD1[i].Pos3D=pos3d[i]
                KAD2[i].Pos3D=pos3d[i]
                KAD2[i].preditxy=Pos2D_2[i]
                i+=1


        #重构误差
        ERR=self.RebuildError(KAD1,KAD2,self.RT_Points[-1])
        for k1 in self.KAD_last:
            count=len(KAD1)
            i=0
            for j in range(count):
                if abs(k1.KeyPoint.pt[0]-KAD1[i].KeyPoint.pt[0])<1\
                    and abs(k1.KeyPoint.pt[1]-KAD1[i].KeyPoint.pt[1])<1:
                    del KAD1[i]
                    del KAD2[i]
                else:
                    i+=1
        self.KAD_last+=KAD1
        self.KAD_current+=KAD2
        self.PointsCloud.append(self.KAD_current)
        self.img_last=self.img_current[:]
        RT=self.PNP_GetCameraAttitude(self.KAD_current,self.KAD_current)
        self.RT_Points[-1]=RT
        #PNP检查
        print("补充路标后的位姿：")
        print(self.RT2Value(RT))
        print("本帧相机位姿：")
        print(self.RT2Value(self.GetRTInv(RT)))
    #def LoopbackVerification(self,):



def test():
    cm=KalmanFilter()
    N = 20
    true_x = np.linspace(0.0, 10.0, N)
    true_y = true_x**2
    observed_x = true_x + 0.05*np.random.random(N)*true_x
    observed_y = true_y + 0.05*np.random.random(N)*true_y
    for meas in zip(observed_x, observed_y):
        print(cm.predit(meas))
    cm.draw()
if __name__ == "__main__": 

    #rtresult=[]
    #e=read_pickle_data(save_data_dir, "跟踪")
    #for rt in e.RT_Points:
    #   rtresult.append(e.GetRTInv(rt))
    #   print(e.RT2Value(rtresult[-1]))
    #np.save('CAMTransfer',np.array(rtresult))
    #np.save('PointsCloud',np.array(e.PointsCloud))


    
    #cap = cv2.VideoCapture('out.avi')
    sum=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES ,init_f1)
    ret, frame0 = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES ,init_f1+init_f2)
    ret, frame1 = cap.read()

    a=FeatureDetection(frame0,300,True)
    b=FeatureDetection(frame1,100,False)
    c=FeatureMatch(a.kad,b.kad)
    d=SLAMInit(c.KAD1)
    RT,error=d.Init(c.KAD2)
    
    print("初始化rt,误差,初始路标个数,相机姿态")
    print(d.RT2Value(RT))
    print(error)
    print(len(d.KAD_current))
    print(d.RT2Value(d.GetRTInv(RT)))
    for kad in d.KAD_last:
        frame0=cv2.circle(frame0, (int(kad.KeyPoint.pt[0]),int(kad.KeyPoint.pt[1])), 6, (0,0,255), -1)
    for kad in d.KAD_current:
        frame0=cv2.circle(frame0, (int(kad.KeyPoint.pt[0]),int(kad.KeyPoint.pt[1])), 6, (0,255,0), -1)
    for i in range(len(c.KAD1)):
        cv2.line(frame0,(int(c.KAD1[i].KeyPoint.pt[0]),int(c.KAD1[i].KeyPoint.pt[1])),(int(c.KAD2[i].KeyPoint.pt[0]),int(c.KAD2[i].KeyPoint.pt[1])), (255, 255, 255),3)
    cv2.imwrite(str(init_f1+init_f2)+".png",frame0)
    #frame0=cv2.resize(frame0, (960, 540))
    save_pickle_data(save_data_dir, d, "初始化")
    
    #cv2.imshow("frameTRACK",frame0)
    #cv2.waitKey(5)
    iu=init_f1+init_f2+step_f
    e=Tracking(frame1,c.KAD2,RT)
    cap.set(cv2.CAP_PROP_POS_FRAMES ,init_f1+init_f2+step_f)
    ret, frame2 = cap.read()
    while iu<=sum-step_f:
        save_pickle_data(save_data_dir, e, "跟踪")
        if e.Refresh(iu,frame2)==False:
            print("第"+str(iu)+"帧未匹配到")
            iu+=step_f
            cap.set(cv2.CAP_PROP_POS_FRAMES ,iu)
            ret, frame2 = cap.read()
            continue
        for kad in e.KAD_last:
           frame2=cv2.circle(frame2, (int(kad.KeyPoint.pt[0]),int(kad.KeyPoint.pt[1])), 8, (0,0,255), -1)
        for kad in e.KAD_current:
           frame2=cv2.circle(frame2, (int(kad.KeyPoint.pt[0]),int(kad.KeyPoint.pt[1])), 8, (0,255,0), -1)
        for i in range(len(e.KAD_last)):
            cv2.line(frame2,(int(e.KAD_last[i].KeyPoint.pt[0]),int(e.KAD_last[i].KeyPoint.pt[1])),(int(e.KAD_current[i].KeyPoint.pt[0]),int(e.KAD_current[i].KeyPoint.pt[1])), (255, 255, 255),8)
        cv2.imwrite(str(iu)+".png",frame2)
        #frame2=cv2.resize(frame2, (960, 540))
        #cv2.imshow("frameTRACK",frame2)
        #cv2.waitKey(5)
        iu+=step_f
        cap.set(cv2.CAP_PROP_POS_FRAMES ,iu)
        ret, frame2 = cap.read()
    rtresult=[]
    save_pickle_data(save_data_dir, e, "跟踪")
    for rt in e.RT_Points:
       rtresult.append(e.GetRTInv(rt))
       print(e.RT2Value(rtresult[-1]))
    np.save('CAMTransfer',np.array(rtresult))
    np.save('PointsCloud',np.array(e.PointsCloud))
