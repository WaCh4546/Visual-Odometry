from socket import *
from result import *
import cv2
import glob
import numpy as np
import math
from math import degrees as dg
import os
import time


x=[]
y=[]
z=[]

co=[]

ol=[]
oxl=[]
oyl=[]
ozl=[]
P=[]
save_data_dir="C:/Users/wang0/Desktop/VisualOdometry/0011new"
save_data_dir1="C:/Users/wang0/Desktop/VisualOdometry/0011new/img"
e=read_pickle_data(save_data_dir, "跟踪")
f=0
fourCC =  cv2.VideoWriter_fourcc(*'MJPG')
#指定视频的名字,编码方式,每秒播放的帧数,每帧的大小
out = cv2.VideoWriter("jack.avi",fourCC,5,(960, 540))
RT=e.GetRTInv(e.RT_Points[0])
d=e.RT2Value(RT)
d0=[d[0]/3,d[1]/3,d[2]/3,d[3]/3,d[4]/3,d[5]/3]
d1=[d0[0]*2,d0[1]*2,d0[2]*2,d0[3]*2,d0[4]*2,d0[5]*2]
t=0.67
aa=[0,0,0,0]
vv=[0,0,0,0]
for rt in e.RT_Points:

    RT=e.GetRTInv(rt)
    d=e.RT2Value(RT)
    a2=d[:]
    f+=1
    v=[(d[0]-d1[0])/t,(d[1]-d1[1])/t,(d[2]-d1[2])/t]
    a=[((d[3]-d1[3])/t-(d1[3]-d0[3])/t)/t,((d[4]-d1[4])/t-(d1[4]-d0[4])/t)/t,((d[5]-d1[5])/t-(d1[5]-d0[5])/t)/t]   
    d0=d1
    d1=d
    print(a)

    img = cv2.imread(save_data_dir1+"/ ("+str(f)+").png")
    #print(img.shape)  #(1080, 1920, 3)
    graph = np.ones((1080,500,3), dtype='uint8')*255

    rt1=e.Value2RT(20,-5,0,0,0,0)
    RT=rt@rt1
    RT=e.GetRTInv(RT)

    dd=120
    o=np.array([[0],[0],[0]])
    ox=np.array([[dd],[0],[0]])+o
    oy=np.array([[0],[dd],[0]])+o
    oz=np.array([[0],[0],[dd]])+o

    oo=RT[:3,:3]@o
    oox=RT[:3,:3]@ox
    ooy=RT[:3,:3]@oy
    ooz=RT[:3,:3]@oz
    dx=200
    dy=100
    x0=int(dx+oo[0,0]  )
    y0=int(dy+oo[1,0]  )
    xx=int(dx+oox[0,0] )
    xy=int(dy+oox[1,0] )
    yx=int(dx+ooy[0,0] )
    yy=int(dy+ooy[1,0] )
    zx=int(dx+ooz[0,0] )
    zy=int(dy+ooz[1,0] )
    text = "Camera Coordinate System"
    cv2.putText(graph, text, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    cv2.line(graph,(x0,y0),(zx,zy), (255, 0, 0),6)
    cv2.putText(graph, "Z", (zx+20,zy+20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 4)
    cv2.line(graph,(x0,y0),(xx,xy), (0, 0, 255),6)
    cv2.putText(graph, "X", (xx+10,xy-10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
    cv2.line(graph,(x0,y0),(yx,yy), (0, 255, 0),6)
    cv2.putText(graph, "Y", (yx+20,yy+20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 4)


    ########################################分隔符#######################################################################
    yf=y0+180
    cv2.line(graph,(0,yf),(500,yf), (0, 0, 0),3)

    #旋转量
    posemax=26
    yp=yf+30
    xpmin=60
    xp=195
    xpmax=195
    cv2.putText(graph, "Pose", (190,yp), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    
    #rz
    ##直方图正负分割线
    for i in range(10):
        i-=5
        cv2.line(graph,(xp+i,yp+10),(xp+i,yp+40), (0, 0, 0),1)
        cv2.line(graph,(xp+i,yp+60),(xp+i,yp+90), (0, 0, 0),1)
        cv2.line(graph,(xp+i,yp+110),(xp+i,yp+140), (0, 0, 0),1)
    cv2.putText(graph, str(round(d[2],3)), (xpmin+330,yp+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(graph, "RZ", (0,yp+40), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    rz=int((xpmax-xpmin)*d[2]/posemax)
    if rz>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp+5,yp+25+i),(rz+xp+5,yp+25+i), (255, 0, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp-5,yp+25+i),(rz-5+xp,yp+25+i), (255, 0, 0),1)
    #ry
    
    cv2.putText(graph, str(round(d[1],3)), (xpmin+330,yp+80), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(graph, "RY", (0,yp+85), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    ry=int((xpmax-xpmin)*d[1]/posemax)
    if ry>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp+5,yp+75+i),(ry+xp+5,yp+75+i), (0, 255, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp-5,yp+75+i),(ry-5+xp,yp+75+i), (0, 255, 0),1)
    #rx
    
    cv2.putText(graph, str(round(d[0],3)), (xpmin+330,yp+130), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(graph, "RX", (0,yp+135), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    rx=int((xpmax-xpmin)*d[0]/posemax)
    if rx>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp+5,yp+125+i),(rx+xp+5,yp+125+i), (0, 0, 255),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp-5,yp+125+i),(rx-5+xp,yp+125+i), (0, 0, 255),1)

   ########################################分隔符#######################################################################
    yf=yf+180
    cv2.line(graph,(0,yf),(500,yf), (0, 0, 0),3)
    #角速度
    Palstance=2
    yp=yf+30
    xpmin=60
    xp=195
    xpmax=195
    for i in range(10):
        i-=5
        cv2.line(graph,(xp+i,yp+10),(xp+i,yp+40), (0, 0, 0),1)
        cv2.line(graph,(xp+i,yp+60),(xp+i,yp+90), (0, 0, 0),1)
        cv2.line(graph,(xp+i,yp+110),(xp+i,yp+140), (0, 0, 0),1)
    cv2.putText(graph, "Palstance", (150,yp), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)

    cv2.putText(graph, str(round(v[2],3)), (xpmin+330,yp+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(graph, "PZ", (0,yp+40), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    pz=int((xpmax-xpmin)*v[2]/Palstance)
    if pz>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp+5,yp+25+i),(pz+xp+5,yp+25+i), (255, 0, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp-5,yp+25+i),(pz-5+xp,yp+25+i), (255, 0, 0),1)

    cv2.putText(graph, str(round(v[1],3)), (xpmin+330,yp+80), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(graph, "PY", (0,yp+85), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    py=int((xpmax-xpmin)*v[1]/Palstance)
    if py>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp+5,yp+75+i),(py+xp+5,yp+75+i), (0, 255, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp-5,yp+75+i),(py-5+xp,yp+75+i), (0, 255, 0),1)

    cv2.putText(graph, str(round(v[0],3)), (xpmin+330,yp+130), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(graph, "PX", (0,yp+135), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    px=int((xpmax-xpmin)*v[0]/Palstance)
    if px>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp+5,yp+125+i),(px+xp+5,yp+125+i), (0, 0, 255),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xp-5,yp+125+i),(px-5+xp,yp+125+i), (0, 0, 255),1)
    ########################################分隔符#######################################################################
    yf=yf+200
    cv2.line(graph,(0,yf),(500,yf), (0, 0, 0),3)
    #平移量
    yt=yf+30
    translationmax=19
    xtmin=60
    xt=195
    xtmax=195
    #dx
     ##直方图正负分割线
    for i in range(10):
        i-=5
        cv2.line(graph,(xt+i,yt+10),(xt+i,yt+40), (0, 0, 0),1)
        cv2.line(graph,(xt+i,yt+60),(xt+i,yt+90), (0, 0, 0),1)
        cv2.line(graph,(xt+i,yt+110),(xt+i,yt+140), (0, 0, 0),1)
    cv2.putText(graph, "Translation", (140,yt), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(graph, "DZ", (0,yt+35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(graph, str(round(d[5],3)), (xtmin+330,yt+35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    dz=int((xtmax-xtmin)*d[5]/translationmax)
    if dz>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt+5,yt+25+i),(dz+xt+5,yt+25+i), (255, 0, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt-5,yt+25+i),(dz-5+xt,yt+25+i), (255, 0, 0),1)



    cv2.putText(graph, "DY", (0,yt+85), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(graph, str(round(d[4],3)), (xtmin+330,yt+85), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    dy=int((xtmax-xtmin)*d[4]/translationmax)
    if dy>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt+5,yt+75+i),(dy+xt+5,yt+75+i), (0, 255, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt-5,yt+75+i),(dy-5+xt,yt+75+i), (0, 255, 0),1)

    cv2.putText(graph, "DX", (0,yt+130), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(graph, str(round(d[3],3)), (xtmin+330,yt+135), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    dx=int((xtmax-xtmin)*d[3]/translationmax)
    if dx>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt+5,yt+125+i),(dx+xt+5,yt+125+i), (0, 0, 255),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt-5,yt+125+i),(dx-5+xt,yt+125+i), (0, 0, 255),1)
    ########################################分隔符#######################################################################
    yf=yf+200
    cv2.line(graph,(0,yf),(500,yf), (0, 0, 0),3)
    #加速度
    yt=yf+30
    acceleration=0.67
    xtmin=60
    xt=195
    xtmax=195
    #dx
     ##直方图正负分割线
    for i in range(10):
        i-=5
        cv2.line(graph,(xt+i,yt+10),(xt+i,yt+40), (0, 0, 0),1)
        cv2.line(graph,(xt+i,yt+60),(xt+i,yt+90), (0, 0, 0),1)
        cv2.line(graph,(xt+i,yt+110),(xt+i,yt+140), (0, 0, 0),1)
    cv2.putText(graph, "Acceleration", (140,yt), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(graph, "AZ", (0,yt+35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(graph, str(round(a[2],3)), (xtmin+330,yt+35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
    az=int((xtmax-xtmin)*a[2]/acceleration)
    if az>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt+5,yt+25+i),(az+xt+5,yt+25+i), (255, 0, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt-5,yt+25+i),(az-5+xt,yt+25+i), (255, 0, 0),1)



    cv2.putText(graph, "AY", (0,yt+85), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(graph, str(round(a[1],3)), (xtmin+330,yt+85), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    ay=int((xtmax-xtmin)*a[1]/acceleration)
    if ay>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt+5,yt+75+i),(ay+xt+5,yt+75+i), (0, 255, 0),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt-5,yt+75+i),(ay-5+xt,yt+75+i), (0, 255, 0),1)

    cv2.putText(graph, "AX", (0,yt+130), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(graph, str(round(a[0],3)), (xtmin+330,yt+135), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    ax=int((xtmax-xtmin)*a[0]/acceleration)
    if ax>0:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt+5,yt+125+i),(ax+xt+5,yt+125+i), (0, 0, 255),1)
    else:
        for i in range(20):
            i-=10
            cv2.line(graph,(xt-5,yt+125+i),(ax-5+xt,yt+125+i), (0, 0, 255),1)


    img=cv2.hconcat([graph,img])
    img=cv2.resize(img, (960, 540))
    out.write(img)
    cv2.imshow("1",img)
    cv2.waitKey(1)
  
out.release()
cv2.destroyAllWindows()    
                   
    #无人机位姿
    #a=180*math.atan(d[4]/d[5])/math.pi
    #rt1=e.Value2RT(-a,0,0,0,0,0)
    #RT=rt@rt1
    #RT=e.GetRTInv(RT)

    #d=e.RT2Value(RT)
    #print(d)

   