# -*- coding: utf-8 -*-
import numpy as np
import cv2
import skvideo.io
import matplotlib.path as mplPath
import os
import pandas as pd
import time
'''
ROI issues need to be fixed!
'''

font = cv2.FONT_HERSHEY_SIMPLEX

class vehicle():
    def __init__(self,ID,pos):
        self.ID = ID
        self.type = 'unknown'
        # self.img = img
        self.pos = pos
        self.pos_history = [pos]
        dL,dR = get_distance(pos)
        if dL < dR:
            self.LorR = 'L'
        else:
            self.LorR = 'R'

    def update(self,pos):
        self.pos = pos
        if np.linalg.norm(pos-self.pos_history[-1]) <= 60:
            self.pos_history += [pos]

    def get_picture(self):
        pass

    def __del__(self):
        pass
        # print('******************************************')
        # print('I am removed {}'.format(ID))
        # print('******************************************')
# 0622 camera
# pts = [[180,60],[395,127.5],[160,245],[65,120],[180,60]]
# Left ROI boarder y = -0.5x +150 (points:(60,120),(180,60))
# Right ROI boarder y = -0.5x +325 (points:(395,127.5),(160,245))


# 0630 camera (香蕉灣?)
pts = [[50,10],[50,280],[350,280],[350,10],[50,10]]
# Left ROI boarder  (points:(20,20),(20,280))
# Right ROI boarder (points:(250,280),(250,20))


def get_distance(P):
    '''
    Get the perpendicular distance to L and R ROI boarder,
    Used to check whether this point is leaving the boarder
    and which way is it coming from.
    '''
    PR1 = np.array(pts[0])
    PR2 = np.array(pts[1])
    PL1 = np.array(pts[2])
    PL2 = np.array(pts[3])
    dL = np.cross(PL2-PL1, P-PL1)/np.linalg.norm(PL2-PL1)
    dR = np.cross(PR2-PR1, P-PR1)/np.linalg.norm(PR2-PR1)
    return dL,dR


def PIPC(pnt,polygon):
    '''
    Check whether a point is inside a polygon
    Return a BOOL
    '''
    crd = np.array(polygon)# poly
    bbPath = mplPath.Path(crd)
    r = 0.001 # accuracy
    isIn = bbPath.contains_point(pnt,radius=r) or bbPath.contains_point(pnt,radius=-r)
    return isIn


# n = 0
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.createBackgroundSubtractorMOG2()
kernel0 = np.ones((2,2),np.uint8)
kernel1 = np.ones((8,8),np.uint8)

pts_d = np.array(pts, np.int32).reshape((-1,1,2))



def run(video,output = False,f_skip_rate = 5):

    oLoR = {'L':0,'R':0} # output
    def DelLogic(veh):
        if len(veh.pos_history) >= 2 and np.linalg.norm(veh.pos_history[0]-veh.pos_history[-1]) >= 30:
            if veh.LorR == 'L':
                oLoR['L'] += 1
                print('L += 1')
            elif veh.LorR == 'R':
                oLoR['R'] += 1
                print('R += 1')
            else:
                print('Something wrong!!')
        else:
            print('This is a noise')

    vehicles = []
    ID = 0
    cap = cv2.VideoCapture(video)

    if output:
        # writer = skvideo.io.FFmpegWriter("output/0630/Detected_{}.mp4".format(video.replace('/','_')[:-4]))
        writer = skvideo.io.FFmpegWriter("TEST_{}.mp4".format(video.replace('/','_')[:-4]))
        f_id = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('*********************************')
            print('End of Video')
            print('*********************************')
            break

        frame = cv2.resize(frame,(400, 300), interpolation=cv2.INTER_AREA)
        cv2.polylines(frame,[pts_d],True,(0,255,255),3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # hist_eq = cv2.equalizeHist(gray)
        # blur = cv2.blur(gray, (12, 12))
        fgmask = fgbg.apply(gray)
        # fgmask[fgmask==127]=0
        erode = cv2.erode(fgmask,kernel0,iterations = 2)
        # dilation = cv2.dilate(erode,kernel0,iterations = 1)
        dilation = cv2.dilate(erode,kernel1,iterations = 2)
        imgs, cnts, __ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicle_poss = []
        for i, c in enumerate(cnts):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if len(c)>=5 and PIPC((cX,cY),pts):
                ellipse = cv2.fitEllipse(c)
                cv2.ellipse(frame,ellipse,(0,255,0),2)
                vehicle_poss.append(np.array([cX,cY]))

        if vehicle_poss: # if something is inside ROI
            if not vehicles:
                # first vehicle come in ROI

                for pos in vehicle_poss:
                    vehicles.append(vehicle(ID = ID,pos = pos))
                    ID += 1

            elif len(vehicle_poss) > len(vehicles):
                # new vehicle come in ROI
                closest_dist = np.array([min(get_distance(pos)) for pos in vehicle_poss])
                # get the closest pos to boarder lines (L or R)
                vehicles.append(vehicle(ID = ID, pos = vehicle_poss[closest_dist.argmin()]))
                ID += 1
                # Create a new vehicle to vehicles
                for veh in vehicles:
                    Dveh2pos = np.array([np.linalg.norm(pos-veh.pos) for pos in vehicle_poss])
                    if min(Dveh2pos) <= 40:
                        veh.update(vehicle_poss[Dveh2pos.argmin()])
                # Update vehicles

            elif len(vehicles) > len(vehicle_poss):
                # One vehicle leave ROI but there are still vehicles in ROI
                vehicles_temp = vehicles.copy()
                for pos in vehicle_poss:
                    Dveh2pos = np.array([np.linalg.norm(veh.pos-pos) for veh in vehicles])
                    try:
                        del vehicles_temp[Dveh2pos.argmin()]
                        # Compare vehicles.pos with vehicle_poss to determine which vehicle to delete
                        DelLogic(vehicles_temp[0])
                        try:
                            vehicles.remove(vehicles_temp[0])
                        except:
                            raise NameError('Fail to delete')
                    except:
                        print('Unknown bug?')

                # Execute delete logic
                for veh in vehicles:
                    Dveh2pos = np.array([np.linalg.norm(pos-veh.pos) for pos in vehicle_poss])
                    veh.update(vehicle_poss[Dveh2pos.argmin()])
                #update
            else:
                # update vehicle pos
                for veh in vehicles:
                    Dveh2pos = np.array([np.linalg.norm(pos-veh.pos) for pos in vehicle_poss])
                    veh.update(vehicle_poss[Dveh2pos.argmin()])

        else: #if there is nothing inside ROI
            if vehicles:
                while vehicles:
                    print('Delete?')
                    # Add delete logic (Not to use destructor)
                    DelLogic(vehicles[0])
                    del vehicles[0]

        cv2.putText(frame, 'R to L: {}'.format(oLoR['R']), (50, 50),font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'L to R: {}'.format(oLoR['L']), (50, 80),font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Total: {}'.format(oLoR['R']+oLoR['L']), (50, 110),font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow('Detected',frame)
        # cv2.imshow('HGBG',dilation)
        if output:
            if f_id % f_skip_rate == 0:
                writer.writeFrame(frame[:,:,::-1])
            f_id += 1
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print(len(vehicles))
            break
    #     n += 1
        
    if output:
        writer.close()            
    cap.release()
    cv2.destroyAllWindows()

    return oLoR

if __name__ == '__main__':
    videos = ['20180630/'+f for f in os.listdir('20180630/')]

    # run(video = '20180630/201807011300.mp4',output = True)
    outdict = {'FileName':[],'R to L':[],'L to R':[],'Total':[],'Process_Time':[]}

    for video in videos:
        S = time.time()
        OLOR = run(video = video,output=True)
        E = time.time()
        outdict['FileName'] += [video]
        outdict['R to L'] += [OLOR['R']]
        outdict['L to R'] += [OLOR['L']]
        outdict['Total'] += [OLOR['R'] + OLOR['L']]
        outdict['Process_Time'] += [round(E-S,2)]

    pd.DataFrame(outdict).to_excel('output/0630/summary.xlsx')