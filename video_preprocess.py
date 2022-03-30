# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:22:10 2021

@author: tsini_1teem35
"""


import face_recognition
import cv2
import os
from functions_preprocess_pre import or_data, video_time_data, edge_removal_video
import pickle as pkl
current_path = os.getcwd()+'\\video_sessions'

vid = 0 

   
for i in range(1,13):
    path_temp = current_path + "\S" + str(i)
    arr_temp = os.listdir(path_temp)
    print(arr_temp)
    for j in range(len(arr_temp)):
        path_video = path_temp + '\\' + arr_temp[j]
        print(path_video)
        if i < 10: 
           video = cv2.VideoCapture(path_video+"/VID_S0"+str(i)+'_0'+arr_temp[j]+'-nosound.mp4')
        else:
           video = cv2.VideoCapture(path_video+"/VID_S"+str(i)+'_0'+arr_temp[j]+'-nosound.mp4')

        fr = int(video.get(5))       
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('ok')
        frame_number = 0
        
        # face detection 
        while True:
           ret, frame = video.read()
           face_locations = face_recognition.face_locations(frame)
           if len(face_locations) != 0:
              top1 = face_locations[0][0]
              right1 = face_locations[0][1]
              bottom1 = face_locations[0][2]
              left1 = face_locations[0][3]
              break
    
        top = 0
        bottom = frame_height
        if right1 + int (100*frame_width/320)> frame_width:
           right = frame_width
        else:
           right = right1 + int (100*frame_width/320)
        if left1 - int (100*frame_width/320) < 0:
           left = 0
        else:
           left = left1 - int (100*frame_width/320)
        print(right)
        print(left)

        #change dimension for the video   
        dim = (128,128)

        #object to create video 
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  
        out = cv2.VideoWriter(current_path + '/vid_' + str(vid) + '.mp4',fourcc, 5, dim, False)
        
        sum = 0 
        sec = 0 
        frameRate = 0.2 
        while(True):
           sec = sec + frameRate 
           sec = round(sec, 2) 
           sum = sum +1
           print(sum)
           video.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
           ret, frame = video.read()
           if ret==True:
        
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              crop = frame[top:bottom , left:right]
              new_frame = cv2.resize(crop,dim,interpolation = cv2.INTER_AREA)
        
        #crop, downsample, gray 
              out.write(new_frame)       
           else:
    
              break
       
        video.release()
        out.release()
        cv2.destroyAllWindows()
        vid = vid + 1
 
with open('./FIC.pkl', 'rb') as fh:
    dataset = pkl.load(fh)

bite_gt_vec = dataset['bite_gt']
proc_data_vec = dataset['signals_proc']

proc, bite, subject, session  = or_data(proc_data_vec,bite_gt_vec)


videos, time = video_time_data(current_path,proc)

print('edge_removal....')
edge_removal_video(videos, time, bite)

dataset_video = {"bite":bite,
"videos":videos,                
"subject":subject,
"session":session, 
"time":time }



pkl.dump( dataset_video, open( "video_data.p", "wb" ) )

