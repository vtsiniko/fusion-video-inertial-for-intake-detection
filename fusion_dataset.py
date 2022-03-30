# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:24:57 2022

@author: tsini_1teem35
"""


import pickle as pkl
from functions_fusion import sync,window_inertial,window_video_new,label_last_frame
import os
import sys



def create_fusion_dataset(time_window, model_fusion):


   current_path = os.getcwd()

   inertial_data = pkl.load( open( "inertial_data.p", "rb" ) )
   video_data = pkl.load( open( "video_data.p", "rb" ) )

   inertial = inertial_data['data_inertial']
   video = video_data['videos']
   video_time = video_data['time']
   bite = inertial_data['bite']
   subject = video_data['subject']

   inertial = sync(inertial)

   # window for inertial
   wl_inertial = time_window*100 
   ws1 = 100
   ws2 = 20
   inertial_window, inertial_time_last,ws_list = window_inertial(inertial,bite,wl_inertial,ws1,ws2)
   for i in range(len(ws_list)):
       for k in range(len(ws_list[i])):
           if ws_list[i][k] == 100 :
               ws_list[i][k] = 5
           if ws_list[i][k] == 20 :
               ws_list[i][k] = 1
            
# window for video
   wl_video = time_window*5
   video_window, video_time_last = window_video_new(video,video_time,wl_video,ws_list)


# label 
   e = 0.5 
   inertial_label = label_last_frame(inertial_time_last,bite,e)


   dataset_window_fusion = {
   "inertial_window":inertial_window,
   "video_window":video_window,
   "label":inertial_label,
   "subject":subject,
   'bite':bite
    }    

   directory = '/model_' + str(model_fusion) + '_wl_' + str(time_window) + 'sec' 
   path = current_path + directory
   os.mkdir(path)

   print('save data..')
   pkl.dump( dataset_window_fusion, open( path + "/data_fusion.p", "wb" ) )
   print('save data..ok')

if __name__ == '__main__':
    time_window = int(sys.argv[1])
    model_fusion = str(sys.argv[2])
    print('model_' + model_fusion + '_time_' + str(time_window))
    create_fusion_dataset(time_window, model_fusion)
    