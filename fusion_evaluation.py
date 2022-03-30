# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 01:05:32 2021

@author: tsini_1teem35
"""



import pickle as pkl
from functions_pred_eval import sync
from functions_pred_eval import predictions_threshold, F1
import sys
import os


    
    
    
    
current_path = os.getcwd()



inertial_data = pkl.load( open( "inertial_data.p", "rb" ) )
video_data = pkl.load( open( "video_data.p", "rb" ) )

inertial = inertial_data['data_inertial']
video = video_data['videos']
video_time = video_data['time']
bite = inertial_data['bite']
subject = video_data['subject']

inertial = sync(inertial)

def fusion_prediction(time_window, model_fusion, num_epochs, architecture, threshold, distance):
  
   path = current_path + '/model_' + str(model_fusion) + '_wl_' + str(time_window) + 'sec' 


   

   pred_fusion_all = pkl.load( open( path + '/pred_fusion_' + architecture + '.p', "rb" ) )

   pred_fusion = []
   for i in range(len(pred_fusion_all)):
       for j in range(len(pred_fusion_all[i])):
           pred_fusion.append(pred_fusion_all[i][j])
        
 
   for i in range(len(bite)):    
      for j in range(len(bite[i])):
         bite[i][j][1] = bite[i][j][1]+0.5
     

   time_last = [] 
   for i in range(len(pred_fusion)):
      time_last.append(inertial[i][time_window*100::20,0]) 
 

   pred_fusion_th, time_stamp_all_fusion = predictions_threshold(pred_fusion,float(threshold),int(distance),time_last)
   F1_fusion,Precision_fusion,Recall_fusion,TP_all_fusion,FP_all_fusion,FN_all_fusion = F1(bite,time_stamp_all_fusion)
   print('F1 = ' + str(F1_fusion))
     
if __name__ == '__main__':
    time_window = int(sys.argv[1])
    model_fusion = str(sys.argv[2])
    num_epochs = int(sys.argv[3])
    architecture = str(sys.argv[4])
    threshold = str(sys.argv[5])
    distance = str(sys.argv[6])

    print('model_' + model_fusion + '_time_' + str(time_window) +'_' + architecture +'_' + str(threshold) +'_' + str(distance)  )
    fusion_prediction(time_window,model_fusion,num_epochs,architecture,threshold,distance)
    
