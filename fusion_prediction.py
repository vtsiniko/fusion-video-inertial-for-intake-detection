# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:53:20 2021

@author: tsini_1teem35
"""


from tensorflow.keras import models
import pickle as pkl
from functions_pred_eval import sync,create_model_fusion,window_inertial,window_video_new,predictions_fusion

import numpy as np
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
subject = np.array(subject)  
session = inertial_data['session']
inertial = sync(inertial)

def fusion_prediction( time_window, model_fusion, num_epochs, architecture):
    
   print('start')
   pred_fusion_all = []
   for i in range(1,13):
      print(i)

      path = current_path + '/model_' + str(model_fusion) + '_wl_' + str(time_window) + 'sec' 
      directory = "\model" + str(i) + '_' + architecture + '_' + str(num_epochs) 
      path2 = path + directory
      m_fusion = "model"+str(i)
      print(m_fusion)
      model_fusion = models.load_model(path2  + '/saved-model-' + architecture + '-epoch_' + str(num_epochs) + '_20.hdf5')
      pred_model_fusion = create_model_fusion(time_window)
      pred_model_fusion.set_weights(model_fusion.get_weights())
   
 

   #####video

      sessions = np.where(subject == i)
      sessions = np.asarray(sessions[0])
      print(sessions)
      bite_gt = []       
      video_time_test = []
      video_data_test = []
      inertial_data_test = []
  
      for i in range(len(sessions)):
      
         bite_gt.append(bite[sessions[i]]) 
         video_data_test.append(video[sessions[i]])
         video_time_test.append(video_time[sessions[i]])
         inertial_data_test.append(inertial[sessions[i]][:,:])
     

      # bite for label [-tmean ,+0.5]  ##### video
      for i in range(len (bite_gt)):
         for j in range(len(bite_gt[i])):
             bite_gt[i][j][1] = bite_gt[i][j][1]+0.5


      print('window data...')
      ws1 = 20 
      ws2 = 20
      wl_inertial = time_window*100
      inertial_window_test, inertial_time_last_test,ws_list = window_inertial(inertial_data_test,bite_gt,wl_inertial,ws1,ws2)
      for i in range(len(ws_list)):
          for k in range(len(ws_list[i])):
              if ws_list[i][k] == 20 :
                  ws_list[i][k] = 1
              if ws_list[i][k] == 20 :
                  ws_list[i][k] = 1
            
# window for video
      wl_video = time_window*5
      video_window_test, video_time_last_test = window_video_new(video_data_test,video_time_test,wl_video,ws_list)


      print('start predictions')
      pred_fusion = predictions_fusion(inertial_window_test, video_window_test,pred_model_fusion)
      pred_fusion_all.append(pred_fusion)
   
   pkl.dump( pred_fusion_all, open( path + '/pred_fusion_' + architecture + '.p', "wb" ) )



if __name__ == '__main__':
    time_window = int(sys.argv[1])
    model_fusion = str(sys.argv[2])
    num_epochs = int(sys.argv[3])
    architecture = str(sys.argv[4])
    print('model_' + model_fusion + '_time_' + str(time_window) +'_' + architecture )
    fusion_prediction(time_window,model_fusion,num_epochs,architecture)
           