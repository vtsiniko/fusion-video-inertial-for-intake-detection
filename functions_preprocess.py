# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:18:09 2022

@author: tsini_1teem35
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:42:49 2021

@author: tsini_1teem35
"""

import copy 
import cv2
import numpy as np
from scipy import signal





################ create_video_dataset  ##################
def or_data(proc_data_vec,bite_gt_vec) :
    proc = []
    bite = []
    
    proc.append(copy.deepcopy(proc_data_vec[10])) 
    proc.append(copy.deepcopy(proc_data_vec[9]))
    proc.append(copy.deepcopy(proc_data_vec[13]))
    proc.append(copy.deepcopy(proc_data_vec[12]))
    proc.append(copy.deepcopy(proc_data_vec[11]))
    proc.append(copy.deepcopy(proc_data_vec[4]))
    proc.append(copy.deepcopy(proc_data_vec[3]))
    proc.append(copy.deepcopy(proc_data_vec[2]))
    proc.append(copy.deepcopy(proc_data_vec[20]))
    proc.append(copy.deepcopy(proc_data_vec[19]))
    proc.append(copy.deepcopy(proc_data_vec[18]))
    proc.append(copy.deepcopy(proc_data_vec[1]))
    proc.append(copy.deepcopy(proc_data_vec[17]))
    proc.append(copy.deepcopy(proc_data_vec[16]))
    proc.append(copy.deepcopy(proc_data_vec[15]))
    proc.append(copy.deepcopy(proc_data_vec[5]))
    proc.append(copy.deepcopy(proc_data_vec[7]))
    proc.append(copy.deepcopy(proc_data_vec[0]))
    proc.append(copy.deepcopy(proc_data_vec[14]))
    proc.append(copy.deepcopy(proc_data_vec[6]))
    proc.append(copy.deepcopy(proc_data_vec[8]))
    
    bite.append(copy.deepcopy(bite_gt_vec[10])) 
    bite.append(copy.deepcopy(bite_gt_vec[9]))
    bite.append(copy.deepcopy(bite_gt_vec[13]))
    bite.append(copy.deepcopy(bite_gt_vec[12]))
    bite.append(copy.deepcopy(bite_gt_vec[11]))
    bite.append(copy.deepcopy(bite_gt_vec[4]))
    bite.append(copy.deepcopy(bite_gt_vec[3]))
    bite.append(copy.deepcopy(bite_gt_vec[2]))
    bite.append(copy.deepcopy(bite_gt_vec[20]))
    bite.append(copy.deepcopy(bite_gt_vec[19]))
    bite.append(copy.deepcopy(bite_gt_vec[18]))
    bite.append(copy.deepcopy(bite_gt_vec[1]))
    bite.append(copy.deepcopy(bite_gt_vec[17]))
    bite.append(copy.deepcopy(bite_gt_vec[16]))
    bite.append(copy.deepcopy(bite_gt_vec[15]))
    bite.append(copy.deepcopy(bite_gt_vec[5]))
    bite.append(copy.deepcopy(bite_gt_vec[7]))
    bite.append(copy.deepcopy(bite_gt_vec[0]))
    bite.append(copy.deepcopy(bite_gt_vec[14]))
    bite.append(copy.deepcopy(bite_gt_vec[6]))
    bite.append(copy.deepcopy(bite_gt_vec[8]))
    subject = [1,1,2,2,2,3,3,3,4,4,4,5,6,6,6,7,8,9,10,11,12]
    session = [1,2,1,2,3,1,2,3,1,2,3,1,1,2,3,1,1,1,1,1,1]
   
    return proc, bite,subject,session

def video_time_data(path,proc):
  
     

  print('create dataset...')  
  tframe = []
  time_all = []
  videos = []
  #time_s = 0 
 
  time = []  
  
  for i in range(len(proc)):
     j = 0 
     video = cv2.VideoCapture(path+'/vid_'+str(i)+".mp4")  
     fps = video.get(cv2.CAP_PROP_FPS)  
     while(True): 
         ret, frame = video.read()
         #if temp > 6 :
         if ret == False:
             break
         #milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
         #seconds = milliseconds/1000
         
         t = j/fps
         tframe.append(frame[:,:,0]) #t_frame.append(frame)
         time_all.append(t)
         j = j + 1 
         
     time.append(np.array(time_all))
     videos.append(np.array(tframe))
     del video
     tframe = []
     time_all = []
     
     
  return videos,time



def edge_removal_video(videos, time, bite):
   
    for i in range(len(videos)):
        print(i)
        flag1 = True
        flag2 = True
        temp1 = 0
        temp2 = videos[i].shape[0]-1
        
        temp_t = len(videos[i])-1
        
        for j in range(len(videos[i])):
           
            if (time[i][j] > bite[i][0][0]-3) and flag1 == True:
                
                flag1 = False
                temp1 = j
                
            if  time[i][j] > bite[i][-1][1]+3 and flag2 == True:
                flag2 = False
                temp2 = j   
                  
                
        for k in range(temp2,temp_t+1):
            videos[i] = np.delete(videos[i],temp2,axis=0)
            time[i]= np.delete(time[i],temp2,axis=0)
        for k in range(0,temp1):
            videos[i]= np.delete(videos[i],0,axis=0)
            time[i]=np.delete(time[i],0,axis=0)
        
################ create_video_dataset  ##################

################ create_inertial_dataset  ##################


def mov_aver(X,l):
    print ("proc: move average filter")
    for i in range(len(X)):
        X[i][:,1] = signal.convolve(X[i][:,1],np.ones(l)/l,mode='same')
        X[i][:,2] = signal.convolve(X[i][:,2],np.ones(l)/l,mode='same')
        X[i][:,3] = signal.convolve(X[i][:,3],np.ones(l)/l,mode='same')
        X[i][:,4] = signal.convolve(X[i][:,4],np.ones(l)/l,mode='same')
        X[i][:,5] = signal.convolve(X[i][:,5],np.ones(l)/l,mode='same')
        X[i][:,6] = signal.convolve(X[i][:,6],np.ones(l)/l,mode='same')

    print("move average filter its done")    
    return X
    
    
 
def fir_filter(cut,numtaps,X):
    X_set = copy.deepcopy(X) 
    
    
    print ("proc: fir filter")
    for i in range(len(X)):
        t = X[i][:,0]
        fs = int(t.shape[0]/(t[-1]-t[0]))
       
        #print(fs)
        coef = signal.firwin(numtaps , cutoff = 2/fs , window= "hamming" , pass_zero = False)
        X_set[i][:,1] = signal.convolve(X_set[i][:,1],coef,mode='same')
        X_set[i][:,2] = signal.convolve(X_set[i][:,2],coef,mode='same')
        X_set[i][:,3] = signal.convolve(X_set[i][:,3],coef,mode='same')
    
    print ("fir filter its done")    
    return X_set

 
def scale_data(data):
    
  data_f = np.concatenate(data)
  std = np.std(data_f,0)
  mean = np.mean(data_f,0)


  data_scale = copy.deepcopy(data)
  for i in range(len(data_scale)):
      for j in range(1,data_scale[i].shape[1]):
          data_scale[i][:,j] = (data_scale[i][:,j] - mean[j])/std[j]
  return data_scale, std, mean       

def edge_removal(data_scale,bite):
    data_scale_edge = []

    for i in range(len(data_scale)):
        flag1 = True
        flag2 = True
        temp1 = 0
        temp2 = data_scale[i].shape[0]-1
      
        for j in range(len(data_scale[i])):
           
            if (data_scale[i][j,0] > bite[i][0][0]-3) and flag1 == True:
                
                flag1 = False
                temp1 = j
                
            if  data_scale[i][j,0] > bite[i][-1][1]+3 and flag2 == True:
                flag2 = False
                temp2 = j   
        
        data_scale_edge.append(data_scale[i][temp1:temp2,:])        
    return data_scale_edge    



################ create_inertial_dataset  ##################
