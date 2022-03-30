# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:55:27 2021

@author: tsini_1teem35
"""

import tensorflow as tf 

import numpy as np 
from scipy.signal import find_peaks
import copy
from keras.layers.convolutional import Conv3D
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D,Input,Concatenate,MaxPooling3D, Dropout




def sync(inertial):
    inertial[0] = np.delete(inertial[0],np.s_[0:17],0)
    inertial[0] = np.delete(inertial[0],np.s_[75381:len(inertial[0])],0)
    inertial[1] = np.delete(inertial[1],np.s_[0:14],0)
    inertial[1] = np.delete(inertial[1],np.s_[90421:len(inertial[1])],0)
    inertial[2] = np.delete(inertial[2],np.s_[0:12],0)
    inertial[2] = np.delete(inertial[2],np.s_[44961:len(inertial[2])],0)
    inertial[3] = np.delete(inertial[3],np.s_[0:13],0)
    inertial[3] = np.delete(inertial[3],np.s_[48761:len(inertial[3])],0)
    inertial[4] = np.delete(inertial[4],np.s_[0:2],0)
    inertial[4] = np.delete(inertial[4],np.s_[52521:len(inertial[4])],0)
    inertial[5] = np.delete(inertial[5],np.s_[0:1],0)
    inertial[5] = np.delete(inertial[5],np.s_[102501:len(inertial[5])],0)
    inertial[6] = np.delete(inertial[6],np.s_[0:13],0)
    inertial[6] = np.delete(inertial[6],np.s_[72361:len(inertial[6])],0)
    inertial[7] = np.delete(inertial[7],np.s_[0:19],0)
    inertial[7] = np.delete(inertial[7],np.s_[80521:len(inertial[7])],0)
    inertial[8] = np.delete(inertial[8],np.s_[0:6],0)
    inertial[8] = np.delete(inertial[8],np.s_[73961:len(inertial[7])],0)
    inertial[9] = np.delete(inertial[9],np.s_[0:8],0)
    inertial[9] = np.delete(inertial[9],np.s_[66241:len(inertial[9])],0)
    inertial[10] = np.delete(inertial[10],np.s_[0:16],0)
    inertial[10] = np.delete(inertial[10],np.s_[52181:len(inertial[10])],0)
    inertial[11] = np.delete(inertial[11],np.s_[0:8],0)
    inertial[11] = np.delete(inertial[11],np.s_[100801:len(inertial[11])],0)
    inertial[12] = np.delete(inertial[12],np.s_[0:4],0)
    inertial[12] = np.delete(inertial[12],np.s_[52821:len(inertial[12])],0)
    inertial[13] = np.delete(inertial[13],np.s_[0:1],0)
    inertial[13] = np.delete(inertial[13],np.s_[78361:len(inertial[13])],0)
    inertial[14] = np.delete(inertial[14],np.s_[0:16],0)
    #video_dataset[14] = np.delete(video_dataset[14],np.s_[52521:len(inertial_data_5fps[14])],0)
    inertial[15] = np.delete(inertial[15],np.s_[0:19],0)
    inertial[15] = np.delete(inertial[15],np.s_[63041:len(inertial[15])],0)
    inertial[16] = np.delete(inertial[16],np.s_[0:17],0)
    inertial[16] = np.delete(inertial[16],np.s_[67621:len(inertial[16])],0)
    inertial[17] = np.delete(inertial[17],np.s_[0:14],0)
    inertial[17] = np.delete(inertial[17],np.s_[103341:len(inertial[17])],0)
    inertial[18] = np.delete(inertial[18],np.s_[0:0],0)#isws den xreaizetai 
    inertial[18] = np.delete(inertial[18],np.s_[82601:len(inertial[18])],0)
    inertial[19] = np.delete(inertial[19],np.s_[0:6],0)
    inertial[19] = np.delete(inertial[19],np.s_[80221:len(inertial[19])],0)
    inertial[20] = np.delete(inertial[20],np.s_[0:16],0)
    inertial[20] = np.delete(inertial[20],np.s_[35541:len(inertial[20])],0)
    
    return inertial
    


###### evaluate video ######


def window_inertial(x,bite,wl,ws1,ws2):
       
   window_inertial = [] 
   time_inertial_last = []
   ws_l = []
   for i in range(len(x)):
     print(i)  
     ws_list = []
     temp = []
     time_t = []   
     j = 1
     while j < len(x[i]): 
        ws = ws1 
        if j + wl > len(x[i]):      
           break
        t = x[i][j:j+wl,1:7]
        time_t.append(x[i][j+wl-1,0])
        for k in range(len(bite[i])):
            if (x[i][j+wl-1,0] > (bite[i][k,1]+ bite[i][k,0])/2 and x[i][j+wl-1,0] < bite[i][k,1])  :               
                ws = ws2
                break                         
        temp.append(t)
        ws_list.append(ws)
        del t 
        j = j + ws
     ws_l.append(np.array(ws_list))    
     window_inertial.append(np.array(temp))
     time_inertial_last.append(np.array(time_t))
   return   window_inertial, time_inertial_last,ws_l



def window_video_new(videos,time,wl,ws_list):
       
   window_frames = [] 
   time_window_last = []

   for i in range(len(videos)):
     print(i)  

     temp = []
     time_t = []   
     j = 1
     for k in range(len(ws_list[i])):
        if j + wl > len(videos[i]): break
        t = videos[i][j:j+wl,:,:]
        time_t.append(time[i][j+wl-1])
        j = j + int(ws_list[i][k])                   
        temp.append(t)       
        
            
     window_frames.append(np.array(temp))
     time_window_last.append(np.array(time_t))
   return   window_frames, time_window_last  

    



def create_model_fusion(time_window): 
   
    
   # INERTIAL MODEL
   wl1 = 100*time_window
   wl2 = 5*time_window
   in_1D = Input((300, 6))
   # conv1D_1
   model_1D = Conv1D(32,input_shape=(wl1, 6), kernel_size= 5 ,padding = 'same', strides=1, activation='relu')(in_1D)
   #model_1D = BatchNormalization()(model_1D)
   model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # conv1D_1
   model_1D = Conv1D(64, input_shape=(round(wl1), 6),kernel_size= 3 ,padding = 'same', strides=1 , activation='relu')(model_1D)
   #model_1D = BatchNormalization()(model_1D)
   model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # conv1D_1
   model_1D = Conv1D(128, input_shape=(round(wl1), 6), kernel_size= 3,padding = 'same', strides=1 , activation='relu')(model_1D)
   #model_1D = BatchNormalization()(model_1D)
   #model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # LSTM_1 
   model_1D = LSTM(128, dropout=0.2,recurrent_dropout=0.5, activation ='hard_sigmoid', return_sequences=False )(model_1D)
   model_1D = Dropout(0.5)(model_1D) 
   model_1D = Flatten()(model_1D)

   # VIDEO
   
   in_3D = Input((wl2, 128, 128, 1))
   # conv3D_2
   model_3D = Conv3D(32,input_shape=(wl2,128,128,1), kernel_size=(3, 3,3) ,padding = 'same', strides = (1,1,1), activation='relu')(in_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'same')(model_3D)
    # conv3D_2
   model_3D = Conv3D(32,input_shape=(round(wl2),64,64,32), kernel_size=(3, 3,3) ,padding = 'same', strides= (1,1,1), activation='relu')(model_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'valid')(model_3D)
    # conv3D_2
   model_3D = Conv3D(64,input_shape=(round(wl2),32,32,32), kernel_size=(3, 3,3) ,padding = 'same', strides = (1,1,1), activation='relu')(model_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'valid')(model_3D)
    # conv3D_2
   model_3D = Conv3D(64,input_shape=(round(wl2),16,16,64), kernel_size=(3, 3,3) ,padding = 'same', strides = (1,1,1), activation='relu')(model_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'valid')(model_3D)
    # conv3D_2
   model_3D = Flatten()(model_3D)
 
    

   # FUSION 

   conc = Concatenate()([model_1D, model_3D])
   fusion_model = Dense(128, activation='relu')(conc) 
   fusion_model = Dropout(0.5)(fusion_model) 
   fusion_model = Dense(32, activation='relu')(fusion_model) 
   fusion_model = Dropout(0.5)(fusion_model) 
   fusion_model = Dense(1, activation='sigmoid')(fusion_model) 
   model = Model(inputs=[in_1D, in_3D], outputs=[fusion_model])


   return model 




class DataGenerator_fusion_test(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_inertial,data_video,batch_size=1, dim_video=(25,128,128),dim_inertial=(500,6), shuffle=False):
        'Initialization'
        self.data_inertial = data_inertial
        self.dim_video = dim_video
        self.batch_size = batch_size
        self.data_video = data_video
        
        self.shuffle = shuffle
        self.on_epoch_end()
      
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int( (len(self.data_inertial))/self.batch_size)
   
            
            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print(index)
        x_inertial_test,x_video_test = self.__data_generation(index)      
        x_video_test = x_video_test[:,:,:,:, np.newaxis] 
        x_test = [x_inertial_test ,x_video_test/255]
        
        return x_test
       
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.videos))
        
    def __data_generation(self,index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
     
        # Generate data
        n = index*0
        
        tempx1 = []
        tempx2 = []
       
        
        x1 = []
        x2 = []
        for j in range(self.batch_size):
            if ((index+j)+n) > len(self.data_inertial):
                break 
           
            tempx1.append(self.data_inertial[(index+j)+n])
            tempx2.append(self.data_video[(index+j)+n])
      
        x1 = np.array(tempx1)
        x2 = np.array(tempx2)
        
        
        
        return x1,x2   
          



def predictions_fusion(test_inertial,test_video,pred_model):
     
    
    predictions = []
    for i in range(len(test_inertial)):
        test_gen = DataGenerator_fusion_test(test_inertial[i],test_video[i])
        pred0 = pred_model.predict(test_gen)
        pred_down = pred0[:,0]
        predictions.append(pred_down)
        
    return predictions  

    
  
def F1(bite,time_stamp):
    TP_all = []
    FN_all = []
    FP_all = []
    for k in range(len(bite)):
        TP,FP,FN = find_metrics(bite[k],time_stamp[k])
        TP_all.append(TP)
        FP_all.append(FP)
        FN_all.append(FN)
            
    Precision = (sum(TP_all)/(sum(TP_all)+sum(FP_all)))   
    Recall = (sum(TP_all)/(sum(TP_all)+sum(FN_all))) 
    F1 = ((2*Precision*Recall/(Precision+Recall)))
    return F1,Precision,Recall,TP_all,FP_all,FN_all
  
  
def predictions_threshold(predictions_th,threshold,d,time):
    p_th = copy.deepcopy(predictions_th)
    peaks_all = []
    time_stamp_all = []
    
    for i in range(len(predictions_th)):
      
      t_s1 = []
      time_s = []
      for j in range(len(predictions_th[i])):
          if predictions_th[i][j] < threshold: 
                p_th[i][j] = 0
          #else:
                #p_th[i][j] = 1       
      peaks,_ = find_peaks(p_th[i], distance = d)
      
      for k in range(len(peaks)):
          
          t_s1.append(time[i][peaks[k]])
         
      time_s = np.array(t_s1)
      peaks_all.append(peaks) 
      time_stamp_all.append(time_s)
    return p_th,time_stamp_all
    



def find_metrics(bite_gt,time_s):
    
   FP = 0 
   flag = np.zeros(len(bite_gt))
   flag2 = np.zeros(len(time_s))
   TP = 0 
   
  
   for i in range(len(time_s)):
    
       
      
       for j in range(len(bite_gt)):
           if time_s[i]>bite_gt[j][0] and time_s[i]<bite_gt[j][1] and flag[j] == 0 and flag2[i] == 0  :
                 TP = TP + 1
                 flag[j] = 1 
                 flag2[i] = 1 
                 
             
                   
       
   FP = len(time_s)-TP   


   FN = sum(flag == 0 )
   return TP,FP,FN
  


  