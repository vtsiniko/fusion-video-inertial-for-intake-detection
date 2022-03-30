# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:55:27 2021

@author: tsini_1teem35
"""


import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras import layers
import tensorflow as tf 
from keras.layers.convolutional import Conv3D
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D,Input,Concatenate,MaxPooling3D, Dropout

import random



####### creaate_dataset_fusion ###########

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

    




def label_last_frame(time,bites,e):
    print ("proc: label")  
    lb = []
   
   
    for i in range(len(time)):
         print(i)
        # inside for 21 subjects
         l = len(time[i])
         
         a = np.empty(l)
         
         for n in range(l):
             
             # inside for all the length (windows)
             a[n] = 0
             
             for k in range(len(bites[i])):
                 # inside for all bites 
                 tmean = (bites[i][k,0]+bites[i][k,1])/2
                 #tmean = bites[i][k,1]-e
                 if (time[i][n] < bites[i][k,1]+e)  and  (time[i][n] > tmean):
                      #print(n)
                      a[n] = 1
                      #print(a[n])
                      break
                 
         lb.append(a)
    print("label done")     
    return lb

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
    
###### create_dataset_fusion #############


###### LOSO_fusion ###########
def split(lb,data_all,i,sub):
    logo = LeaveOneGroupOut()
    d = np.asarray(data_all)
    l = np.asarray(lb)
    sum = 0 
    for train_index, test_index in logo.split(data_all, lb, sub):
       sum = sum + 1
       x_train = d[train_index]
       y_train = l[train_index]
       
       if sum == i :
           break 
    print("start")  
    print(train_index)
    print(test_index)
    x_train = np.concatenate(x_train)    
    y_train = np.concatenate(y_train)
      
    return  x_train, y_train 

###### LOSO_fusion ###########


###### train_model_fusion ###########



def ballance_fusion_data(X_inertial,X_video,Yd,shuffle):
    
   X1_inertial = []
   X1_video = []
   X0_video = []
   X0_inertial = []
   for i in range(X_inertial.shape[0]):
       if Yd[i] == 1 :
          X1_inertial.append(X_inertial[i,:,:])
          X1_video.append(X_video[i,:,:,:])
       elif Yd[i] == 0:
          X0_inertial.append(X_inertial[i,:,:])
          X0_video.append(X_video[i,:,:,:])
   print('0:'+str(len(X0_inertial)))
   print('1:'+str(len(X1_video)))
   dist = len(X0_inertial) - len(X1_video)
   if dist < 0:
      
       dist = dist*(-1)
       tem = 0
   elif dist > 0:
       tem = 1
  
   for i in range (dist):
       if tem == 0:
          
          a = int ( random.uniform(0, len(X1_inertial)-1))
          X1_inertial.pop(a)
          X1_video.pop(a)
       elif tem == 1:
           
          a = int ( random.uniform(0, len(X0_inertial)-1))
          X0_inertial.pop(a)
          X0_video.pop(a)
  # if shuffle == True:        
   #   random.shuffle(X0)
   #   random.shuffle(X1)    
   Y0 = []
   Y1 = []  
   for i in range(len(X0_inertial)):
       Y0.append(0)
       Y1.append(1) 
   
       
   return X0_inertial,X1_inertial,X0_video,X1_video,Y0,Y1   

class DataGenerator_fusion2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_in0,data_in1,data_vid0,data_vid1,label0,label1,batch_size=4, dim_in=(500,6),dim_vid=(25,128,128),shuffle=False):
        'Initialization'
       
        self.dim_in = dim_in
        self.dim_vid = dim_vid
    
        self.batch_size = batch_size
        self.data_in0 = data_in0
        self.data_in1 = data_in1
        self.data_vid0 = data_vid0
        self.data_vid1 = data_vid1
       
        self.label0 = label0
        self.label1 = label1 
        self.shuffle = shuffle
        self.on_epoch_end()
        
      

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int( (len(self.data_in0)+len(self.data_in1))/self.batch_size)
    
   
            
            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        x_inertial_train,x_video_train,y_train = self.__data_generation(index)      
        shuffler = np.random.permutation(len(x_inertial_train))
        x_inertial_train = x_inertial_train[shuffler]
        x_video_train = x_video_train[shuffler]
        x_video_train = x_video_train[:,:,:,:, np.newaxis] 

        y_train = y_train[shuffler]
        x_train = [x_inertial_train ,x_video_train/255]
        return x_train,y_train
        #if self.flag == 'inertial':
            #return x_inertial_train,y_train
        #else:
            #return x_video_train/255,y_train
       
       
       
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.videos))
        
    def __data_generation(self,index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        

        # Generate data
        n = index*1 
        tempx0_in = []
        tempx1_in = []
        tempx0_vid = []
        tempx1_vid = []
        tempy0 = []
        tempy1 = []
        x0_in = []
        x1_in = []
        x0_vid = []
        x1_vid = []
        for j in range(self.batch_size//2):
            if ((index+j)+n) > len(self.data_in0):
                break 
            tempx0_in.append(self.data_in0[(index+j)+n])
            tempx1_in.append(self.data_in1[(index+j)+n])
            tempx0_vid.append(self.data_vid0[(index+j)+n])
            tempx1_vid.append(self.data_vid1[(index+j)+n])
            tempy0.append(self.label0[(index+j)+n])
            tempy1.append(self.label1[(index+j)+n])
        x0_in = np.array(tempx0_in)
        x1_in = np.array(tempx1_in)
        x0_vid = np.array(tempx0_vid)
        x1_vid = np.array(tempx1_vid)
        y0 = np.array(tempy0)
        y1 = np.array(tempy1)
        x_in = np.concatenate((x0_in,x1_in))
        x_vid = np.concatenate((x0_vid,x1_vid))
        y = np.concatenate((y0,y1))
        
        
        return x_in,x_vid,y   
            # Store class
                  


class DataGenerator_fusion(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_in,data_vid,label,batch_size=16, dim_in=(500,6),dim_vid=(25,128,128),shuffle=False):
        'Initialization'
       
        self.dim_in = dim_in
        self.dim_vid = dim_vid
    
        self.batch_size = batch_size
        self.data_in = data_in
        self.data_vid = data_vid

       
        self.label = label

        self.shuffle = shuffle
        self.on_epoch_end()
        
      

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data_in)/self.batch_size)
    
   
            
            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        x_inertial_train,x_video_train,y_train = self.__data_generation(index)      
        shuffler = np.random.permutation(len(x_inertial_train))
        x_inertial_train = x_inertial_train[shuffler]
        x_video_train = x_video_train[shuffler]
        x_video_train = x_video_train[:,:,:,:, np.newaxis] 

        y_train = y_train[shuffler]
        x_train = [x_inertial_train ,x_video_train/255]
        return x_train,y_train
        #if self.flag == 'inertial':
            #return x_inertial_train,y_train
        #else:
            #return x_video_train/255,y_train
       
       
       
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.videos))
        
    def __data_generation(self,index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        

        # Generate data
        n = index*(self.batch_size-1) 
        tempx_in = []
        tempx_vid = []
        tempy = []
        x_in = []
        for j in range(self.batch_size):
            if ((index+j)+n) > len(self.data_in):
                break 
            tempx_in.append(self.data_in[(index+j)+n])
            tempx_vid.append(self.data_vid[(index+j)+n])
            tempy.append(self.label[(index+j)+n])
        x_in = np.array(tempx_in)
        x_vid = np.array(tempx_vid)
        y = np.array(tempy)
        #x_in = np.concatenate((x_in))
        #x_vid = np.concatenate((x_vid))
        #y = np.concatenate((y))
        
        
        return x_in,x_vid,y   
            # Store class
                  








def create_model_fusion(time_window): 
   
    
   # INERTIAL MODEL
   wl1 = time_window*100
   wl2 = time_window*5
   in_1D = Input((wl1, 6))
   # conv1D_1
   model_1D = Conv1D(32,input_shape=(wl1, 6), kernel_size= 5 ,padding = 'same', strides=1, activation='relu')(in_1D)
   #model_1D = BatchNormalization()(model_1D)
   model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # conv1D_1
   model_1D = Conv1D(64, input_shape=(round(wl1/2), 6),kernel_size= 3 ,padding = 'same', strides=1 , activation='relu')(model_1D)
   #model_1D = BatchNormalization()(model_1D)
   model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # conv1D_1
   model_1D = Conv1D(128, input_shape=(round(wl1/4), 6), kernel_size= 3,padding = 'same', strides=1 , activation='relu')(model_1D)
   #model_1D = BatchNormalization()(model_1D)
   #model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # LSTM_1 
   model_1D = LSTM(128, dropout=0.2,recurrent_dropout=0.5, activation ='hard_sigmoid', return_sequences=False )(model_1D)
   model_1D = Dropout(0.5)(model_1D) 
   model_1D = Flatten()(model_1D)

   # VIDEO
   
   in_3D = Input((15, 128, 128, 1))
   # conv3D_2
   model_3D = Conv3D(32,input_shape=(wl2,128,128,1), kernel_size=(3, 3,3) ,padding = 'same', strides = (1,1,1), activation='relu')(in_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'same')(model_3D)
    # conv3D_2
   model_3D = Conv3D(32,input_shape=(round(wl2/2),64,64,32), kernel_size=(3, 3,3) ,padding = 'same', strides= (1,1,1), activation='relu')(model_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'valid')(model_3D)
    # conv3D_2
   model_3D = Conv3D(64,input_shape=(round(wl2/2),32,32,32), kernel_size=(3, 3,3) ,padding = 'same', strides = (1,1,1), activation='relu')(model_3D)
   model_3D = MaxPooling3D(pool_size= (2,2,2), strides=(2,2,2) , padding = 'valid')(model_3D)
    # conv3D_2
   model_3D = Conv3D(64,input_shape=(round(wl2/2),16,16,64), kernel_size=(3, 3,3) ,padding = 'same', strides = (1,1,1), activation='relu')(model_3D)
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



def create_model_fusion2(time_window): 
   
    
   # INERTIAL MODEL
   wl1 = time_window*100
   wl2 = time_window*5
   in_1D = Input((wl1, 6))
   # conv1D_1
   model_1D = Conv1D(32,input_shape=(wl1, 6), kernel_size= 5 ,padding = 'same', strides=1, activation='relu')(in_1D)
   #model_1D = BatchNormalization()(model_1D)
   model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # conv1D_1
   model_1D = Conv1D(64, input_shape=(round(wl1/2), 6),kernel_size= 3 ,padding = 'same', strides=1 , activation='relu')(model_1D)
   #model_1D = BatchNormalization()(model_1D)
   model_1D = MaxPooling1D(pool_size= 2, strides=2, padding = 'same')(model_1D)
   # conv1D_1
   model_1D = Conv1D(128, input_shape=(round(wl1/4), 6), kernel_size= 3,padding = 'same', strides=1 , activation='relu')(model_1D)
   #model_1D = BatchNormalization()(model_1D)
  # LSTM_1 
   model_1D = LSTM(128,  return_sequences=False )(model_1D)
   #dropout=0.2,recurrent_dropout=0.5, activation ='hard_sigmoid',
   model_1D = Dropout(0.5)(model_1D) 
   model_1D = Flatten()(model_1D)

   # VIDEO
   
   in_2D = Input(( wl2, 128, 128, 1))
   # conv3D_2
   model_2D = layers.TimeDistributed(layers.Conv2D(32, (3,3), padding='same', strides=(1,1), activation='relu'),input_shape = ( wl2, 128, 128, 1))(in_2D)
   #model_2D = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")(model_2D)
   model_2D = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))(model_2D)

   # conv3D_2
   model_2D = layers.TimeDistributed(layers.Conv2D(32, (3,3), padding='same', strides=(1,1), activation='relu'),input_shape = (wl2, 64, 64, 32))(model_2D)
  # model_2D = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")(model_2D)
   model_2D = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))(model_2D)

   # conv3D_2
   model_2D = layers.TimeDistributed(layers.Conv2D(64, (3,3), padding='same', strides=(1,1), activation='relu'),input_shape = (wl2, 32, 32, 32))(model_2D)
 #  model_2D = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")(model_2D)
   model_2D = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))(model_2D)

    # conv3D_2
   model_2D = layers.TimeDistributed(layers.Conv2D(64, (3,3), padding='same', strides=(1,1), activation='relu'),input_shape = (wl2, 16, 16, 64))(model_2D)
  # model_2D = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")(model_2D)
   model_2D = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))(model_2D)

    # conv3D_2
   model_2D = layers.TimeDistributed(layers.Flatten())(model_2D)
   model_2D = layers.LSTM(128,  return_sequences=False )(model_2D)
   #dropout=0.2,recurrent_dropout=0.5, activation ='hard_sigmoid',
   model_2D = layers.Dropout(0.5)(model_2D)
   model_2D = Flatten()(model_2D)
 
    

   # FUSION 

   conc = Concatenate()([model_1D, model_2D])
   fusion_model = Dense(128, activation='relu')(conc) 
   fusion_model = Dropout(0.5)(fusion_model) 
   fusion_model = Dense(32, activation='relu')(fusion_model) 
   fusion_model = Dropout(0.5)(fusion_model) 
   fusion_model = Dense(1, activation='sigmoid')(fusion_model) 
   model = Model(inputs=[in_1D, in_2D], outputs=[fusion_model])


   return model 

def plt_history(history,l):
   fig1, ax1 = plt.subplots()
   ax1.plot(history.history['accuracy'])
   if len(history.history)>2:
       ax1.plot(history.history['val_accuracy'])
   ax1.set_title('model accuracy')
   ax1.set_ylabel('accuracy')
   ax1.set_xlabel('epoch')
   ax1.legend(['train', 'test'], loc='upper left')
   plt.savefig(l+'/acc.png')
   
# summarize history for loss
   fig2, ax2 = plt.subplots()
   ax2.plot(history.history['loss'])
   if len(history.history)>2:
       ax2.plot(history.history['val_loss'])
   ax2.set_title('model loss')
   ax2.set_ylabel('loss')
   ax2.set_xlabel('epoch')
   ax2.legend(['train', 'test'], loc='upper left')
   plt.savefig(l+'/loss.png')   
      
###### train_model_fusion ###########
   

