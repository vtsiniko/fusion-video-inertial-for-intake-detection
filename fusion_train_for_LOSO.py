# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 19:19:25 2021

@author: tsini_1teem35
"""


import pickle as pkl

from functions_fusion import split,ballance_fusion_data 
import sys
import os
import numpy as np
from keras.optimizers import Adam,schedules
from functions_fusion import create_model_fusion, create_model_fusion2, plt_history,DataGenerator_fusion
from keras.callbacks import ModelCheckpoint



##### LOSO ####

def train_model(time_window,model_fusion,num_epochs, architecture):

   current_path = os.getcwd()
   path = current_path + '\model_' + str(model_fusion) + '_wl_' + str(time_window) + 'sec' 

   data_fusion = pkl.load( open( path + "/data_fusion.p", "rb" ) )
   inertial_window = data_fusion[ "inertial_window"]
   video_window = data_fusion['video_window']
   label = data_fusion['label']
   subject = data_fusion['subject']



   batch_sizes = 16
   lr = 0.1 



   for i in range(1,2):
       sub = i   
       m = "\model"+str(sub)
       print(m) 
       directory = "\model" + str(i) + '_' + architecture + '_' + str(num_epochs) + 'epochs'
       path2 = path + directory
       os.mkdir(path2)
    
       x_train_inertial, y_train_inertial, = split(label,inertial_window,sub,subject)
       x_train_video, y_train_video, = split(label,video_window,sub,subject)
      
       '''
       print("save data")
       np.save(path2 + "\\x_train_inertial.npy",x_train_inertial)
       np.save(path2 + "\\x_train_video.npy",x_train_video)
       np.save(path2 + "\\y_train.npy",y_train_video)
       print('save data ....ok')   
       '''


       train = DataGenerator_fusion(x_train_inertial,x_train_video,y_train_inertial) 
    

       sdec = len(x_train_inertial)//32
       lr = schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=sdec,decay_rate=0.9)
      
       if architecture == '3D_CNN':
           model = create_model_fusion(time_window)
           print(model.summary())
       else:
          model = create_model_fusion2(time_window)
          print(model.summary())

       model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=lr), metrics = ['accuracy'])
       filepath = path2 + '\saved-model-' + architecture + '-epoch_' + str(num_epochs) + '{epoch:02d}.hdf5'
       checkpoint = ModelCheckpoint(filepath,save_freq="epoch")
       history = model.fit(train, batch_size = batch_sizes, verbose = 1,epochs = num_epochs, shuffle = False,callbacks=[checkpoint] )
       plt_history(history,(path+m))
       

if __name__ == '__main__':
    time_window = int(sys.argv[1])
    model_fusion = str(sys.argv[2])
    num_epochs = int(sys.argv[3])
    architecture = str(sys.argv[4])
    print('model_' + model_fusion + '_time_' + str(time_window))
    train_model(time_window,model_fusion,num_epochs, architecture)
           