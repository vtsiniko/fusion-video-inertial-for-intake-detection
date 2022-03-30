# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:55:25 2021

@author: tsini_1teem35
"""


import pickle as pkl
from functions_preprocess import or_data,mov_aver,fir_filter,scale_data,edge_removal

import os

current_path = os.getcwd()

# load the dataset pickle
with open(current_path + './FIC.pkl', 'rb') as fh:
   dataset = pkl.load(fh)
# Extract all information
raw_data_vec = dataset['signals_raw']
proc_data_vec = dataset['signals_proc']
subject_id_vec = dataset['subject_id']
session_id_vec = dataset['session_id']
bite_gt_vec = dataset['bite_gt']
mm_gt_vec = dataset['mm_gt']


# correct layout
proc, bite, subject, session  = or_data(proc_data_vec,bite_gt_vec)


#moving average filter  
l = 25
proc = mov_aver(proc, l)   
 
#fir filter  
cutoff = 1
num = 513 
data = fir_filter(cutoff, num , proc)   

   
#scale data
data_scale, std, mean = scale_data(data)

#edge remove 
data_scale_edge = edge_removal(data_scale,bite)


dataset = {"bite":bite,
"subject":subject,
"session":session,
"data_inertial":data_scale_edge}

print('save inertial data')
pkl.dump( dataset, open( "inertial_data.p", "wb" ) )
print('save inertial data ..... ok')


