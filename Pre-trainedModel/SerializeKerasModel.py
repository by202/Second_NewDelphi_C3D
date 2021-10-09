#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 23:01:06 2021

@author: bingyuliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:42:47 2021

@author: bingyuliu
"""

import argparse
import numpy as np
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sport1m_model import create_model_functional
import numpy as np
import cv2

#def serialize_weights(model):
    
    
    
"""Serialize Keras model into flattened numpy array in correct shape for
Pytorch in Rust"""
# All the weights need to be flattened into a single array for rust interopt

model = create_model_functional()

network_weights = np.array([])

for i, layer in enumerate(model.layers):
    if "conv" in layer.name:
        A, b = layer.get_weights()
        # Keras stores the filter as the first two dimensions and the
        # channels as the 3rd and 4th. PyTorch does the opposite so flip
        # everything around
        _, _, _, inp_c, out_c = A.shape
        py_tensor = [[A[:,:, :, i, o] for i in range(inp_c)] for o in range(out_c)]
        A = np.array(py_tensor)
    elif "dense" in layer.name:
        A, b = layer.get_weights()
        A = A.T
        # Get the shape of last layer output to transform the FC
        # weights correctly since we don't flatten input to FC in Delphi
        inp_chans = 1
        for prev_i in range(i, 0, -1):
            layer_name = model.layers[prev_i].name
            if ("global" in layer_name):
                inp_chans = model.layers[prev_i].output_shape[2]
                break
            if ("conv" in layer_name) or ("max_pooling3d" in layer_name) or prev_i == 0:
                inp_chans = model.layers[prev_i].output_shape[4]
                break
        # Remap to PyTorch shape
        fc_h, fc_w = A.shape
        channel_cols = [np.hstack([A[:, [i]] for i in range(chan, fc_w, inp_chans)])
                        for chan in range(inp_chans)]
        A = np.hstack(channel_cols)
    else:
        continue
    layer_weights = np.concatenate((A.flatten(), b.flatten()))
    network_weights = np.concatenate((network_weights, layer_weights))

np.save(os.path.join(f"model_newnew.npy"), network_weights.astype(np.float64))


    
     
    
    

    # with open ('keras_labels.txt','r') as f:
    #     class_names = f.readlines()
    #     f.close()
        
# =============================================================================
#     Initi model    
# sports1M_weights.h5/ C3D_Sport1M_weights_keras_2.2.4.h5/c3d-pretrained.pth
# =============================================================================

   
    # try:
    #     c3d_trainedmodel =model.load_weights('/Users/bingyuliu/Desktop/c3d-pretrained.pth')
    # except OSError as err:
    #     print('Check path to the model weights\' file!\n\n', err)
        
    # c3d_trainedmodel.summary()    
    
    #c3d_trainedmodel =  np.load('/Users/bingyuliu/Desktop/c3d-sports1M_weights.h5',allow_pickle=True)
    
   # ws = serialize_weights('/Users/bingyuliu/Desktop/C3D_Sport1M_weights_keras_2.2.4.h5')
    #print ('\nShape:', ws.shape)
    
   