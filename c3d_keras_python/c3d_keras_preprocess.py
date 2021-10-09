#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 12:03:42 2021

@author: bingyuliu
"""


from sport1m_model import create_model_functional
import numpy as np
import cv2


def video_preproc1_old():
    video = '/Users/bingyuliu/Desktop/1.avi'
    cap = cv2.VideoCapture(video)
    
    clip = []
    while True:
        ret, frame = cap.read()
        if ret:
            ## convert image to RGB color for matplotlib
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            #if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            #inputs = np.expand_dims(inputs, axis=0)
            inputs[..., 0] -= 99.9
            inputs[..., 1] -= 92.1
            inputs[..., 2] -= 82.6
            inputs[..., 0] /= 65.8
            inputs[..., 1] /= 62.3
            inputs[..., 2] /= 60.3
            # center crop  (, l, h, w, c)
            #inputs = inputs[:,:,8:120,30:142,:]
            inputs = inputs[:,8:120,30:142,:]
                #inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
    cap.release()
    cv2.destroyAllWindows()   
        
    #clip.pop(0)
    _, _, _, chans = inputs.shape
    rust_tensor = np.array([[inputs[:, :, :, c] for c in range(chans)]])
    
    #np.save(path.join(image_path, f"class.npy"), np.array(classification).flatten().astype(np.int64))
    #cap.release()
     
    #inputs = np.transpose(inputs, (1, 2, 0, 3, 4))  
    #print ('input', inputs)


    # ######generate new image.npy #####
    # _, _, chans = inputs.shape
    # rust_tensor = np.array([[inputs[:, :, c] for c in range(chans)]])
    # np.save(os.path.join(f"image_v3.npy"), rust_tensor.flatten().astype(np.float64))
    
    
    def video_preproc2_new():
        cap = cv2.VideoCapture('/Users/bingyuliu/Desktop/12.avi')
        vid = []
        while True:
           ret, img = cap.read()
           if not ret:
              break
           vid.append(cv2.resize(img, (171, 128)))
        vid = np.array(vid, dtype=np.float32)
        vid = vid.astype(np.float32)
        vid[..., 0] -= 99.9
        vid[..., 1] -= 92.1
        vid[..., 2] -= 82.6
        vid[..., 0] /= 65.8
        vid[..., 1] /= 62.3
        vid[..., 2] /= 60.3
        # center crop  (, l, h, w, c)
        #inputs = inputs[:,:,8:120,30:142,:]
        vids = vid[:,8:120,30:142,:]
        _, _, _, chans = vids.shape
        rust_tensor = np.array([[vids[:, :, :, c] for c in range(chans)]])
        np.save(os.path.join(f"image_test.npy"), rust_tensor.flatten().astype(np.float64))
        
        
        
    def video_preproc3_new():
        cap = cv2.VideoCapture('/Users/bingyuliu/Desktop/1.avi')
        clip = []
        retaining = True
         # with open("result.txt", "w") as outfile:
    #     outfile.write(str('Result'+'\n'))
        while retaining:
           retaining, frame = cap.read()
           if not retaining and frame is None:
               continue
           clip.append(cv2.resize(frame, (171, 128)))
        clip = np.array(clip, dtype=np.float32)
        clip = clip.astype(np.float32)
        if len(clip) == 16:
            clip[..., 0] -= 99.9
            clip[..., 1] -= 92.1
            clip[..., 2] -= 82.6
            clip[..., 0] /= 65.8
            clip[..., 1] /= 62.3
            clip[..., 2] /= 60.3
            # center crop  (, l, h, w, c)
            #inputs = inputs[:,:,8:120,30:142,:]
            clip = clip[:,8:120,30:142,:]
            
        cap.release()
        cv2.destroyAllWindows()   
        
        _, _, _, chans = clip.shape
        rust_tensor2 = np.array([[clip[:, :, :, c] for c in range(chans)]])
        np.save(f"image_newnewnew.npy", rust_tensor2.flatten().astype(np.float64))
        
        
    def classification_preproc():
        classification = [367]
        np.save(f"class_new.npy", np.array(classification).flatten().astype(np.int64))
       
        
       
        
    if __name__ == "__main__":
        
        video_preproc3_new()
        classification_preproc()
    
