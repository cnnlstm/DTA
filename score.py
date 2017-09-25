#coding=utf-8
from __future__ import division 
import numpy as np 
import time

npzfile = ["_epoch_fc_score.npz","_epoch_conv_score.npz","_epoch_numscore.npz","_epoch_mulscore.npz"]

for l in range(100):
    print l
    
    for file in npzfile:
        file_path = "/your_dir/ucf11/score/"+str(l)+file
        #print file.split("_")[2].split(".")[0]
        try:
            file_info = np.load(file_path)["test_info"]
            j = 0
            for i in np.arange(len(file_info)):
                score = file_info[i][0]
                label = file_info[i][1]  
                pred_label = np.argmax(score)
                if pred_label == label:
                    j = j+1
                k = i+1
            print file.split("_")[2].split(".")[0],"_Accuracy:",j/k
        except EOFError:
            print 'ERROR INPUT !'
