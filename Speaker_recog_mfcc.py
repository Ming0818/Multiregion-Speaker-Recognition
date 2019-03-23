from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
import os
import math

i=-1
ds=np.empty((0,14))
for foldername in os.listdir('/Users/Archit/Documents/Speak_Recog/DR3'):
    i=i+1
    for filename in os.listdir('/Users/Archit/Documents/Speak_Recog/DR3/'+foldername):
    
        (rate, sig)= wav.read('/Users/Archit/Documents/Speak_Recog/DR3/'+foldername+'/'+filename)
        mfcc_feat = mfcc(sig, rate)
    
        avg=np.mean(mfcc_feat, axis=0)
        np.reshape(avg, (1,13))
        avg=np.hstack((avg,i))
        print avg
        ds=np.append(ds, [avg], axis=0)
        print ds.shape
np.savetxt("speak_reco_DR3.csv", ds, delimiter=",")


