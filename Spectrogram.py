import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wav
import numpy as np
import os
import math

i=9
for foldername in os.listdir('/Users/Archit/Documents/Speak_Recog/DR3'):
    i=i+1
    j=0
    for filename in os.listdir('/Users/Archit/Documents/Speak_Recog/DR3/'+foldername):
        j=j+1
        (rate, sig)= wav.read('/Users/Archit/Documents/Speak_Recog/DR3/'+foldername+'/'+filename)
        frequencies, times, spectrogram = signal.spectrogram(sig, rate)
    
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        #plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.axis('off')
        
        plt.show()
 #       plt.savefig('reg3_withoutwhite/spec_reg_3_speaker_'+str(i)+'_sentence_'+str(j)+'.png', transparent=True, bbox_inches='tight')
