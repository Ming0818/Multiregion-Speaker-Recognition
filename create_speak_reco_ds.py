import numpy as np
import scipy
from scipy import ndimage
import os

def dataset_train():
    lst=[]
    lst_label=[]
    for filename_train in os.listdir('/Users/Archit/Documents/reg1/'):
        image=np.array(ndimage.imread('/Users/Archit/Documents/reg1/'+filename_train, flatten=False))
        lst.append(image)
        ds_train=np.stack(lst, axis=0)
        if(filename_train[20:21] == '_'):
            lst_label.append(int(filename_train[19:20]))
            label_train=np.stack(lst_label, axis=0)
        else:
            lst_label.append(int(filename_train[19:21]))
            label_train=np.stack(lst_label, axis=0)
            
    return ds_train, label_train

def dataset_test():
    lst2=[]
    lst_label2=[]
    for filename_test in os.listdir('/Users/Archit/Documents/reg1_test/'):
        image=np.array(ndimage.imread('/Users/Archit/Documents/reg1_test/'+filename_test, flatten=False))
        lst2.append(image)
        ds_test=np.stack(lst2, axis=0)
        if(filename_test[20:21] == '_'):
            lst_label2.append(int(filename_test[19:20]))
            label_test=np.stack(lst_label2, axis=0)
        else:
            lst_label2.append(int(filename_test[19:21]))
            label_test=np.stack(lst_label2, axis=0)

    return ds_test, label_test
