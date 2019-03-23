from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib
import math, svmlight, ast

train_file = pd.read_csv('speak_reco_multireg.csv')
test_file = pd.read_csv('speak_reco_multireg_valid.csv')
features = ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12','feat_13']
test_file.dropna(inplace = True)
train_file.dropna(inplace = True)

X_train = train_file[features].values
Y_train = train_file['Speaker'].values

X_test = test_file[features].values
Y_test = test_file['Speaker'].values

#Standardising the values.
'''std_scaler_x = preprocessing.StandardScaler().fit(X_train)
std_scaler_y = preprocessing.StandardScaler().fit(y_train)
X_train_std = std_scaler_x.transform(X_train)
X_test_std = std_scaler_x.transform(X_test)
y_train_std = std_scaler_y.transform(y_train)
y_test_std = std_scaler_y.transform(y_test)'''

# Prediction

#KNN algorithm
KNN_clf= KNeighborsClassifier(n_neighbors=1)
KNN_clf.fit(X_train, Y_train)
Y_pred= KNN_clf.predict(X_test)
print Y_pred
KNN_accuracy= accuracy_score(Y_test, Y_pred)
print KNN_accuracy




Date = input_file['Date'].values
test_date = np.array(Date[456:])
datetimes = [datetime.strptime(t, "%Y-%m-%d") for t in test_date]
date = matplotlib.dates.date2num(datetimes)
hfmt = matplotlib.dates.DateFormatter('%d-%m-%Y')
fig = plt.figure()
fig.canvas.set_window_title('Moisture Prediction using Machine Learning') 
ax = fig.add_subplot(1,1,1)
ax.xaxis.set_major_formatter(hfmt)
plt.setp(ax.get_xticklabels(), rotation=15)
ax.plot(date, y_test, color='#af0505', label='Original Values')
ax.plot(date, y_LR_test_prediction, color='blue', label='LR prediction' )
ax.plot(date, denormalized_y_prediction_svmlight, color='#08ad13', label='SVMLight prediction')
ax.plot(date, denormalized_y_prediction_SVM, color='#af2edd', label='SVR prediction')
plt.xlabel('Time')
plt.ylabel('Moisture')
plt.legend()
plt.grid()
plt.show()'''


