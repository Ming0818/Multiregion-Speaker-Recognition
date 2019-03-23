from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib
import math, svmlight, ast
from sklearn import ensemble

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
std_scaler_x = preprocessing.StandardScaler().fit(X_train)
std_scaler_y = preprocessing.StandardScaler().fit(Y_train)
X_train_std = std_scaler_x.transform(X_train)
X_test_std = std_scaler_x.transform(X_test)
Y_train_std = std_scaler_y.transform(Y_train)
Y_test_std = std_scaler_y.transform(Y_test)

# Prediction Algorithms -


# SVR algorithm
SVM_clf = svm.SVR(kernel='linear')
SVM_clf.fit(X_train_std, Y_train_std)
SVM_accuracy = SVM_clf.score(X_test_std, Y_test_std)
SVM_accuracytrain = SVM_clf.score(X_train_std, Y_train_std)
y_SVM_test_prediction = SVM_clf.predict(X_test_std)
#print y_SVM_test_prediction
print ('SVM Accuracy = '+str(SVM_accuracy))
print ('SVM Train Accuracy = '+str(SVM_accuracytrain))

#RandomForestClf

RF_clf= ensemble.RandomForestClassifier(n_estimators=1000)
RF_clf.fit(X_train, Y_train)
RF_accuracy = RF_clf.score(X_test, Y_test)
RF_accuracytrain = RF_clf.score(X_train, Y_train)
y_RF_test_prediction = RF_clf.predict(X_test)
#print y_RF_test_prediction
print ('RandomForest Accuracy = '+str(RF_accuracy))
print ('Random Forest Training Accuracy =' +str(RF_accuracytrain))



'''# Linear Regression algorithm
LR_clf = LinearRegression()
LR_clf.fit(X_train, y_train)
LR_accuracy = LR_clf.score(X_test, y_test)
LR_accuracytrain = LR_clf.score(X_train, y_train)
y_LR_test_prediction = LR_clf.predict(X_test)'''

'''# SVMLight algorithm
def now(d, c):
	for i in range(len(d)):
		j = list(d[i])
		k = float(c[i])
		s = "(" + str(k) + ", [(1, " + str(j[0]) + "), (2, " + str(j[1]) + "), (3, " + str(j[2]) + \
			"), (4, " + str(j[3]) + "), (5, " + str(j[4]) + "), (6, " + str(j[5]) + "), (7, " + str(j[6]) + \
			"), (8, " + str(j[7]) + "), (9, " + str(j[8]) + ")])"
		s = ast.literal_eval(s)
		yield (s)
		
X_light_train = list(now(X_train_std, y_train_std))
X_light_test = list(now(X_test_std, y_test_std))

model = svmlight.learn(X_light_train, type='regression', kernel='rbf', rbf_gamma=.0005, C=890)
y_svmlight_test_prediction = np.array(svmlight.classify(model, X_light_test))
r4 = mean_squared_error(y_test_std, y_svmlight_test_prediction)
y_mean = np.full(len(y_test_std),y_test_std.mean())
r2 = mean_squared_error(y_test_std, y_mean)
svmlight_accuracy = 1-(r4/r2)

#  R-SQUARED
print "\n\n  R-SQUARED:"
print "SVM :",SVM_accuracy
print "SVM train:",SVM_accuracytrain
print "LR :",LR_accuracy
print "LR train:",LR_accuracytrain
print "svmlight :",svmlight_accuracy


# MEAN SQUARED ERROR
r1 = mean_squared_error(y_test_std, y_SVM_test_prediction)
r3 = mean_squared_error(y_test_std, y_LR_test_prediction)
print "\n\nMEAN SQUARED ERROR"
print "SVM :",r1
print "LR :",r3
print "svmlight :",r4

denormalized_y_prediction_SVM  = std_scaler_y.inverse_transform(y_SVM_test_prediction)
denormalized_y_prediction_svmlight = std_scaler_y.inverse_transform(y_svmlight_test_prediction)

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


