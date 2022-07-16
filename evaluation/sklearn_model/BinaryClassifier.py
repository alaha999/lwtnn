#Import the necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import np_utils

import os
import sys
import warnings
warnings.filterwarnings('ignore')

#define model name
modelname = "sklearn_QCDvsVLL"

col_names=["Event","lepflav","ptlep0","ptjet0","ptjet1","mtlep0","mtjet0","mtjet1","dijetmass",
           "drjet01","MET","HT","dphimetjet0","dphimetjet1","dphimetlep0","njet","nbjet","dijetPT",
           "dijetMT","dphijet0lep0","dphijet1lep0","dphidijetlep0","dphimetdijet","ST"]

cols = np.arange(0,len(col_names),1)

##Load Input data
InputPath = "../InputFiles/"
#Signal
file_vllm100 = InputPath + "VLLM100_Mar7_v0.txt"
#QCD
file_QCD     = InputPath +  "QCD_Mar7_v0.txt"

#DataFrame
vllm100=pd.read_csv(file_vllm100,sep=' ',index_col=None, usecols=cols,names=col_names)
vllm100['label']=1
vllm100['sample']=0

QCD=pd.read_csv(file_QCD,sep=' ',index_col=None, usecols=cols,names=col_names)
QCD['label']=0
QCD['sample']=4

#print(QCD.head(5))
### Some Functions
# Here is a function that returns a dataframe with only selected columns                                                                
def select_columns(data_frame, column_names):                                                                                           
    new_frame = data_frame.loc[:, column_names]                                                                                         
    return new_frame 
#Define a function to plot that we will use later
def plot_prob(tdf,vdf,var):
    
    #bkt = plt.hist(tdf[tdf['truth']==2][tvar],bins=np.arange(0,1.02,0.02), log=False)
    bkv = plt.hist(vdf[vdf['truth']==0][var], bins=np.arange(0,1.02,0.02), log=True)
    #sgt = plt.hist(tdf[tdf['truth']==0][tvar],bins=np.arange(0,1.02,0.02), log=False)
    sgv = plt.hist(vdf[vdf['truth']>0][var], bins=np.arange(0,1.02,0.02), log=True)

    bkverr = np.sqrt(bkv[0])
    sgverr = np.sqrt(sgv[0])

    plt.figure(figsize=(8,6))
    plt.errorbar(bkv[1][1:]-0.01, bkv[0],yerr=bkverr, fmt='r.', color="xkcd:burnt orange", label="Background Test",markersize='10')
    plt.errorbar(sgv[1][1:]-0.01, sgv[0],yerr=sgverr,fmt='b.', label="Signal Test",markersize='10')
    plt.hist(tdf[tdf['truth']==0][var],bins=np.arange(0,1.02,0.02), histtype='step', label="Background Train", linewidth=3, color='xkcd:peach',log=True)
    plt.hist(tdf[tdf['truth']>0][var],bins=np.arange(0,1.02,0.02), histtype='step', label="Signal Train", linewidth=3, color='skyblue',log=True)
    
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'NN Output',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    #plt.suptitle('Result 1: nnscore',fontsize=18)
    plt.savefig('NNoutput.png',facecolor='white')
    plt.show()


## Select training Variables
selcols=["mtlep0","mtjet0","mtjet1","dijetmass","drjet01","MET","HT","dphimetjet0",
         "dphimetjet1","dphimetlep0","njet","nbjet","dijetPT","dijetMT",
         "dphijet0lep0","dphijet1lep0","dphidijetlep0","dphimetdijet","ST",'label']

sigdf = select_columns(vllm100,selcols)
bkgdf = select_columns(QCD,selcols)

#print(sigdf.head(5))    


## Prepare for training
data=pd.concat([sigdf,bkgdf])
X, y = data.values[:, :-1], data.values[:, -1]
#Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(n_features)

## Scaling data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler=StandardScaler()
scaler=MinMaxScaler((0,1))
scaler.fit(X_train)                  #scale training data and the used the same scale on test data
#transform
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

##=======================================================
## Save the scaling in a text file for c++ implementation
##=======================================================
minvalue = X_train.min(axis=0)
rangevar = X_train.max(axis=0)-X_train.min(axis=0)
file = open("scaler_lwtnn.txt", "w+")
for i in range(0,rangevar.shape[0],1):
    #print(i)
    saveit = str(i) + " "+ str(minvalue[i]) + " "+ str(rangevar[i])+"\n"
    #print(saveit)
    file.write(str(saveit))
file.close()    
##=======================================================
#define callbacks. Here the model with highest accuracy will be saved at end...
cb = [ModelCheckpoint(filepath='best_model_'+modelname+'.h5',monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]

#Model Architecture
model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

##Training
#do the training
history = model.fit(X_train_scaled,y_train,epochs=32,batch_size=256,validation_data=(X_test_scaled, y_test), verbose=0, callbacks=cb)
#print the information, and save the final model
model.summary()

##=======================================================================
## Save the Model Arch in json and weights in .h5 for c++ implementation
##=======================================================================
arch = model.to_json()
with open('architecture.json','w') as arch_file:
    arch_file.write(arch)
    
model.save_weights('weights.h5')    
##=======================================================================
model.save(modelname+".h5")


#Now the model is trained. The rest is plotting and evaluating.
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

#Setup dataframes for the probability
t_df = pd.DataFrame()
v_df = pd.DataFrame()
t_df['truth'] = y_train
t_df['prob'] = 0
v_df['truth'] = y_test
v_df['prob'] = 0

print(y_train)
v_prob = model.predict(X_test_scaled);
t_prob = model.predict(X_train_scaled);
v_df['prob'] = v_prob
t_df['prob'] = t_prob

plot_prob(t_df,v_df,'prob')

fpr, tpr, _ = roc_curve(y_test,v_prob)
auc_score = auc(fpr,tpr)
fpr1, tpr1, _ = roc_curve(y_train,t_prob)
auc_score1 = auc(fpr1,tpr1)

##============================================
#                  ROC CURVE
#=============================================
#ROC CURVE
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color='xkcd:denim blue', label='ROC (AUC = %0.4f)' % auc_score)
plt.plot(fpr1,tpr1,color='xkcd:sky blue', label='Train ROC (AUC = %0.4f)' % auc_score1)
plt.legend(loc='lower right',fontsize=18)
plt.title(f'ROC Curve',fontsize=20)
plt.xlabel('Background Efficiency',fontsize=20)
plt.ylabel('Signal Efficiency',fontsize=20)
plt.xlim(0.,1.)
plt.ylim(0.,1.)
#plt.suptitle('Result 2: ROC Curve',fontsize=18)
plt.savefig(modelname+'_ROC_Curve.png',facecolor='white')
#plt.show()




##=================================================================
##                                                                #
##                    EVALUATION                                  #
##                                                                #
##=================================================================
# I'm just being lazy. To compare Sklearn Python Based Performance
# with lwtnn performance, I chose same VLL and QCD events to evaluate
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

X_eval,y_eval = data.values[:,:-1],data.values[:,-1]

X_eval_scaled = scaler.transform(X_eval)

nnevalscore = model.predict(X_eval_scaled)

outdf = pd.DataFrame(columns=["label","score"])
outdf["label"]=y_eval
outdf["score"]=nnevalscore

outdf.to_csv('../sklearn_evaluation.txt',sep=" ",index=False)
#file = open("sklearn_evaluation.txt","w+")
