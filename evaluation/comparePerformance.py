import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sklearnfile =  "sklearn_evaluation.txt"
lwtnn_sigfile= "sigscore_lwtnn.txt"
lwtnn_bkgfile= "bkgscore_lwtnn.txt"

#lwtnn
siglwtnndf=pd.read_csv(lwtnn_sigfile,sep=' ',index_col=None, usecols=[0],names=["sigscore"])
bkglwtnndf=pd.read_csv(lwtnn_bkgfile,sep=' ',index_col=None, usecols=[0],names=["bkgscore"])

#sklearn
sklearndf= pd.read_csv(sklearnfile,sep=' ',index_col=None, usecols=[0,1],names=["label","score"])

#plot
plt.figure(figsize=(8,6))
plt.hist(sklearndf[sklearndf["label"]==1]["score"],bins=100,label="sklearn_signal",alpha=0.5,color="blue",log=True)
plt.hist(sklearndf[sklearndf["label"]==0]["score"],bins=100,label="sklearn_bkg",alpha=0.5,color="orange",log=True)
plt.hist(siglwtnndf["sigscore"],bins=100,label="lwtnn_signal",histtype="step",lw=2,color="blue",log=True)
plt.hist(bkglwtnndf["bkgscore"],bins=100,label="lwtnn_bkg",histtype="step",lw=2,color="orange",log=True)
plt.xlabel("NN Score",fontsize=16)
plt.ylabel("Events",fontsize=16)
plt.title("lwtnn vs sklearn",fontsize=18)
plt.legend(loc="upper center",fontsize=13)
plt.savefig("lwtnnVsSklearn_performance_comparison.png",dpi=250,facecolor="white")
