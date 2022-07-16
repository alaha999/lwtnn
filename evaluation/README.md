Evaluating Sklearn Sequential Model using lwtnn
================================================
Please follow the lwtnn repo to setup the framework (improve later)

To test the lwtnn framework, I chose a sequential model which is a binary classifier. This folder has the following components, 
### sklearn_model
- Binary classifier(Sequential model) 
- Signal: VLL Signal
- Background: QCD
- Model Training and Testing file: **BinaryClassifier.py**
- This should produce,
  > architecture.json : needed for lwtnn conversion
  
  > weights.h5        : needed for lwtnn conversion

  > scaler_lwtnn.txt  : needed later to normalize inputs while reading in c++ code
- trained .h5 file, neural network score distribution and ROC curve  

### InputFiles
You can find the VLL and QCD files and the same scaler_lwtnn.txt file

### ModelFiles
- architecture.json (copy from sklearn_model folder)
- weights.h5 (copy from sklearn_model folder)
- **inputs.json** : prototype of input variables (Be Careful while giving name, you need to call exactly this variable name in c++ code)

Now use **keras2json.py** converter provided in lwtnn/converters/ to produce c++ readable format ***neural_net.json***

## Reading the converted nnmodel file in c++ using lwtnn

- use **evaluate_lwtnn.C** to read the model and evaluate the new events using the trained model
- Use the following command,
  > root -l
 
  > .x evaluate_lwtnn.C("c++readblennmodelfile","SigORbkgfile","scalingfile","outputfile")
- outputfile is consist of score of the process that you create running over signal or background file
- scalingfile: scaler_lwtnn.txt
- as an example,
  > .x evaluate_lwtnn.C("neural_net.json","QCD_Mar7_v0.txt","scaler_lwtnn.txt","bkgscore_lwtnn.txt")
  > 
  >PROTIP: don't forget to load the library (just check!) or use gSystem->Load(../build/lib/liblwtnn.so) in root prompt or include it in
  root macro
  
**REMEMBER: This is just to demonstrate the performance between pure pythonic sklearn evaluation Vs lwtnn c++ evaluation**
**REMEMBER: We will add this into our TTreeReader code, so we will be able to evaluate the event using this framework on FLY**

  


## Performance
Here I compare the performance between evaluation of VLL(signal) and QCD(background) events using sklearn .h5 file in pythonic environment and
using lwtnn converted neural_net.json file in C++ framework.

![Sklearn vs lwtnn](https://github.com/alaha999/lwtnn/blob/master/evaluation/lwtnnVsSklearn_performance_comparison.png)


