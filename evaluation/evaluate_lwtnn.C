// Include the headers needed for sequential model
#include "../include/lwtnn/NNLayerConfig.hh"
#include "../include/lwtnn/LightweightNeuralNetwork.hh"
// Then include the json parsing functions
#include "../include/lwtnn/parse_json.hh"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <cmath>

using namespace std;

//Reading Input variables from a file(optional only for demonstration)
map<string,double>getInputVar(vector<string>row,string scalingfile){

  //Input Normalization File
  ifstream finscaler("InputFiles/"+scalingfile);
  
  map<string,double>inputs;
  vector<string> scaler,minval,rangevar;
  string line,word;
  
  //Read Normalization
  while (getline(finscaler,line)){
    scaler.clear();
    stringstream s(line);
    while (getline(s, word, ' ')){
      //cout<<"SCALER="<< word << endl;
      scaler.push_back(word);      
    }
    
    minval.push_back(scaler[1]);
    rangevar.push_back(scaler[2]);
  }
  
  //cout<<"6th var is ="<<stof(row[5])<<endl;
  
  //Prepare the inputs
  //But be careful, don't forget to normalize the inputs. Use exactly same normalization as you used in traning data
  /*
    Inputs["variable_0"]= value
    Inputs["variable_1"]= value
    ...
    Inputs["variable_n"]= value
  */
  
  //minval,rangevar = getNormalization();
  //cout<<"minval="<<minval.size()<<endl;
  for(int ivar=0;ivar<19;ivar++){
    inputs["variable_"+std::to_string(ivar)]=((stof(row[ivar+5])-stof(minval[ivar]))/stof(rangevar[ivar]));
  }
  
  return inputs;
}

int evaluate_lwtnn(string nnmodelfile, string inpfile, string scalingfile,string outfilename){
  float output_value;
  lwt::JSONConfig network_file;
  // The map for the variables
  std::map<std::string,double> inputs;
  vector<string>row;string line,word;

  // The actual NN instance
  lwt::LightweightNeuralNetwork *nn_instance;
  // Read in the network file
  std::string in_file_name("ModelFiles/"+nnmodelfile);
  std::ifstream in_file(in_file_name);
  network_file = lwt::parse_json(in_file);
  // Create a new lwtn netowrk instance
  nn_instance = new lwt::LightweightNeuralNetwork(network_file.inputs, 
						  network_file.layers, network_file.outputs);
  
  //Loop Over All Events
  ifstream fin("InputFiles/"+inpfile);
  while (getline(fin,line)){
    row.clear();
    stringstream s(line);
    while (getline(s, word, ' ')){
      //cout<< word << endl;
      row.push_back(word);      
    }

    
    //GetInputs
  
    inputs = getInputVar(row,scalingfile);
    

    //Check Inputs
    for(const auto&inp: inputs){
      //cout<<inp.first<<" "<<inp.second<<endl;
    }
    // Calculate the output value of the NN based on the inputs given
    auto out_vals = nn_instance->compute(inputs);
    for (const auto& out: out_vals) {
      output_value = out.second;
    }
    //Print out the score
    std::cout<<"NN output = " << output_value << std::endl;
    
    fstream outfile;
    outfile.open(outfilename,ios::out | ios::app);
    outfile<<output_value<<endl;
  }
  return 0;
}

