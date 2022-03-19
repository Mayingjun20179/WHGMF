# WHGMF
Generalized matrix factorization based on weighted hypergraph learning for microbe-drug relationship prediction
All files are placed in the MATLAB path, and the main function “mianZ.m” can be run to automatically calculate the results of the five-fold cross verification under the three scenarios. The following describes data files, code files, and run results.

Data files:
“drug_micro_MASI.mat” contains all data information for drugs and microbes.
“drug_micro_MASI.micro_dissim” represents the semantic similarity of microbes (S_sem^m).
drug_micro_MASI.micro_funsim represents the functional similarity of microbes (S_fun^m).
“drug_micro_MASI. micro_inf” stores the name and ID of microbes.
“drug_micro_MASI.drug_chemsim” represents the chemical structure similarity of drugs (S_stru^d).
“drug_micro_MASI.drug_inf” stores the name and ID of drugs.
“drug_micro_MASI.drug_micro_assocition” stores the association information of drugs and microbes.

Code file:
“Assist_method_utils.m” represents auxiliary codes, including KSNS, CKA-MKL, etc.
“main_cv.m ” represents the code of the five-fold crossover validation, including the generation of test and training sets, and the calculation of evaluation results.
“process_data.m” converts the data in drug_micro_MASI into matrix form.
“WHGMF_opt.m” represents the code that performs the calculation process of WHGMF.
“minz.m” represents the main function.
 
Result file:
“best_WHGMF_CVa.mat” represents the result in the class imbalance scenario. The first three columns of this matrix are parameters, the fourth column is the scale factor, and the fifth and sixth columns are the AUPR and AUC values, respectively.
“best_WHGMF_CVr.mat” and “best_WHGMF_CVc.mat” represent results for new drugs and new microorganisms, respectively. The first three positions of the resulting vector are parameters, and the 7th to 11th positions are the top 2%, 4%, 6%, 8%, and 10% hit rates, respectively.
