function D_M = process_data( )
load('drug_micro_MASI.mat')

D_M.drug_sim = drug_micro_MASI.drug_chemsim;
D_M.micro_sim = {drug_micro_MASI.micro_funsim,drug_micro_MASI.micro_dissim};
D_M.drug_sim = {drug_micro_MASI.drug_chemsim};
%%%计算交互矩阵
D_M.interaction = drug_micro_MASI.inter_matrix;  

%%%%删除全为0的行和全为0的列
indd = find(sum(D_M.interaction,2)==0);
indm = find(sum(D_M.interaction)==0);



end