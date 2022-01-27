%%%%%%%%%%%%%GRGMF,5-fold
clc
clear

% % %%%%%%%%%%%%step1£º pair prediction
cv_flag = 1;
M_D = process_data( );
prop = [1,10,30,50,70,100];
main_cv(cv_flag,M_D,prop);
% 100.0000    0.1000    0.0100


% %%%%%%%%%%%%step2£º new drug
cv_flag = 2;
M_D = process_data( );
prop = 100;
main_cv(cv_flag,M_D,prop);



% %%%%%%%%%%%%step3£ºnew microbe
cv_flag = 3;
M_D = process_data( );
prop = 100;
main_cv(cv_flag,M_D,prop);


