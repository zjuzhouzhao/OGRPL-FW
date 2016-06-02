%% This is a demo program of predicting CIAO ratings by using OGRPL-FW
% code by Hanqing Lu from Zhejiang University.
%% load data
clear clc;
load CIAOdata.mat
 
%% set the parameter
lambda = 0.0000001;
[n,m] = size(R);
K = 15;
alpha = 0.0002;

%% train data using the OGRPL-FW algorithm
fprintf('Training process...\n');
[W] = OGRPL(R,P,X,L,lambda,K);
RMSE = norm((Test>0).*(R - W'*X),'fro')/sqrt(sum(sum(Test>0)));
fprintf('Training process is over, the testing Root Mean Square Error= %.4f\n',RMSE);



