% Clearing command window, workspace, and closing all figures
clear,clc,close all
% Adding path to custom functions
if ispc  % Check if the system is running Windows    
    dataPath = 'dat_simulations\';
elseif isunix  % Check if the system is running Unix-like system (Linux, macOS, etc.)    
    dataPath = 'dat_simulations/';
else
    disp('Unsupported operating system.');  % Display a message if the operating system is not recognized
end

saveFile=sprintf('dat_rate_distribution_stdL038.mat');
if(~exist(saveFile,'file'))
dtI=0.5;
% std0------------------------------
load([dataPath 'nosignal61_dtI0.50_std0.00_Nx50_Iw2.0_s1.mat'])
tcut=StimSize;
sx_train=sx_train(tcut+1:end,:);
NGroups=size(sx_train,2);
fr_std0=sum(sx_train)/((Nrepetitions-1)*StimSize*dtI)/(NNeure/NGroups)*1000;
%std3--------------------------------------------
load([dataPath 'nosignal61_dtI0.50_std3.00_Nx50_Iw2.0_s1.mat'])
tcut=StimSize;
sx_train=sx_train(tcut+1:end,:);
NGroups=size(sx_train,2);
fr_std3=sum(sx_train)/((Nrepetitions-1)*StimSize*dtI)/(NNeure/NGroups)*1000;
%std8--------------------------------------------
load([dataPath 'nosignal61_dtI0.50_std8.00_Nx50_Iw2.0_s1.mat'])
tcut=StimSize;
sx_train=sx_train(tcut+1:end,:);
NGroups=size(sx_train,2);
fr_std8=sum(sx_train)/((Nrepetitions-1)*StimSize*dtI)/(NNeure/NGroups)*1000;
save(saveFile,'fr_std0','fr_std3','fr_std8');
else
    load(saveFile)
end