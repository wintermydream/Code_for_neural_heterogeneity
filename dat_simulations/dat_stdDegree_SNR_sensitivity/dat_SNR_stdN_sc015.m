% Resetting the random number generator
clear, clc, close all
rng('default')  % Reset RNG to default state
rng('shuffle')  % Shuffle RNG for randomness

% Adding path to custom functions based on the operating system
if ispc  % Windows
    functionPath='F:\Code_for_neural_heterogeneity\dat_simulations\Functions\';
elseif isunix  % Unix-like systems (Linux, macOS, etc.)
    functionPath = '/media/wu/Elements/Code_for_neural_heterogeneity/dat_simulations/Functions/';
else
    disp('Unsupported operating system.');  % Display a message if the operating system is not recognized
end
addpath(functionPath);

sim_folder = 'dat_simulations/';

scaling=0.15;
NxI1=50;
inhwidth=2.0;


flag_presyn='ones'; % input weight: randn or ones or space or nosignal
dtI=0.5;     % Time steps for stimulus (needs to be much smaller than taucorr)
Nrepetitions=62; % sampling number
% parameters for gaussian filter
filter_sigma=2/dtI;
filter_bin=20;
flagFilter=1;

% Array of MG frequencies
sinFreq = 5;

stdN=0;
stdParaList=0:10;
saveFile=sprintf('dat_SNR_stdL_sc%.2f_f%d.mat',scaling,sinFreq);
if(~exist(saveFile,'file'))
    snrmm = zeros(size(stdParaList));
    snrms = zeros(size(stdParaList));
    snrsm = zeros(size(stdParaList));
    snrss = zeros(size(stdParaList));
    for i_std=1:length(stdParaList)
        stdL=stdParaList(i_std);
        saveFile_tem=sprintf('dat_SNR_stdL%dN%d_sc%.2f_f%d.mat',stdL,stdN,scaling,sinFreq);
        if(~exist(saveFile_tem,'file'))
            fprintf('nonexist: %s\n',saveFile_tem)
            trialsList=[1:20];
            snrmAll= [];
            snrsAll= [];
            for trial_index = 1:length(trialsList)
                trials = trialsList(trial_index);
                % load file name
                loadFile = [sim_folder sprintf(...
                        'SNR%d%s%d_f%d_std%d_%d_Nx%d_Iw%.3f_s%d.mat'...
                        ,scaling*100,flag_presyn,Nrepetitions,sinFreq,stdL,stdN,NxI1,inhwidth,trials)];
                load(loadFile)
                % gaussian fittering
%                 sCount=sx_train;
                if flagFilter==1
                    [sCount]=Fun_FixNet.GaussianSmooth(filter_sigma,filter_bin,dtI,sCount);
                end
                % trunction of begening and end
                StimSize=StimTime/dtI;
                sCount=sCount(StimSize+1:end-StimSize,:);
                % delete silent neurons
                thres=mean(mean(sCount))/20;
                temp=sum(sCount,1);
                index=find(temp<thres);
                sCount(:,index)=[];
                % calculating SNR
                [snr,f,P]=SNR(sCount',dtI,sinFreq);
                snr(snr==0)=[];
                snr(isnan(snr))=[];
                snrmAll= [snrmAll; mean(snr)];
                snrsAll=[snrsAll;std(snr)];
            end
            clearvars P sCount snr temp f
            save(saveFile_tem,'snrmAll','snrsAll');
        else
            load(saveFile_tem)
            snrmm(i_std) = mean(snrmAll);
            snrms(i_std) = std(snrmAll);
            snrsm(i_std)  = mean(snrsAll);
            snrss(i_std)  = std(snrsAll);
        end
    end
    save(saveFile,'snrmm','snrms','snrsm','snrss');
end