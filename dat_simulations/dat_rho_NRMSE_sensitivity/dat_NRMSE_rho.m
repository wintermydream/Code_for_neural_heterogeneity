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

% Define path for simulation data folder
sim_folder = 'dat_simulations/';

lambda=100;
Nrepetitions=62;
Ntrain=50;
Ntest=10;
MGfreq=50;
para_mu=20;
flag_presyn='ones'; % input weight: randn or ones or space or nosignal
flag_disp='lognormal'; % taum distribution: lognormal gaussian
dtI=0.5;
scaling=1.0;
NxI1=50;
inhwidth=2.0;

saveFile=sprintf('datC_ones_rho.mat');
if(~exist(saveFile,'file'))

    stdList=[3 8];
    rhoList=[0:0.1:1];
    trialList=1;

    Error_array=zeros(length(trialList),length(stdList),length(rhoList));
    Error_m=zeros(length(stdList),length(rhoList));
    Error_s=zeros(length(stdList),length(rhoList));

    for i_stdNet=1:length(stdList)
        stdNet=stdList(i_stdNet);
        stdLeak=stdList(i_stdNet);

        for i_rho=1:length(rhoList)
            rho=rhoList(i_rho);
            if((stdNet==0) && (stdLeak==0))
                rho=1;
            end

            for i_trial=1:length(trialList)
                trials=trialList(i_trial);

                tic
                loadFile = sprintf('MGC%d%s%d_f%d_std%d_%d_rho%.1f_Nx%d_Iw%.3f_s%d.mat'...
                    ,scaling*10,flag_presyn,Nrepetitions,MGfreq,stdNet,stdLeak,rho,NxI1,inhwidth,trials);
                load([sim_folder loadFile])

                % gaussian filtering============
                filter_sigma=0.5/dtI;
                [Rfilter]=Fun_FixNet.gaussian_filter(filter_sigma,dtI,sCount);
                clearvars sCount
                % Regression================
                % train
                Rfilter_train=Rfilter(StimSize+1:(Ntrain+1)*StimSize,:);
                Itarget_train=Itarget(StimSize+1:(Ntrain+1)*StimSize);
                input = reshape(zscore(Rfilter_train(:)),size(Rfilter_train));
                wo = (pinv(input'*input+lambda*eye(size(input,2))) * input'*Itarget_train')';
                % test
                Rfilter_test=Rfilter(end-Ntest*StimSize+1:end,:);
                Itarget_test=Itarget(end-Ntest*StimSize+1:end);
                input = reshape(zscore(Rfilter_test(:)),size(Rfilter_test));
                zTest = wo*input';
                Error_array(i_trial,i_stdNet,i_rho)=sqrt(mean((zTest-Itarget_test).^2))/sqrt(mean((zTest-mean(Itarget_test)).^2));

                % Stop the timer and display the elapsed time
                elapsedTime = toc;
                fprintf('Simulation time: %.4f seconds\n', elapsedTime);

            end
            Error_m(i_stdNet,i_rho)=squeeze(mean(Error_array(:,i_stdNet,i_rho)));
            Error_s(i_stdNet,i_rho)=squeeze(std(Error_array(:,i_stdNet,i_rho)));

        end
    end
    save(saveFile,'Error_array','Error_m','Error_s','stdList','rhoList','-v7.3')
else
    fprintf('saved file: %s\n',saveFile);

    load(saveFile)
    figure
    plot(rhoList,Error_m(1,:),'r');hold on
    plot(rhoList,Error_m(2,:),'m');hold on
    strList = {'$\sigma_{\tau_{\rm L, \Gamma}}=3$ (ms)', '$\sigma_{\tau_{\rm L, \Gamma}}=8$ (ms)'};
    hLegend = legend(strList,'Interpreter','latex','location','best');
    ylabel('$\rho$')
    ylabel('NRMSE')
end