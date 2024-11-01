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
% Check if simulation folder exists, if not, create it
if ~exist(sim_folder, 'dir')
    mkdir(sim_folder);
    fprintf('Created simulation folder: %s\n', sim_folder);
end

%load network parameter
inhwidth=2.0;
NxI1=50;
paraFile=sprintf('ParamsStable2Unstable_Nx=%d_inhwidth=%.3f.mat',NxI1,inhwidth);
load(paraFile);

rng('default');
rng('shuffle');

Nrepetitions=62; % sampling number
para_mu=20;
flag_presyn='ones'; % input weight: randn or ones or space or nosignal
flag_disp='lognormal'; % taum distribution: lognormal gaussian
dtI=0.5;
scaling=0.15;
stdL=0;
stdN=0;

% timescale diversity
Trunc=[1,100];
[temp_tauL]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdL,Trunc);
tauL_inv=[1./temp_tauL', 1./temp_tauL'];
% Network diversity
Trunc=[1,100];
[temp_tauN]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdN,Trunc);
tauN_inv=[20./temp_tauN',20./temp_tauN'];

trialList=2:20;
for i_trial=1:length(trialList)
    trial=trialList(i_trial);

    sinFreqList=[30];
    for i_freq=1:length(sinFreqList)
        sinFreq=sinFreqList(i_freq);

        stdParaList=[8];
        for i_std=1:length(stdParaList)
            stdPara=stdParaList(i_std);

            scalingList=0.0:0.05:0.5;
            for i_scaling=1:length(scalingList)
                scaling=scalingList(i_scaling);

                saveFile = [sim_folder sprintf(...
                    'sCount_%sN%d_sc%.2f_stdV%.1f_stdLN%d_%d_f%d_Iw%.2f_s%d.mat'...
                    ,flag_presyn,Nrepetitions,scaling,stdPara,stdL,stdN,sinFreq,inhwidth,trial)];
                if(~exist(saveFile,'file'))
                    fprintf('Non-saved file: %s\n', saveFile);
                    % Threshold diversity
                    V_threshold0=-50;
                    V_rest0=-70;
                    muThres=V_threshold0 - V_rest0;
                    Trunc=[0 100];
                    [tempVarThres]=Fun_Hete_g.Hete_Net(Ne,flag_disp,muThres,stdPara,Trunc);
                    tempThres = V_threshold0+tempVarThres-muThres;
                    vThres=[tempThres; tempThres];

                    % time step
                    dt = .05;
                    dtIs=(dtI)/dt;
                    StimTime=1000;
                    StimSize = StimTime/dtI;

                    %==========================
                    %  simulation time
                    StimLength=(StimSize)*dtI;
                    T=StimLength*Nrepetitions;
                    Ilevels=0:dtI:T-dtI;

                    % ==========================
                    % generating Sin signal
                    Imult=sin(2*pi*sinFreq*(dtI:dtI:StimTime*Nrepetitions)/StimTime);
                    % presynaptic weight
                    switch flag_presyn
                        case 'randn'
                            temp=rand(NGroups/2, 1);
                            WinRaw  = scaling*[temp;temp];
                            clearvars temp
                        case 'ones'
                            WinRaw  = scaling*ones(NGroups, 1);
                        case 'space'
                            WinRaw  = scaling*Ispace;
                        case 'nosignal'
                            WinRaw  = zeros(NGroups, 1);
                    end

                    % input signal
                    Imult=zscore(Imult);
                    Isignal = zeros(NGroups,numel(Ilevels)+1);
                    for i=2:size(Isignal,2)
                        Isignal(:,i)=WinRaw.*Imult(:,i-1);
                    end

                    %==============================
                    % Run simulation
                    %==============================
                    % Maximum number of spikes.
                    % Simulation will terminate with a warning if this is exceeded
                    maxns=NNeur*T*.1;

                    tic
                    OutputFlag=0;
                    rng('shuffle');
                    V0min=-75;
                    V0max=-50;
                    V0=(V0max-V0min).*rand(NNeur,1)+V0min;
                    [sCount]=SimThresholdScount(Wx,I0,...
                        vl,NGroups,vThres,vre,vlb,V0,T,dt,dtIs,maxns,tauL_inv,tauN_inv,refractory,...
                        current,taudecay,taurise,Jx,OutputFlag,NLoc,Isignal,Ilevels);
                    save(saveFile,'sCount','tempThres','Nrepetitions','StimSize','dtI','-v7.3')
                    % How long did the simulation take
                    t0=toc;
                    fprintf('\nTime spent simulating network: %1.2f seconds\n',t0);
                    %}

                else
                    %                     fprintf('saved file: %s\n', saveFile);
                end
            end
        end
    end
end