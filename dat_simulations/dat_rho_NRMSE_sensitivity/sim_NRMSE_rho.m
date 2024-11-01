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

MGfreq=50;
Nrepetitions=62; % sampling number
para_mu=20;
flag_presyn='ones'; % input weight: randn or ones or space or nosignal
flag_disp='lognormal'; % taum distribution: lognormal gaussian
dtI=0.5;
scaling=1.0;

trialList=1:20;
for i_trial=1:length(trialList)
    trial=trialList(i_trial);

    inhwidth_array=2.0;
    for i_Iw=1:length(inhwidth_array)
        inhwidth=inhwidth_array(i_Iw);
        paraFile=sprintf('ParamsStable2Unstable_Nx=%d_inhwidth=%.3f.mat',NxI1,inhwidth);
        load(paraFile)

        stdList=[3 8];
        for i_stdNet=1:length(stdList)
            stdNet=stdList(i_stdNet);
            stdLeak=stdList(i_stdNet);

            rhoList=0:0.1:1;
            for i_rho=1:length(rhoList)
                rho=rhoList(i_rho);
                if((stdNet==0) && (stdLeak==0))
                    rho=1;
                end

                saveFile = [sim_folder sprintf(...
                    'MGC%d%s%d_f%d_std%d_%d_rho%.1f_Nx%d_Iw%.3f_s%d.mat'...
                    ,scaling*10,flag_presyn,Nrepetitions,MGfreq,stdNet,stdLeak,rho,NxI1,inhwidth,trial)];
                if(~exist(saveFile,'file'))

                    fprintf('================================\n');
                    fprintf('simulation begin: %s\n',saveFile);

                    Trunc=[1,100];
                    [temp_tauM]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdNet,Trunc);
                    [temp_tauC]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdNet,Trunc);
                    [temp_tauN]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdNet,Trunc);
                    indiceC=randperm(Ne,ceil(Ne*rho));
                    temp_tauM(indiceC) = temp_tauC(indiceC);
                    temp_tauN(indiceC) = temp_tauC(indiceC);
                    % Leakage diversity
                    tauL_inv=[1./temp_tauM', 1./temp_tauM'];
                    % Network diversity
                    tauN_inv=[20./temp_tauN',20./temp_tauN'];

                    % time step
                    dt = .05;
                    dtIs=(dtI)/dt;
                    StimTime=500;
                    StimSize = StimTime/dtI;

                    % generating MG like signal
                    tau_MG=17;
                    [temp,t]=Fun_FixNet.FunMG_match_noZS(StimTime/dtI,tau_MG,dtI*MGfreq/2);
                    datasetTrain=temp';
                    Imult=[];
                    Itarget=[];
                    for i = 1:Nrepetitions
                        Imult=[Imult,datasetTrain];
                        Itarget=[Itarget,datasetTrain];
                    end

                    %  simulation time
                    StimLength=(StimSize)*dtI;
                    T=StimLength*Nrepetitions;
                    Ilevels=0:dtI:T-dtI;

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
                    V0min=-75;
                    V0max=-50;
                    V0=(V0max-V0min).*rand(NNeur,1)+V0min;
                    [sCount]=SimHeteScount(Wx,I0,...
                        vl,NGroups,vth,vre,vlb,V0,T,dt,dtIs,maxns,tauL_inv,tauN_inv,refractory,...
                        current,taudecay,taurise,Jx,OutputFlag,NLoc,Isignal,Ilevels);

                    save(saveFile,'sCount','Itarget','Nrepetitions','StimSize','StimTime','-v7.3')

                    % How long did the simulation take
                    t0=toc;
                    fprintf('\nTime spent simulating network: %1.2f seconds\n',t0);
                else
                    fprintf('saved file: %s\n',saveFile)
                end
            end
        end
    end
end