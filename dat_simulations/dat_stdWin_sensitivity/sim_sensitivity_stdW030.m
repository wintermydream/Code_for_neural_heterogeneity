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

Nrepetitions=62; % sampling number
inhwidth = 2; % ratio of excitatory-inhibitory projections
stdL=0;
stdN=0;

% network parameters
flag_presyn='Wspace';
flag_disp='lognormal'; % tau distribution: lognormal gaussian
Ne=2500; % number of grids in excitatory and inhibitory populations
para_mu=20; % mean of timescales
dtI = 0.5;  % Time step
Trunc=[1,100];

% load network parameters
NxI1=50;
paraFile=[functionPath sprintf('ParamsStable2Unstable_Nx=%d_inhwidth=%.3f.mat',NxI1,inhwidth)];
load(paraFile)

% timescale diversity===================================================
rng('shuffle')  % Shuffle RNG for randomness
% generating random number
[temp_tauL]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdL,Trunc);
[temp_tauN]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdN,Trunc);
% Leakage diversity
tauL_inv=[1./temp_tauL', 1./temp_tauL'];
% response diversity
tauN_inv=[20./temp_tauN',20./temp_tauN'];

% simulation loops========================================================
trialList = 1:2;  % Array of trial
for i_trial = 1:length(trialList)
    trial = trialList(i_trial);

    freqSinList=[30];
    for i_freq=1:length(freqSinList)
        freqSin=freqSinList(i_freq);

        sigmaW_list=[0.30];
        for i_wr=1:length(sigmaW_list)
            sigmaW=sigmaW_list(i_wr);

            % simulation time
            dt = .05;
            dtIs=(dtI)/dt;
            StimTime=1000;
            StimSize = StimTime/dtI;

            T=StimTime*Nrepetitions;
            Ilevels=0:dtI:T-dtI;

            % generating Sin signal
            Imult=sin(2*pi*freqSin*(dtI:dtI:StimTime*Nrepetitions)/StimTime);

            % presynaptic weight
            if(strcmp(flag_presyn,'Wspace') && sigmaW>=1)
                flag_presyn='ones';
            else
                flag_presyn='Wspace';
            end
            switch flag_presyn
                case 'ones'
                    WinRaw  = ones(NGroups, 1);
                case 'space'
                    WinRaw  = Ispace;
                case 'Wspace'
                    % generating heterogeneous input weight
                    StimSeed=829;
                    Ispace0=FunWeightStimulus0(NxI1,sigmaW,StimSeed);
                    Ispace0=(1/(std(Ispace0(:))))*(Ispace0-mean(Ispace0(:)));
                    Ispace=[Ispace0(:);Ispace0(:)];
                    WinRaw  = Ispace;
                case 'nosignal'
                    WinRaw  = zeros(NGroups, 1);
            end

            scalingList=[0.01:0.02:0.40];
            for i_scaling=1:length(scalingList)
                scaling=scalingList(i_scaling);

                % Constructing save path for simulation data
                saveFile = [sim_folder sprintf(...
                    '%sN%d_sigmaW%.2f_stdL%d_N%d_freqSin%d_Iw%.2f_sc%.2f_s%d.mat'...
                    ,flag_presyn,Nrepetitions,sigmaW,stdL,stdN,freqSin,inhwidth,scaling,trial)];

                % if(~exist(saveFile, 'file'))
                fprintf('unsaved file: %s\n', saveFile);

                % input signal
                Imult=zscore(Imult);
                Isignal = zeros(NGroups,numel(Ilevels)+1);
                for i=2:size(Isignal,2)
                    Isignal(:,i)=scaling*WinRaw.*Imult(:,i-1);
                end

                % Run simulation
                maxns=NNeur*T*.1; % Maximum number of spikes.

                tic
                % Generate random initial membrane potentials
                rng('shuffle')  % Shuffle RNG for randomness
                V0 = (-50 + 75) * rand(NNeur, 1) - 75;
                OutputFlag=0;
                [sCount]=SimHeteScount(Wx,I0,...
                    vl,NGroups,vth,vre,vlb,V0,T,dt,dtIs,maxns,tauL_inv,tauN_inv,refractory,...
                    current,taudecay,taurise,Jx,OutputFlag,NLoc,Isignal,Ilevels);
                save(saveFile,'sCount','WinRaw','Nrepetitions','StimSize','dtI','-v7.3')

                t0=toc; % How long did the simulation take
                fprintf('\nTime spent simulating network: %1.2f seconds\n',t0);
                % else
                %     fprintf('Saved file: %s\n', saveFile);
                % end
            end
        end
    end
end