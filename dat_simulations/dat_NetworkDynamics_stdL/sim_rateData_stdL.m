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

tic
%load network parameter
NxI1=50;
inhwidth=2.0;
paraFile=[functionPath sprintf('ParamsStable2Unstable_Nx=%d_inhwidth=%.3f.mat',NxI1,inhwidth)];
load(paraFile)

rng(Wseed);

% Define path for simulation data folder
sim_folder = 'dat_simulations/';
% Check if simulation folder exists, if not, create it
if ~exist(sim_folder, 'dir')
    mkdir(sim_folder);
    fprintf('Created simulation folder: %s\n', sim_folder);
end

Nrepetitions=61; % sampling number
para_mu=20;
stdLList=[0 3 8];%[0 4 5 8 10];
flag_presyn='nosignal'; % input weight: randn or ones or space or nosignal
flag_disp='lognormal'; % taum distribution: lognormal gaussian
dtI=0.5;
trial=1;

for jj=1:length(stdLList)
    stdL=stdLList(jj);

    saveFile = [sim_folder sprintf(...
        '%s%d_dtI%.2f_std%.2f_Nx%d_Iw%.1f_s%d.mat'...
        ,flag_presyn,Nrepetitions,dtI,stdL,NxI1,inhwidth,trial)];

    if(~exist(saveFile,'file'))
        %load data
        Trunc=[1,100];
        [taum_e]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdL,Trunc);
        taum_inv=[1./taum_e', 1./taum_e'];

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
        Isignal = zeros(NGroups,numel(Ilevels)+1);

        %==============================
        % Run simulation
        %==============================
        % Maximum number of spikes.
        % Simulation will terminate with a warning if this is exceeded
        maxns=NNeur*T*.1;

        OutputFlag=0;
        if(OutputFlag==0)
            [sx_train]=Spatial2DNetSim_tauStim(Wx,I0,...
                vl,NGroups,vth,vre,vlb,V0,T,dt,dtIs,maxns,taum_inv,refractory,...
                current,taudecay,taurise,Jx,OutputFlag,NLoc,Isignal,Ilevels);
        end

        save(saveFile,'sx_train','Nrepetitions','StimSize','NNeure','Ne','taum_e','-v7.3')

        % How long did the simulation take
        t0=toc;
        fprintf('\nTime spent simulating network: %1.2f seconds\n',t0);
    end
end