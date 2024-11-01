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

tic
%load network parameter
NxI1=50;
inhwidth=2.0;
paraFile=[functionPath sprintf('ParamsStable2Unstable_Nx=%d_inhwidth=%.3f.mat',NxI1,inhwidth)];
load(paraFile)
rng(Wseed);

Nrepetitions=201; % sampling number
para_mu=20;
stdLList=[0 3 8];
flag_presyn='nosignal'; % input weight: randn or ones or space or nosignal
flag_disp='lognormal'; % taum distribution: lognormal gaussian
dtI=0.5;

for i_stdL=1:length(stdLList)
    stdL=stdLList(i_stdL);

    saveFile = [sim_folder sprintf(...
        'S_%s%d_dtI%.2f_std%.2f_Nx%d_Iw%.1f.mat'...
        ,flag_presyn,Nrepetitions,dtI,stdL,NxI1,inhwidth)];
    if(~exist(saveFile,'file'))
        %load data
        Trunc=[1,100];
        [taum_inve]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdL,Trunc);
        taum_inv=[1./taum_inve', 1./taum_inve'];
        clearvars taum_inve  taum_invi

        % time step
        dt = .05;
        dtIs=(dtI)/dt;
        StimTime=1000;
        StimSize = StimTime/dtI;
        %===================
        % Generating the external stimuli signal
        % generating sinusoidal signal
        Imult=sin(2*pi*(dtI:dtI:StimTime*Nrepetitions)/StimTime);
        Itarget=Imult;

        %==========================
        %  simulation time
        StimLength=(StimSize)*dtI;
        T=StimLength*Nrepetitions;
        Ilevels=0:dtI:T-dtI;

        % ==========================
        % presynaptic weight
        scaling=1.5;
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

        OutputFlag=0;
        [s_out]=Spatial2DNetSim_Sout(Wx,I0,...
            vl,NGroups,vth,vre,vlb,V0,T,dt,dtIs,maxns,taum_inv,refractory,...
            current,taudecay,taurise,Jx,OutputFlag,NLoc,Isignal,Ilevels);
        s_out=s_out(:,s_out(2,:)>0);
        
        save(saveFile,'s_out','Nrepetitions','StimSize','NNeure','Ne','-v7.3')

        % How long did the simulation take
        t0=toc;
        fprintf('\nTime spent simulating network: %1.2f seconds\n',t0);
    end
end