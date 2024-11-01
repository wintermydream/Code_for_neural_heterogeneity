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

networkType='LognormalNetworks'; %  LognormalNetworks

stdDlist = 50:50:800;
for idx_stdD = 1:length(stdDlist)
    % Loading network parameters
    stdDegree=stdDlist(idx_stdD);
    inhwidth=2.0;
    NxI1=50;
    paraFile = [functionPath sprintf('dat_para%s_Nx=%d_stdDegree%d.mat'...
        ,networkType,NxI1,stdDegree)];
    if ~exist(paraFile, 'file')
        % Generate parameters if file doesn't exist
        NNeure=40000;
        NNeuri=10000;
        NNeur = NNeure + NNeuri;
        pee0=.0125;  % Connection probability (exc-to-exc)
        pei0=.05;    % Connection probability (exc-to-inh)
        pie0=.0125;  % Connection probability (exc-to-inh)
        pii0=.05;    % Connection probability (inh-to-inh)
        % Number of outgoing connections
        Kee=pee0*NNeure;
        Kei=pei0*NNeure;
        Kie=pie0*NNeuri;
        Kii=pii0*NNeuri;
        clearvars pee0 pei0 pie0 pii0
        % in-degree of exc:
        sumKinExc = Kee*NNeure + Kei*NNeuri;
        % in-degree of inh:
        sumKinInh = Kie*NNeure + Kii*NNeuri;
        flag_disp = 'lognormal_int';
        inDegree_muExc = sumKinExc/NNeure;
        inDegree_muInh = sumKinInh/NNeuri;
        Trunc=[10, 1e4];
        [inDegreeList] = Fun_Hete_g.Hete_Net(NNeur, flag_disp, inDegree_muExc, stdDegree, Trunc);
        % truncation
        if(stdDegree>0)
            [inDegreeList] = fct_inDegreeList(inDegreeList, NNeure,NNeuri,sumKinExc,sumKinInh, Trunc(2), Trunc(1));
        end
        fct_genNetworkParameters(NxI1,inhwidth,networkType,paraFile,inDegreeList);
    end
end

load(paraFile);

Nrepetitions = 62;  % Number of repetitions
para_mu = 20;  % Mean parameter
flag_presyn = 'ones';  % Presynaptic weight flag
flag_disp = 'lognormal';  % Distribution flag
dtI = 0.5;  % Time step
scaling = 0.15;  % Scaling factor
inhwidth = 2.0;  % Inhibitory width
stdN=0;

trialList = 1:20;  % Array of trial
for trial_index = 1:length(trialList)
    trial = trialList(trial_index);

    for idx_stdD = 1:length(stdDlist)
        stdDegree=stdDlist(idx_stdD);
        paraFile = [functionPath sprintf('dat_para%s_Nx=%d_stdDegree%d.mat'...
            ,networkType,NxI1,stdDegree)];
        load(paraFile);  % Load parameters

        % Generate random initial membrane potentials
        V0 = (-50 + 75) * rand(NNeur, 1) - 75;

        % Array of sinusoidal frequencies
        sinFreqList=[30];
        for i_freq=1:length(sinFreqList)
            sinFreq=sinFreqList(i_freq);

            stdParaList=[0 3 8];
            for i_stdL=1:length(stdParaList)
                stdL=stdParaList(i_stdL);

                saveFile = [sim_folder sprintf(...
                    'SNR%s_N%d_f%d_stdLND%d_%d_%d_sc%.2f_s%d.mat'...
                    ,flag_presyn,Nrepetitions,sinFreq,stdL,stdN,stdDegree,scaling,trial)];
                if(~exist(saveFile,'file'))

                    tic
                    % timescale diversity
                    Trunc=[1,100];
                    [temp_Leak]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdL,Trunc);
                    tauL_inv=[1./temp_Leak', 1./temp_Leak'];
                    % Network diversity
                    Trunc=[1,100];
                    [temp_Gamma]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,stdN,Trunc);
                    tauN_inv=[20./temp_Gamma',20./temp_Gamma'];

                    % time step
                    dt = .05;
                    dtIs=(dtI)/dt;
                    StimTime=1000;
                    StimSize = StimTime/dtI;

                    % generating Sin signal
                    Imult=sin(2*pi*sinFreq*(dtI:dtI:StimTime*Nrepetitions)/StimTime);

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
                    % Simulation will terminate with a warning if this is exceeded
                    maxns=NNeur*T*.1; % Maximum number of spikes.

                    OutputFlag=0;
                    [sCount]=SimHeteScount(Wx,I0,...
                        vl,NGroups,vth,vre,vlb,V0,T,dt,dtIs,maxns,tauL_inv,tauN_inv,refractory,...
                        current,taudecay,taurise,Jx,OutputFlag,NLoc,Isignal,Ilevels);

                    save(saveFile,'dtI','temp_Leak','sCount','StimTime','NNeure','Ne','-v7.3');

                    t0=toc; % Simulation time
                    fprintf('\nTime spent simulating network: %1.2f seconds\n',t0);
                else
                    fprintf('saved file:%s\n', saveFile);
                end
            end
        end
    end
end