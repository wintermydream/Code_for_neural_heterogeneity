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
%% load raster data====================
saveFile=sprintf('dat_spk2rate_N21_stdL08.mat');
if(~exist(saveFile,'file'))
    Tbegin = 10000;
    Tend=Tbegin+10000;
    Ne1 = 200;
    loadFile = [sim_folder 'nosignal21_dtI0.50_std0.00_Nx50_Iw2.0.mat'];
    load([sim_folder loadFile])
    s_std0=sx_train;
    s_std0=s_std0(:,s_std0(2,:)<=Ne1*Ne1);
    s_std0=s_std0(:,s_std0(1,:)>Tbegin);
    s_std0=s_std0(:,s_std0(1,:)<=Tend);
    clearvars sx_train Itarget

    loadFile = [sim_folder 'nosignal21_dtI0.50_std8.00_Nx50_Iw2.0.mat'];
    load([sim_folder loadFile])
    s_std8=sx_train;
    s_std8=s_std8(:,s_std8(2,:)<=Ne1*Ne1);
    s_std8=s_std8(:,s_std8(1,:)>Tbegin);
    s_std8=s_std8(:,s_std8(1,:)<=Tend);
    clearvars sx_train Itarget

    % raster -------------------------------------------
    % data parameters
    dtI=0.5;
    DispTime=Tbegin+500;
    DispDuration=2000+1000;
    timeInterval=DispTime:dtI:DispTime+DispDuration;
    dx=0.02;
    Nx=1/dx;
    Ne1=200;
    StepSizeE=Ne1/Nx;
    %sampling
    samplingNum=3;
    % normal: [24 18; 20 29; ]
    % spase: [32 30; 4 18; 45 38; 29 1;]
    % dense: [7 40; 37 39; 14 41; 13 37; 39 3; 44 23]
    % XIndex=randperm(Nx,samplingNum);
    % YIndex=randperm(Nx,samplingNum);
    XIndex=[24 29 44];
    YIndex=[18 1 23];

    s_std0 = s_std0(:,s_std0(1,:)>DispTime & s_std0(1,:)<=DispTime+DispDuration);
    s_std8 = s_std8(:,s_std8(1,:)>DispTime & s_std8(1,:)<=DispTime+DispDuration);
    [X0,Y0]=ind2sub([Ne1 Ne1],s_std0(2,:));
    [X8,Y8]=ind2sub([Ne1 Ne1],s_std8(2,:));

    spkData_std0=cell(samplingNum,1);
    spkData_std8=cell(samplingNum,1);
    rate_std0=cell(samplingNum,1);
    rate_std8=cell(samplingNum,1);
    smoothRate_std0=cell(samplingNum,1);
    smoothRate_std8=cell(samplingNum,1);
    for i=1:length(XIndex)
        % spike
        spkData_std0{i} = s_std0(:, X0>(XIndex(i)-1)*StepSizeE &...
            X0<=XIndex(i)*StepSizeE & Y0>(YIndex(i)-1)*StepSizeE & Y0<=YIndex(i)*StepSizeE);
        % neuron indice
        temp=unique( spkData_std0{i}(2,:));
        for j=1:length(temp)
            % neuron indice from 1 to length(temp)
            spkData_std0{i}(2,spkData_std0{i}(2,:)==temp(j))=j;
        end
        spkData_std8{i} = s_std8(:, X8>(XIndex(i)-1)*StepSizeE &...
            X8<=XIndex(i)*StepSizeE & Y8>(YIndex(i)-1)*StepSizeE & Y8<=YIndex(i)*StepSizeE);
        temp=unique( spkData_std8{i}(2,:));
        for j=1:length(temp)
            spkData_std8{i}(2,spkData_std8{i}(2,:)==temp(j))=j;
        end
        % rate
        % calculating firing rate
        [temp0] = histcounts(spkData_std0{i}(1,:), timeInterval)/dtI/StepSizeE/StepSizeE*1000;
        [temp8] = histcounts(spkData_std8{i}(1,:), timeInterval)/dtI/StepSizeE/StepSizeE*1000;
        if(size(temp0,2)~=1)
            temp0=temp0';
            temp8=temp8';
        end
        rate_std0{i}=temp0;
        rate_std8{i}=temp8;
    end

    filter_sigma=2/dtI;
    filter_bin=20;
    for i=1:length(rate_std8)
        [smoothRate_std0{i}]=Fun_FixNet.GaussianSmooth(filter_sigma,filter_bin,dtI,rate_std0{i});
        [smoothRate_std8{i}]=Fun_FixNet.GaussianSmooth(filter_sigma,filter_bin,dtI,rate_std8{i});
    end

    save(saveFile,'spkData_std0','spkData_std8','rate_std0','rate_std8','DispTime','DispDuration','dtI','smoothRate_std0','smoothRate_std8','XIndex','YIndex');
else
    load(saveFile)
    figure
    subplot(211)
    plot(rate_std0{1})
    subplot(212)
    filter_sigma=2/dtI;
    filter_bin=20;
    smoothRate_std0{1}=Fun_FixNet.GaussianSmooth(filter_sigma,filter_bin,dtI,rate_std0{1}');
    plot(smoothRate_std0{1})
end