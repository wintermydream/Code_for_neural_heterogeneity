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

saveFile=sprintf('dat_powerSpectrum_rate_stdL038.mat');
if(~exist(saveFile,'file'))
    % std0 ======================
    dtI=0.5;
    Fs=1000/dtI; % sampling rate
    load([dataPath 'nosignal61_dtI0.50_std0.00_Nx50_Iw2.0_s1.mat'])
    tcut=StimSize;
    sx_train=sx_train(tcut+1:end,:);
    thres=mean(mean(sx_train))/20;
    temp=sum(sx_train,1);
    index=find(temp<thres);
    sx_train(:,index)=[];
    NGroups=size(sx_train,2);

    Ptemp=zeros(NGroups,20000-1);
    for i_group=1:NGroups
        tic
        sp=zscore(sx_train(:,i_group)');
        nfft=2^nextpow2(length(sp));
        Y=fft(sp-mean(sp),nfft);
        Pxx1=Y.*conj(Y); %/nfft; %power spectrum 1
        fs1=Fs*(0:(nfft/2))/nfft;
        [~,index1]=min(abs(fs1-500));
        Ptemp(i_group,:)=Pxx1(2:20000);
        toc
    end
    Ptemp=log10(Ptemp);
    Pm_std0=mean(Ptemp);
    Ps_std0=std(Ptemp);
    f0_std0=fs1(2:20000);

    sx_population=mean(sx_train,2)';
    sp=zscore(sx_population);
    nfft=2^nextpow2(length(sp));
    Y=fft(sp-mean(sp),nfft);
    Pxx1=Y.*conj(Y); %/nfft; %power spectrum 1
    fs1=Fs*(0:(nfft/2))/nfft;
    [~,index1]=min(abs(fs1-500));
    Pall_std0=log10(Pxx1(2:20000));

    % std3 ======================
    load([dataPath 'nosignal61_dtI0.50_std3.00_Nx50_Iw2.0_s1.mat'])
    tcut=StimSize;
    sx_train=sx_train(tcut+1:end,:);
    thres=mean(mean(sx_train))/20;
    temp=sum(sx_train,1);
    index=find(temp<thres);
    sx_train(:,index)=[];
    NGroups=size(sx_train,2);

    Ptemp=zeros(NGroups,20000-1);
    for i_group=1:NGroups
        tic
        sp=zscore(sx_train(:,i_group)');
        nfft=2^nextpow2(length(sp));
        Y=fft(sp-mean(sp),nfft);
        Pxx1=Y.*conj(Y); %/nfft; %power spectrum 1
        fs1=Fs*(0:(nfft/2))/nfft;
        [~,index1]=min(abs(fs1-500));
        Ptemp(i_group,:)=Pxx1(2:20000);
        toc
    end
    Ptemp=log10(Ptemp);
    Pm_std3=mean(Ptemp);
    Ps_std3=std(Ptemp);
    f0_std3=fs1(2:20000);

    sx_population=mean(sx_train,2)';
    sp=zscore(sx_population);
    nfft=2^nextpow2(length(sp));
    Y=fft(sp-mean(sp),nfft);
    Pxx1=Y.*conj(Y); %/nfft; %power spectrum 1
    fs1=Fs*(0:(nfft/2))/nfft;
    [~,index1]=min(abs(fs1-500));
    Pall_std3=log10(Pxx1(2:20000));

    % std8 ======================
    load([dataPath 'nosignal61_dtI0.50_std8.00_Nx50_Iw2.0_s1.mat'])
    tcut=StimSize;
    sx_train=sx_train(tcut+1:end,:);
    thres=mean(mean(sx_train))/20;
    temp=sum(sx_train,1);
    index=find(temp<thres);
    sx_train(:,index)=[];
    NGroups=size(sx_train,2);

    Ptemp=zeros(NGroups,20000-1);
    for i_group=1:NGroups
        tic
        sp=zscore(sx_train(:,i_group)');
        nfft=2^nextpow2(length(sp));
        Y=fft(sp-mean(sp),nfft);
        Pxx1=Y.*conj(Y); %/nfft; %power spectrum 1
        fs1=Fs*(0:(nfft/2))/nfft;
        [~,index1]=min(abs(fs1-500));
        Ptemp(i_group,:)=Pxx1(2:20000);
        toc
    end
    Ptemp=log10(Ptemp);
    Pm_std8=mean(Ptemp);
    Ps_std8=std(Ptemp);
    f0_std8=fs1(2:20000);

    sx_population=mean(sx_train,2)';
    sp=zscore(sx_population);
    nfft=2^nextpow2(length(sp));
    Y=fft(sp-mean(sp),nfft);
    Pxx1=Y.*conj(Y); %/nfft; %power spectrum 1
    fs1=Fs*(0:(nfft/2))/nfft;
    [~,index1]=min(abs(fs1-500));
    Pall_std8=log10(Pxx1(2:20000));

    save(saveFile,'f0_std0','Pm_std0','Ps_std0',...
        'f0_std3','Pm_std3','Ps_std3',...
        'f0_std8','Pm_std8','Ps_std8')
else
    fprintf('saved file: %s\n',saveFile)
end