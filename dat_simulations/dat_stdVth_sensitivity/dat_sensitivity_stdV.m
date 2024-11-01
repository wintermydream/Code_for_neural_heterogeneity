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

sinFreq=30;
flag_presyn='ones'; % input weight: randn or ones or space or nosignal
dtI=0.5;     % Time steps for stimulus (needs to be much smaller than taucorr)
Nrepetitions=62; % sampling number
% parameters for gaussian filter
filter_sigma=2/dtI;
filter_bin=20;
flagFilter=1;
NxI1=50;
inhwidth=2;
StimSize=2000;
stdL=0;
stdN=0;

trialsList=1:20;
stdParaList=[0 3 8];
scalingList=0.00:0.05:0.5;
num_channels=2500;

for i_trial=1:length(trialsList)
    trial=trialsList(i_trial);
    for i_stdPara=1:length(stdParaList)
        stdPara=stdParaList(i_stdPara);


        saveFile_temp=sprintf(...
            'dat_sensitivity_std%d_sinFreq%d_s%d.mat'...
            ,stdPara,sinFreq,trial);
        if(~exist(saveFile_temp,'file'))
            snrMean = zeros(length(scalingList),1);
            snrStd = zeros(length(scalingList),1);
            fprintf('Calculating begin: %s\n',saveFile_temp);

            % Initialize variables
            idx_silent = [];
            snrTemp = zeros(length(scalingList), num_channels);
            scaledAmplitudes = scalingList * 20;

            saveFlag=1;
            % Loop through each scaling factor in scalingList
            for i_scaling = 1:length(scalingList)
                scaling = scalingList(i_scaling);
                tic; % Start timing the iteration

                % Load file with dynamic filename generation
                filename = [sim_folder sprintf(...
                    'sCount_%sN%d_sc%.2f_stdV%.1f_stdLN%d_%d_f%d_Iw%.2f_s%d.mat'...
                    ,flag_presyn,Nrepetitions,scaling,stdPara,stdL,stdN,sinFreq,inhwidth,trial)];
                if(~exist(filename,'file'))
                    fprintf('unexisted file: %s\n',filename);
                    saveFlag=0;
                    break;
                else
                    load(filename);

                    % Define Gaussian filter parameters
                    filter_sigma = 2 / dtI;
                    filter_bin = 20;
                    flagFilter = 1;

                    % Apply Gaussian smoothing if enabled
                    if flagFilter
                        sCount = Fun_FixNet.GaussianSmooth(filter_sigma, filter_bin, dtI, sCount);
                    end

                    % Truncate beginning and end based on StimSize
                    sCount = sCount(StimSize + 1:end - StimSize, :);

                    % Identify and remove silent neurons
                    thres = mean(sCount(:)) / 20;
                    active_neurons = sum(sCount, 1) >= thres;
                    sCount = sCount(:, active_neurons);

                    % Update the list of silent neuron indices
                    idx_silent = unique([idx_silent, find(~active_neurons)]);

                    % Calculate SNR for active neurons
                    [snr, ~, ~] = SNR(sCount', dtI, sinFreq);
                    snrTemp(i_scaling, active_neurons) = snr;

                    % Display elapsed time for each scaling
                    fprintf('Elapsed time: %.4f s\n', scaling, toc);
                end
            end

            if(saveFlag==1)
                % Final adjustments after all scaling factors processed
                snrTemp(:, idx_silent) = []; % Remove columns of silent neurons from snrTemp
                snrTemp(snrTemp<0) = 0;
                snrTemp(isnan(snrTemp)) = 0;
                sqrtSNR = sqrt(snrTemp);      % Calculate square root of SNR

                % Calculate slope for each channel
                slope = zeros(size(sqrtSNR, 2), 1);
                for idx_channel = 1:size(sqrtSNR, 2)
                    % Perform linear fit for scaled amplitudes vs. sqrtSNR
                    pFit = polyfit(scaledAmplitudes, sqrtSNR(:, idx_channel), 1);
                    slope(idx_channel) = pFit(1);
                end

                save(saveFile_temp,'slope','idx_silent');
            end
        end
    end
end