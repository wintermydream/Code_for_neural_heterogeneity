clear,clc,close all
saveFile = 'dat_sensitivity_stdW_selectedTrials.mat';
if(~exist(saveFile,'file'))
    sensitivity=cell(3,1);
    for idx = 1:length(sensitivity)
        sensitivity{idx}.slope=[];
    end
    for idx_trial=1:2
        dataFiles = {['dat_sensitivity_stdW2.00_freqSin30_s' num2str(idx_trial,'%d') '.mat'],...
            ['dat_sensitivity_stdW0.30_freqSin30_s' num2str(idx_trial,'%d') '.mat'], ...
            ['dat_sensitivity_stdW0.20_freqSin30_s' num2str(idx_trial,'%d') '.mat']};
        if(idx_trial<=2)
            for idx = 1:length(dataFiles)
                load(dataFiles{idx}, 'slope'); % Load 'slope' variable from each file
                sensitivity{idx}.slope=[sensitivity{idx}.slope; slope];
            end
        elseif(idx_trial<=5)
            for idx = 1:2
                load(dataFiles{idx}, 'slope'); % Load 'slope' variable from each file
                sensitivity{idx}.slope=[sensitivity{idx}.slope; slope];
            end
        else
                load(dataFiles{1}, 'slope'); % Load 'slope' variable from each file
                sensitivity{1}.slope=[sensitivity{1}.slope; slope];
        end
    end
    save(saveFile,'sensitivity')
end

figure
load('dat_sensitivity_stdW2.00_freqSin30_s1.mat')
plot(mean(sqrtSNR,2));hold on
load('dat_sensitivity_stdW0.30_freqSin30_s1.mat')
plot(mean(sqrtSNR,2));hold on
load('dat_sensitivity_stdW0.20_freqSin30_s1.mat')
plot(mean(sqrtSNR,2));hold on
legend('2','0.15','0.05')