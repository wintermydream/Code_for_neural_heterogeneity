clear,clc,close all

sim_folder = 'dat_simulations/';

Ne1=200;
Tbegin = 1000*1;
Tend=Tbegin+200*1000; % ms
T=Tend-Tbegin;
% parameters for correlation
Tw=100;
saveFile=sprintf('dat_corrBasedSynchrony_Tw%d.mat',Tw);
if(~exist(saveFile,'file'))
    stdList=[0 3 8];
    for i_std=1:length(stdList)
        if(~exist(['Corr_std',num2str(stdList(i_std),'%d')],'var'))
            varName=['Corr_std',num2str(stdList(i_std),'%d')];

            % load raster data====================
            loadFile=[sim_folder sprintf('S_nosignal201_dtI0.50_std%.2f_Nx50_Iw2.0.mat',stdList(i_std))];
            load(loadFile)

            % spike data processing
            s_out=s_out(:,s_out(2,:)<=Ne1*Ne1);
            s_out=s_out(:,s_out(1,:)>Tbegin);
            s_out=s_out(:,s_out(1,:)<=Tend);
            s_out(1,:)=s_out(1,:)-Tbegin;
            % parameters
            N1=40000;
            NGroup=2500;
            Ne1=200;
            StepSizeE=sqrt(N1/NGroup);
            % get the X and Y indice of neurons
            [X0,Y0]=ind2sub([Ne1 Ne1],s_out(2,:));
            % loops for calculating correlation of each subgroup
            Corr_buff=[];
            for i=1:sqrt(NGroup)
                for j=1:sqrt(NGroup)
                    tic
                    % dividing spikes into groups
                    s_out_temp = s_out(:, X0>(i-1)*StepSizeE & X0<=i*StepSizeE &...
                        Y0>(j-1)*StepSizeE & Y0<=j*StepSizeE);
                    % calculating firing rate
                    [GC,GR] = groupcounts(s_out_temp(2,:)');
                    GR(GC<400) = []; % Remove those neurons with low firing rate
                    if (numel(GR)>2)
                        s_out_temp = s_out_temp(:, ismember(s_out_temp(2,:), GR));
                        % compute spike counts using sliding window
                        time=0:Tw:T;
                        re_temp=zeros(numel(GR),length(time)-1);
                        for mm=1:numel(GR)
                            re_temp(mm,:)=histcounts(s_out(1, s_out(2,:)==GR(mm)), time);
                        end
                        corr_temp=corr(re_temp');
                        corr_temp=corr_temp-diag(diag(corr_temp));
                        corr_temp=corr_temp(:);
                        corr0=mean(corr_temp(corr_temp~=0));
                        Corr_buff=[Corr_buff;corr0];
                    end
                    elapsedTime = toc;
                    fprintf('Elapsed time: %.4f\n', elapsedTime);
                end
            end
            eval([varName,'=Corr_buff;']);
        end
    end
    save(saveFile,'Corr_std0','Corr_std3','Corr_std8','Tw')
end
