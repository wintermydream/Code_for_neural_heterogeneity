clear,clc,close all
loadfile=sprintf('dat_spkCntCorr_hist.mat');
if(~exist(loadfile,'file'))
    N1=40000;
    dim='2D';
    Nc=500;
    Tw=100;
    %% calculating spike count correlation
    load([sim_folder 'S_nosignal201_dtI0.50_std0.00_Nx50_Iw2.0.mat'])
    [Corr, Cov, rate1, var1]=fct_spkCntCorr(s_out,N1,dim,Nc,Tw);
    % only keep Upper triangle of correlation matrix
    U=triu(ones(size(Corr)),1);
    % U is of same size as Corr
    corr_R=Corr(U==1);
    eval(['corr_std0','=','corr_R;']);

    load([sim_folder 'S_nosignal201_dtI0.50_std3.00_Nx50_Iw2.0.mat'])
    [Corr, Cov, rate1, var1]=fct_spkCntCorr(s_out,N1,dim,Nc,Tw);
    % only keep Upper triangle of correlation matrix
    U=triu(ones(size(Corr)),1);
    % U is of same size as Corr
    corr_R=Corr(U==1);
    eval(['corr_std3','=','corr_R;']);

    load([sim_folder 'S_nosignal201_dtI0.50_std8.00_Nx50_Iw2.0.mat'])
    [Corr, Cov, rate1, var1]=fct_spkCntCorr(s_out,N1,dim,Nc,Tw);
    % only keep Upper triangle of correlation matrix
    U=triu(ones(size(Corr)),1);
    % U is of same size as Corr
    corr_R=Corr(U==1);
    eval(['corr_std8','=','corr_R;']);
    save(loadfile,'corr_std8','corr_std3','corr_std0');
else
    load(loadfile);
end

figure()
FontSize=13;
subplot(121)
edges=-0.5:1/100:0.5;
[bincountsL2,edges] = histcounts(corr_std0,edges);
histogram('BinCounts', bincountsL2/sum(bincountsL2), 'BinEdges', edges,'FaceColor','black');hold on
xlabel('Spike count correlation','fontsize',FontSize)
ylabel('probability','fontsize',FontSize)
box off
set(gca,'TickDir','out')
title(['$\sigma_{\tau_m}$=0 ms'],'fontsize',FontSize,'Interpreter','latex')

subplot(122)
edges=-0.1:0.2/100:0.1;
[bincountsL2,edges] = histcounts(corr_std8,edges);
histogram('BinCounts', bincountsL2/sum(bincountsL2), 'BinEdges', edges,'FaceColor','black');
box off
set(gca,'TickDir','out')
title(['$\sigma_{\tau_m}$=8 ms'],'fontsize',FontSize,'Interpreter','latex')
xlabel('Spike count correlation','fontsize',FontSize)
ylabel('probability','fontsize',FontSize)

