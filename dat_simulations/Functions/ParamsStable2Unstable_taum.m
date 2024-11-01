%{
This code generates network parameters based on predefined values for Nx1 and inhwidth.
Some parameters are adapted from the following paper:
Pyle R, Rosenbaum R. Spatiotemporal dynamics and reliable computations in 
recurrent spiking neural networks[J]. Physical review letters, 2017, 118(1): 018103.

Example usage:
    NxI1 = 50;
    inhwidth = 2;
%}

clearvars -except inhwidth NxI1 inhwidth_array
rng('default')
Wseed=100;
rng(Wseed);

current = [3, 2.3];
sigmarece=.05;
sigmareci=inhwidth*sigmarece;

% Number of neurons in network along each dimension (e.g., the exc network is Ne1xNe1)
Ne1=200;
Ni1 = 100;
NNeure=Ne1*Ne1;
NNeuri=Ni1*Ni1;
NNeur=NNeure+NNeuri;

% Neuron params
% taum_inv=[1/20 1/15];
vlb=[-100 -100];
vth=[-50 -50];
vl=[-70 -70]; % mV
vre=[-75 -70];
% refractory=[1 1];
taudecay=[.05 .05];
taurise=[.05 .05];

% Recurrent connection probabilities averaged over
% all distances (scale the Gaussians with these numbers)
% pab0 is the proportion of postsynaptic neurons in pop.
% a that receive input from a randomly chosen presynaptic
% neuron in pop. b
pee0=.0125;
pei0=.05;
pie0=.0125;
pii0=.05;

% Recurrent connection strength scalings
Jee=0.1;
Jei=-0.25;
Jie=0.2;
Jii=-0.25;
% Jx = [Jee Jei;Jie Jii];

% Number of outgoing connections
Kee=pee0*NNeure;
Kei=pei0*NNeure;
Kie=pie0*NNeuri;
Kii=pii0*NNeuri;
clearvars pee0 pei0 pie0 pii0

% Width of recurrent connections
sigmaee=sigmarece;
sigmaie=sigmarece;
sigmaei=sigmareci;
sigmaii=sigmareci;
[Wx,I0]=FunGenSpaceW2D(Ne1,Ni1,Kee,Kei,Kie,Kii,sigmaee,sigmaei,sigmaie,sigmaii,Wseed);
clearvars Ne1 Ni1 Kee Kei Kie Kii sigmaee sigmaei sigmaie sigmaii

% Min and max for initial values of membrane potentials
V0min=vre(1);
V0max=vth(1);
V0=(V0max-V0min).*rand(NNeur,1)+V0min;
clearvars V0min V0max

% spatial stimuli
% Space mesh width (number of e and i groups will each be NxI1^2)
% eg: allocate Neure=40000 exc neurons into NxI1^2 grids and allocate Neuri=10000 inh neurons into NxI1^2 grids
StimSeed=10;
sigmastim=.1; % Spatial correlation size of stimulus (.2 most, .25 DmitriSim)
% NxI1=10;
NGroups = 2*NxI1^2;
Ne = NxI1^2;
% Generate spatial shape
% StimSeed=15;
% sigmastim=.1; % Spatial correlation size of stimulus (.2 most, .25 DmitriSim)
Ispace=FunWeightStimulus0(NxI1,sigmastim,StimSeed);
Ispace=(1/(std(Ispace(:))))*[Ispace(:); Ispace(:)];
temp=randn(NGroups/2, 1);
Irandn=(1/std(temp(:)))*[temp;temp];
clearvars temp;


refractory=[ones(1,Ne)*2 ones(1,NGroups-Ne)*1];
%{
flag_disp='lognormal';
para_mu=15;
para_std=10;
Trunc=[1,100];
[taum_inve]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,para_std,Trunc);
% para_mu=20;
% para_std=10;
% Trunc=[1,100];
% [taum_invi]=Fun_Hete_g.Hete_Net(Ne,flag_disp,para_mu,para_std,Trunc);
taum_inv=[1./taum_inve', 1./taum_inve'];
clearvars taum_inve  taum_invi
%}
Jeem = Jee * ones(Ne,Ne);
Jeim = Jei * ones(Ne, NGroups-Ne);
Jiem = Jie * ones(NGroups-Ne, Ne);
Jiim = Jii * ones(NGroups-Ne, NGroups-Ne);
Jx = [Jeem Jeim;Jiem Jiim];
clearvars Jeem Jeim Jiem Jiim Jee Jei Jie Jii

% construct group of position for spatial stimuli
NLoce = zeros(1,NNeure);
NLoci = zeros(1,NNeur-NNeure);
% E group (equal X,Y dim, Ne must be square)
NGe = sqrt(Ne); %50
NNeure1D = sqrt(NNeure); %200
StepSizeE = NNeure1D/NGe; %4
StepVecE = 0:StepSizeE:NNeure1D;
StepVecE = round(StepVecE); % cutoff points, include right endpoint but not left
for i = 1:NGe    % X Dim
    for j = 1:NGe % Y Dim
        for k = 1:(StepVecE(i+1)-StepVecE(i))
            indices = ((StepVecE(i) + (k-1)) * NNeure1D) + (StepVecE(j)+1:StepVecE(j+1));
            NLoce(indices) = (i-1)*NGe + j - 1; % Group ID - X*Group1DSize + Y
        end
    end
end
% I Group (equal X,Y dim N-Ne must be square)
NGi = sqrt(NGroups-Ne);
NNeuri1D = sqrt(NNeur-NNeure);
StepSizeI = NNeuri1D/NGi;
StepVecI = 0:StepSizeI:NNeuri1D;
StepVecI = round(StepVecI); % cutoff points, include right endpoint but not left
for i = 1:NGi    % X Dim
    for j = 1:NGi % Y Dim
        for k = 1:(StepVecI(i+1)-StepVecI(i))
            indices = ((StepVecI(i) + (k-1)) * NNeuri1D) + (StepVecI(j)+1:StepVecI(j+1));
            NLoci(indices) = (i-1)*NGi + j - 1 + Ne; % Group ID - X*Group1DSize + Y
        end
    end
end
NLoc = [NLoce, NLoci];
clearvars NLoce NLoci NGe NGi NNeure1D StepVecE StepVecI StepSizeE StepSizeI indices
paraFile=sprintf('ParamsStable2Unstable_Nx=%d_inhwidth=%.3f.mat',NxI1,inhwidth);
save(paraFile)