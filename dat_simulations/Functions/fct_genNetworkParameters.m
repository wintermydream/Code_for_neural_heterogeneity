function fct_genNetworkParameters(Nx1, inhwidth, networkType, paraFile, inDegreeList)
%{
    Function to generate network parameters for various types of neural networks (random, spatial, lognormal).
    This function initializes network parameters, connection strengths, and neuron characteristics 
    based on the specified network type and other input parameters.

    Inputs:
    - Nx1: Dimension size for spatial networks.
    - inhwidth: Width for inhibitory connections.
    - networkType: Type of network to generate ('SpatialNetworks', 'RandNetworks', 'LognormalNetworks').
    - paraFile: Optional filename to save parameters.
    - inDegreeList: List of in-degrees for lognormal networks (used only if networkType is 'LognormalNetworks').

    Outputs:
    - Saves network parameters in a .mat file specified by paraFile.
%}

if nargin < 4
    % If no parameter file is provided, generate a default filename.
    paraFile = sprintf('dat_para%s_Nx=%d_inhwidth=%.3f.mat', networkType, Nx1, inhwidth);
end

% Check if parameter file already exists; if not, generate new parameters.
if (~exist(paraFile, 'file'))
    % Set random seed for reproducibility
    rng('default');
    Wseed = 100;
    rng(Wseed);


    % Define network dimensions
    Ne1 = 200;  % Number of excitatory neurons along one dimension
    Ni1 = 100;  % Number of inhibitory neurons along one dimension
    NNeure = Ne1 * Ne1;  % Total excitatory neurons
    NNeuri = Ni1 * Ni1;  % Total inhibitory neurons
    NNeur = NNeure + NNeuri;  % Total neurons in the network

    % Neuron parameters (membrane potential characteristics)
    current = [3, 2.3];  % external current
    vlb = [-100, -100];  % Lower bounds for potentials
    vth = [-50, -50];    % Threshold potentials
    vl = [-70, -70];     % Resting potentials (mV)
    vre = [-75, -70];    % Reversal potentials
    taudecay = [0.05, 0.05];  % Decay time constants
    taurise = [0.05, 0.05];    % Rise time constants

    % Connection probabilities for recurrent connections
    % all distances (scale the Gaussians with these numbers)
    % pab0 is the proportion of postsynaptic neurons in pop.
    % a that receive input from a randomly chosen presynaptic
    % neuron in pop. b
    pee0 = 0.0125;  
    pei0 = 0.05;    
    pie0 = 0.0125;  
    pii0 = 0.05;    

    % Connection strength scalings
    Jee = 0.1;      
    Jei = -0.25;    
    Jie = 0.2;      
    Jii = -0.25;    

    % Calculate number of outgoing connections based on probabilities
    Kee = pee0 * NNeure;  
    Kei = pei0 * NNeure;  
    Kie = pie0 * NNeuri;  
    Kii = pii0 * NNeuri;  

    clearvars pee0 pei0 pie0 pii0

    % Generate connectivity based on network type
    switch networkType
        case 'SpatialNetworks'
            sigmarece = 0.05;  % Standard deviation for excitatory connections
            sigmareci = inhwidth * sigmarece;  % for inhibitory connections
            % Width of recurrent connections
            sigmaee=sigmarece;
            sigmaie=sigmarece;
            sigmaei=sigmareci;
            sigmaii=sigmareci;
            % Generate spatial connectivity matrix
            [Wx,I0]=FunGenSpaceW2D(Ne1,Ni1,Kee,Kei,Kie,Kii,sigmaee,sigmaei,sigmaie,sigmaii,Wseed);            
        case 'RandNetworks'
            % Generate random connectivity matrix
            [Wx, I0] = FunGenRandW2D(Ne1, Ni1, Kee, Kei, Kie, Kii, Wseed);
        case 'LognormalNetworks'
            % Generate lognormal network connectivity
            [Wx, I0] = FunGenHeteNet(NNeure, NNeuri, Kee, Kei, Kie, Kii, inDegreeList);
    end

    clearvars Ne1 Ni1 Kee Kei Kie Kii sigmaee sigmaei sigmaie sigmaii

    % Initialize membrane potentials uniformly
    V0min = vre(1);
    V0max = vth(1);
    V0 = (V0max - V0min) .* rand(NNeur, 1) + V0min;  % Random initial membrane potentials

    clearvars V0min V0max

    % Parameters for spatial stimuli
    % Space mesh width (number of e and i groups will each be Nx1^2)
    % eg: allocate Neure=40000 exc neurons into Nx1^2 grids and allocate 
    % Neuri=10000 inh neurons into Nx1^2 grids
    StimSeed = 10;  % Seed for spatial stimulus generation
    sigmastim = 0.1;  % Spatial correlation size of stimulus
    NGroups = 2 * Nx1^2;  % Total number of stimulus groups
    Ne = Nx1^2;  % Number of excitatory neurons per group

    % Generate spatial stimulus patterns
    Ispace = FunWeightStimulus0(Nx1, sigmastim, StimSeed);
    Ispace = (1 / (std(Ispace(:)))) * [Ispace(:); Ispace(:)];  % Normalize stimulus

    % Create random noise for stimuli
    temp = randn(NGroups / 2, 1);
    Irandn = (1 / std(temp(:))) * [temp; temp];  % Normalize noise

    clearvars temp

    % Define refractory periods and connection matrices
    refractory = [ones(1, Ne) * 2, ones(1, NGroups - Ne) * 1];  % Refractory periods for excitatory and inhibitory neurons
    Jeem = Jee * ones(Ne, Ne);  % Excitatory-to-excitatory connection matrix
    Jeim = Jei * ones(Ne, NGroups - Ne);  
    Jiem = Jie * ones(NGroups - Ne, Ne);  
    Jiim = Jii * ones(NGroups - Ne, NGroups - Ne); 
    Jx = [Jeem, Jeim; Jiem, Jiim];  % Combined connection matrix

    clearvars Jeem Jeim Jiem Jiim Jee Jei Jie Jii

    % Construct spatial position indices for excitatory and inhibitory neurons
    NLoce = zeros(1, NNeure);  % Position indices for excitatory neurons
    NLoci = zeros(1, NNeur - NNeure);  % Position indices for inhibitory neurons

    % Assign positions for excitatory neurons in a 2D grid
    NGe = sqrt(Ne);  % Number of groups along one dimension for excitatory neurons
    NNeure1D = sqrt(NNeure);  % 1D representation of excitatory neurons
    StepSizeE = NNeure1D / NGe;  % Step size for positioning
    StepVecE = 0:StepSizeE:NNeure1D;  % Vector of step positions
    StepVecE = round(StepVecE);  % Round step positions to integers

    for i = 1:NGe  % Loop over X dimension for excitatory groups
        for j = 1:NGe  % Loop over Y dimension for excitatory groups
            for k = 1:(StepVecE(i + 1) - StepVecE(i))
                indices = ((StepVecE(i) + (k - 1)) * NNeure1D) + (StepVecE(j) + 1:StepVecE(j + 1));
                NLoce(indices) = (i - 1) * NGe + j - 1;  % Assign group ID for excitatory neurons
            end
        end
    end

    % Assign positions for inhibitory neurons in a similar manner
    NGi = sqrt(NGroups - Ne);  % Number of groups for inhibitory neurons
    NNeuri1D = sqrt(NNeur - NNeure);  % 1D representation of inhibitory neurons
    StepSizeI = NNeuri1D / NGi;  % Step size for positioning
    StepVecI = 0:StepSizeI:NNeuri1D;  % Vector of step positions
    StepVecI = round(StepVecI);  % Round step positions to integers

    for i = 1:NGi  % Loop over X dimension for inhibitory groups
        for j = 1:NGi  % Loop over Y dimension for inhibitory groups
            for k = 1:(StepVecI(i + 1) - StepVecI(i))
                indices = ((StepVecI(i) + (k - 1)) * NNeuri1D) + (StepVecI(j) + 1:StepVecI(j + 1));
                NLoci(indices) = (i - 1) * NGi + j - 1 + Ne;  % Assign group ID for inhibitory neurons
            end
        end
    end

    % Combine position indices for all neurons
    NLoc = [NLoce, NLoci];  
    % clear unused variables
    clearvars NLoce NLoci NGe NGi NNeure1D StepVecE StepVecI StepSizeE StepSizeI indices...
    % Save generated parameters to specified file
    save(paraFile);
end
