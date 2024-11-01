/*
The code is writen by winter-my-dream@hotmail.com
Comparing to the code LIFnet_Lreadout_oldgaussian_dupplicate.c, output of this code only includes readout signals

Oct 14rd 2022:
- applying the stimuli in the condition when the time point i satisfying (i%dtI<=dtIs)
- considering the filter of presynaptic stimuli, like f = f_target + tau_pre*df_target/dt;
Oct 24th 2022
- taum_inv[NLoc[j]];
Oct 25th 2022
int Nrt = (int)round(Nt / dtI);
int Nx = (int)(NLoc[Ne]+1);
int Nrl = Nrt*Nx;
*/

#include "mex.h"
#include "math.h"
#include "time.h"
#include "malloc.h"
#include "float.h"

#ifndef MPI
#    define MPI 3.14159265358979323846
#endif

// 2-d array for temporaly storing spikes in force learning
//int buffer[100][201];

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{


	int printfperiod, kk, Ni, Ne, N, j, j1, j2, k, i, Nt, m1, m2, maxns, ns, jj, ii;
	double *sx, *sx2, *s, *IsyneO, *IsyniO, *vO, *temp;
	double *v, *v0, *JnextE, *JnextI, *JnextX, *Iext, *xdecaye, *xdecayi, *xrisee, *xrisei, *taudecay, *taurise;
	double *taum_inv, *Vleak, *tref, *Vth, *Vre, *Vlb, *current, *Jx, *NLocIn;
	int *Wx, *I0;
	double dt, T;
	int postcell, NGroups, outputflag, Nw, dtIs, timeind, spaceind;

	// Total number of each type of neuron
	int Ne1 = 200;
	int Ni1 = 100;
	Ne = Ne1*Ne1;
	Ni = Ni1*Ni1;
	N = Ne + Ni;

	//================
	//* Import variables from matlab
	//================
	//W = mxGetPr(prhs[1]);
	temp = mxGetPr(prhs[0]);
	m1 = (int)mxGetM(prhs[0]);
	Nw = (int)mxGetN(prhs[0]);
	if (Nw == 1 && m1 != 1) {
		Nw = m1;
		m1 = 1;
	}
	if (Nw == 1 || m1 != 1)
		mexErrMsgTxt("Weight vector must be Nwx1 or 1xNw where Nw is numer of connections.");

	Wx = (int*)mxMalloc((Nw) * sizeof(int));
	for (j = 0; j < Nw; j++) {
		Wx[j] = (int)round(temp[j]);
	}

	temp = mxGetPr(prhs[1]);
	m1 = (int)mxGetM(prhs[1]);
	m2 = (int)mxGetN(prhs[1]);
	if (!((m1 == 1 && m2 == N + 1) || (m1 == N + 1 && m2 == 1)))
		mexErrMsgTxt("I0 should be (N+1)x1 or 1x(N+1).");
	I0 = (int*)mxMalloc((N + 1) * sizeof(int));
	for (j = 0; j < N + 1; j++) {
		I0[j] = (int)round(temp[j]);
		if (I0[j]<0 || I0[j]>Nw)
			mexErrMsgTxt("Bad index in I0");
		if (j > 0)
			if (I0[j] < I0[j - 1])
				mexErrMsgTxt("I0 should be non-decreasing");
	}
	if (I0[0] != 0 || I0[N] != Nw)
		mexErrMsgTxt("First element of I0 should be zero and last element should be Nw.");

	int argindex = 2;
	Vleak = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);

	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;

	// dividing neurons into N Groups for receiving heterogenious stimuli
	NGroups = ((int)mxGetScalar(prhs[argindex]));
	argindex = argindex + 1;


	Vth = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);
	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;

	Vre = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);
	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;

	Vlb = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);
	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;

	v0 = mxGetPr(prhs[argindex]);
	N = mxGetM(prhs[argindex]);
	m2 = mxGetN(prhs[argindex]);
	if (N == 1 && m2 != 1)
		N = m2;
	argindex = argindex + 1;


	T = mxGetScalar(prhs[argindex]);
	argindex = argindex + 1;

	dt = mxGetScalar(prhs[argindex]);
	argindex = argindex + 1;

	dtIs = ((int)mxGetScalar(prhs[argindex]));
	argindex = argindex + 1;

	maxns = ((int)mxGetScalar(prhs[argindex]));
	argindex = argindex + 1;

	// Considering different membraine time constant associated different groups
	taum_inv = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);

	if (m1*m2 != NGroups)
		mexErrMsgTxt("taum_inv should be NGroupsx1");
	argindex = argindex + 1;

	// Considering different refractory periods associated different groups
	double *refractory_in;
	refractory_in = mxGetPr(prhs[argindex]);
	m1 = (int)mxGetM(prhs[argindex]);
	m2 = (int)mxGetN(prhs[argindex]);
	if (!((m1 == 1 && m2 == NGroups) || (m1 == NGroups && m2 == 1)))
	{
		mexPrintf("\n Ref[1] : %f  \n", refractory_in[1]);
		mexEvalString("drawnow;");
		mexErrMsgTxt("refractory should be NGroupsx1 or 1xNGroups.");
	}
	int *refractory;
	// Convert refractory to int;
	refractory = (int*)malloc(NGroups * sizeof(int)); // refractory associated w/ each group
	for (int n = 0; n < NGroups; n++)
	{
		refractory[n] = (int)round(refractory_in[n] / dt); // cast to int
	}
	int *refractory_status;
	refractory_status = (int*)malloc(N * sizeof(int));
	for (int n = 0; n < N; n++) {
		refractory_status[n] = 0;
	}
	argindex = argindex + 1;


	// external constant current input to Exc and Inh
	current = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);
	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;


	// synaptic decay time constant
	taudecay = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);
	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;


	// synaptic rise time constant
	taurise = mxGetPr(prhs[argindex]);
	m1 = mxGetN(prhs[argindex]);
	m2 = mxGetM(prhs[argindex]);
	if (m1*m2 != 2)
		mexErrMsgTxt("All neuron parameters should be 2x1");
	argindex = argindex + 1;


	// weight of synaptic connections
	Jx = mxGetPr(prhs[argindex]);
	m1 = (int)mxGetM(prhs[argindex]);
	m2 = (int)mxGetN(prhs[argindex]);
	if (m1 != NGroups || m2 != NGroups)
		mexErrMsgTxt("Jx should be NGroups x NGroups.");
	argindex = argindex + 1;


	outputflag = (int)mxGetScalar(prhs[argindex]);
	argindex = argindex + 1;


	// NLoc allocated neurons to special position where recieve special stimulation
	NLocIn = mxGetPr(prhs[argindex]);
	m1 = (int)mxGetM(prhs[argindex]);
	m2 = (int)mxGetN(prhs[argindex]);
	if (!((m1 == 1 && m2 == N) || (m1 == N && m2 == 1)))
		mexErrMsgTxt("NLoc should be Nx1 or 1xN.");
	// Convert NLoc to int;
	int* NLoc = (int*)malloc(N * sizeof(int)); // records neuron location from 1 to N
	for (int n = 0; n < N; n++)
	{
		NLoc[n] = (int)NLocIn[n]; // cast to int
	}
	argindex = argindex + 1;


	// external time-varying driven signals

	double *group_timevar_signal, *levels_timevar_signal;
	int NumSignalLevels;
	group_timevar_signal = mxGetPr(prhs[argindex]);
	m1 = (int)mxGetM(prhs[argindex]);
	m2 = (int)mxGetN(prhs[argindex]);
	if (!(m1 == NGroups))
		mexErrMsgTxt("group_timevar_signal should be NGroupsxNLevels");
	NumSignalLevels = m2;
	argindex = argindex + 1;

	// levels_timevar_signal=0:dtI:(T-dtI);
	levels_timevar_signal = mxGetPr(prhs[argindex]);
	m1 = (int)mxGetM(prhs[argindex]);
	m2 = (int)mxGetN(prhs[argindex]);
	if (!((m1 == 1 && m2 == NumSignalLevels - 1) || (m1 == NumSignalLevels - 1 && m2 == 1)))
		mexErrMsgTxt("levels_timevar_signal should be 1x(NLevels-1) or (NLevels-1)x1");

	//bool jj_stim;
	int dtI = (int)((levels_timevar_signal[2] - levels_timevar_signal[1]) / dt);
	argindex = argindex + 1;

	// record relected neuron, NeurTrace is the indices of neurons
	int *NeurTrace;
	double *NeurTraceIn;
	int Nrecord = 1; // recorded number
	if (outputflag == 1) {
		NeurTraceIn = mxGetPr(prhs[argindex]);
		m1 = (int)mxGetM(prhs[argindex]);
		m2 = (int)mxGetN(prhs[argindex]);
		if (!((m1 == 1) || (m2 == 1)))
			mexErrMsgTxt("noisesd should be 1xNumber or Numberx1.");
		Nrecord = m1*m2; // number of neurons to trace
		mexPrintf("\nNrecord%d\n", Nrecord);
		//Convert NeurTrace to int
		NeurTrace = (int*)malloc(Nrecord * sizeof(int));
		for (i = 0; i < Nrecord; i++) {
			NeurTrace[i] = (int)NeurTraceIn[i]; // cast to int
		}
		argindex = argindex + 1;
	}

	//================
	//* Finished importing variables.
	//================

	// Numebr of time bins
	Nt = (int)round(T / dt);
	int Nrt = (int)round(Nt / dtI);
	int Nx = (int)(NLoc[Ne]);
	int Nx1 = (int)sqrt(Nx);

	//================
	// Allocate output vector
	//================
	//plhs[0] = mxCreateDoubleMatrix(Nrt, Nx, mxREAL);
	//sx = mxGetPr(plhs[0]);
	plhs[0] = mxCreateDoubleMatrix(2, maxns, mxREAL);
	s = mxGetPr(plhs[0]);

	if (outputflag == 1 && Nrecord > 0) {
		plhs[1] = mxCreateDoubleMatrix(Nrecord, Nt, mxREAL);
		IsyneO = mxGetPr(plhs[1]);
		plhs[2] = mxCreateDoubleMatrix(Nrecord, Nt, mxREAL);
		IsyniO = mxGetPr(plhs[2]);
		plhs[3] = mxCreateDoubleMatrix(Nrecord, Nt, mxREAL);
		vO = mxGetPr(plhs[3]);
	}

	//for (int i = 1; i < Nrt*Nx; i++) {
	//	sx[i] = 0;
	//}


	// Check for consistency with total number of neurons
	if (N != Ne + Ni)
		mexErrMsgTxt("Ne and/or Ni not consistent with size of V0");

	// Allocate local variables
	v = (double*)mxMalloc(N * sizeof(double));
	JnextE = (double*)mxMalloc(N * sizeof(double));
	JnextI = (double*)mxMalloc(N * sizeof(double));
	xdecaye = (double*)mxMalloc(N * sizeof(double));
	xdecayi = (double*)mxMalloc(N * sizeof(double));
	xrisee = (double*)mxMalloc(N * sizeof(double));
	xrisei = (double*)mxMalloc(N * sizeof(double));
	JnextX = (double*)mxMalloc(N * sizeof(double));
	Iext = (double*)mxMalloc(N * sizeof(double));

	//================
	//* Finished allocating variables
	//================

	// Inititalize variables
	for (j = 0; j < N; j++) {
		JnextE[j] = 0;
		JnextI[j] = 0;
		xdecaye[j] = 0;
		xdecayi[j] = 0;
		Iext[j] = 0;
		xrisee[j] = 0;
		xrisei[j] = 0;
		if (j < Ne) {
			JnextX[j] = current[0];
			v[j] = current[0] * dt + v0[j];
		}
		else {
			JnextX[j] = current[1];
			v[j] = current[1] * dt + v0[j];
		}
	}

	// Initialize number of spikes
	ns = 0;

	// Print portion complete every printperiod steps
	printfperiod = (int)(round(Nt / 10.0));
	mexPrintf("\nprintfperiod: %d\n", printfperiod);
	mexEvalString("drawnow;");

	//================
	// Time loop for simulation
	//================
	int SignalTimeLevel = 0;
	
	/*
	int irow, icol;
	double xloc, yloc;
	double yloc_test = 0;

	double *spaceind1, *spaceind2;
	plhs[2] = mxCreateDoubleMatrix(1, Ne, mxREAL);
	spaceind1 = mxGetPr(plhs[2]);
	plhs[3] = mxCreateDoubleMatrix(1, Ne, mxREAL);
	spaceind2 = mxGetPr(plhs[3]);

	
	for (int j = 0; j < Ne; j++) {
		spaceind1[j] = NLoc[j];

		xloc = (double)(j%Ne1 + 1) / Ne1 - FLT_EPSILON;
		yloc = (double)((j - j%Ne1) / Ne1 + 1) / Ne1 - FLT_EPSILON;
		spaceind2[j] = ((int)floor(xloc * Nx1))*Nx1 + (int)floor(yloc * Nx1);
	}
	*/

	// Exit loop and issue a warning if max number of spikes is exceeded 
	for (i = 1; i < Nt && ns < maxns; i++) {

		if (SignalTimeLevel < NumSignalLevels - 1)
		{
			if (levels_timevar_signal[SignalTimeLevel] <= i*dt)
			{
				SignalTimeLevel++; // Assumed one timestep will not cover multiple levels
			}
		}
		//jj_stim = (i%dtI==0)

		for (j = 0; j < N; j++) {
			// Update the synaptic variables
			xdecaye[j] -= (xdecaye[j] - xrisee[j])*(dt / taudecay[0]);
			xdecayi[j] -= (xdecayi[j] - xrisei[j])*(dt / taudecay[1]);
			xrisee[j] -= xrisee[j] * (dt / taurise[0]);
			xrisei[j] -= xrisei[j] * (dt / taurise[1]);

			// Update membrane potential
			if (refractory_status[j] > 0) {
				refractory_status[j] = refractory_status[j] - 1;
			}
			else {
				if (j < Ne) {
					v[j] += fmax((group_timevar_signal[SignalTimeLevel*NGroups + NLoc[j]] * (i%dtI <= dtIs) + \
						xdecaye[j] + xdecayi[j] + Iext[j] - taum_inv[NLoc[j]] * (v[j] - Vleak[0]))*dt, \
						Vlb[0] - v[j]);

					// If a spike occurs
					if (v[j] >= Vth[0] && ns < maxns) {
						refractory_status[j] = refractory[NLoc[j]]; // add refractory period
						v[j] = Vre[0];       // reset membrane potential

						s[0 + 2 * ns] = i*dt; // spike time
						s[1 + 2 * ns] = j + 1;  // neuron index		
                        
						timeind = (int)floor((i*dt - FLT_EPSILON) / (dtI*dt));
						spaceind = NLoc[j];

						//xloc = (double)(j%Ne1+1)/Ne1-FLT_EPSILON;
						//yloc = (double)((j - j%Ne1)/Ne1 + 1)/Ne1-FLT_EPSILON;
						//spaceind = ((int)floor(xloc * Nx1))*Nx1 + (int)floor(yloc * Nx1);
						//if(spaceindtest<spaceind)
						// spaceindtest=spaceind;						
						//sx[spaceind*Nrt + timeind] += 1;

						//irow=(int)(j/Ne1/4);
						//icol=(int)(j%Ne1/4);
						//sx2[(irow*50+icol)+Nx*(int)floor(i/dtI)] += 1;

						ns++;// update total number of spikes

							 // For each postsynaptic target, propagate spike into JnextE
						for (k = I0[j]; k < I0[j + 1]; k++) {
							postcell = ((int)round(Wx[k])); // the postsynaptic cell index
							JnextE[postcell] += Jx[NLoc[j] * NGroups + NLoc[postcell]];
						}

					}
				}
				else { // If cell is inhibitory
					v[j] += fmax((group_timevar_signal[SignalTimeLevel*NGroups + NLoc[j]] * (i%dtI <= dtIs) + \
						xdecaye[j] + xdecayi[j] + Iext[j] - taum_inv[NLoc[j]] * (v[j] - Vleak[1]))*dt, \
						Vlb[1] - v[j]);

					// If a spike occurs
					if (v[j] >= Vth[1] && ns < maxns) {
						refractory_status[j] = refractory[NLoc[j]]; // add refractory period
						v[j] = Vre[1];       // reset membrane potential
                        
						s[0 + 2 * ns] = i*dt; // spike time
						s[1 + 2 * ns] = j + 1;  // neuron index												
						ns++;// update total number of spikes

							 // For each postsynaptic target, propagate spike into JnextE
						for (k = I0[j]; k < I0[j + 1]; k++) {
							postcell = ((int)round(Wx[k])); // the postsynaptic cell index
							JnextI[postcell] += Jx[NLoc[j] * NGroups + NLoc[postcell]];
						}

					}
				}
			}

		}

		// Use Jnext vectors to update synaptic variables
		for (j = 0; j < N; j++) {
			xrisee[j] += JnextE[j] / taurise[0];
			xrisei[j] += JnextI[j] / taurise[1];
			Iext[j] = JnextX[j];
			JnextE[j] = 0;
			JnextI[j] = 0;
		}

		// Store recorded variables
		if (outputflag == 1 && Nrecord > 0) {
			for (j = 0; j < Nrecord; j++) {
				IsyneO[j + Nrecord*i] = xdecaye[NeurTrace[j]] + JnextX[NeurTrace[j]];
				IsyniO[j + Nrecord*i] = xdecayi[NeurTrace[j]];
				vO[j + Nrecord*i] = v[NeurTrace[j]];
			}
		}

		// Print percent complete every printfperiod time steps
		//* This might not actually print until the full simulation
		//* is complete due to how some versions of Matlab treat the
		//* drawnow signal coming from a mex file

		if (i%printfperiod == 0) {
		mexPrintf("\n%d percent complete  rate = %2.2fHz", i * 100 / Nt, 1000 * ((double)(ns)) / (((double)(N))*((double)(i))*dt));
		mexEvalString("drawnow;");
		}

	}
	//mexPrintf("spaceindtest: %d\n", spaceindtest);
	//mexEvalString("drawnow;");

	// Issue a warning if max number of spikes reached //
	if (ns >= maxns)
		mexWarnMsgTxt("Maximum number of spikes reached, simulation terminated.");

	// Free allocated memory //
	mxFree(v);
	mxFree(JnextE);
	mxFree(JnextI);
	mxFree(xdecaye);
	mxFree(xdecayi);
	mxFree(xrisee);
	mxFree(xrisei);
	mxFree(JnextX);
	mxFree(Iext);

	mxFree(Wx);
	mxFree(I0);

	free(refractory);
	free(refractory_status);
	free(NLoc);
	if (outputflag == 1) {
		free(NeurTrace);
	}
}
