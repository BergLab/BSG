import numpy as np
import copy
from numba import njit
from itertools import product
from scipy.optimize import lsq_linear 
from astropy.stats.circstats import circcorrcoef,circmean,circstd
from scipy.integrate import solve_ivp


@njit(fastmath=True)
def tanh_f_V(V,threshold=20.,gain=1.,fmax=100.):
    if (V-threshold)<=0.:
        f = threshold*np.tanh(gain*(V-threshold)/threshold)
    elif (V-threshold)>0:
        f = fmax*np.tanh(gain*(V-threshold)/fmax)        
    return f+threshold

def calc_slope_at_f_V(V,eps,f_V_func,threshold,gain,fmax,**kwargs):
	f_plus = f_V_func(V+eps,threshold,gain,fmax)
	f_minus = f_V_func(V-eps,threshold,gain,fmax)
	return (f_plus-f_minus)/(2.*eps)


@njit(fastmath=True)
def stim_const_common(i,t,I_e):
    return I_e

@njit(fastmath=True)
def stim_const_individual(i,t,I_e):
    return I_e[i]

@njit(fastmath=True)
def stim_varying_common(i,t,I_e):
    return I_e[t]

@njit(fastmath=True)
def stim_varying_individual(i,t,I_e):
    return I_e[i,t]

@njit(fastmath=True)
def gain_const_common(i,t,gain):
    return gain

@njit(fastmath=True)
def gain_const_individual(i,t,gain):
    return gain[i]

@njit(fastmath=True)
def gain_varying_common(i,t,gain):
    return gain[t]

@njit(fastmath=True)
def gain_varying_individual(i,t,gain):
    return gain[i,t]

@njit(fastmath=True)
def simulate_network(W,tau_V,t_steps,\
                    noise_ampl,seed,stim_func,gain_func,I_e,V_init,\
                    f_V_func,threshold,gain,fmax):#,f_V_dict):

    np.random.seed(seed)

    N = W.shape[0]
    R = np.zeros((t_steps,N))
    V = np.zeros((N,t_steps))

    V[:,0] = V_init
    R[0,:] = V_init
    
    for t in range(1,t_steps):
        for i in range(N):
            I_rec = np.dot(W[i],R[t-1])
            I_noise = np.random.normal(loc=0.,scale=noise_ampl)
            I_tot = I_rec + I_noise + stim_func(i,t,I_e) #- w_A*A[i,t]
            dV = (1./tau_V)*(-V[i,t-1]+I_tot)            
            V[i,t] = V[i,t-1]+dV            
            R[t,i] = f_V_func(V[i,t],threshold,gain_func(i,t,gain),fmax)

    return R.T
    

def generate_W_dale_by_radius(N,C,frac_inh,g,W_radius,seed,**kwargs):
    W = np.zeros((N,N))
    N_inh = int(N*frac_inh)
    N_exc = N-N_inh
    n_exc = int(C*N_exc)
    n_inh = int(C*N_inh)

    exc_idx = np.arange(0,N_exc)
    inh_idx = np.arange(N_exc,N)

    w0 = W_radius/np.sqrt(C*(1.-C)*(1.+g**2.)/2.)
    w_e = w0/np.sqrt(1.*N)
    w_i = -g*w0/np.sqrt(1.*N)
 
    rng = np.random.default_rng(seed)  

    for i in range(N):
        sel_exc_idx = rng.choice(exc_idx,size=n_exc,replace=False)
        W[i,sel_exc_idx] = w_e
    
        sel_inh_idx = rng.choice(inh_idx,size=n_inh,replace=False)
        W[i,sel_inh_idx] = w_i

    return W


def get_weight_bounds(N,max_weight=1.,frac_inh=.5):
    lb = np.zeros(N)
    Ne = int(frac_inh*N)
    lb[0:Ne] = 0.
    lb[Ne::] = -max_weight
    ub = np.zeros(N)
    ub[0:Ne] = max_weight
    ub[Ne::] = 0.
    return (lb,ub)

def simple_readout(R,neuron_phase,N_exc,exc_phase,inh_phase,exc_spread,inh_spread):

    exc_neuron_phase = neuron_phase[:N_exc]
    inh_neuron_phase = neuron_phase[N_exc:]

    nerve = {}
    nerve['exc_phase'] = exc_phase
    nerve['inh_phase'] = inh_phase
    nerve['exc_spread'] = exc_spread
    nerve['inh_spread'] = inh_spread
    
    nerve['exc_idx'] = np.where(np.abs(np.angle(np.exp(1j*(exc_neuron_phase-nerve['exc_phase']))))<nerve['exc_spread'])[0]
    nerve['inh_idx'] = N_exc+np.where(np.abs(np.angle(np.exp(1j*(inh_neuron_phase-nerve['inh_phase']))))<nerve['inh_spread'])[0]

    nerve['exc_inp'] = np.mean(R[nerve['exc_idx']],0)
    nerve['inh_inp'] = np.mean(R[nerve['inh_idx']],0)
    nerve['tot_inp'] = nerve['exc_inp']-nerve['inh_inp']

    nerve['drive'] = copy.copy(nerve['tot_inp'])
    nerve['drive'][np.where(nerve['drive']<0.)] = 0.

    nerve['nerve'] = np.random.normal(loc=0,scale=nerve['drive'])

    return nerve

def get_readout_weights(R,target,max_iter=1000,max_weight=1.):
    N = R.shape[0]
    bounds = get_weight_bounds(N,max_weight)
    res = lsq_linear(R.T,target,bounds=bounds,verbose=1,max_iter=max_iter)
    return res['x']

def calc_nerve_output(R,weights,seed,bg_noise = 0.1):
    input = np.dot(weights,R)
    ampl = copy.copy(input)
    ampl[np.where(ampl<bg_noise)]=bg_noise
    rng = np.random.default_rng(seed)
    output = rng.normal(loc=0,scale=ampl)
    nerve = {}
    nerve['input'] = input
    nerve['ampl'] = ampl
    nerve['output'] = output
    return nerve


def calc_PCA(R,normalize_to_mean_std=False):
    M = copy.copy(R)
    if normalize_to_mean_std:
        M = (M-M.mean(axis=1,keepdims=True))/np.mean(np.std(M,1))
    else:
        M = (M-M.mean(axis=1,keepdims=True))

    cov_matrix = np.cov(M)
    eigvals,eigvecs = np.linalg.eig(cov_matrix)

    sort_idx = np.argsort(eigvals)[::-1]
    proj_PC =np.dot(eigvecs[:,sort_idx].T,M)

    tot = sum(eigvals)
    var_exp = [(i / tot)*100 for i in eigvals[sort_idx]]

    return eigvecs[:,sort_idx],proj_PC,var_exp
    
def calc_psd(R):
    fft_freqvec = np.fft.fftfreq(np.shape(R)[1],1.e-3)  
    fft_of_R = np.fft.fft(R,axis=1)
    fft_psd = np.mean(np.abs(fft_of_R)**2.,0)
    pos_idx = np.where(fft_freqvec>0)[0] 
    freqvec = fft_freqvec[pos_idx]
    psd_vec = fft_psd[pos_idx]
    peak_idx = psd_vec.argmax()
    phase_at_peak = np.angle(fft_of_R[:,pos_idx[peak_idx]])
    ampl_at_peak = np.abs(fft_of_R[:,pos_idx[peak_idx]])
    return freqvec,psd_vec,peak_idx,phase_at_peak,ampl_at_peak
