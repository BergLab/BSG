"""
Script to generate panels fo Figure 4

Henrik Linden 2022
Rune Berg Lab, University of Copenhagen
"""


import numpy as np
import matplotlib.pyplot as plt
from spinalsim import *
from astropy.stats.circstats import circcorrcoef,circmean

plt.close('all')
plt.ion()

### Set up simulation
neuron_params = {
    'tau_V' : 50.,
    'seed' : 6,
    'stim_func' : stim_const_common,
	'gain_func' : gain_const_individual,
	'I_e':20.,
    'noise_ampl' : 4.,
    'f_V_func':tanh_f_V,
    'threshold':20.,
    'gain': 1.2,
    'fmax' : 50.,
    'V_init':20.,
    't_steps' : 10000,
    }

conn_params = {
    'N' : 200,
    'C' : .1,
    'frac_inh':.5,
    'g':1.,
    'W_radius':1.0,
    'seed':1,
}

# Network connectivity
N = conn_params['N']
N_exc = int(N*(1.-conn_params['frac_inh']))
W = generate_W_dale_by_radius(**conn_params)
eigvals,eigvecs = np.linalg.eig(W)

max_idx = eigvals.real.argmax()
osc_mode = np.conj(eigvecs[:,max_idx])

neuron_phase = np.angle(osc_mode)
neuron_ampl = np.abs(osc_mode)
phase_sort = np.argsort(neuron_phase)

neuron_params['gain'] = 1.2*np.ones(N)
R = simulate_network(W,**neuron_params)
PCs,proj_PC,var_exp = calc_PCA(R)


# Find frequency modulation capacity of individual neurons

pos_mod = np.zeros(N)
neg_mod = np.zeros(N)

orig_gain = 1.2

print('Finding frequency modulation capacity')

for i in range(N):
    gain = orig_gain*np.ones(N)
    gain[i]+=.1
    
    mod_eigvals = np.linalg.eigvals(gain[:,None]*W)
    mod_imag = mod_eigvals.imag[mod_eigvals.real.argmax()]

    pos_mod[i] = mod_imag

    gain = orig_gain*np.ones(N)
    gain[i]-=.1
    
    mod_eigvals = np.linalg.eigvals(gain[:,None]*W)
    mod_imag = mod_eigvals.imag[mod_eigvals.real.argmax()]

    neg_mod[i] = mod_imag

mod_factor = pos_mod-neg_mod
mod_factor*=1./np.max(np.abs(mod_factor))

mod_sort = np.argsort(mod_factor)

# Plot Fig4 panel a

plt.figure(figsize=(4,3))
plt.fill_between(np.arange(N),mod_factor[mod_sort],color='.5',lw=2)
plt.plot(np.zeros(N),'k--',lw=1)
plt.xlabel('Sorted neurons')
plt.ylabel('Modulation capacity')
plt.tight_layout()
plt.savefig('Fig4_a.pdf')

# Simulate for three levels of gain modulation

gain_mod_vec = [-.15,0,.15]#np.arange(-.15,.16,.15)
n_gains = len(gain_mod_vec)

fig = plt.figure(figsize=(10,6))

for i,gain_mod in enumerate(gain_mod_vec):

    gain = orig_gain*np.ones(N)
    gain[mod_sort[0:int(.1*N)]] -= gain_mod
    gain[mod_sort[-int(.1*N):]] += gain_mod

    neuron_params['gain'] = gain
    
    R = simulate_network(W,**neuron_params)
    proj_PC = np.dot(PCs.T,R)

    flexor = simple_readout(R,neuron_phase,N_exc,exc_phase=np.pi/2,inh_phase=-np.pi/2.,exc_spread=np.pi/8.,inh_spread=np.pi/8.)
    extensor = simple_readout(R,neuron_phase,N_exc,exc_phase=0,inh_phase=np.pi,exc_spread=np.pi/8.,inh_spread=np.pi/8.)

    ax = fig.add_subplot(3,6,i*2+1)
    ax.fill_between(np.arange(N),neuron_params['gain'][np.argsort(mod_factor)],orig_gain*np.ones(N),color='.5',lw=2)
    ax.plot(orig_gain*np.ones(N),'k--',lw=1)
    ax.set_ylim([1.,1.4])
    ax.axis('off')

    ax = fig.add_subplot(3,6,i*2+2)
    ax.plot(proj_PC[0],proj_PC[1],'k')
    ax.axis('equal')
    ax.axis('off')
    
    ax = fig.add_subplot(3,3,i+4)
    ax.imshow(R[phase_sort],interpolation=None,vmin=10,vmax=50)
    ax.axis('tight')    
    ax.set_xticks([])
        
    ax = fig.add_subplot(3,3,i+7)
    ax.plot(flexor['nerve'])
    ax.plot(100+extensor['nerve'])    
    ax.set_ylim([-150,250])
    ax.axis('off')
     
plt.tight_layout()
plt.savefig('Fig4_cde.pdf')

# Vary degree of gain modulation

gain_mod_vec = np.arange(-.5,.7,.1)
osc_freq_vec_all = np.zeros_like(gain_mod_vec)
osc_freq_vec_exc = np.zeros_like(gain_mod_vec)
osc_freq_vec_inh = np.zeros_like(gain_mod_vec)

# Set longer simulation times to improve frequency resolution
neuron_params['t_steps']=40000

for i,gain_mod in enumerate(gain_mod_vec):

    gain = orig_gain*np.ones(N)
    gain[mod_sort[0:int(.1*N)]] -= gain_mod
    gain[mod_sort[-int(.1*N):]] += gain_mod

    neuron_params['gain'] = gain

    R = simulate_network(W,**neuron_params)
    freqvec,psd_vec,peak_idx,phase_at_peak,ampl_at_peak = calc_psd(R)

    osc_freq_vec_all[i] = freqvec[peak_idx]

for i,gain_mod in enumerate(gain_mod_vec):

    gain = orig_gain*np.ones(N)
    gain[mod_sort[0:int(.1*N)]] -= gain_mod
    gain[mod_sort[-int(.1*N):]] += gain_mod

    gain[int(N/2.)::]=orig_gain

    neuron_params['gain'] = gain

    R = simulate_network(W,**neuron_params)
    freqvec,psd_vec,peak_idx,phase_at_peak,ampl_at_peak = calc_psd(R)

    osc_freq_vec_exc[i] = freqvec[peak_idx]

for i,gain_mod in enumerate(gain_mod_vec):

    gain = orig_gain*np.ones(N)
    gain[mod_sort[0:int(.1*N)]] -= gain_mod
    gain[mod_sort[-int(.1*N):]] += gain_mod

    gain[0:int(N/2.)]=orig_gain

    neuron_params['gain'] = gain

    R = simulate_network(W,**neuron_params)
    freqvec,psd_vec,peak_idx,phase_at_peak,ampl_at_peak = calc_psd(R)

    osc_freq_vec_inh[i] = freqvec[peak_idx]

fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(1,3,1)
ax.plot(gain_mod_vec,osc_freq_vec_all,'.-',label='all')
ax.set_xlabel('Modulation index')
ax.set_ylabel('Frequency (Hz)')
ax.legend(loc='best',frameon=False)

ax = fig.add_subplot(1,3,2)
ax.hist(mod_factor[0:N_exc],20,label='exc')
ax.hist(mod_factor[N_exc:],20,label='inh')
ax.set_xlabel('Modulation capacity')
ax.set_ylabel('Count')
ax.legend(loc='best',frameon=False)

ax = fig.add_subplot(1,3,3)
ax.plot(gain_mod_vec,osc_freq_vec_exc,'.-',label='exc')
ax.plot(gain_mod_vec,osc_freq_vec_inh,'.-',label='inh')
ax.set_xlabel('Modulation index')
ax.set_ylabel('Frequency (Hz)')
ax.legend(loc='best',frameon=False)
plt.tight_layout()
plt.savefig('Fig4_fgh.pdf')

