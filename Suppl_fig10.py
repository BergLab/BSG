"""
Script to generate panels Supplementary figure 10

First run script for main Fig 5 which sets up *.pkl files used by this script

Henrik Linden 2022
Rune Berg Lab, University of Copenhagen
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
from spinalsim import *
from astropy.stats.circstats import circcorrcoef,circmean
plt.close('all')
plt.ion()

### Set up simulation
neuron_params = {
    'tau_V' : 50.,
    'seed' : 10,
    'stim_func' : stim_const_common,
	'gain_func' : gain_const_common,
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

orig_gain = 1.1

with open('switch_neurons.pkl','rb') as file:
    switch_neurons = pickle.load(file)
with open('weights.pkl','rb') as file:
    weights = pickle.load(file)
with open('pocket_gain.pkl','rb') as file:
    pocket_gain = pickle.load(file)
with open('rostral_gain.pkl','rb') as file:
    rostral_gain = pickle.load(file)

np.random.seed(1)
cut_time=1000
n_trials = 5
gain_list = []
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(n_trials,6)


ax_gain = {}
for i in range(n_trials):
    ax_gain[i] = fig.add_subplot(gs[i,0])
    if i >0:
        ax_gain[i].sharey(ax_gain[0])

ax_rates = {}
for i in range(n_trials):
    ax_rates[i] = fig.add_subplot(gs[i,1:3])
    
ax_pca = {}
for i in range(n_trials):
    ax_pca[i] = fig.add_subplot(gs[i,3])
    if i >0:
        ax_pca[i].sharey(ax_pca[0])

ax_nerves = {}
for i in range(n_trials):
    ax_nerves[i] = fig.add_subplot(gs[i,4:6])
    if i >0:
        ax_nerves[i].sharey(ax_nerves[0])

for trial in range(n_trials):
    gain = orig_gain*np.ones(N)

    if trial>0:
        gain[switch_neurons] = orig_gain+np.random.normal(0,.5,size=len(switch_neurons))
    gain_list.append(gain)
    neuron_params['gain_func']=gain_const_individual
    neuron_params['gain'] = gain
    neuron_params['seed'] = trial

    R = simulate_network(W,**neuron_params)

    if trial==0:
        PCs,proj_PC_ref,var_exp = calc_PCA(R[:,cut_time:])

    R_norm = (R-R.mean(axis=1,keepdims=True))
    proj_PC =np.dot(PCs.T,R_norm)

    nerve = {}
    for key in ['hf','he','kf','ke']:
        nerve[key] = calc_nerve_output(R,weights[key],seed=neuron_params['seed'],bg_noise = 0.)

    ax_gain[trial].plot(gain)
    ax_rates[trial].imshow(R[phase_sort],interpolation=None,vmin=10,vmax=50)
    ax_rates[trial].axis('tight')
    ax_pca[trial].plot(proj_PC[0],proj_PC[1])
    ax_pca[trial].axis('equal')
    ax_pca[trial].axis('off')
    ax_nerves[trial].plot(nerve['hf']['output'],'.5')
    ax_nerves[trial].plot(2+nerve['he']['output'],'.5')
    ax_nerves[trial].plot(4+nerve['kf']['output'],'.5')
    ax_nerves[trial].plot(6+nerve['ke']['output'],'.5')
    ax_nerves[trial].axis('off')

    ax_gain[trial].set_xlabel('Neuron')
    ax_gain[trial].set_ylabel('Gain')
    ax_rates[trial].set_xlabel('Time (ms)')
    ax_rates[trial].set_ylabel('Neuron')

ax_gain[0].set_title('Activation profile')
ax_rates[0].set_title('Sorted firing rates')
ax_nerves[0].set_title('Motor nerve activity')
ax_pca[0].set_title('PC space')
        
plt.tight_layout()
plt.savefig('Suppl_fig10.pdf')
