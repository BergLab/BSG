import numpy as np
import matplotlib.pyplot as plt
from spinalsim import *
from astropy.stats.circstats import circcorrcoef,circmean
from figure_functions import *

plt.close('all')
plt.ion()

### Set up simulation
neuron_params = {
    'tau_V' : 50.,
    'seed' : 6,
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

# Run trials with varying gain

cut_time = 1000 

gain_vec = np.arange(.9,1.5,.05)
gain_vec_to_plot = copy.copy(gain_vec[1::2])

# gain_vec = gain_vec_to_plot

rate_rms_vec = np.zeros_like(gain_vec)
nerve_rms_vec = np.zeros_like(gain_vec)
PC_rms_vec = np.zeros_like(gain_vec)

fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(len(gain_vec_to_plot)+1,6)

ax_spectrum ={}
for i in range(len(gain_vec_to_plot)):
    ax_spectrum[i] = fig.add_subplot(gs[i,0])
    set_spines_at_zero(ax_spectrum[i])
    trim_spines(ax_spectrum[i])
    
ax_rates ={}
for i in range(len(gain_vec_to_plot)):
    ax_rates[i] = fig.add_subplot(gs[i,1:3])

ax_pca ={}
for i in range(len(gain_vec_to_plot)):
    ax_pca[i] = fig.add_subplot(gs[i,3])

for i in range(len(gain_vec_to_plot)):
    ax_pca[i].sharex(ax_pca[len(gain_vec_to_plot)-1])

ax_nerves ={}
for i in range(len(gain_vec_to_plot)):
    ax_nerves[i] = fig.add_subplot(gs[i,4:6])

for i in range(len(gain_vec_to_plot)):
    ax_nerves[i].sharey(ax_nerves[len(gain_vec_to_plot)-1])

ax_aux = {}
for i in range(4):
    ax_aux[i] = fig.add_subplot(gs[len(gain_vec_to_plot),1+i])


# Reference simulation for PCA
neuron_params['gain'] = 1.2
neuron_params['seed'] = 0

R = simulate_network(W,**neuron_params)
PCs,proj_PC_ref,var_exp = calc_PCA(R[:,cut_time:])

for i,gain in enumerate(gain_vec):
    print(i)
    
    neuron_params['gain'] = gain
    neuron_params['seed'] = i
    
    R = simulate_network(W,**neuron_params)
    flexor = simple_readout(R,neuron_phase,N_exc,exc_phase=np.pi/2,inh_phase=-np.pi/2.,exc_spread=np.pi/8.,inh_spread=np.pi/8.)
    extensor = simple_readout(R,neuron_phase,N_exc,exc_phase=-np.pi/2,inh_phase=np.pi/2,exc_spread=np.pi/8.,inh_spread=np.pi/8.)

    R_norm = (R-R.mean(axis=1,keepdims=True))
    proj_PC =np.dot(PCs.T,R_norm)
    
    rate_rms_vec[i] = np.sqrt(np.mean(R_norm**2.))
    nerve_rms_vec[i] = np.sqrt(np.mean(flexor['nerve']**2.))
    PC_rms_vec[i] = np.sqrt(np.mean(proj_PC[0]**2.))
    
    if gain in gain_vec_to_plot:
        i_plot = int(np.where(gain_vec_to_plot==gain)[0])

        eigvals = np.linalg.eigvals(gain*W)

        ax_spectrum[i_plot].scatter(eigvals.real,eigvals.imag,c='.5',s=2)
        ax_spectrum[i_plot].plot([1.,1.],[-1.2,1.2],'k--')
        ax_spectrum[i_plot].set_yticklabels([])
        ax_spectrum[i_plot].set_xticklabels([])
        ax_spectrum[i_plot].set_xlim([-2.1,2.1])
        ax_spectrum[i_plot].axis('equal')
        ax_spectrum[i_plot].set_title('gain={:.2f}'.format(gain))
        
        ax_rates[i_plot].imshow(R[phase_sort],interpolation=None,vmin=10,vmax=50)
        ax_rates[i_plot].axis('tight')    
        ax_rates[i_plot].set_xticks([])

        ax_pca[i_plot].plot(proj_PC[0],proj_PC[1])
        ax_pca[i_plot].axis('equal')
        ax_pca[i_plot].axis('off')
        ax_pca[i_plot].set_xlim((1.5*min(proj_PC_ref[0]),1.5*max(proj_PC_ref[1])))
        ax_pca[i_plot].set_ylim((1.5*min(proj_PC_ref[0]),1.5*max(proj_PC_ref[1])))

        ax_nerves[i_plot].plot(flexor['nerve'],'.5')
        ax_nerves[i_plot].plot(50+extensor['nerve'],'.5')    
        ax_nerves[i_plot].set_xticks([])
        ax_nerves[i_plot].axis('off')


ax = ax_aux[0]
ax.plot(gain_vec,rate_rms_vec,'.')
ax.set_xlabel('gain')
ax.set_ylabel('rate (RMS)')

ax = ax_aux[1]
ax.plot(gain_vec,PC_rms_vec,'.')
ax.set_xlabel('gain')
ax.set_ylabel('PC1 (RMS)')

ax = ax_aux[2]
ax.plot(gain_vec,nerve_rms_vec,'.')
ax.set_xlabel('gain')
ax.set_ylabel('flexor nerve (RMS)')

ax = ax_aux[3]
ax.plot(PC_rms_vec,nerve_rms_vec,'.')
ax.set_xlabel('PC1 (RMS)')
ax.set_ylabel('flexor nerve (RMS)')

plt.tight_layout()

plt.savefig('Suppl_fig6.pdf')
