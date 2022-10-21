"""
Script to generate panels Supplementary figure 5

Henrik Linden 2022
Rune Berg Lab, University of Copenhagen
"""


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
    'seed' : 5,
    'stim_func' : stim_varying_common,
	'gain_func' : gain_const_common,
	'I_e':20.,
    'noise_ampl' : 4.,
    'f_V_func':tanh_f_V,
    'threshold':20.,
    'gain': 1.2,
    'fmax' : 50.,
    'V_init':0.,
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

# Generate network connectivity
N = conn_params['N']
N_exc = int(N*(1.-conn_params['frac_inh']))
W = generate_W_dale_by_radius(**conn_params)
eigvals,eigvecs = np.linalg.eig(W)

# Set up input vector
stim_start = 1000
stim_stop = 9000
I_e = neuron_params['threshold']
I_e_vec = np.zeros(neuron_params['t_steps'])
I_e_vec[stim_start:stim_stop] = neuron_params['I_e']
neuron_params['I_e'] = I_e_vec

# Simulate network activity
R = simulate_network(W,**neuron_params)


# Extended data Figure 5
max_idx = eigvals.real.argmax()
osc_mode = eigvecs[:,max_idx]
neuron_phase = np.angle(osc_mode)
neuron_ampl = np.abs(osc_mode)
phase_sort = np.argsort(neuron_phase)
exc_phase_sort = np.argsort(neuron_phase[0:N_exc])[::-1]
inh_phase_sort = np.argsort(neuron_phase[N_exc:])[::-1]

fig = plt.figure(figsize=(6,8))
gs = fig.add_gridspec(6,3)

ax_exc_rates = fig.add_subplot(gs[0,:])
ax_exc_imshow = fig.add_subplot(gs[1,:])

ax_inh_rates = fig.add_subplot(gs[2,:],sharey=ax_exc_rates)
ax_inh_imshow = fig.add_subplot(gs[3,:])

ax_all_phase = fig.add_subplot(gs[4,0])
ax_all_phase_linear = fig.add_subplot(gs[5,0])
trim_spines(ax_all_phase_linear)

ax_exc_phase = fig.add_subplot(gs[4,1],sharey=ax_all_phase)
ax_exc_phase_linear = fig.add_subplot(gs[5,1])
trim_spines(ax_exc_phase_linear)

ax_inh_phase = fig.add_subplot(gs[4,2],sharey=ax_all_phase)
ax_inh_phase_linear = fig.add_subplot(gs[5,2])
trim_spines(ax_inh_phase_linear)

# Plot excitatory rates
ax_exc_rates.set_title('Excitatory neurons')
ax_exc_rates.plot(R[0:N_exc:10].T,'.5')
# ax_exc_rates.plot(np.mean(R[0:N_exc],0),lw=2,c='cadetblue')
ax_exc_rates.set_xlim([0,neuron_params['t_steps']])
ax_exc_rates.set_ylabel('Rate')
ax_exc_rates.set_xlabel('Time (ms)')

ax_exc_imshow.imshow(R[0:N_exc][exc_phase_sort],interpolation=None,vmin=10,vmax=50)
ax_exc_imshow.axis('tight')    
ax_exc_imshow.set_xticks([])
ax_exc_imshow.set_ylabel('Neuron')

# Plot inhibitory rates
ax_inh_rates.set_title('Inhibitory neurons')
ax_inh_rates.plot(R[N_exc::10].T,'.5')
# ax_inh_rates.plot(np.mean(R[N_exc:],0),lw=2,c='orangered')
ax_inh_rates.set_xlim([0,neuron_params['t_steps']])
ax_inh_rates.set_ylabel('Rate')
ax_inh_rates.set_xlabel('Time (ms)')

ax_inh_imshow.imshow(R[N_exc:][inh_phase_sort],interpolation=None,vmin=10,vmax=50)
ax_inh_imshow.axis('tight')    
ax_inh_imshow.set_xticks([])
ax_inh_imshow.set_ylabel('Neuron')

# Plot phase distribution for all neurons
rmax = 1.*np.max(np.sqrt(osc_mode.real**2.+osc_mode.imag**2.))
x_circle = rmax*np.cos(np.linspace(-np.pi,np.pi,100))
y_circle = rmax*np.sin(np.linspace(-np.pi,np.pi,100))

ax_all_phase.set_title('All neurons')
ax_all_phase.scatter(osc_mode.real,osc_mode.imag,s=2.,c='k')
ax_all_phase.plot(x_circle,y_circle,lw=1,c='.5')
ax_all_phase.axis('equal')
ax_all_phase.axis('off')

ax_all_phase_linear.hist(neuron_phase,color='black',bins=np.linspace(-np.pi,np.pi,15))
ax_all_phase_linear.set_xlim([-np.pi,np.pi])
ax_all_phase_linear.set_xlabel('Phase')
ax_all_phase_linear.set_ylabel('Count')

# Plot phase distribution for excitatory neurons
ax_exc_phase.set_title('Excitatory')
ax_exc_phase.scatter(osc_mode.real[0:N_exc],osc_mode.imag[0:N_exc],s=2.,c='cadetblue')
ax_exc_phase.plot(x_circle,y_circle,lw=1,c='.5')
ax_exc_phase.axis('equal')
ax_exc_phase.axis('off')

ax_exc_phase_linear.hist(neuron_phase[0:N_exc],color='cadetblue',bins=np.linspace(-np.pi,np.pi,15))
ax_exc_phase_linear.set_xlim([-np.pi,np.pi])
ax_exc_phase_linear.set_xlabel('Phase')
ax_exc_phase_linear.set_ylabel('Count')

# Plot phase distribution for excitatory neurons
ax_inh_phase.set_title('Inhibitory')
ax_inh_phase.scatter(osc_mode.real[N_exc:],osc_mode.imag[N_exc:],s=2.,c='orangered')
ax_inh_phase.plot(x_circle,y_circle,lw=1,c='.5')
ax_inh_phase.axis('equal')
ax_inh_phase.axis('off')

ax_inh_phase_linear.hist(neuron_phase[N_exc:],color='orangered',bins=np.linspace(-np.pi,np.pi,15))
ax_inh_phase_linear.set_xlim([-np.pi,np.pi])
ax_inh_phase_linear.set_xlabel('Phase')
ax_inh_phase_linear.set_ylabel('Count')

plt.tight_layout()
plt.savefig('Suppl_fig5.pdf')