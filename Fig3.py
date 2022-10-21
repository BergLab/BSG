"""
Script to generate panels fo Figure 3

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
    'seed' : 3,
    'stim_func' : stim_const_common,
	'gain_func' : gain_varying_common,
	'I_e':20.,
    'noise_ampl' : 4.,
    'f_V_func':tanh_f_V,
    'threshold':20.,
    'gain': 1.2,
    'fmax' : 50.,
    'V_init':0.,
    't_steps' : 30000,
    }

conn_params = {
    'N' : 200,
    'C' : .1,
    'frac_inh':.5,
    'g':1.,
    'W_radius':1.,
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

# Input
gain_vec = np.zeros(neuron_params['t_steps'])
gain_vec[0:10000] = 1.05#1.04#1.05
gain_vec[10000:20000] = 1.10#1.08#1.15
gain_vec[20000:] = 1.2#1.14#1.25

neuron_params['gain'] = gain_vec

# Simulation and analysis
R= simulate_network(W,**neuron_params)
PCs,proj_PC,var_exp = calc_PCA(R)

R[:,0] = R[:,1] # to avoid zeroes in circmean below
population_phase = np.zeros(neuron_params['t_steps'])
population_radius = np.zeros(neuron_params['t_steps'])
for i in range(neuron_params['t_steps']):
    population_phase[i] = circmean(neuron_phase,weights=R[:,i])
    population_radius[i] = np.std(R[:,i])

# Nerve readout
flexor = simple_readout(R,neuron_phase,N_exc,exc_phase=np.pi/2,inh_phase=-np.pi/2.,exc_spread=np.pi/4.,inh_spread=np.pi/4.)
extensor = simple_readout(R,neuron_phase,N_exc,exc_phase=-np.pi/2,inh_phase=np.pi/2,exc_spread=np.pi/4.,inh_spread=np.pi/4.)
tvec_s = np.linspace(0,neuron_params['t_steps']/1000,neuron_params['t_steps'])

RMS_periods = np.zeros(3)

RMS_periods[0] = np.sqrt(np.mean(flexor['drive'][3000:8000]**2.))
RMS_periods[1] = np.sqrt(np.mean(flexor['drive'][13000:18000]**2.))
RMS_periods[2] = np.sqrt(np.mean(flexor['drive'][23000:28000]**2.))

plt.figure(figsize=(10,4))
ax = plt.subplot(244)
ax.plot(proj_PC[0,3000:8000],proj_PC[1,3000:8000],'orangered')
ax.plot(proj_PC[0,13000:18000],proj_PC[1,13000:18000],'cadetblue')
ax.plot(proj_PC[0,23000:28000],proj_PC[1,23000:28000],'dimgray')
ax.axis('equal')
ax.axis('off')

ax = plt.subplot(248)
ax.bar(np.arange(3),RMS_periods)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['1','2','3'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Nerve (RMS)')

ax = plt.subplot(241)
ax.imshow(R[phase_sort,3000:8000],interpolation=None,vmin=10,vmax=50.)
ax.set_xticks([])
ax.axis('tight')
ax.set_ylabel('Neuron')
ax = plt.subplot(242)
ax.imshow(R[phase_sort,13000:18000],interpolation=None,vmin=10,vmax=50.)
ax.set_xticks([])
ax.axis('tight')
ax = plt.subplot(243)
ax.imshow(R[phase_sort,23000:28000],interpolation=None,vmin=10,vmax=50.)
ax.set_xticks([])
ax.axis('tight')

ax = plt.subplot(245)
ax.plot(tvec_s[3000:8000],extensor['nerve'][3000:8000],c='.5')#,alpha=.7)
ax.plot(tvec_s[3000:8000],1.2*np.max(extensor['nerve'])+flexor['nerve'][3000:8000],c='.5')#,alpha=.7)
ax.set_xlim([3.,8.])
ax.set_ylim([np.min(extensor['nerve']),1.2*np.max(extensor['nerve'])+np.max(flexor['nerve'])])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Nerve',labelpad=10)
ax.set_yticks([0,1.2*np.max(extensor['nerve'])])
ax.set_yticklabels(['2','1'])

ax = plt.subplot(246)
ax.plot(tvec_s[13000:18000],extensor['nerve'][13000:18000],c='.5')#,alpha=.7)
ax.plot(tvec_s[13000:18000],1.2*np.max(extensor['nerve'])+flexor['nerve'][13000:18000],c='.5')#,alpha=.7)
ax.set_xlim([13.,18.])
ax.set_ylim([np.min(extensor['nerve']),1.2*np.max(extensor['nerve'])+np.max(flexor['nerve'])])

ax.set_xlabel('Time (s)')
# ax.set_ylabel('Nerve',labelpad=10)
ax.set_yticks([0,1.2*np.max(extensor['nerve'])])
ax.set_yticklabels(['2','1'])

ax = plt.subplot(247)
ax.plot(tvec_s[23000:28000],extensor['nerve'][23000:28000],c='.5')#,alpha=.7)
ax.plot(tvec_s[23000:28000],1.2*np.max(extensor['nerve'])+flexor['nerve'][23000:28000],c='.5')#,alpha=.7)
ax.set_xlim([23.,28.])
ax.set_ylim([np.min(extensor['nerve']),1.2*np.max(extensor['nerve'])+np.max(flexor['nerve'])])
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Nerve',labelpad=10)
ax.set_yticks([0,1.2*np.max(extensor['nerve'])])
ax.set_yticklabels(['2','1'])


plt.tight_layout()
plt.savefig('Fig3.pdf')
