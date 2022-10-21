"""
Script to generate panels for Figure 2

Henrik Linden 2022
Rune Berg Lab, University of Copenhagen
"""


import numpy as np
import matplotlib.pyplot as plt
from spinalsim import *
from astropy.stats.circstats import circcorrcoef,circmean
plt.close('all')
plt.ion()

# Define neuronal parameters
neuron_params = {
    'tau_V' : 50., # combined neuronal and synaptic timescale (ms)
    'seed' : 5, # random seed for noise generator
    'stim_func' : stim_varying_common, #defines type of input I_e
	'gain_func' : gain_const_common, # defines type of gain
	'I_e':20., # external (synaptic) input
    'noise_ampl' : 4., # standard deviation of neuronal noise term
    'f_V_func':tanh_f_V, # firing rate function 
    'threshold':20., # threshold V_th (mV)
    'gain': 1.2, # default gain
    'fmax' : 50., # maximal firing, relative to threshold (1/s)
    'V_init':0., # initial value for membrane potential
    't_steps' : 10000, # simulation time (ms)
    }

# Define network connectivity parameters
conn_params = {
    'N' : 200, # number of neurons
    'C' : .1, # pair-wise connection probability
    'frac_inh':.5, # fraction of inhibitory neurons
    'g':1., # strength of inhibitory synaptic weight, relative to excitatory
    'W_radius':1.0, # expected spectral radius
    'seed':1, # random seed for generating network connectivity 
}

# Generate network connectivity
N = conn_params['N']
N_exc = int(N*(1.-conn_params['frac_inh']))
W = generate_W_dale_by_radius(**conn_params)
eigvals,eigvecs = np.linalg.eig(W)

# Calculate eigenvalue spectrum and dominant eigenmode
max_idx = eigvals.real.argmax()
osc_mode = np.conj(eigvecs[:,max_idx])
neuron_phase = np.angle(osc_mode)
neuron_ampl = np.abs(osc_mode)
phase_sort = np.argsort(neuron_phase)

# Set up input vector
stim_start = 1000
stim_stop = 9000
I_e = neuron_params['threshold']
I_e_vec = np.zeros(neuron_params['t_steps'])
I_e_vec[stim_start:stim_stop] = neuron_params['I_e']
neuron_params['I_e'] = I_e_vec

# Simulate network activity
R = simulate_network(W,**neuron_params)
PCs,proj_PC,var_exp = calc_PCA(R[:,stim_start:stim_stop])

# Calculate nerve readout
flexor = simple_readout(R,neuron_phase,N_exc,exc_phase=np.pi/2,inh_phase=-np.pi/2.,exc_spread=np.pi/8.,inh_spread=np.pi/8.)
extensor = simple_readout(R,neuron_phase,N_exc,exc_phase=-np.pi/2,inh_phase=np.pi/2,exc_spread=np.pi/8.,inh_spread=np.pi/8.)
tvec_s = np.linspace(0,neuron_params['t_steps']/1000,neuron_params['t_steps'])

### For transfer function illustration
V_vec = np.linspace(-10,30,100)
f_vec = np.zeros_like(V_vec)
f_vec2 = np.zeros_like(V_vec)

slope_vec =np.zeros_like(V_vec)
[f_V_func,threshold,gain,fmax] = [neuron_params[key] for key in ['f_V_func','threshold','gain','fmax']]

for i,V in enumerate(V_vec):
    f_vec[i] = f_V_func(V,threshold,gain,fmax)
    f_vec2[i] = f_V_func(V,threshold,1.5*gain,fmax)  
    slope_vec[i] = calc_slope_at_f_V(V,.1,f_V_func,threshold,gain,fmax)

slope_at_rest = calc_slope_at_f_V(0.,.1,f_V_func,threshold,gain,fmax)
slope_at_input = calc_slope_at_f_V(I_e,.1,f_V_func,threshold,gain,fmax)
slope_above_one = np.where(slope_vec>1.)[0]

#### Format Figure 2
from figure_functions import *

figsize = [8,6]    
fig = plt.figure(figsize=figsize)

n_rows = 4
n_cols = 3

pad_x = .06
pad_y = .035

tot_width_ax = (1.)/n_cols
tot_height_ax = (1.-pad_y)/n_rows
width_ax = tot_width_ax-1.5*pad_x
height_ax = tot_height_ax-2.*pad_y

ax_dict = {}
ax_dict['a']=fig.add_axes(rect=[pad_x,3.*tot_height_ax+2.*pad_y,width_ax,height_ax])
ax_dict['b']=fig.add_axes(rect=[tot_width_ax+pad_x,3.*tot_height_ax+2.*pad_y,width_ax,height_ax])
ax_dict['c']=fig.add_axes(rect=[2.*tot_width_ax+pad_x,3.*tot_height_ax+2.*pad_y,width_ax,height_ax])
ax_dict['d']=fig.add_axes(rect=[pad_x,2.*tot_height_ax+2.*pad_y,2.*(width_ax+pad_x),height_ax])
# ax_dict['e']=fig.add_axes(rect=[2.*tot_width_ax+pad_x,2.*tot_height_ax+2.*pad_y,width_ax,height_ax])
ax_dict['f']=fig.add_axes(rect=[pad_x,tot_height_ax+2.*pad_y,2.*(width_ax+pad_x),height_ax])
ax_dict['g']=fig.add_axes(rect=[2.*tot_width_ax+pad_x,tot_height_ax+2.*pad_y,width_ax,height_ax])#,projection='polar')
ax_dict['h']=fig.add_axes(rect=[pad_x,2.*pad_y,2.*(width_ax+pad_x),height_ax])
ax_dict['i']=fig.add_axes(rect=[2.*tot_width_ax+pad_x,2.*pad_y,width_ax,height_ax])#,projection='polar')

for key in ax_dict.keys():
    ax_dict[key].set_title(key,loc='left',fontweight='bold')

for key in ['b','c','d','h']:
    trim_spines(ax_dict[key])

for key in ['g']:
    remove_spines(ax_dict[key])

for key in ['b','c']:
    set_spines_at_zero(ax_dict[key])

#### Plot subpanels

ax = ax_dict['a']
ax.pcolor(W[::2,::2],cmap='bwr_r',alpha=.75)#,interpolation=None)
# ax.axis('equal')
ax.set_title('Connectivity')
ax.set_xticks([0,100])
ax.set_xticklabels(['0','N'])
ax.set_yticks([0,100])
ax.set_yticklabels(['0','N'])
# ax.axis('off')

ax = ax_dict['b']
ax.plot(V_vec,f_vec,c='k',lw=2)
ax.plot(V_vec,f_vec2,c='.5',lw=2)
ax.plot([0],[f_V_func(0,threshold,gain,fmax)],marker='o',c='.5')
ax.plot([I_e],[f_V_func(I_e,threshold,gain,fmax)],'ko')
ax.fill_between(V_vec,np.zeros(len(V_vec)),f_vec,where=slope_vec>1.,color='.9')
thresh_idx = np.min(np.where(slope_vec>1.)[0])
ax.plot([V_vec[thresh_idx],V_vec[thresh_idx]],[0,f_V_func(V_vec[thresh_idx],threshold,gain,fmax)],'k--')
ax.set_yticks([20])
ax.set_xlabel('Input')
ax.set_ylabel('Rate',labelpad=10)

ax = ax_dict['c']
ax.scatter(slope_at_rest*eigvals.real,slope_at_rest*eigvals.imag,s=2,c='.5')
ax.scatter(slope_at_input*eigvals.real,slope_at_input*eigvals.imag,s=2,c='.2')
ax.plot([1.,1.],[-1.2,1.2],'k--')
ax.set_yticklabels([])
ax.set_xlim([-2.1,2.1])
ax.axis('equal')
ax.set_xlabel('Real($\lambda$)',labelpad=20)
ax.set_ylabel('Imag($\lambda$)',labelpad=60)

ax = ax_dict['d']
ax.plot(tvec_s,R[::40].T,'.5')
ax.plot(tvec_s,40+.5*I_e_vec,ls='--',c='.5')
ax.set_xlim([0,tvec_s[-1]])
ax.set_ylabel('Rate')

ax = ax_dict['f']
ax.imshow(R[phase_sort],vmin=10,vmax=50.,origin='upper',interpolation=None)

ax.set_xticks([])
ax.set_yticks([0,N-1])
ax.set_yticklabels(['0',str(N)])
ax.axis('tight')
ax.set_ylabel('Neuron',labelpad=-5)

ax = ax_dict['g']
ax.plot(proj_PC[0],-proj_PC[1],c='k')
ax.axis('equal')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2',labelpad=-25)

ax = ax_dict['h']
ax.plot(tvec_s,extensor['nerve'],c='green',alpha=.7)
ax.plot(tvec_s,1.2*np.max(extensor['nerve'])+flexor['nerve'],c='blue',alpha=.7)
ax.set_xlim([0,tvec_s[-1]])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Nerve',labelpad=10)
ax.set_yticks([0,1.2*np.max(extensor['nerve'])])
ax.set_yticklabels(['2','1'])

ax = ax_dict['i']
rmax = .8*np.max(np.abs(osc_mode))
x_circle = rmax*np.cos(np.linspace(-np.pi,np.pi,100))
y_circle = rmax*np.sin(np.linspace(-np.pi,np.pi,100))
x_pie_flexor = 1.2*rmax*np.cos(np.linspace(flexor['exc_phase']-flexor['exc_spread'],flexor['exc_phase']+flexor['exc_spread'],100))
y_pie_flexor = 1.2*rmax*np.sin(np.linspace(flexor['exc_phase']-flexor['exc_spread'],flexor['exc_phase']+flexor['exc_spread'],100))
x_pie_extensor = 1.2*rmax*np.cos(np.linspace(extensor['exc_phase']-extensor['exc_spread'],extensor['exc_phase']+extensor['exc_spread'],100))
y_pie_extensor = 1.2*rmax*np.sin(np.linspace(extensor['exc_phase']-extensor['exc_spread'],extensor['exc_phase']+extensor['exc_spread'],100))
ax.fill_between(x_pie_flexor,np.r_[np.linspace(y_pie_flexor[0],0,50),np.linspace(0,y_pie_flexor[-1],50)],y_pie_flexor,color='blue',alpha=.2)
ax.fill_between(x_pie_extensor,np.r_[np.linspace(y_pie_extensor[0],0,50),np.linspace(0,y_pie_extensor[-1],50)],y_pie_extensor,color='green',alpha=.2)
ax.plot(x_circle,y_circle,c='.7',lw=1,ls='-')
ax.scatter(osc_mode.real,osc_mode.imag,s=2.,c='k')
ax.scatter(osc_mode.real[flexor['exc_idx']],osc_mode.imag[flexor['exc_idx']],s=2.,c='b')
ax.scatter(osc_mode.real[extensor['exc_idx']],osc_mode.imag[extensor['exc_idx']],s=2.,c='g')
ax.axis('equal')
ax.axis('off')

plt.savefig('Fig2.pdf')
