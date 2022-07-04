import numpy as np
import matplotlib.pyplot as plt
import pickle
from spinalsim import *
from astropy.stats.circstats import circcorrcoef,circmean
from scipy.io import savemat
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
    'gain': 1.1,
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

with open('weights.pkl','rb') as file:
    weights = pickle.load(file)
with open('pocket_gain.pkl','rb') as file:
    pocket_gain = pickle.load(file)
with open('rostral_gain.pkl','rb') as file:
    rostral_gain = pickle.load(file)

# reference simulation
neuron_params['gain_func'] = gain_const_individual
neuron_params['gain'] = pocket_gain
R = simulate_network(W,**neuron_params)
PCs,proj_PC,var_exp = calc_PCA(R)
nerve = {}
for key in ['hf','he','kf','ke']:
    nerve[key] = calc_nerve_output(R,weights[key],seed=neuron_params['seed'],bg_noise = 0.)


# Figure with panels a,b,c

np.random.seed(5)
noise = np.random.normal(0,1.,size=15000)
filter = np.exp(-np.arange(5001)/1000)
ou_noise = np.convolve(noise,filter,'valid')

neuron_params['stim_func']=stim_varying_common
neuron_params['I_e']=20.+2.*ou_noise/np.std(ou_noise)

R = simulate_network(W,**neuron_params)
proj_PC =np.dot(PCs.T,R)

nerve = {}
for key in ['hf','he','kf','ke']:
    nerve[key] = calc_nerve_output(R,weights[key],seed=neuron_params['seed'],bg_noise = 0.)

plt.figure(figsize=(6,6))
plt.subplot(421)
plt.plot(neuron_params['gain'])
plt.title('Gain')

plt.subplot(422)
plt.plot(proj_PC[0],proj_PC[1])
plt.axis('equal')
plt.title('PCA')
plt.axis('off')

plt.subplot(412)
plt.plot(neuron_params['I_e'],'.5')
plt.xlim(0,neuron_params['t_steps'])
plt.ylabel('Input')
plt.xticks()

plt.subplot(413)
plt.imshow(R[phase_sort],interpolation=None,vmin=10,vmax=50)
plt.axis('tight')
plt.ylabel('Neurons')
plt.xticks()

plt.subplot(414)
plt.ylabel('Nerves')
plt.plot(nerve['hf']['output'],'.5')
plt.plot(2+nerve['he']['output'],'.5')
plt.plot(4+nerve['kf']['output'],'.5')
plt.plot(6+nerve['ke']['output'],'.5')
plt.xlim(0,neuron_params['t_steps'])
plt.xlim([0,neuron_params['t_steps']])
plt.ylim([np.min(nerve['hf']['output']),6+np.max(nerve['ke']['output'])])
plt.xticks([1000*i for i in range(10)],[str(i) for i in range(10)])
plt.xlabel('Time (s)')
plt.tight_layout()

plt.savefig('Suppl_fig12_abc.pdf')
    
    
t_deletion = [[4773.,7734.],[7512.],[4595.],]
t_bursts = [[1650.,3050.,6090.,9460.],\
            [3170.,4610.,6030.,9460.],\
            [1600.,3060.,6120.,7900.,9320.]]


fig = plt.figure(figsize=(10,6))
from matplotlib.gridspec import GridSpec

gs = GridSpec(6, 6, figure=fig)
ax_imshow = {}
ax_input = {}
ax_nerves = {}
ax_pca = {}
ax_imshow[0] = fig.add_subplot(gs[1,0:3])
ax_imshow[1] = fig.add_subplot(gs[4,0:3])
ax_imshow[2] = fig.add_subplot(gs[1,3:])
ax_input[0] = fig.add_subplot(gs[0,0:3])
ax_input[1] = fig.add_subplot(gs[3,0:3])
ax_input[2] = fig.add_subplot(gs[0,3:])
ax_nerves[0] = fig.add_subplot(gs[2,0:3])
ax_nerves[1] = fig.add_subplot(gs[5,0:3])
ax_nerves[2] = fig.add_subplot(gs[2,3:])
ax_pca[0] = fig.add_subplot(gs[3:,3])
ax_pca[1] = fig.add_subplot(gs[3:,4])
ax_pca[2] = fig.add_subplot(gs[3:,5])

for i,noise_seed in enumerate([0,7,9]):

    np.random.seed(noise_seed)
    noise = np.random.normal(0,1.,size=15000)
    filter = np.exp(-np.arange(5001)/1000)
    ou_noise = np.convolve(noise,filter,'valid')

    neuron_params['stim_func']=stim_varying_common
    neuron_params['I_e']=20.+2.*ou_noise/np.std(ou_noise)
    
    R = simulate_network(W,**neuron_params)
    proj_PC =np.dot(PCs.T,R)

    nerve = {}
    for key in ['hf','he','kf','ke']:
        nerve[key] = calc_nerve_output(R,weights[key],seed=neuron_params['seed'],bg_noise = 0.)

    ax_input[i].plot(neuron_params['I_e'],'.5')
    ax_input[i].plot(20*np.ones(len(neuron_params['I_e'])),'k--')
    ax_input[i].axis('off')

    ax_imshow[i].imshow(R[phase_sort],interpolation=None,vmin=10,vmax=50)
    ax_imshow[i].axis('tight')
    ax_imshow[i].set_xticks([])
    ax_imshow[i].set_yticks([])
        
    ax_nerves[i].plot(nerve['hf']['output'],'.5')
    ax_nerves[i].plot(2+nerve['he']['output'],'.5')        
    ax_nerves[i].plot(4+nerve['kf']['output'],'.5')
    for t in t_bursts[i]:
        ax_nerves[i].plot([t],[4.],'bo')
    for t in t_deletion[i]:
        ax_nerves[i].plot([t],[4.],'ro')
    ax_nerves[i].plot(6+nerve['ke']['output'],'.5')
    ax_nerves[i].set_xlim(0,neuron_params['t_steps'])
    for j in range(len(t_deletion[i])):
        ax_nerves[i].plot([t_deletion[i][j]-500.,t_deletion[i][j]-500.],[-1.,7.],'r')
        ax_nerves[i].plot([t_deletion[i][j]+500.,t_deletion[i][j]+500.],[-1.,7.],'r')
    
    ax_nerves[i].axis('off')

    ax_pca[i].plot(proj_PC[0],proj_PC[1],'k')
    for j in range(len(t_deletion[i])):
        ax_pca[i].plot(proj_PC[0][int(t_deletion[i][j]-500):int(t_deletion[i][j]+500)],\
                        proj_PC[1][int(t_deletion[i][j]-500):int(t_deletion[i][j]+500)],'r')
        for t in t_deletion[i]:
            ax_pca[i].plot(proj_PC[0][int(t)],proj_PC[1][int(t)],'ro')
        for t in t_bursts[i]:
            ax_pca[i].plot(proj_PC[0][int(t)],proj_PC[1][int(t)],'bo')

            
        
    
    ax_pca[i].axis('equal')
    ax_pca[i].axis('off')

plt.tight_layout()
plt.savefig('Supple_fig12_defg.pdf')

