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


# Find the ability for each neuron to modulate phase distribution of population activity

orig_gain = 1.1

def find_switch_neurons(N,W,orig_gain=1.2,top_percent=10):
    phase_mod = np.zeros(N)

    for i in range(N):
        print(i)
        gain = orig_gain*np.ones(N)
        gain[i]+=.2
    
        mod_eigvals,mod_eigvecs = np.linalg.eig(gain[:,None]*W)
        max_idx = mod_eigvals.real.argmax()
        mod_osc_mode = np.conj(mod_eigvecs[:,max_idx])
        mod_neuron_phase = np.angle(mod_osc_mode)
        mod_neuron_ampl = np.abs(osc_mode)

        a = np.angle(np.exp(1j*(mod_neuron_phase-neuron_phase)))
        phase_mod[i] = circstd(a)

        
    switch_neurons = np.argsort(phase_mod)[-int(N*(.01*top_percent)):]
    return switch_neurons, phase_mod
    
switch_neurons,phase_mod = find_switch_neurons(N,W,orig_gain,top_percent=10)

pocket_gain = orig_gain*np.ones(N)
rostral_gain = orig_gain*np.ones(N)

pocket_gain[switch_neurons] = orig_gain+np.linspace(.3,-.3,len(switch_neurons))
rostral_gain[switch_neurons] = orig_gain+np.linspace(-.3,.3,len(switch_neurons))


# Generate training set for readout weights with alternating pocket and rostral scratch
R_dict = {}
target_dict = {}

n_trials = 4
label_list = ['pocket','rostral','pocket','rostral']
switch_list = [False,True,False,True]
phase_shift_list = [np.pi/6.,np.pi/6+np.pi,np.pi/6.,np.pi/6+np.pi]
hip_ampl_list = [.5,.2,.5,.2]
knee_ampl_list = [.5,.8,.5,.8]
seed_list = [1,2,3,4]
cut_idx = 2000

for trial in range(n_trials):
    print('trial',trial,' switch ',switch_list[trial])

    if switch_list[trial]:
        gain = rostral_gain
    else:
        gain = pocket_gain

    neuron_params['gain_func']=gain_const_individual
    neuron_params['gain'] = gain
    neuron_params['seed'] = seed_list[trial]

    R = simulate_network(W,**neuron_params)
    R[:,0] = R[:,1] # to avoid zeroes in circmean below
    R_dict[trial] = R[:,cut_idx:]

    population_phase = np.zeros(neuron_params['t_steps'])
    population_radius = np.zeros(neuron_params['t_steps'])
    for i in range(neuron_params['t_steps']):
        population_phase[i] = circmean(neuron_phase,weights=R[:,i])
        population_radius[i] = np.std(R[:,i])

    phase_shift = phase_shift_list[trial]

    target = {}
    target['hf'] = hip_ampl_list[trial]*np.cos(population_phase[cut_idx:])

    target['he'] = hip_ampl_list[trial]*np.cos(population_phase[cut_idx:]-np.pi)

    target['kf'] = knee_ampl_list[trial]*np.cos(population_phase[cut_idx:]-phase_shift)

    target['ke'] = knee_ampl_list[trial]*np.cos(population_phase[cut_idx:]-np.pi-phase_shift)

    target_dict[trial] = target


# Put trials together and optimize readout weights

R_combined = np.hstack([R_dict[trial] for trial in range(n_trials)])
target_combined = {}
weights = {}
nerve = {}
for key in ['hf','he','kf','ke']:
    target_combined[key] = np.hstack([target_dict[trial][key] for trial in range(n_trials)])
    weights[key] = get_readout_weights(R_combined,target_combined[key],max_iter=1000,max_weight=1.)
    

# Set up functions for generating the limb movement        
        
def get_foot_pos(l1,l2,hip_angle,knee_angle):
    x = (l1 * np.sin(hip_angle) - l2 * np.sin(hip_angle-knee_angle))
    y = (l1 * np.cos(hip_angle) - l2 * np.cos(hip_angle-knee_angle))
    return x,y

def get_knee_pos(l1,hip_angle):
    x = (l1 * np.sin(hip_angle))
    y = (l1 * np.cos(hip_angle))
    return x,y    

def calc_limb_angles(t_steps,tau_muscle,hip_weight,knee_weight,\
                    hf,he,kf,ke,\
                    hip_angle_start=np.pi/4.,knee_angle_start=np.pi/2.,\
                    hip_angle_limits=[0,np.pi],knee_angle_limits=[0,np.pi]):    

    hip_angle = np.zeros(t_steps)
    knee_angle = np.zeros(t_steps)

    hip_angle[0] = hip_angle_start
    knee_angle[0] =knee_angle_start

    leak_weight = 1.e-5

    for t in range(1,t_steps):
        d_ha = 1./tau_muscle*(hip_weight*(np.abs(hf[t])-np.abs(he[t]))-leak_weight*(hip_angle[t-1]-hip_angle_start))
        hip_angle[t] = hip_angle[t-1]+d_ha
        if hip_angle[t] < hip_angle_limits[0]:
            hip_angle[t]=hip_angle_limits[0]
        elif hip_angle[t] > hip_angle_limits[1]:
            hip_angle[t]=hip_angle_limits[1]

        d_ka = 1./tau_muscle*(hip_weight*(np.abs(kf[t])-np.abs(ke[t]))-leak_weight*(knee_angle[t-1]-knee_angle_start))
        knee_angle[t] = knee_angle[t-1]+d_ka      
        if knee_angle[t] < knee_angle_limits[0]:
            knee_angle[t]=knee_angle_limits[0]
        elif knee_angle[t] > knee_angle_limits[1]:
            knee_angle[t]=knee_angle_limits[1]

    return hip_angle,knee_angle

# Parameters for limb movement
l1 = 10
l2 = 10    
tau_muscle = .01
hip_weight = 1.e-4
knee_weight = 1.e-4

# Simulate two behaviours and plot results (Fig 5 d-e: firing rates, nerves, Fig 5 c: limb movements)

cut_start = 4500
cut_stop = 8000

# Pocket
neuron_params['t_steps'] = 10000
neuron_params['gain'] = pocket_gain
neuron_params['seed'] = 1004
R = simulate_network(W,**neuron_params)
print(neuron_params)
nerve = {}
for key in ['hf','he','kf','ke']:
    nerve[key] = calc_nerve_output(R,weights[key],seed=neuron_params['seed'],bg_noise = 0.)

hip_angle,knee_angle=calc_limb_angles(neuron_params['t_steps'],tau_muscle,hip_weight,knee_weight,\
                                        nerve['hf']['output'],\
                                        nerve['he']['output'],\
                                        nerve['kf']['output'],\
                                        nerve['ke']['output'])

plt.figure(1,figsize=(8,8))

plt.subplot(411)
plt.imshow(R[phase_sort])
plt.ylabel('Neuron')
plt.title('Pocket')
plt.axis('tight')
plt.xticks([1000*i for i in range(11)],[str(i) for i in range(11)])
plt.xlabel('Time (s)')

plt.subplot(412)
plt.plot(nerve['hf']['output'],'.5')
plt.text(0,.2,'Hip flexor')
plt.plot(2+nerve['kf']['output'],'.5')
plt.text(0,2.2,'Knee flexor')
plt.plot(4+nerve['ke']['output'],'.5')
plt.text(0,4.2,'Knee extensor')
plt.plot(6+nerve['he']['output'],'.5')
plt.text(0,6.2,'Hip extensor')
plt.xlim((0,10000))
plt.axis('off')

plt.figure(2,figsize=(4,4))

plt.subplot(222)
plt.plot(knee_angle[cut_start:cut_stop],label='knee')
plt.plot(hip_angle[cut_start:cut_stop],label='hip')
plt.legend(loc='upper right')
plt.ylim([0,np.pi])
plt.xlim([0,3000])
plt.ylabel('Angle (rad)')
plt.xticks([0,1000,2000],['1','2','3'])
plt.xlabel('Time (s)')

plt.subplot(221)
x_foot,y_foot = get_foot_pos(l1,l2,hip_angle,knee_angle)
plt.plot(x_foot[cut_start:cut_stop],y_foot[cut_start:cut_stop],'.2')

for idx in [4600,5500]:
    x_k,y_k = get_knee_pos(l1,hip_angle[idx])
    x_f,y_f = get_foot_pos(l1,l2,hip_angle[idx],knee_angle[idx])
    plt.plot([0,x_k],[0,y_k],'k-',lw=1)    
    plt.plot([x_k,x_f],[y_k,y_f],'k-',lw=1)

plt.axis('equal')
plt.axis('off')
plt.title('Pocket')


# Rostral
cut_start = 5000
cut_stop = 8000

neuron_params['t_steps'] = 10000
neuron_params['gain'] = rostral_gain
neuron_params['seed'] = 2004
print(neuron_params)
R = simulate_network(W,**neuron_params)
nerve = {}
for key in ['hf','he','kf','ke']:
    nerve[key] = calc_nerve_output(R,weights[key],seed=neuron_params['seed'],bg_noise = 0.)

hip_angle,knee_angle=calc_limb_angles(neuron_params['t_steps'],tau_muscle,hip_weight,knee_weight,\
                                        nerve['hf']['output'],\
                                        nerve['he']['output'],\
                                        nerve['kf']['output'],\
                                        nerve['ke']['output'])

plt.figure(1)

plt.subplot(413)
plt.imshow(R[phase_sort])
plt.ylabel('Neuron')
plt.title('Rostral')
plt.axis('tight')
plt.xticks([1000*i for i in range(11)],[str(i) for i in range(11)])
plt.xlabel('Time (s)')

plt.subplot(414)
plt.plot(nerve['hf']['output'],'.5')
plt.text(0,.2,'Hip flexor')
plt.plot(2+nerve['kf']['output'],'.5')
plt.text(0,2.2,'Knee flexor')
plt.plot(4+nerve['ke']['output'],'.5')
plt.text(0,4.2,'Knee extensor')
plt.plot(6+nerve['he']['output'],'.5')
plt.text(0,6.2,'Hip extensor')
plt.xlim((0,10000))
plt.axis('off')

plt.figure(2)
plt.subplot(224)
plt.plot(knee_angle[cut_start:cut_stop])
plt.plot(hip_angle[cut_start:cut_stop])
plt.ylim([0,np.pi])
plt.xlim([0,3000])
plt.ylabel('Angle (rad)')
plt.xticks([0,1000,2000],['1','2','3'])
plt.xlabel('Time (s)')

plt.subplot(223)
x_foot,y_foot = get_foot_pos(l1,l2,hip_angle,knee_angle)
plt.plot(x_foot[cut_start:cut_stop],y_foot[cut_start:cut_stop],'.2')

for idx in [2700,3100]:
    x_k,y_k = get_knee_pos(l1,hip_angle[idx])
    x_f,y_f = get_foot_pos(l1,l2,hip_angle[idx],knee_angle[idx])
    plt.plot([0,x_k],[0,y_k],'k-',lw=1)    
    plt.plot([x_k,x_f],[y_k,y_f],'k-',lw=1)

plt.axis('equal')
plt.axis('off')
plt.title('Rostral')

plt.figure(1)
plt.tight_layout()
plt.savefig('Fig5_de.pdf')

plt.figure(2)
plt.tight_layout()
plt.savefig('Fig5_c.pdf')

with open('switch_neurons.pkl','wb') as file:
    pickle.dump(switch_neurons,file)

with open('weights.pkl','wb') as file:
    pickle.dump(weights,file)

with open('pocket_gain.pkl','wb') as file:
    pickle.dump(pocket_gain,file)

with open('rostral_gain.pkl','wb') as file:
    pickle.dump(rostral_gain,file)

