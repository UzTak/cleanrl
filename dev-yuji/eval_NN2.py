import numpy as np
import matplotlib.pyplot as plt
import sys 


sys.path.append('C:/Users/yujit/github/cleanrl')
root_dir = 'C:/Users/yujit/github/cleanrl'
from gym_examples.envs.rpo_detumble3d import RPO_Detumble3DEnv
from gym_examples.envs.plot_misc_envs import plot_sol_qw, plot_input, plot_fuel_remaining


# data = np.load("/home/cyrus/cleanrl/dev-max/traj_data.npy",allow_pickle=True).item()
# t   = data["t"]     # 1 x n_time
# rtn = data["rtn"]   # 6 x n_time
# qw  = data["qw"]    # 7 x n_time
# qw = np.zeros(qw.shape)
# qw[0,:] = 1

# plot_sol_qw(qw,t)

env = RPO_Detumble3DEnv(None)
# env.reset()

sim_length = 800

t = np.linspace(0, sim_length, 1000)
obslist = np.zeros((sim_length, 10))
actions = np.zeros((sim_length, 3))
u_rem = np.zeros((sim_length, 1))
num_burns = np.zeros((sim_length, 1))

episodic_returns = []
all_obs = []
all_actions = []
for j in range(1):
    r=0
    rs = None
    obs = None
    dws = None
    actions = None
    t_step = None 
    env.reset()
    minutes = 0
    for i in range(sim_length):
        # action =  np.random.rand()
        action = 1.0
        next_obs, reward, done, _, infos = env.step(action)
        r+=reward
        # obslist[i] = next_obs
        # actions[i] = infos['T_I']
        # num_burns[i] = infos['burns']
        # u_rem[i] = infos['u_rem']

        # print(next_obs[8])
        # minutes+=next_obs[8]
        # print(f"num_burn;{next_obs[7]*env.num_burn_max}")

        if i != 0:
            rs = np.vstack((rs,reward))
            obs = np.vstack((obs,next_obs))
            dws = np.vstack((dws,infos['T_I']))
            actions = np.vstack((actions,infos['r_a']))
            norms = np.vstack((norms,infos['r_state_norm']))
            burns = np.vstack((burns,infos['r_burns']))
            t_step = np.vstack((t_step,infos['t_step']))
        else:
            rs = reward
            obs = next_obs
            dws = infos['T_I']
            actions = infos['r_a']
            norms = infos['r_state_norm']
            burns = infos['r_burns']
            t_step = infos['t_step']

        if done:
            print(f"terminated at step {i}")
            break
        # elif 
    episodic_returns.append(r)
    all_obs.append(obs)
    all_actions.append(dws)

#PLOT output
for i in range(1):#range(len(all_obs)):
    plot_sol_qw(all_obs[i][:,0:7].T, range(len(all_obs[i][:,0:7])), env.J_targ)
    # plot_total_burns(all_obs[i][:,7]*env.num_burn_max, range(len(all_obs[i][:,7])))
    # plt.plot(range(len(all_obs[i][:,7])), t_step/env.t_max * env.num_burn_max, "--." )
    # plt.show()

plt.figure()
plt.plot(rs,label='reward')
plt.plot(actions,label='actions')
plt.plot(norms,label='norms')
plt.plot(burns,label='burns')
plt.legend()
plt.show()

# plot_sol_qw(obslist[:,0:7].T, range(sim_length), env.J_targ)
# plt.show()
# plot_input(actions.T,range(22))
# plot_remaining_burns(num_burns, range(22))
# plot_fuel_remaining(u_rem, range(22))
