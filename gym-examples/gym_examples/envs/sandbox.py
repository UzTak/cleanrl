import numpy as np
import matplotlib.pyplot as plt
from plot_misc_envs import plot_sol_qw, plot_input, plot_remaining_burns, plot_fuel_remaining
from rpo_detumble3d_max import RPO_Detumble3DEnv


# data = np.load("/home/cyrus/cleanrl/dev-max/traj_data.npy",allow_pickle=True).item()
# t   = data["t"]     # 1 x n_time
# rtn = data["rtn"]   # 6 x n_time
# qw  = data["qw"]    # 7 x n_time
# qw = np.zeros(qw.shape)
# qw[0,:] = 1

# plot_sol_qw(qw,t)


env = RPO_Detumble3DEnv(None)
env.reset()

t = np.linspace(0, 100, 1000)
obslist = np.zeros((99, 10))
actions = np.zeros((99, 3))
u_rem = np.zeros((99, 1))
num_burns = np.empty((99, 1))
for i in range(99):
    # action =  np.random.rand()
    action = 1.0
    obs, reward, done, _, infos = env.step(action)
    obslist[i] = obs
    actions[i] = infos['T_I']
    num_burns[i] = infos['burns']
    u_rem[i] = infos['u_rem']
    print(obs, reward, done)
    if done:
        break
    # elif 

#PLOT output
plot_sol_qw(obslist[:,0:7].T, range(99), env.J_targ)
plot_input(actions.T,range(99))
plot_remaining_burns(num_burns, range(99))
plot_fuel_remaining(u_rem, range(99))
plt.show()
