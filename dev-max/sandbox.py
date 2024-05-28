import numpy as np
import matplotlib.pyplot as plt
from plot_misc import plot_sol_qw, plot_input
from guidance_torque import RPO_Detumble3DEnv



data = np.load("/home/cyrus/cleanrl/dev-max/traj_data.npy",allow_pickle=True).item()
# t   = data["t"]     # 1 x n_time
# rtn = data["rtn"]   # 6 x n_time
# qw  = data["qw"]    # 7 x n_time
# qw = np.zeros(qw.shape)
# qw[0,:] = 1

# plot_sol_qw(qw,t)


env = RPO_Detumble3DEnv(None)
env.reset()

t = np.linspace(0, 100, 1000)
obslist = np.empty((99, 10))
actions = np.empty((99, 3))
for i in range(99):
    action =  np.random.rand(1)*5
    obs, reward, done, _, _, torque = env.step(action)
    obslist[i] = obs
    actions[i] = torque
    print(obs, reward, done)
    if done:
        break


#PLOT output
plot_sol_qw(obslist[:,0:7].T, range(99), env.J_targ)
plot_input(actions.T,range(99))
