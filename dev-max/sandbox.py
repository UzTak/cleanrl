import numpy as np



data = np.load("/home/cyrus/cleanrl/dev-max/traj_data.npy",allow_pickle=True).item()
t   = data["t"]     # 1 x n_time
rtn = data["rtn"]   # 6 x n_time
qw  = data["qw"]    # 7 x n_time

