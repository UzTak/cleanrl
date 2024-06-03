"""
Plotting / Animation functions from the history of quaternion 
"""

import numpy as np
import matplotlib.pyplot as plt
# from dynamics_rot import * 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter, PillowWriter # Import FFMpegWriter
import matplotlib.animation as animation  



def plot_remaining_burns(num_burns, t):
    plt.figure()
    plt.plot(t, num_burns)
    plt.xlabel('time [s]')
    plt.ylabel('Remaining burns')
    plt.grid(True)
    # plt.savefig('/home/cyrus/AA203_project/src/LQRtest_remaining_burns.png', dpi = 600)
    # plt.show()


def plot_fuel_remaining(u_rem, t):
    plt.figure()
    plt.plot(t, u_rem)
    plt.xlabel('time [s]')
    plt.ylabel('Remaining fuel')
    plt.grid(True)
    # plt.savefig('/home/cyrus/AA203_project/src/LQRtest_fuel_remaining.png', dpi = 600)
    # plt.show()    


def plot_input(action, t):

    plt.figure()
    plt.scatter(t, action[0,:], c='r', label='u1')
    plt.scatter(t, action[1,:], c='g', label='u2')
    plt.scatter(t, action[2,:], c='b', label='u3')
    plt.xlabel('time [s]')
    plt.ylabel('w [rad/s]')
    plt.legend()
    plt.grid(True)
    # plt.savefig('/home/cyrus/AA203_project/src/LQRtest_inputs.png', dpi = 600)
    # plt.show()



def plot_attitude(ax, rtn, qw, height=20):
    """
    plot attiude in 3D (just one frame)
    """
    R_i = q2rotmat(qw[0:4])
    ex, ey, ez = R_i @ np.array([height, 0, 0]), R_i @ np.array([0, height, 0]), R_i @ np.array([0, 0, height])
    # plot axis
    ax.plot3D([rtn[1], rtn[1] + ex[1]], [rtn[2], rtn[2] + ex[2]], [rtn[0], rtn[0] + ex[0]], '-r', linewidth=2)
    ax.plot3D([rtn[1], rtn[1] + ey[1]], [rtn[2], rtn[2] + ey[2]], [rtn[0], rtn[0] + ey[0]], '-g', linewidth=2)
    ax.plot3D([rtn[1], rtn[1] + ez[1]], [rtn[2], rtn[2] + ez[2]], [rtn[0], rtn[0] + ez[0]], '-b', linewidth=2)


def plot_attitude_track(ax, rtn, qw, coneAngle, height=20):
    if rtn.shape[1] != qw.shape[1]:
        raise ValueError("rtn and qw have different length. Check the input variable")

    Nfreq = rtn.shape[1] // 10  # frequency of plotting axes
    N = rtn.shape[1]

    # Cone parameters
    radius = height * np.tan(coneAngle)
    # Meshgrid for polar coordinates
    r = np.linspace(0, radius, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    r, theta = np.meshgrid(r, theta)
    # Convert polar to cartesian coordinates
    coneX = r * np.cos(theta)
    coneY = r * np.sin(theta)
    coneZ = r * height / radius  # Scale Z based on height and radius
    coneX, coneY, coneZ = coneZ, coneX, coneY  # RTN trick
    
    for i in range(N):
        if i % Nfreq == 0 or i == N-1:
            R_i = q2rotmat(qw[:4, i])
            ex, ey, ez = R_i @ np.array([height, 0, 0]), R_i @ np.array([0, height, 0]), R_i @ np.array([0, 0, height])

            # plot axis
            ax.plot3D([rtn[1, i], rtn[1, i] + ex[1]], [rtn[2, i], rtn[2, i] + ex[2]], [rtn[0, i], rtn[0, i] + ex[0]], '-r', linewidth=2)
            ax.plot3D([rtn[1, i], rtn[1, i] + ey[1]], [rtn[2, i], rtn[2, i] + ey[2]], [rtn[0, i], rtn[0, i] + ey[0]], '-g', linewidth=2)
            ax.plot3D([rtn[1, i], rtn[1, i] + ez[1]], [rtn[2, i], rtn[2, i] + ez[2]], [rtn[0, i], rtn[0, i] + ez[0]], '-b', linewidth=2)

            # plot cone 
            # coneVertices = np.vstack([coneX.flatten(), coneY.flatten(), coneZ.flatten()])
            # coneVertices = R_i @ coneVertices 
            # coneVertices = coneVertices + rtn[:3, i].reshape(-1, 1)
            # coneXRotated = coneVertices[0, :].reshape(coneX.shape)
            # coneYRotated = coneVertices[1, :].reshape(coneY.shape)
            # coneZRotated = coneVertices[2, :].reshape(coneZ.shape)
            # ax.plot_surface(coneYRotated, coneZRotated, coneXRotated, color='red', alpha=0.2, linewidth=0, antialiased=False)


def animate_attitude(rtn,qw):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writer = PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    lines = [ax.plot([], [], [], color)[0] for color in ['r', 'g', 'b']]
    
    e_b = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        return lines
    
    def update(i):
        R_i = q2rotmat(qw[0:4,i])
        e_i = R_i @ e_b
        for k, color in enumerate(['r', 'g', 'b']):
            lines[k].set_data([rtn[1, i], rtn[1, i] + e_i[1,k]], 
                              [rtn[2, i], rtn[2, i] + e_i[2,k]])
            lines[k].set_3d_properties([rtn[0, i], rtn[0, i] + e_i[0,k]])
            lines[k].set_color(color)
        return lines
    
    ani = animation.FuncAnimation(fig, update, rtn.shape[1], init_func=init, blit=True, interval=10)
    ani.save('/home/cyrus/AA203_project/src/LQRtest.gif', writer=writer)
    plt.show()



def plot_sol_qw(sol, t, J):
    
    w_lb = -1e-1
    w_ub = 1e-1
    
    plt.figure(figsize=(10,8))
    for j in range(9):
        plt.subplot(4,3,j+1)
        
        qw = sol.T 
        # qw_ref = sol[i]["qw_ref"]
        # qw_cvx = sol[i]["qw"]
        # qw_nl  = sol[i]["qw_nl"]
        # dw     = sol[i]["dw"]
        # vqw    = sol[i]["vqw"]    
        
        if j < 7:  # qw
            plt.plot(t, qw[:,j], 'b')
            # plt.plot(t, qw_ref[j,:], '--ro', label='ref.')
            # plt.plot(t, qw_cvx[j,:], ':g.', label='cvx.')
            # plt.plot(t, qw_nl[j,:],  'b', label='nonlin.')
            
            # if j == 0: plt.legend()
            
        elif j == 7:
            plt.plot(t, np.sqrt(qw[:,0]**2 + qw[:,1]**2 + qw[:,2]**2 + qw[:,3]**2), 'b', label='|q|')
            # plt.plot(t, np.sqrt(qw_cvx[0,:]**2 + qw_cvx[1,:]**2 + qw_cvx[2,:]**2 + qw_cvx[3,:]**2), 'g', label='|q_rel|')
            # plt.plot(t, np.sqrt(qw_nl[0,:]**2 + qw_nl[1,:]**2 + qw_nl[2,:]**2 + qw_nl[3,:]**2), 'b', label='|q_rel|')

        elif j == 8: 
            plt.plot(t, np.sqrt((J[0,0]*qw[:,4])**2 + (J[1,1]*qw[:,5])**2 + (J[2,2]*qw[:,6])**2), 'b', label='|L|')


        if j == 0:
            plt.xlabel('time [s]')
            plt.ylabel('$q_1$')
            # plt.grid(True)
            # plt.xlim([-1,1])
            plt.ylim([-1,1])
        elif j == 1:
            plt.xlabel('time [s]')
            plt.ylabel('$q_2$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 2:
            plt.xlabel('time [s]')
            plt.ylabel('$q_3$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 3:
            plt.xlabel('time [s]')
            plt.ylabel('$q_4$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 4:
            plt.xlabel('time [s]')
            plt.ylabel('$w_1$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([w_lb,w_ub]) 
        elif j == 5:
            plt.xlabel('time [s]')
            plt.ylabel('$w_2$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([w_lb,w_ub]) 
        elif j == 6:
            plt.xlabel('time [s]')
            plt.ylabel('$w_3$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([w_lb,w_ub]) 
        elif j == 7:
            plt.xlabel('time [s]')
            plt.ylabel('$|q|$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([0.9,1.1])
        elif j == 8:
            plt.xlabel('time [s]')
            plt.ylabel('$|L|$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([0.9,1.1])
        elif j == 9:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_x$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([-1,1])
        elif j == 10:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_y$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([-1,1])
        elif j == 11:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_z$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            # plt.ylim([-1,1])
            
        plt.tight_layout()
        plt.legend()
        # plt.savefig('/home/cyrus/AA203_project/src/torquefree_states.png', dpi = 600)
    # plt.show()
        # fname = root_folder + '\\optimization\\saved_files\\plots\\scp\\iter' + str(i) + '.png'
        # plt.savefig('./rot_history.png', dpi = 600)