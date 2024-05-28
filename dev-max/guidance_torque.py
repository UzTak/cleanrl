"""
2D version of the detumbling problem. 
Assumption: 
- translational motion of the servicer and target is fixed (ignored)
- rotational motion of the servicer is fixed, reaction of the thrust is ignored 
- the target is a circular object (spherical) in a 2D plane, so attitude does not exist 
- only variable is angular velocity of the target
"""

import numpy as np
import numpy.linalg as la
from scipy.integrate import odeint

import gymnasium as gym
from gym import spaces


class RPO_Detumble2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        
        self.t = 0  
        self.dt = 10  # sec
        
        # target info
        self.oe0  = np.array([7000e3, 0.001, 0, 0, 0, 0])  # [m]
        self.oe   = self.oe0.copy()
        self.J_targ = np.diag([100,100,100])
        self.n    = np.sqrt(398600e9/self.oe[0]**3)
        
        self.d = 10 # distance between the target and the satellite (currently constant)
        self.r = 1  # radius of the target 
        
        # state = [q, w, ROE]
        self.state = np.zeros((13))
        
        # action = [u_mag, thetea]
        # self.action = np.empty((2))  
        # self.action_space = spaces.Box(np.array([0,1]),np.array([-np.pi/4,np.pi/4]),dtype=np.float32)        
        high = np.array([1,1,1,1, 1,1,1, 200, 200, 200, 200, 200, 200])
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)
        high = np.array([1,np.pi/2])
        low = np.array([0, -np.pi/2])
        self.action_space = spaces.Box(low, high, dtype = np.float64)
        self.umax = 1   # max output of the thrustr [unit?]

        # threshold of detumbling (TBD) rad/s
        self.w_tol = 1e-1
        
    # assuminng perfect observation as of now 
    def _get_obs(self):
        return self.state


    def _get_info(self):
        return {}
        
    def reset(self, seed=None, options=None):
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset environment state
        self.state = np.zeros((13))

        # Return initial observation
        # return self._get_obs
        return self.state, {}
    
    
    def step(self, action):

        qw = self.state[0:7]        
        T = a2t_laser_2d(action, self.d, self.r) 
        J = 1e-3
        # qw = odeint(ode_qw, qw, [0, self.dt], args=(T,))[1]
        qw = odeint(ode_qw, qw, [0, self.dt], args=(J, T))[1]
        self.state[3:6] = qw[4:7]   # only update angular velocity now... 
        
        self.t += self.dt 
        self.oe += np.array([0,0,0,0,0,self.n*self.dt])
        
        reward = -action[0]
        
        if (abs(self.state[3:6]) < self.w_tol).all():
            terminated = True
        else:
            terminated = False    
        
        info = {}
        
        return self.state, reward, terminated, False, info
    

class RPO_Detumble3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, size=5):
            data = np.load("/home/cyrus/cleanrl/dev-max/traj_data.npy",allow_pickle=True).item()
            # self.t   = data["t"]     # 1 x n_time
            self.rtn = data["rtn"]   # 6 x n_time
            self.qw_SC_RTN  = data["qw"]    # 7 x n_time
            self.qw_SC_RTN = np.zeros(self.qw_SC_RTN.shape)
            self.qw_SC_RTN[0,:] = 1

            # self.dt = 10  # sec
            
            # target info
            self.oe0  = np.array([7000e3, 0.001, 0, 0, 0, 0])  # [m]
            self.oe   = self.oe0.copy()
            self.J_targ = np.array([[562.07457,   0.     ,   0.0],
              [  0.     , 562.07465,   0.     ],
              [  0.0,   0.     , 192.29662]])
            
            self.n    = np.sqrt(398600e9/self.oe[0]**3)
            self.period = 2*np.pi/self.n
            
            #TODO: add the relative position of the target
            self.d = 10 # distance between the target and the satellite (currently constant)
            #TODO: add the relative position of the target
            self.r = 1  # radius of the target 
            

            # state = [q_rel, w_rel, num_burn, u_rem, t]
            self.state = np.zeros((10))
            
            # action = [u_mag]
            self.action = np.empty((1))  
            # self.action_space = spaces.Box(np.array([0,1]),dtype=np.float64)      

            self.w_max = 0.1
            self.num_burn_max = 10
            self.urem_max = 10
            self.K_max = 200
            self.num_rev = 20
            self.n_orbit = 100 # d_orbit
            self.freq_thrust = 25 # time steps between thrusts


            ub = np.array([1,1,1,1, self.w_max, self.w_max, self.w_max, self.num_burn_max, self.urem_max, self.num_rev])#TODO figure out nondimensional time, #revolutions?
            lb = np.array([-1,-1,-1,-1,-self.w_max,-self.w_max,-self.w_max, 0, 0.0, 0.0])
        
            self.observation_space = spaces.Box(lb, ub, dtype=np.float64)
            high = self.K_max #np.array([K_max])
            low = 0.0 #np.array([0])
            self.action_space = spaces.Box(low, high, dtype = np.float64)
            # self.umax = K_max*   # max output of the thrustr [unit?]

            # threshold of detumbling (TBD) rad/s
            self.w_tol = 1e-2
        
    # assuminng perfect observation as of now 
    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}
        
    def reset(self, seed=None, options=None):
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset environment state
        q_res = np.random.rand(4)
        q_res = q_res/la.norm(q_res)
        w_res = np.random.rand(3)*self.w_max
        self.state = np.hstack((q_res, w_res, 0, 1.0, 0))

        # Return initial observation
        # return self._get_obs
        return self.state, {}
    
    
    def step(self, action):        
        num_burn, u_rem, t = self.state[7:10]
        t = int(t)%self.n_orbit

        rtn_SC    = self.rtn[:,t]
        self.d = la.norm(rtn_SC[0:3])
        qw_RSO_SC = self.state[0:7]
        qw_SC_I   = self.qw_SC_RTN[:,t]
        q_RSO_I   = q_mul(q_conj(q_inv(qw_SC_I[0:4])),qw_RSO_SC[0:4])
        w_RSO_I   = q2rotmat(qw_SC_I[0:4]).T@qw_RSO_SC[4:7]        
        
        h_I = self.J_targ @ w_RSO_I
        
        T_I = a2t_laser_3D(action, h_I, self.d)
        dw_RSO_I  = T_I#la.inv(self.J_targ)@T_I
        w_RSO_I   = w_RSO_I + dw_RSO_I
        qw_RSO_I  = np.hstack((q_RSO_I, w_RSO_I))


        qw_RSO_I   = odeint(ode_qw, qw_RSO_I, [0, self.period/self.n_orbit*self.freq_thrust], args=(self.J_targ, np.zeros(T_I.shape)))[1]
        qw_SC_I    = self.qw_SC_RTN[:,t+1]     
        q_RSO_SC   = q_mul(q_conj(qw_SC_I[0:4]),qw_RSO_I[0:4])
        w_RSO_SC   = q2rotmat(qw_SC_I[0:4])@qw_RSO_I[4:7]
        self.state = np.hstack((q_RSO_SC,w_RSO_SC, num_burn+1, u_rem-action, t+self.freq_thrust))   # only update angular velocity now... 
        
        
        reward = (-action - la.norm(self.state[4:7]) - t)#TODO: weight this sum
        
        if (abs(self.state[4:7]) < self.w_tol).all():
            terminated = True
        else:
            terminated = False    
        
        info = {}
        
        return self.state, reward, terminated, False, info, T_I

def a2t_laser_3D(a, hrel,d):
    u_mag = a/d**2
    hrel_norm = la.norm(hrel)

    control_input = - u_mag * hrel / hrel_norm
    return control_input


def a2t_laser_2d(a, d, r):
    """
    Convert action to torque
    Assuming the target is circular (spherical)
    Args:
        a: action [u_mag, theta]
        d: distance between the target and the satellite
        r: radius of the target 
    Returns: 
        torque: [tau_z]   
    """
    
    u_mag, θ = a 
    fvec = np.array([u_mag*np.cos(θ), u_mag*np.sin(θ)])

    ϕ = np.arcsin(r/d*np.sin(abs(θ)))

    if abs(ϕ) < np.pi/2:
        ϕ = np.pi - ϕ
    
    λ = np.pi - θ - ϕ
    
    if θ > 0: 
        rvec = np.array([r*np.cos(λ), r*np.sin(λ)])
    else: 
        rvec = np.array([r*np.cos(λ), -r*np.sin(λ)])
    
    torque = np.cross(rvec, fvec)
    
    return torque 
    
def ode_qw(qw,t,J,T):
    q = qw[0:4]
    w = qw[4:7]
    return np.concatenate((q_kin(q, w),  euler_dyn(w, J, T)))

def q_kin(q, omega):
    # qdot = 0.5 * q * omega
    return 0.5 * q_mul(q, np.array([0, omega[0], omega[1], omega[2]]))

def euler_dyn(w, I, tau):
    return la.inv(I).dot(tau - np.cross(w, I.dot(w)))

### Quaternion setup 
# q = [q0, q1, q2, q3] = [scalar, vector]

def q_mul(q0, q1):
    return np.array([
        q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2] - q0[3]*q1[3],
        q0[0]*q1[1] + q0[1]*q1[0] + q0[2]*q1[3] - q0[3]*q1[2],
        q0[0]*q1[2] - q0[1]*q1[3] + q0[2]*q1[0] + q0[3]*q1[1],
        q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1] + q0[3]*q1[0]
    ])   

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_inv(q):
    return q_conj(q) / la.norm(q)

def q2rotmat(q):
    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,     1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1,     1 - 2*q1**2 - 2*q2**2]
    ])