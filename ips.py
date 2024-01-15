import numpy as np
from math import sin, cos, pi
from scipy.linalg import solve_continuous_are


class Ips(object):
  '''
    Define the Inverted Pendulum System model
  '''
  def __init__(self):
    # Define k_p and k_d gains for swing up controller
    self.kp = 7
    self.kd = 4
    self.g = 9.81

    # Define mass of pole and cart, length of pole
    self.mc = 1
    self.mp = 1
    self.L = 1

    # Computes the lqr gains for the final stabilizing controller
    A = np.array([[0,                  0,                                1, 0],
                  [0,                  0,                                0, 1],
                  [0, self.g * (self.mc / self.mp),                      1, 0],
                  [0, self.g * (self.mc + self.mp) / (self.L * self.mc), 0, 0]])
    B = np.array([[0], [0], [1/self.mc], [1/(self.L * self.mc)]])
    Q = np.eye((4))
    Q[3, 3] = 10
    R = np.array([[1]])

    P = solve_continuous_are(A, B, Q, R)
    self.K = np.linalg.inv(R) @ B.T @ P
    self.x_des = np.array([0, pi, 0, 0])


  def compute_efforts(self, t, x):
    q = x[:2]
    qdot = x[-2:]

    e_tilde = 0.5 * qdot[1] ** 2 - self.g * cos(q[1]) - self.g
    angular_distance  = x[3]**2 + (x[1]-pi)**2

    if np.abs(e_tilde) < 1 and angular_distance < 1:
      return self.compute_lqr_input(x)
    else:
      return self.compute_energy_shaping_input(t, x)


  def compute_energy_shaping_input(self, t, x):
    '''
    Computes the energy shaping inputs to stabilize the cartpole
    '''
    q = x[:2]
    qdot = x[-2:]
    q1_ddot_pd_term = - self.kd * qdot[0] - self.kp * q[0]

    E = (0.5 * qdot[1]**2) - (self.g * np.cos(q[1])) - self.g
    q1_ddot_des = qdot[1] * np.cos(q[1]) * E
    q1_ddot_des += q1_ddot_pd_term

    u = (2 - (np.cos(q[1]))**2) * q1_ddot_des - self.g * np.cos(q[1]) * np.sin(q[1]) - np.sin(q[1]) * (qdot[1]**2);
    return u

  def compute_lqr_input(self, x):
    '''
    Stabilizes the cartpole at the final location using lqr
    '''
    return -self.K @ (x - self.x_des)


