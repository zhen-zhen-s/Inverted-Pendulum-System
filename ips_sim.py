import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
from ips_traj import plot_trajectory
from ips import Ips

def simulate_cartpole(ips, x0, tf, plotting=False):
  n_frames = 30
  # Model parameters
  g = 9.81
  mc = 1
  mp = 1
  L = 1

  def f(t, x):
    M = np.array([[mc + mp, mp * L * cos(x[1])], [mp * L * cos(x[1]), mp * L**2]])
    C = np.array([-mp * L * sin(x[1]) * x[3]**2, mp * g * L * sin(x[1])])
    B = np.array([1, 0])

    u = ips.compute_efforts(t, x)

    x_dot = np.hstack((x[-2:], np.linalg.solve(M, B * u  - C)))
    return x_dot

  sol = solve_ivp(f, (0, tf), x0, max_step=1e-3)

  if plotting:
    anim, fig = plot_trajectory(sol.t, sol.y, n_frames, L)
    return anim, fig
  else:
    return sol.y

if __name__ == '__main__':
  x0 = np.zeros(4)
  x0[1] = pi/6
  tf = 10
  ips = Ips()
  simulate_cartpole(ips, x0, tf, True)