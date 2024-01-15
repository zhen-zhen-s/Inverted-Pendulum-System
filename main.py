from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np
from math import sin, cos, pi
from ips import Ips
from ips_sim import simulate_cartpole

# Initial state (Adjustable)
x0 = np.zeros(4)
x0[0] = 0.5
x0[1] = 4.7
x0[2] = 0
x0[3] = 0
tf = 10

# Create a cartpole system with LQR controller
cartpole = Ips()

# Simulate and create animation
anim, fig = simulate_cartpole(cartpole, x0, tf, True)
anim
anim.save('cartpole_animation.gif', writer='pillow')