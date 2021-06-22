# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:15:06 2020

@author: James
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(x))
    return line,


def animate(i):
    line.set_ydata(np.sin(x + i / 100))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=False, save_count=100)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=18000)
ani.save("movie.mp4", writer=writer, dpi=200)

# plt.show()