'''Plot error graph and show convergence rate.'''
from __future__ import division
from fe_utils.solvers.one_dimensional import solve_one_dimensional
from matplotlib import pyplot as plt
import numpy as np

res = [2 ** i for i in range(3, 7)]
graph = [solve_one_dimensional(1, r, return_error=True)[0] for r in res]
error = [solve_one_dimensional(1, r)[1] for r in res]

fig = plt.figure(figsize=(12, 8))
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.annotate(f'nc = {res[i]}', xy=(0.4, 0.8), xycoords='axes fraction', fontsize=16)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.ticklabel_format(style='sci', scilimits=(0, 0), useMathText=True)
    ax.yaxis.offsetText.set_fontsize(12)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('error', fontsize=18)
    ax.tick_params(which='both', width=2, direction='in', labelsize=12)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=4)
    graph[i].plot(ax)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(res, error, 'k^-', lw=2, ms=16, mec='k', mfc='c', mew=2)
ax.set_xlabel('nc', fontsize=18)
ax.set_ylabel('error', fontsize=18)
# ax.grid(which='both', b=True)
ax.tick_params(which='both', width=2, direction='in', labelsize=14)
ax.tick_params(which='major', length=10)
ax.tick_params(which='minor', length=6)
plt.show()
