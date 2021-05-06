

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import copy
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import math
# import beta_functions2 as bf2

def angle_between_vectors(a,b):
    return np.arccos(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))

def normalize(vec):
    return vec / np.linalg.norm(vec)


def getL(rad, d_back, d_front,func):
    z = (1 - np.cos(rad)) / 2
    L = d_back + ((d_front - d_back) * (z**3 + (z * 0.3)) / 1.3) #calculate distance limit in that direction
    return 0-(func(rad)*L), L


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.axis('equal')
ax.axis('off')



border_x, border_y = [], []
for i in np.linspace(0,(2*np.pi),600):
    border_x.append(getL(i,5,15,func=np.cos)[0])
    border_y.append(getL(i,5,15,func=np.sin)[0])

plt.scatter(border_x, border_y, s=10, c='k')


# opponents
plt.scatter(9, 4.03, s=50, c='w', edgecolors='k')
plt.annotate("$o_{1}$",(8.5, 5), c='r', fontsize=25)
plt.plot([0,9], [-40,4], linestyle='--', c='r')
plt.annotate("$\\theta_{1}$",(0.3, -23), fontsize=20)


plt.scatter(-15.02, -24.97, s=50, c='w', edgecolors='k')
plt.annotate("$o_{2}$",(-15.1, -24), c='r', fontsize=25)
plt.plot([0,-15], [-40,-25], linestyle='--', c='r')
plt.annotate("$\\theta_{2}$",(-3.5, -36.5), fontsize=20)


plt.scatter(-10, 10, s=50, c='w', edgecolors='k')
plt.annotate("$o_{3}$",(-10.1, 11), c='r', fontsize=25)



# possession team

plt.scatter(0, -40, s=50, c='k')
plt.annotate("$p$",(-0.2, -42.5), fontsize=25)

plt.scatter(0, 0, s=50, c='k')
plt.annotate("$r$",(-0.1, 1), fontsize=25)

plt.plot([0,0], [-40,0], linestyle=':', c='k')

plt.arrow(-10,-48.5,40,0, head_width=3, head_length=5, color='k')
ax.annotate("$p$ offensive direction",(-5, -47.5),fontsize=20)

ax.annotate("$d(p, r)$",(-4, -20), rotation=90, fontsize=20)


legend_elements = [Line2D([0], [0], marker='o', color='w', label='Possessing Team',
                markerfacecolor='k', markersize=10),
               Line2D([0], [0], marker='o', color='k', label='Opponents ($o$)',
                      markerfacecolor='w', markersize=10),
               Line2D([0], [0], color='k', lw=4, label='Pressure Boundary'),
               Line2D([0], [0], color='k', linestyle=":", lw=4, label='Direct Passing Lane'),
               Line2D([0], [0], color='r', linestyle="--", lw=4, label='Orientation to Opponents'),]

plt.legend(handles=legend_elements, loc='upper right', framealpha=1)

fname = "../../Paper/paper_imgs/possessions/passing_lane_diagram_deg.png"
plt.savefig(fname,bbox_inches='tight', dpi=300)
plt.close()
# plt.show()
