import numpy as np
import pandas as pd
import math

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

import matplotlib.pyplot as plt

mate = np.array([5,-1])
opp = np.array([-3,3])
posessor = np.array([0,0])


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def normalize(vec):
    norm_denom = np.linalg.norm(vec)
    if norm_denom == 0.0:
        return np.zeros(vec.shape)
    return vec / np.linalg.norm(vec)

def angle_between_vectors(a,b):
    return np.arccos(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))

def getRadians(x1, y1, x2, y2):
    ''' Calculates the degrees of angle between two points... posessor should always be first '''
    return math.atan2(y2-y1, x2-x1)

def getL(rad, d_back, d_front):
    z = (1 - np.cos(rad)) / 2
    L = d_back + ((d_front - d_back) * (z**3 + (z * 0.3)) / 1.3) #calculate distance limit in that direction
    return L


border_x, border_y = [], []
rad_to_net = 0
for i in np.linspace(0,(2*np.pi),200):
    L = getL(i-rad_to_net, 5, 15)
    border_x.append(posessor[0]-(np.cos(i)*L))
    border_y.append(posessor[1]-(np.sin(i)*L))

plt.scatter(posessor[0], posessor[1])

print("min: ", np.amin(np.array(border_x)))
print("max: ", np.amax(np.array(border_x)))

plt.scatter(border_x, border_y)
plt.ylim(-20,20)
plt.xlim(-20,30)
plt.show()
exit()



me_mate = getRadians(posessor[0], posessor[1], mate[0], mate[1])
me_opp = getRadians(posessor[0], posessor[1], opp[0], opp[1])

v1 = posessor - opp
v2 = posessor - mate
beta = np.degrees(angle_between_vectors(v1, v2))
# print(beta)
#
# print(np.degrees(me_mate))
# print(np.degrees(me_opp))

to_mate = mate - posessor
to_opp = opp - posessor
projection = np.dot(to_opp, to_mate) / euclideanDistance(posessor[0], posessor[1], mate[0], mate[1])
print(projection)

mate_proj = mate - (posessor - mate)
plt.scatter(mate_proj[0], mate_proj[1], c='k')
plt.scatter(mate[0], mate[1])
plt.scatter(opp[0], opp[1])
plt.scatter(posessor[0], posessor[1])
plt.show()


# v1 = posessor - np.array([opp_x, opp_y])
# mate_proj = mate - (posessor - mate)
# v2 = mate_proj - np.array([opp_x, opp_y])
#
# rad = angle_between_vectors(v1, v2)
