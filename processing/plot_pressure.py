import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import math


def getRadians(x1, y1, x2, y2):
    ''' Calculates the degrees of angle between two points... posessor should always be first '''
    return math.atan2(y2-y1, x2-x1)

def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def getL(rad, d_back, d_front,func):
    z = (1 - np.cos(rad)) / 2
    L = d_back + ((d_front - d_back) * (z**3 + (z * 0.3)) / 1.3) #calculate distance limit in that direction
    return 0-(func(rad)*L), L


def plotPosessionTimestep(border=None):
    ''' This function makes the heatmap of pressure zone '''
    # fig, ax = self._game.getRinkPlot()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    back = np.linspace(0,5,30)
    front = np.linspace(0,15,30)
    q = 1

    for theta in np.linspace(0,(2*np.pi)-(1/(4*np.pi)),100):
        Lbx,Lb = getL(theta,5,15,func=np.cos)
        Lby,Lb = getL(theta,5,15,func=np.sin)
        for count, val in enumerate(back):
            dL = euclideanDistance(0, 0, Lbx, Lby)
            x,_ = getL(theta,back[count],front[count],func=np.cos)
            y,_ = getL(theta,back[count],front[count],func=np.sin)
            d = euclideanDistance(0, 0, x, y)
            pressure = 1 - (d / dL)
            # pressure = (1 - (d / Lb) ** q)
            # print(pressure)
            plt.scatter(x, y, s=70, c='b', alpha=pressure, edgecolors='None')

    if border is not None:
        plt.scatter(border[0], border[1], s=10, c='k')

    plt.scatter(0, 0, c='k',s=90)
    plt.xlabel("Front/Back Distance (ft)", fontsize=25)
    plt.ylabel("Side/Side Distance (ft)", fontsize=25)
    plt.title("Zone of Pressure", fontsize=25)
    # ax.annotate("$p$",(0.5, 0.5),fontsize=10)
    plt.arrow(0,0,10,0, head_width=1, head_length=3, color='k')
    ax.annotate("$V^{threat}$",(12, -1.5),fontsize=15)


    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Puck Possessor ($p$)',
                    markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Pressure',
                          markerfacecolor='b', markersize=10),
                   Line2D([0], [0], color='k', lw=4, label='Pressure Boundary ($L$)')]

    plt.legend(handles=legend_elements, loc='upper right', framealpha=1)
    plt.show()

def plotPresser(border=None):
    ''' This function makes the pressure zone with opponents in it '''
    # fig, ax = self._game.getRinkPlot()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = [7]
    y = [-4]
    tot_pressure = 0
    for count, loc in enumerate(x):
        # theta = getRadians(0,0,15,0) + np.pi
        theta = getRadians(0,0,x[count],y[count]) + np.pi

        Lbx,Lb = getL(theta,5,15,func=np.cos)
        Lby,Lb = getL(theta,5,15,func=np.sin)
        dL = euclideanDistance(0, 0, Lbx, Lby)
        d = euclideanDistance(0, 0, x[count], y[count])

        pressure = 1 - (d / dL)
        plt.scatter(x[count], y[count], s=90, c='w', edgecolors='k')
        if count == 0: anno_loc = [x[count]-2, y[count]-2]
        else: anno_loc = [x[count]-2, y[count]+1]
        ax.annotate("$o(p)$ = " + str(round(pressure,2)),(anno_loc[0],anno_loc[1]),fontsize=15)
        tot_pressure+=pressure

    if border is not None:
        plt.scatter(border[0], border[1], s=10, c='k')

    plt.scatter(0, 0, c='k',s=90)
    ax.annotate("$O(p)$ = " + str(round(tot_pressure,2)),(-2,-2),fontsize=15)
    plt.xlabel("Front/Back Distance (ft)", fontsize=25)
    plt.ylabel("Side/Side Distance (ft)", fontsize=25)
    plt.title("One Presser", fontsize=25)
    # ax.annotate("$p$",(0.5, 0.5),fontsize=10)
    plt.arrow(0,0,10,0, head_width=1, head_length=3, color='k')
    ax.annotate("$V^{threat}$",(12, -1.5),fontsize=15)


    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Puck Possessor ($p$)',
                    markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='o', color='k', label='Opponents ($o$)',
                          markerfacecolor='w', markersize=10),
                   Line2D([0], [0], color='k', lw=4, label='Pressure Boundary ($L$)')]

    plt.legend(handles=legend_elements, loc='upper right', framealpha=1)
    plt.show()

rad_to_net = 0
border_x, border_y = [], []
for i in np.linspace(0,(2*np.pi),600):
    border_x.append(getL(i,5,15,func=np.cos)[0])
    border_y.append(getL(i,5,15,func=np.sin)[0])

# plotPosessionTimestep(border=(border_x,border_y))
plotPresser(border=(border_x,border_y))
