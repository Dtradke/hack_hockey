'''
    File name: passing_lane_functions.py
    Author: Daniel Radke
    Date created: 22.04.2021
    Date last modified: 24.04.2021
    Python Version: 3.7.5
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

def distance(a,b):
    # returns euclidean distance from a to b
    return np.linalg.norm(a - b)

def normalize(vec):
    # returns the normalized vector or vec
    return vec / np.linalg.norm(vec)

def point_in_circle(p,c,r):
    if (distance(p,c) < r):
        return True
    else:
        return False

def circles_intersections(c0, r0, c1, r1):
# def circles_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    x0 = c0[0]
    y0 = c0[1]
    x1 = c1[0]
    y1 = c1[1]


    d=np.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=np.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        c3 = np.asarray([x3,y3])
        c4 = np.asarray([x4,y4])
        # return (x3, y3, x4, y4)
        return (c3, c4)


def opp_in_ROI(p,r,o,gamma,t):
    d = distance(p,r)
    pr_vec = normalize(r-p)
    p_radius = gamma*d*t
    r_radius = gamma*d*(1-t)

    if (gamma >= 2):
        if point_in_circle(o,r,r_radius):
            # opp is inside the circle around r
            return True
        else:
            return False
    else:
        arc_radius = 4*d/gamma
        # radius of circles around p and r to find centers of arc circles
        arc_p = arc_radius - p_radius
        arc_r = arc_radius - r_radius
        c_arc3, c_arc4 = circles_intersections(p,arc_p,r,arc_r)

        if ((not point_in_circle(o,c_arc3,arc_radius)) | (not point_in_circle(o,c_arc4,arc_radius))):
            # opp is outside the "lune"
            # print("Not In Lune")
            return False
        elif point_in_circle(o,r,r_radius):
            # opp is inside the circle around r
            # print("In R Circle")
            return True
        elif point_in_circle(o,p,p_radius):
            # opp is inside the circle around p
            # print("In P Circle")
            return True
        else:
            # opp is inside the lune, but not in circles around p and r
            # need to check if opp is "between" the circles
            # point_on_p = circles_intersections(p,p_radius,c_arc3,arc_radius)
            # point_on_r = circles_intersections(r,r_radius,c_arc3,arc_radius)
            point_on_p = p + normalize(p-c_arc3)*p_radius #circles_intersections(p,p_radius,c_arc3,arc_radius)
            point_on_r = r + normalize(r-c_arc3)*r_radius #circles_intersections(r,r_radius,c_arc3,arc_radius)
            # print("p >>> ", point_on_p)
            # print("r >>> ", point_on_r)
            # exit()
            # on_p_dot_pr = np.dot(point_on_p[0],pr_vec)
            # on_r_dot_pr = np.dot(point_on_r[0],pr_vec)
            on_p_dot_pr = np.dot(point_on_p,pr_vec)
            on_r_dot_pr = np.dot(point_on_r,pr_vec)
            if (on_p_dot_pr < np.dot(o,pr_vec) < on_r_dot_pr):
                # print("In Lune, between Circles")
                return True
            else:
                # print("In Lune, not between Circles")
                return False

def getValueOfA(posessor,mate,opp,t, bounds=[0,2,4]):

    in_lune = opp_in_ROI(posessor,mate,opp,bounds[1],t)

    if in_lune is None:
        return bounds[1]

    resolution = bounds[1] - bounds[0]
    if resolution < 0.01:
        return bounds[1]

    if not in_lune and bounds[1] >= 2:
        # compute exactly
        radius = distance(mate,opp)
        d = distance(posessor,mate)
        return radius / (d*(1-t))
    else:
        b0, b1, b2 = copy.deepcopy(bounds[0]), copy.deepcopy(bounds[1]), copy.deepcopy(bounds[2])
        new_bounds = np.zeros((bounds.shape))
        if in_lune:     # bounds need to decrease
            new_bounds[0] = b0
            new_bounds[1] = b1 - ((b2 - b1)/2)
            new_bounds[2] = b1
        else:           # bounds need to increase
            new_bounds[0] = b1
            new_bounds[1] = b1 + ((b2 - b1)/2)
            new_bounds[2] = b2

    return getValueOfA(posessor, mate, opp, t, new_bounds)



def main():
    p = np.asarray([0,0])
    r = np.asarray([8,0])
    gamma = 1
    t = 0.25
    o1 = np.asarray([-3,0])
    o7 = np.asarray([-2.5,0.5])
    o2 = np.asarray([1,1])
    o3 = np.asarray([2,2])
    o4 = np.asarray([4,4])
    o5 = np.asarray([16,6])
    o6 = np.asarray([20,10])

    print(opp_in_ROI(p,r,o1,gamma,t))
    print(opp_in_ROI(p,r,o7,gamma,t))
    print(opp_in_ROI(p,r,o2,gamma,t))
    print(opp_in_ROI(p,r,o3,gamma,t))
    print(opp_in_ROI(p,r,o4,gamma,t))
    print(opp_in_ROI(p,r,o5,gamma,t))
    print(opp_in_ROI(p,r,o6,gamma,t))


def plotGraph():
    ''' make passing lane plot '''
    fig=plt.figure(figsize=(8, 6))
    p = np.asarray([0,0])
    r = np.asarray([8,0])
    o1 = np.asarray([5,-3.23])
    o2 = np.asarray([2.25,-4.13])
    o3 = np.asarray([4,-7.65])


    # alphas = [1.4,1]
    alphas = [1.4, 1, 0.6]

    for a_count, a in enumerate(alphas):

        t = 0.25
        # a = 1

        d = np.sqrt((p[0] - r[0])**2 + (p[1] - r[1])**2)
        pr = a * d * t
        rr = a * d * (1-t)

        u = np.linspace(np.pi/2,5*np.pi/2,3600)

        xp = pr*np.sin(u) + p[0]
        yp = pr*np.cos(u) + p[1]

        xr = rr*np.sin(u) + r[0]
        yr = rr*np.cos(u) + r[1]

        if a == 0.6:
            plt.plot(xp, yp, c='k', linestyle='--')
            plt.plot(xr, yr, c='k', linestyle='--')
        # else:
        #     plt.plot(xp, yp, c='k', linestyle='--', alpha=0.1)
        #     plt.plot(xr, yr, c='k', linestyle='--', alpha=0.1)

        c = 4
        er = (c*d) / a

        ep = er - pr
        eR = er - rr

        b = (ep**2 - eR**2 + d**2) / (2*d)
        h = np.sqrt(ep**2 - b**2)

        x2 = p[0] + ((b*(r[0] - p[0])) / d)
        y2 = p[1] + ((b*(r[1] - p[1])) / d)

        x3 = x2 + ((h*(r[1] - p[1])) / d)
        y3 = y2 - ((h*(r[0] - p[0])) / d)

        x4 = x2 - ((h*(r[1] - p[1])) / d)
        y4 = y2 + ((h*(r[0] - p[0])) / d)

        u = np.linspace(np.pi/2,5*np.pi/2,3600)

        xr2 = er*np.sin(u) + x3
        yr2 = er*np.cos(u) + y3

        xr22 = er*np.sin(u) + x4
        yr22 = er*np.cos(u) + y4

        if a == 0.6:
            plt.plot(xr2, yr2, c='k', linestyle='--') # top curve
            plt.plot(xr22, yr22, c='k', linestyle='--') # bottom curve
        # else:
        #     plt.plot(xr2, yr2, c='k', linestyle='--', alpha=0.1) # top curve
        #     plt.plot(xr22, yr22, c='k', linestyle='--', alpha=0.1) # bottom curve

        # fill between
        if a < 2:
            point_on_p = p + normalize(p-np.asarray([x3,y3]))*pr
            point_on_r = r + normalize(r-np.asarray([x3,y3]))*rr
            fillx1 = xp[(xp < point_on_p[0]) & (yp > 0)]
            filly1 = yp[(xp < point_on_p[0]) & (yp > 0)]
            order1 = np.argsort(fillx1)
            fillx2 = xr2[(point_on_p[0] < xr2) & (xr2 < point_on_r[0]) & (yr2 > 0)]
            filly2 = yr2[(point_on_p[0] < xr2) & (xr2 < point_on_r[0]) & (yr2 > 0)]
            order2 = np.argsort(fillx2)
            fillx3 = xr[(xr > point_on_r[0]) & (yr > 0)]
            filly3 = yr[(xr > point_on_r[0]) & (yr > 0)]
            order3 = np.argsort(fillx3)

            fillx_all = np.append(fillx1,np.append(fillx2,fillx3))
            filly_all = np.append(filly1,np.append(filly2,filly3))
            order_all = np.argsort(fillx_all)

            if a_count == (len(alphas)-1): facecol = 'b'
            elif a_count == (len(alphas)-2): facecol = 'g'
            elif a_count == (len(alphas)-3): facecol = 'y'
            plt.fill_between(fillx_all[order_all], filly_all[order_all], -filly_all[order_all],facecolor=facecol, alpha=0.2)

            # plt.fill_between(fillx1[order1], filly1[order1], -filly1[order1])
            # plt.fill_between(fillx2[order2], filly2[order2], -filly2[order2])
            # plt.fill_between(fillx3[order3], filly3[order3], -filly3[order3])
        else:
            if a_count == (len(alphas)-1): facecol = 'b'
            elif a_count == (len(alphas)-2): facecol = 'g'
            elif a_count == (len(alphas)-3): facecol = 'y'
            plt.fill_between(xr[yr > 0],yr[yr > 0],-yr[yr > 0],facecolor=facecol, alpha=0.2)


    plt.plot([p[0],r[0]],[p[1],r[1]],c='k',label='$pr$')
    plt.scatter(p[0], p[1], color='k', s=90)
    plt.annotate("$p$", (p[0], p[1]-1), fontsize=20)

    plt.scatter(r[0], r[1], color='k', s=90)
    plt.annotate("$r$", (r[0], r[1]-1), fontsize=20)

    plt.scatter(o1[0], o1[1], color='r', facecolor='w', s=90)
    plt.annotate("$o_{1}$", (o1[0], o1[1]-1), color='r', fontsize=20)

    plt.scatter(o2[0], o2[1], color='r', facecolor='w', s=90)
    plt.annotate("$o_{2}$", (o2[0], o2[1]-1), color='r', fontsize=20)

    plt.scatter(o3[0], o3[1], color='r', facecolor='w', s=90)
    plt.annotate("$o_{3}$", (o3[0], o3[1]-1), color='r', fontsize=20)

    plt.text(6,2, r'$\gamma$ = 0.6', fontsize=20)
    plt.text(6,4.5, r'$\gamma$ = 1.0', fontsize=20)
    plt.text(6,7, r'$\gamma$ = 1.4', fontsize=20)


    plt.xlim(-4,20)
    plt.ylim(-9,9)
    plt.xlabel("Front/Back Distance (ft)", fontsize=20)
    plt.ylabel("Side/Side Distance (ft)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', fontsize=20)

    # custom_lines = [Line2D([0], [0], color=cmap(0.0), lw=4),
    #             Line2D([0], [0], color=cmap(0.2), lw=4),
    #             Line2D([0], [0], color=cmap(0.4), lw=4),
    #             Line2D([0], [0], color=cmap(0.6), lw=4),
    #             Line2D([0], [0], color=cmap(0.8), lw=4),
    #             Line2D([0], [0], color=cmap(1.0), lw=4),
    #             Line2D([0], [0], color='b', lw=8),
    #             Line2D([0], [0], color='g', lw=8)]
    # plt.legend(custom_lines, ['+0', '+1', '+2', '+3', '+4', '+5', 'Tampa Bay Goal', 'Dallas Goal'], fontsize=13,ncol=8, framealpha=1)


    fname = "../../Paper/paper_imgs/scatterplots/passing_lanes_diagram.png"
    plt.savefig(fname,bbox_inches='tight', dpi=300)
    plt.close()
    exit()
    # plt.show()
    # exit()


if __name__ == "__main__":
    # main()
    plotGraph()
