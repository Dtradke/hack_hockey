import json
import sys
import numpy as np
import pandas as pd
import pickle
import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy

# fake
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

import passing_lane_functions


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def isInHalfCircle(posessor, mate, opp):
    dpr = euclideanDistance(posessor[0], posessor[1], mate[0], mate[1])
    dro = euclideanDistance(opp[0], opp[1], mate[0], mate[1])
    if dro <= dpr:
        return True
    return False

def isOppInCorridor(posessor, mate, opp):
    to_mate = mate - posessor
    to_opp = opp - posessor
    dist_posessor_to_mate = euclideanDistance(posessor[0], posessor[1], mate[0], mate[1])
    projection = np.dot(to_opp, to_mate) / dist_posessor_to_mate

    if projection >= 0 and projection <= dist_posessor_to_mate:
        return True
    return False

# def getPRQradians(a, b, c):
#     ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
#     if ang < 0:
#         ang+=360
#     print("degrees: ", ang)
#     ang = ang*(np.pi/180)
#     print("angle: ", ang)
#     return ang

def angle_between_vectors(a,b):
    return np.arccos(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))

def normalize(vec):
    norm_denom = np.linalg.norm(vec)
    if norm_denom == 0.0:
        return np.zeros(vec.shape)
    return vec / np.linalg.norm(vec)

def getRadians(x1, y1, x2, y2):
    ''' Calculates the degrees of angle between two points... posessor should always be first '''
    return math.atan2(y2-y1, x2-x1)


def getL(rad, d_back, d_front):
    z = (1 - np.cos(rad)) / 2
    L = d_back + ((d_front - d_back) * (z**3 + (z * 0.3)) / 1.3) #calculate distance limit in that direction
    return L


def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)


def isInPressureZone(game, mate, opp, mate_obj, period):
    dist_mate_to_opp = euclideanDistance(mate[0], mate[1], opp[0], opp[1])
    attacking_net,_ = game.getAttackingNet(mate_obj._team, period)
    rad_to_net = getRadians(mate[0], mate[1], attacking_net["X"], attacking_net["Y"])

    rad_to_opp = getRadians(mate[0], mate[1], opp[0], opp[1])
    rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)
    L = getL(rad_diff, 5, 15)

    if dist_mate_to_opp < L:
        return True, dist_mate_to_opp
    else:
        return False, dist_mate_to_opp


def getUTCtimekey(obj, time):
    key = np.argmin(np.abs(np.array(list(obj._hd_UTC_update.keys())) - time))
    return list(obj._hd_UTC_update.keys())[key]


def getPassingLane_General(game, posessor, receiver, time):
    ''' Calculates passlanes

    input:
        game object,
        puck possessor object,
        potential receiver,
        current timestep

    returns:
        current bete for potential receiver

    '''

    # time_key = np.argmin((np.abs(np.array(list(game._entities['1']._hd_UTC_update.keys())) - time)))
    # key = list(game._entities['1']._hd_UTC_update.keys())[time_key]
    # posessor = np.array([game._entities['1']._hd_UTC_update[key]["X"], game._entities['1']._hd_UTC_update[key]["Y"]])

    betas, min_betas_lst, min_betas_player = [], [], []



    arg_key = np.argmin(np.abs(np.array(list(receiver._hd_UTC_update.keys())) - time))
    key = list(receiver._hd_UTC_update.keys())[arg_key]
    # time_key = getUTCtimekey(receiver, time)
    mate = np.array([receiver._hd_UTC_update[key]["X"], receiver._hd_UTC_update[key]["Y"]])
    dist_to_mate = euclideanDistance(posessor[0], posessor[1], mate[0], mate[1])

    opponents = game.getOpponentsOnIce(receiver._team, time)

    t = 0.25
    betas, opps = [], []
    for o in opponents:
        opp_obj = game._entities[o["entity_id"]]
        if opp_obj._pos == 'G': continue
        arg_key = np.argmin(np.abs(np.array(list(opp_obj._hd_UTC_update.keys())) - time))
        key = list(opp_obj._hd_UTC_update.keys())[arg_key]
        # time_key = getUTCtimekey(opp_obj, time)
        opp = np.array([opp_obj._hd_UTC_update[key]["X"], opp_obj._hd_UTC_update[key]["Y"]])
        opps.append(opp)

        a = passing_lane_functions.getValueOfA(posessor,mate,opp,t, bounds=np.array([0,2,4]))
        betas.append(a)

    # opponents = game.getOpponentsOnIce(receiver._team, time)
    # teammates = game.getTeammatesOnIce(receiver._team, time)
    # fig, ax = plotGamma(game, np.amin(np.array(betas)), posessor, mate)
    # for o in opps:
    #     plt.scatter(o[0], o[1], c='g')
    # plt.scatter(posessor[0], posessor[1], c='r')
    # plt.scatter(mate[0], mate[1], c='b')
    # plt.show()

    return np.amin(np.array(betas)), opps[np.argmin(np.array(betas))]




def plotGamma(game, gamma, p, r):
    # fig = plt.figure()
    fig, ax = game.getRinkPlot()
    t = 0.25

    d = np.sqrt((p[0] - r[0])**2 + (p[1] - r[1])**2)
    pr = gamma * d * t
    rr = gamma * d * (1-t)

    u = np.linspace(np.pi/2,5*np.pi/2,3600)

    xp = pr*np.sin(u) + p[0]
    yp = pr*np.cos(u) + p[1]

    xr = rr*np.sin(u) + r[0]
    yr = rr*np.cos(u) + r[1]

    plt.plot(xp, yp, c='k', linestyle='--')
    plt.plot(xr, yr, c='k', linestyle='--')

    c = 4
    er = (c*d) / gamma

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

    plt.plot(xr2, yr2, c='k', linestyle='--') # top curve
    plt.plot(xr22, yr22, c='k', linestyle='--') # bottom curve

    # fill between
    # if gamma < 2:
    #     point_on_p = p + normalize(p-np.asarray([x3,y3]))*pr
    #     point_on_r = r + normalize(r-np.asarray([x3,y3]))*rr
    #     fillx1 = xp[(xp < point_on_p[0]) & (yp > 0)]
    #     filly1 = yp[(xp < point_on_p[0]) & (yp > 0)]
    #     order1 = np.argsort(fillx1)
    #     fillx2 = xr2[(point_on_p[0] < xr2) & (xr2 < point_on_r[0]) & (yr2 > 0)]
    #     filly2 = yr2[(point_on_p[0] < xr2) & (xr2 < point_on_r[0]) & (yr2 > 0)]
    #     order2 = np.argsort(fillx2)
    #     fillx3 = xr[(xr > point_on_r[0]) & (yr > 0)]
    #     filly3 = yr[(xr > point_on_r[0]) & (yr > 0)]
    #     order3 = np.argsort(fillx3)
    #
    #     fillx_all = np.append(fillx1,np.append(fillx2,fillx3))
    #     filly_all = np.append(filly1,np.append(filly2,filly3))
    #     order_all = np.argsort(fillx_all)
    #
    #     plt.fill_between(fillx_all[order_all], filly_all[order_all], -filly_all[order_all],facecolor='b', alpha=0.2)
    #
    #     # plt.fill_between(fillx1[order1], filly1[order1], -filly1[order1])
    #     # plt.fill_between(fillx2[order2], filly2[order2], -filly2[order2])
    #     # plt.fill_between(fillx3[order3], filly3[order3], -filly3[order3])
    # else:
    #     plt.fill_between(xr[yr > 0],yr[yr > 0],-yr[yr > 0],facecolor='b', alpha=0.2)

    return fig, ax



def getAnyPassStats(game, player, period, time):
    ''' Calculates pressure similar to paper: A visual analysis of pressure in football'''
    d_front = 15
    d_back = 5
    q = 1
    # self._posession_pressures = {}
    attacking_net,_ = game.getAttackingNet(player._team, period)

    time_key = np.argmin((np.abs(np.array(list(player._hd_UTC_update.keys())) - time)))
    rad_to_net = getRadians(player._hd_UTC_update[list(player._hd_UTC_update.keys())[time_key]]["X"], player._hd_UTC_update[list(player._hd_UTC_update.keys())[time_key]]["Y"], attacking_net["X"], attacking_net["Y"])
    opponents = game.getOpponentsOnIce(player._team, time)

    timestep_pressure = []
    timestep_pressers = []
    betas = []
    for i in opponents:
        opp = game._entities[i["entity_id"]]
        opp_x = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["X"]
        opp_y = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["Y"]

        d = euclideanDistance(player._hd_UTC_update[list(player._hd_UTC_update.keys())[time_key]]["X"], player._hd_UTC_update[list(player._hd_UTC_update.keys())[time_key]]["Y"], opp_x, opp_y)

        if d <= d_front:
            rad_to_opp = getRadians(player._hd_UTC_update[list(player._hd_UTC_update.keys())[time_key]]["X"], player._hd_UTC_update[list(player._hd_UTC_update.keys())[time_key]]["Y"], opp_x, opp_y)
            rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)
            L = getL(rad_diff, d_back, d_front)
            # pressure = 100 + (((1 - d) / L) ** q) * 100
            pressure = (1 - (d / L) ** q)

            if d < L:
                timestep_pressure.append(pressure)
                timestep_pressers.append(opp)


    time_key = np.argmin((np.abs(np.array(list(game._entities['1']._hd_UTC_update.keys())) - time)))
    key = list(game._entities['1']._hd_UTC_update.keys())[time_key]
    possessor_loc = [game._entities['1']._hd_UTC_update[key]['X'], game._entities['1']._hd_UTC_update[key]['Y']]

    pass_availability, threat_o = getPassingLane_General(game, game._entities['1'], player, time)

    return np.sum(np.array(timestep_pressure)), pass_availability


class Posession(object):
    def __init__(self, game, posession, next_posession):
        self._game = game
        self._UTC_start = posession["Marker"]["MarkerData"][0]["MarkerUTC"]
        self._UTC_end = next_posession["Marker"]["MarkerData"][0]["MarkerUTC"]
        self._descriptor = posession["Marker"]["MarkerData"][0]["Descriptor_"]
        self._period = self._descriptor = posession["Marker"]["MarkerData"][0]["ETime"]["Period"]
        self._clocktime_start = self._descriptor = posession["Marker"]["MarkerData"][0]["ETime"]
        self._clocktime_end = self._descriptor = next_posession["Marker"]["MarkerData"][0]["ETime"]
        self._clocktime_seconds = self.getClocktimeDuration()
        self._posessor = game._entities[posession["Marker"]["MarkerData"][0]["Participants"][0]["EntityId"]]
        self._stolen = self.wasTurnover(posession)
        self._turnover = self.wasTurnover(next_posession)
        self._posession_UTC_keys = self.getTimeKeys()
        self._d_front = 15
        self._d_back = 5
        self._pressure = 0.0
        self._posession_pressures = {}
        self.getPressure(time=round(self._UTC_end,1))

        for time_key in self._posession_UTC_keys:
            # self.getPasslanes(time_key)
            try:
                self.getPasslanes_HalfCircleBeta(time_key)
            except:
                print(">>> Passing lane fail at time: ", time_key)
                pass
            self._game._who_has_possession[time_key] = self._posessor


# this is for total possession at once, not just a timestep like in game2
        self._total_pressure = 0
        for p in self._posession_pressures.keys():
            self._total_pressure+=np.sum(np.array(self._posession_pressures[p]))

        try:
            self._pressure_at_posession_loss = np.sum(np.array(self._posession_pressures[max(list(self._posession_pressures.keys()))]))
        except:
            self._pressure_at_posession_loss = 0.0

        # print(self._total_pressure)
        # print(self._pressure_at_posession_loss)
        # exit()

        # ----- update player
        self._posessor._pressure_to_pass.append(self._pressure_at_posession_loss)
        self._posessor._posession_time+=self._clocktime_seconds
        if self._stolen: self._posessor._steals+=1
        if self._turnover: self._posessor._turnovers+=1

    def plotPosession(self, border=None):

        for count, time in enumerate(self._posession_UTC_keys):
            fig, ax = self._game.getRinkPlot()

            opponents = self._game.getOpponentsOnIce(self._posessor._team, time)
            teammates = self._game.getTeammatesOnIce(self._posessor._team, time)
            for i in opponents:
                opp = self._game._entities[i["entity_id"]]
                plt.scatter(opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["X"], opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["Y"], c=opp._color, label=opp._number+opp._last_name)
                ax.annotate(opp._number,(opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["X"], opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["Y"]))

            for i in teammates:
                mate = self._game._entities[i["entity_id"]]
                if mate._id != self._posessor._id:
                    plt.scatter(mate._hd_UTC_update[mate._update_time[i["time_idx"]]]["X"], mate._hd_UTC_update[mate._update_time[i["time_idx"]]]["Y"], c=mate._color, label=mate._number+mate._last_name)
                    ax.annotate(mate._number,(mate._hd_UTC_update[mate._update_time[i["time_idx"]]]["X"], mate._hd_UTC_update[mate._update_time[i["time_idx"]]]["Y"]))


            plt.scatter(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], s=120, c='y', label="Posessor")
            plt.scatter(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], c=self._posessor._color, label=self._posessor._last_name)

            plt.arrow(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], (self._posessor._hd_UTC_update[time]["X"]+self._posessor._hd_UTC_update[time]["velX"]),
                        (self._posessor._hd_UTC_update[time]["Y"]+self._posessor._hd_UTC_update[time]["velY"]), length_includes_head=True,head_width=1, head_length=3)
            plt.scatter(self._game._entities['1']._hd_UTC_update[time]["X"], self._game._entities['1']._hd_UTC_update[time]["Y"], s=10, c=self._game._entities['1']._color, label="Puck")
            # plt.legend()
            plt.title("Posession: "+ self._posessor._last_name+" Clock: "+str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"])+" Duration: "+str(self._clocktime_seconds), fontsize=20)

            if border is not None:
                plt.scatter(border[0], border[1], s=10, c='m',label="Pressure Boundary")

            # theta = np.linspace(0, 2*np.pi, 100)
            # r = np.sqrt(30)
            # x1 = r*np.cos(theta)
            # x2 = r*np.sin(theta)
            # plt.plot(x1, x2)

            plt.show()
            # exit()

    def plotPosessionTimestep(self, time, game_num, possession_count, possession_timestep, border=None):

        # self.getPasslanes(time)
        self.getPasslanes_HalfCircleBeta(time)

        fig, ax = self._game.getRinkPlot()

        opponents = self._game.getOpponentsOnIce(self._posessor._team, time)
        teammates = self._game.getTeammatesOnIce(self._posessor._team, time)
        for i in opponents:
            opp = self._game._entities[i["entity_id"]]
            plt.scatter(opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["X"], opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["Y"], c=opp._color,s=120, edgecolors='k', label=opp._number+opp._last_name)
            ax.annotate(opp._number,(opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["X"], opp._hd_UTC_update[opp._update_time[i["time_idx"]]]["Y"]), fontsize=18)

        for i in teammates:
            mate = self._game._entities[i["entity_id"]]
            if mate._id != self._posessor._id:
                plt.scatter(mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["X"], mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["Y"], c=mate._color,s=120, edgecolors='k', label=mate._number+mate._last_name)
                ax.annotate(mate._number,(mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["X"], mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["Y"]), fontsize=18)

        colors = ['r', 'c', 'b', 'g']
        opps_x, opps_y = [], []
        for lane_count, lane in enumerate(self._betas):
            plt.plot([lane[0][0], self._game._entities['1']._hd_UTC_update[time]["X"]], [lane[0][1], self._game._entities['1']._hd_UTC_update[time]["Y"]], c='r') #colors[lane_count]
            x = self._game._entities['1']._hd_UTC_update[time]["X"] + ((lane[0][0] - self._game._entities['1']._hd_UTC_update[time]["X"]) / 2)
            y = self._game._entities['1']._hd_UTC_update[time]["Y"] + ((lane[0][1] - self._game._entities['1']._hd_UTC_update[time]["Y"]) / 2)
            ax.annotate(round(lane[1],2),(x-5, y), fontsize=20, c='r', bbox={'facecolor':'w', 'alpha':1, 'pad':2}) #colors[lane_count]
            # if lane[2][0] in opps_x and lane[2][1] in opps_y:
            #     lane[2][0]+=0.5
            #     lane[2][1]+=0.5
            # plt.plot([lane[2][0], self._game._entities['1']._hd_UTC_update[time]["X"]], [lane[2][1], self._game._entities['1']._hd_UTC_update[time]["Y"]], c=colors[lane_count], alpha=0.3, linestyle='--')
            # opps_x.append(lane[2][0])
            # opps_y.append(lane[2][1])

        plt.scatter(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], s=300, c='y', label="Possessor")
        ax.annotate(self._posessor._number,(self._posessor._hd_UTC_update[round(self._posessor._update_time[i["time_idx"]],1)]["X"], self._posessor._hd_UTC_update[round(self._posessor._update_time[i["time_idx"]],1)]["Y"]), fontsize=18)
        plt.scatter(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], c=self._posessor._color,s=120, edgecolors='k', label=self._posessor._last_name)

        # plt.arrow(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], (self._posessor._hd_UTC_update[time]["X"]+self._posessor._hd_UTC_update[time]["velX"]),
        #             (self._posessor._hd_UTC_update[time]["Y"]+self._posessor._hd_UTC_update[time]["velY"]), length_includes_head=True,head_width=1, head_length=3)
        plt.scatter(self._game._entities['1']._hd_UTC_update[time]["X"], self._game._entities['1']._hd_UTC_update[time]["Y"], s=15, c=self._game._entities['1']._color, label="Puck")
        # plt.legend()
        plt.title("Possessor: "+ self._posessor._first_name + " "+ self._posessor._last_name+", Clock: "+str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"])+", Duration: "+str(self._clocktime_seconds)+ " seconds", fontsize=20)

        if border is not None:
            plt.scatter(border[0], border[1], s=10, c='m',label="Pressure Boundary")

        custom_lines = [Line2D([0], [0], marker='o', markerfacecolor='b', markersize=12),
                        Line2D([0], [0], marker='o', markerfacecolor='g', markersize=12),
                        Line2D([0], [0], color='b', lw=8),
                        Line2D([0], [0], color='g', lw=8)]
        ax.legend(custom_lines, ['Tampa Bay Players', 'Dallas Players', 'Tampa Bay Goal', 'Dallas Goal'], fontsize=13,ncol=1, framealpha=1, loc="upper left", bbox_to_anchor=(1, 1))

        plt.show()
        exit()
        fname = "../../Paper/paper_imgs/possessions/game"+str(game_num+5)+"/"+str(self._posessor._last_name)+"_pos"+str(possession_count)+"_t"+str(possession_timestep)
        plt.savefig(fname,bbox_inches='tight', dpi=300)
        plt.close()
        exit()


    def getPressure(self, time):
        ''' Calculates pressure similar to paper: A visual analysis of pressure in football'''

        q = 1
        # self._posession_pressures = {}
        attacking_net,_ = self._game.getAttackingNet(self._posessor._team, self._period)

        time_key = np.argmin((np.abs(np.array(list(self._posessor._hd_UTC_update.keys())) - time)))

        rad_to_net = getRadians(self._posessor._hd_UTC_update[list(self._posessor._hd_UTC_update.keys())[time_key]]["X"], self._posessor._hd_UTC_update[list(self._posessor._hd_UTC_update.keys())[time_key]]["Y"], attacking_net["X"], attacking_net["Y"])
        # print(rad_to_net)
        self._opponents = self._game.getOpponentsOnIce(self._posessor._team, time)
        self._teammates = self._game.getTeammatesOnIce(self._posessor._team, time)

        # border_x, border_y = [], []
        # for i in np.linspace(0,(2*np.pi),200):
        #     L = self.getL(i-rad_to_net)
        #     border_x.append(self._posessor._hd_UTC_update[time]["X"]-(np.cos(i)*L))
        #     border_y.append(self._posessor._hd_UTC_update[time]["Y"]-(np.sin(i)*L))


        timestep_pressure = []
        timestep_pressers = []
        for i in self._opponents:
            opp = self._game._entities[i["entity_id"]]
            opp_x = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["X"]
            opp_y = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["Y"]

            d = euclideanDistance(self._posessor._hd_UTC_update[list(self._posessor._hd_UTC_update.keys())[time_key]]["X"], self._posessor._hd_UTC_update[list(self._posessor._hd_UTC_update.keys())[time_key]]["Y"], opp_x, opp_y)

            if d <= self._d_front:
                rad_to_opp = getRadians(self._posessor._hd_UTC_update[list(self._posessor._hd_UTC_update.keys())[time_key]]["X"], self._posessor._hd_UTC_update[list(self._posessor._hd_UTC_update.keys())[time_key]]["Y"], opp_x, opp_y)
                rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)
                L = getL(rad_diff, self._d_back, self._d_front)
                # pressure = 100 + (((1 - d) / L) ** q) * 100
                pressure = (1 - (d / L) ** q)

                if d < L:
                    timestep_pressure.append(pressure)
                    timestep_pressers.append(opp)

        self._pressure = np.sum(np.array(timestep_pressure))
        self._pressers = timestep_pressers
        self._posession_pressures[time] = timestep_pressure

        # print("Pressers: ", len(timestep_pressure), " Pressure: ", np.sum(np.array(timestep_pressure)))
        # self.plotPosessionTimestep(time, border=(border_x,border_y))

        # q = 1
        # self._posession_pressures = {}
        # for count, time in enumerate(self._posession_UTC_keys):
        #     attacking_net = self._game.getAttackingNet(self._posessor._team, self._period)
        #     rad_to_net = self.getRadians(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], attacking_net["X"], attacking_net["Y"])
        #     # print(rad_to_net)
        #     opponents = self._game.getOpponentsOnIce(self._posessor._team, time)
        #     teammates = self._game.getTeammatesOnIce(self._posessor._team, time)
        #
        #     border_x, border_y = [], []
        #     for i in np.linspace(0,(2*np.pi),200):
        #         L = self.getL(i-rad_to_net)
        #         border_x.append(self._posessor._hd_UTC_update[time]["X"]-(np.cos(i)*L))
        #         border_y.append(self._posessor._hd_UTC_update[time]["Y"]-(np.sin(i)*L))
        #
        #
        #     timestep_pressure = []
        #     for i in opponents:
        #         opp = self._game._entities[i["entity_id"]]
        #         opp_x = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["X"]
        #         opp_y = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["Y"]
        #
        #         d = euclideanDistance(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], opp_x, opp_y)
        #
        #         if d <= self._d_front:
        #             rad_to_opp = self.getRadians(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], opp_x, opp_y)
        #             rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)
        #             L = self.getL(rad_diff)
        #             pressure = 100 + (((1 - d) / L) ** q) * 100
        #
        #             if d < L:
        #                 timestep_pressure.append(pressure)
        #
            # self._posession_pressures[time] = timestep_pressure
        #
        #     # print("Pressers: ", len(timestep_pressure), " Pressure: ", np.sum(np.array(timestep_pressure)))
        #     self.plotPosessionTimestep(time, border=(border_x,border_y))


    def getPasslanes(self, time):
        ''' Calculates passlanes using beta skeletons
        minimum beta is the most covered the person is
        '''

        # posessor = np.array([self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"]])

        time_key = np.argmin((np.absolute(np.array(list(self._game._entities['1']._hd_UTC_update.keys())) - time)))
        posessor = np.array([self._game._entities['1']._hd_UTC_update[list(self._game._entities['1']._hd_UTC_update.keys())[time_key]]["X"], self._game._entities['1']._hd_UTC_update[list(self._game._entities['1']._hd_UTC_update.keys())[time_key]]["Y"]])
        self._betas = []
        min_betas_lst, min_betas_player = [], []

        for m in self._teammates:
            # print()
            if m["entity_id"] != self._posessor._id:
                mate_obj = self._game._entities[m["entity_id"]]
                if mate_obj._pos == 'G': continue

                time_key = np.argmin((np.abs(np.array(list(mate_obj._hd_UTC_update.keys())) - time)))
                key = list(mate_obj._hd_UTC_update.keys())[time_key]
                mate = np.array([mate_obj._hd_UTC_update[key]['X'], mate_obj._hd_UTC_update[key]['Y']])
                # time_key_mate = getUTCtimekey(mate_obj, time)
                # mate = np.array([mate_obj._hd_UTC_update[time_key_mate]["X"], mate_obj._hd_UTC_update[time_key_mate]["Y"]])

                pass_availability, threat_o = getPassingLane_General(self._game, posessor, mate_obj, time)

                min_betas_lst.append(pass_availability)
                min_betas_player.append(mate_obj)
                self._betas.append((mate,pass_availability))
                mate_obj._openness.append(pass_availability)
        return min_betas_player[np.argmax(min_betas_lst)]

    def getPasslanes_HalfCircleBeta(self, time):
        ''' Calculates passlanes using half-circle beta skeletons
        minimum beta is the most covered the person is
        '''

        # posessor = np.array([self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"]])

        time_key = np.argmin((np.absolute(np.array(list(self._game._entities['1']._hd_UTC_update.keys())) - time)))
        posessor = np.array([self._game._entities['1']._hd_UTC_update[list(self._game._entities['1']._hd_UTC_update.keys())[time_key]]["X"], self._game._entities['1']._hd_UTC_update[list(self._game._entities['1']._hd_UTC_update.keys())[time_key]]["Y"]])
        self._betas = []
        min_betas_lst, min_betas_player = [], []

        for m in self._teammates:
            # print()
            if m["entity_id"] != self._posessor._id:
                mate_obj = self._game._entities[m["entity_id"]]
                if mate_obj._pos == 'G': continue
                # print(mate_obj._last_name)

                time_key = np.argmin((np.abs(np.array(list(mate_obj._hd_UTC_update.keys())) - time)))
                key = list(mate_obj._hd_UTC_update.keys())[time_key]
                mate = np.array([mate_obj._hd_UTC_update[key]['X'], mate_obj._hd_UTC_update[key]['Y']])

                # time_key_mate = getUTCtimekey(mate_obj, time)
                # mate = np.array([mate_obj._hd_UTC_update[time_key_mate]["X"], mate_obj._hd_UTC_update[time_key_mate]["Y"]])

                pass_availability, threat_o = getPassingLane_General(self._game, posessor, mate_obj, time)

                min_betas_lst.append(pass_availability)
                min_betas_player.append(mate_obj)

                self._betas.append((mate,pass_availability, threat_o))
                mate_obj._openness.append(pass_availability)

        return min_betas_player[np.argmin(min_betas_lst)]


    def getTimeKeys(self):
        key_list = np.array(list(self._posessor._hd_UTC_update.keys()))

        key_list = key_list[key_list < (self._UTC_end)]
        key_list = key_list[key_list > (self._UTC_start)]
        return key_list

    def getClocktimeDuration(self):
        start_seconds = (self._clocktime_start["ClockMinutes"] * 60) + self._clocktime_start["ClockSeconds"]
        end_seconds = (self._clocktime_end["ClockMinutes"] * 60) + self._clocktime_end["ClockSeconds"]
        return start_seconds - end_seconds

    def wasTurnover(self, posession):
        participants = posession["Marker"]["MarkerData"][0]["Participants"]
        try:
            if participants[0]["EntityId"] == '' or participants[1]["EntityId"] == '':
                return False
        except:
            return False # NOTE: this catches cases where there are no entities in the next posession
        if self._game._entities[participants[0]["EntityId"]]._team != self._game._entities[participants[1]["EntityId"]]._team:
            return True
        else:
            return False

    def __repr__(self):
        return "Posession(Team: {}, Player: {}, Period: {}, Clock: {}, Duration: {}, UTC_Start: {}, UTC_diff: {}, Turnover?: {}, PassPressure: {})".format(self._posessor._team, self._posessor._last_name, self._period, str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"]),
                                            self._clocktime_seconds, self._UTC_start, self._posession_UTC_keys.shape[0], self._turnover, self._pressure)

        # return "Posession(Team: {}, Player: {}, Period: {}, Clock: {}, Duration: {}, UTC_diff: {}, Turnover?: {}, PassPressure: {})".format(self._posessor._team, self._posessor._last_name, self._period, str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"]),
        #                                     self._clocktime_seconds, self._posession_UTC_keys.shape[0], self._turnover, self._pressure_at_posession_loss)

class Faceoff(object):
    def __init__(self, game, time, info):
        self._game = game
        self._time = time
        self._UTC_update = info["Marker"]["MarkerData"][0]["MarkerUTC"]
        self._period = info["Marker"]["MarkerData"][0]["ETime"]["Period"]
        self._clock_minutes = info["Marker"]["MarkerData"][0]["ETime"]["ClockMinutes"]
        self._clock_seconds = info["Marker"]["MarkerData"][0]["ETime"]["ClockSeconds"]
        self._participants = info["Marker"]["MarkerData"][0]["Participants"]
        self._loc = {"X": info["Marker"]["MarkerData"][0]["Properties"][0]["Location"]["X"],
                    "Y": info["Marker"]["MarkerData"][0]["Properties"][0]["Location"]["Y"]}

    def __repr__(self):
        return "Faceoff(Period: {}, Clock: {}:{}, Loc: {}, {})".format(self._period, self._clock_minutes, self._clock_seconds, round(self._loc["X"], 1), round(self._loc["Y"],1))

class Penalty(object):
    def __init__(self, game, scoreboard, team):
        self._game = game
        self._clock_minutes = scoreboard["ClockMinutes"]
        self._clock_seconds = scoreboard["ClockSeconds"]
        self._team = team
        self._teamId = team

    def __repr__(self):
        return "Penalty(Team: {}, Clock: {}:{})".format(self._team, self._clock_minutes, self._clock_seconds)


class Goal(object):
    def __init__(self, game, time, info, UTC_update=None, period=None, clock_mins=None, clock_secs=None, participants=None, scorer=None):
        self._game = game
        self._time = time
        self._UTC_update = info["Marker"]["MarkerData"][0]["MarkerUTC"] if UTC_update is None else UTC_update
        self._period = info["Marker"]["MarkerData"][0]["ETime"]["Period"] if period is None else period
        self._clock_minutes = info["Marker"]["MarkerData"][0]["ETime"]["ClockMinutes"] if clock_mins is None else clock_mins
        self._clock_seconds = info["Marker"]["MarkerData"][0]["ETime"]["ClockSeconds"] if clock_secs is None else clock_secs
        self._participants = info["Marker"]["MarkerData"][0]["Participants"] if participants is None else participants
        self.assists = []
        self._goalie = None
        if scorer is None:
            self.getGoalStats()
        else:
            self._scorer = scorer
            self._teamId = scorer._team
            self._team_name = game._teams[self._scorer._team]

    def getGoalStats(self):
        for i, player in enumerate(self._participants):
            if player["Role"] == "MapRoleShooter":
                self._scorer = self._game._entities[player["EntityId"]]

                time_key = np.argmin(np.array(list(self._scorer._hd_UTC_update.keys())) - self._UTC_update)
                self._loc = {"X": self._scorer._hd_UTC_update[list(self._scorer._hd_UTC_update.keys())[time_key]]["X"],
                            "Y": self._scorer._hd_UTC_update[list(self._scorer._hd_UTC_update.keys())[time_key]]["Y"]}

                self._teamId = self._scorer._team
                self._team_name = self._game._teams[self._teamId]
            elif player["Role"] == "MapRoleShotAssister":
                self.assists.append(self._game._entities[player["EntityId"]])
            elif player["Role"] == "MapRoleGoalie":
                self._goalie = self._game._entities[player["EntityId"]]

    def __repr__(self):
        assists = []
        for i in self.assists:
            assists.append(i._last_name)
        return "Goal(Team: {}, Period: {}, Clock: {}:{}, Scorer: {}, Assists: {})".format(self._team_name, self._period, self._clock_minutes, self._clock_seconds, self._scorer._last_name, assists)

class Shot(object):
    def __init__(self, game, time, descriptor, info):
        self._game = game
        self._time = time # when the shot was taken
        self._period = info[0]["ETime"]["Period"]
        self._clock_minutes = info[0]["ETime"]["ClockMinutes"]
        self._clock_seconds = info[0]["ETime"]["ClockSeconds"]
        self._shooter = None
        self._loc = None
        self._distance = None
        self._blocker = None
        self._blocked = False
        self._miss = False
        self._d_front = 15
        self._d_back = 5
        self._shot_pressure = 0.0
        self.getShotData(game, descriptor, info)
        self.getDistance()

        if self._shooter is not None:
            self.getShotPressure()

        # self.getInformation(info)
        # self.beat = 0
        # self.beatOpponents(info, game)

    def getDistance(self):
        if self._loc is not None and self._shooter is not None:
            self._attacking_net,_ = self._game.getAttackingNet(self._shooter._team, self._period)
            self._distance = euclideanDistance(self._attacking_net["X"], self._attacking_net["Y"], self._loc["X"], self._loc["Y"])

    def getShotData(self, game, descriptor, info):
        desc_arr = descriptor.split(" ")
        if "MISSED." in desc_arr:
            self._miss = True

        # if 'Location' in list(info[0]["Properties"][0].keys()):
            # self._loc = info[0]["Properties"][0]["Location"]

        for p in info[0]["Participants"]:
            if p["Role"] == "MapRoleShooter":
                self._shooter = self._game._entities[p["EntityId"]]
                self.updateLoc()
            if p["Role"] == "MapRoleNonGoalieShotBlocker":
                self._blocker = self._game._entities[p["EntityId"]]
                self._blocked = True


    def updateShot(self, info):
        for p in info[0]["Participants"]:
            if p["Role"] == "MapRoleShooter" and self._shooter is None:
                self._shooter = self._game._entities[p["EntityId"]]
                self._updateLoc()
            if p["Role"] == "MapRoleNonGoalieShotBlocker" and self._blocker is None:
                self._blocker = self._game._entities[p["EntityId"]]
                self._blocked = True

        if 'Location' in list(info[0]["Properties"][0].keys()):
            self._loc = info[0]["Properties"][0]["Location"]
            self.getDistance()

    def updateLoc(self):
        time_key = np.argmin(np.array(list(self._shooter._hd_UTC_update.keys())) - self._time)
        self._loc = {"X": self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["X"],
                    "Y": self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["Y"],
                    "Z": self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["Z"]}

        self.getShotPressure()

    def getShotPressure(self):
        ''' Calculates pressure similar to paper: A visual analysis of pressure in football'''


        time = round(self._time,1)

        q = 1
        # self._posession_pressures = {}
        attacking_net,_ = self._game.getAttackingNet(self._shooter._team, self._period)

        time_key = np.argmin((np.abs(np.array(list(self._shooter._hd_UTC_update.keys())) - time)))
        rad_to_net = getRadians(self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["X"], self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["Y"], attacking_net["X"], attacking_net["Y"])
        # print(rad_to_net)
        self._opponents = self._game.getOpponentsOnIce(self._shooter._team, time)
        self._teammates = self._game.getTeammatesOnIce(self._shooter._team, time)

        timestep_pressure = []
        timestep_pressers = []
        self._shot_presser_lst = {}
        for i in self._opponents:
            opp = self._game._entities[i["entity_id"]]
            opp_x = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["X"]
            opp_y = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["Y"]

            d = euclideanDistance(self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["X"], self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["Y"], opp_x, opp_y)

            if d <= self._d_front:
                rad_to_opp = getRadians(self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["X"], self._shooter._hd_UTC_update[list(self._shooter._hd_UTC_update.keys())[time_key]]["Y"], opp_x, opp_y)
                rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)
                L = getL(rad_diff, self._d_back, self._d_front)
                # pressure = 100 + (((1 - d) / L) ** q) * 100
                pressure = (1 - (d / L) ** q)

                if d < L:
                    timestep_pressure.append(pressure)
                    timestep_pressers.append(opp)
                    self._shot_presser_lst[opp._id] = pressure
                    opp._pressure_on_shooter.append(pressure)

        self._shot_pressure = np.sum(np.array(timestep_pressure))
        # self._shot_presser_lst = timestep_pressers
        # self._shot_pressures = timestep_pressure

    def __repr__(self):
        return "Shot(Period: {}, Clock: {}:{}, Shooter: {}, Distance: {}, Blocked?: {}, Miss?: {})".format(self._period, self._clock_minutes, self._clock_seconds, self._shooter._last_name, self._distance, self._blocked, self._miss)
        # return "Shot(Team: {}, Period: {}, Clock: {}:{}, Shooter: {}, Distance: {}, Blocked?: {}, Miss?: {})".format(self._shooter._team, self._period, self._clock_minutes, self._clock_seconds, self._shooter._last_name, self._distance, self._blocked, self._miss)


class Hit(object):
    def __init__(self, game, event_entry):
        self._time = event_entry["Marker"]["MarkerData"][0]["MarkerUTC"]
        self._descriptor = event_entry["Marker"]["MarkerData"][0]["Descriptor_"]
        self._hitter = game._entities[event_entry["Marker"]["MarkerData"][0]["Participants"][0]["EntityId"]]
        self._hittee = game._entities[event_entry["Marker"]["MarkerData"][0]["Participants"][1]["EntityId"]]
        self._period = event_entry["Marker"]["MarkerData"][0]["ETime"]['Period']
        self._clock_minutes = event_entry["Marker"]["MarkerData"][0]["ETime"]['ClockMinutes']
        self._clock_seconds = event_entry["Marker"]["MarkerData"][0]["ETime"]['ClockSeconds']
        self.getHitLoc()

    def getRealHitLoc(self):

        hitter_keys = list(self._hitter._hd_UTC_update.keys())
        hitter_min = np.argmin(np.absolute(np.array(hitter_keys) - self._time))
        hitter_keys = np.array(list(self._hitter._hd_UTC_update.keys())[(hitter_min - 70):(hitter_min)])

        hittee_keys = list(self._hittee._hd_UTC_update.keys())
        hittee_min = np.argmin(np.absolute(np.array(hittee_keys) - self._time))
        hittee_keys = np.array(list(self._hittee._hd_UTC_update.keys())[(hittee_min - 70):(hittee_min)])

        # print("hitter: ", hitter_keys)
        # print("hittee: ", hittee_keys)

        dists = []
        for i, val in enumerate(hitter_keys):
            dists.append(euclideanDistance(self._hitter._hd_UTC_update[val]["X"], self._hitter._hd_UTC_update[val]["Y"], self._hittee._hd_UTC_update[hittee_keys[i]]["X"], self._hittee._hd_UTC_update[hittee_keys[i]]["Y"]))

        hitter_min = np.argmin(np.array(dists))
        self._loc = {"X": self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["X"],
                    "Y": self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["Y"],
                    "Z": self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["Z"]}


        self._time = hitter_keys[hitter_min]



    def getHitLoc(self):
        key = np.argmin(np.absolute(np.array(list(self._hitter._hd_UTC_update.keys())) - self._time))
        hitter_keys = list(self._hitter._hd_UTC_update.keys())
        hitter_min = np.argmin(np.absolute(np.array(hitter_keys) - self._time))

        hittee_keys = list(self._hittee._hd_UTC_update.keys())
        hittee_min = np.argmin(np.absolute(np.array(hittee_keys) - self._time))

        dist = euclideanDistance(self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["X"], self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["Y"],
                                self._hittee._hd_UTC_update[hittee_keys[hittee_min]]["X"], self._hittee._hd_UTC_update[hittee_keys[hittee_min]]["Y"])

        if dist > 5:
            self.getRealHitLoc()
        else:
            self._loc = {"X": self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["X"],
                        "Y": self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["Y"],
                        "Z": self._hitter._hd_UTC_update[hitter_keys[hitter_min]]["Z"]}



        # key = np.argmin(np.absolute(np.array(list(self._hitter._hd_UTC_update.keys())) - self._time))
        # self._loc = {"X": self._hitter._hd_UTC_update[list(self._hitter._hd_UTC_update.keys())[key]]["X"],
        #             "Y": self._hitter._hd_UTC_update[list(self._hitter._hd_UTC_update.keys())[key]]["Y"],
        #             "Z": self._hitter._hd_UTC_update[list(self._hitter._hd_UTC_update.keys())[key]]["Z"]}

    def __repr__(self):
        return "Hit(Period: {}, Clock: {}:{}, Hitter: {}, Desc: {})".format(self._period, self._clock_minutes, self._clock_seconds, self._hitter._last_name, self._descriptor)


class TakeAway(object):
    def __init__(self, game, time, descriptor, info):
        self._time = time # when the takeaway happened
        self._game = game
        self._UTC_update = info[0]["MarkerUTC"]
        self._clock_minutes = info[0]["ETime"]['ClockMinutes']
        self._clock_seconds = info[0]["ETime"]['ClockSeconds']
        self.getInformation(info)
        self.getParticipants(info)
        self.getPassAvailability()

    def getInformation(self, info):
        self._period = info[0]["ETime"]["Period"]

    def getParticipants(self, info):
        participants = info[0]["Participants"]
        for p in participants:
            if p['Role'] == 'MapRoleCommittedTurnover':
                self._taker = self._game._entities[p['EntityId']]
                self._taker._takeaways+=1

        possession_keys = copy.deepcopy(np.array(list(self._game._who_has_possession.keys())))
        possession_keys = possession_keys - self._UTC_update
        possession_key_max = np.where(possession_keys < 0, possession_keys, -np.inf).argmax()
        for i in range(50):
            self._giver = self._game._who_has_possession[list(self._game._who_has_possession.keys())[possession_key_max-i]]
            if (self._giver._id != self._taker._id) and (self._giver._team != self._taker._team):
                break
        else:
            time_key_taker = getUTCtimekey(self._taker, self._UTC_update)
            taker_coord = np.array([self._taker._hd_UTC_update[time_key_taker]["X"], self._taker._hd_UTC_update[time_key_taker]["Y"]])
            opponents = self._game.getOpponentsOnIce(self._taker._team, self._UTC_update)
            dists = []
            for o in opponents:
                opp = self._game._entities[o["entity_id"]]
                time_key_opp = getUTCtimekey(opp, self._UTC_update)
                opp_coord = np.array([opp._hd_UTC_update[time_key_opp]["X"], opp._hd_UTC_update[time_key_opp]["Y"]])
                dists.append(euclideanDistance(taker_coord[0], taker_coord[1], opp_coord[0], opp_coord[1]))
            self._giver = self._game._entities[opponents[np.argmin(np.array(dists))]["entity_id"]]

    def getPassAvailability(self):
        time_key = np.argmin((np.absolute(np.array(list(self._game._entities['1']._hd_UTC_update.keys())) - self._UTC_update)))
        posessor = np.array([self._game._entities['1']._hd_UTC_update[list(self._game._entities['1']._hd_UTC_update.keys())[time_key]]["X"], self._game._entities['1']._hd_UTC_update[list(self._game._entities['1']._hd_UTC_update.keys())[time_key]]["Y"]])
        betas, min_betas_lst, min_betas_player = [], [], []

        teammates = self._game.getTeammatesOnIce(self._giver._team, self._UTC_update)

        t = 0.25
        time_key = getUTCtimekey(self._taker, self._UTC_update)
        opp = np.array([self._taker._hd_UTC_update[time_key]["X"], self._taker._hd_UTC_update[time_key]["Y"]])

        lanes = []
        for m in teammates:
            if m["entity_id"] != self._giver._id:
                mate_obj = self._game._entities[m["entity_id"]]
                if mate_obj._pos == 'G': continue
                time_key_mate = getUTCtimekey(mate_obj, self._UTC_update)
                mate = np.array([mate_obj._hd_UTC_update[time_key_mate]["X"], mate_obj._hd_UTC_update[time_key_mate]["Y"]])


                a = passing_lane_functions.getValueOfA(posessor,mate,opp,t, bounds=np.array([0,2,4]))
                lanes.append(a)
        self._takeaway_difficulty = np.amin(np.array(lanes))
        self._taker._takeaway_difficulty.append(self._takeaway_difficulty)

    def __repr__(self):
        return "TakeAway(Period: {}, Clock: {}:{}, taker: {}, giver: {}, Difficulty: {})".format(self._period, self._clock_minutes, self._clock_seconds, self._taker._last_name, self._giver._last_name, self._takeaway_difficulty)


class Pass(object):
    def __init__(self, game, time, descriptor, info):
        self._time = time # when the pass was caught
        self._game = game
        self._UTC_update = info[0]["MarkerUTC"]
        self.getPassData(game, descriptor, info)
        self._distance = euclideanDistance(self._origin["X"], self._origin["Y"], self._destination["X"], self._destination["Y"])
        self.getInformation(info)
        self.beat = 0
        self._overtook_ids = []
        self.getPassRisk()
        self.beatOpponents(info, game)


        # FIXME: origin of the pass is where the passer is at _UTC_update and the destination (x,y) is where the receiver is at self._time

    def getPassData(self, game, descriptor, info_participants):
        desc_arr = descriptor.split(" ")

        for i in info_participants[0]["Participants"]:
            if i["Role"] == "MapRolePasser":
                passer_id = i["EntityId"]
            elif i["Role"] == "MapRolePassReceiver":
                receiver_id = i["EntityId"]

        passer_name = desc_arr[2]
        receiver_name = desc_arr[-1]

        for e in game._entities.keys():
            if game._entities[e]._last_name == passer_name:
                self._passer = game._entities[e]
                # print(self._passer._hd_UTC_update[round(self._time,1)])
                self._passer.total_passes+=1
            if game._entities[e]._last_name == receiver_name:
                self._receiver = game._entities[e]
                # print(self._receiver._hd_UTC_update[round(self._time,1)])

        # fig, ax = game.getRinkPlot()
        # print(game._entities["1"]._hd_UTC_update[round(self._time,1)])
        # print(info_participants[0]["Participants"][0]["Location"])
        # exit()
        # plt.scatter(info_participants[0]["Participants"][0]["Location"]["X"], info_participants[0]["Participants"][0]["Location"]["Y"], s=10, c='m', label="Marker")
        # plt.scatter(game._entities["1"]._hd_UTC_update[round(self._time,1)]["X"], game._entities["1"]._hd_UTC_update[round(self._time,1)]["Y"], s=10, c='k', label="HD")
        # plt.scatter(self._passer._hd_UTC_update[round(self._time,1)]["X"], self._passer._hd_UTC_update[round(self._time,1)]["Y"], c='r', label="Passer")
        # plt.scatter(self._receiver._hd_UTC_update[round(self._time,1)]["X"], self._receiver._hd_UTC_update[round(self._time,1)]["Y"], c='c', label="Receiver")
        # plt.legend()
        # plt.show()
        # exit()

        # try:
        # time_key = getUTCtimekey(self._passer, self._time)

        arg_key = np.argmin(np.abs(np.array(list(self._passer._hd_UTC_update.keys())) - self._UTC_update))
        key = list(self._passer._hd_UTC_update.keys())[arg_key]

        self._origin =  {"X": self._passer._hd_UTC_update[round(key,1)]["X"], "Y": self._passer._hd_UTC_update[round(key,1)]["Y"]} #info_participants[0]["Participants"][1]["Location"]
        self._destination = info_participants[0]["Participants"][0]["Location"] # NOTE: THIS IS SET TO THE PUCK... THE PUCKS "LOCATION" IS WHEN IT IS CAUGHT



        # get speed of pass
        arg_key = np.argmin(np.abs(np.array(list(game._entities['1']._hd_UTC_update.keys())) - self._UTC_update))
        key = list(game._entities['1']._hd_UTC_update.keys())[arg_key]

        time_key = round(key,1)
        time_key_p1 = round(list(game._entities['1']._hd_UTC_update.keys())[arg_key+1],1)
        time_diff = time_key_p1 - time_key
        puck_t0 =  {"X": game._entities['1']._hd_UTC_update[time_key]["X"], "Y": game._entities['1']._hd_UTC_update[time_key]["Y"]} #info_participants[0]["Participants"][1]["Location"]
        puck_t1 =  {"X": game._entities['1']._hd_UTC_update[time_key_p1]["X"], "Y": game._entities['1']._hd_UTC_update[time_key_p1]["Y"]}
        self._speed = euclideanDistance(puck_t0["X"], puck_t0["Y"], puck_t1["X"], puck_t1["Y"])
        self._speed = self._speed / (10*time_diff)

        # except:
        #     time_idx = np.where((np.abs(np.array(self._passer._update_time) - round(info_participants[0]["MarkerUTC"],1)) < 0.5))[0][0]
        #     self._origin = {"X": self._passer._locX[time_idx], "Y": self._passer._locY[time_idx], "Z": self._passer._locZ[time_idx]} #info_participants[0]["Participants"][1]["Location"]
        #     self._destination = info_participants[0]["Participants"][0]["Location"] # NOTE: THIS IS SET TO THE PUCK... THE PUCKS "LOCATION" IS WHEN IT IS CAUGHT

        # self.getRealPassOrigin()

        # fig, ax = game.getRinkPlot()
        # plt.scatter(self._origin["X"], self._origin["Y"], s=40, c='r', label='Origin')
        # plt.scatter(self._destination["X"], self._destination["Y"], s=40, c='c', label='Destination')

        # print(info_participants[0]["Participants"][2])
        # print(info_participants)
        # exit()

        # received_time_frame = np.where((np.abs(np.array(self._receiver._update_time) - self._time) < 0.5))[0][0]
        # print(received_time_frame)
        # print(self._time)
        # print(self._receiver._locX[received_time_frame:received_time_frame+10])
        # print(self._receiver._locY[received_time_frame:received_time_frame+10])
        # print(self._destination)
        # exit()

    def getPassRisk(self):
        ''' Calculates passlanes using beta skeletons
        minimum beta is the most covered the person is
        '''
        # arg_key = np.argmin(np.abs(np.array(list(opp_obj._hd_UTC_update.keys())) - self._UTC_update))
        # key = list(opp_obj._hd_UTC_update.keys())[arg_key]

        # posessor = np.array([self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"]])
        time_key = np.argmin((np.abs(np.array(list(self._game._entities['1']._hd_UTC_update.keys())) - self._UTC_update)))
        key = list(self._game._entities['1']._hd_UTC_update.keys())[time_key]
        posessor = np.array([self._game._entities['1']._hd_UTC_update[key]["X"], self._game._entities['1']._hd_UTC_update[key]["Y"]])

        self._betas = []
        min_betas_lst, min_betas_player = [], []

        arg_key = np.argmin(np.abs(np.array(list(self._receiver._hd_UTC_update.keys())) - self._UTC_update))
        key = list(self._receiver._hd_UTC_update.keys())[arg_key]
        # key = getUTCtimekey(self._receiver, self._UTC_update)
        mate = np.array([self._receiver._hd_UTC_update[key]["X"], self._receiver._hd_UTC_update[key]["Y"]])

        pass_availability, threat_o = getPassingLane_General(self._game, posessor, self._receiver, self._UTC_update)

        self._threat_o = threat_o
        self._pass_risk = pass_availability
        self._passer._pass_risk.append(pass_availability)
        return


    def getRealPassOrigin(self):
        puck = self._game._entities["1"]
        puck_keys = list(puck._hd_UTC_update.keys())
        puck_min = np.argmin(np.absolute(np.array(puck_keys) - self._time))
        puck_keys = np.array(list(puck._hd_UTC_update.keys())[(puck_min - 50):(puck_min)])

        passer_keys = list(self._passer._hd_UTC_update.keys())
        passer_min = np.argmin(np.absolute(np.array(passer_keys) - self._time))
        passer_keys = np.array(list(self._passer._hd_UTC_update.keys())[(passer_min - 50):(passer_min)])

        dists = []
        for i, val in enumerate(puck_keys):
            dists.append(euclideanDistance(puck._hd_UTC_update[val]["X"], puck._hd_UTC_update[val]["Y"], self._passer._hd_UTC_update[passer_keys[i]]["X"], self._passer._hd_UTC_update[passer_keys[i]]["Y"]))

        puck_min = np.argmin(np.array(dists))
        self._origin = {"X": puck._hd_UTC_update[puck_keys[puck_min]]["X"],
                    "Y": puck._hd_UTC_update[puck_keys[puck_min]]["Y"],
                    "Z": puck._hd_UTC_update[puck_keys[puck_min]]["Z"]}


        # self._time = puck_keys[puck_min]

    def getInformation(self, info):
        # print(info[0]["Properties"][0])
        # exit()
        self._period = info[0]["ETime"]["Period"]
        self._homeplate = info[0]["Properties"][0]["IsInHomePlateArea"]

    def plotPass(self, game, opponents, teammates, beat=None, fig=None, ax=None):
        if fig is None:
            fig, ax = game.getRinkPlot()

        # arg_key = np.argmin(np.abs(np.array(list(self._passer._hd_UTC_update.keys())) - self._UTC_update))
        # key = list(self._passer._hd_UTC_update.keys())[arg_key]
        # self._origin =  {"X": self._passer._hd_UTC_update[round(key,1)]["X"], "Y": self._passer._hd_UTC_update[round(key,1)]["Y"]} #info_participants[0]["Participants"][1]["Location"]


        for i in teammates:
            mate = game._entities[i["entity_id"]]
            # key = round(mate._update_time[i["time_idx"]], 1)
            arg_key = np.argmin(np.abs(np.array(list(mate._hd_UTC_update.keys())) - self._UTC_update))
            key = list(mate._hd_UTC_update.keys())[arg_key]
            plt.scatter(mate._hd_UTC_update[key]["X"], mate._hd_UTC_update[key]["Y"], c=game._entities[i["entity_id"]]._color, label=game._entities[i["entity_id"]]._number+game._entities[i["entity_id"]]._last_name)
            ax.annotate(game._entities[i["entity_id"]]._number,(mate._hd_UTC_update[key]["X"], mate._hd_UTC_update[key]["Y"]))

        plt.scatter(self._origin["X"], self._origin["Y"], s=40, c='r', label='Origin')
        plt.scatter(self._destination["X"], self._destination["Y"], s=40, c='c', label='Destination')
        plt.plot([self._origin["X"], self._destination["X"]], [self._origin["Y"], self._destination["Y"]], c='r')

        plt.scatter(self._threat_o[0], self._threat_o[1], s=240, c='r', label='Threat')

        if beat is not None:
            plt.scatter(game._entities[beat["entity_id"]]._locX[beat["time_idx"]], game._entities[beat["entity_id"]]._locY[beat["time_idx"]], s = 100, c='y', label="Beaten")

        for i in opponents:
            opp = game._entities[i["entity_id"]]
            # key = round(opp._update_time[i["time_idx"]], 1)
            arg_key = np.argmin(np.abs(np.array(list(opp._hd_UTC_update.keys())) - self._UTC_update))
            key = list(opp._hd_UTC_update.keys())[arg_key]
            plt.scatter(opp._hd_UTC_update[key]["X"], opp._hd_UTC_update[key]["Y"], c=game._entities[i["entity_id"]]._color, label=game._entities[i["entity_id"]]._number+game._entities[i["entity_id"]]._last_name)
            ax.annotate(game._entities[i["entity_id"]]._number,(opp._hd_UTC_update[key]["X"], opp._hd_UTC_update[key]["Y"]))

        plt.title(self)
        plt.legend()
        plt.show()


    # def passBeta(self, i, game):
    #     ''' return true if player is in a small beta for a pass '''
    #
    #     opp = [game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]]]
    #
    #     passer_loc = np.array([self._origin["X"], self._origin["Y"]])
    #     receiver_loc = np.array([self._destination["X"], self._destination["Y"]])
    #     circle_mid = receiver_loc
    #     # r = euclideanDistance(passer_loc[0], passer_loc[1], receiver_loc[0], receiver_loc[1])
    #     dist_to_mate = euclideanDistance(passer_loc[0], passer_loc[1], receiver_loc[0], receiver_loc[1])
    #     dist_to_opp = euclideanDistance(passer_loc[0], passer_loc[1], opp[0], opp[1])
    #
    #     ''' find po~r '''
    #     v1 = passer_loc - opp
    #     mate_proj = receiver_loc - (passer_loc - receiver_loc)
    #     v2 = mate_proj - opp
    #     rad = angle_between_vectors(v1, v2)
    #
    #     inPressureZone, dist_mate_to_opp = isInPressureZone(self._game, receiver_loc, opp, self._receiver, self._period)
    #
    #     ''' do degree passing lane below '''
    #     # if dist_to_opp <= dist_to_mate or inPressureZone:
    #     #     v1 = passer_loc - opp
    #     #     v2 = passer_loc - receiver_loc
    #     #     beta = np.absolute(np.degrees(angle_between_vectors(v1, v2)))
    #     #
    #     #     if np.absolute(beta) < 5:
    #     #         return True
    #     ''' do degree passing lane above '''
    #
    #     ''' do HCBS below '''
    #     is_in_corridor = isOppInCorridor(passer_loc, receiver_loc, opp)
    #     if is_in_corridor or inPressureZone:
    #         beta = np.sin(rad)
    #     else:
    #         beta = 1 / np.sin(rad)
    #     ''' do HCBS above '''
    #
    #     # if (dist_mate_to_opp <= r and dist_to_opp <= r) or inPressureZone:
    #     #     beta = np.sin(rad)
    #     # else:
    #     #     beta = 1 / np.sin(rad)
    #     if beta < np.sin(2.8):
    #         return True
    #     return False


    def plotDirection(self):
        opponents = self._game.getOpponentsOnIce(self._passer._team, self._UTC_update)
        teammates = self._game.getTeammatesOnIce(self._passer._team, self._UTC_update)

        gammas = np.linspace(0,2,10)
        # for gam in gammas:
        #     print("gam: ", gam)
        fig, ax = plotGamma(self._game, self._pass_risk, [self._origin["X"], self._origin["Y"]], [self._destination["X"], self._destination["Y"]])
        #     # plt.scatter(self._origin["X"], self._origin["Y"], c='r')
        #     # plt.scatter(self._destination["X"], self._destination["Y"], c='b')
        #     # plt.scatter(self._attacking_net["X"], self._attacking_net["Y"], c='k')
        #     # plt.show()
        #
        self.plotPass(self._game, opponents, teammates, fig=fig, ax=ax)
        # exit()


    def getDirection(self):
        self._attacking_net,_ = self._game.getAttackingNet(self._passer._team, self._period)
        rad_to_net = getRadians(self._origin["X"], self._origin["Y"], self._attacking_net["X"], self._attacking_net["Y"])
        rad_to_mate = getRadians(self._origin["X"], self._origin["Y"], self._destination["X"], self._destination["Y"])
        # self._direction = np.degrees(rad_to_net - rad_to_mate)

        origin = np.array([self._origin["X"], self._origin["Y"]])
        dest = np.array([self._destination["X"], self._destination["Y"]])
        net = np.array([self._attacking_net["X"], self._attacking_net["Y"]])

        # print(origin)
        # print(dest)
        # print(net)

        v1 = origin - net
        v2 = origin - dest
        rad = angle_between_vectors(v1, v2)
        # print("before shift: rad: ", rad, " deg: ", np.degrees(rad))
        self._direction = rad - (np.pi/2)
        # print("after shift: rad: ", self._direction, " deg: ", np.degrees(self._direction))
        # print()


        # vector / np.linalg.norm(vector)
        # v1_u = v1 / np.linalg.norm(v1) #unit_vector(v1)
        # v2_u = v2 / np.linalg.norm(v2) #unit_vector(v2)
        # ang = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        # print("angle: ", np.degrees(ang))
        # print()


        # else:
        #     self._direction = (rad - (np.pi))
        # print(np.degrees(rad))
        #
        # # print(self._direction)
        # print("net: ", np.degrees(rad_to_net))
        # print("mate: ", np.degrees(rad_to_mate))
        # exit()


        # rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)

    def resetPass(self):
        self._pass_overtook = 0
        self._beat_by_pass = 0
        self.beat = 0
        self._overtook_ids = []


    def beatOpponents(self):
        ''' find which opponents were beaten with the pass
        TODO: use radius around players to determine if the pass was close (necessary?)
        '''

        # attacking_net,_ = self._game.getAttackingNet(self._passer._team, self._period)
        pass_origin_len = euclideanDistance(self._attacking_net["X"], self._attacking_net["Y"], self._origin["X"], self._origin["Y"])
        pass_dest_len = euclideanDistance(self._attacking_net["X"], self._attacking_net["Y"], self._destination["X"], self._destination["Y"])


        opponents = self._game.getOpponentsOnIce(self._passer._team, self._time)
        teammates = self._game.getTeammatesOnIce(self._passer._team, self._time)

        can_overtake = 0
        for i in opponents:
            opp_net_diff = euclideanDistance(self._attacking_net["X"], self._attacking_net["Y"], self._game._entities[i["entity_id"]]._locX[i["time_idx"]], self._game._entities[i["entity_id"]]._locY[i["time_idx"]])
            if pass_origin_len > opp_net_diff and self._game._entities[i["entity_id"]]._pos != 'G':
                can_overtake+=1

        overtook_count = 0
        self._pass_overtook, self._beat_by_pass = 0, 0
        for i in opponents:
            opp_net_diff = euclideanDistance(self._attacking_net["X"], self._attacking_net["Y"], self._game._entities[i["entity_id"]]._locX[i["time_idx"]], self._game._entities[i["entity_id"]]._locY[i["time_idx"]])
            # self.plotPass(self._game, opponents, teammates, i)

            # if self.passBeta(i, self._game):
            #     self.plotPass(self._game, opponents, teammates, i)
                # exit()

            key = np.argmin(np.abs(np.array(list(self._game._scoreboard.keys())) - self._UTC_update))
            key = list(self._game._scoreboard.keys())[key]

            home_strength = self._game._scoreboard[key]["HomeStrength"]
            visitor_strength = self._game._scoreboard[key]["VisitorStrength"]

            # if (pass_origin_len > opp_net_diff and pass_dest_len < opp_net_diff and home_strength == 6 and visitor_strength == 6) or self.passBeta(i, self._game):
            # if (pass_origin_len > opp_net_diff and pass_dest_len < opp_net_diff and home_strength == 6 and visitor_strength == 6) or (self._pass_risk < 0.1):
            if pass_origin_len > opp_net_diff and pass_dest_len < opp_net_diff and home_strength == 6 and visitor_strength == 6 and self._game._entities[i["entity_id"]]._pos != 'G':
                self._game._entities[i["entity_id"]]._beaten.append(self._passer._id)
                self._game._entities[i["entity_id"]]._beaten_time[round(self._time,1)] = True

                if can_overtake == 0:
                    overtook_one = 0
                else:
                    overtook_one = (1/can_overtake)

                # scale overtook
                self._passer._overtook_val+=overtook_one
                self._game._entities[i["entity_id"]]._beaten_val+=overtook_one
                self._game._entities[i["entity_id"]]._beaten_ratio.append(overtook_one)


                self._pass_overtook+=overtook_one
                self._beat_by_pass+=overtook_one

                self.beat+=overtook_one

                self._passer._overtook.append(i["entity_id"])
                self._overtook_ids.append(i["entity_id"])
                overtook_count+=1
                # self.beat+=1
                # print("passer: ", self._passer._last_name, " receiver: ", self._receiver._last_name, " beat: ", self._game._entities[i["entity_id"]]._last_name)
                # self.plotPass(self._game, opponents, teammates, i)

        # for o in self._overtook_ids:
            # self._game._entities[o]._beaten_ratio.append(overtook_count/can_overtake)
        # if can_overtake == 0:
        #     self._passer._overtook_ratio.append(0)
        # else:
        if overtook_count > 0:
            self._passer._overtook_ratio.append(overtook_count/can_overtake)

        self._passer._overtook_time[round(self._time,1)] = overtook_count
        # return self._pass_overtook - self._beat_by_pass, self._pass_overtook, self._beat_by_pass




# TODO: record the high res data for each player's location and use that at the time of the pass... just use events to get the UTC time of a pass

    def __repr__(self):
        return "Pass(Team: {}, From: {}, To: {}, Speed (ft/0.1s): {}, Distance: {} ft, Overtook: {}, Period: {}, Risk: {})".format(self._passer._team, self._passer._last_name, self._receiver._last_name, self._speed, round(self._distance,1), self.beat, self._period, self._pass_risk)




class GameSecond(object):
    def __init__(self, elapsed_seconds, game, time):
        self._elapsed_seconds = elapsed_seconds
        self._game = game
        self._time = time
        self._stats = {}
        self.initSecond()
        self.updateSecond(self._time)

    def initSecond(self):
        for player_info in list(self._game._game_stats[self._time].keys()):
            player = self._game._game_stats[self._time][player_info]
            team_id = self.getTeamId(player)
            if team_id not in list(self._stats.keys()):
                self._stats[team_id] = {}
                self._stats[team_id]["goals"] = 0
                self._stats[team_id]["hits"] = 0
                self._stats[team_id]["passes"] = 0
                self._stats[team_id]["pass_plus_minus"] = 0
                self._stats[team_id]["blocks"] = 0
                self._stats[team_id]["shots"] = 0
                self._stats[team_id]["shot_attempts"] = 0
                self._stats[team_id]["turnover"] = 0 #False
                self._stats[team_id]["possession"] = 0 #False

    def updateSecond(self, t):
        # block_info = {}
        for player_info in list(self._game._game_stats[t].keys()):
            player = self._game._game_stats[t][player_info]
            team_id = self.getTeamId(player)
            # if team_id not in block_info.keys(): block_info[team_id] = 0
            self._stats[team_id]["goals"]+=player["goals"]
            self._stats[team_id]["hits"]+=player["hits"]
            self._stats[team_id]["passes"]+=player["passes"]
            self._stats[team_id]["pass_plus_minus"]+=player["pass_plus_minus"]
            self._stats[team_id]["blocks"]+=player["blocks"]
            # block_info[team_id] += player["blocks"]
            self._stats[team_id]["shots"]+=player["shots"]
            self._stats[team_id]["shot_attempts"]+=player["shot_attempts"]
            self._stats[team_id]["turnover"]+=player["turnover"]
            self._stats[team_id]["possession"]+=player["possession"]
            # if player["turnover"]: self._stats[team_id]["turnover"] = True
            # if player["possession"]: self._stats[team_id]["possession"] = True
            if self._stats[team_id]["possession"] > 1: self._stats[team_id]["possession"] = 1 #boolean

        # if np.sum(np.array(list(block_info.values()))) > 0:
        #     for team_id in self._stats.keys():
        #         print("before: ", self._stats[team_id]["shots"], " blocks: ", block_info[list(self._stats.keys())-team_id])
        #         self._stats[team_id]["shots"]  = self._stats[team_id]["shots"] - block_info[list(self._stats.keys())-team_id]
        #         print("after: ", self._stats[team_id]["shots"])

    def getTeamId(self, player):
        if player["player_team"] == "TAMPA BAY LIGHTNING":
            return "14"
        return "25"

    def __repr__(self):
        col_lst = ['Metric', 'Tampa_Bay', 'Dallas']
        second = pd.DataFrame(columns=col_lst)
        for i, val in enumerate(list(self._stats["14"].keys())):
            second.loc[i] = [val, self._stats["14"][val], self._stats["25"][val]]
        second.loc[len(list(self._stats["14"].keys()))] = ["Seconds", self._elapsed_seconds, self._elapsed_seconds]
        return repr(second)
