import json
import sys
import numpy as np
import pandas as pd
import pickle
import math
import os
import matplotlib.pyplot as plt

# fake
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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





def getUTCtimekey(obj, time):
    key = np.argmin(np.abs(np.array(list(obj._hd_UTC_update.keys())) - time))
    return list(obj._hd_UTC_update.keys())[key]

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
            plt.legend()
            plt.title("Posession: "+ self._posessor._last_name+" "+str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"])+" Duration: "+str(self._clocktime_seconds), fontsize=20)

            if border is not None:
                plt.scatter(border[0], border[1], s=10, c='m',label="Pressure Boundary")

            # theta = np.linspace(0, 2*np.pi, 100)
            # r = np.sqrt(30)
            # x1 = r*np.cos(theta)
            # x2 = r*np.sin(theta)
            # plt.plot(x1, x2)

            plt.show()
            # exit()

    def plotPosessionTimestep(self, time, border=None):

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
                plt.scatter(mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["X"], mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["Y"], c=mate._color, label=mate._number+mate._last_name)
                ax.annotate(mate._number,(mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["X"], mate._hd_UTC_update[round(mate._update_time[i["time_idx"]],1)]["Y"]))


        plt.scatter(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], s=120, c='y', label="Posessor")
        plt.scatter(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], c=self._posessor._color, label=self._posessor._last_name)

        plt.arrow(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], (self._posessor._hd_UTC_update[time]["X"]+self._posessor._hd_UTC_update[time]["velX"]),
                    (self._posessor._hd_UTC_update[time]["Y"]+self._posessor._hd_UTC_update[time]["velY"]), length_includes_head=True,head_width=1, head_length=3)
        plt.scatter(self._game._entities['1']._hd_UTC_update[time]["X"], self._game._entities['1']._hd_UTC_update[time]["Y"], s=10, c=self._game._entities['1']._color, label="Puck")
        plt.legend()
        plt.title("Posession: "+ self._posessor._last_name+" "+str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"])+" Duration: "+str(self._clocktime_seconds), fontsize=20)

        if border is not None:
            plt.scatter(border[0], border[1], s=10, c='m',label="Pressure Boundary")

        plt.show()


    def getPressure(self, time):
        ''' Calculates pressure similar to paper: A visual analysis of pressure in football'''

        q = 1
        # self._posession_pressures = {}
        attacking_net = self._game.getAttackingNet(self._posessor._team, self._period)

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
                pressure = 100 + (((1 - d) / L) ** q) * 100

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
        posessor = np.array([self._game._entities['1']._hd_UTC_update[time]["X"], self._game._entities['1']._hd_UTC_update[time]["Y"]])
        self._betas = []
        min_betas_lst, min_betas_player = [], []

        for m in self._teammates:
            # print()
            if m["entity_id"] != self._posessor._id:
                mate_obj = self._game._entities[m["entity_id"]]
                if mate_obj._pos == 'G': continue
                # print(mate_obj._last_name)
                time_key = getUTCtimekey(mate_obj, time)
                mate = np.array([mate_obj._hd_UTC_update[time_key]["X"], mate_obj._hd_UTC_update[time_key]["Y"]])
                dist = euclideanDistance(posessor[0], posessor[1], mate[0], mate[1])
                r = dist / 2
                circle_mid = np.mean(np.array([posessor, mate]), axis=0)

                betas = []
                for o in self._opponents:
                    opp_obj = self._game._entities[o["entity_id"]]
                    if opp_obj._pos == 'G': continue
                    time_key = getUTCtimekey(opp_obj, time)
                    opp = np.array([opp_obj._hd_UTC_update[time_key]["X"], opp_obj._hd_UTC_update[time_key]["Y"]])

                    # betas.append(distance_numpy(posessor, mate, opp) / 200)

                    # print(circle_mid, " ", opp)
                    dist_to_opp = euclideanDistance(circle_mid[0], circle_mid[1], opp[0], opp[1])

                    v1 = posessor - opp
                    # print(v1)
                    v2 = mate - opp
                    # print(v2)
                    rad = angle_between_vectors(v1, v2)
                    # print(rad)
                    # exit()
                    # rad = getPRQradians(posessor, opp, mate)

                    if dist_to_opp > r: #beta > 1
                        beta = 1 / np.sin(rad)
                    else:
                        beta = np.sin(rad)
                    # print(opp_obj._last_name, ": ", round(beta,3), " radius: ", r, " radians: ", rad, " dist: ", dist_to_opp)

                    # if self._posessor._number == "27" and opp_obj._number == "47" and mate_obj._number == "22":
                    #     print("rad: ", rad)
                    #     print("midpoint: ", circle_mid)
                    #     print("radius: ", r)
                    #     print("beta: ", beta)
                    #     print("dist to opp: ", dist_to_opp)
                    #     print(posessor)
                    #     print(opp)
                    #     print(mate)
                    #     exit()
                    betas.append(beta)
                min_beta = np.amin(np.array(betas))
                min_betas_lst.append(min_beta)
                min_betas_player.append(mate_obj)
                # print(min_beta)
                self._betas.append((mate,min_beta))

        return min_betas_player[np.argmax(min_betas_lst)]


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
        return "Posession(Team: {}, Player: {}, Period: {}, Clock: {}, Duration: {}, UTC_diff: {}, Turnover?: {}, PassPressure: {})".format(self._posessor._team, self._posessor._last_name, self._period, str(self._clocktime_start["ClockMinutes"])+":"+str(self._clocktime_start["ClockSeconds"]),
                                            self._clocktime_seconds, self._posession_UTC_keys.shape[0], self._turnover, self._pressure)

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
            self._attacking_net = self._game.getAttackingNet(self._shooter._team, self._period)
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
        attacking_net = self._game.getAttackingNet(self._shooter._team, self._period)

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
                pressure = 100 + (((1 - d) / L) ** q) * 100

                if d < L:
                    timestep_pressure.append(pressure)
                    timestep_pressers.append(opp)
                    self._shot_presser_lst[opp._id] = pressure

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

        key = np.argmin(np.abs(np.array(list(self._passer._hd_UTC_update.keys())) - self._UTC_update))
        key = list(self._passer._hd_UTC_update.keys())[key]

        self._origin =  {"X": self._passer._hd_UTC_update[round(key,1)]["X"], "Y": self._passer._hd_UTC_update[round(key,1)]["Y"]} #info_participants[0]["Participants"][1]["Location"]
        self._destination = info_participants[0]["Participants"][0]["Location"] # NOTE: THIS IS SET TO THE PUCK... THE PUCKS "LOCATION" IS WHEN IT IS CAUGHT
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

    def getRealPassOrigin(self):
        puck = self._game._entities["1"]
        puck_keys = list(puck._hd_UTC_update.keys())
        puck_min = np.argmin(np.absolute(np.array(puck_keys) - self._time))
        puck_keys = np.array(list(puck._hd_UTC_update.keys())[(puck_min - 5):(puck_min)])

        passer_keys = list(self._passer._hd_UTC_update.keys())
        passer_min = np.argmin(np.absolute(np.array(passer_keys) - self._time))
        passer_keys = np.array(list(self._passer._hd_UTC_update.keys())[(passer_min - 5):(passer_min)])

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

    def plotPass(self, game, opponents, teammates, beat):
        fig, ax = game.getRinkPlot()

        for i in teammates:
            mate = game._entities[i["entity_id"]]
            key = round(mate._update_time[i["time_idx"]], 1)
            plt.scatter(mate._hd_UTC_update[key]["X"], mate._hd_UTC_update[key]["Y"], c=game._entities[i["entity_id"]]._color, label=game._entities[i["entity_id"]]._number+game._entities[i["entity_id"]]._last_name)
            ax.annotate(game._entities[i["entity_id"]]._number,(mate._hd_UTC_update[key]["X"], mate._hd_UTC_update[key]["Y"]))

        plt.scatter(self._origin["X"], self._origin["Y"], s=40, c='r', label='Origin')
        plt.scatter(self._destination["X"], self._destination["Y"], s=40, c='c', label='Destination')
        plt.plot([self._origin["X"], self._destination["X"]], [self._origin["Y"], self._destination["Y"]], c='r')

        plt.scatter(game._entities[beat["entity_id"]]._locX[beat["time_idx"]], game._entities[beat["entity_id"]]._locY[beat["time_idx"]], s = 100, c='y', label="Beaten")

        for i in opponents:
            opp = game._entities[i["entity_id"]]
            key = round(opp._update_time[i["time_idx"]], 1)
            plt.scatter(opp._hd_UTC_update[key]["X"], opp._hd_UTC_update[key]["Y"], c=game._entities[i["entity_id"]]._color, label=game._entities[i["entity_id"]]._number+game._entities[i["entity_id"]]._last_name)
            ax.annotate(game._entities[i["entity_id"]]._number,(opp._hd_UTC_update[key]["X"], opp._hd_UTC_update[key]["Y"]))
        plt.legend()
        plt.show()

        # for i in teammates:
        #     plt.scatter(game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]], c=game._entities[i["entity_id"]]._color, label=game._entities[i["entity_id"]]._number+game._entities[i["entity_id"]]._last_name)
        #     ax.annotate(game._entities[i["entity_id"]]._number,(game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]]))
        #
        # plt.scatter(self._origin["X"], self._origin["Y"], s=40, c='r', label='Origin')
        # plt.scatter(self._destination["X"], self._destination["Y"], s=40, c='c', label='Destination')
        # plt.plot([self._origin["X"], self._destination["X"]], [self._origin["Y"], self._destination["Y"]], c='r')
        #
        # plt.scatter(game._entities[beat["entity_id"]]._locX[beat["time_idx"]], game._entities[beat["entity_id"]]._locY[beat["time_idx"]], s = 100, c='y', label="Beaten")
        #
        # for i in opponents:
        #     plt.scatter(game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]], c=game._entities[i["entity_id"]]._color, label=game._entities[i["entity_id"]]._number+game._entities[i["entity_id"]]._last_name)
        #     ax.annotate(game._entities[i["entity_id"]]._number,(game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]]))
        # plt.legend()
        # plt.show()
        # exit()

    def beatOpponents(self, info, game):
        ''' find which opponents were beaten with the pass
        TODO: use radius around players to determine if the pass was close (necessary?)
        '''

        attacking_net = game.getAttackingNet(self._passer._team, self._period)
        pass_origin_len = euclideanDistance(attacking_net["X"], attacking_net["Y"], self._origin["X"], self._origin["Y"])
        pass_dest_len = euclideanDistance(attacking_net["X"], attacking_net["Y"], self._destination["X"], self._destination["Y"])


        opponents = game.getOpponentsOnIce(self._passer._team, self._time)
        teammates = game.getTeammatesOnIce(self._passer._team, self._time)
        overtook_count = 0
        for i in opponents:
            opp_net_diff = euclideanDistance(attacking_net["X"], attacking_net["Y"], game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]])
            # self.plotPass(game, opponents, teammates, i)
            if pass_origin_len > opp_net_diff and pass_dest_len < opp_net_diff and len(opponents) == 6 and len(teammates) == 6:
                game._entities[i["entity_id"]]._beaten.append(self._passer._id)
                game._entities[i["entity_id"]]._beaten_time[round(self._time,1)] = True

                self._passer._overtook.append(i["entity_id"])
                self._overtook_ids.append(i["entity_id"])
                overtook_count+=1
                self.beat+=1
                # print("passer: ", self._passer._last_name, " receiver: ", self._receiver._last_name, " beat: ", game._entities[i["entity_id"]]._last_name)
                # self.plotPass(game, opponents, teammates, i)

        self._passer._overtook_time[round(self._time,1)] = overtook_count

# TODO: record the high res data for each player's location and use that at the time of the pass... just use events to get the UTC time of a pass

    def __repr__(self):
        return "Pass(Team: {}, From: {}, To: {}, Distance: {} ft, Overtook: {}, Period: {})".format(self._passer._team, self._passer._last_name, self._receiver._last_name, round(self._distance,1), self.beat, self._period)
