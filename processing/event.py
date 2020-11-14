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


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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
        self.getPressure()

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

        plt.show()

    def getRadians(self, x1, y1, x2, y2):
        ''' Calculates the degrees of angle between two points... posessor should always be first '''
        return math.atan2(y2-y1, x2-x1)

    def getL(self, rad):
        z = (1 - np.cos(rad)) / 2
        L = self._d_back + ((self._d_front - self._d_back) * (z**3 + (z * 0.3)) / 1.3) #calculate distance limit in that direction
        return L

    def getPressure(self):
        ''' Calculates pressure similar to paper: A visual analysis of pressure in football'''

        q = 1
        self._posession_pressures = {}
        for count, time in enumerate(self._posession_UTC_keys):
            attacking_net = self._game.getAttackingNet(self._posessor._team, self._period)
            rad_to_net = self.getRadians(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], attacking_net["X"], attacking_net["Y"])
            # print(rad_to_net)
            opponents = self._game.getOpponentsOnIce(self._posessor._team, time)
            teammates = self._game.getTeammatesOnIce(self._posessor._team, time)

            border_x, border_y = [], []
            for i in np.linspace(0,(2*np.pi),200):
                L = self.getL(i-rad_to_net)
                border_x.append(self._posessor._hd_UTC_update[time]["X"]-(np.cos(i)*L))
                border_y.append(self._posessor._hd_UTC_update[time]["Y"]-(np.sin(i)*L))


            timestep_pressure = []
            for i in opponents:
                opp = self._game._entities[i["entity_id"]]
                opp_x = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["X"]
                opp_y = opp._hd_UTC_update[round(opp._update_time[i["time_idx"]],1)]["Y"]

                d = euclideanDistance(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], opp_x, opp_y)

                if d <= self._d_front:
                    rad_to_opp = self.getRadians(self._posessor._hd_UTC_update[time]["X"], self._posessor._hd_UTC_update[time]["Y"], opp_x, opp_y)
                    rad_diff = np.pi - np.absolute(rad_to_net - rad_to_opp)
                    L = self.getL(rad_diff)
                    pressure = 100 + (((1 - d) / L) ** q) * 100

                    if d < L:
                        timestep_pressure.append(pressure)

            self._posession_pressures[time] = timestep_pressure

            # print("Pressers: ", len(timestep_pressure), " Pressure: ", np.sum(np.array(timestep_pressure)))
            # self.plotPosessionTimestep(time, border=(border_x,border_y))


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
                                            self._clocktime_seconds, self._posession_UTC_keys.shape[0], self._turnover, self._pressure_at_posession_loss)


class Pass(object):
    def __init__(self, game, time, descriptor, info):
        self._time = time # when the pass was caught
        self.getPassData(game, descriptor, info)
        self._distance = euclideanDistance(self._origin["X"], self._origin["Y"], self._destination["X"], self._destination["Y"])
        self.getInformation(info)
        self.beat = 0
        self.beatOpponents(info, game)



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

        try:
            self._origin = info_participants[0]["Participants"][1]["Location"]
            self._destination = info_participants[0]["Participants"][0]["Location"] # NOTE: THIS IS SET TO THE PUCK... THE PUCKS "LOCATION" IS WHEN IT IS CAUGHT
        except:
            time_idx = np.where((np.abs(np.array(self._passer._update_time) - round(info_participants[0]["MarkerUTC"],1)) < 0.5))[0][0]
            self._origin = {"X": self._passer._locX[time_idx], "Y": self._passer._locY[time_idx], "Z": self._passer._locZ[time_idx]} #info_participants[0]["Participants"][1]["Location"]
            self._destination = info_participants[0]["Participants"][0]["Location"] # NOTE: THIS IS SET TO THE PUCK... THE PUCKS "LOCATION" IS WHEN IT IS CAUGHT


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
        for i in opponents:
            opp_net_diff = euclideanDistance(attacking_net["X"], attacking_net["Y"], game._entities[i["entity_id"]]._locX[i["time_idx"]], game._entities[i["entity_id"]]._locY[i["time_idx"]])
            # self.plotPass(game, opponents, teammates, i)
            if pass_origin_len > opp_net_diff and pass_dest_len < opp_net_diff and len(opponents) == 6 and len(teammates) == 6:
                game._entities[i["entity_id"]]._beaten.append(self._passer._id)
                self._passer._overtook.append(i["entity_id"])
                self.beat+=1
                # print("passer: ", self._passer._last_name, " receiver: ", self._receiver._last_name, " beat: ", game._entities[i["entity_id"]]._last_name)
                # self.plotPass(game, opponents, teammates, i)

# TODO: record the high res data for each player's location and use that at the time of the pass... just use events to get the UTC time of a pass

    def __repr__(self):
        return "Pass(Team: {}, From: {}, To: {}, Distance: {}, Overtook: {}, Period: {})".format(self._passer._team, self._passer._last_name, self._receiver._last_name, self._distance, self.beat, self._period)
