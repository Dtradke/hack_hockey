import json
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rink


def loadPlayer(fname):
    with open(fname, "rb") as f:
        player = pickle.load(f)
    return player


class Team(object):
    def __init__(self, team):
        self._teamId = team["OfficialId"]
        self._VisOrHome = team["VisOrHome"]
        self._city = team["City"]
        self._mascot = team["Mascot"]
        self._full_name = team["FullName"]
        self._code = team["Code"]
        self._conference = team["Conference"]
        self._division = team["Division"]
        self._posessions = []

    def __repr__(self):
        return "Team(ID: {}, Name: {})".format(self._teamId, self._full_name)




class Player(object):
    def __init__(self, info=None, game=None):
        self._id = info["EntityId"]
        self._type = info["EntityType"]
        self._VisOrHome = info["VisOrHome"]
        self._number = info["JerseyNum"]
        self._first_name = info["FirstName"]
        self._last_name = info["LastName"]
        self._pos = info["Position"]
        self._height = info["Height"]
        self._weight = info["Weight"]
        self._dob = info["DOB"]
        self._pob = info["POB"]
        self._age = info["Age"]
        self._years_exp = info["YearsExp"]
        self._team = info["EntityTeamId"]
        self._onice = []
        self._counter = []
        self._update_time = []
        self._possession_time = []
        self._locX, self._locY, self._locZ = [], [], []
        self._velX, self._velY, self._velZ = [], [], []
        self._accX, self._accY, self._accZ = [], [], []
        self._posession_lst = []
        self._turnovers = 0
        self._steals = 0
        self._posession_time = 0
        self._pressure_to_pass = [] # array with the pressures on a player when they give up the puck


        # for HD resolution tracking
        self._hd_UTC_update = {}

        # These are for counting passes
        self._beaten = []
        self._beaten_time = {}
        self._overtook = []
        self._overtook_time = {}
        self.total_passes = 0
        self.blocks = 0
        self._pressed_shooter_success = []
        self._pressed_shooter_fail = []
        self._pressure_to_shoot = []



        if self._team == '25':
            self._color = 'g'
            self._size = 30
        elif self._team == '14':
            self._color = 'b'
            self._size = 30
        else:
            self._color = 'k'
            self._size = 10

        # player = data[0]["EntityTracking"]["TrackingData"][0]["EntityId"]


    def plotMovement(self):
        fig = rink.getRink()


        start = self._shifts[0]["start"]
        end = self._shifts[0]["end"]

        # colors = []
        # for i in range(1,end-start+1):
        #     colors.append(np.ones(3,)*i)
        c = np.linspace(0,1,(end-start))
        color_base = np.ones(3,)



        for i in range(start,end):
            color_arr = color_base.reshape(1,-1)*c[i-start]
            if self._team == "25": #TBL colors
                color_arr[0][1] = 0.9
                color_arr[0][2] = 0.2
            else: #DAL colors
                color_arr[0][1] = 0.2
                color_arr[0][2] = 0.9
        # plt.plot(self._locX[start:end], self._locY[start:end], '-o',c=colors[0],linewidth=1)
            if i == start:
                plt.scatter(self._locX[i], self._locY[i],s=30, c=color_arr, label="shiftStart")
            if i == (end-1):
                plt.scatter(self._locX[i], self._locY[i],s=30, c=color_arr, label="shiftEnd")
            else:
                plt.scatter(self._locX[i], self._locY[i],s=30, c=color_arr)
        # plt.scatter(self._locX, self._locY, c=self._locY, cmap="RdYlGn", s=500, edgecolors="black")
        plt.legend()
        plt.show()

    def getShifts(self, rink):
        self._zones = []
        self._shifts = []
        shift = {"start": -1, "end": -1}
        for i, val in enumerate(self._hd_UTC_update.keys()):
            if i > 0:
                previous_key = list(self._hd_UTC_update.keys())[np.argmin(np.absolute(np.array(list(self._hd_UTC_update.keys())) - val)) - 1]
                if self._hd_UTC_update[val]["_onice"] and not self._hd_UTC_update[previous_key]["_onice"]:
                    shift["start"] = val
                if not self._hd_UTC_update[val]["_onice"] and self._hd_UTC_update[previous_key]["_onice"] and shift["start"] != -1:
                    shift["end"] = val-1
                if shift["start"] != -1 and shift["end"] != -1:
                    self._shifts.append(shift)
                    shift = {"start": -1, "end": -1}


    def savePlayer(self, fname):
        fname = fname+'players/'+self._last_name+self._id+".pkl"
        self._fname = fname
        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return fname


    def __repr__(self):
        # try:
        #     return "Player(ID: {}, Name: {} {}, Passes: {}, Overtook: {} ({}%), Beat: {})".format(self._id, self._first_name, self._last_name, self.total_passes, len(self._overtook), round((len(self._overtook)/self.total_passes)*100,1), len(self._beaten))
        # except:
        #     pass
        return "Player(ID: {}, Name: {} {}, Team: {}, Number: {}, Position: {}, Update: {})".format(self._id, self._first_name, self._last_name, self._team, self._number, self._pos, len(self._update_time))
