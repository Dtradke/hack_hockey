import json
import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from matplotlib.patches import Rectangle

fname = sys.argv[-1]

def close_event():
    plt.close()

class Game(object):
    def __init__(self, info, teams):
        self._gameId = info["EventId"]
        self._sport = info["Sport"]
        self._league = info["League"]
        self._visitor_team_num = info["VisitorTeamId"]
        self._home_team_num = info["HomeTeamId"]
        self._UTC_start = info["ActualStartUTC"]
        self._UTC_end = info["ActualEndUTC"]
        self._code = info["OfficialCode"]
        self._teams = teams
        self._posessions = {}


    def getRinkPlot(self):
        fig, ax = getRink()
        return fig, ax

    def graphGame(self):
        fig, ax = getRink()

        for e in self._entities.keys():
            if (self._entities[e]._pos != "Referee" or self._entities[e]._pos != "Linesman") and len(self._entities[e]._locX) > 0:
                if self._entities[e]._onice[-1]:
                    ax.scatter(self._entities[e]._locX[-1], self._entities[e]._locY[-1], c=self._entities[e]._color, s=self._entities[e]._size, label=self._entities[e]._number+"-"+self._entities[e]._last_name)
                    if e != '1': ax.annotate(self._entities[e]._number,(self._entities[e]._locX[-1], self._entities[e]._locY[-1]))
                else:
                    ax.scatter(self._entities[e]._locX[-1], self._entities[e]._locY[-1], c=self._entities[e]._color, s=self._entities[e]._size)
                    if e != '1': ax.annotate(self._entities[e]._number,(self._entities[e]._locX[-1], self._entities[e]._locY[-1]))

        # timer = fig.canvas.new_timer(interval = 1000) #creating a timer object and setting an interval of 3000 milliseconds
        # timer.add_callback(close_event)
        # timer.start()
        plt.legend()
        plt.show()

    def getAttackingNet(self, my_team, period):
        ''' get the attacking net location
        NOTE: home attacks right in first period
        '''

        if my_team == '14':             #NOTE: TAMPA IS 14 ----> ASSUMPTION BECAUSE THEY DIDN'T SWITCH SIDES
            if period%2 == 1:
                attacking = 'LeftGoal'
            else:
                attacking = 'RightGoal'
        else:
            if period%2 == 1:
                attacking = 'RightGoal'
            else:
                attacking = 'LeftGoal'

        for z in self._rink._zones.keys():
            if self._rink._zones[z]._name == attacking:
                x_mean = (self._rink._zones[z]._sx + self._rink._zones[z]._ex) / 2
                y_mean = (self._rink._zones[z]._sy + self._rink._zones[z]._ey) / 2
                return {"X": x_mean, "Y": y_mean}


    def getOpponentsOnIce(self, my_team,cur_time,mode=None):
        ''' Returns the keys of the opponents on the ice during an event (pass) '''

        opponents_on_ice = []
        for e in self._entities.keys():
            if e == '1' or self._entities[e]._team == my_team:
                continue
            # print(self._entities[e])
            # if (self._entities[e]._pos != "Referee" or self._entities[e]._pos != "Linesman"):
            # if self._entities[e]._team != my_team:
            try: #Try catches players who dont get updated, like backup goalie
                # if mode == 'posession':
                time_idx = np.where((np.abs(np.array(self._entities[e]._update_time) - cur_time) < 0.5))[0][0]
                if self._entities[e]._onice[time_idx] and self._entities[e]._id != '1':
                    opponents_on_ice.append({'entity_id':e, 'time_idx':time_idx})
                # else:
                #     time_idx = np.where((np.abs(np.array(self._entities[e]._update_time) - cur_time) < 0.5))[0][0]
                #     if self._entities[e]._onice[time_idx] and self._entities[e]._id != '1':
                #         opponents_on_ice.append({'entity_id':e, 'time_idx':time_idx})
            except:
                pass

        # exit()
        return opponents_on_ice

    def getTeammatesOnIce(self, my_team,cur_time):
        ''' Returns the keys of the teammates on the ice during an event (pass) '''
        teammates_on_ice = []
        for e in self._entities.keys():
            if e == '1' or self._entities[e]._team != my_team:
                continue
            # if (self._entities[e]._pos != "Referee" or self._entities[e]._pos != "Linesman"):
                # if self._entities[e]._team == my_team:
            try:
                time_idx = np.where((np.abs(np.array(self._entities[e]._update_time) - cur_time) < 0.5))[0][0]
                if self._entities[e]._onice[time_idx] and self._entities[e]._id != '1':
                    teammates_on_ice.append({'entity_id':e, 'time_idx':time_idx})
            except:
                pass
        return teammates_on_ice



    def __repr__(self):
        return "Game(Id: {}, Home: {}, Away: {}, StartUTC: {}, EndUTC: {})".format(self._gameId, self._home_team, self._visitor_team, self._UTC_start, self._UTC_end)

class Rink(object):
    def __init__(self):
        self._zones = getZones()

    def getZone(self, x, y):
        zones = []
        for zone in list(self._zones.keys()):
            if x <= self._zones[zone]._ex and x >= self._zones[zone]._sx and y <= self._zones[zone]._ey and y >= self._zones[zone]._sy:
                zones.append(zone)
        return zones

    def __repr__(self):
        return "Rink(Zones: {})".format(self._zones)


class Zone(object):
    def __init__(self,name,sx,sy,ex,ey):
        self._name = name
        self._sx = sx
        self._sy = sy
        self._ex = ex
        self._ey = ey

    def __repr__(self):
        return "Zone(Name: {}, SX: {}, SY: {}, EX: {}, EY: {})".format(self._name, self._sx, self._sy, self._ex, self._ey)



def getZones():
    zone_dict = {}
    with open(fname+'PlayingSurface.json') as f:
        data = json.load(f)


    rink_sx = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['SX']
    rink_sy = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['SY']
    rink_x_len = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EX'] - rink_sx
    rink_y_len = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EY'] - rink_sy
    zone_dict['Rink'] = Zone('Rink', rink_sx, rink_sy, data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EX'], data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EY'])

    zones = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Sections"]
    for i in zones:
        name = i["Name"]
        sx = i["Rectangle"]["SX"]
        sy = i["Rectangle"]["SY"]
        x_len = i["Rectangle"]["EX"] - sx
        y_len = i["Rectangle"]["EY"] - sy
        zone_dict[name] = Zone(name, sx, sy, i["Rectangle"]["EX"], i["Rectangle"]["EY"])
    return zone_dict



def getRink():

    with open(fname+'PlayingSurface.json') as f:
        data = json.load(f)

    fig = plt.figure(figsize=(14,5))
    ax = fig.add_subplot(111)

    # print(data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"])
    # ---- Gets full rink
    rink_sx = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['SX']
    rink_sy = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['SY']
    rink_x_len = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EX'] - rink_sx
    rink_y_len = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EY'] - rink_sy


    zones = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Sections"]
    for i in zones:
        name = i["Name"]
        sx = i["Rectangle"]["SX"]
        sy = i["Rectangle"]["SY"]
        x_len = i["Rectangle"]["EX"] - sx
        y_len = i["Rectangle"]["EY"] - sy
        ax.add_patch(Rectangle((sx,sy),x_len,y_len, ec='k', lw=2,fill=False)) #np.random.rand(3,) ,label=name


    # print(data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Sections"][0]["Rectangle"])
    # exit()


    ax.add_patch(Rectangle((rink_sx,rink_sy),rink_x_len,rink_y_len, ec='k', lw=2,fill=False,label="Rink"))
    plt.xlim(left=-110, right=160)
    plt.ylim(bottom=-55, top=55)
    plt.axvline(0,ymin=0.12, ymax=0.88, c='r')
    # plt.legend()
    return fig, ax
    # plt.show()
