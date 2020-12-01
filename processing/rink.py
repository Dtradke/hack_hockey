import json
import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from matplotlib.patches import Rectangle

def loadGame(fname):
    print(fname)
    with open(fname, "rb") as f:
        game = pickle.load(f)
    return game

def close_event():
    plt.close()

class Game(object):
    def __init__(self, info, teams, fname):
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
        self._passes = {}
        self._shots = {}
        self._hits = {}
        self._faceoffs = {}
        self._posessions = {}
        self.getRinkDims(fname)
        self._game_stats = {}
        self._scoreboard = {}
        # self._fig, self._ax = getRink(fname)


    def getRinkPlot(self):
        fig, ax = getRink(fname)
        return fig, ax

    def getRinkDims(self, fname):

        with open(fname+'PlayingSurface.json') as f:
            data = json.load(f)

        # fig = plt.figure(figsize=(14,5))
        # ax = fig.add_subplot(111)

        # print(data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"])
        # ---- Gets full rink
        self._rink_sx = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['SX']
        self._rink_sy = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['SY']
        self._rink_x_len = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EX'] - self._rink_sx
        self._rink_y_len = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Boundary"]["Rectangle"]['EY'] - self._rink_sy


        zones = data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Sections"]
        self._zone_arr = []
        for i in zones:
            sx = i["Rectangle"]["SX"]
            sy = i["Rectangle"]["SY"]
            x_len = i["Rectangle"]["EX"] - sx
            y_len = i["Rectangle"]["EY"] - sy
            self._zone_arr.append([sx, sy, x_len, y_len])


    def runGameSynchonus(self, time, cur_posession, time_key):
        ''' This plays the game synchonusly '''

        time_data = {"entities": [],
                    "faceoffs":[],
                    "hit": [],
                    "pass": [],
                    "posession": []}

        # fig, ax = self.getRinkFast() #getRink()

        if cur_posession is not None:
            key = np.argmin(np.abs(np.array(list(cur_posession._posessor._hd_UTC_update.keys())) - time))
            key = list(cur_posession._posessor._hd_UTC_update.keys())[key]
            cur_posession.getPressure(key)

            # cur_posessor = np.array([cur_posession._posessor._hd_UTC_update[key]["X"], cur_posession._posessor._hd_UTC_update[key]["Y"]])
            cur_posessor = np.array([self._entities['1']._hd_UTC_update[key]["X"], self._entities['1']._hd_UTC_update[key]["Y"]])

            most_open_mate = cur_posession.getPasslanes(key)
            # bs = []
            # for b in cur_posession._betas:
            #     bs.append(b[1])
            #
            # for b in cur_posession._betas:
            #
            #     # annotations = np.mean(np.array([cur_posessor, np.array(b[0])]), axis=0)
            #     if b[1] == np.amax(np.array(bs)):
            #         plt.plot([cur_posessor[0], b[0][0]], [cur_posessor[1], b[0][1]], c='green')
            #         ax.annotate(round(b[1],3),(b[0][0], b[0][1]+5), c='green', fontsize=15)
            #     else:
            #         plt.plot([cur_posessor[0], b[0][0]], [cur_posessor[1], b[0][1]], c='red')
            #         ax.annotate(round(b[1],3),(b[0][0], b[0][1]+5), c='red', fontsize=15)
            #

            # time_data["posession"] = cur_posession
            # plt.scatter(cur_posessor[0], cur_posessor[1], s=160, c='y', label='Posession')
            # ax.annotate("Pressure: " + str(round(cur_posession._pressure,1)),(50, -50), fontsize=15)
            time_data["posession"].append({"possessor_id": cur_posession._posessor._id,
                                            "pressure": cur_posession._pressure})
                # TODO: add in passing lanes


        # ---- faceoffs
        if np.amin(np.absolute(np.array(list(self._faceoffs.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._faceoffs.keys())) - time))
            faceoff = self._faceoffs[list(self._faceoffs.keys())[key]]
            time_data["faceoffs"].append({"X": faceoff._loc["X"],
                                        "Y":faceoff._loc["Y"]})
            # plt.scatter(faceoff._loc["X"], faceoff._loc["Y"], s=160, c='c', label='Faceoff')
            # plt.title(faceoff)
            # time_data["faceoffs"].append(faceoff)

        # ---- hits
        if np.amin(np.absolute(np.array(list(self._hits.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._hits.keys())) - time))
            hit_event = self._hits[list(self._hits.keys())[key]]
            time_data["hit"].append({"hitter": hit_event._hitter._id,
                                    "hitter_team":self._teams[hit_event._hitter._team]._full_name,
                                    "hitter_number":hit_event._hitter._number,
                                    "hitter_last_name":hit_event._hitter._last_name,
                                    "X": hit_event._loc["X"],
                                    "Y": hit_event._loc["Y"],
                                    "points": 1})
            # plt.scatter(hit_event._loc["X"], hit_event._loc["Y"], s=160, c='y', label='Hit')
            # plt.title(hit_event)
            # time_data["hit"].append(hit_event)

        for e in self._entities.keys():
            try:
                key = np.argmin(np.abs(np.array(list(self._entities[e]._hd_UTC_update.keys())) - time))
                key = list(self._entities[e]._hd_UTC_update.keys())[key]
                if self._entities[e]._hd_UTC_update[key]["_onice"]:
                    # plt.scatter(self._entities[e]._hd_UTC_update[key]["X"], self._entities[e]._hd_UTC_update[key]["Y"], s=self._entities[e]._size, c=self._entities[e]._color, label=self._entities[e]._number+self._entities[e]._last_name)
                    # if self._entities[e]._id != '1':
                    #     ax.annotate(self._entities[e]._number,(self._entities[e]._hd_UTC_update[key]["X"], self._entities[e]._hd_UTC_update[key]["Y"]))
                    # time_data["entities"].append(self._entities[e])
                    time_data["entities"].append({"player_id":self._entities[e]._id,
                                                "player_team":self._teams[self._entities[e]._team]._full_name,
                                                "player_number":self._entities[e]._number,
                                                "player_last_name":self._entities[e]._last_name,
                                                "X": self._entities[e]._hd_UTC_update[key]["X"],
                                                "Y": self._entities[e]._hd_UTC_update[key]["Y"]})
            except:
                pass

        # ---- passes
        if np.amin(np.absolute(np.array(list(self._passes.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._passes.keys())) - time))
            pass_event = self._passes[list(self._passes.keys())[key]]
            # print("PASS: ", pass_event)
            # plt.scatter(pass_event._origin["X"], pass_event._origin["Y"], s=40, c='r', label='Origin')
            # plt.scatter(pass_event._destination["X"], pass_event._destination["Y"], s=40, c='c', label='Destination')
            # plt.plot([pass_event._origin["X"], pass_event._destination["X"]], [pass_event._origin["Y"], pass_event._destination["Y"]], c='r')
            # plt.title(pass_event)
            # time_data["pass"].append(pass_event)

            multiplier = 1
            if pass_event._turnover:
                multiplier = -1

            time_data["pass"].append({"passer_id": pass_event._passer._id,
                                    "passer_team":self._teams[pass_event._passer._team]._full_name,
                                    "passer_number":pass_event._passer._number,
                                    "passer_last_name":pass_event._passer._last_name,
                                    "overtook": pass_event._overtook,
                                    "turnover": pass_event._turnover,
                                    "points": (multiplier*(cur_posession._pressure/100)*pass_event._overtook)})

        # ---- shots
        # if np.amin(np.array(list(self._shots.keys())) - time) < 0.2:
        #     key = np.argmin(np.array(list(self._shots.keys())) - time)
        #     shot_event = self._shots[list(self._shots.keys())[key]]
        #     del self._shots[list(self._shots.keys())[key]]
        #     plt.scatter(shot_event._loc["X"], shot_event._loc["Y"], s=40, c='r', label='Origin')
        #     plt.plot([shot_event._loc["X"], shot_event._attacking_net["X"]], [shot_event._loc["Y"], shot_event._attacking_net["Y"]], c='r')
        #     plt.title(shot_event)



        # else:
        #     plt.title(key)

        # plt.legend()
        # # plt.show()
        # plt.savefig(sys.argv[-1]+'imgs/'+str(time)+".pdf", dpi=300) #,bbox_inches='tight'
        # plt.close()
        return time_data

    def runGameSynchonusOffline(self, time, cur_posession, possession_change):
        ''' This plays the game synchonusly '''

        # time_data = {"entities": [],
        #             "faceoffs":[],
        #             "hit": [],
        #             "pass": [],
        #             "posession": []}
        time_data = {} # keys will be player ids

        for e in self._entities.keys():
            try:
                key = np.argmin(np.abs(np.array(list(self._entities[e]._hd_UTC_update.keys())) - time))
                key = list(self._entities[e]._hd_UTC_update.keys())[key]
                if self._entities[e]._hd_UTC_update[key]["_onice"]:
                    time_data[self._entities[e]._id] = {
                                                "player_team":self._teams[self._entities[e]._team]._full_name,
                                                "player_number":self._entities[e]._number,
                                                "player_last_name":self._entities[e]._last_name,
                                                "X": self._entities[e]._hd_UTC_update[key]["X"],
                                                "Y": self._entities[e]._hd_UTC_update[key]["Y"],
                                                "hits": 0,
                                                "passes": 0,
                                                "shots": 0,
                                                "blocks": 0,
                                                "let_shot_off": 0,
                                                "pass_plus_minus": 0,
                                                "turnover": False,
                                                "possession":False,
                                                "points": 0}
            except:
                pass

        # print("cur pos: ", cur_posession)
        if cur_posession is not None:
            key = np.argmin(np.abs(np.array(list(cur_posession._posessor._hd_UTC_update.keys())) - time))
            pressure_key = list(cur_posession._posessor._hd_UTC_update.keys())[key]
            cur_posession.getPressure(pressure_key)

            key = np.argmin(np.abs(np.array(list(self._entities['1']._hd_UTC_update.keys())) - time))
            key = list(self._entities['1']._hd_UTC_update.keys())[key]
            # cur_posessor = np.array([cur_posession._posessor._hd_UTC_update[key]["X"], cur_posession._posessor._hd_UTC_update[key]["Y"]])
            cur_posessor = np.array([self._entities['1']._hd_UTC_update[key]["X"], self._entities['1']._hd_UTC_update[key]["Y"]])

            most_open_mate = cur_posession.getPasslanes(key)

            time_data[cur_posession._posessor._id]["possession"] = True
            time_data[cur_posession._posessor._id]["points"]+= cur_posession._pressure/1000 #mini points bump for handling pressure

            if possession_change:
                time_data[cur_posession._posessor._id]["points"]-=4
                time_data[cur_posession._posessor._id]["turnover"] = True

            # time_data["posession"] = cur_posession
            # time_data["posession"].append({"possessor_id": cur_posession._posessor._id,
            #                                 "pressure": cur_posession._pressure})
                # TODO: add in passing lanes


        # ---- faceoffs -> dont need faceoffs for points... wins are added to passes
        # if np.amin(np.absolute(np.array(list(self._faceoffs.keys())) - time)) == 0:
        #     key = np.argmin(np.absolute(np.array(list(self._faceoffs.keys())) - time))
        #     faceoff = self._faceoffs[list(self._faceoffs.keys())[key]]
        #     time_data["faceoffs"].append({"X": faceoff._loc["X"],
        #                                 "Y":faceoff._loc["Y"]})

        # ---- hits
        if np.amin(np.absolute(np.array(list(self._hits.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._hits.keys())) - time))
            hit_event = self._hits[list(self._hits.keys())[key]]
            del self._hits[list(self._hits.keys())[key]]

            time_data[hit_event._hitter._id]["points"]+= 1 # plus 1 for a hit
            # time_data[hit_event._hittee._id]["points"]-= 1 # minus 1 when getting hit
            time_data[hit_event._hitter._id]["hits"]+= 1 # plus 1 for a hit
            # time_data[hit_event._hittee._id]["hits"]-= 1 # minus 1 when getting hit



        # ---- passes
        if np.amin(np.absolute(np.array(list(self._passes.keys())) - time)) <= 0.1:
            key = np.argmin(np.absolute(np.array(list(self._passes.keys())) - time))
            pass_event = self._passes[list(self._passes.keys())[key]]
            del self._passes[list(self._passes.keys())[key]]

            # print("pass: ", pass_event)
            try:
                if pass_event._receiver._id == most_open_mate._id: #made most open pass
                    pass_points = 1 + ((cur_posession._pressure/100)+pass_event.beat) * 2
                else:
                    pass_points = 1 + ((cur_posession._pressure/100)+pass_event.beat)
            except:
                pass_points = 1 + ((0.5)+pass_event.beat)


            time_data[pass_event._passer._id]["points"]+= pass_points
            time_data[pass_event._passer._id]["passes"]+= 1

            for beat_player in pass_event._overtook_ids:
                time_data[beat_player]["points"]-= 1
                time_data[beat_player]["passes"]-= 1

        # ---- shots
        if np.amin(np.absolute(np.array(list(self._shots.keys())) - time)) < 0.2:
            key = np.argmin(np.absolute(np.array(list(self._shots.keys())) - time))
            shot_event = self._shots[list(self._shots.keys())[key]]
            del self._shots[list(self._shots.keys())[key]]

            if cur_posession is not None:
                shot_points = ((cur_posession._pressure/100)+2)
            else:
                shot_points = 1

            if shot_event._shooter._id not in list(time_data.keys()):
                self.addPlayer(shot_event._shooter, time_data)


            # _blocked, _blocker
            if shot_event._blocked:
                time_data[shot_event._shooter._id]["points"]-=shot_points

                if shot_event._blocker._id not in list(time_data.keys()):
                    self.addPlayer(shot_event._blocker, time_data)                      # negative points if shot blocked
                time_data[shot_event._blocker._id]["points"]+=shot_points                       # points go to  blocker
            else:
                # print(time_data.keys())
                time_data[shot_event._shooter._id]["points"]+=shot_points                       # points for a shot
                time_data[shot_event._shooter._id]["shots"]+=1                                  # increments number of shots
                if cur_posession is not None:
                    for presser in cur_posession._pressers:
                        time_data[presser._id]["points"]-=((cur_posession._pressure/100)+1)          # punish if presser lets player get shot off
                        time_data[presser._id]["let_shot_off"]+=1                                # times a presser let a shot get off
        return time_data, cur_posession

    def addPlayer(self, player, time_data):
        time_data[player._id] = {
                                    "player_team":self._teams[player._team]._full_name,
                                    "player_number":player._number,
                                    "player_last_name":player._last_name,
                                    "X": 0,
                                    "Y": 0,
                                    "hits": 0,
                                    "passes": 0,
                                    "shots": 0,
                                    "blocks": 0,
                                    "let_shot_off": 0,
                                    "pass_plus_minus": 0,
                                    "turnover": False,
                                    "possession":False,
                                    "points": 0}
        return time_data

    def getRinkFast(self, fname):
        fig = plt.figure(figsize=(14,5))
        ax = fig.add_subplot(111)


        for i in self._zone_arr:
            ax.add_patch(Rectangle((i[0],i[1]),i[2],i[3], ec='k', lw=2,fill=False)) #np.random.rand(3,) ,label=name


        ax.add_patch(Rectangle((self._rink_sx,self._rink_sy),self._rink_x_len,self._rink_y_len, ec='k', lw=2,fill=False,label="Rink"))
        plt.xlim(left=-110, right=160)
        plt.ylim(bottom=-55, top=55)
        plt.axvline(0,ymin=0.12, ymax=0.88, c='r')
        # plt.legend()
        return fig, ax
        # plt.show()


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
                # time_idx = np.where((np.abs(np.array(self._entities[e]._update_time) - cur_time) < 0.5))[0][0]

                time_idx = np.argmin((np.abs(np.array(self._entities[e]._update_time) - cur_time)))

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
                # time_idx = np.where((np.abs(np.array(self._entities[e]._update_time) - cur_time) < 0.5))[0][0]
                time_idx = np.argmin((np.abs(np.array(self._entities[e]._update_time) - cur_time)))
                if self._entities[e]._onice[time_idx] and self._entities[e]._id != '1':
                    teammates_on_ice.append({'entity_id':e, 'time_idx':time_idx})
            except:
                pass
        return teammates_on_ice



    def __repr__(self):
        return "Game(Id: {}, Home: {}, Away: {}, StartUTC: {}, EndUTC: {})".format(self._gameId, self._home_team_num, self._visitor_team_num, self._UTC_start, self._UTC_end)

class Rink(object):
    def __init__(self, fname=None):
        self._zones = getZones(fname)

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



def getZones(fname):
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






def getRink(fname):

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
