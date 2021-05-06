import json
import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from scipy.special import softmax
from matplotlib.lines import Line2D

def loadGame(fname):
    # print(fname)
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
        self._passes = {}
        self._shots = {}
        self._hits = {}
        self._faceoffs = {}
        self._posessions = {}
        self.getRinkDims(fname)
        self._game_stats = {}
        self._scoreboard = {}
        self._goals = {}
        self._who_has_possession = {}
        self._takeaways = {}

        # self._fig, self._ax = getRink(fname)


    def getRinkPlot(self):
        fig, ax = getRink(fname='../../2019030415/')
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

    def cleanPossessions(self):
        ''' gets rid of dirty data possessions '''
        new_possessions = {}
        for p in self._posessions.keys():
            if self._posessions[p]._clocktime_seconds >= 0:
                new_possessions[p] = self._posessions[p]
                # self._posessions[p]._posessor._time_of_possession+=self._posessions[p]._posession_UTC_keys.size
            else:
                print("Dirty Possession: ", self._posessions[p])
        # print("Deleted: ", len(self._posessions.keys()) - len(new_possessions.keys()))
        self._posessions = new_possessions

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

            # most_open_mate = cur_posession.getPasslanes(time) #key
            most_open_mate = cur_posession.getPasslanes_HalfCircleBeta(time) #key

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
            if self._entities[e]._team in list(self._teams.keys()) and len(list(self._entities[e]._hd_UTC_update.keys())) > 100:
                key = np.argmin(np.abs(np.array(list(self._entities[e]._hd_UTC_update.keys())) - time))
                key = list(self._entities[e]._hd_UTC_update.keys())[key]
                # if self._entities[e]._hd_UTC_update[key]["_onice"]:
                time_data[self._entities[e]._id] = {
                                            "player_team":self._teams[self._entities[e]._team]._full_name,
                                            "player_number":self._entities[e]._number,
                                            "player_last_name":self._entities[e]._last_name,
                                            "X": self._entities[e]._hd_UTC_update[key]["X"],
                                            "Y": self._entities[e]._hd_UTC_update[key]["Y"],
                                            "goals": 0,
                                            "hits": 0,
                                            "passes": 0,
                                            "shots": 0,
                                            "shot_attempts": 0,
                                            "blocks": 0,
                                            "let_shot_off": 0,
                                            "pass_plus_minus": 0,
                                            "turnover": 0,
                                            "possession":0,
                                            "points": 0}


        # print("cur pos: ", cur_posession)
        if cur_posession is not None:
            key = np.argmin(np.abs(np.array(list(cur_posession._posessor._hd_UTC_update.keys())) - time))
            pressure_key = list(cur_posession._posessor._hd_UTC_update.keys())[key]
            cur_posession.getPressure(pressure_key)

            key = np.argmin(np.abs(np.array(list(self._entities['1']._hd_UTC_update.keys())) - time))
            key = list(self._entities['1']._hd_UTC_update.keys())[key]
            # cur_posessor = np.array([cur_posession._posessor._hd_UTC_update[key]["X"], cur_posession._posessor._hd_UTC_update[key]["Y"]])
            cur_posessor = np.array([self._entities['1']._hd_UTC_update[key]["X"], self._entities['1']._hd_UTC_update[key]["Y"]])

            # most_open_mate = cur_posession.getPasslanes(key)
            most_open_mate = cur_posession.getPasslanes_HalfCircleBeta(key)

            attacking_zone = self.isAttackingZone(time, cur_posession, cur_posession._posessor)
            if attacking_zone == 'offensive': zone_multiplier = 1.25
            elif attacking_zone == 'defensive': zone_multiplier = 0.75
            else: zone_multiplier = 1

            time_data[cur_posession._posessor._id]["possession"]+=(1 * zone_multiplier)
            time_data[cur_posession._posessor._id]["points"]+= cur_posession._pressure/1000 #mini points bump for handling pressure

            if possession_change:
                time_data[cur_posession._posessor._id]["points"]-=4

                if attacking_zone == 'offensive': zone_multiplier = 0.75
                elif attacking_zone == 'defensive': zone_multiplier = 1.25
                else: zone_multiplier = 1
                time_data[cur_posession._posessor._id]["turnover"]+=(1 * zone_multiplier)

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
            # del self._hits[list(self._hits.keys())[key]]

            time_data[hit_event._hitter._id]["points"]+= 1 # plus 1 for a hit
            # time_data[hit_event._hittee._id]["points"]-= 1 # minus 1 when getting hit
            time_data[hit_event._hitter._id]["hits"]+= 1 # plus 1 for a hit
            # time_data[hit_event._hittee._id]["hits"]-= 1 # minus 1 when getting hit

        # ----- goals
        if np.amin(np.absolute(np.array(list(self._goals.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._goals.keys())) - time))
            goal_event = self._goals[list(self._goals.keys())[key]]
            # del self._goals[list(self._goals.keys())[key]]

            time_data[goal_event._scorer._id]["points"]+= 5 # plus 1 for a hit
            # time_data[hit_event._hittee._id]["points"]-= 1 # minus 1 when getting hit
            time_data[goal_event._scorer._id]["goals"]+= 1 # plus 1 for a hit
            # time_data[hit_event._hittee._id]["hits"]-= 1 # minus 1 when getting hit

        # ---- passes
        if np.amin(np.absolute(np.array(list(self._passes.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._passes.keys())) - time))
            pass_event = self._passes[list(self._passes.keys())[key]]
            # del self._passes[list(self._passes.keys())[key]]

            # print("pass: ", pass_event)
            try:
                if pass_event._receiver._id == most_open_mate._id: #made most open pass
                    pass_points = 1 + ((cur_posession._pressure/100)+pass_event.beat) * 2
                else:
                    pass_points = 1 + ((cur_posession._pressure/100)+pass_event.beat)
            except:
                pass_points = 1 + ((0.5)+pass_event.beat)


            attacking_zone = self.isAttackingZone(time, pass_event, pass_event._passer)
            if attacking_zone == 'offensive': zone_multiplier = 1.25
            elif attacking_zone == 'defensive': zone_multiplier = 0.75
            else: zone_multiplier = 1


            time_data[pass_event._passer._id]["points"]+= pass_points
            time_data[pass_event._passer._id]["passes"]+= (1 * zone_multiplier)
            time_data[pass_event._passer._id]["pass_plus_minus"]+=len(pass_event._overtook_ids)

            # for beat_player in pass_event._overtook_ids:
            #     try:                                                            #FIXME: this is a hack
            #         time_data[beat_player]["points"]-= 1
            #         time_data[beat_player]["pass_plus_minus"]-= 1
            #     except:
            #         pass

        # ---- shots
        if np.amin(np.absolute(np.array(list(self._shots.keys())) - time)) == 0:
            key = np.argmin(np.absolute(np.array(list(self._shots.keys())) - time))
            shot_event = self._shots[list(self._shots.keys())[key]]
            # del self._shots[list(self._shots.keys())[key]]

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
                if shot_event._miss:
                    time_data[shot_event._shooter._id]["shot_attempts"]+=1
                else:
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
                                    "goals":0,
                                    "hits": 0,
                                    "passes": 0,
                                    "shots": 0,
                                    "shot_attempts": 0,
                                    "blocks": 0,
                                    "let_shot_off": 0,
                                    "pass_plus_minus": 0,
                                    "turnover": 0,
                                    "possession":0,
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

        if my_team == '14':             #NOTE: TAMPA IS 14 ----> ASSUMPTION BECAUSE THEY DIDN'T SWITCH SIDES BETWEEN GAMES
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
                return {"X": x_mean, "Y": y_mean}, attacking


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

    def updateMomentumData(self, gameSecond, cur_posession):
        for i, team_id in enumerate(gameSecond._stats):
            for j, event in enumerate(self._momentum_data[team_id].keys()):
                # if event != 'turnover' or event != "possession":
                # attacking_zone = isAttackingZone(gameSecond._time, cur_posession)
                # if attacking_zone:
                #     zone_multiplier = 2
                # else:
                #     zone_multiplier = 1
                self._momentum_data[team_id][event].append(gameSecond._stats[team_id][event])
                # else:
                #     attacking_zone = isAttackingZone(gameSecond._time)
                #     self._momentum_data[team_id][event] = any(gameSecond._stats[team_id][event][0])

    def isAttackingZone(self, t, event, player):
        ''' works with updateMomentumData to determine if puck is in attacking zone '''

        time_key = np.argmin(np.abs(np.array(list(self._entities["1"]._hd_UTC_update.keys())) - t))
        puck_loc = [self._entities["1"]._hd_UTC_update[list(self._entities["1"]._hd_UTC_update)[time_key]]["X"],
                    self._entities["1"]._hd_UTC_update[list(self._entities["1"]._hd_UTC_update)[time_key]]["Y"]]
        zone = self.getPuckZone(puck_loc)

        # if cur_posession is not None:
        _, attacking_net_name = self.getAttackingNet(player._team, event._period)
        if attacking_net_name[:4] == zone[:4]:
            return "offensive"               # NOTE: THIS IS USED FOR OFFENSIVE ZONE PRESSURE
        elif zone[0] != 'N': #defensive zone
            return "defensive"
        return "neutral"

    def getPuckZone(self, puck_loc):
        ''' this returns the name of the zone that the puck is in...
        more calculations are going to need to be done in calculateMomentum to determine
        the offensive/defensive team '''

        zoi = ["NeutralZone", "LeftOffensiveZone", "RightOffensiveZone"]
        for zone_key in self._rink._zones:
            zone = self._rink._zones[zone_key]
            if puck_loc[0] > zone._sx and puck_loc[1] > zone._sy and puck_loc[0] <= zone._ex and puck_loc[1] <= zone._ey:
                if zone._name in zoi:
                    return zone._name
        return None


    def calculateMomentum(self, elapsed_seconds, t, cur_posession):
        window = 30
        scalar = 1


        # if elapsed_seconds%(20*60) > window:
        #     start = elapsed_seconds - window
        # else:
        #     start = ((elapsed_seconds//(20*60))*(20*60))
        #
        # end = elapsed_seconds + 1


        self._momentum[elapsed_seconds] = {'14':0, '25':0}


        # home_strength = self._scoreboard[t]["HomeStrength"]
        # visitor_strength = self._scoreboard[t]["VisitorStrength"]

        team_momentums = []
        for i, team_id in enumerate(list(self._momentum[elapsed_seconds].keys())):
            goals = np.sum(np.array(self._momentum_data[team_id]["goals"][-window:]))
            hits = np.sum(np.array(self._momentum_data[team_id]["hits"][-window:]))
            passes = np.sum(np.array(self._momentum_data[team_id]["passes"][-window:]))
            pass_plus_minus = np.sum(np.array(self._momentum_data[team_id]["pass_plus_minus"][-window:]))
            blocks = np.sum(np.array(self._momentum_data[team_id]["blocks"][-window:]))
            shots = np.sum(np.array(self._momentum_data[team_id]["shots"][-window:]))
            shot_attempts = np.sum(np.array(self._momentum_data[team_id]["shot_attempts"][-window:]))
            turnover = np.sum(np.array(self._momentum_data[team_id]["turnover"][-window:]))
            possession = np.sum(np.array(self._momentum_data[team_id]["possession"][-window:]))

            # if team_id == '14':
            #     print("possession: ", possession, ", passes: ", passes, " shot_attempts: ", shot_attempts)


            # print(possession, " - ", np.array(self._momentum_data[team_id]["possession"][start:end]))

            # dave momentum
            # team_momentums.append(((3*goals) + (0.75*hits) + (0.2*passes) +
            #                 (pass_plus_minus) + (blocks) + (shots) +
            #                 (-2*turnover) + (0.15*possession)) + scalar)

            # alex momentum
            team_momentums.append((0.75 *shots) +
                                    (0.1 + shot_attempts) +
                                    (0.05   *possession) +
                                    (0.2   *pass_plus_minus) +
                                    (0.6   *turnover) +
                                    (0.4   *hits) +
                                    (0.3 *blocks) +
                                    (0.2 *passes) +
                                    scalar)



            # print(team_id, ": ", shots, " ", shot_attempts, " ", possession, " ", pass_plus_minus, " ", turnover, " ", hits, " ", blocks, " ", passes)

            # self._momentum[elapsed_seconds][team_id] = team_momentum
            # total_momentum+=team_momentum

        # print()
        team_momentums = softmax(np.array(team_momentums))
        # print(team_momentums)
        for i, team_id in enumerate(list(self._momentum[elapsed_seconds].keys())):
            # self._momentum[elapsed_seconds][team_id] = np.exp(self._momentum[elapsed_seconds][team_id]) / np.exp(total_momentum)
            self._momentum[elapsed_seconds][team_id] = team_momentums[i]

    def getXTicks(self, y, event_name, clock, event_count, game_idx):
        if event_name == 'Goal':
            if event_count == 4:
                y = y[-30:]
            else:
                y = y[-60:]
        else:
            if event_count == 1 and game_idx == 1:
                y = y[60:203]
            else:
                # y = y[-240:]
                y = y[60:]

        y = np.array(y)

        print("Y SHAPE: ", y.shape)

        xticks = [y.shape[0]]
        xtick_labels = [str(clock[0])+":"+str(clock[1]-30)]
        if event_name == 'Goal': second_resolution = 10
        else: second_resolution = 30
        # =[right_min, right_sec]
        last_mins = clock[0]
        if event_name == 'Goal' and event_count == 4:
            last_secs = clock[1] - 30
            loop_len = y.shape[0]*2
        else:
            last_secs = clock[1]
            loop_len = y.shape[0]

        for x in range(loop_len,0,-1):
            if x%second_resolution == 0 and x != y.shape[0]:
                if last_secs + (second_resolution) > 59:
                    last_mins = last_mins+1
                    last_secs = (last_secs + second_resolution)-60
                    # xtick_labels.append(str(last_mins)+":"+str(last_secs))
                else:
                    last_secs = last_secs + second_resolution
                xtick_labels.append(str(last_mins)+":"+str(last_secs))
                # xtick_labels.append(str(x))
                xticks.append(x)

        return y, xticks, xtick_labels

    def plotMomentum(self, fname, event, event_count, clock, event_name='Goal', game_idx=0):
        y = []

        for key, val in enumerate(self._momentum.keys()):
            y.append(self._momentum[val]['14'])

        fig, ax1 = plt.subplots(1,1, figsize=(15,6))
        # y = np.array(y)[-60:]
        # y = np.array(y)[-120:]


        y, xticks, xtick_labels = self.getXTicks(y, event_name, clock, event_count, game_idx)
        ax1.set_xlabel("Game Clock", fontsize=25)


        print("yshape: ", y.shape[0])

        print("ticks: ", xticks)
        print("lab: ", xtick_labels)

        dal_mom, tbl_mom = [], []
        for mom in y:
            dal_mom.append(1-mom)
            tbl_mom.append(mom)

        # np.save(fname+"/momentum"+str(y.shape[0])+"sec.npy", y)
        # y = np.load(fname)

        ax1.fill_between(np.arange(y.shape[0]), y, 0,color='b', label='TBL Momentum')
        ax1.fill_between(np.arange(y.shape[0]), y, 1,color='g', label='DAL Momentum')
        # for tg in tampa_goals:
        if event._teamId == '14':
            if event_name == 'Goal':
                ax1.axvline(y.shape[0]//2, c='k', linewidth=5, label='TBL '+event_name)
                mom_percentage = round((np.mean(np.array(tbl_mom[30:])) * 100),1)
                mom_percentage_before = round((np.mean(np.array(tbl_mom[:30])) * 100),1)
                team_name = "TBL"
                if game_idx == 1 and event_count == 0:
                    ax1.axvline(6, c='y', linewidth=5, linestyle="--", label='DAL Penalty')
                    custom_lines = [Line2D([0], [0], linestyle="--", color='y', lw=8)]
                    ax1.legend(custom_lines, ['DAL Penalty'], fontsize=17, framealpha=0.8, loc="upper left")
            else:
                ax1.axvline(60, c='k', linewidth=5, linestyle='--', label='TBL '+event_name) #y.shape[0]-1
                # if y.shape[0] > 180:
                    # ax1.axvline(180, c='k', linestyle=':', linewidth=5, label='Penalty Over')
                if game_idx == 0 and event_count == 2:
                    ax1.axvline(188, c='y', linewidth=5, label='DAL Goal')
                    custom_lines = [Line2D([0], [0], color='y', lw=8)]
                    ax1.legend(custom_lines, ['DAL Goal'], fontsize=17, framealpha=0.8, loc="upper left")
                if game_idx == 1 and event_count == 3:
                    ax1.axvline(139, c='y', linewidth=5, linestyle="--", label='DAL Penalty')
                    custom_lines = [Line2D([0], [0], color='y', linestyle="--", lw=8)]
                    ax1.legend(custom_lines, ['DAL Penalty'], fontsize=17, framealpha=0.8, loc="upper left")

                if y.shape[0] > 180:
                    ax1.axvline(180, c='k', linestyle=':', linewidth=5, label='Penalty Over')
                mom_percentage = 100 - round((np.mean(np.array(tbl_mom[60:180])) * 100),1)
                mom_percentage_before = 100 - round((np.mean(np.array(tbl_mom[:60])) * 100),1)
                mom_percentage_last = 100 - round((np.mean(np.array(tbl_mom[180:])) * 100),1)
                team_name = "DAL" #"Tampa Bay"
        # for dg in dallas_goals:
        else:
            if event_name == 'Goal':
                ax1.axvline(29, c='y', linewidth=5, label='DAL '+event_name)
                mom_percentage = round((np.mean(np.array(dal_mom[30:])) * 100),1)
                mom_percentage_before = round((np.mean(np.array(dal_mom[:30]))* 100),1)
                team_name = "DAL"
                if game_idx == 0 and event_count == 3:
                    ax1.axvline(21, c='k', linewidth=5, linestyle=':', label='TBL Penalty Over')
                    custom_lines = [Line2D([0], [0], color='k', linestyle=':', lw=8)]
                    ax1.legend(custom_lines, ['TBL Penalty Over'], fontsize=17, framealpha=0.8, loc="upper left")
            else:
                ax1.axvline(60, c='y', linewidth=5, linestyle='--', label='DAL '+event_name)
                mom_percentage = 100 - round((np.mean(np.array(dal_mom[60:180])) * 100),1)
                mom_percentage_before = 100 - round((np.mean(np.array(dal_mom[:60])) * 100),1)
                mom_percentage_last = 100 - round((np.mean(np.array(dal_mom[180:])) * 100),1)
                team_name = "TBL" #"Dallas"
                if game_idx == 1 and event_count == 1 and event_name == 'Penalty':
                    ax1.axvline(84, c='k', linewidth=5, label='TBL Goal/Penalty Over')
                    ax1.axvline(84, c='y', linewidth=5, linestyle=':', label='Penalty Over')
                    # plt.legend(loc="upper left", fontsize=17, framealpha=0.8)
                    mom_percentage = 100 - round((np.mean(np.array(dal_mom[60:84])) * 100),1)
                    mom_percentage_last = 100 - round((np.mean(np.array(dal_mom[84:])) * 100),1)
                    custom_lines = [Line2D([0], [0], color='k', lw=8),
                                    Line2D([0], [0], color='y', linestyle=":", lw=8)]
                    ax1.legend(custom_lines, ['TBL Goal', 'DAL Penalty Over'], fontsize=17, framealpha=0.8, loc="upper left")
                elif game_idx == 1 and event_count == 4:
                    ax1.axvline(99, c='k', linewidth=5, linestyle=":", label='TBL Penalty Over')
                    custom_lines = [Line2D([0], [0], linestyle=":", color='k', lw=8)]
                    ax1.legend(custom_lines, ['TBL Penalty Over'], fontsize=17, framealpha=0.8, loc="upper left")
                    ax1.axvline(180, c='y', linestyle=':', linewidth=5, label='Penalty Over')
                else:
                    ax1.axvline(180, c='y', linestyle=':', linewidth=5, label='Penalty Over')

        if (y.shape[0] < 235 and event_name == 'Penalty') or (y.shape[0] < 60 and event_name == 'Goal'):
            ax1.axvline(y.shape[0]-0.5, c='r', linestyle='-', linewidth=5, label='End of Period')
        # else:
        if y.shape[0] > 180:
            ax1.text(180,-0.08, team_name+": "+str(round(mom_percentage_last,1))+"%", fontsize=23, bbox={'facecolor':'w', 'alpha':0.9, 'pad':2})

        ax1.set_ylabel("Momentum", fontsize=25)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtick_labels, fontsize=20)

        if event_name == 'Goal':
            ax1.text(0,-0.08, team_name+" Before Goal: "+str(round(mom_percentage_before,1))+"%", fontsize=23, bbox={'facecolor':'w', 'alpha':0.9, 'pad':2})
            if event_count == 4:
                ax1.text(29,-0.08, "Game Over", fontsize=23, bbox={'facecolor':'w', 'alpha':0.9, 'pad':2})
            else:
                ax1.text(30,-0.08, team_name+" After Goal: "+str(mom_percentage)+"%", fontsize=23, bbox={'facecolor':'w', 'alpha':0.9, 'pad':2})
        else:
            ax1.text(0,-0.08, team_name+": "+str(round(mom_percentage_before,1))+"%", fontsize=23, bbox={'facecolor':'w', 'alpha':0.9, 'pad':2})
            ax1.text(60,-0.08, team_name+" PP: "+str(round(mom_percentage,1))+"%", fontsize=23, bbox={'facecolor':'w', 'alpha':0.9, 'pad':2})


        plt.title("Game: "+str(game_idx+5)+", "+self._teams[event._teamId]._full_name+" "+event_name+", Period:   , Clock: "+str(event._clock_minutes)+":"+str(event._clock_seconds), fontsize=25)
        # plt.legend(loc="upper left", fontsize=17, framealpha=0.8)
        plt.yticks(fontsize=20)
        plt.ylim(bottom=-0.12)
        if event_name == 'Goal': fname = "../../Paper/paper_imgs/momentum/GOAL-30sec-30sec_"+self._gameId+str(event._clock_minutes)+"-"+str(event._clock_seconds)+event_name+str(event_count) #event._scorer._last_name+
        else: fname = "../../Paper/paper_imgs/momentum/PP-1-2-1mins_sec_"+self._gameId+str(event._clock_minutes)+"-"+str(event._clock_seconds)+event_name+str(event_count) #event._scorer._last_name+

        print(fname)
        plt.savefig(fname,bbox_inches='tight', dpi=300)
        plt.close()


        # plt.show()
        # exit()


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
        if name in ['RightSubstitutionArea', 'LeftSubstitutionArea', 'LeftPenaltyBox', 'RightPenaltyBox', 'LeftBench', 'RightBench']: continue
        # print(name)
        sx = i["Rectangle"]["SX"]
        sy = i["Rectangle"]["SY"]
        x_len = i["Rectangle"]["EX"] - sx
        y_len = i["Rectangle"]["EY"] - sy
        ax.add_patch(Rectangle((sx,sy),x_len,y_len, ec='k', lw=2,fill=False)) #np.random.rand(3,) ,label=name
        if name == 'RightGoal' or name == 'RightCrease': ax.add_patch(Rectangle((sx,sy),x_len,y_len, ec='k', lw=2,fill=True, facecolor='b', label="Tampa Bay Goal"))
        elif name == 'LeftGoal' or name == 'LeftCrease': ax.add_patch(Rectangle((sx,sy),x_len,y_len, ec='k', lw=2,fill=True, facecolor='g', label="Dallas Goal"))
    # exit()


    # print(data[0]["PlayingSurface"]["PlayingSurfaceInfo"]["Sections"][0]["Rectangle"])
    # exit()


    ax.add_patch(Rectangle((rink_sx,rink_sy),rink_x_len,rink_y_len, ec='k', lw=2,fill=False,label="Rink"))
    plt.xlim(left=-110, right=110) #160
    plt.ylim(bottom=-45, top=45) #was 55
    # plt.axvline(0,ymin=0.12, ymax=0.88, c='r')
    plt.axvline(0,ymin=0.03, ymax=0.97, c='r')
    ax.text(101,-38, "Tampa Bay Defensive Zone", rotation=-90, fontsize=18)
    ax.text(-106,-33, "Dallas Defensive Zone", rotation=90, fontsize=18)
    # plt.show()
    # exit()
    # plt.legend()
    return fig, ax
    # plt.show()
