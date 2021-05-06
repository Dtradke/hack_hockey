import json
import sys
import numpy as np
import pandas as pd
import pickle
import os
import math
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler

import player_factory
import rink
import event
import print_to_terminal
import paper_plots


# STATS TO TRACK:
# - shots
# - hits [check]
# - significant passes [check]
# - posessions/pressure [check]
# - shift length [check - use _onice]
# - run the game completely synchonusly, push data real-time


# NOTES: HOME DEFENDS THE RIGHT GOAL IN FIRST PERIOD


def makePlayers(fname, game):
    '''Creates the entities in the selected game

    returns: dict of entities in the game
    '''
    position_arr = ['D', 'G', 'LW', 'C', "RW"]
    print("Creating Entities...")
    with open(fname+'EntityRegistration.json') as f:
        data = json.load(f)

    # data[0]["EntityRegistration"]['Entities'] ---> list of players
    entity_list = data[0]["EntityRegistration"]['Entities']
    entities = {}
    for i in entity_list:
        ent = player_factory.Player(i, game)
        if ent._pos in position_arr or ent._id == '1':
            entities[ent._id] = ent

    print("Entities Made...")
    return entities

def makeTeams(fname):
    ''' This function makes the teams '''

    print("Creating Teams...")
    with open(fname+'EntityRegistration.json') as f:
        data = json.load(f)

    teams = {}
    for team in data[0]["EntityRegistration"]["Teams"]:
        t = player_factory.Team(team)
        teams[t._teamId] = t

    return teams

def getHeartbeat(fname):
    ''' This function loads the heartbeat of the sensor '''

    print("Getting Heartbeat...")
    with open(fname+'Heartbeat.json') as f:
        data = json.load(f)

    counters = []
    for i in data:
        counters.append(i["Counter"]) # Note: i["EventStatus"] tells WaitingForStart, InProgress, or Finished
    return np.array(counters)

def getGame(fname, teams):
    ''' This function loads the Event/Game information '''

    print("Creating Game...")
    with open(fname+'Event.json') as f:
        data = json.load(f)

    game = rink.Game(data[0]["Event"], teams=teams, fname=fname)

    for update in data:
        if update["Event"]["ActualStartUTC"] > game._UTC_start:
            game._UTC_start = update["Event"]["ActualStartUTC"]
        if update["Event"]["ActualEndUTC"] > game._UTC_end:
            game._UTC_end = update["Event"]["ActualEndUTC"]
    return game

def getPosessions(fname, game):
    ''' This gets significant events in the game (ex: faceoffs, hits, goals, etc...)
    '''

    print("Loading Posessions...")
    with open(fname+'Marker.json') as f:
        data = json.load(f)

    possession_changes = []

    possessions = []
    for t in data:
        change_time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        if event_type == "EventPossessionChangePlayer":
            possession_changes.append(t)

            # print(t["Marker"]["MarkerData"][0]["Descriptor_"])
            # possession_event = event.Possession(game, start_time, t["Marker"]["MarkerData"][0]['Descriptor_'], t["Marker"]["MarkerData"])

    last_period = -1
    for i in range(len(possession_changes)-1):
        if len(possession_changes[i]["Marker"]["MarkerData"][0]["Participants"]) > 0 and possession_changes[i]["Marker"]["MarkerData"][0]["Participants"][0]["EntityId"] != '':
            posession = event.Posession(game, possession_changes[i], possession_changes[i+1])
            game._teams[posession._posessor._team]._posessions.append(posession)
            game._entities[posession._posessor._id]._posession_lst.append(posession)
            game._posessions[round(posession._UTC_start,1)] = posession
            if posession._period != last_period:
                print("Possessions in Period: ", posession._period)
                last_period = posession._period

    return game

def getHits(fname, game):

    print("Loading Hits...")
    with open(fname+'Marker.json') as f:
        data = json.load(f)

    counter = 0
    for count, t in enumerate(data):
        change_time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        if event_type == "EventHit":
            if len(t["Marker"]["MarkerData"][0]["Participants"]) == 2:
                hit = event.Hit(game, t)
                if round(hit._time, 1) not in (game._hits.keys()):
                    game._hits[round(hit._time, 1)] = hit
                    counter+=1
                    # print(hit)
                    # if counter == 10:
                    #     print(game._hits.keys())
                    #     exit()

    return game




def getShots(fname, game):
    ''' This gets shot events in the game
    '''
    print("Loading Shots...")
    with open(fname+'Marker.json') as f:
        data = json.load(f)

    last_total_seconds = 0
    last_shooter = 0
    count = 0
    passes = []
    for t in data:
        time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        # print(event_type)
        count +=1
        # if count == 100: exit()
        if event_type == "EventShot":


            # if int(t["Marker"]["MarkerData"][0]['ETime']['ClockMinutes']) == 7 and int(t["Marker"]["MarkerData"][0]['ETime']['ClockSeconds']) == 37:
            #     participants = t["Marker"]["MarkerData"][0]["Participants"]
            #     shooter = None
            #     for i in participants:
            #         if i['Role'] == 'MapRoleShooter':
            #             shooter = game._entities[i["EntityId"]]
            #     if shooter is not None:
            #         print(">Shot: ", t["Marker"]["MarkerData"][0]['ETime']['ClockMinutes'], ":",t["Marker"]["MarkerData"][0]['ETime']['ClockSeconds'], " > ", shooter)



            participants = t["Marker"]["MarkerData"][0]["Participants"]
            shooter_bool = False
            for i in participants:
                if i['Role'] == 'MapRoleShooter':
                    shooter_bool = True

            if len(participants) > 0 and shooter_bool:
                shot_seconds = (t["Marker"]["MarkerData"][0]["ETime"]["ClockMinutes"]*60) + t["Marker"]["MarkerData"][0]["ETime"]["ClockSeconds"]
                if np.abs(shot_seconds - last_total_seconds) > 5:
                    shot_event = event.Shot(game, time, t["Marker"]["MarkerData"][0]['Descriptor_'], t["Marker"]["MarkerData"])
                    # print(shot_event)
                    last_total_seconds = (shot_event._clock_minutes*60) + shot_event._clock_seconds
                    last_shooter = shot_event._shooter._id
                    last_key = round(time,1)
                    # print(shot_event)
                    game._shots[round(time,1)] = shot_event
                else:
                    # print(t["Marker"]["MarkerData"])
                    game._shots[last_key].updateShot(t["Marker"]["MarkerData"])

    for shot_key in list(game._shots.keys()):
        shot = game._shots[shot_key]
        # shot._shooter._shots+=1
        if shot._blocked:
            shot._blocker.blocks+=1
            try:
                shot._blocker._pressed_shooter_success.append(shot._shot_presser_lst[shot._blocker._id])
            except:
                pass
        else:
            for player in list(shot._shot_presser_lst.keys()):
                game._entities[player]._pressed_shooter_fail.append(shot._shot_presser_lst[game._entities[player]._id])

            shot._shooter._pressure_to_shoot.append(shot._shot_pressure)


    # print(game._shots)
    # print("shots: ", len(game._shots.keys()))

    return game


def getScoreBoard(fname, game):
    with open(fname+'Scoreboard.json') as f:
        data = json.load(f)

    for t in data:
        now_UTC = round(t["Scoreboard"]["NowUTC"],1)
        game._scoreboard[now_UTC] = {"Period": t["Scoreboard"]["Period"],
                                    "ClockState": t["Scoreboard"]["ClockState"],
                                    "ClockMinutes": t["Scoreboard"]["ClockMinutes"],
                                    "ClockSeconds": t["Scoreboard"]["ClockSeconds"],
                                    "HomeStrength": t["Scoreboard"]["PowerPlayInfo"]["HomeStrength"],
                                    "VisitorStrength": t["Scoreboard"]["PowerPlayInfo"]["VisitorStrength"]}

    return game


def getGoals(fname, game):
    ''' gets the goals of the game '''
    print("Loading Goals...")

    with open(fname+'Marker.json') as f:
        data = json.load(f)

    faceoffs = []
    for t in data:
        time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        if event_type == "EventGoal":
            # print(">>>>> GOAL", t)
            goal = event.Goal(game, time, t)
            game._goals[round(goal._UTC_update,1)] = goal
    # exit()
    return game

def getFaceOffs(fname, game):
    ''' gets the faceoffs of the game '''
    print("Loading Faceoffs...")

    with open(fname+'Marker.json') as f:
        data = json.load(f)

    faceoffs = []
    for t in data:
        time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        if event_type == "EventFaceoff":
            faceoff = event.Faceoff(game, time, t)
            game._faceoffs[round(faceoff._UTC_update,1)] = faceoff

    return game

def getPasses(fname, game):
    ''' This gets passing events in the game
    '''
    print("Loading Passes...")
    with open(fname+'Marker.json') as f:
        data = json.load(f)

    speeds = []
    dists = []

    events = []

    last_period = -1
    passes = []
    for t in data:
        time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        # if event_type not in events:
        #     events.append(event_type)
        if event_type == "EventPass":
        #     print(t["Marker"]["MarkerData"][0]["Descriptor_"], " turnover:", t["Marker"]["MarkerData"][0]["Properties"][0]['TurnoverType'])
            pass_event = event.Pass(game, time, t["Marker"]["MarkerData"][0]['Descriptor_'], t["Marker"]["MarkerData"])
            game._passes[round(pass_event._UTC_update,1)] = pass_event
            if pass_event._period != last_period:
                print("Passes in Period: ", pass_event._period)
                last_period = pass_event._period

    # speeds = np.array(speeds)
    # dists = np.array(dists)
    #
    # plt.plot(speeds, dists, 'o')
    # plt.xlabel("Speed (ft/0.1s)",fontsize=18)
    # plt.ylabel("Distance (ft)",fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    #
    # m, b = np.polyfit(speeds, dists, 1)
    # plt.plot(speeds, m*speeds + b)
    # plt.show()


    # for e in events:
    #     print(e)


    return game


def getTakeAways(fname, game):
    ''' This gets takeaways events in the game
    '''
    print("Loading TakeAways...")
    with open(fname+'Marker.json') as f:
        data = json.load(f)

    turnovers = []
    for t in data:
        time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        if event_type == "EventTurnover":
            to_type = t["Marker"]["MarkerData"][0]["Properties"][0]['TurnoverType']
            if to_type == 'TurnoverTypeTakeaway':
                takeaway_event = event.TakeAway(game, time, t["Marker"]["MarkerData"][0]['Descriptor_'], t["Marker"]["MarkerData"])
                game._takeaways[round(takeaway_event._UTC_update,1)] = takeaway_event
    return game

def getDistanceChange(player, time):
    arg_key = np.argmin(np.abs(np.array(list(player._hd_UTC_update.keys())) - time))
    key = list(player._hd_UTC_update.keys())[arg_key]

    time_key = round(key,1)
    time_key_p1 = round(list(player._hd_UTC_update.keys())[arg_key+1],1)
    time_diff = time_key_p1 - time_key
    puck_t0 =  {"X": player._hd_UTC_update[time_key]["X"], "Y": player._hd_UTC_update[time_key]["Y"]} #info_participants[0]["Participants"][1]["Location"]
    puck_t1 =  {"X": player._hd_UTC_update[time_key_p1]["X"], "Y": player._hd_UTC_update[time_key_p1]["Y"]}
    dist = event.euclideanDistance(puck_t0["X"], puck_t0["Y"], puck_t1["X"], puck_t1["Y"])
    dist = dist / (10*time_diff)
    player._speeds.append(dist)
    return dist

def getSpeeds(game):
    for e in list(game._entities.keys()):
        if (game._entities[e]._team == '14' or game._entities[e]._team == '25') and game._entities[e]._pos == 'G':
            print(game._entities[e])
            for shift in game._entities[e]._shifts:
                start = shift['start']
                end = shift['end']
                while start < end:
                    dist = getDistanceChange(start, end)
                    print(dist)
                    exit()
                    start+=0.1


def getEventSummary(fname, entities, game):
    ''' This function loads the Event/Game Summary of actions '''

    game._entities = entities
    print("Loading Events Summary...")
    with open(fname+'LiveEventSummary.json') as f:
        data = json.load(f)

    pos = False
    for t in range(len(data)):
        entity_data = data[t]["LiveEventSummary"]["EntitySummaries"]
        for ent in entity_data:
            if "Location" in ent.keys():
                game._entities[ent["EntityId"]]._onice.append(ent["OnPlayingSurface"])
                game._entities[ent["EntityId"]]._update_time.append(ent["LocationUTC"])
                game._entities[ent["EntityId"]]._possession_time.append(ent["PossessionTime"])
                game._entities[ent["EntityId"]]._locX.append(ent["Location"]["X"])
                game._entities[ent["EntityId"]]._locY.append(ent["Location"]["Y"])
                game._entities[ent["EntityId"]]._locZ.append(ent["Location"]["Z"])
                game._entities[ent["EntityId"]]._time_on_ice = ent['TimeOnTotal']

        # if pos:
            # game.graphGame()
        #     exit()

    # game.graphGame()
    return game

def getStartTime(clock_time, mins):
    ''' takes in a clock time of a goal/penalty and returns time idx of mins minutes before '''

    start_time_arr = clock_time.split("-")

    if (int(start_time_arr[1])+mins) >= 20:
        return start_time_arr[0] + "-20-0"
    elif (int(start_time_arr[1])+mins) < 0:
        return start_time_arr[0] + "-0-0"
    else:
        return start_time_arr[0] + "-" + str(int(start_time_arr[1])+mins) + "-" + start_time_arr[2]

def getTimeDicts(game, time_arr):
    # id_ch = {v: k for k, v in ch_id.items()}
    clock_ids = {}
    period = 1
    for i, t in enumerate(time_arr):
        if i > 1:
            if game._scoreboard[t]["ClockMinutes"] > game._scoreboard[time_arr[i-1]]["ClockMinutes"]:
                period+=1
        # print("Period: ", period, " CLOCK: ", game._scoreboard[t]["ClockMinutes"], ":",game._scoreboard[t]["ClockSeconds"])
        key = str(period) + "-" + str(game._scoreboard[t]["ClockMinutes"]) + "-" + str(game._scoreboard[t]["ClockSeconds"])
        if key not in clock_ids.keys():
            clock_ids[key] = t
    ids_clock = {v: k for k, v in clock_ids.items()}
    return ids_clock, clock_ids

def getPenalties(game, time_arr):
    penalties = {}
    for t_count, t in enumerate(time_arr):
        # home penalty
        if game._scoreboard[t]["HomeStrength"] < game._scoreboard[time_arr[t_count-1]]["HomeStrength"]:
            penalties[t] = event.Penalty(game, game._scoreboard[t], game._home_team_num)
        # away penalty
        elif game._scoreboard[t]["VisitorStrength"] < game._scoreboard[time_arr[t_count-1]]["VisitorStrength"]:
            penalties[t] = event.Penalty(game, game._scoreboard[t], game._visitor_team_num)
    return penalties

def getMomentum(games, fnames, game_idx):
    ''' calculates the momentum of the game '''

    game = games[game_idx]


    all_players_init = {}
    # scoreboard = {}
    for i, val in enumerate(list(game._entities.keys())):
        if val != '1':
            all_players_init[game._entities[val]._id] = {
                                        "player_team":game._teams[game._entities[val]._team]._full_name,
                                        "player_number":game._entities[val]._number,
                                        "player_last_name":game._entities[val]._last_name,
                                        "X": 0,
                                        "Y": 0,
                                        "goals": 0,
                                        "hits": 0,
                                        "passes": 0,
                                        "shots": 0,
                                        "blocks": 0,
                                        "let_shot_off": 0,
                                        "pass_plus_minus": 0,
                                        "turnover": 0,
                                        "possession":0,
                                        "points": 0}

    game._game_stats[0] = all_players_init # initialize all of the players to 0 values at the start
    game._momentum_data_empty = {"14":{"goals":[],
                                "hits":[],
                                "passes":[],
                                "pass_plus_minus":[],
                                "blocks":[],
                                "shots":[],
                                "shot_attempts": [],
                                "turnover":[],
                                "possession":[]},
                            "25":{"goals":[],
                                "hits":[],
                                "passes":[],
                                "pass_plus_minus":[],
                                "blocks":[],
                                "shots":[],
                                "shot_attempts": [],
                                "turnover":[],
                                "possession":[]}}

    game._momentum_data = copy.deepcopy(game._momentum_data_empty)


    # period = 0
    time_arr = np.sort(np.array(list(game._scoreboard.keys()))) #list(game._entities['1']._hd_UTC_update)
    ids_clock, clock_ids = getTimeDicts(game, time_arr)

    # --- set event
    penalties = getPenalties(game, time_arr)
    events = penalties #game._goals #penalties
    event_keys = np.sort(np.array(list(events.keys())))
    time_diff = -2 # penalty = -2, goal = 2

    # print(">>>EVENTS: ", events)

    for event_count, e in enumerate(event_keys):
        clock = 1
        gameSecond = None
        game_seconds = 0
        tampa_goals, dallas_goals = [], []
        possession_change = False
        cur_posession = None


        game._momentum = {}

        ''' Filter which charts I want to make '''
        if event_count == 4:
            event_oi = events[e]
            print(">>> Event: ", event_oi)
            key = np.argmin(np.absolute(np.array(list(ids_clock.keys())) - e))
            clock_time = ids_clock[list(ids_clock.keys())[key]]
            range_time = getStartTime(clock_time, mins=time_diff)

            # ----- goal
            # right_min = event_oi._clock_minutes
            # right_sec = event_oi._clock_seconds
            #
            # # -- run for 30 seconds after goal
            # # if right_sec - 30 < 0:
            # #     right_sec = 60 + (right_sec - 30)
            # #     right_min-=1
            # # else:
            # #     right_sec = right_sec - 30
            #
            # clock_time = range_time
            # event_name_str = 'Goal'
            # print(range_time, " --> ", right_min, ":", right_sec)

            # ----- Penalty
            right_min = int(range_time.split("-")[1]) - 1
            right_sec = int(range_time.split("-")[2])
            if right_min < 0:
                right_min = 0
                right_sec = 0
            # clock_time = clock_time[:-1] + str(int(clock_time[-1])-1)
            clock_time = getStartTime(clock_time, mins=2)
            event_name_str = 'Penalty'
            print(clock_time, " --> ", right_min, ":", right_sec)
            # print(event_count)
            # exit()




            for counter, t in enumerate(time_arr):
                # if t > clock_ids[start_time]:
                if t > clock_ids[clock_time]:

                    # game.plotMomentum(fnames[game_idx], event_oi, event_count, event_name=event_name_str, game_idx=game_idx)
                    # exit()

                    if np.amin(np.absolute(np.array(list(game._posessions.keys())) - t)) <= 0.1:
                        key_idx = np.argmin(np.absolute(np.array(list(game._posessions.keys())) - t))
                        cur_posession = game._posessions[list(game._posessions.keys())[key_idx]]

                    if cur_posession is not None:
                        if np.amin(np.absolute(cur_posession._UTC_end - t)) == 0.0:
                            possession_change = cur_posession._turnover
                        if t > cur_posession._UTC_end:
                            cur_posession = None
                            possession_change = False
                    else:
                        possession_change = False

                    # if game._scoreboard[t]["ClockState"] == "ClockStateRunning":
                    time_data, cur_posession = game.runGameSynchonusOffline(t, cur_posession, possession_change)

                    game._game_stats[t] = time_data

                    # new_clock = (period*(20*60)) + ((20*60) - (60*game._scoreboard[t]["ClockMinutes"]) - game._scoreboard[t]["ClockSeconds"])
                    new_clock = (game._scoreboard[t]["Period"]*(20*60)) + (60*game._scoreboard[t]["ClockMinutes"]) + (60-game._scoreboard[t]["ClockSeconds"])

                    # print(clock, " -- ", new_clock, " ", game._scoreboard[t]["ClockMinutes"], ":", game._scoreboard[t]["ClockSeconds"], " t: ", t)
                    if gameSecond == None:
                        gameSecond = event.GameSecond(0, game, t) #new_clock
                        game_seconds+=1
                    if new_clock == clock:
                        gameSecond.updateSecond(t)
                    else:
                        game_seconds+=1
                        game.updateMomentumData(gameSecond, cur_posession)


                        # try:
                        #     tbl_mom = game._momentum[list(game._momentum.keys())[-1]]['14']
                        # except:
                        #     tbl_mom = 0
                        # print(int(game._scoreboard[t]["ClockMinutes"]), " : ", int(game._scoreboard[t]["ClockSeconds"]), " rightmin: ", int(right_min), " right_sec: ", int(right_sec),
                                # " ", int(game._scoreboard[t]["ClockMinutes"]) == int(right_min), " ", int(game._scoreboard[t]["ClockSeconds"]) == int(right_sec), " TBL: ", tbl_mom)
                        # print(gameSecond)
                        # print(int(game._scoreboard[t]["ClockMinutes"]), " : ", int(game._scoreboard[t]["ClockSeconds"]), " elapsed: ", gameSecond._elapsed_seconds, " TBL: ", round(tbl_mom, 3), " // pos: ", gameSecond._stats['14']['possession'], " passes: ", gameSecond._stats['14']['passes'])

                        # if gameSecond._stats['14']["goals"] > 0:
                        #     tampa_goals.append(gameSecond._elapsed_seconds)
                        # elif gameSecond._stats['25']["goals"] > 0:
                        #     dallas_goals.append(gameSecond._elapsed_seconds)


                        # if game_seconds%(20*60) == 1 and game_seconds > 1: #gameSecond._elapsed_seconds%(20*60)==0 and gameSecond._elapsed_seconds!=0:
                        # if game._scoreboard[t]["ClockMinutes"] > game._scoreboard[time_arr[count-1]]["ClockMinutes"]:
                        if int(game._scoreboard[t]["ClockMinutes"]) == int(right_min) and int(game._scoreboard[t]["ClockSeconds"]) == int(right_sec):
                            # period+=1
                            # print("Timestep: ", counter, " of ", len(time_arr),
                            #                             "Period: ", period, #game._scoreboard[t]["Period"],
                            #                             " Time: ", game._scoreboard[t]["ClockMinutes"], ":",
                            #                             game._scoreboard[t]["ClockSeconds"], " t: ", t)
                            print(gameSecond)
                            # print(period)
                            print("clock: ", clock)
                            print("new clock: ", new_clock)
                            game.plotMomentum(fnames[game_idx], event_oi, event_count, clock=[right_min, right_sec], event_name=event_name_str, game_idx=game_idx)


                            game._momentum_data = copy.deepcopy(game._momentum_data_empty)
                            break

                        game.calculateMomentum(gameSecond._elapsed_seconds, t, cur_posession)
                        gameSecond = event.GameSecond(game_seconds, game, t) #new_clock
                        clock = new_clock


        # clock_seconds = game.updateMomentum(t) #_momentum_data[str(game._scoreboard["Period"])+"-"+game._scoreboard["ClockMinutes"]+":"+game._scoreboard["ClockSeconds"]]
        # print(clock_seconds)
        # if counter == 10: exit()
    return
    # exit()
    # game_fname = fname+'game/'+str(game._gameId)+".pkl"
    # with open(game_fname, 'wb') as output:
    #     pickle.dump(game, output, pickle.HIGHEST_PROTOCOL)
    # print("Game Saved after loading timesteps")
    # game.plotMomentum(fnames[game_idx], tampa_goals, dallas_goals)
    # game_fname = fnames[game_idx]+'game/'+str(game._gameId)+"_"+str(gameSecond._elapsed_seconds)+".pkl"
    # with open(game_fname, 'wb') as output:
    #     pickle.dump(game, output, pickle.HIGHEST_PROTOCOL)
    # print("Final game Saved after ", gameSecond._elapsed_seconds, " Seconds...")




def playGame(fname, game): #time is the time_key for the game
    ''' plays the game synchonously '''

    col_lst = ['ID','First','Last','Points']
    home_results = pd.DataFrame(columns=col_lst)
    away_results = pd.DataFrame(columns=col_lst)

    all_players_init = {}
    # scoreboard = {}
    for i, val in enumerate(list(game._entities.keys())):
        if val != '1':
            if game._entities[val]._team == game._home_team_num:
                home_results.loc[i] = [game._entities[val]._id, game._entities[val]._first_name, game._entities[val]._last_name, 0]
            if game._entities[val]._team == game._visitor_team_num:
                away_results.loc[i] = [game._entities[val]._id, game._entities[val]._first_name, game._entities[val]._last_name, 0]


            all_players_init[game._entities[val]._id] = {
                                        "player_team":game._teams[game._entities[val]._team]._full_name,
                                        "player_number":game._entities[val]._number,
                                        "player_last_name":game._entities[val]._last_name,
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

    game._game_stats[0] = all_players_init # initialize all of the players to 0 values at the start
    # print(game._game_stats[0])

    possession_change = False
    cur_posession = None
    time_arr = list(game._scoreboard.keys()) #list(game._entities['1']._hd_UTC_update)
    for counter, t in enumerate(time_arr):
        if counter <= 5000: #print("Timestep: ", counter, " of ", len(time_arr))
            if counter%500 == 0: print("Timestep: ", counter, " of ", len(time_arr))
                # print()
            if np.amin(np.absolute(np.array(list(game._posessions.keys())) - t)) <= 0.1:
                key_idx = np.argmin(np.absolute(np.array(list(game._posessions.keys())) - t))
                time_key = list(game._posessions.keys())[key_idx]
                cur_posession = game._posessions[time_key]

            if cur_posession is not None:
                if np.amin(np.absolute(cur_posession._UTC_end - t)) == 0.0:
                    possession_change = cur_posession._turnover
                if t > cur_posession._UTC_end:
                    cur_posession = None
                    possession_change = False

            else:
                possession_change = False
            time_data, cur_posession = game.runGameSynchonusOffline(t, cur_posession, possession_change)
            game._game_stats[t] = time_data

    # game_fname = fname+'game/'+str(game._gameId)+".pkl"
    # with open(game_fname, 'wb') as output:
    #     pickle.dump(game, output, pickle.HIGHEST_PROTOCOL)
    # print("Game Saved after loading timesteps")

    return game


def assignMovement(fname, entities):
    ''' This function traverses the EntityTracking.json file
    and assigns position data to entities. I think this is higher resolution
    data than the EventSummary.'''

    with open(fname+'EntityTracking.json') as f:
        tracking_data = json.load(f)

    print("Tracking data loading...")
    # tracking_data = tracking_data[0]

    for i,event in enumerate(tracking_data):
        id = event["EntityTracking"]["TrackingData"][0]["EntityId"]
        entities[id]._counter.append(event["Counter"])
        entities[id]._update_time.append(round(event["EntityTracking"]["TrackingData"][0]["LocationUTC"],1))
        entities[id]._hd_UTC_update[round(event["EntityTracking"]["TrackingData"][0]["LocationUTC"],1)] = {'_onice':event["EntityTracking"]["TrackingData"][0]["OnPlayingSurface"],
                                                            'X': event["EntityTracking"]["TrackingData"][0]["Location"]['X'],
                                                            'Y': event["EntityTracking"]["TrackingData"][0]["Location"]['Y'],
                                                            'Z': event["EntityTracking"]["TrackingData"][0]["Location"]['Z']}
        entities[id]._locX.append(event["EntityTracking"]["TrackingData"][0]["Location"]['X'])
        entities[id]._locY.append(event["EntityTracking"]["TrackingData"][0]["Location"]['Y'])
        entities[id]._locZ.append(event["EntityTracking"]["TrackingData"][0]["Location"]['Z'])
        entities[id]._onice.append(event["EntityTracking"]["TrackingData"][0]["OnPlayingSurface"])
        try:
            entities[id]._hd_UTC_update[round(event["EntityTracking"]["TrackingData"][0]["LocationUTC"],1)]['velX'] = event["EntityTracking"]["TrackingData"][0]["Velocity"]['X']
            entities[id]._hd_UTC_update[round(event["EntityTracking"]["TrackingData"][0]["LocationUTC"],1)]['velY'] = event["EntityTracking"]["TrackingData"][0]["Velocity"]['Y']
            entities[id]._hd_UTC_update[round(event["EntityTracking"]["TrackingData"][0]["LocationUTC"],1)]['velZ'] = event["EntityTracking"]["TrackingData"][0]["Velocity"]['Z']
            entities[id]._velX.append(event["EntityTracking"]["TrackingData"][0]["Velocity"]['X'])
            entities[id]._velY.append(event["EntityTracking"]["TrackingData"][0]["Velocity"]['Y'])
            entities[id]._velZ.append(event["EntityTracking"]["TrackingData"][0]["Velocity"]['Z'])
            entities[id]._accX.append(event["EntityTracking"]["TrackingData"][0]["Acceleration"]['X'])
            entities[id]._accY.append(event["EntityTracking"]["TrackingData"][0]["Acceleration"]['Y'])
            entities[id]._accZ.append(event["EntityTracking"]["TrackingData"][0]["Acceleration"]['Z'])
        except:
            pass
            # print(event["EntityTracking"]["TrackingData"][0]["OnPlayingSurface"])

        # if len(entities[id]._locX) > 10:
        #     entities[id].plotMovement()
        #     exit()
    return entities



# NOTE: START HERE

def initGame(fname,loadplayers=False, loadgame=False):
    # print("hello")
    if loadgame:
        # print("Loading game")
        game_fnames = os.listdir(fname+"game/")
        for game_fname in game_fnames:
            # if game_fname[-8:] != 'Game.pkl':
            if game_fname[-7:-5] == '41': #'415.pkl'
                # print("Loading game: ", fname+"/game/"+game_fname)
                game = rink.loadGame(fname+"/game/"+game_fname)
        return game
    else:
        print("Making new game")
        rink_obj = rink.Rink(fname)
        teams = makeTeams(fname)
        game = getGame(fname,teams)
        game._rink = rink_obj

        if not loadplayers:
            entities = makePlayers(fname, game)
            entities = assignMovement(fname, entities)

            # print("Saving")
            for i in list(entities.keys()):
                # entities[i].getShifts(rink_obj)
                entities[i].savePlayer(fname)
        else:
            print("Loading Entities...")
            entities = {}
            entity_fnames = os.listdir(fname+"/players/")
            for player_fname in entity_fnames:
                if player_fname[-3:] == 'pkl':
                    player = player_factory.loadPlayer(fname+"/players/"+player_fname)
                    # player.getShifts(rink_obj)
                    entities[player._id] = player

        # heartbeat = getHeartbeat(fname)
        game = getEventSummary(fname, entities, game)

        # SCOREBOARD
        game = getScoreBoard(fname, game)

        # GOALS
        game = getGoals(fname, game)

        # FACEOFFS
        game = getFaceOffs(fname, game)

        # SHOTS
        game = getShots(fname, game)

        # HITS
        game = getHits(fname, game)

        # PASSES
        game = getPasses(fname, game)
        # print_to_terminal.rankPassingData(game)

        # POSESSION/PRESSURE
        game = getPosessions(fname, game)

        game = getTakeAways(fname, game)

        game_fname = fname+'game/'+str(game._gameId)+".pkl"
        with open(game_fname, 'wb') as output:
            pickle.dump(game, output, pickle.HIGHEST_PROTOCOL)
        print("Game Saved")

    return game #playGame(game)


def SOGonly(game):
    ''' limit shots to those "on goal" '''

    for e in game._entities.keys():
        game._entities[e]._pressure_to_shoot = []
        game._entities[e]._pressure_on_shooter = []

    sog = {}
    for p in game._shots.keys():
        if not game._shots[p]._blocked and not game._shots[p]._miss:
            sog[p] = game._shots[p]
            game._shots[p]._shooter._pressure_to_shoot.append(game._shots[p]._shot_pressure)
            for presser_id in game._shots[p]._shot_presser_lst.keys():
                game._entities[presser_id]._pressure_on_shooter.append(game._shots[p]._shot_presser_lst[presser_id])
    game._shots = sog
    return game





def recalculatePasses(games, fnames):

    for g in games:
        for pl in g._entities.keys():
            g._entities[pl].resetPasses()

    risk = []
    for game_num, g in enumerate(games):
        for p in g._passes.keys():
            g._passes[p].resetPass()
            g._passes[p].beatOpponents()
            g._entities[g._passes[p]._passer._id]._pass_risk.append(g._passes[p]._pass_risk)


        game_fname = fnames[game_num]+'game/'+str(g._gameId)+".pkl"
        with open(game_fname, 'wb') as output:
            pickle.dump(g, output, pickle.HIGHEST_PROTOCOL)
        print("Game Saved")


fnames = ['../../2019030415/', '../../2019030416/']
# fnames = ['../../2019030415/']

if __name__=="__main__":

    games = []
    for f_count, f in enumerate(fnames):
        g = initGame(fname=f, loadgame=True)

        if f_count == 1:
            # adding goal to second game because of shitty data
            for s in g._shots.keys():
                if g._shots[s]._period == 1 and g._shots[s]._clock_minutes == 7 and g._shots[s]._clock_seconds == 39:
                    g._goals[s] = event.Goal(g, s, [], s, g._shots[s]._period, g._shots[s]._clock_minutes, g._shots[s]._clock_seconds, g._shots[s]._shooter, g._shots[s]._shooter)
            # print(g._goals)

        # g.cleanPossessions()
        # g = SOGonly(g)
        games.append(g)



    # exit()
    # for game_num in range(2):
    #     getMomentum(games, fnames, game_num)
    #     exit()
    # getMomentum(games, fnames, 1)
    ''' reset functions '''
    # recalculatePasses(games, fnames)

    ''' calculate '''
    # _ = print_to_terminal.rankAnalyticsCustom(games)

    ''' plotting for paper '''
    # paper_plots.makeZoP()
    # paper_plots.PAvHist(games)
    paper_plots.makePAvPlots(games)
    # paper_plots.passPressureLaneScatter(games)
    # paper_plots.plot2D(games)
    # paper_plots.plotPassChart(games)
    # paper_plots.plotPressureChart(games)
    # paper_plots.plotMomentumLoad(fname='../../2019030415/', game=game)

    ''' play game temporally '''
    # playGame(fname='../../2019030415/', game=game)
