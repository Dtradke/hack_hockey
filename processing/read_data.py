import json
import sys
import numpy as np
import pandas as pd
import pickle
import os

import player_factory
import rink
import event
import print_to_terminal

fname = sys.argv[-1]

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

    game = rink.Game(data[0]["Event"], teams=teams)

    for update in data:
        if update["Event"]["ActualStartUTC"] > game._UTC_start:
            game._UTC_start = update["Event"]["ActualStartUTC"]
        if update["Event"]["ActualEndUTC"] > game._UTC_end:
            game._UTC_end = update["Event"]["ActualEndUTC"]
    return game

def getPressure(fname, game):
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


    for i in range(len(possession_changes)-1):
        if len(possession_changes[i]["Marker"]["MarkerData"][0]["Participants"]) > 0 and possession_changes[i]["Marker"]["MarkerData"][0]["Participants"][0]["EntityId"] != '':
            posession = event.Posession(game, possession_changes[i], possession_changes[i+1])
            game._teams[posession._posessor._team]._posessions.append(posession)
            game._entities[posession._posessor._id]._posession_lst.append(posession)
            # print(s)
            # posession.plotPosession()


    return game

def getPasses(fname, game):
    ''' This gets passing events in the game
    '''
    print("Loading Passes...")
    with open(fname+'Marker.json') as f:
        data = json.load(f)

    passes = []
    for t in data:
        time = t["PacketSendUTC"]
        event_type = t["Marker"]["MarkerData"][0]["MinorType"]
        if event_type == "EventPass":
            pass_event = event.Pass(game, time, t["Marker"]["MarkerData"][0]['Descriptor_'], t["Marker"]["MarkerData"])
            passes.append(pass_event)

    # print(len(passes))
    # for e in game._entities.keys():
    #     print(game._entities[e])
    # exit()

    return game


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
            # if ent["PossessionTime"] > 0:
            #     print(ent)
            #     print(game._entities[ent["EntityId"]])
            #     pos = True
            if "Location" in ent.keys():
                game._entities[ent["EntityId"]]._onice.append(ent["OnPlayingSurface"])
                game._entities[ent["EntityId"]]._update_time.append(ent["LocationUTC"])
                game._entities[ent["EntityId"]]._possession_time.append(ent["PossessionTime"])
                game._entities[ent["EntityId"]]._locX.append(ent["Location"]["X"])
                game._entities[ent["EntityId"]]._locY.append(ent["Location"]["Y"])
                game._entities[ent["EntityId"]]._locZ.append(ent["Location"]["Z"])
        # if pos:
            # game.graphGame()
        #     exit()

    # game.graphGame()
    return game

def assignMovement(entities):
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

rink_obj = rink.Rink()

teams = makeTeams(fname)
game = getGame(fname,teams)
game._rink = rink_obj



if "load" not in sys.argv:
    entities = makePlayers(fname, game)
    entities = assignMovement(entities)

    print("Saving")
    for i in list(entities.keys()):
        entities[i].savePlayer()
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


# PASSES
# game = getPasses(fname, game)
# print_to_terminal.rankPassingData(game)

# POSESSION/PRESSURE
game = getPressure(fname, game)
print_to_terminal.rankPosessionData(game)


# for i in entities.keys():
    # print(entities[i]._last_name, ": ", len(entities[i]._counter))
    # if entities[i]._last_name == "Johnson":
    #     entities[i].plotMovement()
    #     exit()
