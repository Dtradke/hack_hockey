import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getAvgORZero(arr):
    if len(arr) == 0:
        return 0.0
    else:
        arr = np.array(arr)
        return round(np.mean(arr),2)


def combinedDF(teams):
    passing_lanes, positions, names = [], [], []

    # col_lst = ['Last', '&', 'POS', '&', 'TOP', '&', 'PrMOV', '&', 'PrSH', '&', 'PrSHOP', "&",'PASS', "&",'MEAN-POV', "&",'MEAN-TBT', "&", "MRatio", "&",'POV', "&",'TBT', "&",
                # 'PPM', "&", 'SPPM', "&", "TO", "&", 'PAv', '&', 'Omega', 'newline']
    col_lst = ['Last', '&', 'PAv', '&', 'Omega', '&', 'PASS', '&', 'MOV', '&', 'TOV', "&",'TBT', "&",'PPM', "&",'NPPM', "&", "TO", "&",'PrMOV', "&",'PrSH', "&",
                'PrOS', 'newline']
    df = pd.DataFrame(columns=col_lst)

    count = -1
    for i, t in enumerate(teams.keys()):
        team = teams[t] #list of players on a team
        for j, player in enumerate(team):
            # clean players without possessions
            if player["possessions"] == 0: dop = 0
            else: dop = round(player["DOP"] / player["possessions"],2)

            if getAvgORZero(player["MTBT"]) != 0:
                mean_ratio = getAvgORZero(player["MPOV"]) / getAvgORZero(player["MTBT"])
            else:
                mean_ratio = 0

            count+=1
            df.loc[count] = [
                        # player["id"],
                        # player["first_name"],
                        player["last_name"],
                        " & ",
                        # openness
                        getAvgORZero(player["PA"]),
                        " & ",
                        getAvgORZero(player["omega"]),
                        # " & ",
                        # player["position"],
                        # " & ",
                        " & ",
                        # pass
                        round(player["TP"] / player["games"],2),
                        # means
                        " & ",
                        getAvgORZero(player["MPOV"]),
                        " & ",
                        round(player["PO"] / player["games"],2),
                        " & ",
                        round(player["TB"] / player["games"],2),
                        " & ",
                        round(player["PPM"] / player["games"],2),
                        " & ",
                        round(player["SPPM"] / player["games"],2),
                        " & ",
                        round(player["turnovers"] / player["games"],2),

                        # possession
                        # round(player["possessions"] / player["games"],2),
                        # " & ",
                        # dop, #duration of possession
                        " & ",
                        getAvgORZero(player["PMv"]),
                        " & ",
                        getAvgORZero(player["PWS"]),
                        " & ",
                        getAvgORZero(player["PSh"]),

                        # " & ",
                        # getAvgORZero(player["MTBT"]),
                        # " & ",
                        # (mean_ratio),
                        # totals


                        " \\\\\\hline"
                        ]
    df = df.sort_values('NPPM', ascending=False)
    print(df.to_string(index=False))
    print("AVG PPM: ", np.mean(np.array(df["PPM"].to_list())))
    # passing_lanes = passing_lanes + df["Omega"].to_list()
    # positions = positions + df["Position"].to_list()
    # names = names + df["Last"].to_list()
    exit()


def possessionDF(teams):
    for t in teams.keys():
        team = teams[t] #list of players on a team
        # col_lst = ['Last', "&1", 'Position', "&2",'POS', "&3",'TOP', "&4",'Steals', "&4", 'TO', "&5", 'PMv'
        #             , "&6", 'Shots', "&7", 'PWS', "&8", 'Omega', "&9", "PSh"]
        col_lst = ['Last', "&", 'Position', '&', 'TOI', '&', 'TOP', '&', 'CTRL', "&",'POS', "&",'DOP', "&", 'PMv'
                    , "&", 'PWS', "&", "PSh", 'newline']
        df = pd.DataFrame(columns=col_lst)
        for i, player in enumerate(team):
            df.loc[i] = [
                        # player["id"],
                        # player["first_name"],
                        player["last_name"],
                        " & ",
                        player["position"],
                        " & ",
                        round(player["time_on_ice"] / player["games"],2), # avg time on ice in minutes
                        " & ",
                        round(player["time_of_possession"] / player["games"],2), # avg time of possession in minutes
                        " & ",
                        round((player["time_of_possession"] / player["games"]) / (player["time_on_ice"] / player["games"]),2),
                        " & ",
                        round(player["possessions"] / player["games"],2),
                        " & ",
                        round(player["DOP"] / player["possessions"],2), #duration of possession
                        # " & ",
                        # round(player["steals"] / player["games"],2),
                        # " & ",
                        # round(player["turnovers"] / player["games"],2),
                        " & ",
                        getAvgORZero(player["PMv"]),
                        # " & ",
                        # round(player["shots"] / player["games"],2),
                        " & ",
                        getAvgORZero(player["PWS"]),
                        " & ",
                        getAvgORZero(player["PSh"]),
                        " \\\\\\hline"
                        ]
        print("TEAM: ", t)
        df = df.sort_values('POS', ascending=False)
        print(df.to_string(index=False))

def passingDF(teams):
    passing_lanes, positions, names = [], [], []
    for t in teams.keys():
        team = teams[t] #list of players on a team
        col_lst = ['Last', "&", 'Position', "&",'PASS', "&",'POV', "&", "OFF_IMP", "&",'TBEAT', "&",
                    'PPM', "&", 'SPPM', "&", "TO", "&", "TAKE", "&", "TAKE_DIFF", "&", 'PAv', '&', 'Omega', 'newline']
        df = pd.DataFrame(columns=col_lst)
        for i, player in enumerate(team):
            df.loc[i] = [
                        # player["id"],
                        # player["first_name"],
                        player["last_name"],
                        " & ",
                        player["position"],
                        " & ",
                        round(player["TP"] / player["games"],2),
                        " & ",
                        round(player["PO"] / player["games"],2),
                        " & ",
                        round(player["PO"] / player["TP"],2), # offensive impact
                        " & ",
                        round(player["TB"] / player["games"],2),
                        " & ",
                        round(player["PPM"] / player["games"],2),
                        " & ",
                        round(player["SPPM"] / player["games"],2),
                        " & ",
                        round(player["turnovers"] / player["games"],2),
                        " & ",
                        round(player["takeaways"] / player["games"],2),
                        " & ",
                        getAvgORZero(player["takeaway_difficulty"]),
                        " & ",
                        getAvgORZero(player["PA"]),
                        " & ",
                        getAvgORZero(player["omega"]),
                        " \\\\\\hline"
                        ]
        print("TEAM: ", t)
        df = df.sort_values('SPPM', ascending=False)
        print(df.to_string(index=False))
        print("AVG PPM: ", np.mean(np.array(df["PPM"].to_list())))
        passing_lanes = passing_lanes + df["Omega"].to_list()
        positions = positions + df["Position"].to_list()
        names = names + df["Last"].to_list()

    # lanes = np.array(passing_lanes)
    # poses = np.array(positions)
    #
    # p = lanes.argsort()
    # lanes = lanes[p]
    # poses = poses[p]
    #
    # fig, ax = plt.subplots()
    # N, bins, patches = ax.hist(lanes, bins=38, edgecolor='white', linewidth=1)
    #
    #
    # for i in range(len(patches)):
    #     if poses[i] == 'Def':
    #         patches[i].set_facecolor('r')
    #     else:
    #         patches[i].set_facecolor('black')
    #
    # # plt.hist(lanes, bins=lanes.shape[0])
    # plt.show()
    # exit()

def rankAnalyticsCustom(games):
    players = {}
    teams = {}

    # oratio, bratio = [], []
    # for game in games:
    #     for e in game._entities.keys():
    #         if game._entities[e]._last_name == 'Bogosian':
    #             oratio = oratio + game._entities[e]._overtook_ratio
    #             bratio = bratio + game._entities[e]._beaten_ratio
    #
    # print("overtook ratio: ", oratio)
    # print("overtook ratio mean: ", np.mean(np.array(oratio)))
    #
    # print("overtook ratio: ", bratio)
    # print("overtook ratio mean: ", np.mean(np.array(bratio)))
    # exit()

    for game in games:
        for e in game._entities.keys():
            # if game._entities[e]._pos == 'G':
            #     continue
            if game._entities[e]._team == game._home_team_num or game._entities[e]._team == game._visitor_team_num: #checks if player
                # ----- add team to dict
                if game._entities[e]._team not in teams.keys():
                    teams[game._teams[game._entities[e]._team]._full_name] = []

                players[game._entities[e]._id] = {"id": game._entities[e]._id,
                                                    "first_name": game._entities[e]._first_name,
                                                    "last_name": game._entities[e]._last_name,
                                                    "position": game._entities[e]._pos,
                                                    # "position": "Def" if game._entities[e]._pos == "D" else "Fwd",
                                                    "time_on_ice": 0,
                                                    "time_of_possession":0,
                                                    "possessions":0,
                                                    "DOP":0,
                                                    "steals":0,
                                                    "turnovers":0,
                                                    "takeaways":0,
                                                    "takeaway_difficulty":[],
                                                    "PMv":[],
                                                    "shots":0,
                                                    "PWS":[],
                                                    "omega":[],
                                                    "PSh":[],
                                                    "TP":0,
                                                    "MPOV":[],
                                                    "MTBT":[],
                                                    "PO":0,
                                                    "TB":0,
                                                    "PPM":0,
                                                    "SPPM":0,
                                                    "PA":[],
                                                    "team":game._teams[game._entities[e]._team]._full_name,
                                                    "code":game._entities[e]._team,
                                                    "games":0}


    for i, game in enumerate(games):
        for e in game._entities.keys():
            # if game._entities[e]._pos == 'G':
            #     continue
            if game._entities[e]._team == game._home_team_num or game._entities[e]._team == game._visitor_team_num: #checks if player
                if players[game._entities[e]._id]["position"] == 'G' and i == 0: players[game._entities[e]._id]["time_on_ice"] += 89.65 #overtime
                elif players[game._entities[e]._id]["position"] == 'G' and i == 1: players[game._entities[e]._id]["time_on_ice"] += 60
                else: players[game._entities[e]._id]["time_on_ice"] += (game._entities[e]._time_on_ice / 60)                      # time on ice in minutes

                players[game._entities[e]._id]["time_of_possession"] += (game._entities[e]._time_of_possession / 10) / 60 # time of possession in minutes
                players[game._entities[e]._id]["possessions"] += len(game._entities[e]._posession_lst)
                players[game._entities[e]._id]["DOP"] += game._entities[e]._posession_time
                players[game._entities[e]._id]["steals"] += game._entities[e]._steals
                players[game._entities[e]._id]["turnovers"] += game._entities[e]._turnovers
                players[game._entities[e]._id]["takeaways"] += game._entities[e]._takeaways
                players[game._entities[e]._id]["takeaway_difficulty"] += game._entities[e]._takeaway_difficulty
                players[game._entities[e]._id]["PMv"] += game._entities[e]._pressure_to_pass #array
                players[game._entities[e]._id]["shots"] += len(game._entities[e]._pressure_to_shoot)
                players[game._entities[e]._id]["PWS"] += game._entities[e]._pressure_to_shoot #array
                players[game._entities[e]._id]["omega"] += game._entities[e]._openness #array
                players[game._entities[e]._id]["PSh"] += game._entities[e]._pressure_on_shooter #array
                players[game._entities[e]._id]["TP"] += game._entities[e].total_passes
                players[game._entities[e]._id]["PO"] += game._entities[e]._overtook_val # len(game._entities[e]._overtook)
                players[game._entities[e]._id]["TB"] += game._entities[e]._beaten_val # len(game._entities[e]._beaten)
                players[game._entities[e]._id]["MPOV"] += game._entities[e]._overtook_ratio # len(game._entities[e]._overtook)
                players[game._entities[e]._id]["MTBT"] += game._entities[e]._beaten_ratio # len(game._entities[e]._beaten)
                players[game._entities[e]._id]["PPM"] += game._entities[e]._overtook_val - game._entities[e]._beaten_val #len(game._entities[e]._overtook) - len(game._entities[e]._beaten)

                if game._entities[e].total_passes == 0:
                    players[game._entities[e]._id]["SPPM"] = 0
                else:
                    players[game._entities[e]._id]["SPPM"] += ((game._entities[e]._overtook_val - game._entities[e]._beaten_val) / (len(game._entities[e]._overtook_ratio) + len(game._entities[e]._beaten_ratio)))
                    # players[game._entities[e]._id]["SPPM"] += (game._entities[e]._overtook_val - game._entities[e]._beaten_val) / game._entities[e].total_passes #(len(game._entities[e]._overtook) - len(game._entities[e]._beaten)) / game._entities[e].total_passes
                players[game._entities[e]._id]["PA"] += game._entities[e]._pass_risk
                if i == 0: players[game._entities[e]._id]["games"] += (89.65/60) #overtime
                else: players[game._entities[e]._id]["games"] += 1

    for p in players.keys():
        # if game._entities[p]._last_name == 'Cernak':
        #     print(len(players[p]["PA"]))
        #     print(getAvgORZero(players[p]["PA"]))
        #     exit()
        teams[players[p]["team"]].append(players[p])

    # possessionDF(teams)
    # passingDF(teams)
    # combinedDF(teams)
    return teams
