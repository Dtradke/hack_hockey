import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import player_factory
import rink
import event
import print_to_terminal



salaries = {
            "Hedman":	7875000,
            "Cernak":	890000,
            "McDonagh":	6750000,
            "Sergachev":	7007498,
            "Shattenkirk":	3900000,
            "Rutta":	1300000,
            "Vasilevskiy":	9500000,
            "Point":	6750000,
            "Palat":	5300000,
            "McElhinney":	1300000,
            "Gourde":	5166666,
            "Goodrow":	925000,
            "Kucherov":	9500000,
            "Paquette":	1650000,
            "Johnson":	5000000,
            "Cirelli":	935833,
            "Killorn":	4450000,
            "Maroon":	900000,
            "Verhaeghe":	1000000,
            "Coleman":	1800000,
            "Volkov": 925000,
            "Bogosian": 5142857,

            # dallas
            "Lindell":	5800000,
            "Hanley":	700000,
            "Heiskanen":	3394166,
            "Klingberg":	4250000,
            "Oleksiak":	2137500,
            "Khudobin":	3333333,
            "Dickinson":	1500000,
            "Sekera":	1500000,
            "Oettinger":	1387500,
            "Janmark":	2300000,
            "Caamano":	738333,
            "Cogliano":	3250000,
            "Radulov":	6250000,
            "Dowling":	750000,
            "Kiviranta":	925000,
            "Perry":	8625000,
            "Gurianov":	2550000,
            "Seguin":	9850000,
            "Benn":	9500000,
            "Pavelski":	7000000,
            "Bishop":	4916666
            }

def func3(x,y, polygon):
    # return (1- x/2 + x**5 + y**3)*np.exp(-x**2-y**2)

    # cx = int(x.shape[0] * (1/3))
    # cy = int(x.shape[0] * (1/2))


    Z2 = np.zeros((x.shape))

    for i, yy in enumerate(y[:,0]):
        for j, xx in enumerate(x[0]):
            pt = Point(xx, yy)
            if polygon.contains(pt):
                rad_to_opp = event.getRadians(0,0,xx,yy)
                rad_diff = np.pi - rad_to_opp
                L = event.getL(rad_diff, 5, 15)
                d = event.euclideanDistance(0,0,xx,yy)
                Z2[i,j] = 1 - (d/L)

    return Z2

def makeZoP():
    plt.style.use('default')
    border = []
    bx, by = [], []
    for i in np.linspace(0,(2*np.pi),200):
        L = event.getL(i, 5, 15)
        border.append((-(np.cos(i)*L), -(np.sin(i)*L)))
        bx.append(-(np.cos(i)*L))
        by.append(-(np.sin(i)*L))

    polygon = Polygon(border)

    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    x = np.arange(-7.0, 17.0, dx)
    y = np.arange(-10.0, 10.0, dy)
    X,Y = np.meshgrid(x, y)

    xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
    extent = xmin, xmax, ymin, ymax
    fig = plt.figure()
    Z2 = func3(X, Y, polygon)
    im2 = plt.imshow(Z2, cmap='hot_r', origin='lower', interpolation='bilinear', extent=extent) #, alpha=.9
    plt.plot(bx, by, c='darkviolet', linewidth=2, label='Zone of Pressure')
    # plt.plot(np.array(bx)/2, np.array(by)/2, c='k', linewidth=1, linestyle='--')
    # plt.text(5,-4.5,"$o(p) = 0.5$", fontsize=14)
    plt.scatter(0,0, c='k', s=70, label='$p$')
    plt.xlabel("Front/Back Distance (ft)", fontsize=20)
    plt.ylabel("Side/Side Distance (ft)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.text(8.5,0.2,"$V^{threat}$", fontsize=18)
    plt.arrow(0, 0, 11, 0, head_width=1, head_length=3, color='k')
    plt.legend(loc='upper left', fontsize=14)
    plt.colorbar(im2)
    plt.title("Zone of Pressure for $p$", fontsize=20)
    fname = "../../Paper/paper_imgs/ZoP/ZoP_gradient_red.png"
    plt.savefig(fname,bbox_inches='tight', dpi=300)
    plt.close()
    # plt.show()
    exit()



def plotMomentumLoad(fname, game):
    # y = []
    #
    # for i, val in enumerate(self._momentum.keys()):
    #     y.append(self._momentum[val]['14'])

    fig, ax1 = plt.subplots(1,1)
    y = np.load(fname+"/momentum"+str((20*60))+"sec.npy")

    min_count = 0
    min_resolution = 2
    xtick_vals, xtick_labels = [], []
    for i in range(y.shape[0]):
        if i%(min_resolution*60) == 0:
            xtick_vals.append(i)
            xtick_labels.append(str(min_resolution*(i/(min_resolution*60))))

    ax1.fill_between(np.arange(y.shape[0]), y, 0,color='b', label='Tampa Bay Momentum')
    ax1.fill_between(np.arange(y.shape[0]), y, 1,color='g', label='Dallas Momentum')

    tampa_goals, dallas_goals = getQuickPlotGoals(game)
    tampa_penalty, dallas_penalty = getQuickPlotPenalties(game)

    for i, tg in enumerate(tampa_goals):
        if tg <= y.shape[0]:
            if i == 0: ax1.axvline(tg, c='w', linewidth=5, label='Tampa Bay Goal')
            else: ax1.axvline(tg, c='w', linewidth=5)
    for i, dg in enumerate(dallas_goals):
        if dg <= y.shape[0]:
            if i == 0: ax1.axvline(dg, c='y', linewidth=5, label='Dallas Goal')
            else: ax1.axvline(dg, c='y', linewidth=5)
    for i, tp in enumerate(tampa_penalty):
        if tp <= y.shape[0]:
            if i == 0: ax1.axvline(tp, c='w', linestyle=':', linewidth=5, label='Tampa Bay Penalty')
            else: ax1.axvline(tp, c='w', linestyle=':', linewidth=5)
    for i, dp in enumerate(dallas_penalty):
        if dp <= y.shape[0]:
            if i == 0: ax1.axvline(dp, c='y', linestyle=':', linewidth=5, label='Dallas Penalty')
            else: ax1.axvline(dp, c='y', linestyle=':', linewidth=5)

    ax1.set_ylabel("Momentum", fontsize=25)
    plt.xticks(xtick_vals, xtick_labels)
    ax1.set_xlabel("Minutes", fontsize=25)
    # plt.title("Dallas Stars")
    plt.legend()
    plt.show()
    exit()

def getQuickPlotGoals(game):
    tampa_goals, dallas_goals = [], []
    for g in list(game._goals.keys()):
        goal = game._goals[g]
        second = ((goal._period-1)*(20*60)) + ((20*60) - (60*goal._clock_minutes) - goal._clock_seconds)
        if goal._teamId == '14':
            tampa_goals.append(second)
        else:
            dallas_goals.append(second)
    return tampa_goals, dallas_goals

def getQuickPlotPenalties(game):                                        # FIXME: home and away switch between games
    tampa_penalty, dallas_penalty = [], []
    last_home_strength, last_visitor_strength = 6, 6

    time_arr = list(game._scoreboard.keys()) #list(game._entities['1']._hd_UTC_update)
    for counter, t in enumerate(time_arr):
        home_strength = game._scoreboard[t]["HomeStrength"]
        visitor_strength = game._scoreboard[t]["VisitorStrength"]
        if home_strength < last_home_strength:
            penalty_second = (game._scoreboard[t]["Period"]-1) + ((20*60) - (60*game._scoreboard[t]["ClockMinutes"]) - game._scoreboard[t]["ClockSeconds"])
            tampa_penalty.append(penalty_second)
        elif visitor_strength < last_visitor_strength:
            penalty_second = (game._scoreboard[t]["Period"]-1) + ((20*60) - (60*game._scoreboard[t]["ClockMinutes"]) - game._scoreboard[t]["ClockSeconds"])
            dallas_penalty.append(penalty_second)
        last_home_strength = home_strength
        last_visitor_strength = visitor_strength
    return tampa_penalty, dallas_penalty

def plot2D(games):
    players = {}

    for game in games:
        for e in game._entities.keys():
            if game._entities[e]._pos == 'G':
                continue
            if game._entities[e]._team == game._home_team_num or game._entities[e]._team == game._visitor_team_num: #checks if player
                players[game._entities[e]._last_name] = {"team":game._teams[game._entities[e]._team]._full_name,
                                                        "code":game._entities[e]._team,
                                                        "pos":[0, 0],
                                                        "games":0}

    # for game in games:
    #     for e in game._entities.keys():
    #         if game._entities[e]._pos == 'G':
    #             continue
    #         if game._entities[e]._team == game._home_team_num or game._entities[e]._team == game._visitor_team_num: #checks if player
    #             ppm = (len(game._entities[e]._overtook) - len(game._entities[e]._beaten)) #/ game._entities[e].total_passes
    #             x = game._entities[e]._turnovers #ppm #game._entities[e]._turnovers
    #             # if game._entities[e]._team == '14': x = 0
    #             # else: x = 1
    #             y = print_to_terminal.getAvgORZero(game._entities[e]._pressure_to_pass) #salaries[game._entities[e]._last_name]  # print_to_terminal.getAvgORZero(game._entities[e]._pressure_to_pass)
    #             players[game._entities[e]._last_name]["pos"][0] += x
    #             players[game._entities[e]._last_name]["pos"][1] += y
    #             players[game._entities[e]._last_name]["games"] += 1
                 # = {"team":game._teams[game._entities[e]._team]._full_name,
                 #                                        "code":game._entities[e]._team,
                 #                                        "pos":[x, y]}

    teams = print_to_terminal.rankAnalyticsCustom(games)
    for player_last_name in players.keys():
        for team_lst in teams.values():
            for player in team_lst:
                if player_last_name == player["last_name"]:
                    x = player["turnovers"] / player["games"] #player["PPM"] / player["games"]
                    y = print_to_terminal.getAvgORZero(player["PMv"]) #print_to_terminal.getAvgORZero(player["PA"]) #print_to_terminal.getAvgORZero(player["PMv"]) # salaries[player_last_name]
                    players[player_last_name]["pos"][0] = x
                    players[player_last_name]["pos"][1] = y
                    players[player_last_name]["games"] = player["games"]
                    players[player_last_name]["position"] = player["position"]



    fig=plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    xxs, yys = [], []
    for p in players.keys():
        xxs.append(players[p]["pos"][0])
        yys.append(players[p]["pos"][1])

    plt.axhline(np.median(np.array(yys)), c='r', label='PrMOV Median')
    plt.axvline(np.median(np.array(xxs)), c='c', label='TO Median')


    xs, ys = [], []
    for p in players.keys():

        # print(p, ": ", players[p]["position"])

        ''' by team '''
        if players[p]["code"] == '14':
            color = 'b'
        else:
            color = 'g'

        ''' by position '''
        if players[p]["position"] == 'D':
            # color = 'tab:orange'
            marker_sty = 's'
        elif players[p]["position"] == 'G' or players[p]["position"] == 'C' or players[p]["position"] == 'RW' or players[p]["position"] == 'LW':
            # color = 'tab:purple'
            marker_sty = 'o'
        else:
            # color = 'tab:cyan'
            # marker_sty = '*'
            continue


        plt.scatter(players[p]["pos"][0], players[p]["pos"][1], c=color, s=50, edgecolors='k', marker=marker_sty, label=players[p]["team"])
        # plt.annotate(p,((players[p]["pos"][0])-0.05, (players[p]["pos"][1])+0.02), fontsize=15)
        if p == 'Klingberg' or p == 'Dickinson' or p == 'Coleman' or p == 'Verhaeghe' or p == 'Benn':
            plt.scatter(players[p]["pos"][0], players[p]["pos"][1], c=color, s=150, edgecolors='k', marker=marker_sty, label=players[p]["team"])
            plt.annotate(p,((players[p]["pos"][0])-0.2, (players[p]["pos"][1])+0.01), fontsize=15)
        elif p == 'Volkov' or p == 'Cirelli' or p == 'Shattenkirk': #below player
            plt.scatter(players[p]["pos"][0], players[p]["pos"][1], c=color, s=150, edgecolors='k', marker=marker_sty, label=players[p]["team"])
            plt.annotate(p,((players[p]["pos"][0])-0.2, (players[p]["pos"][1])-0.03), fontsize=15)

        xs.append(players[p]["pos"][0])
        ys.append(players[p]["pos"][1])

        # if p == 'Palat' or p == 'Hanley' or p == 'McDonagh' or p == 'Benn' or p == 'Perry' or p == 'Pavelski':
        #     plt.scatter(players[p]["pos"][0], players[p]["pos"][1], c=color, s=150, edgecolors='k', marker=marker_sty, label=players[p]["team"])
        #     plt.annotate(p,((players[p]["pos"][0])-0.02, (players[p]["pos"][1])+250000), fontsize=15)
        # if p == 'Hedman' or p == 'Lindell': #above player
        #     plt.scatter(players[p]["pos"][0]/players[p]["games"], players[p]["pos"][1]/players[p]["games"], c=color, s=150, label=players[p]["team"])
        #     plt.annotate(p,((players[p]["pos"][0]/players[p]["games"])-0.05, (players[p]["pos"][1]/players[p]["games"])+3), fontsize=15)
        # if p == 'Hanley': #below player
        #     plt.scatter(players[p]["pos"][0]/players[p]["games"], players[p]["pos"][1]/players[p]["games"], c=color, s=150, label=players[p]["team"])
        #     plt.annotate(p,((players[p]["pos"][0]/players[p]["games"])-5, (players[p]["pos"][1]/players[p]["games"])-700000), fontsize=15)

    # xs = np.array(xs)
    # ys = np.array(ys)
    # m, b = np.polyfit(xs, ys, 1)
    # plt.plot(xs, m*xs + b, c='r')
    # plt.show()


    # plt.text(np.amin(np.array(xs)), np.amax(np.array(ys)), "This is my text", fontsize=15)
    plt.xlabel("TO", fontsize=25)
    plt.ylabel("PrMOV", fontsize=25)
    # plt.xticks([0,1], ["Tampa Bay", "Dallas"],fontsize=20)
    # plt.xticks([0,1, 2, 3, 4], ["0", "1", "2", "3", "4"],fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.xlim(left=-43, right=43)
    # plt.ylim(top=85)
    plt.ylim(bottom=0.22,top=0.65)
    # plt.ylim(bottom=0.55)



    custom_lines = [ Line2D([0], [0], color='r', lw=4, label='PAv Mean'),
                Line2D([0], [0], color='c', lw=4, label='TO Mean'),
                Line2D([0], [0], marker='s', color='k', label='Def', markerfacecolor='w', markersize=15),
                Line2D([0], [0], marker='o', color='k', label='Fwd', markerfacecolor='w', markersize=15)]
    plt.legend(custom_lines, ['PrMOV Median', 'TO Median', 'Def', 'Fwd'], fontsize=13,ncol=2, framealpha=0.8, loc='lower right')


    # plt.show()
    fname = "../../Paper/paper_imgs/scatterplots/PrMOV_TO.png"
    plt.savefig(fname,bbox_inches='tight', dpi=300)
    plt.close()
    exit()


def makePAvPlots(games):
    dones = []
    for g in games:
        for p in games[0]._entities.keys():
            if p not in dones and g._entities[p].total_passes > 0:
                print("Making chart for: ", g._entities[p]._last_name)
                g._entities[p].generatePassHeatmap(games)
            dones.append(p)
    exit()

def plotHist(player, data, title_str, raw_str):
    fig=plt.figure(figsize=(10, 6))
    plt.hist(data, color=player._color, bins='auto')
    plt.axvline(np.mean(data), c='r', label='Mean')
    plt.axvline(np.mean(data)+np.std(data), c='k', linestyle='--', label='STD')
    if np.mean(data)-np.std(data) > 0:
        plt.axvline(np.mean(data)-np.std(data), c='k', linestyle='--')
    plt.title(player._first_name+" "+player._last_name+" Histogram: "+title_str, fontsize=20)
    plt.xlabel(title_str+" Value", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20)
    plt.xlim(left=0)
    fname = "../../Paper/paper_imgs/histograms/"+player._last_name+"_"+raw_str+".png"
    plt.savefig(fname,bbox_inches='tight', dpi=300)
    plt.close()


def PAvHist(games):
    for game_num, g in enumerate(games):
        for p in g._entities.keys():
            plotHist(g._entities[p], g._entities[p]._openness, r'$\Omega$', "Omega")
            plotHist(g._entities[p], g._entities[p]._pass_risk, "PAv", "PAv")
    exit()


def passPressureLaneScatter(games):

    for game_num, game in enumerate(games):
        for p in game._passes.keys():
            key = np.argmin(np.abs(np.array(list(game._scoreboard.keys())) - game._passes[p]._UTC_update))
            key = list(game._scoreboard.keys())[key]
            # home_strength = game._scoreboard[key]["HomeStrength"]
            # visitor_strength = game._scoreboard[key]["VisitorStrength"]
            opponents = game.getOpponentsOnIce(game._passes[p]._passer._team, game._passes[p]._UTC_update)
            # print(game._passes[p]._pass_risk)
            teammates = game.getTeammatesOnIce(game._passes[p]._passer._team, key)

            for mate in teammates:
                pressure, beta = event.getAnyPassStats(game, game._entities[mate['entity_id']], game._passes[p]._period, key)
                if game._passes[p]._receiver._id == mate['entity_id']:
                    c = 'r'
                else:
                    c = 'k'
                # plt.scatter(pressure, beta, color = c)
                if beta > 10:
                    game._passes[p].plotPass(game, opponents, teammates, opponents[0])
            #     print(pressure)
            #     print(beta)
            #     exit()
            #
            # print(game._passes[p]._opponents)
            # for o in game._passes[p]._opponents:
            #     opp_obj = game._entities[o["entity_id"]]
            #     if opp_obj._pos == 'G': continue
            #     time_key = getUTCtimekey(opp_obj, p._UTC_update)
            #     opp = np.array([opp_obj._hd_UTC_update[time_key]["X"], opp_obj._hd_UTC_update[time_key]["Y"]])
        plt.xlabel("pressure")
        plt.ylabel("beta")
        plt.show()
        exit()


def plotPassChart(games):
    ''' This function makes a passing chart for a player

    to run: 1. Change the player's name in 'if' statement below
            2. Make sure the direction in correct on line 'if attacking_net['X'] < 0:' Dallas = <, Tampa = >
     '''
    player_arr = []
    for game in games:
        for e in game._entities.keys():
            if game._entities[e]._last_name not in player_arr and game._entities[e]._team in list(game._teams.keys()):
                player_arr.append(game._entities[e]._last_name)

    for player_name in player_arr:
        # player_name = "Cernak"
        cmap = plt.cm.coolwarm
        for game_num, game in enumerate(games):
            fig, ax = games[0].getRinkPlot()
            for p in game._passes.keys():
                # check strength
                key = np.argmin(np.abs(np.array(list(game._scoreboard.keys())) - game._passes[p]._UTC_update))
                key = list(game._scoreboard.keys())[key]
                home_strength = game._scoreboard[key]["HomeStrength"]
                visitor_strength = game._scoreboard[key]["VisitorStrength"]

                if game._passes[p]._passer._last_name == player_name and home_strength == 6 and visitor_strength == 6:
                    # getAttackingNet(self, my_team, period)
                    attacking_net,_ = game.getAttackingNet(game._passes[p]._passer._team, game._passes[p]._period)

                    if (attacking_net['X'] > 0 and game._passes[p]._passer._team == '14') or (attacking_net['X'] < 0 and game._passes[p]._passer._team == '25'):
                        game._passes[p]._origin['X'] = game._passes[p]._origin['X'] * -1 # flip sign
                        game._passes[p]._origin['Y'] = game._passes[p]._origin['Y'] * -1 # flip sign
                        game._passes[p]._destination['X'] = game._passes[p]._destination['X'] * -1 # flip sign
                        game._passes[p]._destination['Y'] = game._passes[p]._destination['Y'] * -1 # flip sign

                    overtook = len(game._passes[p]._overtook_ids)
                    if overtook == 0:   color_var = cmap(0.0)
                    elif overtook == 1: color_var = cmap(0.2)
                    elif overtook == 2: color_var = cmap(0.4)
                    elif overtook == 3: color_var = cmap(0.6)
                    elif overtook == 4: color_var = cmap(0.8)
                    elif overtook == 5: color_var = cmap(1.0)

                    plt.arrow(game._passes[p]._origin["X"], game._passes[p]._origin["Y"], (game._passes[p]._destination["X"]-game._passes[p]._origin["X"]), (game._passes[p]._destination["Y"]-game._passes[p]._origin["Y"]), head_width=3, color=color_var)

                    # plt.plot([game._passes[p]._origin["X"], game._passes[p]._destination["X"]], [game._passes[p]._origin["Y"], game._passes[p]._destination["Y"]], c=color_var)
                    # break
            # ax.set_xlim(right=110)

            for e in game._entities.keys():
                if game._entities[e]._last_name == player_name:
                    player_key = e
                    pos_string = "Def" if game._entities[e]._pos == "D" else "Fwd"
                    title_string = game._teams[game._entities[e]._team]._full_name + ": " + game._entities[e]._first_name + " " + game._entities[e]._last_name + " ("+pos_string+") "
                    break

            try:
                ax.text(-101,-42,"$PPM_{p}$ = "+str(len(game._entities[player_key]._overtook) - len(game._entities[player_key]._beaten)), fontsize=18, bbox={'facecolor':'y', 'alpha':1, 'pad':2})
            except:
                pass
            plt.title(title_string + "Pass Chart Game "+str(game_num+5), fontsize=20)
            custom_lines = [Line2D([0], [0], color=cmap(0.0), lw=4),
                        Line2D([0], [0], color=cmap(0.2), lw=4),
                        Line2D([0], [0], color=cmap(0.4), lw=4),
                        Line2D([0], [0], color=cmap(0.6), lw=4),
                        Line2D([0], [0], color=cmap(0.8), lw=4),
                        Line2D([0], [0], color=cmap(1.0), lw=4),
                        Line2D([0], [0], color='b', lw=8),
                        Line2D([0], [0], color='g', lw=8)]
            # ax.legend(custom_lines, ['+0', '+1', '+2', '+3', '+4', '+5', 'Tampa Bay Goal', 'Dallas Goal'], fontsize=13,ncol=8, framealpha=1)
            # ax.set_xlim(right=120)
            # plt.show()
            fname = "../../Paper/paper_imgs/passcharts/game"+str(game_num+5)+"/"+player_name+"_passchart_noleg"
            plt.savefig(fname,bbox_inches='tight', dpi=300)
            plt.close()
        # exit()

def plotPressureChart(games):
    ''' This function makes a pressure chart for a player

    to run: 1. Change the player's name in 'if' statement below
            2. Make sure the direction in correct on line 'if attacking_net['X'] < 0:' Dallas = <, Tampa = >
     '''
    player_arr = []
    for game in games:
        for e in game._entities.keys():
            if game._entities[e]._last_name not in player_arr and game._entities[e]._team in list(game._teams.keys()):
                player_arr.append(game._entities[e]._last_name)

    for player_name in player_arr:
        player_name = 'Shattenkirk'
        cmap = plt.cm.coolwarm
        # pos_count = 0
        for game_num, game in enumerate(games):
            for possession_count, p in enumerate(game._posessions.keys()):
                # creates graph for player possessions and betas
                if game._posessions[p]._posessor._last_name == player_name:
                    for possession_timestep, t in enumerate(game._posessions[p]._posession_UTC_keys):

                        game._posessions[p].plotPosessionTimestep(t, game_num, possession_count, possession_timestep)
                        exit()
                else:
                    continue
                if game._posessions[p]._posessor._last_name == player_name:
                    # pos_count+=1

                    for t in game._posessions[p]._posession_UTC_keys:
                        game._posessions[p].plotPosessionTimestep(t)
                    exit()

                    # get direction
                    attacking_net,_ = game.getAttackingNet(game._posessions[p]._posessor._team, game._posessions[p]._period)
                    # if attacking_net['X'] > 0:
                    if (attacking_net['X'] > 0 and game._posessions[p]._posessor._team == '14') or (attacking_net['X'] < 0 and game._posessions[p]._posessor._team == '25'):
                        c = -1
                    else:
                        c = 1

                    if game._posessions[p]._posessor._team == '14': dot_color = 'b'
                    else: dot_color = 'g'

                    for i, key in enumerate(game._posessions[p]._posession_UTC_keys):
                        game._posessions[p].getPressure(time=key)
                        if game._posessions[p]._pressure > 1: game._posessions[p]._pressure = 1
                        elif game._posessions[p]._pressure == 0.0: game._posessions[p]._pressure = 0.05

                        if i == 0:
                            plt.scatter(game._posessions[p]._posessor._hd_UTC_update[key]["X"]*c, game._posessions[p]._posessor._hd_UTC_update[key]["Y"]*c, s=50, facecolor='None', edgecolor='k')
                            # plt.scatter(game._posessions[p]._posessor._hd_UTC_update[key]["X"]*c, game._posessions[p]._posessor._hd_UTC_update[key]["Y"]*c, s=80, facecolor='None', edgecolor='k')
                        # if i == len(game._posessions[p]._posession_UTC_keys)-1: plt.scatter(game._posessions[p]._posessor._hd_UTC_update[key]["X"]*c, game._posessions[p]._posessor._hd_UTC_update[key]["Y"]*c, s=50, c=dot_color, marker='D', alpha=(game._posessions[p]._pressure))
                        plt.scatter(game._posessions[p]._posessor._hd_UTC_update[key]["X"]*c, game._posessions[p]._posessor._hd_UTC_update[key]["Y"]*c, s=50, c=dot_color, alpha=(game._posessions[p]._pressure))
                        # plt.scatter(game._posessions[p]._posessor._hd_UTC_update[key]["X"]*c, game._posessions[p]._posessor._hd_UTC_update[key]["Y"]*c, facecolor='None', edgecolor='k')
                    # if pos_count == 6:

            for e in game._entities.keys():
                if game._entities[e]._last_name == player_name:
                    pos_string = "Def" if game._entities[e]._pos == "D" else "Fwd"
                    title_string = game._teams[game._entities[e]._team]._full_name + ": " + game._entities[e]._first_name + " " + game._entities[e]._last_name + " ("+pos_string+") "
                    break
            # plt.title(title_string+"Pressure Experienced Game "+str(game_num+5), fontsize=20)
            # fname = "../../Paper/paper_imgs/pressurecharts/game"+str(game_num+5)+"/"+player_name+"_pressurechart_noleg"
            # plt.savefig(fname,bbox_inches='tight', dpi=300)
            # plt.close()

def examplePassPlot(game):
    fig, ax = game.getRinkPlot()


    # turnover

    x = 35
    y = -18
    ax.scatter(x, y, c='g', s=50)
    ax.annotate("$p$",(x, y-7), fontsize=25)
    ax.annotate("$TO_{p} = 1$",(x-5, y-15), fontsize=16)

    plt.arrow(36, -17, 15, 13, head_width=3, color='k')
    ax.annotate("Pass",(38,-12), fontsize=16, rotation=41)


    x = 71
    y = 15
    ax.scatter(x, y, c='g', s=50)
    ax.annotate("$r$",(x, y-7), fontsize=25)

    x = 55
    y = 0
    ax.scatter(x, y, c='b', s=50)
    ax.annotate("$o$",(x, y-7), fontsize=25)
    # ax.annotate("$TB_{o} = 1$",(x-5, y-15), fontsize=16)

    # pass

    x = -35
    y = -18
    ax.scatter(x, y, c='b', s=50)
    ax.annotate("$p$",(x-7, y-7), fontsize=25)
    ax.annotate("$PO_{p} = 1$",(x-15, y-15), fontsize=16)

    plt.arrow(-36, -17, -30, 27, head_width=3, color='k')
    ax.annotate("Pass",(-56,-6), fontsize=16, rotation=-41)


    x = -71
    y = 15
    ax.scatter(x, y, c='b', s=50)
    ax.annotate("$r$",(x-7, y-7), fontsize=25)

    x = -55
    y = -10
    ax.scatter(x, y, c='g', s=50)
    ax.annotate("$o$",(x-7, y-7), fontsize=25)
    ax.annotate("$TB_{o} = 1$",(x-15, y-15), fontsize=16)



    plt.show()
    exit()
