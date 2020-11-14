import numpy as np
import pandas as pd


def rankPosessionData(game):
    ''' This function presents the posession data for each player '''

    col_lst = ['ID','First','Last','Posessions','Seconds','Avg_Time','Steals', 'Turnovers']
    home_results = pd.DataFrame(columns=col_lst)
    away_results = pd.DataFrame(columns=col_lst)

    for count, e in enumerate(game._entities.keys()):
        # ----- dont include goalies in posession
        if game._entities[e]._pos == 'G':
            continue

        try:
            avg_poss = (game._entities[e]._posession_time/len(game._entities[e]._posession_lst))
        except:
            avg_poss = 0
        if game._entities[e]._team == game._home_team_num:
            home_results.loc[count] = [game._entities[e]._id, game._entities[e]._first_name, game._entities[e]._last_name, len(game._entities[e]._posession_lst), game._entities[e]._posession_time, avg_poss,
                                    game._entities[e]._steals, game._entities[e]._turnovers]
        elif game._entities[e]._team == game._visitor_team_num:
            away_results.loc[count] = [game._entities[e]._id, game._entities[e]._first_name, game._entities[e]._last_name, len(game._entities[e]._posession_lst), game._entities[e]._posession_time, avg_poss,
                                    game._entities[e]._steals, game._entities[e]._turnovers]


    print("HOME - ", game._teams[game._home_team_num])
    print(home_results.sort_values('Posessions', ascending=False))
    tot_team_posessions_home = np.sum(home_results["Posessions"])
    tot_seconds_home = np.sum(home_results["Seconds"])
    tot_avg_time_home = np.sum(home_results["Avg_Time"]) / len(home_results["Avg_Time"])
    tot_turnovers_home = np.sum(home_results["Turnovers"])
    print(game._teams[game._home_team_num]._full_name, " - Posessions: ", tot_team_posessions_home, " Seconds: ", tot_seconds_home,
                    " Avg Duration: ", tot_avg_time_home, " Turnovers: ", tot_turnovers_home)
    print("*********************************************************")
    print("AWAY - ", game._teams[game._visitor_team_num])
    print(away_results.sort_values('Posessions', ascending=False))
    tot_team_posessions_visitor = np.sum(away_results["Posessions"])
    tot_seconds_visitor = np.sum(away_results["Seconds"])
    tot_avg_time_visitor = np.sum(away_results["Avg_Time"]) / len(away_results["Avg_Time"])
    tot_turnovers_visitor = np.sum(away_results["Turnovers"])
    print(game._teams[game._visitor_team_num]._full_name, " - Posessions: ", tot_team_posessions_visitor, " Seconds: ", tot_seconds_visitor,
                    " Avg Duration: ", tot_avg_time_visitor, " Turnovers: ", tot_turnovers_visitor)
    print("*********************************************************")
    print("COMPARE: home - away")
    posession_diff = tot_team_posessions_home - tot_team_posessions_visitor
    seconds_diff = tot_seconds_home - tot_seconds_visitor
    avg_time_diff = tot_avg_time_home - tot_avg_time_visitor
    turnovers_diff = tot_turnovers_home - tot_turnovers_visitor
    print("Posessions: ", posession_diff, " Seconds: ", seconds_diff, " Avg Duration: ", avg_time_diff, " Turnovers: ", turnovers_diff)
    exit()


def rankPassingData(game):
    ''' This function presents the data for each player '''

    col_lst = ['ID','First','Last','Tot_Pass','Overtook','Beat','PPM']
    home_results = pd.DataFrame(columns=col_lst)
    away_results = pd.DataFrame(columns=col_lst)

    for count, e in enumerate(game._entities.keys()):
        if game._entities[e]._team == game._home_team_num:
            home_results.loc[count] = [game._entities[e]._id, game._entities[e]._first_name, game._entities[e]._last_name, game._entities[e].total_passes, len(game._entities[e]._overtook), len(game._entities[e]._beaten), (len(game._entities[e]._overtook) - len(game._entities[e]._beaten))]
        elif game._entities[e]._team == game._visitor_team_num:
            away_results.loc[count] = [game._entities[e]._id, game._entities[e]._first_name, game._entities[e]._last_name, game._entities[e].total_passes, len(game._entities[e]._overtook), len(game._entities[e]._beaten), (len(game._entities[e]._overtook) - len(game._entities[e]._beaten))]

    print("HOME - ", game._teams[game._home_team_num])
    print(home_results.sort_values('PPM', ascending=False))
    tot_team_pass_home = np.sum(home_results["Tot_Pass"])
    tot_overtaken_home = np.sum(home_results["Overtook"])
    tot_beat_home = np.sum(home_results["Beat"])
    tot_ppm_home = np.sum(home_results["PPM"])
    print(game._teams[game._home_team_num]._full_name, " - Passes made: ", tot_team_pass_home, " Overtook: ", tot_overtaken_home, " Beaten: ", tot_beat_home, " Team PPM: ", tot_ppm_home)
    print("*********************************************************")
    print("AWAY - ", game._teams[game._visitor_team_num])
    print(away_results.sort_values('PPM', ascending=False))
    tot_team_pass_visitor = np.sum(away_results["Tot_Pass"])
    tot_overtaken_visitor = np.sum(away_results["Overtook"])
    tot_beat_visitor = np.sum(away_results["Beat"])
    tot_ppm_visitor = np.sum(away_results["PPM"])
    print(game._teams[game._visitor_team_num]._full_name, " - Passes made: ", tot_team_pass_visitor, " Overtook: ", tot_overtaken_visitor, " Beaten: ", tot_beat_visitor, " Team PPM: ", tot_ppm_visitor)
    print("*********************************************************")
    print("COMPARE: home - away")
    pass_diff = tot_team_pass_home - tot_team_pass_visitor
    overtake_diff = tot_overtaken_home - tot_overtaken_visitor
    beat_diff = tot_beat_home - tot_beat_visitor
    ppm_diff = tot_ppm_home - tot_ppm_visitor
    print("Passes made: ", pass_diff, " Overtook: ", overtake_diff, " Beaten: ", beat_diff, " Team PPM: ", ppm_diff)
    exit()
