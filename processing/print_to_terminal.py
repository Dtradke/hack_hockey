import numpy as np
import pandas as pd


def rankPosessionData(game):
    ''' This function presents the posession data for each player '''

    col_lst = ['ID','First','Last','Posessions','Avg_Time','Steals', 'Turnovers', 'Avg_PassPressure', 'Shots', 'Avg_ShotPressure']
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
            home_results.loc[count] = [game._entities[e]._id, game._entities[e]._first_name, game._entities[e]._last_name, len(game._entities[e]._posession_lst), avg_poss,
                                    game._entities[e]._steals, game._entities[e]._turnovers, round((np.sum(np.array(game._entities[e]._pressure_to_pass)) / len(game._entities[e]._pressure_to_pass)),1),
                                    len(game._entities[e]._pressure_to_shoot), round((np.sum(np.array(game._entities[e]._pressure_to_shoot)) / len(game._entities[e]._pressure_to_shoot)),1)]
        elif game._entities[e]._team == game._visitor_team_num:
            away_results.loc[count] = [game._entities[e]._id, game._entities[e]._first_name, game._entities[e]._last_name, len(game._entities[e]._posession_lst), avg_poss,
                                    game._entities[e]._steals, game._entities[e]._turnovers, round((np.sum(np.array(game._entities[e]._pressure_to_pass)) / len(game._entities[e]._pressure_to_pass)),1),
                                    len(game._entities[e]._pressure_to_shoot), round((np.sum(np.array(game._entities[e]._pressure_to_shoot)) / len(game._entities[e]._pressure_to_shoot)),1)]


    print("HOME - ", game._teams[game._home_team_num])
    print(home_results.sort_values('Posessions', ascending=False))
    tot_team_posessions_home = np.sum(home_results["Posessions"])
    # tot_seconds_home = np.sum(home_results["Seconds"])
    tot_avg_time_home = np.sum(home_results["Avg_Time"]) / len(home_results["Avg_Time"])
    tot_turnovers_home = np.sum(home_results["Turnovers"])
    avg_pressurepass_home = np.sum(home_results["Avg_PassPressure"]) / len(home_results["Avg_PassPressure"])
    tot_shots_home = np.sum(home_results["Shots"])
    avg_shot_pressure_home = np.sum(home_results["Avg_ShotPressure"]) / len(home_results["Avg_ShotPressure"])
    print(game._teams[game._home_team_num]._full_name, " - Posessions: ", tot_team_posessions_home,
                    " Avg Duration: ", tot_avg_time_home, " Turnovers: ", tot_turnovers_home, " PassPressure: ", avg_pressurepass_home,
                    " Shots: ", tot_shots_home , " ShotPressure: ", avg_shot_pressure_home)
    print("*********************************************************")
    print("AWAY - ", game._teams[game._visitor_team_num])
    print(away_results.sort_values('Posessions', ascending=False))
    tot_team_posessions_visitor = np.sum(away_results["Posessions"])
    # tot_seconds_visitor = np.sum(away_results["Seconds"])
    tot_avg_time_visitor = np.sum(away_results["Avg_Time"]) / len(away_results["Avg_Time"])
    tot_turnovers_visitor = np.sum(away_results["Turnovers"])
    avg_pressurepass_visitor = np.sum(away_results["Avg_PassPressure"]) / len(away_results["Avg_PassPressure"])
    tot_shots_visitor = np.sum(away_results["Shots"])
    avg_shot_pressure_visitor = np.sum(away_results["Avg_ShotPressure"]) / len(away_results["Avg_ShotPressure"])
    print(game._teams[game._visitor_team_num]._full_name, " - Posessions: ", tot_team_posessions_visitor,
                    " Avg Duration: ", tot_avg_time_visitor, " Turnovers: ", tot_turnovers_visitor, " PassPressure: ", avg_pressurepass_visitor,
                    " Shots: ", tot_shots_visitor , " ShotPressure: ", avg_shot_pressure_visitor)
    print("*********************************************************")
    print("COMPARE: home - away")
    posession_diff = tot_team_posessions_home - tot_team_posessions_visitor
    # seconds_diff = tot_seconds_home - tot_seconds_visitor
    avg_time_diff = tot_avg_time_home - tot_avg_time_visitor
    turnovers_diff = tot_turnovers_home - tot_turnovers_visitor
    pressure_diff = avg_pressurepass_home - avg_pressurepass_visitor
    shot_diff = tot_shots_home - tot_shots_visitor
    shot_pressure_diff = avg_shot_pressure_home - avg_shot_pressure_visitor
    print("Posessions: ", posession_diff, " Avg Duration: ", avg_time_diff, " Turnovers: ", turnovers_diff, " Pressure: ", pressure_diff, " Shots: ", shot_diff, " ShotPressure: ", shot_pressure_diff)
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
