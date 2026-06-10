import pandas as pd
import numpy as np

df = pd.read_csv("results.csv")

df["date"] = pd.to_datetime(df["date"])

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

df["result"] = df.apply(get_result, axis=1)

df["goal_difference"] = df["home_score"] - df["away_score"]
df["total_goals"] = df["home_score"] + df["away_score"]

matches = df[["date", "home_team", "away_team", "home_score", "away_score", "result", "goal_difference", "total_goals", "tournament", "country", "neutral"]]

home_stats = df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].copy()
home_stats = home_stats.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "goals_for", "away_score": "goals_against"})
home_stats["is_home"] = 1

away_stats = df[["date", "away_team", "home_team", "away_score", "home_score", "tournament", "neutral"]].copy()
away_stats = away_stats.rename(columns={"away_team": "team", "home_team": "opponent", "away_score": "goals_for", "home_score": "goals_against"})
away_stats["is_home"] = 0

team_matches = pd.concat([home_stats, away_stats], ignore_index=True)

def get_team_result(row):
    if row["goals_for"] > row["goals_against"]:
        return "win"
    elif row["goals_for"] < row["goals_against"]:
        return "loss"
    else:
        return "draw"

team_matches["team_result"] = team_matches.apply(get_team_result, axis=1)

def get_points(result):
    if result == "win":
        return 3
    elif result == "draw":
        return 1
    else:
        return 0
    
team_matches["points"] = team_matches["team_result"].apply(get_points)

team_summary = team_matches.groupby("team").agg(
    matches_played = ("team", "count"),
    avg_goals_for = ("goals_for", "mean"),
    avg_goals_against = ("goals_against", "mean"),
    avg_points = ("points", "mean")
).reset_index()

def predict_match(team_a, team_b):
    team_a_data = team_summary[team_summary["team"] == team_a]
    team_b_data = team_summary[team_summary["team"] == team_b]
    if team_a_data.empty:
        print(f"No data for team: {team_a}")
        
    if team_b_data.empty:
        print(f"No data for team: {team_b}")

    team_a_points = team_a_data["avg_points"].values[0]
    team_b_points = team_b_data["avg_points"].values[0]

    team_a_goals_for = team_a_data["avg_goals_for"].values[0]
    team_a_goals_against = team_a_data["avg_goals_against"].values[0]

    team_b_goals_for = team_b_data["avg_goals_for"].values[0]
    team_b_goals_against = team_b_data["avg_goals_against"].values[0]

    expected_goals_a = (team_a_goals_for + team_b_goals_against) / 2
    expected_goals_b = (team_b_goals_for + team_a_goals_against) / 2

    predicted_score_a = round(expected_goals_a)
    predicted_score_b = round(expected_goals_b)

    difference = team_a_points - team_b_points

    if difference > 0.2:
        predicted_result = team_a
    elif difference < -0.2:
        predicted_result = team_b
    else:
        predicted_result = "draw"

    confidence = min(90, 50 + abs(difference) * 20)

    reason = f"{team_a} has an average of {team_a_points:.2f} points per match, while {team_b} has {team_b_points:.2f} points per match."

    return{
        "Match": f"{team_a} vs {team_b}",
        "Prediction": f"{team_a} {predicted_score_a} - {predicted_score_b} {team_b}",
        "Winner": predicted_result,
        "Confidence": f"{confidence:.1f}%",
        "Reason": reason

    }

prediction = predict_match("Mexico", "South Africa")
print(prediction)