import pandas as pd
import numpy as np

df = pd.read_csv("results.csv")
market_values = pd.read_csv("team_market_values.csv")
fifa_rankings = pd.read_csv("fifa_rankings_all_teams.csv")
host_teams = ["Mexico", "United States", "Canada"]
df["date"] = pd.to_datetime(df["date"])

train_df = df[df["date"] < "2020-01-01"].copy()
validation_df = df[(df["date"] >= "2020-01-01") & (df["date"] < "2023-01-01")].copy()
test_df = df[df["date"] >= "2023-01-01"].copy()

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

home_stats = train_df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].copy()
home_stats = home_stats.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "goals_for", "away_score": "goals_against"})
home_stats["is_home"] = 1

away_stats = train_df[["date", "away_team", "home_team", "away_score", "home_score", "tournament", "neutral"]].copy()
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
latest_year = team_matches["date"].dt.year.max()
team_matches["year"] = team_matches["date"].dt.year
team_matches["recency_weight"] = 1 / (latest_year - team_matches["year"] + 1)

opponent_strength = fifa_rankings[["team", "fifa_points"]].rename(columns={"team": "opponent", "fifa_points": "opponent_fifa_points"})
team_matches = team_matches.merge(opponent_strength, on="opponent", how="left")
average_fifa_points = fifa_rankings["fifa_points"].mean()
team_matches["opponent_fifa_points"] = team_matches["opponent_fifa_points"].fillna(average_fifa_points)
team_matches["opponent_strength_multiplier"] = (team_matches["opponent_fifa_points"] / average_fifa_points)
team_matches["adjusted_points"] = (team_matches["points"] * team_matches["opponent_strength_multiplier"])

team_matches["adjusted_goals_for"] = (team_matches["goals_for"] * team_matches["opponent_strength_multiplier"])
team_matches["adjusted_goals_against"] = (team_matches["goals_against"] / team_matches["opponent_strength_multiplier"])

def weight_average(values, weights):
    return (values * weights).sum() / weights.sum()
team_summary = team_matches.groupby("team").apply(
    lambda x: pd.Series({
        "matches_played": len(x),
        "avg_goals_for": weight_average(x["adjusted_goals_for"], x["recency_weight"]),
        "avg_goals_against": weight_average(x["adjusted_goals_against"], x["recency_weight"]),
        "avg_points": weight_average(x["adjusted_points"], x["recency_weight"])
    })
).reset_index()

fifa_rankings = fifa_rankings[["team", "fifa_rank", "fifa_points"]]
team_summary = team_summary.merge(market_values, on="team", how="left")
team_summary = team_summary.merge(fifa_rankings, on="team", how="left")

global_average_fifa_points = fifa_rankings["fifa_points"].mean()
global_median_fifa_ranking = fifa_rankings["fifa_rank"].median()

team_summary["fifa_points"] = team_summary["fifa_points"].fillna(global_average_fifa_points)
team_summary["fifa_rank"] = team_summary["fifa_points"].fillna(global_median_fifa_ranking)


def predict_match(team_a, team_b, match_date = None):
    team_a_data = team_summary[team_summary["team"] == team_a]
    team_b_data = team_summary[team_summary["team"] == team_b]

    if team_a_data.empty:
        return {"Error": f"No data for team: {team_a}"}
        
    if team_b_data.empty:
        return {"Error": f"No data for team: {team_b}"}

    team_a_points = team_a_data["avg_points"].values[0]
    team_b_points = team_b_data["avg_points"].values[0]

    team_a_goals_for = team_a_data["avg_goals_for"].values[0]
    team_a_goals_against = team_a_data["avg_goals_against"].values[0]

    team_b_goals_for = team_b_data["avg_goals_for"].values[0]
    team_b_goals_against = team_b_data["avg_goals_against"].values[0]

    market_value_a = team_a_data["market_value_million_eur"].values[0]
    market_value_b = team_b_data["market_value_million_eur"].values[0]

    team_a_fifa = team_a_data["fifa_points"].values[0]
    team_b_fifa = team_b_data["fifa_points"].values[0]

    expected_goals_a = (team_a_goals_for + team_b_goals_against) / 2
    expected_goals_b = (team_b_goals_for + team_a_goals_against) / 2

    points_difference = team_a_points - team_b_points
    if pd.isna(market_value_a) or pd.isna(market_value_b):
        market_value_difference = 0
    else:
        market_value_difference = np.log1p(market_value_a) - np.log1p(market_value_b)
    fifa_difference = (team_a_fifa - team_b_fifa) / 400

    goal_adjustment = (0.2 * fifa_difference + 0.1 * market_value_difference)

    expected_goals_a = expected_goals_a + goal_adjustment
    expected_goals_b = expected_goals_b - goal_adjustment

    expected_goals_a = max(0, expected_goals_a)
    expected_goals_b = max(0, expected_goals_b)

    predicted_score_a = round(expected_goals_a)
    predicted_score_b = round(expected_goals_b)

    if match_date is not None:
        match_date = pd.to_datetime(match_date)
    
    world_cup_start_date = pd.to_datetime("2026-06-11")
    host_difference = 0
    host_advantages = 0.2
    if match_date is not None and match_date >= world_cup_start_date:
        if team_a in host_teams:
            host_difference += host_advantages
        if team_b in host_teams:
            host_difference -= host_advantages

    difference = points_difference + 0.20 * market_value_difference + 0.7 * fifa_difference + host_difference

    if difference > 0.2:
        predicted_result = team_a
        if predicted_score_a <= predicted_score_b:
            predicted_score_a = predicted_score_b + 1
    elif difference < -0.2:
        predicted_result = team_b
        if predicted_score_b <= predicted_score_a:
            predicted_score_b = predicted_score_a + 1
    else:
        predicted_result = "draw"
        if predicted_score_a != predicted_score_b:
            average_score = round((predicted_score_a + predicted_score_b) / 2)
            predicted_score_a = average_score
            predicted_score_b = average_score
    
    confidence = min(100, abs(difference) * 40)

    reason = f"{team_a} has an average of {team_a_points:.2f} points per match, while {team_b} has {team_b_points:.2f} points per match."

    return{
        "Match": f"{team_a} vs {team_b}",
        "Prediction": f"{team_a} {predicted_score_a} - {predicted_score_b} {team_b}",
        "Predicted Home Goals": predicted_score_a,
        "Predicted Away Goals": predicted_score_b,
        "Winner": predicted_result,
        "Confidence": f"{confidence:.1f}%",
        "Reason": reason

    }

def get_actual_winner(row):
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    elif row["home_score"] < row["away_score"]:
        return row["away_team"]
    else:
        return "draw"

validation_results = []

for index, row in validation_df.iterrows():
    prediction = predict_match(row["home_team"], row["away_team"], row["date"])

    if "Error" in prediction:
        continue

    actual_winner = get_actual_winner(row)
    actual_score = f'{row["home_team"]} {row["home_score"]} - {row["away_score"]} {row["away_team"]}'

    validation_results.append({
        "Match": prediction["Match"],
        "Predicted Score": prediction["Prediction"],
        "Predicted Winner": prediction["Winner"],
        "Actual Score": actual_score,
        "Actual Winner": prediction["Winner"] == actual_winner,
        "Correct Exact Score": (prediction["Predicted Home Goals"] == row["home_score"] and prediction["Predicted Away Goals"] == row["away_score"])
    })

validation_results_df = pd.DataFrame(validation_results)

winner_accuracy = validation_results_df["Actual Winner"].mean() * 100
exact_score_accuracy = validation_results_df["Correct Exact Score"].mean() * 100

print(validation_results_df.head(20))
print(f"Validation matches tested: {len(validation_results_df)}")
print(f"Winner accuracy: {winner_accuracy:.2f}%")
print(f"Exact score accuracy: {exact_score_accuracy:.2f}%")