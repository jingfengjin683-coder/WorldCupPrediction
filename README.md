# World Cup Predictor

A python project that predicst football match results for the FIFA world cup 2026 using historical international match result, FIFA rankings, team market values, recent performance weighting, opponent strength and host advantages

# Features
The predictor uses:
. Historical international match results
. Goals scored and conceded by each team
. Average points per match
. Recency weighting, giving more importance to recent matches
. FIFA ranking points
. Team market values
. Opponent strength adjustment
. Neutral stadium information
. Home and away performance statistics
. Host advantage for Mexico, United States, and Canada in the 2026 World Cup
. Validation and test evaluation

# How the prediction works

For each team, the program calculates:
. Average adjusted goals scored
. Average adjusted goals conceded
. Average adjusted points per match
. FIFA ranking points
. Team market value
. Home and away performance averages

The model then compares both teams using a weighted formula.
It calculates expected goals using attacking and defensive performance:
expected_goals_a = (team_a_goals_for + team_b_goals_against) / 2
expected_goals_b = (team_b_goals_for + team_a_goals_against) / 2
For non-neutral matches, the model uses home statistics for the home team and away statistics for the away team.
For neutral matches, it uses overall team averages instead.

The final predicted winner is based on:
. Difference in average points
. Difference in market value
. Difference in FIFA points
. Host advantage when applicable