import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

DATA = Path("/Users/yohands/downloads/medium_code")
FIGS = Path("/Users/yohands/downloads/medium_code")
FIGS.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6.75)   
plt.rcParams["font.size"] = 12

league = pd.read_csv(DATA / "nba_3pt_shooting_2010_2024.csv")
teams = pd.read_csv(DATA / "nba_3pt_revolution_teams.csv")

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=300, bbox_inches="tight") 

# 1) League trend: 3PA per game over time
plt.figure()
sns.lineplot(data=league, x="Year", y="3PA_per_game", marker="o", linewidth=3)
plt.title("NBA 3-Point Attempts Per Game (2010–2024)")
plt.xlabel("Season (ending year)")
plt.ylabel("3-Point Attempts Per Game")
plt.grid(alpha=0.3)
savefig("league_trend.png")
plt.close()

# 2) Timeline vs league: Warriors, Rockets, Celtics vs league average
timeline = league[["Season", "3PA_per_game"]].rename(columns={"3PA_per_game": "League_Avg"})
timeline["Season"] = timeline["Season"]
w = teams[teams["Team"] == "Warriors"][["Season", "3PA_per_game"]].rename(columns={"3PA_per_game":"Warriors"})
r = teams[teams["Team"] == "Rockets"][["Season", "3PA_per_game"]].rename(columns={"3PA_per_game":"Rockets"})
c = teams[teams["Team"] == "Celtics"][["Season", "3PA_per_game"]].rename(columns={"3PA_per_game":"Celtics"})

# Merge by season
tl = timeline.merge(w, on="Season", how="left").merge(r, on="Season", how="left").merge(c, on="Season", how="left")
tl_long = tl.melt(id_vars=["Season"], value_vars=["League_Avg","Warriors","Rockets","Celtics"],
                  var_name="Team", value_name="3PA_per_game")

plt.figure()
sns.lineplot(data=tl_long, x="Season", y="3PA_per_game", hue="Team", linewidth=2.5)
plt.title("3-Point Attempts Timeline: Warriors, Rockets, Celtics vs League Avg")
plt.xlabel("Season")
plt.ylabel("3-Point Attempts Per Game")
plt.xticks(rotation=45)
plt.legend(title="")
savefig("timeline_vs_league.png")
plt.close()

# 3) Team comparison: average vs peak 3PA
summary = []
for team in ["Warriors","Rockets","Celtics"]:
    t = teams[teams["Team"] == team]
    summary.append({"Team": team, "Metric": "Average", "3PA": t["3PA_per_game"].mean()})
    summary.append({"Team": team, "Metric": "Peak", "3PA": t["3PA_per_game"].max()})
comp = pd.DataFrame(summary)

plt.figure()
sns.barplot(data=comp, x="Team", y="3PA", hue="Metric")
plt.title("3-Point Attempts: Team Averages vs Peaks")
plt.xlabel("")
plt.ylabel("3-Point Attempts Per Game")
for c in plt.gca().containers:
    plt.bar_label(c, fmt="%.1f", label_type="edge", padding=2)
savefig("teams_comparison.png")
plt.close()

# 4) Volume vs efficiency: league-wide scatter with trendline
plt.figure()

w_seasons = set(teams.loc[teams["Team"]=="Warriors", "Season"])
r_seasons = set(teams.loc[teams["Team"]=="Rockets", "Season"])
c_seasons = set(teams.loc[teams["Team"]=="Celtics", "Season"])

# Map each league row (by Season) to a label
def team_label(season):
    if season in r_seasons:
        return "Rockets"
    if season in w_seasons:
        return "Warriors"
    if season in c_seasons:
        return "Celtics"
    return "Other league seasons"

plot_df = league.copy()
plot_df["TeamLabel"] = plot_df["Season"].apply(team_label)
palette = {
    "Rockets": "#d62728",  
    "Warriors": "#1f77b4",  
    "Celtics": "#2ca02c",  
    "Other league seasons": "#bfbfbf"  
}

sns.scatterplot(
    data=plot_df,
    x="3PA_per_game",
    y="3P_percentage",
    hue="TeamLabel",
    palette=palette,
    s=70,
    edgecolor="white",
    linewidth=0.6,
    alpha=0.85
)

# Global regression (whole league, not just one team)
sns.regplot(
    data=league,
    x="3PA_per_game",
    y="3P_percentage",
    scatter=False,
    line_kws={"color": "red", "linewidth": 2}
)

plt.title("3-Point Volume vs Efficiency (2010–2024)")
plt.xlabel("3-Point Attempts Per Game")
plt.ylabel("3-Point Percentage")
plt.legend(title="", frameon=True, edgecolor="#e0e0e0")
savefig("volume_vs_efficiency_colored.png")
plt.close()
