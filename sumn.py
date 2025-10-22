import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

DATA = Path("/Users/yohands/downloads/medium_code")
FIGS = Path("/Users/yohands/downloads/medium_code")
FIGS.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6.75)
plt.rcParams["font.size"] = 12

league = pd.read_csv(DATA / "nba_3pt_shooting_2010_2024.csv")
teams = pd.read_csv(DATA / "nba_3pt_revolution_teams.csv")

#Team yearly 3pt % vs league average 

def season_end_year(s):
    return int(s[:4]) + 1

# Build league % timeline
league_pct = league[["Season", "3P_percentage"]].rename(columns={"3P_percentage": "League_3P%"})
league_pct["YearEnd"] = league_pct["Season"].apply(season_end_year)

# Build team % timeline
teams_pct = teams[["Team", "Season", "3P_percentage"]].copy()
teams_pct["YearEnd"] = teams_pct["Season"].apply(season_end_year)

t_long = (teams_pct
          .merge(league_pct[["Season","YearEnd","League_3P%"]], on=["Season","YearEnd"], how="left")
          .sort_values("YearEnd"))

plt.figure()
palette = {"Rockets":"#d62728", "Warriors":"#1f77b4", "Celtics":"#2ca02c", "League Avg":"#7f7f7f"}

# Plot each team 3pt% trajectory
for team, df_t in t_long.groupby("Team"):
    df_t = df_t.sort_values("YearEnd")
    sns.lineplot(
        data=df_t, x="YearEnd", y="3P_percentage",
        label=team, color=palette[team], marker="o", linewidth=2.5
    )

# League-average 3pt% across the union of seasons 
league_subset = (t_long[["YearEnd","League_3P%"]]
                 .drop_duplicates()
                 .sort_values("YearEnd"))
sns.lineplot(
    data=league_subset, x="YearEnd", y="League_3P%",
    label="League Avg", color=palette["League Avg"], linestyle="--", linewidth=2
)
def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=300, bbox_inches="tight")

plt.title("Team 3-Point Percentage by Season vs League Average (Ascending Timeline)")
plt.xlabel("Season (ending year)")
plt.ylabel("3-Point Percentage")
plt.xticks(sorted(league_subset["YearEnd"]), rotation=0)
plt.legend(title="", frameon=True, edgecolor="#e0e0e0")
savefig("team_3p_pct_yearly_vs_league_ascending.png")
plt.close()



