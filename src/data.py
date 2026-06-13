from pathlib import Path
import pandas as pd

def load_results():
    path = Path(__file__).parent.parent / "data" / "raw" / "results.csv"
    if not path.exists():
        df = pd.read_csv('https://github.com/martj42/international_results/raw/refs/heads/master/results.csv')
        df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral']]
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df

def process_results(df):
    df = df.dropna(subset=['home_score', 'away_score'])
    df['date'] = pd.to_datetime(df['date']) 
    return df