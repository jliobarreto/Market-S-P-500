from __future__ import annotations
from pathlib import Path
import pandas as pd


def export_weekly_rank(df_rank: pd.DataFrame, path: str = "storage/history/weekly_rank.csv") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_rank.to_csv(path, index=False)
    return path
