import pandas as pd
import os
from pathlib import Path

def create_sliced_ohlcv_files(input_csv='data\SPY.csv', output_folder='sliced_data'):
    # Load data
    df = pd.read_csv(input_csv, parse_dates=['datetime'])

    # Sort by datetime to ensure proper order
    df.sort_values('datetime', inplace=True)

    # Extract trading dates
    df['date'] = df['datetime'].dt.date
    unique_dates = df['date'].unique()

    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Running index pointer for efficiency
    last_idx = 0
    for current_date in unique_dates:
        # Mask up to and including current date using vectorized filtering
        mask = df['date'] <= current_date
        sliced_df = df.loc[mask].copy()

        # Drop helper column
        sliced_df.drop(columns='date', inplace=True)

        # Output path
        out_path = os.path.join(output_folder, f"{current_date}.csv")

        # Write to CSV
        sliced_df.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")

# Call the function
create_sliced_ohlcv_files()
