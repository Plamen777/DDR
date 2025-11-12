#!/usr/bin/env python3
"""
CL Futures Candle Data Analyzer
Analyzes 5-minute candle data to find daily high/low extremes and their timing.
Optimized for large datasets using pandas vectorized operations.
"""

import pandas as pd
import numpy as np
from datetime import time
import sys
from pathlib import Path


def load_and_prepare_data(csv_path):
    """
    Load CSV and prepare data with optimized dtypes.
    Expects columns: timestamp/datetime, open, high, low, close
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV with optimized dtypes
    df = pd.read_csv(csv_path)
    
    # Detect timestamp column (common names)
    timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Date', 'DateTime', 'Timestamp']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        # Assume first column is timestamp
        timestamp_col = df.columns[0]
    
    print(f"Using '{timestamp_col}' as timestamp column")
    
    # Convert to datetime
    df['datetime'] = pd.to_datetime(df[timestamp_col])
    
    # Extract date and time components
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    
    # Optimize numeric columns
    for col in ['high', 'low', 'open', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df):,} candles")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def filter_time_range(df, start_time, end_time):
    """
    Filter dataframe by time range (inclusive).
    Uses vectorized operations for efficiency.
    """
    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    return df[mask]


def analyze_daily_extremes(df, session_name, start_time, end_time):
    """
    Analyze daily high/low extremes for a given time range.
    Returns DataFrame with daily statistics.
    """
    print(f"\nAnalyzing {session_name} session ({start_time} - {end_time})...")
    
    # Filter to time range
    session_df = filter_time_range(df, start_time, end_time)
    
    if len(session_df) == 0:
        print(f"Warning: No data found in {session_name} time range")
        return pd.DataFrame()
    
    print(f"Processing {len(session_df):,} candles across {session_df['date'].nunique()} days")
    
    # Group by date for vectorized operations
    grouped = session_df.groupby('date')
    
    # Find daily highs
    high_idx = grouped['high'].idxmax()
    daily_highs = session_df.loc[high_idx, ['date', 'datetime', 'high']].copy()
    daily_highs['high_time'] = daily_highs['datetime'].dt.time
    daily_highs = daily_highs.rename(columns={'high': 'highest_high'})
    
    # Find daily lows
    low_idx = grouped['low'].idxmin()
    daily_lows = session_df.loc[low_idx, ['date', 'low']].copy()
    daily_lows['low_time'] = session_df.loc[low_idx, 'datetime'].dt.time.values
    daily_lows = daily_lows.rename(columns={'low': 'lowest_low'})
    
    # Merge results
    results = daily_highs.merge(daily_lows, on='date')
    results = results[['date', 'highest_high', 'high_time', 'lowest_low', 'low_time']]
    results = results.sort_values('date')
    
    return results


def format_time(t):
    """Format time object for display."""
    return t.strftime('%H:%M:%S') if isinstance(t, time) else str(t)


def time_in_range(t, start, end):
    """Check if time t is within range [start, end] inclusive."""
    return start <= t <= end


def classify_extremity(high_time, low_time, cluster_start, cluster_end, odrs_start, odrs_end):
    """
    Classify extremity status based on time clusters.
    
    Returns tuple: (status, ext_time)
    - status: 'False' if both in ODRS, 'High' if high in cluster, 'Low' if low in cluster, None otherwise
    - ext_time: the time of the extremity if status is High/Low, None otherwise
    """
    high_in_odrs = time_in_range(high_time, odrs_start, odrs_end)
    low_in_odrs = time_in_range(low_time, odrs_start, odrs_end)
    high_in_cluster = time_in_range(high_time, cluster_start, cluster_end)
    low_in_cluster = time_in_range(low_time, cluster_start, cluster_end)
    
    # Both in ODRS
    if high_in_odrs and low_in_odrs:
        return 'False', None
    
    # High in cluster
    if high_in_cluster:
        return 'High', high_time
    
    # Low in cluster
    if low_in_cluster:
        return 'Low', low_time
    
    # Neither condition met
    return None, None


def main():
    # Configuration
    CSV_PATH = 'candle_data.csv'
    
    # Session times
    FULLDAY_START = time(4, 0, 0)
    FULLDAY_END = time(15, 55, 0)
    HALFDAY_START = time(4, 0, 0)
    HALFDAY_END = time(9, 25, 0)
    BOX_START = time(4, 0, 0)
    BOX_END = time(10, 25, 0)
    
    # Time clusters
    ODRS_START = time(4, 0, 0)
    ODRS_END = time(8, 25, 0)
    RDRT_START = time(8, 30, 0)
    RDRT_END = time(9, 25, 0)
    RDRB_START = time(9, 30, 0)
    RDRB_END = time(10, 25, 0)
    RDRS_START = time(10, 30, 0)
    RDRS_END = time(15, 55, 0)
    
    # Check if CSV exists
    if not Path(CSV_PATH).exists():
        print(f"Error: {CSV_PATH} not found in current directory")
        print("Please ensure candle_data.csv is in the same directory as this script")
        sys.exit(1)
    
    # Load data
    df = load_and_prepare_data(CSV_PATH)
    
    # Analyze full day session
    fullday_results = analyze_daily_extremes(df, 'FULLDAY', FULLDAY_START, FULLDAY_END)
    
    # Analyze half day session
    halfday_results = analyze_daily_extremes(df, 'HALFDAY', HALFDAY_START, HALFDAY_END)
    
    # Analyze box session
    box_results = analyze_daily_extremes(df, 'BOX', BOX_START, BOX_END)
    
    # Add extremity classification columns to fullday_results
    if not fullday_results.empty:
        # Initialize new columns
        fullday_results['RDRT_EXT_STATUS'] = None
        fullday_results['RDRB_EXT_STATUS'] = None
        fullday_results['RDRT_EXT_TIME'] = None
        fullday_results['RDRB_EXT_TIME'] = None
        
        # Process each date
        for idx, row in fullday_results.iterrows():
            date = row['date']
            
            # Check RDRT status using halfday data
            if not halfday_results.empty and date in halfday_results['date'].values:
                halfday_row = halfday_results[halfday_results['date'] == date].iloc[0]
                rdrt_status, rdrt_time = classify_extremity(
                    halfday_row['high_time'],
                    halfday_row['low_time'],
                    RDRT_START,
                    RDRT_END,
                    ODRS_START,
                    ODRS_END
                )
                fullday_results.at[idx, 'RDRT_EXT_STATUS'] = rdrt_status
                fullday_results.at[idx, 'RDRT_EXT_TIME'] = rdrt_time
            
            # Check RDRB status using box data
            if not box_results.empty and date in box_results['date'].values:
                box_row = box_results[box_results['date'] == date].iloc[0]
                rdrb_status, rdrb_time = classify_extremity(
                    box_row['high_time'],
                    box_row['low_time'],
                    RDRB_START,
                    RDRB_END,
                    ODRS_START,
                    ODRS_END
                )
                fullday_results.at[idx, 'RDRB_EXT_STATUS'] = rdrb_status
                fullday_results.at[idx, 'RDRB_EXT_TIME'] = rdrb_time
    
    # Save results
    if not fullday_results.empty:
        fullday_output = 'fullday_analysis.csv'
        fullday_results.to_csv(fullday_output, index=False)
        print(f"\n✓ Full day results saved to: {fullday_output}")
        
        # Display sample
        print("\nFull Day Analysis (first 10 days):")
        print("=" * 100)
        for _, row in fullday_results.head(10).iterrows():
            print(f"Date: {row['date']}")
            print(f"  Highest High: {row['highest_high']:.2f} at {format_time(row['high_time'])}")
            print(f"  Lowest Low:   {row['lowest_low']:.2f} at {format_time(row['low_time'])}")
            print(f"  RDRT Status:  {row['RDRT_EXT_STATUS']}", end="")
            if row['RDRT_EXT_TIME'] is not None:
                print(f" at {format_time(row['RDRT_EXT_TIME'])}")
            else:
                print()
            print(f"  RDRB Status:  {row['RDRB_EXT_STATUS']}", end="")
            if row['RDRB_EXT_TIME'] is not None:
                print(f" at {format_time(row['RDRB_EXT_TIME'])}")
            else:
                print()
            print()
    
    if not halfday_results.empty:
        halfday_output = 'halfday_analysis.csv'
        halfday_results.to_csv(halfday_output, index=False)
        print(f"\n✓ Half day results saved to: {halfday_output}")
        
        # Display sample
        print("\nHalf Day Analysis (first 10 days):")
        print("=" * 80)
        for _, row in halfday_results.head(10).iterrows():
            print(f"Date: {row['date']}")
            print(f"  Highest High: {row['highest_high']:.2f} at {format_time(row['high_time'])}")
            print(f"  Lowest Low:   {row['lowest_low']:.2f} at {format_time(row['low_time'])}")
            print()
    
    if not box_results.empty:
        box_output = 'box_analysis.csv'
        box_results.to_csv(box_output, index=False)
        print(f"\n✓ Box results saved to: {box_output}")
        
        # Display sample
        print("\nBox Analysis (first 10 days):")
        print("=" * 80)
        for _, row in box_results.head(10).iterrows():
            print(f"Date: {row['date']}")
            print(f"  Highest High: {row['highest_high']:.2f} at {format_time(row['high_time'])}")
            print(f"  Lowest Low:   {row['lowest_low']:.2f} at {format_time(row['low_time'])}")
            print()
    
    # Summary statistics
    if not fullday_results.empty:
        print("\nFull Day Summary:")
        print("=" * 80)
        print(f"Total days analyzed: {len(fullday_results)}")
        print(f"Average high: {fullday_results['highest_high'].mean():.2f}")
        print(f"Average low: {fullday_results['lowest_low'].mean():.2f}")
        print(f"Absolute highest: {fullday_results['highest_high'].max():.2f}")
        print(f"Absolute lowest: {fullday_results['lowest_low'].min():.2f}")
        
        # Extremity cluster statistics
        print("\nExtremity Cluster Statistics:")
        print("-" * 80)
        rdrt_counts = fullday_results['RDRT_EXT_STATUS'].value_counts()
        print(f"RDRT (8:30-9:25):")
        for status, count in rdrt_counts.items():
            if status:
                print(f"  {status}: {count} days ({count/len(fullday_results)*100:.1f}%)")
        
        rdrb_counts = fullday_results['RDRB_EXT_STATUS'].value_counts()
        print(f"RDRB (9:30-10:25):")
        for status, count in rdrb_counts.items():
            if status:
                print(f"  {status}: {count} days ({count/len(fullday_results)*100:.1f}%)")
    
    if not halfday_results.empty:
        print("\nHalf Day Summary:")
        print("=" * 80)
        print(f"Total days analyzed: {len(halfday_results)}")
        print(f"Average high: {halfday_results['highest_high'].mean():.2f}")
        print(f"Average low: {halfday_results['lowest_low'].mean():.2f}")
        print(f"Absolute highest: {halfday_results['highest_high'].max():.2f}")
        print(f"Absolute lowest: {halfday_results['lowest_low'].min():.2f}")
    
    if not box_results.empty:
        print("\nBox Summary:")
        print("=" * 80)
        print(f"Total days analyzed: {len(box_results)}")
        print(f"Average high: {box_results['highest_high'].mean():.2f}")
        print(f"Average low: {box_results['lowest_low'].mean():.2f}")
        print(f"Absolute highest: {box_results['highest_high'].max():.2f}")
        print(f"Absolute lowest: {box_results['lowest_low'].min():.2f}")
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
