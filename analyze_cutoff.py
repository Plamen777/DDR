#!/usr/bin/env python3
"""
CL Futures Cut-off Time Analysis
Analyzes how often highs and lows established by a given time hold until session end.
Uses 30-minute intervals from 4:30 to 15:55.
"""

import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta
import sys
from pathlib import Path


def load_and_prepare_data(csv_path):
    """
    Load CSV and prepare data with optimized dtypes.
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Detect timestamp column
    timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Date', 'DateTime', 'Timestamp']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
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


def generate_cutoff_times():
    """
    Generate cut-off times every 30 minutes from 4:30 to 15:55.
    """
    cutoff_times = []
    
    # Start at 4:30
    current = datetime.strptime("04:30", "%H:%M")
    end = datetime.strptime("15:55", "%H:%M")
    
    while current <= end:
        cutoff_times.append(current.time())
        current += timedelta(minutes=30)
    
    return cutoff_times


def analyze_cutoff_time(df, cutoff_time, session_start, session_end):
    """
    Analyze how often highs and lows established by cutoff_time hold until session end.
    OPTIMIZED VERSION - uses vectorized operations for large datasets.
    
    Args:
        df: DataFrame with candle data
        cutoff_time: time object - the cut-off time to analyze
        session_start: time object - session start (4:00)
        session_end: time object - session end (15:55)
    
    Returns:
        Dictionary with analysis results
    """
    # Create masks for before and after cutoff
    df['before_cutoff'] = df['time'] <= cutoff_time
    df['after_cutoff'] = df['time'] > cutoff_time
    
    # Group by date and compute aggregations in one pass
    grouped = df.groupby('date')
    
    # Get counts to filter valid days (must have data before and after cutoff)
    before_counts = grouped['before_cutoff'].sum()
    after_counts = grouped['after_cutoff'].sum()
    valid_dates = (before_counts > 0) & (after_counts > 0)
    
    if valid_dates.sum() == 0:
        return None
    
    valid_date_list = valid_dates[valid_dates].index
    df_valid = df[df['date'].isin(valid_date_list)].copy()
    
    # For each date, compute high/low before and after cutoff
    # Split data into before and after cutoff
    before_df = df_valid[df_valid['before_cutoff']].copy()
    after_df = df_valid[df_valid['after_cutoff']].copy()
    
    # Aggregate before cutoff
    before_agg = before_df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'cutoff_high', 'low': 'cutoff_low'})
    
    # Aggregate after cutoff
    after_agg = after_df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'post_high', 'low': 'post_low'})
    
    # Merge and compute hold conditions
    results = before_agg.join(after_agg, how='inner')
    
    # Check if extremes held
    results['high_held'] = results['post_high'] <= results['cutoff_high']
    results['low_held'] = results['post_low'] >= results['cutoff_low']
    results['both_held'] = results['high_held'] & results['low_held']
    results['neither_held'] = ~results['high_held'] & ~results['low_held']
    
    # Calculate statistics
    total_days = len(results)
    
    stats = {
        'cutoff_time': cutoff_time,
        'total_days': total_days,
        'high_held_count': int(results['high_held'].sum()),
        'high_held_pct': float(results['high_held'].sum() / total_days * 100),
        'low_held_count': int(results['low_held'].sum()),
        'low_held_pct': float(results['low_held'].sum() / total_days * 100),
        'both_held_count': int(results['both_held'].sum()),
        'both_held_pct': float(results['both_held'].sum() / total_days * 100),
        'neither_held_count': int(results['neither_held'].sum()),
        'neither_held_pct': float(results['neither_held'].sum() / total_days * 100),
        'only_high_held_count': int((results['high_held'] & ~results['low_held']).sum()),
        'only_high_held_pct': float((results['high_held'] & ~results['low_held']).sum() / total_days * 100),
        'only_low_held_count': int((results['low_held'] & ~results['high_held']).sum()),
        'only_low_held_pct': float((results['low_held'] & ~results['high_held']).sum() / total_days * 100),
    }
    
    # Clean up temporary columns
    df.drop(['before_cutoff', 'after_cutoff'], axis=1, inplace=True)
    
    return stats


def format_time(t):
    """Format time object for display."""
    return t.strftime('%H:%M')


def main():
    # Configuration
    CSV_PATH = 'candle_data.csv'
    SESSION_START = time(4, 0, 0)
    SESSION_END = time(15, 55, 0)
    
    # Check if CSV exists
    if not Path(CSV_PATH).exists():
        print(f"Error: {CSV_PATH} not found in current directory")
        print("Please ensure candle_data.csv is in the same directory as this script")
        sys.exit(1)
    
    # Load data
    df = load_and_prepare_data(CSV_PATH)
    
    # Generate cut-off times
    cutoff_times = generate_cutoff_times()
    print(f"\nAnalyzing {len(cutoff_times)} cut-off times (every 30 minutes from 04:30 to 15:55)")
    print("=" * 80)
    
    # Analyze each cut-off time
    all_results = []
    
    for cutoff_time in cutoff_times:
        print(f"Analyzing cut-off time: {format_time(cutoff_time)}...", end=" ")
        
        stats = analyze_cutoff_time(df, cutoff_time, SESSION_START, SESSION_END)
        
        if stats:
            all_results.append(stats)
            print(f"✓ ({stats['total_days']} days)")
        else:
            print("✗ (insufficient data)")
    
    # Create results DataFrame
    if not all_results:
        print("\nError: No valid results generated")
        sys.exit(1)
    
    results_df = pd.DataFrame(all_results)
    
    # Format cutoff_time for display
    results_df['cutoff_time_str'] = results_df['cutoff_time'].apply(format_time)
    
    # Save detailed results
    output_file = 'cutoff_analysis.csv'
    output_df = results_df[[
        'cutoff_time_str', 'total_days',
        'high_held_count', 'high_held_pct',
        'low_held_count', 'low_held_pct',
        'both_held_count', 'both_held_pct',
        'neither_held_count', 'neither_held_pct',
        'only_high_held_count', 'only_high_held_pct',
        'only_low_held_count', 'only_low_held_pct'
    ]].copy()
    
    output_df.columns = [
        'Cutoff_Time', 'Total_Days',
        'High_Held_Count', 'High_Held_Pct',
        'Low_Held_Count', 'Low_Held_Pct',
        'Both_Held_Count', 'Both_Held_Pct',
        'Neither_Held_Count', 'Neither_Held_Pct',
        'Only_High_Held_Count', 'Only_High_Held_Pct',
        'Only_Low_Held_Count', 'Only_Low_Held_Pct'
    ]
    
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Display summary table
    print("\n" + "=" * 80)
    print("CUT-OFF TIME ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Time':<10} {'Days':<8} {'High Held':<15} {'Low Held':<15} {'Both Held':<15}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{format_time(row['cutoff_time']):<10} "
              f"{row['total_days']:<8} "
              f"{row['high_held_count']:>4} ({row['high_held_pct']:>5.1f}%)   "
              f"{row['low_held_count']:>4} ({row['low_held_pct']:>5.1f}%)   "
              f"{row['both_held_count']:>4} ({row['both_held_pct']:>5.1f}%)")
    
    print("\n" + "=" * 80)
    print("INSIGHTS")
    print("=" * 80)
    
    # Find best times for each metric
    best_high_held = results_df.loc[results_df['high_held_pct'].idxmax()]
    best_low_held = results_df.loc[results_df['low_held_pct'].idxmax()]
    best_both_held = results_df.loc[results_df['both_held_pct'].idxmax()]
    
    print(f"\n📈 Best time for HIGH to hold: {format_time(best_high_held['cutoff_time'])} "
          f"({best_high_held['high_held_pct']:.1f}% hold rate)")
    
    print(f"📉 Best time for LOW to hold: {format_time(best_low_held['cutoff_time'])} "
          f"({best_low_held['low_held_pct']:.1f}% hold rate)")
    
    print(f"🎯 Best time for BOTH to hold: {format_time(best_both_held['cutoff_time'])} "
          f"({best_both_held['both_held_pct']:.1f}% hold rate)")
    
    # Trend analysis
    print("\n📊 TREND ANALYSIS:")
    print("-" * 80)
    
    # Compare early vs late
    early_cutoffs = results_df[results_df['cutoff_time'] <= time(8, 0)]
    late_cutoffs = results_df[results_df['cutoff_time'] >= time(12, 0)]
    
    if len(early_cutoffs) > 0 and len(late_cutoffs) > 0:
        early_high_avg = early_cutoffs['high_held_pct'].mean()
        late_high_avg = late_cutoffs['high_held_pct'].mean()
        early_low_avg = early_cutoffs['low_held_pct'].mean()
        late_low_avg = late_cutoffs['low_held_pct'].mean()
        
        print(f"Early session (≤08:00) avg high hold rate: {early_high_avg:.1f}%")
        print(f"Late session (≥12:00) avg high hold rate: {late_high_avg:.1f}%")
        print(f"Difference: {late_high_avg - early_high_avg:+.1f}%")
        print()
        print(f"Early session (≤08:00) avg low hold rate: {early_low_avg:.1f}%")
        print(f"Late session (≥12:00) avg low hold rate: {late_low_avg:.1f}%")
        print(f"Difference: {late_low_avg - early_low_avg:+.1f}%")
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print(f"✓ Detailed results saved to: {output_file}")
    print("\nUse the CSV file to create charts showing probability trends over time.")


if __name__ == '__main__':
    main()
