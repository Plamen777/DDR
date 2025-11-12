#!/usr/bin/env python3
"""
Crude Oil High/Low Probability Analysis
Analyzes the probability of intraday highs and lows holding until end of trading day.
Optimized for large datasets using vectorized operations.
"""

import pandas as pd
import numpy as np
from datetime import time, datetime
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path):
    """
    Load candle data and prepare it for analysis.
    Expects columns: timestamp, open, high, low, close
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    else:
        # Try to find a datetime column
        for col in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df[col])
                break
            except:
                continue
    
    # Extract date and time components
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    
    # Filter only trading hours (4:00 - 15:55)
    df = df[
        (df['time'] >= time(4, 0)) & 
        (df['time'] <= time(15, 55))
    ].copy()
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} candles across {df['date'].nunique()} trading days")
    
    return df


def generate_observation_times():
    """
    Generate observation time buckets: 4:30, 5:00, 5:30, ..., 16:00
    """
    times = [time(4, 30)]  # First observation at 4:30
    
    # Then every 30 minutes from 5:00 to 16:00
    for hour in range(5, 16):
        times.append(time(hour, 0))
        times.append(time(hour, 30))
    times.append(time(16, 0))
    
    return times


def analyze_day(day_data, observation_time):
    """
    Analyze a single day for a specific observation time.
    Returns: dict with high/low values, their times, break status, and breaking times.
    """
    # Formation period: 4:00 until observation time
    formation_mask = day_data['time'] < observation_time
    formation_data = day_data[formation_mask]
    
    if len(formation_data) == 0:
        return None
    
    # Find high and low in formation period
    high_idx = formation_data['high'].idxmax()
    low_idx = formation_data['low'].idxmin()
    
    high_value = formation_data.loc[high_idx, 'high']
    low_value = formation_data.loc[low_idx, 'low']
    high_time = formation_data.loc[high_idx, 'time']
    low_time = formation_data.loc[low_idx, 'time']
    
    # Testing period: after observation time until end of day
    testing_mask = day_data['time'] >= observation_time
    testing_data = day_data[testing_mask]
    
    if len(testing_data) == 0:
        # If no data after observation time, levels hold by default
        return {
            'high_value': high_value,
            'high_time': high_time,
            'low_value': low_value,
            'low_time': low_time,
            'high_broken': False,
            'low_broken': False,
            'high_break_time': None,
            'low_break_time': None
        }
    
    # Check if high/low were broken and find the first break time
    high_break_mask = testing_data['high'] > high_value
    low_break_mask = testing_data['low'] < low_value
    
    high_broken = high_break_mask.any()
    low_broken = low_break_mask.any()
    
    # Find the time when each level was broken (first occurrence)
    high_break_time = None
    low_break_time = None
    
    if high_broken:
        high_break_idx = testing_data[high_break_mask].index[0]
        high_break_time = testing_data.loc[high_break_idx, 'time']
    
    if low_broken:
        low_break_idx = testing_data[low_break_mask].index[0]
        low_break_time = testing_data.loc[low_break_idx, 'time']
    
    return {
        'high_value': high_value,
        'high_time': high_time,
        'low_value': low_value,
        'low_time': low_time,
        'high_broken': high_broken,
        'low_broken': low_broken,
        'high_break_time': high_break_time,
        'low_break_time': low_break_time
    }


def calculate_probabilities(df):
    """
    Calculate hold/break probabilities for all observation times across all days.
    Optimized for large datasets.
    """
    observation_times = generate_observation_times()
    results = []
    
    # Group by date for efficient processing
    grouped = df.groupby('date')
    total_days = len(grouped)
    
    print(f"\nAnalyzing {total_days} days across {len(observation_times)} observation times...")
    
    for obs_idx, obs_time in enumerate(observation_times):
        print(f"Processing observation time {obs_idx + 1}/{len(observation_times)}: {obs_time.strftime('%H:%M')}", end='\r')
        
        day_results = []
        
        # Process each day
        for date, day_data in grouped:
            result = analyze_day(day_data, obs_time)
            if result is not None:
                day_results.append(result)
        
        if len(day_results) == 0:
            continue
        
        # Calculate statistics
        total_valid_days = len(day_results)
        high_breaks = sum(1 for r in day_results if r['high_broken'])
        low_breaks = sum(1 for r in day_results if r['low_broken'])
        
        high_hold_pct = ((total_valid_days - high_breaks) / total_valid_days) * 100
        high_break_pct = (high_breaks / total_valid_days) * 100
        low_hold_pct = ((total_valid_days - low_breaks) / total_valid_days) * 100
        low_break_pct = (low_breaks / total_valid_days) * 100
        
        # Calculate average values and times
        avg_high = np.mean([r['high_value'] for r in day_results])
        avg_low = np.mean([r['low_value'] for r in day_results])
        
        results.append({
            'observation_time': obs_time.strftime('%H:%M'),
            'formation_period': f"04:00 - {obs_time.strftime('%H:%M')}",
            'testing_period': f"{obs_time.strftime('%H:%M')} - 15:55",
            'days_analyzed': total_valid_days,
            'high_hold_pct': round(high_hold_pct, 2),
            'high_break_pct': round(high_break_pct, 2),
            'low_hold_pct': round(low_hold_pct, 2),
            'low_break_pct': round(low_break_pct, 2),
            'avg_high_value': round(avg_high, 2),
            'avg_low_value': round(avg_low, 2),
            'high_breaks_count': high_breaks,
            'low_breaks_count': low_breaks
        })
    
    print("\n")
    return pd.DataFrame(results)


def generate_detailed_report(df, results_df):
    """
    Generate a detailed report with per-day breakdown.
    Returns a DataFrame with each day's high/low times, break status, and breaking times.
    """
    print("Generating detailed per-day report...")
    
    observation_times = generate_observation_times()
    grouped = df.groupby('date')
    
    detailed_records = []
    
    for date, day_data in grouped:
        for obs_time in observation_times:
            result = analyze_day(day_data, obs_time)
            
            if result is not None:
                detailed_records.append({
                    'date': date,
                    'observation_time': obs_time.strftime('%H:%M'),
                    'high_value': round(result['high_value'], 2),
                    'high_time': result['high_time'].strftime('%H:%M'),
                    'high_broken': result['high_broken'],
                    'high_break_time': result['high_break_time'].strftime('%H:%M') if result['high_break_time'] else None,
                    'low_value': round(result['low_value'], 2),
                    'low_time': result['low_time'].strftime('%H:%M'),
                    'low_broken': result['low_broken'],
                    'low_break_time': result['low_break_time'].strftime('%H:%M') if result['low_break_time'] else None
                })
    
    return pd.DataFrame(detailed_records)


def main(csv_path, output_summary='probability_summary.csv', output_detailed='detailed_breakdown.csv'):
    """
    Main execution function.
    """
    print("=" * 80)
    print("CRUDE OIL HIGH/LOW PROBABILITY ANALYSIS")
    print("=" * 80)
    
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    
    # Calculate probabilities
    results_df = calculate_probabilities(df)
    
    # Generate detailed report
    detailed_df = generate_detailed_report(df, results_df)
    
    # Save results
    results_df.to_csv(output_summary, index=False)
    detailed_df.to_csv(output_detailed, index=False)
    
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}\n")
    print(results_df.to_string(index=False))
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to:")
    print(f"  - Summary: {output_summary}")
    print(f"  - Detailed: {output_detailed}")
    print(f"{'=' * 80}\n")
    
    return results_df, detailed_df


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python candle_probability_analysis.py <path_to_candle_data.csv>")
        print("\nOptional arguments:")
        print("  python candle_probability_analysis.py <csv_path> <summary_output.csv> <detailed_output.csv>")
        print("\nExample:")
        print("  python candle_probability_analysis.py candle_data.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Create output filenames in the same directory as input file
    input_dir = os.path.dirname(os.path.abspath(csv_path)) if os.path.dirname(csv_path) else '.'
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    default_summary = os.path.join(input_dir, f'{base_name}_probability_summary.csv')
    default_detailed = os.path.join(input_dir, f'{base_name}_detailed_breakdown.csv')
    
    output_summary = sys.argv[2] if len(sys.argv) > 2 else default_summary
    output_detailed = sys.argv[3] if len(sys.argv) > 3 else default_detailed
    
    results_df, detailed_df = main(csv_path, output_summary, output_detailed)
