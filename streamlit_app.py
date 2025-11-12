#!/usr/bin/env python3
"""
CL Futures Candle Data Analyzer - Streamlit Dashboard
Interactive analysis of daily high/low extremities with time clustering.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import time, datetime, timedelta
import numpy as np


# Time cluster definitions
CLUSTERS = {
    'ODRS': (time(4, 0, 0), time(8, 25, 0)),
    'RDRT': (time(8, 30, 0), time(9, 25, 0)),
    'RDRB': (time(9, 30, 0), time(10, 25, 0)),
    'RDRS': (time(10, 30, 0), time(15, 55, 0))
}

SESSION_START = time(4, 0, 0)
SESSION_END = time(15, 55, 0)

# Color schemes
CLUSTER_COLORS = {
    'ODRS': '#3498db',  # Blue
    'RDRT': '#e74c3c',  # Red
    'RDRB': '#f39c12',  # Orange
    'RDRS': '#9b59b6'   # Purple
}

EXT_COLORS = {
    'High': '#e74c3c',  # Red
    'Low': '#3498db',   # Blue
    'N/A': '#95a5a6'    # Gray
}


def get_percentage_color(percentage):
    """
    Generate color based on percentage using green-red spectrum with transparency.
    Higher percentages have more opacity, lower percentages are more transparent.
    Returns rgba color code for subtle visualization.
    """
    # Calculate opacity based on percentage (0.3 to 1.0 range for visibility)
    opacity = 0.3 + (percentage / 100) * 0.7
    
    if percentage >= 40:
        # Green for high percentages
        return f'rgba(46, 204, 113, {opacity})'  # #2ecc71 with opacity
    elif percentage >= 30:
        # Light green
        return f'rgba(52, 152, 219, {opacity})'  # #3498db blue-green with opacity
    elif percentage >= 20:
        # Yellow
        return f'rgba(241, 196, 15, {opacity})'  # #f1c40f with opacity
    elif percentage >= 10:
        # Orange
        return f'rgba(230, 126, 34, {opacity})'  # #e67e22 with opacity
    else:
        # Red for low percentages
        return f'rgba(231, 76, 60, {opacity})'  # #e74c3c with opacity


def get_text_color_for_background(percentage):
    """
    Return appropriate text color (black or white) based on background percentage.
    Darker backgrounds (high percentage) need white text, lighter backgrounds need black text.
    """
    if percentage >= 40:
        return 'white'
    elif percentage >= 20:
        return 'black'
    else:
        return 'black'



@st.cache_data
def load_data():
    """Load the fullday analysis CSV file."""
    try:
        df = pd.read_csv('fullday_analysis.csv')
        
        # Convert time columns to time objects
        for col in ['high_time', 'low_time', 'RDRT_EXT_TIME', 'RDRB_EXT_TIME']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    except FileNotFoundError:
        st.error("Error: fullday_analysis.csv not found. Please run analyze_candles.py first.")
        return None


@st.cache_data
def load_candle_data():
    """Load raw candle data for cut-off analysis."""
    try:
        df = pd.read_csv('candle_data.csv')
        
        # Detect timestamp column
        timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Date', 'DateTime', 'Timestamp']
        timestamp_col = None
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            timestamp_col = df.columns[0]
        
        # Convert to datetime
        df['datetime'] = pd.to_datetime(df[timestamp_col])
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        # Optimize numeric columns
        for col in ['high', 'low', 'open', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except FileNotFoundError:
        st.warning("candle_data.csv not found. Cut-off analysis will not be available.")
        return None


def time_in_range(t, start, end):
    """Check if time t is within range [start, end] inclusive."""
    if t is None:
        return False
    return start <= t <= end


def get_color_from_percentage(percentage):
    """
    Generate color based on percentage using green-red spectrum.
    High percentage (>30%) = Green
    Medium percentage (10-30%) = Yellow/Orange
    Low percentage (<10%) = Red
    
    Returns hex color code.
    """
    if percentage >= 30:
        # Green spectrum (30% and above)
        # Interpolate from yellow-green to dark green
        ratio = min((percentage - 30) / 40, 1.0)  # 30-70% range
        r = int(144 - ratio * 44)  # 144 -> 100
        g = int(238 - ratio * 68)  # 238 -> 170
        b = int(144 - ratio * 94)  # 144 -> 50
    elif percentage >= 10:
        # Yellow to Orange spectrum (10-30%)
        ratio = (percentage - 10) / 20  # 10-30% range
        r = int(255 - ratio * 111)  # 255 -> 144
        g = int(193 + ratio * 45)   # 193 -> 238
        b = int(7 + ratio * 137)    # 7 -> 144
    else:
        # Red spectrum (below 10%)
        # Darker red for very low percentages
        ratio = percentage / 10
        r = int(139 + ratio * 116)  # 139 -> 255
        g = int(0 + ratio * 193)    # 0 -> 193
        b = int(0 + ratio * 7)      # 0 -> 7
    
    return f'#{r:02x}{g:02x}{b:02x}'


def classify_time_to_cluster(t):
    """Classify a time into one of the clusters."""
    if t is None:
        return None
    
    for cluster_name, (start, end) in CLUSTERS.items():
        if time_in_range(t, start, end):
            return cluster_name
    return None


def analyze_cutoff_by_cluster(candle_df, fullday_df, cutoff_time, cluster_filter=None):
    """
    Analyze how often highs/lows from specific clusters hold until session end
    when checked at a given cut-off time.
    
    Args:
        candle_df: Raw candle data
        fullday_df: Full day analysis with cluster classifications
        cutoff_time: Time to check if extremes hold
        cluster_filter: Optional cluster to filter by (ODRS, RDRT, RDRB, RDRS)
    
    Returns:
        Dictionary with hold statistics by cluster
    """
    if candle_df is None or fullday_df is None:
        return None
    
    # Add cluster classifications to fullday_df if not present
    if 'high_cluster' not in fullday_df.columns:
        fullday_df['high_cluster'] = fullday_df['high_time'].apply(classify_time_to_cluster)
    if 'low_cluster' not in fullday_df.columns:
        fullday_df['low_cluster'] = fullday_df['low_time'].apply(classify_time_to_cluster)
    
    # Filter by cluster if specified
    if cluster_filter:
        fullday_filtered = fullday_df[
            (fullday_df['high_cluster'] == cluster_filter) | 
            (fullday_df['low_cluster'] == cluster_filter)
        ].copy()
    else:
        fullday_filtered = fullday_df.copy()
    
    if len(fullday_filtered) == 0:
        return None
    
    # Get dates to analyze
    dates_to_analyze = fullday_filtered['date'].dt.date.unique()
    
    # Filter candle data to relevant dates
    candle_filtered = candle_df[candle_df['date'].isin(dates_to_analyze)].copy()
    
    # Create masks for before and after cutoff
    candle_filtered['before_cutoff'] = candle_filtered['time'] <= cutoff_time
    candle_filtered['after_cutoff'] = candle_filtered['time'] > cutoff_time
    
    # Group by date
    grouped = candle_filtered.groupby('date')
    
    # Check which dates have data before and after cutoff
    before_counts = grouped['before_cutoff'].sum()
    after_counts = grouped['after_cutoff'].sum()
    valid_dates = (before_counts > 0) & (after_counts > 0)
    
    if valid_dates.sum() == 0:
        return None
    
    valid_date_list = valid_dates[valid_dates].index
    candle_valid = candle_filtered[candle_filtered['date'].isin(valid_date_list)].copy()
    
    # Split into before and after cutoff
    before_df = candle_valid[candle_valid['before_cutoff']]
    after_df = candle_valid[candle_valid['after_cutoff']]
    
    # Aggregate
    before_agg = before_df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'cutoff_high', 'low': 'cutoff_low'})
    
    after_agg = after_df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'post_high', 'low': 'post_low'})
    
    # Merge
    results = before_agg.join(after_agg, how='inner')
    results['high_held'] = results['post_high'] <= results['cutoff_high']
    results['low_held'] = results['post_low'] >= results['cutoff_low']
    
    # Join with fullday data to get cluster info
    fullday_for_merge = fullday_filtered.copy()
    fullday_for_merge['date'] = fullday_for_merge['date'].dt.date
    fullday_for_merge = fullday_for_merge.set_index('date')
    
    results = results.join(fullday_for_merge[['high_cluster', 'low_cluster', 'high_time', 'low_time']], how='inner')
    
    # Calculate statistics by cluster
    stats = {}
    
    for cluster in ['ODRS', 'RDRT', 'RDRB', 'RDRS']:
        # Highs from this cluster
        high_in_cluster = results[results['high_cluster'] == cluster]
        if len(high_in_cluster) > 0:
            high_held_count = high_in_cluster['high_held'].sum()
            high_held_pct = (high_held_count / len(high_in_cluster)) * 100
        else:
            high_held_count = 0
            high_held_pct = 0
        
        # Lows from this cluster
        low_in_cluster = results[results['low_cluster'] == cluster]
        if len(low_in_cluster) > 0:
            low_held_count = low_in_cluster['low_held'].sum()
            low_held_pct = (low_held_count / len(low_in_cluster)) * 100
        else:
            low_held_count = 0
            low_held_pct = 0
        
        stats[cluster] = {
            'high_total': len(high_in_cluster),
            'high_held_count': int(high_held_count),
            'high_held_pct': float(high_held_pct),
            'low_total': len(low_in_cluster),
            'low_held_count': int(low_held_count),
            'low_held_pct': float(low_held_pct)
        }
    
    return stats


def filter_data(df, high_filters, low_filters, rdrt_ext_filter=None, rdrb_ext_filter=None):
    """
    Filter dataframe based on high and low time cluster selections and EXT status.
    
    Args:
        df: DataFrame with fullday analysis
        high_filters: List of cluster names or custom time ranges for highs
        low_filters: List of cluster names or custom time ranges for lows
        rdrt_ext_filter: Filter for RDRT_EXT_STATUS ('High', 'Low', 'N/A', or None for all)
        rdrb_ext_filter: Filter for RDRB_EXT_STATUS ('High', 'Low', 'N/A', or None for all)
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Add cluster classification columns if not present
    if 'high_cluster' not in df.columns:
        df['high_cluster'] = df['high_time'].apply(classify_time_to_cluster)
    if 'low_cluster' not in df.columns:
        df['low_cluster'] = df['low_time'].apply(classify_time_to_cluster)
    
    # Start with all rows
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Apply high filters
    if high_filters and 'All' not in high_filters:
        high_mask = pd.Series([False] * len(df), index=df.index)
        for filter_item in high_filters:
            if isinstance(filter_item, tuple):  # Custom time range
                start, end = filter_item
                high_mask |= df['high_time'].apply(lambda t: time_in_range(t, start, end))
            else:  # Cluster name
                high_mask |= (df['high_cluster'] == filter_item)
        mask &= high_mask
    
    # Apply low filters
    if low_filters and 'All' not in low_filters:
        low_mask = pd.Series([False] * len(df), index=df.index)
        for filter_item in low_filters:
            if isinstance(filter_item, tuple):  # Custom time range
                start, end = filter_item
                low_mask |= df['low_time'].apply(lambda t: time_in_range(t, start, end))
            else:  # Cluster name
                low_mask |= (df['low_cluster'] == filter_item)
        mask &= low_mask
    
    # Apply RDRT_EXT filter
    if rdrt_ext_filter:
        if rdrt_ext_filter == 'N/A':
            # N/A means False or neither High nor Low (None/NaN)
            mask &= (df['RDRT_EXT_STATUS'] == 'False') | (df['RDRT_EXT_STATUS'].isna())
        else:
            mask &= (df['RDRT_EXT_STATUS'] == rdrt_ext_filter)
    
    # Apply RDRB_EXT filter
    if rdrb_ext_filter:
        if rdrb_ext_filter == 'N/A':
            # N/A means False or neither High nor Low (None/NaN)
            mask &= (df['RDRB_EXT_STATUS'] == 'False') | (df['RDRB_EXT_STATUS'].isna())
        else:
            mask &= (df['RDRB_EXT_STATUS'] == rdrb_ext_filter)
    
    return df[mask]


def time_to_minutes(t):
    """Convert time object to minutes since 00:00."""
    if t is None:
        return None
    return t.hour * 60 + t.minute


def bucket_time(t, bucket_size=5):
    """
    Bucket a time into intervals.
    
    Args:
        t: time object
        bucket_size: bucket size in minutes (5, 15, or 30)
    
    Returns:
        Start time of the bucket
    """
    if t is None:
        return None
    
    minutes = time_to_minutes(t)
    bucketed_minutes = (minutes // bucket_size) * bucket_size
    
    hours = bucketed_minutes // 60
    mins = bucketed_minutes % 60
    
    return time(hours, mins)


def create_time_distribution_chart(df, bucket_size=5):
    """
    Create a bar chart showing time distribution of highs and lows.
    
    Args:
        df: Filtered DataFrame
        bucket_size: Time bucket size in minutes (5, 15, or 30)
    
    Returns:
        Plotly figure
    """
    if df is None or df.empty:
        return None
    
    # Create bucketed times
    high_buckets = df['high_time'].apply(lambda t: bucket_time(t, bucket_size))
    low_buckets = df['low_time'].apply(lambda t: bucket_time(t, bucket_size))
    
    # Count occurrences
    high_counts = high_buckets.value_counts().sort_index()
    low_counts = low_buckets.value_counts().sort_index()
    
    # Create time labels
    all_buckets = sorted(set(high_counts.index) | set(low_counts.index))
    time_labels = [t.strftime('%H:%M') if t else 'Unknown' for t in all_buckets]
    
    # Prepare data
    high_values = [high_counts.get(t, 0) for t in all_buckets]
    low_values = [low_counts.get(t, 0) for t in all_buckets]
    
    # Create figure with consistent colors
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='High',
        x=time_labels,
        y=high_values,
        marker_color='#e74c3c'  # Consistent red
    ))
    
    fig.add_trace(go.Bar(
        name='Low',
        x=time_labels,
        y=low_values,
        marker_color='#3498db'  # Consistent blue
    ))
    
    fig.update_layout(
        title=f'Time Distribution of Highs and Lows ({bucket_size}-minute buckets)',
        xaxis_title='Time',
        yaxis_title='Count',
        barmode='group',
        height=500,
        hovermode='x unified',
        xaxis={'tickangle': -45}
    )
    
    return fig


def display_cluster_statistics(df):
    """Display dataset count and percentage breakdown by cluster."""
    if df is None or df.empty:
        st.warning("No data to display.")
        return
    
    # Add cluster classifications
    df['high_cluster'] = df['high_time'].apply(classify_time_to_cluster)
    df['low_cluster'] = df['low_time'].apply(classify_time_to_cluster)
    
    total_count = len(df)
    
    st.subheader(f"📊 Dataset Statistics (Total: {total_count} days)")
    
    # Create two columns for High and Low statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High Extremities")
        high_counts = df['high_cluster'].value_counts()
        
        # Create dataframe for display with color-coded percentages
        high_stats = []
        high_colors = []
        for cluster in ['ODRS', 'RDRT', 'RDRB', 'RDRS']:
            count = high_counts.get(cluster, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            color = get_percentage_color(percentage)  # Use percentage-based color
            high_colors.append(color)
            high_stats.append({
                'Cluster': cluster,
                'Time Range': f"{CLUSTERS[cluster][0].strftime('%H:%M')}-{CLUSTERS[cluster][1].strftime('%H:%M')}",
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        high_df = pd.DataFrame(high_stats)
        
        # Display table with colored styling
        st.markdown("""
        <style>
        .cluster-table {
            width: 100%;
            border-collapse: collapse;
        }
        .cluster-table th {
            background-color: #34495e;
            color: white;
            padding: 10px;
            text-align: left;
        }
        .cluster-table td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Build HTML table with dynamic coloring
        table_html = '<table class="cluster-table"><tr><th>Cluster</th><th>Time Range</th><th>Count</th><th>Percentage</th></tr>'
        for idx, row in high_df.iterrows():
            count = row['Count']
            percentage_val = float(row['Percentage'].rstrip('%'))
            bg_color = get_percentage_color(percentage_val)
            text_color = get_text_color_for_background(percentage_val)
            table_html += f'<tr><td><strong>{row["Cluster"]}</strong></td><td>{row["Time Range"]}</td>'
            table_html += f'<td>{count}</td><td style="background-color: {bg_color}; color: {text_color}; font-weight: bold; text-align: center;">{row["Percentage"]}</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Pie chart for highs - use soft pastel colors with gradient
        pastel_colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD']  # Soft pastel palette
        fig_high = go.Figure(data=[go.Pie(
            labels=high_df['Cluster'],
            values=high_df['Count'],
            marker=dict(
                colors=pastel_colors[:len(high_df)],
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05 if high_df.iloc[i]['Count'] == high_df['Count'].max() else 0 for i in range(len(high_df))]  # Highlight max
        )])
        fig_high.update_layout(
            title='High Distribution',
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_high, use_container_width=True)
    
    with col2:
        st.markdown("### Low Extremities")
        low_counts = df['low_cluster'].value_counts()
        
        # Create dataframe for display with color-coded percentages
        low_stats = []
        low_colors = []
        for cluster in ['ODRS', 'RDRT', 'RDRB', 'RDRS']:
            count = low_counts.get(cluster, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            color = get_percentage_color(percentage)  # Use percentage-based color
            low_colors.append(color)
            low_stats.append({
                'Cluster': cluster,
                'Time Range': f"{CLUSTERS[cluster][0].strftime('%H:%M')}-{CLUSTERS[cluster][1].strftime('%H:%M')}",
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        low_df = pd.DataFrame(low_stats)
        
        # Build HTML table with dynamic coloring
        table_html = '<table class="cluster-table"><tr><th>Cluster</th><th>Time Range</th><th>Count</th><th>Percentage</th></tr>'
        for idx, row in low_df.iterrows():
            count = row['Count']
            percentage_val = float(row['Percentage'].rstrip('%'))
            bg_color = get_percentage_color(percentage_val)
            text_color = get_text_color_for_background(percentage_val)
            table_html += f'<tr><td><strong>{row["Cluster"]}</strong></td><td>{row["Time Range"]}</td>'
            table_html += f'<td>{count}</td><td style="background-color: {bg_color}; color: {text_color}; font-weight: bold; text-align: center;">{row["Percentage"]}</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Pie chart for lows - use complementary soft colors
        pastel_colors_low = ['#B0E0E6', '#FFDAB9', '#E6E6FA', '#F0E68C']  # Complementary soft palette
        fig_low = go.Figure(data=[go.Pie(
            labels=low_df['Cluster'],
            values=low_df['Count'],
            marker=dict(
                colors=pastel_colors_low[:len(low_df)],
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05 if low_df.iloc[i]['Count'] == low_df['Count'].max() else 0 for i in range(len(low_df))]  # Highlight max
        )])
        fig_low.update_layout(
            title='Low Distribution',
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_low, use_container_width=True)


def display_ext_statistics(df):
    """Display RDRT_EXT and RDRB_EXT statistics."""
    if df is None or df.empty:
        st.warning("No data to display.")
        return
    
    total_count = len(df)
    
    st.subheader(f"🎯 EXT Status Statistics")
    
    # Create two columns for RDRT_EXT and RDRB_EXT
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### RDRT_EXT Status")
        st.markdown("*8:30 - 9:25 window*")
        
        # Count statuses, treating False and NaN as N/A
        rdrt_status = df['RDRT_EXT_STATUS'].copy()
        rdrt_status = rdrt_status.fillna('N/A')
        rdrt_status = rdrt_status.replace('False', 'N/A')
        rdrt_counts = rdrt_status.value_counts()
        
        # Create dataframe for display
        rdrt_stats = []
        rdrt_colors = []
        for status in ['High', 'Low', 'N/A']:
            count = rdrt_counts.get(status, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            color = get_percentage_color(percentage)  # Use percentage-based color
            rdrt_colors.append(color)
            rdrt_stats.append({
                'Status': status,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        rdrt_df = pd.DataFrame(rdrt_stats)
        
        # Build HTML table with dynamic coloring
        table_html = '<table class="cluster-table"><tr><th>Status</th><th>Count</th><th>Percentage</th></tr>'
        for idx, row in rdrt_df.iterrows():
            count = row['Count']
            percentage_val = float(row['Percentage'].rstrip('%'))
            bg_color = get_percentage_color(percentage_val)
            text_color = get_text_color_for_background(percentage_val)
            table_html += f'<tr><td><strong>{row["Status"]}</strong></td>'
            table_html += f'<td>{count}</td><td style="background-color: {bg_color}; color: {text_color}; font-weight: bold; text-align: center;">{row["Percentage"]}</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Pie chart for RDRT_EXT - use semantic colors
        status_colors = {
            'High': '#FF6B6B',    # Soft red for highs
            'Low': '#4ECDC4',     # Soft teal for lows
            'N/A': '#95A5A6'      # Gray for N/A
        }
        rdrt_pie_colors = [status_colors.get(status, '#95A5A6') for status in rdrt_df['Status']]
        
        fig_rdrt = go.Figure(data=[go.Pie(
            labels=rdrt_df['Status'],
            values=rdrt_df['Count'],
            marker=dict(
                colors=rdrt_pie_colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05 if rdrt_df.iloc[i]['Count'] == rdrt_df['Count'].max() else 0 for i in range(len(rdrt_df))]
        )])
        fig_rdrt.update_layout(
            title='RDRT_EXT Distribution',
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_rdrt, use_container_width=True)
    
    with col2:
        st.markdown("### RDRB_EXT Status")
        st.markdown("*9:30 - 10:25 window*")
        
        # Count statuses, treating False and NaN as N/A
        rdrb_status = df['RDRB_EXT_STATUS'].copy()
        rdrb_status = rdrb_status.fillna('N/A')
        rdrb_status = rdrb_status.replace('False', 'N/A')
        rdrb_counts = rdrb_status.value_counts()
        
        # Create dataframe for display
        rdrb_stats = []
        rdrb_colors = []
        for status in ['High', 'Low', 'N/A']:
            count = rdrb_counts.get(status, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            color = get_percentage_color(percentage)  # Use percentage-based color
            rdrb_colors.append(color)
            rdrb_stats.append({
                'Status': status,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        rdrb_df = pd.DataFrame(rdrb_stats)
        
        # Build HTML table with dynamic coloring
        table_html = '<table class="cluster-table"><tr><th>Status</th><th>Count</th><th>Percentage</th></tr>'
        for idx, row in rdrb_df.iterrows():
            count = row['Count']
            percentage_val = float(row['Percentage'].rstrip('%'))
            bg_color = get_percentage_color(percentage_val)
            text_color = get_text_color_for_background(percentage_val)
            table_html += f'<tr><td><strong>{row["Status"]}</strong></td>'
            table_html += f'<td>{count}</td><td style="background-color: {bg_color}; color: {text_color}; font-weight: bold; text-align: center;">{row["Percentage"]}</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Pie chart for RDRB_EXT - use semantic colors
        status_colors = {
            'High': '#FF6B6B',    # Soft red for highs
            'Low': '#4ECDC4',     # Soft teal for lows
            'N/A': '#95A5A6'      # Gray for N/A
        }
        rdrb_pie_colors = [status_colors.get(status, '#95A5A6') for status in rdrb_df['Status']]
        
        fig_rdrb = go.Figure(data=[go.Pie(
            labels=rdrb_df['Status'],
            values=rdrb_df['Count'],
            marker=dict(
                colors=rdrb_pie_colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05 if rdrb_df.iloc[i]['Count'] == rdrb_df['Count'].max() else 0 for i in range(len(rdrb_df))]
        )])
        fig_rdrb.update_layout(
            title='RDRB_EXT Distribution',
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_rdrb, use_container_width=True)


def main():
    st.set_page_config(
        page_title="CL Futures Analysis Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 CL Futures Candle Data Analysis Dashboard")
    st.markdown("Interactive analysis of daily high/low extremities with time clustering")
    
    # Load data
    df = load_data()
    
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # High filters
    st.sidebar.subheader("High Extremity Filters")
    high_cluster_options = ['ODRS', 'RDRT', 'RDRB', 'RDRS']
    high_selections = st.sidebar.multiselect(
        "Select time clusters for Highs",
        options=high_cluster_options,
        default=high_cluster_options  # All selected by default
    )
    
    # Custom high range checkbox
    high_use_custom = st.sidebar.checkbox("Use Custom Time Range for Highs", value=False, key='high_custom_check')
    
    high_filters = []
    if high_use_custom:
        # Custom range overwrites multiselection
        st.sidebar.markdown("**Custom High Time Range:**")
        high_custom_start = st.sidebar.time_input(
            "Start time",
            value=time(4, 0),
            key='high_start'
        )
        high_custom_end = st.sidebar.time_input(
            "End time",
            value=time(15, 55),
            key='high_end'
        )
        
        if high_custom_start >= SESSION_START and high_custom_end <= SESSION_END and high_custom_start < high_custom_end:
            high_filters.append((high_custom_start, high_custom_end))
        else:
            st.sidebar.error(f"Custom range must be between {SESSION_START.strftime('%H:%M')} and {SESSION_END.strftime('%H:%M')} with start < end")
    else:
        # Use multiselect
        if not high_selections:
            high_filters = ['All']  # If nothing selected, include all
        else:
            high_filters = high_selections
    
    # Low filters
    st.sidebar.subheader("Low Extremity Filters")
    low_cluster_options = ['ODRS', 'RDRT', 'RDRB', 'RDRS']
    low_selections = st.sidebar.multiselect(
        "Select time clusters for Lows",
        options=low_cluster_options,
        default=low_cluster_options  # All selected by default
    )
    
    # Custom low range checkbox
    low_use_custom = st.sidebar.checkbox("Use Custom Time Range for Lows", value=False, key='low_custom_check')
    
    low_filters = []
    if low_use_custom:
        # Custom range overwrites multiselection
        st.sidebar.markdown("**Custom Low Time Range:**")
        low_custom_start = st.sidebar.time_input(
            "Start time",
            value=time(4, 0),
            key='low_start'
        )
        low_custom_end = st.sidebar.time_input(
            "End time",
            value=time(15, 55),
            key='low_end'
        )
        
        if low_custom_start >= SESSION_START and low_custom_end <= SESSION_END and low_custom_start < low_custom_end:
            low_filters.append((low_custom_start, low_custom_end))
        else:
            st.sidebar.error(f"Custom range must be between {SESSION_START.strftime('%H:%M')} and {SESSION_END.strftime('%H:%M')} with start < end")
    else:
        # Use multiselect
        if not low_selections:
            low_filters = ['All']  # If nothing selected, include all
        else:
            low_filters = low_selections
    
    st.sidebar.markdown("---")
    
    # RDRT_EXT filter
    st.sidebar.subheader("RDRT_EXT Filter")
    st.sidebar.markdown("*8:30 - 9:25 window*")
    rdrt_ext_filter = st.sidebar.selectbox(
        "RDRT_EXT Status",
        options=['All', 'High', 'Low', 'N/A'],
        index=0,
        key='rdrt_ext'
    )
    if rdrt_ext_filter == 'All':
        rdrt_ext_filter = None
    
    # RDRB_EXT filter
    st.sidebar.subheader("RDRB_EXT Filter")
    st.sidebar.markdown("*9:30 - 10:25 window*")
    rdrb_ext_filter = st.sidebar.selectbox(
        "RDRB_EXT Status",
        options=['All', 'High', 'Low', 'N/A'],
        index=0,
        key='rdrb_ext'
    )
    if rdrb_ext_filter == 'All':
        rdrb_ext_filter = None
    
    # Apply filters
    filtered_df = filter_data(df.copy(), high_filters, low_filters, rdrt_ext_filter, rdrb_ext_filter)
    
    # Display active filters
    active_filters = []
    
    # High filters display
    if high_use_custom and high_filters:
        if isinstance(high_filters[0], tuple):
            start, end = high_filters[0]
            active_filters.append(f"High: Custom ({start.strftime('%H:%M')}-{end.strftime('%H:%M')})")
    elif high_filters != ['All'] and high_selections != high_cluster_options:
        active_filters.append(f"High: {', '.join(high_filters)}")
    
    # Low filters display
    if low_use_custom and low_filters:
        if isinstance(low_filters[0], tuple):
            start, end = low_filters[0]
            active_filters.append(f"Low: Custom ({start.strftime('%H:%M')}-{end.strftime('%H:%M')})")
    elif low_filters != ['All'] and low_selections != low_cluster_options:
        active_filters.append(f"Low: {', '.join(low_filters)}")
    
    # EXT filters display
    if rdrt_ext_filter:
        active_filters.append(f"RDRT_EXT: {rdrt_ext_filter}")
    if rdrb_ext_filter:
        active_filters.append(f"RDRB_EXT: {rdrb_ext_filter}")
    
    if active_filters:
        st.info(f"🔍 Active Filters: {' | '.join(active_filters)} | Showing {len(filtered_df)} of {len(df)} days")
    else:
        st.info(f"📊 Showing all data: {len(filtered_df)} days")
    
    # Main content
    st.markdown("---")
    
    # Display statistics
    display_cluster_statistics(filtered_df)
    
    st.markdown("---")
    
    # Display EXT statistics
    display_ext_statistics(filtered_df)
    
    st.markdown("---")
    
    # Cut-off Time Analysis
    st.subheader("🎯 Cut-off Time Analysis")
    st.markdown("*Analyze how often highs/lows from specific time clusters hold until session end (15:55)*")
    
    # Load candle data
    candle_df = load_candle_data()
    
    if candle_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Cut-off Time:**")
            st.markdown("*The time at which you check if extremes will hold*")
            cutoff_time = st.time_input(
                "Cut-off time",
                value=time(9, 25, 0),
                step=300,  # 5-minute increments
                key='cutoff_time'
            )
        
        with col2:
            st.markdown("**Filter by Cluster (Optional):**")
            st.markdown("*Analyze only highs/lows from specific time clusters*")
            cluster_filter_options = ['All Clusters', 'ODRS', 'RDRT', 'RDRB', 'RDRS']
            cluster_filter_selection = st.selectbox(
                "Cluster filter",
                options=cluster_filter_options,
                index=0,
                key='cluster_filter'
            )
            cluster_filter = None if cluster_filter_selection == 'All Clusters' else cluster_filter_selection
        
        # Run analysis
        if st.button("🔍 Analyze Cut-off Time", type="primary"):
            with st.spinner("Analyzing..."):
                stats = analyze_cutoff_by_cluster(candle_df, filtered_df, cutoff_time, cluster_filter)
                
                if stats:
                    st.success(f"✓ Analysis complete for cut-off time: {cutoff_time.strftime('%H:%M')}")
                    
                    # Display results
                    st.markdown(f"### Results: Probability Highs/Lows Hold from Cut-off Time ({cutoff_time.strftime('%H:%M')}) to 15:55")
                    
                    if cluster_filter:
                        st.info(f"Filtered to extremes occurring in: **{cluster_filter}**")
                    
                    # Create DataFrame for display
                    results_data = []
                    for cluster in ['ODRS', 'RDRT', 'RDRB', 'RDRS']:
                        cluster_stats = stats[cluster]
                        results_data.append({
                            'Cluster': cluster,
                            'Time Range': f"{CLUSTERS[cluster][0].strftime('%H:%M')}-{CLUSTERS[cluster][1].strftime('%H:%M')}",
                            'High Total': cluster_stats['high_total'],
                            'High Held': cluster_stats['high_held_count'],
                            'High Hold %': f"{cluster_stats['high_held_pct']:.1f}%",
                            'Low Total': cluster_stats['low_total'],
                            'Low Held': cluster_stats['low_held_count'],
                            'Low Hold %': f"{cluster_stats['low_held_pct']:.1f}%"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Display in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 High Hold Probabilities")
                        high_display = results_df[['Cluster', 'Time Range', 'High Total', 'High Held', 'High Hold %']].copy()
                        
                        # Color code the percentages
                        st.dataframe(
                            high_display.style.applymap(
                                lambda x: f'background-color: {get_percentage_color(float(x.rstrip("%")))}; color: {get_text_color_for_background(float(x.rstrip("%")))}' 
                                if isinstance(x, str) and '%' in x else '',
                                subset=['High Hold %']
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Bar chart for highs
                        fig_high = go.Figure()
                        fig_high.add_trace(go.Bar(
                            x=results_df['Cluster'],
                            y=[float(x.rstrip('%')) for x in results_df['High Hold %']],
                            marker_color=[CLUSTER_COLORS[c] for c in results_df['Cluster']],
                            text=results_df['High Hold %'],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Hold Rate: %{y:.1f}%<br>Total: ' + 
                                         results_df['High Total'].astype(str) + '<extra></extra>'
                        ))
                        fig_high.update_layout(
                            title='High Hold Rate by Cluster',
                            xaxis_title='Cluster',
                            yaxis_title='Hold Probability (%)',
                            yaxis_range=[0, 100],
                            height=400
                        )
                        st.plotly_chart(fig_high, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 📉 Low Hold Probabilities")
                        low_display = results_df[['Cluster', 'Time Range', 'Low Total', 'Low Held', 'Low Hold %']].copy()
                        
                        # Color code the percentages
                        st.dataframe(
                            low_display.style.applymap(
                                lambda x: f'background-color: {get_percentage_color(float(x.rstrip("%")))}; color: {get_text_color_for_background(float(x.rstrip("%")))}' 
                                if isinstance(x, str) and '%' in x else '',
                                subset=['Low Hold %']
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Bar chart for lows
                        fig_low = go.Figure()
                        fig_low.add_trace(go.Bar(
                            x=results_df['Cluster'],
                            y=[float(x.rstrip('%')) for x in results_df['Low Hold %']],
                            marker_color=[CLUSTER_COLORS[c] for c in results_df['Cluster']],
                            text=results_df['Low Hold %'],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Hold Rate: %{y:.1f}%<br>Total: ' + 
                                         results_df['Low Total'].astype(str) + '<extra></extra>'
                        ))
                        fig_low.update_layout(
                            title='Low Hold Rate by Cluster',
                            xaxis_title='Cluster',
                            yaxis_title='Hold Probability (%)',
                            yaxis_range=[0, 100],
                            height=400
                        )
                        st.plotly_chart(fig_low, use_container_width=True)
                    
                    # Insights
                    st.markdown("#### 💡 Insights")
                    
                    # Find best/worst hold rates
                    high_hold_pcts = [stats[c]['high_held_pct'] for c in ['ODRS', 'RDRT', 'RDRB', 'RDRS']]
                    low_hold_pcts = [stats[c]['low_held_pct'] for c in ['ODRS', 'RDRT', 'RDRB', 'RDRS']]
                    clusters_list = ['ODRS', 'RDRT', 'RDRB', 'RDRS']
                    
                    best_high_idx = high_hold_pcts.index(max(high_hold_pcts))
                    best_low_idx = low_hold_pcts.index(max(low_hold_pcts))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Best High Hold Rate",
                            f"{clusters_list[best_high_idx]}",
                            f"{high_hold_pcts[best_high_idx]:.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Best Low Hold Rate",
                            f"{clusters_list[best_low_idx]}",
                            f"{low_hold_pcts[best_low_idx]:.1f}%"
                        )
                else:
                    st.warning("No data available for analysis with current filters and cut-off time.")
    else:
        st.warning("⚠️ candle_data.csv not found. Cut-off analysis requires the raw candle data file.")
        st.info("To enable cut-off analysis, place candle_data.csv in the same directory as this script.")
    
    st.markdown("---")
    
    # Time distribution chart
    st.subheader("⏰ Time Distribution Analysis")
    
    # Bucket size selector
    col1, col2 = st.columns([1, 4])
    with col1:
        bucket_size = st.selectbox(
            "Time bucket size (minutes)",
            options=[5, 15, 30],
            index=0
        )
    
    # Create and display chart
    fig = create_time_distribution_chart(filtered_df, bucket_size)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to display in chart.")
    
    # Data preview
    st.markdown("---")
    st.subheader("📋 Filtered Data Preview")
    
    if not filtered_df.empty:
        # Format the dataframe for display
        display_df = filtered_df[[
            'date', 'highest_high', 'high_time', 'lowest_low', 'low_time',
            'RDRT_EXT_STATUS', 'RDRT_EXT_TIME', 'RDRB_EXT_STATUS', 'RDRB_EXT_TIME'
        ]].copy()
        
        # Format date
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data matches the current filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Time Clusters:**
    - **ODRS** (Overnight/Pre-Market): 04:00 - 08:25
    - **RDRT** (Regular Day Range Top): 08:30 - 09:25
    - **RDRB** (Regular Day Range Bottom): 09:30 - 10:25
    - **RDRS** (Regular Day Range Session): 10:30 - 15:55
    
    **Color Guide:**
    - **Cluster Colors:** ODRS (Blue), RDRT (Red), RDRB (Orange), RDRS (Purple)
    - **EXT Status:** High (Red), Low (Blue), N/A (Gray)
    - **Percentage Heat:** 🟢 Green (<10%) → 🟡 Yellow (10-30%) → 🟠 Orange (30-40%) → 🔴 Red (>40%)
    """)


if __name__ == '__main__':
    main()
