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


def time_in_range(t, start, end):
    """Check if time t is within range [start, end] inclusive."""
    if t is None:
        return False
    return start <= t <= end


def classify_time_to_cluster(t):
    """Classify a time into one of the clusters."""
    if t is None:
        return None
    
    for cluster_name, (start, end) in CLUSTERS.items():
        if time_in_range(t, start, end):
            return cluster_name
    return None


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
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='High',
        x=time_labels,
        y=high_values,
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Low',
        x=time_labels,
        y=low_values,
        marker_color='#4ECDC4'
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
        
        # Create dataframe for display
        high_stats = []
        for cluster in ['ODRS', 'RDRT', 'RDRB', 'RDRS']:
            count = high_counts.get(cluster, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            high_stats.append({
                'Cluster': cluster,
                'Time Range': f"{CLUSTERS[cluster][0].strftime('%H:%M')}-{CLUSTERS[cluster][1].strftime('%H:%M')}",
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        high_df = pd.DataFrame(high_stats)
        st.dataframe(high_df, hide_index=True, use_container_width=True)
        
        # Pie chart for highs
        fig_high = px.pie(
            values=high_df['Count'],
            names=high_df['Cluster'],
            title='High Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_high, use_container_width=True)
    
    with col2:
        st.markdown("### Low Extremities")
        low_counts = df['low_cluster'].value_counts()
        
        # Create dataframe for display
        low_stats = []
        for cluster in ['ODRS', 'RDRT', 'RDRB', 'RDRS']:
            count = low_counts.get(cluster, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            low_stats.append({
                'Cluster': cluster,
                'Time Range': f"{CLUSTERS[cluster][0].strftime('%H:%M')}-{CLUSTERS[cluster][1].strftime('%H:%M')}",
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        low_df = pd.DataFrame(low_stats)
        st.dataframe(low_df, hide_index=True, use_container_width=True)
        
        # Pie chart for lows
        fig_low = px.pie(
            values=low_df['Count'],
            names=low_df['Cluster'],
            title='Low Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel
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
        for status in ['High', 'Low', 'N/A']:
            count = rdrt_counts.get(status, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            rdrt_stats.append({
                'Status': status,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        rdrt_df = pd.DataFrame(rdrt_stats)
        st.dataframe(rdrt_df, hide_index=True, use_container_width=True)
        
        # Pie chart for RDRT_EXT
        fig_rdrt = px.pie(
            values=rdrt_df['Count'],
            names=rdrt_df['Status'],
            title='RDRT_EXT Distribution',
            color_discrete_map={'High': '#FF6B6B', 'Low': '#4ECDC4', 'N/A': '#95A5A6'}
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
        for status in ['High', 'Low', 'N/A']:
            count = rdrb_counts.get(status, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            rdrb_stats.append({
                'Status': status,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        rdrb_df = pd.DataFrame(rdrb_stats)
        st.dataframe(rdrb_df, hide_index=True, use_container_width=True)
        
        # Pie chart for RDRB_EXT
        fig_rdrb = px.pie(
            values=rdrb_df['Count'],
            names=rdrb_df['Status'],
            title='RDRB_EXT Distribution',
            color_discrete_map={'High': '#FF6B6B', 'Low': '#4ECDC4', 'N/A': '#95A5A6'}
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
    """)


if __name__ == '__main__':
    main()
