#!/usr/bin/env python3
"""
Streamlit App for Crude Oil High/Low Probability Analysis
Interactive visualization with time range filters for high/low formation periods
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import time, datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Crude Oil Probability Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Crude Oil High/Low Probability Analysis")
st.markdown("""
Analyze the probability of intraday highs and lows holding until end of trading day.
Filter by specific time ranges to see when your highs and lows were established.
""")

# File uploader
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload detailed breakdown CSV",
    type=['csv'],
    help="Upload the candle_data_detailed_breakdown.csv file"
)

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        df['high_time'] = pd.to_datetime(df['high_time'], format='%H:%M').dt.time
        df['low_time'] = pd.to_datetime(df['low_time'], format='%H:%M').dt.time
        df['observation_time'] = pd.to_datetime(df['observation_time'], format='%H:%M').dt.time
        
        # Handle break times (may be None/NaN)
        if 'high_break_time' in df.columns:
            df['high_break_time'] = pd.to_datetime(df['high_break_time'], format='%H:%M', errors='coerce').dt.time
        if 'low_break_time' in df.columns:
            df['low_break_time'] = pd.to_datetime(df['low_break_time'], format='%H:%M', errors='coerce').dt.time
        
        return df
    
    df = load_data(uploaded_file)
    
    st.sidebar.success(f"✅ Loaded {len(df)} records")
    st.sidebar.info(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Time helper function
    def time_to_minutes(t):
        """Convert time object to minutes since midnight"""
        return t.hour * 60 + t.minute
    
    def minutes_to_time(minutes):
        """Convert minutes since midnight to time object"""
        return time(minutes // 60, minutes % 60)
    
    # Generate 5-minute intervals from 4:00 to 15:55
    def generate_5min_intervals():
        """Generate list of times in 5-minute intervals from 4:00 to 15:55"""
        intervals = []
        start_minutes = 4 * 60  # 4:00 in minutes
        end_minutes = 15 * 60 + 55  # 15:55 in minutes
        
        for minutes in range(start_minutes, end_minutes + 1, 5):
            hours = minutes // 60
            mins = minutes % 60
            intervals.append(time(hours, mins))
        
        return intervals
    
    time_intervals = generate_5min_intervals()
    time_interval_strings = [t.strftime('%H:%M') for t in time_intervals]
    
    # High time range filter
    st.sidebar.subheader("⬆️ High Formation Time Range")
    high_start_str = st.sidebar.selectbox(
        "High Start Time",
        options=time_interval_strings,
        index=0,  # Default to 4:00
        help="Only consider highs that occurred after this time"
    )
    high_end_str = st.sidebar.selectbox(
        "High End Time",
        options=time_interval_strings,
        index=len(time_interval_strings) - 1,  # Default to 15:55
        help="Only consider highs that occurred before this time (inclusive)"
    )
    
    high_start = datetime.strptime(high_start_str, '%H:%M').time()
    high_end = datetime.strptime(high_end_str, '%H:%M').time()
    
    # Low time range filter
    st.sidebar.subheader("⬇️ Low Formation Time Range")
    low_start_str = st.sidebar.selectbox(
        "Low Start Time",
        options=time_interval_strings,
        index=0,  # Default to 4:00
        help="Only consider lows that occurred after this time"
    )
    low_end_str = st.sidebar.selectbox(
        "Low End Time",
        options=time_interval_strings,
        index=len(time_interval_strings) - 1,  # Default to 15:55
        help="Only consider lows that occurred before this time (inclusive)"
    )
    
    low_start = datetime.strptime(low_start_str, '%H:%M').time()
    low_end = datetime.strptime(low_end_str, '%H:%M').time()
    
    # Observation time filter
    st.sidebar.subheader("👁️ Observation Time")
    available_obs_times = sorted(df['observation_time'].unique())
    
    # Convert to datetime for display
    obs_time_display = [datetime.combine(datetime.today(), t).strftime('%H:%M') for t in available_obs_times]
    
    default_index = len(available_obs_times) - 1  # Default to last time (16:00)
    
    selected_obs_time_str = st.sidebar.selectbox(
        "Select Observation Time",
        options=obs_time_display,
        index=default_index,
        help="Time at which to check if high/low levels hold"
    )
    
    # Convert back to time object
    selected_obs_time = datetime.strptime(selected_obs_time_str, '%H:%M').time()
    
    # Apply filters
    filtered_df = df[
        (df['high_time'] >= high_start) &
        (df['high_time'] <= high_end) &
        (df['low_time'] >= low_start) &
        (df['low_time'] <= low_end) &
        (df['observation_time'] == selected_obs_time)
    ].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Records", len(filtered_df))
    st.sidebar.metric("Total Trading Days", filtered_df['date'].nunique())
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches your filter criteria. Please adjust the time ranges.")
    else:
        # Calculate statistics
        total_days = len(filtered_df)
        high_holds = (~filtered_df['high_broken']).sum()
        high_breaks = filtered_df['high_broken'].sum()
        low_holds = (~filtered_df['low_broken']).sum()
        low_breaks = filtered_df['low_broken'].sum()
        
        high_hold_pct = (high_holds / total_days) * 100
        high_break_pct = (high_breaks / total_days) * 100
        low_hold_pct = (low_holds / total_days) * 100
        low_break_pct = (low_breaks / total_days) * 100
        
        # Main metrics
        st.header("📈 Key Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "High Hold %",
                f"{high_hold_pct:.1f}%",
                delta=f"{high_holds} days",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "High Break %",
                f"{high_break_pct:.1f}%",
                delta=f"{high_breaks} days",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Low Hold %",
                f"{low_hold_pct:.1f}%",
                delta=f"{low_holds} days",
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                "Low Break %",
                f"{low_break_pct:.1f}%",
                delta=f"{low_breaks} days",
                delta_color="inverse"
            )
        
        # Break sequence analysis
        if 'high_break_time' in filtered_df.columns and 'low_break_time' in filtered_df.columns:
            st.markdown("---")
            st.header("⚡ Break Sequence Analysis")
            st.markdown("**Which level breaks first when both break?**")
            
            # Helper function to convert time to comparable format
            def time_to_minutes_safe(t):
                if pd.isna(t) or t is None:
                    return None
                return t.hour * 60 + t.minute
            
            # Calculate break sequences
            filtered_df['high_break_minutes'] = filtered_df['high_break_time'].apply(time_to_minutes_safe)
            filtered_df['low_break_minutes'] = filtered_df['low_break_time'].apply(time_to_minutes_safe)
            
            # Categorize each day
            def categorize_break(row):
                high_broke = row['high_broken']
                low_broke = row['low_broken']
                high_time = row['high_break_minutes']
                low_time = row['low_break_minutes']
                
                if not high_broke and not low_broke:
                    return 'Both Hold'
                elif high_broke and not low_broke:
                    return 'Only High Breaks'
                elif low_broke and not high_broke:
                    return 'Only Low Breaks'
                else:  # Both broke
                    if high_time is not None and low_time is not None:
                        if high_time < low_time:
                            return 'High Breaks First'
                        elif low_time < high_time:
                            return 'Low Breaks First'
                        else:
                            return 'Both Break Same Time'
                    return 'Both Break (Unknown Order)'
            
            filtered_df['break_sequence'] = filtered_df.apply(categorize_break, axis=1)
            
            # Calculate statistics
            sequence_counts = filtered_df['break_sequence'].value_counts()
            sequence_pcts = (sequence_counts / len(filtered_df) * 100).round(2)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                count = sequence_counts.get('Only High Breaks', 0)
                pct = sequence_pcts.get('Only High Breaks', 0.0)
                st.metric(
                    "🟠 Only High Breaks",
                    f"{pct:.1f}%",
                    delta=f"{count} days"
                )
            
            with col2:
                count = sequence_counts.get('Only Low Breaks', 0)
                pct = sequence_pcts.get('Only Low Breaks', 0.0)
                st.metric(
                    "🟣 Only Low Breaks",
                    f"{pct:.1f}%",
                    delta=f"{count} days"
                )
            
            with col3:
                # Calculate "Both Break %" - sum of all scenarios where both broke
                both_break_categories = ['High Breaks First', 'Low Breaks First', 'Both Break Same Time', 'Both Break (Unknown Order)']
                both_break_count = sum(sequence_counts.get(cat, 0) for cat in both_break_categories)
                both_break_pct = (both_break_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
                st.metric(
                    "⚫ Both Break",
                    f"{both_break_pct:.1f}%",
                    delta=f"{both_break_count} days"
                )
            
            with col4:
                count = sequence_counts.get('Both Hold', 0)
                pct = sequence_pcts.get('Both Hold', 0.0)
                st.metric(
                    "🟢 Both Hold",
                    f"{pct:.1f}%",
                    delta=f"{count} days"
                )
            
            st.markdown("---")
            
            # Bar chart showing which level breaks first (including single breaks)
            st.markdown("### 🔄 Which Level Breaks First?")
            st.markdown("**Overall probability of which extremity gets taken out first**")
            
            # New logic: Include ALL days where at least one level breaks
            # Categories:
            # - High Breaks First: high breaks before low (whether or not low breaks later)
            # - Low Breaks First: low breaks before high (whether or not high breaks later)
            # - Both Hold: neither breaks
            # - Both Break Same Time: both break in same candle (rare)
            
            def categorize_first_break(row):
                high_broke = row['high_broken']
                low_broke = row['low_broken']
                high_time = row['high_break_minutes']
                low_time = row['low_break_minutes']
                
                if not high_broke and not low_broke:
                    return 'Both Hold'
                elif high_broke and not low_broke:
                    return 'High Breaks First'
                elif low_broke and not high_broke:
                    return 'Low Breaks First'
                else:  # Both broke - compare times
                    if high_time is not None and low_time is not None:
                        if high_time < low_time:
                            return 'High Breaks First'
                        elif low_time < high_time:
                            return 'Low Breaks First'
                        else:
                            return 'Both Break Same Time'
                    return 'Both Break (Unknown Order)'
            
            filtered_df['first_break'] = filtered_df.apply(categorize_first_break, axis=1)
            
            # Calculate statistics
            first_break_counts = filtered_df['first_break'].value_counts()
            first_break_pcts = (first_break_counts / len(filtered_df) * 100).round(2)
            
            # Filter to show only the breaking scenarios (exclude Both Hold for the chart)
            break_data = filtered_df[filtered_df['first_break'] != 'Both Hold']
            
            if len(break_data) > 0:
                break_counts = break_data['first_break'].value_counts()
                break_pcts = (break_counts / len(break_data) * 100).round(2)
                
                fig_first_break = go.Figure()
                
                # Define colors for each category
                colors_map = {
                    'High Breaks First': '#e74c3c',
                    'Low Breaks First': '#3498db',
                    'Both Break Same Time': '#95a5a6'
                }
                
                bar_colors = [colors_map.get(cat, '#95a5a6') for cat in break_counts.index]
                
                fig_first_break.add_trace(go.Bar(
                    x=break_counts.index,
                    y=break_pcts.values,
                    text=[f"{pct:.1f}%<br>({count} days)" for pct, count in zip(break_pcts.values, break_counts.values)],
                    textposition='outside',
                    marker=dict(
                        color=bar_colors,
                        line=dict(color='rgba(0,0,0,0.3)', width=2)
                    ),
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br><extra></extra>'
                ))
                
                fig_first_break.update_layout(
                    title=f"Which Level Breaks First? ({len(break_data)} days with at least one break)",
                    yaxis_title="Percentage of Breaking Days",
                    xaxis_title="First Break",
                    height=450,
                    yaxis_range=[0, max(break_pcts.values) * 1.2] if len(break_pcts) > 0 else [0, 100],
                    showlegend=False
                )
                
                st.plotly_chart(fig_first_break, use_container_width=True)
                
                # Add summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "🔴 High Breaks First",
                        f"{break_pcts.get('High Breaks First', 0):.1f}%",
                        delta=f"{break_counts.get('High Breaks First', 0)} days"
                    )
                
                with col2:
                    st.metric(
                        "🔵 Low Breaks First",
                        f"{break_pcts.get('Low Breaks First', 0):.1f}%",
                        delta=f"{break_counts.get('Low Breaks First', 0)} days"
                    )
                
                # Add insight
                if 'High Breaks First' in break_pcts.index and 'Low Breaks First' in break_pcts.index:
                    high_first_pct = break_pcts['High Breaks First']
                    low_first_pct = break_pcts['Low Breaks First']
                    
                    if high_first_pct > low_first_pct:
                        leader = "High"
                        leader_pct = high_first_pct
                        diff = high_first_pct - low_first_pct
                    else:
                        leader = "Low"
                        leader_pct = low_first_pct
                        diff = low_first_pct - high_first_pct
                    
                    st.info(f"💡 **Insight:** When a level breaks, the **{leader}** breaks first **{leader_pct:.1f}%** of the time (difference of {diff:.1f}%). This tells you which extremity is more likely to be taken out first, regardless of whether it becomes a double breakout day.")
            else:
                st.info("No breaks in the filtered data.")
            
            # Timeline analysis - when do breaks typically happen
            st.subheader("⏰ Break Timing Distribution")
            
            # Add bucket size selector
            bucket_size = st.selectbox(
                "Time Bucket Size (minutes)",
                options=[5, 15, 30],
                index=1,  # Default to 15 minutes
                help="Group break times into buckets for easier visualization"
            )
            
            def bucket_minutes(minutes, bucket_size):
                """Bucket minutes into specified intervals"""
                if pd.isna(minutes) or minutes is None:
                    return None
                return (minutes // bucket_size) * bucket_size
            
            def minutes_to_time_label(minutes):
                """Convert minutes to time label"""
                hours = int(minutes // 60)
                mins = int(minutes % 60)
                return f"{hours:02d}:{mins:02d}"
            
            col1, col2 = st.columns(2)
            
            with col1:
                high_break_times = filtered_df[filtered_df['high_broken'] & filtered_df['high_break_minutes'].notna()].copy()
                if len(high_break_times) > 0:
                    # Create bucketed data
                    high_break_times['bucketed_minutes'] = high_break_times['high_break_minutes'].apply(
                        lambda x: bucket_minutes(x, bucket_size)
                    )
                    
                    # Count occurrences per bucket
                    bucket_counts = high_break_times['bucketed_minutes'].value_counts().sort_index()
                    
                    # Create labels for x-axis
                    bucket_labels = [minutes_to_time_label(m) for m in bucket_counts.index]
                    
                    # Create detailed hover text
                    hover_text = [
                        f"Time: {minutes_to_time_label(m)} - {minutes_to_time_label(m + bucket_size)}<br>Breaks: {count}<br>Bucket: {bucket_size} min"
                        for m, count in zip(bucket_counts.index, bucket_counts.values)
                    ]
                    
                    fig_high_timing = go.Figure()
                    
                    fig_high_timing.add_trace(go.Bar(
                        x=bucket_labels,
                        y=bucket_counts.values,
                        marker=dict(
                            color='#e74c3c',
                            line=dict(color='#c0392b', width=2)  # Border for bars
                        ),
                        hovertext=hover_text,
                        hoverinfo='text',
                        name='High Breaks'
                    ))
                    
                    fig_high_timing.update_layout(
                        title=f"High Break Time Distribution ({bucket_size}-min buckets)",
                        xaxis_title="Time Range",
                        yaxis_title="Number of Breaks",
                        height=350,
                        xaxis=dict(
                            tickangle=-45,
                            tickmode='linear'
                        ),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_high_timing, use_container_width=True)
                else:
                    st.info("No high breaks in filtered data.")
            
            with col2:
                low_break_times = filtered_df[filtered_df['low_broken'] & filtered_df['low_break_minutes'].notna()].copy()
                if len(low_break_times) > 0:
                    # Create bucketed data
                    low_break_times['bucketed_minutes'] = low_break_times['low_break_minutes'].apply(
                        lambda x: bucket_minutes(x, bucket_size)
                    )
                    
                    # Count occurrences per bucket
                    bucket_counts = low_break_times['bucketed_minutes'].value_counts().sort_index()
                    
                    # Create labels for x-axis
                    bucket_labels = [minutes_to_time_label(m) for m in bucket_counts.index]
                    
                    # Create detailed hover text
                    hover_text = [
                        f"Time: {minutes_to_time_label(m)} - {minutes_to_time_label(m + bucket_size)}<br>Breaks: {count}<br>Bucket: {bucket_size} min"
                        for m, count in zip(bucket_counts.index, bucket_counts.values)
                    ]
                    
                    fig_low_timing = go.Figure()
                    
                    fig_low_timing.add_trace(go.Bar(
                        x=bucket_labels,
                        y=bucket_counts.values,
                        marker=dict(
                            color='#3498db',
                            line=dict(color='#2980b9', width=2)  # Border for bars
                        ),
                        hovertext=hover_text,
                        hoverinfo='text',
                        name='Low Breaks'
                    ))
                    
                    fig_low_timing.update_layout(
                        title=f"Low Break Time Distribution ({bucket_size}-min buckets)",
                        xaxis_title="Time Range",
                        yaxis_title="Number of Breaks",
                        height=350,
                        xaxis=dict(
                            tickangle=-45,
                            tickmode='linear'
                        ),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_low_timing, use_container_width=True)
                else:
                    st.info("No low breaks in filtered data.")
        
        st.markdown("---")
        
        # Visualization section
        st.header("📊 Visualizations")
        
        st.subheader("Hold Probability Over Time")
        
        # Calculate probabilities across all observation times for line chart
        line_data = []
        for obs_time in available_obs_times:
            temp_df = df[
                (df['high_time'] >= high_start) &
                (df['high_time'] <= high_end) &
                (df['low_time'] >= low_start) &
                (df['low_time'] <= low_end) &
                (df['observation_time'] == obs_time)
            ]
            
            if len(temp_df) > 0:
                line_data.append({
                    'observation_time': obs_time,
                    'high_hold_pct': (~temp_df['high_broken']).sum() / len(temp_df) * 100,
                    'low_hold_pct': (~temp_df['low_broken']).sum() / len(temp_df) * 100,
                    'days': len(temp_df)
                })
        
        line_df = pd.DataFrame(line_data)
        line_df['time_str'] = line_df['observation_time'].apply(lambda x: x.strftime('%H:%M'))
        
        # Create line chart
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=line_df['time_str'],
            y=line_df['high_hold_pct'],
            mode='lines+markers',
            name='High Hold %',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8)
        ))
        
        fig_line.add_trace(go.Scatter(
            x=line_df['time_str'],
            y=line_df['low_hold_pct'],
            mode='lines+markers',
            name='Low Hold %',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        # Highlight selected observation time
        selected_point = line_df[line_df['observation_time'] == selected_obs_time]
        if not selected_point.empty:
            fig_line.add_trace(go.Scatter(
                x=[selected_point.iloc[0]['time_str']],
                y=[selected_point.iloc[0]['high_hold_pct']],
                mode='markers',
                name='Selected (High)',
                marker=dict(size=15, color='#2ecc71', symbol='star'),
                showlegend=False
            ))
            
            fig_line.add_trace(go.Scatter(
                x=[selected_point.iloc[0]['time_str']],
                y=[selected_point.iloc[0]['low_hold_pct']],
                mode='markers',
                name='Selected (Low)',
                marker=dict(size=15, color='#e74c3c', symbol='star'),
                showlegend=False
            ))
        
        fig_line.update_layout(
            title="Probability of Levels Holding Throughout the Day",
            xaxis_title="Observation Time",
            yaxis_title="Hold Probability (%)",
            yaxis_range=[0, 100],
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.info("💡 **Insight:** Later observation times typically show higher hold probabilities since there's less time remaining for levels to break.")
        
        st.markdown("---")
        # Data table
        st.markdown("---")
        st.markdown("---")
        st.header("📋 Filtered Data Table")
        
        # Build display columns based on what's available
        display_cols = ['date', 'high_value', 'high_time', 'high_broken']
        if 'high_break_time' in filtered_df.columns:
            display_cols.append('high_break_time')
        display_cols.extend(['low_value', 'low_time', 'low_broken'])
        if 'low_break_time' in filtered_df.columns:
            display_cols.append('low_break_time')
        if 'break_sequence' in filtered_df.columns:
            display_cols.append('break_sequence')
        
        display_df = filtered_df[display_cols].copy()
        display_df['date'] = display_df['date'].dt.date
        display_df = display_df.sort_values('date', ascending=False)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download filtered data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_analysis_{selected_obs_time_str.replace(':', '')}.csv",
            mime="text/csv"
        )

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload your `candle_data_detailed_breakdown.csv` file using the sidebar.")
    
    st.markdown("""
    ### How to use this app:
    
    1. **Upload your data**: Use the file uploader in the sidebar to upload `candle_data_detailed_breakdown.csv`
    
    2. **Set time filters**:
       - **High Formation Range**: Time window when you want to consider highs (e.g., 4:00-4:20)
       - **Low Formation Range**: Time window when you want to consider lows (e.g., 5:15-7:15)
       - **Observation Time**: When to check if levels hold (default: 16:00)
    
    3. **Explore visualizations**:
       - **Line Chart**: See how hold probabilities change throughout the day
       - **Heatmap**: Analyze patterns across dates and weekdays
       - **Scatter Plot**: Visualize high/low values vs their formation times
       - **Bar Charts**: Compare hold vs break statistics
    
    4. **Download results**: Export filtered data for further analysis
    
    ### Example Use Case:
    *"I want to know: if a high forms between 4:00-4:20 and a low forms between 5:15-7:15, 
    what's the probability they'll both hold until 16:00 (end of day)?"*
    
    This app will show you exactly that, with detailed visualizations! 📊
    """)
