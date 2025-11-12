#!/usr/bin/env python3
"""
Visualize Cut-off Time Analysis Results
Creates interactive charts showing how hold probabilities change throughout the day.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path


def create_visualizations(csv_path='cutoff_analysis.csv'):
    """
    Create interactive visualizations from cut-off analysis results.
    """
    # Load results
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        print("Please run analyze_cutoff.py first to generate the analysis.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} cut-off time analysis results")
    
    # Create main figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'High & Low Hold Probability Over Time',
            'Both Held vs Neither Held',
            'Only High Held vs Only Low Held',
            'Summary Statistics'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Chart 1: High and Low Hold Probability
    fig.add_trace(
        go.Scatter(
            x=df['Cutoff_Time'],
            y=df['High_Held_Pct'],
            mode='lines+markers',
            name='High Held',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>High Held: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Cutoff_Time'],
            y=df['Low_Held_Pct'],
            mode='lines+markers',
            name='Low Held',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>Low Held: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Chart 2: Both vs Neither
    fig.add_trace(
        go.Scatter(
            x=df['Cutoff_Time'],
            y=df['Both_Held_Pct'],
            mode='lines+markers',
            name='Both Held',
            line=dict(color='#95E1D3', width=3),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>Both Held: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Cutoff_Time'],
            y=df['Neither_Held_Pct'],
            mode='lines+markers',
            name='Neither Held',
            line=dict(color='#F38181', width=3),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>Neither Held: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Chart 3: Only High vs Only Low
    fig.add_trace(
        go.Scatter(
            x=df['Cutoff_Time'],
            y=df['Only_High_Held_Pct'],
            mode='lines+markers',
            name='Only High Held',
            line=dict(color='#AA96DA', width=3),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>Only High: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Cutoff_Time'],
            y=df['Only_Low_Held_Pct'],
            mode='lines+markers',
            name='Only Low Held',
            line=dict(color='#FCBAD3', width=3),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>Only Low: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Table: Summary statistics
    # Find key insights
    max_high_idx = df['High_Held_Pct'].idxmax()
    max_low_idx = df['Low_Held_Pct'].idxmax()
    max_both_idx = df['Both_Held_Pct'].idxmax()
    min_both_idx = df['Both_Held_Pct'].idxmin()
    
    summary_data = [
        ['Metric', 'Time', 'Value'],
        ['Best High Hold', df.loc[max_high_idx, 'Cutoff_Time'], 
         f"{df.loc[max_high_idx, 'High_Held_Pct']:.1f}%"],
        ['Best Low Hold', df.loc[max_low_idx, 'Cutoff_Time'], 
         f"{df.loc[max_low_idx, 'Low_Held_Pct']:.1f}%"],
        ['Best Both Hold', df.loc[max_both_idx, 'Cutoff_Time'], 
         f"{df.loc[max_both_idx, 'Both_Held_Pct']:.1f}%"],
        ['Worst Both Hold', df.loc[min_both_idx, 'Cutoff_Time'], 
         f"{df.loc[min_both_idx, 'Both_Held_Pct']:.1f}%"],
        ['Avg High Hold', 'All Times', f"{df['High_Held_Pct'].mean():.1f}%"],
        ['Avg Low Hold', 'All Times', f"{df['Low_Held_Pct'].mean():.1f}%"],
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Time</b>', '<b>Value</b>'],
                fill_color='#506784',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[[row[i] for row in summary_data[1:]] for i in range(3)],
                fill_color='#F0F0F0',
                align='left',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Cut-off Time", row=1, col=1, tickangle=-45)
    fig.update_xaxes(title_text="Cut-off Time", row=1, col=2, tickangle=-45)
    fig.update_xaxes(title_text="Cut-off Time", row=2, col=1, tickangle=-45)
    
    fig.update_yaxes(title_text="Probability (%)", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Probability (%)", row=1, col=2, range=[0, 100])
    fig.update_yaxes(title_text="Probability (%)", row=2, col=1, range=[0, 100])
    
    fig.update_layout(
        title_text="<b>CL Futures Cut-off Time Analysis</b><br>" +
                   "<sub>How often do highs/lows hold from cut-off time until session end (15:55)?</sub>",
        title_font_size=20,
        height=900,
        showlegend=True,
        hovermode='closest'
    )
    
    # Save interactive HTML
    output_file = 'cutoff_analysis_charts.html'
    fig.write_html(output_file)
    print(f"\n✓ Interactive charts saved to: {output_file}")
    print("  Open this file in a web browser to explore the visualizations")
    
    return fig


def create_single_chart(csv_path='cutoff_analysis.csv'):
    """
    Create a simple, focused chart for quick analysis.
    """
    df = pd.read_csv(csv_path)
    
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=df['Cutoff_Time'],
        y=df['High_Held_Pct'],
        mode='lines+markers',
        name='High Held',
        line=dict(color='#FF6B6B', width=4),
        marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>High Held: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Cutoff_Time'],
        y=df['Low_Held_Pct'],
        mode='lines+markers',
        name='Low Held',
        line=dict(color='#4ECDC4', width=4),
        marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Low Held: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Cutoff_Time'],
        y=df['Both_Held_Pct'],
        mode='lines+markers',
        name='Both Held',
        line=dict(color='#95E1D3', width=4, dash='dash'),
        marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Both Held: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>Probability that Highs/Lows Hold Until Session End</b><br>' +
              '<sub>Based on cut-off time when extreme was established</sub>',
        xaxis_title='Cut-off Time',
        yaxis_title='Hold Probability (%)',
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        yaxis=dict(range=[0, 100])
    )
    
    fig.update_xaxes(tickangle=-45)
    
    output_file = 'cutoff_simple_chart.html'
    fig.write_html(output_file)
    print(f"✓ Simple chart saved to: {output_file}")
    
    return fig


def main():
    print("Creating visualizations from cut-off analysis...\n")
    
    # Create comprehensive dashboard
    create_visualizations()
    
    # Create simple chart
    create_single_chart()
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("\nGenerated files:")
    print("  1. cutoff_analysis_charts.html - Comprehensive 4-panel dashboard")
    print("  2. cutoff_simple_chart.html - Simple focused chart")
    print("\nOpen these HTML files in your web browser to view interactive charts.")


if __name__ == '__main__':
    main()
