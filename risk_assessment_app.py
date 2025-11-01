"""
Risk Assessment Tool - Streamlit Application
Probabilistic Risk Analysis with Monte Carlo Simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Risk Assessment Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎯 Risk Assessment Tool - Monte Carlo Simulation</p>', unsafe_allow_html=True)

# Helper functions
def parse_currency(value):
    """Parse currency string to float"""
    if pd.isna(value) or value == '' or value == '0 CHF':
        return 0.0
    try:
        # Remove CHF, commas, and spaces
        clean_value = str(value).replace('CHF', '').replace(',', '').replace(' ', '').strip()
        # Handle quotes
        clean_value = clean_value.replace('"', '')
        return float(clean_value)
    except:
        return 0.0

def parse_percentage(value):
    """Parse percentage string to float"""
    if pd.isna(value) or value == '':
        return 0.0
    try:
        clean_value = str(value).replace('%', '').strip()
        return float(clean_value) / 100
    except:
        return 0.0

@st.cache_data
def load_risk_data(file_path):
    """Load and process risk register data"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse currency columns
    currency_cols = ['Initial risk', 'Likely Initial Risk', 'Residual risk', 
                     'Likely Residual Risk', 'Cost of Measures']
    for col in currency_cols:
        if col in df.columns:
            df[col + '_Value'] = df[col].apply(parse_currency)
    
    # Parse likelihood columns
    df['Initial_Likelihood'] = df['Likelihood'].apply(parse_percentage)
    
    # Handle the second Likelihood column (for residual)
    likelihood_cols = [col for col in df.columns if 'Likelihood' in col]
    if len(likelihood_cols) > 1:
        df['Residual_Likelihood'] = df[likelihood_cols[1]].apply(parse_percentage)
    
    # Parse schedule impact
    df['Schedule_Impact'] = df['Schedule Impact?'].str.strip().str.lower().isin(['yes', 'yes '])
    
    return df

def run_monte_carlo(df, n_simulations=10000, risk_type='initial', random_numbers=None):
    """
    Run Monte Carlo simulation for risk portfolio

    Parameters:
    - df: DataFrame with risk data
    - n_simulations: Number of Monte Carlo iterations
    - risk_type: 'initial' or 'residual'
    - random_numbers: Optional pre-generated random numbers for Common Random Numbers technique
                     Shape should be (n_simulations, len(df))
    """
    if risk_type == 'initial':
        impact_col = 'Initial risk_Value'
        likelihood_col = 'Initial_Likelihood'
    else:
        impact_col = 'Residual risk_Value'
        likelihood_col = 'Residual_Likelihood'

    # Prepare data
    impacts = df[impact_col].values
    likelihoods = df[likelihood_col].values

    # Generate or use provided random numbers
    if random_numbers is None:
        random_numbers = np.random.random((n_simulations, len(df)))

    # Monte Carlo simulation
    results = np.zeros(n_simulations)
    risk_occurrences = np.zeros((n_simulations, len(df)))

    for i in range(n_simulations):
        # For each risk, determine if it occurs (Bernoulli trial)
        # Using the same random numbers ensures variance reduction
        occurred = random_numbers[i] < likelihoods
        risk_occurrences[i] = occurred

        # Sum the impacts of risks that occurred
        results[i] = np.sum(impacts * occurred)

    return results, risk_occurrences

def calculate_statistics(results):
    """Calculate statistical measures from Monte Carlo results"""
    return {
        'mean': np.mean(results),
        'median': np.median(results),
        'std': np.std(results),
        'p50': np.percentile(results, 50),
        'p80': np.percentile(results, 80),
        'p95': np.percentile(results, 95),
        'min': np.min(results),
        'max': np.max(results)
    }

def create_risk_heatmap(df, risk_type='initial'):
    """Create risk heatmap showing concentration of risks in each cell"""
    if risk_type == 'initial':
        impact_col = 'Initial risk_Value'
        likelihood_col = 'Initial_Likelihood'
        title = 'Initial Risk Heatmap'
    else:
        impact_col = 'Residual risk_Value'
        likelihood_col = 'Residual_Likelihood'
        title = 'Residual Risk Heatmap'
    
    # Create categories
    impact_bins = [0, 100000, 1000000, 10000000, 100000000, np.inf]
    impact_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    likelihood_bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    likelihood_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    # Categorize risks (use absolute values for impact to handle opportunities)
    df_temp = df.copy()
    df_temp['Impact_Cat'] = pd.cut(np.abs(df_temp[impact_col]), bins=impact_bins, labels=impact_labels)
    df_temp['Likelihood_Cat'] = pd.cut(df_temp[likelihood_col], bins=likelihood_bins, labels=likelihood_labels)
    
    # Create heatmap data
    heatmap_data = []
    for imp_cat in impact_labels:
        row_data = []
        for lik_cat in likelihood_labels:
            # Count risks in this cell
            mask = (df_temp['Impact_Cat'] == imp_cat) & (df_temp['Likelihood_Cat'] == lik_cat)
            count = mask.sum()
            
            # Calculate total value in this cell
            total_value = (df_temp.loc[mask, impact_col] * df_temp.loc[mask, likelihood_col]).sum()
            
            # Get risk IDs in this cell
            risk_ids = df_temp.loc[mask, 'Risk ID'].tolist()
            
            row_data.append({
                'count': count,
                'value': total_value,
                'risk_ids': risk_ids
            })
        heatmap_data.append(row_data)
    
    # Prepare data for plotly
    z_counts = [[cell['count'] for cell in row] for row in heatmap_data]
    z_values = [[cell['value']/1e6 for cell in row] for row in heatmap_data]  # Convert to millions
    
    # Create hover text
    hover_text = []
    for i, imp_cat in enumerate(impact_labels):
        row_text = []
        for j, lik_cat in enumerate(likelihood_labels):
            cell = heatmap_data[i][j]
            text = f"<b>{imp_cat} Impact / {lik_cat} Likelihood</b><br>"
            text += f"Number of Risks: {cell['count']}<br>"
            text += f"Total Expected Value: {cell['value']/1e6:.2f}M CHF<br>"
            if cell['risk_ids']:
                text += f"Risk IDs: {', '.join(map(str, cell['risk_ids']))}"
            row_text.append(text)
        hover_text.append(row_text)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_counts,
        x=likelihood_labels,
        y=impact_labels,
        customdata=z_values,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        colorscale='Reds',
        colorbar=dict(title="Number<br>of Risks"),
        text=[[f"{count}<br>{value:.1f}M" for count, value in zip(count_row, value_row)] 
              for count_row, value_row in zip(z_counts, z_values)],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    # Add grid lines and styling
    fig.update_xaxes(side="bottom", title="Likelihood →")
    fig.update_yaxes(title="← Impact")
    
    fig.update_layout(
        title=f'{title}<br><sub>Cell shows: Risk Count / Total Expected Value (M CHF)</sub>',
        height=600,
        xaxis=dict(showgrid=True, gridcolor='white', gridwidth=2),
        yaxis=dict(showgrid=True, gridcolor='white', gridwidth=2)
    )
    
    return fig

def create_risk_bubble_chart(df, risk_type='initial'):
    """Create bubble chart with risk size representing expected value"""
    if risk_type == 'initial':
        impact_col = 'Initial risk_Value'
        likelihood_col = 'Initial_Likelihood'
        title = 'Initial Risk Bubble Chart'
    else:
        impact_col = 'Residual risk_Value'
        likelihood_col = 'Residual_Likelihood'
        title = 'Residual Risk Bubble Chart'
    
    df_plot = df.copy()
    
    # Calculate expected value and use for bubble size
    df_plot['Expected_Value'] = np.abs(df_plot[impact_col] * df_plot[likelihood_col])
    df_plot['Risk_Type'] = df_plot[impact_col].apply(lambda x: 'Opportunity' if x < 0 else 'Risk')
    
    # Create risk score for coloring
    df_plot['Risk_Score'] = df_plot[impact_col] * df_plot[likelihood_col]
    
    # Scale bubble sizes (cube root for better visualization)
    df_plot['Bubble_Size'] = np.power(df_plot['Expected_Value'], 1/3) / 100
    
    fig = px.scatter(df_plot,
                     x=likelihood_col,
                     y=impact_col,
                     size='Bubble_Size',
                     color='Risk_Score',
                     hover_data=['Risk ID', 'Risk Description', 'Expected_Value', 'Risk_Type'],
                     labels={
                         likelihood_col: 'Likelihood',
                         impact_col: 'Impact (CHF)',
                         'Expected_Value': 'Expected Value (CHF)'
                     },
                     color_continuous_scale='RdYlGn_r',
                     title=f'{title}<br><sub>Bubble size = Expected Value | Color = Risk Score</sub>')
    
    # Add quadrant lines
    median_likelihood = df_plot[likelihood_col].median()
    median_impact = df_plot[impact_col].median()
    
    fig.add_vline(x=median_likelihood, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Median Likelihood", annotation_position="top")
    fig.add_hline(y=median_impact, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Median Impact", annotation_position="right")
    
    # Add zero line for opportunities
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    
    # Add quadrant labels
    max_likelihood = df_plot[likelihood_col].max()
    max_impact = df_plot[impact_col].max()
    min_impact = df_plot[impact_col].min()
    
    fig.add_annotation(x=median_likelihood/2, y=max_impact*0.9,
                      text="Low Likelihood<br>High Impact",
                      showarrow=False, font=dict(size=10, color="gray"))
    
    fig.add_annotation(x=median_likelihood + (max_likelihood-median_likelihood)/2, y=max_impact*0.9,
                      text="High Likelihood<br>High Impact",
                      showarrow=False, font=dict(size=10, color="red"))
    
    fig.add_annotation(x=median_likelihood/2, y=min_impact*0.5,
                      text="Low Likelihood<br>Low Impact",
                      showarrow=False, font=dict(size=10, color="green"))
    
    fig.add_annotation(x=median_likelihood + (max_likelihood-median_likelihood)/2, y=min_impact*0.5,
                      text="High Likelihood<br>Low Impact",
                      showarrow=False, font=dict(size=10, color="orange"))
    
    fig.update_layout(height=700, showlegend=True)
    
    return fig

def create_risk_matrix(df, risk_type='initial'):
    """Create interactive risk matrix"""
    if risk_type == 'initial':
        impact_col = 'Initial risk_Value'
        likelihood_col = 'Initial_Likelihood'
        title = 'Initial Risk Matrix'
    else:
        impact_col = 'Residual risk_Value'
        likelihood_col = 'Residual_Likelihood'
        title = 'Residual Risk Matrix'
    
    # Create a copy to avoid modifying original dataframe
    df_plot = df.copy()
    
    # Create categories for impact (using absolute values for categorization)
    df_plot['Impact_Category'] = pd.cut(np.abs(df_plot[impact_col]), 
                                    bins=[0, 100000, 1000000, 10000000, np.inf],
                                    labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create categories for likelihood
    df_plot['Likelihood_Category'] = pd.cut(df_plot[likelihood_col],
                                        bins=[0, 0.2, 0.4, 0.6, 1.0],
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create risk score
    df_plot['Risk_Score'] = df_plot[impact_col] * df_plot[likelihood_col]
    
    # Create size for markers (must be positive, use absolute value + small offset)
    df_plot['Marker_Size'] = np.abs(df_plot['Risk_Score']) + 1000  # Add offset to ensure visibility
    
    # Create hover text showing if risk is negative (opportunity)
    df_plot['Risk_Type'] = df_plot[impact_col].apply(lambda x: 'Opportunity' if x < 0 else 'Risk')
    
    fig = px.scatter(df_plot, 
                     x=likelihood_col, 
                     y=impact_col,
                     size='Marker_Size',
                     color='Risk_Score',
                     hover_data=['Risk ID', 'Risk Description', 'Risk_Type'],
                     labels={likelihood_col: 'Likelihood', 
                            impact_col: 'Impact (CHF)'},
                     title=title,
                     color_continuous_scale='RdYlGn_r')
    
    # Add grid lines for zones
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)  # Zero line
    fig.add_hline(y=1000000, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=10000000, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=-1000000, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.2, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.4, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.6, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(height=600, showlegend=True)
    
    return fig

def create_cdf_plot(results, stats, risk_type):
    """Create cumulative distribution function plot"""
    sorted_results = np.sort(results)
    cdf = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
    
    fig = go.Figure()
    
    # CDF line
    fig.add_trace(go.Scatter(
        x=sorted_results,
        y=cdf * 100,
        mode='lines',
        name='CDF',
        line=dict(color='blue', width=2)
    ))
    
    # Add percentile lines
    percentiles = [('P50', stats['p50'], 'green'), 
                   ('P80', stats['p80'], 'orange'), 
                   ('P95', stats['p95'], 'red')]
    
    for label, value, color in percentiles:
        fig.add_vline(x=value, line_dash="dash", line_color=color,
                     annotation_text=f"{label}: {value/1e6:.2f}M CHF",
                     annotation_position="top")
    
    fig.update_layout(
        title=f'Cumulative Distribution Function - {risk_type.title()} Risk Exposure',
        xaxis_title='Total Risk Exposure (CHF)',
        yaxis_title='Cumulative Probability (%)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_box_plot(initial_results, residual_results):
    """Create box plot comparing initial vs residual risk"""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=initial_results / 1e6,
        name='Initial Risk',
        marker_color='indianred'
    ))
    
    fig.add_trace(go.Box(
        y=residual_results / 1e6,
        name='Residual Risk',
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        title='Risk Exposure Distribution: Initial vs Residual',
        yaxis_title='Total Risk Exposure (Million CHF)',
        height=500,
        showlegend=True
    )
    
    return fig

def create_histogram(results, stats, risk_type):
    """Create histogram with statistics overlay"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=results / 1e6,
        nbinsx=50,
        name='Frequency',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add mean line
    fig.add_vline(x=stats['mean']/1e6, line_dash="dash", line_color="blue",
                 annotation_text=f"Mean: {stats['mean']/1e6:.2f}M",
                 annotation_position="top right")
    
    # Add median line
    fig.add_vline(x=stats['median']/1e6, line_dash="dash", line_color="green",
                 annotation_text=f"Median: {stats['median']/1e6:.2f}M",
                 annotation_position="top left")
    
    fig.update_layout(
        title=f'Distribution of Total Risk Exposure - {risk_type.title()}',
        xaxis_title='Total Risk Exposure (Million CHF)',
        yaxis_title='Frequency',
        height=500,
        showlegend=True
    )
    
    return fig

def calculate_mitigation_roi(df):
    """Calculate ROI for risk mitigation measures"""
    # Calculate initial expected value
    df['Initial_EV'] = df['Initial risk_Value'] * df['Initial_Likelihood']
    
    # Calculate residual expected value
    df['Residual_EV'] = df['Residual risk_Value'] * df['Residual_Likelihood']
    
    # Calculate risk reduction
    df['Risk_Reduction'] = df['Initial_EV'] - df['Residual_EV']
    
    # Calculate ROI (only for risks with mitigation cost > 0)
    df['ROI'] = np.where(
        df['Cost of Measures_Value'] > 0,
        (df['Risk_Reduction'] - df['Cost of Measures_Value']) / df['Cost of Measures_Value'] * 100,
        0
    )
    
    # Calculate benefit-cost ratio
    df['BC_Ratio'] = np.where(
        df['Cost of Measures_Value'] > 0,
        df['Risk_Reduction'] / df['Cost of Measures_Value'],
        0
    )
    
    return df

def create_tornado_chart(df, top_n=15):
    """Create tornado chart for sensitivity analysis"""
    # Calculate risk range (impact * likelihood) - use absolute value for sorting
    df_temp = df.copy()
    df_temp['Risk_Range'] = df_temp['Initial risk_Value'] * df_temp['Initial_Likelihood']
    df_temp['Risk_Range_Abs'] = np.abs(df_temp['Risk_Range'])
    
    # Sort by absolute risk range and take top N
    df_sorted = df_temp.nlargest(top_n, 'Risk_Range_Abs')
    
    # Sort by actual risk range for display
    df_sorted = df_sorted.sort_values('Risk_Range')
    
    # Determine color based on whether it's positive or negative
    colors = ['green' if x < 0 else 'red' for x in df_sorted['Risk_Range']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['Risk Description'],
        x=df_sorted['Risk_Range'] / 1e6,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=df_sorted['Risk_Range'].apply(lambda x: f'{x/1e6:.2f}M'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Expected Value: %{x:.2f}M CHF<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Risks by Expected Value (Tornado Chart)<br><sub>Red = Risk (Cost), Green = Opportunity (Benefit)</sub>',
        xaxis_title='Expected Value (Million CHF)',
        yaxis_title='',
        height=600,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    
    return fig

def perform_sensitivity_analysis(df, n_simulations=10000):
    """
    Enhanced sensitivity analysis with variance contribution
    Shows which risks drive the most uncertainty in total exposure

    Uses Common Random Numbers (CRN) technique to reduce sampling noise
    and ensure accurate variance decomposition (no negative contributions)
    """
    # Generate random numbers ONCE for all simulations (Common Random Numbers)
    # This eliminates Monte Carlo sampling noise in variance comparisons
    random_numbers = np.random.random((n_simulations, len(df)))

    # Baseline simulation using the common random numbers
    baseline_results, _ = run_monte_carlo(df, n_simulations, 'initial', random_numbers)
    baseline_variance = np.var(baseline_results)
    baseline_mean = np.mean(baseline_results)

    # Calculate contribution of each risk to total variance
    risk_contributions = []

    for idx, risk in df.iterrows():
        # Create modified dataframe with this risk set to zero probability
        df_modified = df.copy()
        df_modified.loc[idx, 'Initial_Likelihood'] = 0

        # Run simulation without this risk using SAME random numbers
        # This is the key: same random draws, only the likelihood changes
        modified_results, _ = run_monte_carlo(df_modified, n_simulations, 'initial', random_numbers)
        modified_variance = np.var(modified_results)
        modified_mean = np.mean(modified_results)

        # Calculate variance reduction and mean impact
        # With CRN, this should always be >= 0 (or very close due to numerical precision)
        variance_contribution = baseline_variance - modified_variance
        mean_contribution = baseline_mean - modified_mean

        risk_contributions.append({
            'Risk ID': risk['Risk ID'],
            'Risk Description': risk['Risk Description'],
            'Variance Contribution': variance_contribution,
            'Variance %': (variance_contribution / baseline_variance * 100) if baseline_variance > 0 else 0,
            'Mean Contribution': mean_contribution,
            'Expected Value': risk['Initial risk_Value'] * risk['Initial_Likelihood']
        })

    # Create DataFrame and sort by variance contribution
    sensitivity_df = pd.DataFrame(risk_contributions)
    sensitivity_df = sensitivity_df.sort_values('Variance Contribution', ascending=False)

    # Calculate cumulative percentage (Pareto)
    sensitivity_df['Cumulative %'] = sensitivity_df['Variance %'].cumsum()

    return sensitivity_df

def create_enhanced_tornado_chart(sensitivity_df, top_n=15):
    """Create enhanced tornado chart showing variance contribution"""
    # Take top N risks
    df_plot = sensitivity_df.head(top_n).copy()
    
    # Sort by variance contribution for display
    df_plot = df_plot.sort_values('Variance Contribution')
    
    # Create figure with secondary y-axis for cumulative %
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": False}]]
    )
    
    # Add variance contribution bars
    fig.add_trace(
        go.Bar(
            y=df_plot['Risk Description'],
            x=df_plot['Variance %'],
            orientation='h',
            name='Variance Contribution',
            marker=dict(
                color=df_plot['Variance %'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Variance %")
            ),
            text=df_plot['Variance %'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Variance Contribution: %{x:.2f}%<br>Mean Impact: %{customdata:.2f}M CHF<extra></extra>',
            customdata=df_plot['Mean Contribution'] / 1e6
        )
    )
    
    fig.update_layout(
        title=f'Sensitivity Analysis - Top {top_n} Risks by Variance Contribution<br><sub>Shows which risks drive the most uncertainty in total exposure</sub>',
        xaxis_title='Contribution to Total Variance (%)',
        yaxis_title='',
        height=600,
        showlegend=False
    )
    
    return fig

def create_pareto_chart(sensitivity_df, top_n=20):
    """Create Pareto chart showing cumulative variance contribution"""
    df_plot = sensitivity_df.head(top_n).copy()

    # Prepare hover data with full risk description
    df_plot['hover_text'] = df_plot.apply(
        lambda row: f"<b>{row['Risk ID']}</b>: {row['Risk Description']}", axis=1
    )

    # Create figure WITH secondary y-axis (bars auto-scaled, cumulative 0-100%)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bars for individual variance contribution on PRIMARY y-axis (auto-scaled)
    fig.add_trace(
        go.Bar(
            x=df_plot['Risk ID'],  # Use Risk IDs only for clean x-axis
            y=df_plot['Variance %'],
            name='Individual Variance %',
            marker_color='#E74C3C',  # Vibrant red
            customdata=df_plot['hover_text'],
            hovertemplate='%{customdata}<br>Individual Variance: %{y:.2f}%<extra></extra>'
        ),
        secondary_y=False
    )

    # Add line for cumulative percentage on SECONDARY y-axis (0-100% scale)
    fig.add_trace(
        go.Scatter(
            x=df_plot['Risk ID'],  # Use Risk IDs only
            y=df_plot['Cumulative %'],
            name='Cumulative %',
            mode='lines+markers',
            line=dict(color='#2C3E50', width=4),  # Dark slate, thick line
            marker=dict(size=10, symbol='diamond', color='#2C3E50'),
            customdata=df_plot['hover_text'],
            hovertemplate='%{customdata}<br>Cumulative: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    # Add 80% reference line (on secondary y-axis)
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#27AE60",  # Bright green
        line_width=4,  # Thicker for prominence
        annotation_text="80% Threshold",
        annotation_position="right",
        annotation=dict(
            font=dict(size=12, color="#27AE60"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#27AE60",
            borderwidth=2,
            borderpad=4
        ),
        secondary_y=True
    )

    # Update x-axis with 45-degree rotation
    fig.update_xaxes(
        title_text="Risk ID",
        tickangle=-45,  # Rotate labels 45 degrees
        tickfont=dict(size=11),
        title_font=dict(size=12)
    )

    # Update PRIMARY y-axis (left) - Individual variance (auto-scaled)
    fig.update_yaxes(
        title_text="Individual Variance Contribution (%)",
        secondary_y=False,
        showgrid=True,  # Subtle gridlines
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        tickfont=dict(size=11),
        title_font=dict(size=12)
    )

    # Update SECONDARY y-axis (right) - Cumulative % (0-100% fixed scale)
    fig.update_yaxes(
        title_text="Cumulative Variance Contribution (%)",
        secondary_y=True,
        range=[0, 100],  # Fixed 0-100% scale for cumulative
        showgrid=False,  # No gridlines on secondary axis
        tickfont=dict(size=11),
        title_font=dict(size=12)
    )

    # Update layout with proper dimensions and margins
    fig.update_layout(
        title={
            'text': 'Pareto Analysis - Risk Variance Contribution<br><sub>Identify the vital few risks driving most uncertainty (80/20 rule)</sub>',
            'font': {'size': 16}
        },
        width=1400,  # Increased width for better spacing
        height=700,  # Good height for readability
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=80, r=80, t=120, b=120),  # Proper margins for rotated labels
        font=dict(size=11),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

# Main application
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # File upload or use default
    uploaded_file = st.sidebar.file_uploader("Upload Risk Register (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = load_risk_data(uploaded_file)
    else:
        # Use default file from same directory as app
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_file = os.path.join(script_dir, 'risk_register.csv')
        
        if os.path.exists(default_file):
            df = load_risk_data(default_file)
            st.sidebar.info("📁 Using risk_register.csv from application directory")
        else:
            st.sidebar.error("⚠️ Default risk register not found. Please upload your CSV file.")
            st.error("### 📁 No Risk Register Found")
            st.write("Please upload your risk register CSV file using the sidebar, or place a file named `risk_register.csv` in the same directory as the application.")
            st.stop()
    
    # Simulation parameters
    st.sidebar.subheader("Monte Carlo Parameters")
    n_simulations = st.sidebar.slider("Number of Simulations", 
                                      min_value=1000, 
                                      max_value=100000, 
                                      value=10000, 
                                      step=1000)
    
    confidence_level = st.sidebar.selectbox("Confidence Level", 
                                           ["P50", "P80", "P95"], 
                                           index=2)
    
    # Run simulation button
    if st.sidebar.button("🎲 Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Run simulations
            initial_results, initial_occurrences = run_monte_carlo(df, n_simulations, 'initial')
            residual_results, residual_occurrences = run_monte_carlo(df, n_simulations, 'residual')
            
            # Calculate statistics
            initial_stats = calculate_statistics(initial_results)
            residual_stats = calculate_statistics(residual_results)
            
            # Calculate mitigation ROI
            df_with_roi = calculate_mitigation_roi(df)
            
            # Perform enhanced sensitivity analysis
            st.write("Performing sensitivity analysis...")
            sensitivity_df = perform_sensitivity_analysis(df, n_simulations)
            
            # Store in session state
            st.session_state['initial_results'] = initial_results
            st.session_state['residual_results'] = residual_results
            st.session_state['initial_stats'] = initial_stats
            st.session_state['residual_stats'] = residual_stats
            st.session_state['df'] = df
            st.session_state['df_with_roi'] = df_with_roi
            st.session_state['sensitivity_df'] = sensitivity_df
            st.session_state['simulation_run'] = True
            
        st.sidebar.success("✅ Simulation completed!")
    
    # Display results if simulation has been run
    if st.session_state.get('simulation_run', False):
        initial_results = st.session_state['initial_results']
        residual_results = st.session_state['residual_results']
        initial_stats = st.session_state['initial_stats']
        residual_stats = st.session_state['residual_stats']
        df = st.session_state['df']
        df_with_roi = st.session_state['df_with_roi']
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Dashboard", 
            "📈 Risk Matrix & Heatmap", 
            "🔍 Sensitivity Analysis",
            "🎲 Monte Carlo Results",
            "💰 Cost-Benefit Analysis",
            "📋 Risk Register",
            "📥 Export"
        ])
        
        with tab1:
            st.header("Risk Portfolio Dashboard")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Risks", len(df))
                st.metric("Risks with Schedule Impact", df['Schedule_Impact'].sum())
            
            with col2:
                st.metric("Initial Risk (P95)", f"{initial_stats['p95']/1e6:.2f}M CHF")
                st.metric("Initial Risk (Mean)", f"{initial_stats['mean']/1e6:.2f}M CHF")
            
            with col3:
                st.metric("Residual Risk (P95)", f"{residual_stats['p95']/1e6:.2f}M CHF")
                st.metric("Residual Risk (Mean)", f"{residual_stats['mean']/1e6:.2f}M CHF")
            
            with col4:
                risk_reduction = ((initial_stats['p95'] - residual_stats['p95']) / initial_stats['p95'] * 100)
                st.metric("Risk Reduction (P95)", f"{risk_reduction:.1f}%")
                total_mitigation_cost = df['Cost of Measures_Value'].sum()
                st.metric("Total Mitigation Cost", f"{total_mitigation_cost/1e6:.2f}M CHF")
            
            st.markdown("---")
            
            # Tornado chart
            st.subheader("Top Risks by Expected Value")
            tornado_fig = create_tornado_chart(df_with_roi, top_n=15)
            st.plotly_chart(tornado_fig, use_container_width=True)
            
            # Statistics comparison table
            st.subheader("Statistical Summary")
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median (P50)', 'P80', 'P95', 'Std Dev', 'Min', 'Max'],
                'Initial Risk (M CHF)': [
                    f"{initial_stats['mean']/1e6:.2f}",
                    f"{initial_stats['p50']/1e6:.2f}",
                    f"{initial_stats['p80']/1e6:.2f}",
                    f"{initial_stats['p95']/1e6:.2f}",
                    f"{initial_stats['std']/1e6:.2f}",
                    f"{initial_stats['min']/1e6:.2f}",
                    f"{initial_stats['max']/1e6:.2f}"
                ],
                'Residual Risk (M CHF)': [
                    f"{residual_stats['mean']/1e6:.2f}",
                    f"{residual_stats['p50']/1e6:.2f}",
                    f"{residual_stats['p80']/1e6:.2f}",
                    f"{residual_stats['p95']/1e6:.2f}",
                    f"{residual_stats['std']/1e6:.2f}",
                    f"{residual_stats['min']/1e6:.2f}",
                    f"{residual_stats['max']/1e6:.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.header("Risk Visualization - Matrix, Heatmap & Bubble Charts")
            
            # Add view selector
            view_type = st.radio(
                "Select Visualization Type:",
                ["Risk Matrix (Scatter)", "Risk Heatmap (Grid)", "Risk Bubble Chart"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            if view_type == "Risk Matrix (Scatter)":
                with col1:
                    st.subheader("Initial Risk Matrix")
                    initial_matrix = create_risk_matrix(df, 'initial')
                    st.plotly_chart(initial_matrix, use_container_width=True)
                
                with col2:
                    st.subheader("Residual Risk Matrix")
                    residual_matrix = create_risk_matrix(df, 'residual')
                    st.plotly_chart(residual_matrix, use_container_width=True)
            
            elif view_type == "Risk Heatmap (Grid)":
                with col1:
                    st.subheader("Initial Risk Heatmap")
                    st.info("📊 Shows concentration and total value of risks in each likelihood-impact cell")
                    initial_heatmap = create_risk_heatmap(df, 'initial')
                    st.plotly_chart(initial_heatmap, use_container_width=True)
                
                with col2:
                    st.subheader("Residual Risk Heatmap")
                    st.info("📊 Shows risk concentration after mitigation measures")
                    residual_heatmap = create_risk_heatmap(df, 'residual')
                    st.plotly_chart(residual_heatmap, use_container_width=True)
                
                # Add summary statistics
                st.markdown("---")
                st.subheader("Heatmap Insights")
                
                # Calculate high-risk quadrant
                high_impact_high_prob_initial = df[(np.abs(df['Initial risk_Value']) > 1e6) & 
                                                    (df['Initial_Likelihood'] > 0.25)].shape[0]
                high_impact_high_prob_residual = df[(np.abs(df['Residual risk_Value']) > 1e6) & 
                                                     (df['Residual_Likelihood'] > 0.25)].shape[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High-Risk Quadrant (Initial)", 
                             high_impact_high_prob_initial,
                             help="Risks with impact >1M CHF and likelihood >25%")
                with col2:
                    st.metric("High-Risk Quadrant (Residual)", 
                             high_impact_high_prob_residual,
                             delta=high_impact_high_prob_residual - high_impact_high_prob_initial)
                with col3:
                    reduction = ((high_impact_high_prob_initial - high_impact_high_prob_residual) / 
                                high_impact_high_prob_initial * 100) if high_impact_high_prob_initial > 0 else 0
                    st.metric("High-Risk Reduction", f"{reduction:.1f}%")
            
            else:  # Bubble Chart
                with col1:
                    st.subheader("Initial Risk Bubble Chart")
                    st.info("📊 Bubble size represents expected value of each risk")
                    initial_bubble = create_risk_bubble_chart(df, 'initial')
                    st.plotly_chart(initial_bubble, use_container_width=True)
                
                with col2:
                    st.subheader("Residual Risk Bubble Chart")
                    st.info("📊 Shows risk positioning after mitigation")
                    residual_bubble = create_risk_bubble_chart(df, 'residual')
                    st.plotly_chart(residual_bubble, use_container_width=True)
        
        with tab3:
            st.header("Enhanced Sensitivity Analysis")
            
            st.info("""
            🔍 **Sensitivity Analysis** identifies which risks contribute most to overall portfolio uncertainty.
            This helps prioritize risk management efforts on the risks that matter most.
            """)
            
            # Get sensitivity data
            sensitivity_df = st.session_state['sensitivity_df']
            
            # Key insights
            st.subheader("Key Findings")
            
            col1, col2, col3 = st.columns(3)
            
            # Top risk by variance
            top_risk = sensitivity_df.iloc[0]
            with col1:
                st.metric(
                    "Top Risk Driver",
                    f"Risk {top_risk['Risk ID']}",
                    f"{top_risk['Variance %']:.1f}% of variance"
                )
            
            # Number of risks for 80% variance
            risks_80 = (sensitivity_df['Cumulative %'] <= 80).sum()
            with col2:
                st.metric(
                    "Risks Driving 80% Variance",
                    f"{risks_80} risks",
                    f"{risks_80/len(df)*100:.1f}% of total"
                )
            
            # Total variance of top 5
            top5_variance = sensitivity_df.head(5)['Variance %'].sum()
            with col3:
                st.metric(
                    "Top 5 Risks Contribution",
                    f"{top5_variance:.1f}%",
                    "of total variance"
                )
            
            st.markdown("---")
            
            # Enhanced Tornado Chart
            st.subheader("Variance Contribution (Tornado Chart)")
            tornado_enhanced = create_enhanced_tornado_chart(sensitivity_df, top_n=15)
            st.plotly_chart(tornado_enhanced, use_container_width=True)
            
            st.markdown("---")
            
            # Pareto Chart
            st.subheader("Pareto Analysis (80/20 Rule)")
            st.write("""
            The **Pareto principle** suggests that roughly 80% of effects come from 20% of causes.
            This chart identifies the "vital few" risks that drive most of the uncertainty.
            """)
            pareto_chart = create_pareto_chart(sensitivity_df, top_n=20)
            st.plotly_chart(pareto_chart, use_container_width=True)
            
            # Find 80% threshold
            risks_for_80 = sensitivity_df[sensitivity_df['Cumulative %'] <= 80]
            st.success(f"""
            📊 **Pareto Insight**: {len(risks_for_80)} risks ({len(risks_for_80)/len(df)*100:.1f}% of total) 
            account for 80% of the portfolio variance.
            
            **Recommendation**: Focus intensive risk management efforts on these {len(risks_for_80)} critical risks.
            """)
            
            st.markdown("---")
            
            # Detailed sensitivity table
            st.subheader("Detailed Sensitivity Analysis")
            
            # Prepare display dataframe
            display_sensitivity = sensitivity_df[['Risk ID', 'Risk Description', 'Variance %', 
                                                   'Cumulative %', 'Mean Contribution', 'Expected Value']].copy()
            display_sensitivity['Mean Contribution (M CHF)'] = (display_sensitivity['Mean Contribution'] / 1e6).round(2)
            display_sensitivity['Expected Value (M CHF)'] = (display_sensitivity['Expected Value'] / 1e6).round(2)
            display_sensitivity['Variance %'] = display_sensitivity['Variance %'].round(2)
            display_sensitivity['Cumulative %'] = display_sensitivity['Cumulative %'].round(1)
            
            display_sensitivity = display_sensitivity[['Risk ID', 'Risk Description', 'Variance %', 
                                                       'Cumulative %', 'Mean Contribution (M CHF)', 
                                                       'Expected Value (M CHF)']]
            
            st.dataframe(
                display_sensitivity,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Export sensitivity analysis
            csv_sensitivity = display_sensitivity.to_csv(index=False)
            st.download_button(
                label="📥 Download Sensitivity Analysis (CSV)",
                data=csv_sensitivity,
                file_name=f"sensitivity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.header("Monte Carlo Simulation Results")
            
            st.info(f"📊 Performed {n_simulations:,} simulations to model probabilistic risk exposure")
            
            # Box plot
            st.subheader("Risk Exposure Variability")
            box_fig = create_box_plot(initial_results, residual_results)
            st.plotly_chart(box_fig, use_container_width=True)
            
            # Histograms
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Initial Risk Distribution")
                hist_initial = create_histogram(initial_results, initial_stats, 'initial')
                st.plotly_chart(hist_initial, use_container_width=True)
            
            with col2:
                st.subheader("Residual Risk Distribution")
                hist_residual = create_histogram(residual_results, residual_stats, 'residual')
                st.plotly_chart(hist_residual, use_container_width=True)
            
            # CDF plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Initial Risk CDF")
                cdf_initial = create_cdf_plot(initial_results, initial_stats, 'initial')
                st.plotly_chart(cdf_initial, use_container_width=True)
            
            with col2:
                st.subheader("Residual Risk CDF")
                cdf_residual = create_cdf_plot(residual_results, residual_stats, 'residual')
                st.plotly_chart(cdf_residual, use_container_width=True)
        
        with tab5:
            st.header("Mitigation Cost-Benefit Analysis")
            
            # Filter risks with mitigation measures
            df_with_measures = df_with_roi[df_with_roi['Cost of Measures_Value'] > 0].copy()
            
            if len(df_with_measures) > 0:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_reduction = df_with_measures['Risk_Reduction'].sum()
                    st.metric("Total Risk Reduction", f"{total_reduction/1e6:.2f}M CHF")
                
                with col2:
                    total_cost = df_with_measures['Cost of Measures_Value'].sum()
                    st.metric("Total Mitigation Cost", f"{total_cost/1e6:.2f}M CHF")
                
                with col3:
                    net_benefit = total_reduction - total_cost
                    st.metric("Net Benefit", f"{net_benefit/1e6:.2f}M CHF")
                
                st.markdown("---")
                
                # ROI chart
                st.subheader("Return on Investment by Risk")
                df_roi_sorted = df_with_measures.nlargest(20, 'ROI')
                
                fig_roi = px.bar(df_roi_sorted,
                                x='ROI',
                                y='Risk Description',
                                orientation='h',
                                title='Top 20 Mitigation Measures by ROI (%)',
                                color='ROI',
                                color_continuous_scale='RdYlGn',
                                labels={'ROI': 'ROI (%)'})
                
                fig_roi.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_roi, use_container_width=True)
                
                # Benefit-Cost Ratio chart
                st.subheader("Benefit-Cost Ratio Analysis")
                df_bc_sorted = df_with_measures.nlargest(20, 'BC_Ratio')
                
                fig_bc = px.scatter(df_bc_sorted,
                                   x='Cost of Measures_Value',
                                   y='Risk_Reduction',
                                   size='BC_Ratio',
                                   color='BC_Ratio',
                                   hover_data=['Risk ID', 'Risk Description'],
                                   title='Risk Reduction vs Mitigation Cost',
                                   labels={
                                       'Cost of Measures_Value': 'Mitigation Cost (CHF)',
                                       'Risk_Reduction': 'Risk Reduction (CHF)'
                                   },
                                   color_continuous_scale='Viridis')
                
                # Add break-even line
                max_val = max(df_bc_sorted['Cost of Measures_Value'].max(), 
                             df_bc_sorted['Risk_Reduction'].max())
                fig_bc.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                           mode='lines',
                                           name='Break-even line',
                                           line=dict(dash='dash', color='red')))
                
                st.plotly_chart(fig_bc, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Cost-Benefit Analysis")
                
                display_cols = ['Risk ID', 'Risk Description', 'Initial_EV', 'Residual_EV', 
                               'Risk_Reduction', 'Cost of Measures_Value', 'ROI', 'BC_Ratio']
                
                display_df = df_with_measures[display_cols].copy()
                display_df.columns = ['Risk ID', 'Risk Description', 'Initial EV (CHF)', 
                                     'Residual EV (CHF)', 'Risk Reduction (CHF)', 
                                     'Mitigation Cost (CHF)', 'ROI (%)', 'B/C Ratio']
                
                # Format numbers
                for col in ['Initial EV (CHF)', 'Residual EV (CHF)', 'Risk Reduction (CHF)', 'Mitigation Cost (CHF)']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
                
                display_df['ROI (%)'] = display_df['ROI (%)'].apply(lambda x: f"{x:.1f}")
                display_df['B/C Ratio'] = display_df['B/C Ratio'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No risks with mitigation measures found in the register.")
        
        with tab6:
            st.header("Risk Register")
            
            # Add filters
            st.subheader("Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                schedule_filter = st.multiselect("Schedule Impact", 
                                                options=[True, False],
                                                default=[True, False],
                                                format_func=lambda x: "Yes" if x else "No")
            
            with col2:
                risk_threshold = st.slider("Min Initial Risk (M CHF)", 
                                          min_value=0.0,
                                          max_value=float(df['Initial risk_Value'].max()/1e6),
                                          value=0.0)
            
            with col3:
                likelihood_threshold = st.slider("Min Likelihood (%)",
                                                min_value=0,
                                                max_value=100,
                                                value=0)
            
            # Apply filters
            filtered_df = df_with_roi[
                (df_with_roi['Schedule_Impact'].isin(schedule_filter)) &
                (df_with_roi['Initial risk_Value'] >= risk_threshold * 1e6) &
                (df_with_roi['Initial_Likelihood'] >= likelihood_threshold / 100)
            ].copy()
            
            st.write(f"Showing {len(filtered_df)} of {len(df)} risks")
            
            # Display table
            display_cols = ['Risk ID', 'Risk Description', 'Initial risk_Value', 'Initial_Likelihood',
                           'Residual risk_Value', 'Residual_Likelihood', 'Cost of Measures_Value',
                           'Schedule_Impact', 'ROI', 'BC_Ratio']
            
            display_df = filtered_df[display_cols].copy()
            display_df.columns = ['Risk ID', 'Description', 'Initial Risk (CHF)', 'Initial Likelihood',
                                 'Residual Risk (CHF)', 'Residual Likelihood', 'Mitigation Cost (CHF)',
                                 'Schedule Impact', 'ROI (%)', 'B/C Ratio']
            
            # Format
            display_df['Initial Risk (CHF)'] = display_df['Initial Risk (CHF)'].apply(lambda x: f"{x:,.0f}")
            display_df['Residual Risk (CHF)'] = display_df['Residual Risk (CHF)'].apply(lambda x: f"{x:,.0f}")
            display_df['Mitigation Cost (CHF)'] = display_df['Mitigation Cost (CHF)'].apply(lambda x: f"{x:,.0f}")
            display_df['Initial Likelihood'] = display_df['Initial Likelihood'].apply(lambda x: f"{x:.1%}")
            display_df['Residual Likelihood'] = display_df['Residual Likelihood'].apply(lambda x: f"{x:.1%}")
            display_df['Schedule Impact'] = display_df['Schedule Impact'].apply(lambda x: "Yes" if x else "No")
            display_df['ROI (%)'] = display_df['ROI (%)'].apply(lambda x: f"{x:.1f}")
            display_df['B/C Ratio'] = display_df['B/C Ratio'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
        
        with tab7:
            st.header("Export Results")
            
            st.write("Download your risk assessment results in various formats:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to Excel
                st.subheader("📊 Excel Export")
                
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary statistics
                    summary_data = {
                        'Metric': ['Mean', 'Median (P50)', 'P80', 'P95', 'Std Dev'],
                        'Initial Risk (CHF)': [
                            initial_stats['mean'],
                            initial_stats['p50'],
                            initial_stats['p80'],
                            initial_stats['p95'],
                            initial_stats['std']
                        ],
                        'Residual Risk (CHF)': [
                            residual_stats['mean'],
                            residual_stats['p50'],
                            residual_stats['p80'],
                            residual_stats['p95'],
                            residual_stats['std']
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Full risk register with ROI
                    df_with_roi.to_excel(writer, sheet_name='Risk Register', index=False)
                    
                    # Monte Carlo results sample
                    mc_results = pd.DataFrame({
                        'Simulation': range(1, len(initial_results) + 1),
                        'Initial Risk (CHF)': initial_results,
                        'Residual Risk (CHF)': residual_results
                    })
                    mc_results.to_excel(writer, sheet_name='Monte Carlo Results', index=False)
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Export to CSV
                st.subheader("📄 CSV Export")
                
                csv = df_with_roi.to_csv(index=False)
                
                st.download_button(
                    label="Download Risk Register (CSV)",
                    data=csv,
                    file_name=f"risk_register_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Export Monte Carlo results
                mc_csv = pd.DataFrame({
                    'Simulation': range(1, len(initial_results) + 1),
                    'Initial_Risk_CHF': initial_results,
                    'Residual_Risk_CHF': residual_results
                }).to_csv(index=False)
                
                st.download_button(
                    label="Download Monte Carlo Results (CSV)",
                    data=mc_csv,
                    file_name=f"monte_carlo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            
            # Summary report
            st.subheader("📋 Summary Report")
            
            report = f"""
# Risk Assessment Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Monte Carlo Simulations:** {n_simulations:,}

## Executive Summary

### Initial Risk Exposure
- **P50 (Median):** {initial_stats['p50']/1e6:.2f} Million CHF
- **P80:** {initial_stats['p80']/1e6:.2f} Million CHF
- **P95:** {initial_stats['p95']/1e6:.2f} Million CHF

### Residual Risk Exposure (After Mitigation)
- **P50 (Median):** {residual_stats['p50']/1e6:.2f} Million CHF
- **P80:** {residual_stats['p80']/1e6:.2f} Million CHF
- **P95:** {residual_stats['p95']/1e6:.2f} Million CHF

### Risk Reduction
- **Total Risk Reduction (P95):** {(initial_stats['p95']-residual_stats['p95'])/1e6:.2f} Million CHF ({((initial_stats['p95']-residual_stats['p95'])/initial_stats['p95']*100):.1f}%)
- **Total Mitigation Investment:** {df['Cost of Measures_Value'].sum()/1e6:.2f} Million CHF
- **Net Benefit (P95):** {((initial_stats['p95']-residual_stats['p95'])-df['Cost of Measures_Value'].sum())/1e6:.2f} Million CHF

### Key Findings
- Total number of identified risks: {len(df)}
- Risks with schedule impact: {df['Schedule_Impact'].sum()}
- Risks with mitigation measures: {len(df[df['Cost of Measures_Value'] > 0])}

## Recommendations
Based on the Monte Carlo simulation, the project should reserve at least **{residual_stats['p95']/1e6:.2f} Million CHF** 
(P95 confidence level) for residual risk exposure after implementing all planned mitigation measures.
"""
            
            st.markdown(report)
            
            st.download_button(
                label="Download Summary Report (MD)",
                data=report,
                file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    else:
        # Initial state - show instructions
        st.info("👈 Click **'Run Simulation'** in the sidebar to begin risk analysis")
        
        st.markdown("""
        ## How to Use This Tool
        
        1. **Upload your risk register** (CSV format) or use the default dataset
        2. **Configure simulation parameters** in the sidebar:
           - Number of Monte Carlo simulations (higher = more accurate)
           - Confidence level for reporting
        3. **Run the simulation** to generate probabilistic risk analysis
        4. **Explore the results** across multiple tabs:
           - **Dashboard**: Overview and key metrics
           - **Risk Matrix**: Visual risk mapping
           - **Monte Carlo Results**: Statistical distributions and confidence intervals
           - **Cost-Benefit**: ROI analysis for mitigation measures
           - **Risk Register**: Detailed risk data with filters
           - **Export**: Download results in Excel/CSV formats
        
        ### Features
        - ✅ Monte Carlo simulation with 1,000-100,000 iterations
        - ✅ P50/P80/P95 confidence intervals
        - ✅ Interactive risk matrices (Initial vs Residual)
        - ✅ Cost-benefit and ROI analysis
        - ✅ Distribution plots, CDF, and box plots
        - ✅ Tornado charts for sensitivity analysis
        - ✅ Export to Excel and CSV
        
        ### Risk Register Format
        Your CSV should contain these columns:
        - Risk ID
        - Risk Description
        - Initial risk (with currency)
        - Likelihood (as percentage)
        - Residual risk
        - Residual likelihood
        - Cost of Measures
        - Schedule Impact (Yes/No)
        """)

if __name__ == "__main__":
    main()
