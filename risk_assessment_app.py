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

def get_confidence_value(stats, confidence_level):
    """Get the risk value for the selected confidence level

    Parameters:
    - stats: Dictionary with statistical measures from calculate_statistics()
    - confidence_level: String like 'P50', 'P80', or 'P95'

    Returns:
    - The corresponding percentile value
    """
    percentile_map = {
        'P50': 'p50',
        'P80': 'p80',
        'P95': 'p95'
    }
    return stats[percentile_map[confidence_level]]

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

def create_cdf_plot(results, stats, risk_type, selected_confidence='P95'):
    """Create cumulative distribution function plot with highlighted confidence level"""
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

    # Add percentile lines with highlighting for selected confidence level
    percentiles = [('P50', stats['p50'], 'green'),
                   ('P80', stats['p80'], 'orange'),
                   ('P95', stats['p95'], 'red')]

    for label, value, color in percentiles:
        # Make selected confidence level more prominent
        if label == selected_confidence:
            line_width = 4
            line_dash = "solid"
            annotation_text = f"★ {label}: {value/1e6:.2f}M CHF (ACTIVE)"
        else:
            line_width = 2
            line_dash = "dash"
            annotation_text = f"{label}: {value/1e6:.2f}M CHF"

        fig.add_vline(x=value, line_dash=line_dash, line_color=color, line_width=line_width,
                     annotation_text=annotation_text,
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

    # Classify as Threat or Opportunity based on Initial_EV
    # Threats have positive EV (costs), Opportunities have negative EV (benefits)
    df['Risk_Type'] = df['Initial_EV'].apply(lambda x: 'Opportunity' if x < 0 else 'Threat')

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

def calculate_threat_opportunity_metrics(df):
    """
    Calculate separate metrics for threats and opportunities.

    In quantitative risk analysis:
    - Threats: Positive EV (potential costs/losses)
    - Opportunities: Negative EV (potential benefits/savings)
    - Net Exposure: Threat EV + Opportunity EV (where Opportunity EV is negative)

    Returns:
        dict with threat_ev, opportunity_ev, net_exposure, and counts
    """
    # Separate threats and opportunities
    df_threats = df[df['Initial_EV'] > 0]
    df_opportunities = df[df['Initial_EV'] < 0]

    # Calculate EVs (Opportunity EV will be negative)
    threat_initial_ev = df_threats['Initial_EV'].sum()
    opportunity_initial_ev = df_opportunities['Initial_EV'].sum()  # This is negative

    threat_residual_ev = df_threats['Residual_EV'].sum()
    opportunity_residual_ev = df_opportunities['Residual_EV'].sum()  # This is negative

    # Net exposure = Threats + Opportunities (opportunities reduce the total)
    net_initial_exposure = threat_initial_ev + opportunity_initial_ev
    net_residual_exposure = threat_residual_ev + opportunity_residual_ev

    # Calculate risk reduction for each category
    threat_reduction = threat_initial_ev - threat_residual_ev
    # For opportunities: "improvement" means opportunity EV becomes more negative (larger benefit)
    opportunity_change = opportunity_initial_ev - opportunity_residual_ev

    return {
        'threat_count': len(df_threats),
        'opportunity_count': len(df_opportunities),
        'threat_initial_ev': threat_initial_ev,
        'opportunity_initial_ev': opportunity_initial_ev,  # Negative value
        'threat_residual_ev': threat_residual_ev,
        'opportunity_residual_ev': opportunity_residual_ev,  # Negative value
        'net_initial_exposure': net_initial_exposure,
        'net_residual_exposure': net_residual_exposure,
        'threat_reduction': threat_reduction,
        'opportunity_change': opportunity_change,
        'net_reduction': net_initial_exposure - net_residual_exposure
    }

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
    # Take top N risks - already sorted by variance contribution descending
    df_plot = sensitivity_df.head(top_n).copy()

    # IMPORTANT: Ensure data is sorted by Variance % descending for proper Pareto chart
    # This ensures bars descend from left to right and cumulative curve ascends smoothly
    df_plot = df_plot.sort_values('Variance %', ascending=False).reset_index(drop=True)

    # CRITICAL: Recalculate cumulative % after sorting to ensure correct Pareto curve
    # The cumulative % from sensitivity_df was based on a different order
    df_plot['Cumulative %'] = df_plot['Variance %'].cumsum()

    # CRITICAL FIX: Remove any duplicate Risk IDs to prevent multiple line segments
    # Keep only the first occurrence (highest variance) if duplicates exist
    df_plot = df_plot.drop_duplicates(subset=['Risk ID'], keep='first').reset_index(drop=True)

    # Create truncated risk names for x-axis (max 35 chars for readability)
    df_plot['Risk_Name_Short'] = df_plot['Risk Description'].apply(
        lambda x: x[:35] + '...' if len(x) > 35 else x
    )

    # Prepare hover data with full risk description
    df_plot['hover_text'] = df_plot.apply(
        lambda row: f"<b>{row['Risk ID']}</b>: {row['Risk Description']}", axis=1
    )

    # Create figure WITH secondary y-axis (bars auto-scaled, cumulative 0-100%)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bars for individual variance contribution on PRIMARY y-axis (auto-scaled)
    fig.add_trace(
        go.Bar(
            x=df_plot['Risk_Name_Short'],  # Use truncated risk names for readability
            y=df_plot['Variance %'],
            name='Individual Variance %',
            marker_color='#E74C3C',  # Vibrant red
            text=df_plot['Variance %'].apply(lambda x: f'{x:.1f}%'),  # Bar labels
            textposition='outside',  # Labels above bars
            textfont=dict(size=10, color='#E74C3C'),
            customdata=df_plot['hover_text'],
            hovertemplate='%{customdata}<br>Individual Variance: %{y:.2f}%<extra></extra>'
        ),
        secondary_y=False
    )

    # CRITICAL FIX: Add cumulative line as SINGLE trace with explicit connectgaps
    # This ensures ONE continuous line instead of multiple segments
    fig.add_trace(
        go.Scatter(
            x=df_plot['Risk_Name_Short'],  # Same x-axis as bars
            y=df_plot['Cumulative %'],
            name='Cumulative %',
            mode='lines+markers',  # Both lines and markers
            line=dict(color='#2C3E50', width=4),  # Dark slate, thick line
            marker=dict(size=10, symbol='diamond', color='#2C3E50'),
            connectgaps=True,  # Ensure continuous line even if gaps exist
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

    # Update x-axis with 45-degree rotation and preserve order
    fig.update_xaxes(
        title_text="Risk Description (ordered by variance contribution)",
        tickangle=-45,  # Rotate labels 45 degrees
        tickfont=dict(size=10),
        title_font=dict(size=12),
        categoryorder='array',  # Preserve the exact order from data
        categoryarray=df_plot['Risk_Name_Short'].tolist()  # Explicit order: highest to lowest variance
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

    # Update layout with 16x8 inch figure size and proper dimensions
    fig.update_layout(
        title={
            'text': 'Pareto Analysis - Risk Variance Contribution<br><sub>Identify the vital few risks driving most uncertainty (80/20 rule)</sub>',
            'font': {'size': 17}
        },
        width=1536,  # 16 inches * 96 DPI = 1536 pixels
        height=768,  # 8 inches * 96 DPI = 768 pixels
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
        margin=dict(l=100, r=100, t=140, b=200),  # Larger bottom margin for rotated risk names
        font=dict(size=11),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

# ============================================================================
# WHAT-IF SCENARIO ANALYSIS FUNCTIONS
# ============================================================================

def run_scenario_monte_carlo(df, scenario_adjustments, n_simulations=10000, risk_type='initial'):
    """
    Run Monte Carlo simulation with scenario adjustments applied

    Parameters:
    - df: Original DataFrame with risk data
    - scenario_adjustments: Dict of {risk_id: {'likelihood': float, 'impact': float}}
    - n_simulations: Number of Monte Carlo iterations
    - risk_type: 'initial' or 'residual'

    Returns:
    - results: Array of simulation results
    - modified_df: DataFrame with adjustments applied
    """
    # Create a copy of the dataframe to apply adjustments
    modified_df = df.copy()

    if risk_type == 'initial':
        impact_col = 'Initial risk_Value'
        likelihood_col = 'Initial_Likelihood'
    else:
        impact_col = 'Residual risk_Value'
        likelihood_col = 'Residual_Likelihood'

    # Apply scenario adjustments
    for risk_id, adjustments in scenario_adjustments.items():
        mask = modified_df['Risk ID'] == risk_id
        if mask.any():
            if 'likelihood' in adjustments:
                modified_df.loc[mask, likelihood_col] = adjustments['likelihood']
            if 'impact' in adjustments:
                modified_df.loc[mask, impact_col] = adjustments['impact']

    # Prepare data for simulation
    impacts = modified_df[impact_col].values
    likelihoods = modified_df[likelihood_col].values

    # Generate random numbers
    random_numbers = np.random.random((n_simulations, len(modified_df)))

    # Monte Carlo simulation
    results = np.zeros(n_simulations)

    for i in range(n_simulations):
        occurred = random_numbers[i] < likelihoods
        results[i] = np.sum(impacts * occurred)

    return results, modified_df

def create_scenario_comparison_chart(baseline_stats, scenario_stats, scenario_name, confidence_level='P95'):
    """Create a comparison chart between baseline and scenario results"""

    metrics = ['Mean', 'P50', 'P80', 'P95']
    baseline_values = [
        baseline_stats['mean'] / 1e6,
        baseline_stats['p50'] / 1e6,
        baseline_stats['p80'] / 1e6,
        baseline_stats['p95'] / 1e6
    ]
    scenario_values = [
        scenario_stats['mean'] / 1e6,
        scenario_stats['p50'] / 1e6,
        scenario_stats['p80'] / 1e6,
        scenario_stats['p95'] / 1e6
    ]

    fig = go.Figure()

    # Baseline bars
    fig.add_trace(go.Bar(
        name='Baseline',
        x=metrics,
        y=baseline_values,
        marker_color='#3498DB',
        text=[f'{v:.2f}M' for v in baseline_values],
        textposition='outside'
    ))

    # Scenario bars
    fig.add_trace(go.Bar(
        name=scenario_name,
        x=metrics,
        y=scenario_values,
        marker_color='#E74C3C',
        text=[f'{v:.2f}M' for v in scenario_values],
        textposition='outside'
    ))

    # Highlight the selected confidence level
    confidence_idx = metrics.index(confidence_level)

    fig.update_layout(
        title=f'Scenario Comparison: Baseline vs {scenario_name}',
        xaxis_title='Metric',
        yaxis_title='Risk Exposure (Million CHF)',
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    return fig

def create_scenario_cdf_comparison(baseline_results, scenario_results, baseline_stats, scenario_stats,
                                    scenario_name, confidence_level='P95'):
    """Create overlaid CDF curves for baseline vs scenario"""

    # Sort results for CDF
    baseline_sorted = np.sort(baseline_results)
    scenario_sorted = np.sort(scenario_results)

    # Create probability array (0 to 1)
    n = len(baseline_sorted)
    probabilities = np.arange(1, n + 1) / n

    fig = go.Figure()

    # Baseline CDF
    fig.add_trace(go.Scatter(
        x=baseline_sorted / 1e6,
        y=probabilities * 100,
        mode='lines',
        name='Baseline',
        line=dict(color='#3498DB', width=3)
    ))

    # Scenario CDF
    fig.add_trace(go.Scatter(
        x=scenario_sorted / 1e6,
        y=probabilities * 100,
        mode='lines',
        name=scenario_name,
        line=dict(color='#E74C3C', width=3)
    ))

    # Add confidence level lines
    percentile_map = {'P50': 50, 'P80': 80, 'P95': 95}
    conf_percentile = percentile_map[confidence_level]

    # Horizontal line at confidence level
    fig.add_hline(y=conf_percentile, line_dash='dash', line_color='#2ECC71', line_width=2,
                  annotation_text=f'{confidence_level} Level', annotation_position='right')

    # Vertical lines at baseline and scenario values
    baseline_val = get_confidence_value(baseline_stats, confidence_level) / 1e6
    scenario_val = get_confidence_value(scenario_stats, confidence_level) / 1e6

    fig.add_vline(x=baseline_val, line_dash='dot', line_color='#3498DB', line_width=2)
    fig.add_vline(x=scenario_val, line_dash='dot', line_color='#E74C3C', line_width=2)

    fig.update_layout(
        title=f'CDF Comparison: Baseline vs {scenario_name}',
        xaxis_title='Total Risk Exposure (Million CHF)',
        yaxis_title='Cumulative Probability (%)',
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        yaxis=dict(range=[0, 100])
    )

    return fig

def create_scenario_impact_table(df, scenario_adjustments, risk_type='initial'):
    """Create a table showing which risks were modified in the scenario"""

    if risk_type == 'initial':
        impact_col = 'Initial risk_Value'
        likelihood_col = 'Initial_Likelihood'
    else:
        impact_col = 'Residual risk_Value'
        likelihood_col = 'Residual_Likelihood'

    modified_risks = []

    for risk_id, adjustments in scenario_adjustments.items():
        original_row = df[df['Risk ID'] == risk_id]
        if len(original_row) > 0:
            original_row = original_row.iloc[0]

            orig_likelihood = original_row[likelihood_col]
            orig_impact = original_row[impact_col]

            new_likelihood = adjustments.get('likelihood', orig_likelihood)
            new_impact = adjustments.get('impact', orig_impact)

            orig_ev = orig_likelihood * orig_impact
            new_ev = new_likelihood * new_impact
            ev_change = new_ev - orig_ev

            modified_risks.append({
                'Risk ID': risk_id,
                'Description': original_row['Risk Description'][:50] + '...' if len(str(original_row['Risk Description'])) > 50 else original_row['Risk Description'],
                'Original Likelihood': f'{orig_likelihood*100:.1f}%',
                'New Likelihood': f'{new_likelihood*100:.1f}%',
                'Original Impact (M CHF)': f'{orig_impact/1e6:.2f}',
                'New Impact (M CHF)': f'{new_impact/1e6:.2f}',
                'EV Change (M CHF)': f'{ev_change/1e6:+.2f}'
            })

    return pd.DataFrame(modified_risks) if modified_risks else None

# ============================================================================
# SANKEY DIAGRAM FOR RISK FLOW
# ============================================================================

def create_risk_sankey(df, initial_stats, residual_stats, confidence_level='P95'):
    """
    Create a Sankey diagram showing the flow of both threats and opportunities.

    Shows how initial risk exposure (threats + opportunities) flows through
    mitigation to result in residual exposure, with threats and opportunities
    clearly distinguished.

    Parameters:
    - df: DataFrame with risk data including Initial_EV, Residual_EV, Risk_Reduction
    - initial_stats: Statistics dict for initial risk (from Monte Carlo)
    - residual_stats: Statistics dict for residual risk (from Monte Carlo)
    - confidence_level: Selected confidence level (P50, P80, P95)
    """

    # Separate threats (positive EV) and opportunities (negative EV)
    df_threats = df[df['Initial_EV'] > 0].copy()
    df_opportunities = df[df['Initial_EV'] < 0].copy()

    # Calculate threat values
    threat_initial = df_threats['Initial_EV'].sum()
    threat_residual = df_threats['Residual_EV'].sum()
    threat_reduced = max(0, threat_initial - threat_residual)

    # Calculate opportunity values (these are negative, so we use abs for display)
    opp_initial = abs(df_opportunities['Initial_EV'].sum()) if len(df_opportunities) > 0 else 0
    opp_residual = abs(df_opportunities['Residual_EV'].sum()) if len(df_opportunities) > 0 else 0
    opp_change = opp_residual - opp_initial  # Positive if opportunity improved

    # Net exposure calculations
    net_initial = threat_initial - opp_initial  # Threats minus opportunities
    net_residual = threat_residual - opp_residual

    # Node labels - Threats flow (red tones), Opportunities flow (green tones)
    # Node indices:
    # 0: Initial Threats
    # 1: Threat Reduced
    # 2: Residual Threats
    # 3: Initial Opportunities (shown as benefit)
    # 4: Opportunity Enhanced (if improved)
    # 5: Residual Opportunities
    # 6: Net Residual Exposure

    labels = [
        f"Initial Threats\n{threat_initial/1e6:.1f}M CHF",                    # 0
        f"Threat Reduced\n{threat_reduced/1e6:.1f}M CHF",                     # 1
        f"Residual Threats\n{threat_residual/1e6:.1f}M CHF",                  # 2
        f"Initial Opportunities\n-{opp_initial/1e6:.1f}M CHF",                # 3
        f"Opp. Change\n{opp_change/1e6:+.1f}M CHF",                           # 4
        f"Residual Opportunities\n-{opp_residual/1e6:.1f}M CHF",              # 5
        f"Net Residual\n{net_residual/1e6:.1f}M CHF"                          # 6
    ]

    # Colors: Red tones for threats, Green/Blue tones for opportunities
    colors = [
        "#C0392B",  # 0: Initial Threats - Dark Red
        "#27AE60",  # 1: Threat Reduced - Green (good!)
        "#E74C3C",  # 2: Residual Threats - Red
        "#1ABC9C",  # 3: Initial Opportunities - Teal
        "#16A085",  # 4: Opportunity Change - Dark Teal
        "#2ECC71",  # 5: Residual Opportunities - Green
        "#8E44AD"   # 6: Net Residual - Purple
    ]

    links_source = []
    links_target = []
    links_value = []
    links_color = []

    # Threat flows (warm colors)
    # Flow: Initial Threats -> Threat Reduced
    if threat_reduced > 0:
        links_source.append(0)
        links_target.append(1)
        links_value.append(threat_reduced)
        links_color.append("rgba(39, 174, 96, 0.6)")  # Green - reduction is good

    # Flow: Initial Threats -> Residual Threats
    if threat_residual > 0:
        links_source.append(0)
        links_target.append(2)
        links_value.append(threat_residual)
        links_color.append("rgba(231, 76, 60, 0.6)")  # Red

    # Flow: Residual Threats -> Net Residual
    if threat_residual > 0:
        links_source.append(2)
        links_target.append(6)
        links_value.append(threat_residual)
        links_color.append("rgba(142, 68, 173, 0.5)")  # Purple

    # Opportunity flows (cool colors) - only if opportunities exist
    if len(df_opportunities) > 0:
        # Flow: Initial Opportunities -> Opportunity Change (if changed)
        if abs(opp_change) > 0:
            links_source.append(3)
            links_target.append(4)
            links_value.append(abs(opp_change))
            links_color.append("rgba(22, 160, 133, 0.6)")  # Teal

        # Flow: Initial Opportunities -> Residual Opportunities
        # The flow represents the portion that remains
        min_opp = min(opp_initial, opp_residual)
        if min_opp > 0:
            links_source.append(3)
            links_target.append(5)
            links_value.append(min_opp)
            links_color.append("rgba(46, 204, 113, 0.6)")  # Green

        # If opportunity improved (residual > initial), add flow from change to residual
        if opp_change > 0:
            links_source.append(4)
            links_target.append(5)
            links_value.append(opp_change)
            links_color.append("rgba(46, 204, 113, 0.6)")  # Green

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=35,
            line=dict(color="black", width=2),
            label=labels,
            color=colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color,
            hovertemplate='%{source.label} → %{target.label}<br>Value: %{value:,.0f} CHF<extra></extra>'
        ),
        textfont=dict(size=14, color='black', family='Arial Black')
    )])

    # Calculate metrics for title
    threat_reduction_pct = (threat_reduced / threat_initial * 100) if threat_initial > 0 else 0
    opp_count = len(df_opportunities)
    threat_count = len(df_threats)

    fig.update_layout(
        title={
            'text': f'<b>Risk Mitigation Flow: Threats & Opportunities</b><br>'
                   f'<sup>{threat_count} Threats (Initial: {threat_initial/1e6:.1f}M, Reduced: {threat_reduction_pct:.0f}%) | '
                   f'{opp_count} Opportunities (Benefit: {opp_residual/1e6:.1f}M) | '
                   f'Net Exposure: {net_residual/1e6:.1f}M CHF</sup>',
            'font': {'size': 16, 'color': '#2C3E50'},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=14, color='#2C3E50', family='Arial'),
        height=550,
        paper_bgcolor='#FAFAFA',
        margin=dict(l=30, r=30, t=100, b=30)
    )

    return fig

def create_risk_sankey_detailed(df):
    """
    Create a more detailed Sankey diagram with risk categories.

    Shows risk flow broken down by schedule impact status.
    """

    # Filter to only include risks (positive EV), excluding opportunities
    df_risks = df[df['Initial_EV'] > 0].copy()

    # Separate by schedule impact
    df_schedule = df_risks[df_risks['Schedule_Impact'] == True]
    df_no_schedule = df_risks[df_risks['Schedule_Impact'] == False]

    # Calculate values
    total_initial = df_risks['Initial_EV'].sum()
    schedule_initial = df_schedule['Initial_EV'].sum()
    no_schedule_initial = df_no_schedule['Initial_EV'].sum()

    schedule_residual = df_schedule['Residual_EV'].abs().sum()
    no_schedule_residual = df_no_schedule['Residual_EV'].abs().sum()

    schedule_reduced = max(0, schedule_initial - schedule_residual)
    no_schedule_reduced = max(0, no_schedule_initial - no_schedule_residual)

    total_residual = df_risks['Residual_EV'].abs().sum()
    total_reduced = schedule_reduced + no_schedule_reduced

    labels = [
        f"Total Initial Risk\n{total_initial/1e6:.1f}M CHF",        # 0
        f"Schedule Impact\n{schedule_initial/1e6:.1f}M CHF",         # 1
        f"No Schedule Impact\n{no_schedule_initial/1e6:.1f}M CHF",   # 2
        f"Reduced (Schedule)\n{schedule_reduced/1e6:.1f}M CHF",      # 3
        f"Reduced (No Schedule)\n{no_schedule_reduced/1e6:.1f}M CHF",# 4
        f"Residual (Schedule)\n{schedule_residual/1e6:.1f}M CHF",    # 5
        f"Residual (No Schedule)\n{no_schedule_residual/1e6:.1f}M CHF", # 6
        f"Total Risk Reduced\n{total_reduced/1e6:.1f}M CHF",         # 7
        f"Total Residual\n{total_residual/1e6:.1f}M CHF"             # 8
    ]

    colors = [
        "#C0392B",  # Total Initial - Dark Red
        "#922B21",  # Schedule Impact - Darker Red
        "#D35400",  # No Schedule Impact - Dark Orange
        "#1E8449",  # Reduced (Schedule) - Dark Green
        "#28B463",  # Reduced (No Schedule) - Green
        "#2471A3",  # Residual (Schedule) - Dark Blue
        "#5DADE2",  # Residual (No Schedule) - Light Blue
        "#117A65",  # Total Reduced - Teal
        "#B9770E"   # Total Residual - Dark Gold
    ]

    # Build links dynamically (skip zero values)
    links_source = []
    links_target = []
    links_value = []
    links_color = []

    link_definitions = [
        (0, 1, schedule_initial, "rgba(146, 43, 33, 0.5)"),
        (0, 2, no_schedule_initial, "rgba(211, 84, 0, 0.5)"),
        (1, 3, schedule_reduced, "rgba(30, 132, 73, 0.5)"),
        (1, 5, schedule_residual, "rgba(36, 113, 163, 0.5)"),
        (2, 4, no_schedule_reduced, "rgba(40, 180, 99, 0.5)"),
        (2, 6, no_schedule_residual, "rgba(93, 173, 226, 0.5)"),
        (3, 7, schedule_reduced, "rgba(17, 122, 101, 0.5)"),
        (4, 7, no_schedule_reduced, "rgba(17, 122, 101, 0.5)"),
        (5, 8, schedule_residual, "rgba(185, 119, 14, 0.5)"),
        (6, 8, no_schedule_residual, "rgba(185, 119, 14, 0.5)")
    ]

    for source, target, value, color in link_definitions:
        if value > 0:
            links_source.append(source)
            links_target.append(target)
            links_value.append(value)
            links_color.append(color)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,
            thickness=30,
            line=dict(color="black", width=2),
            label=labels,
            color=colors
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color
        ),
        textfont=dict(size=14, color='black', family='Arial Black')
    )])

    # Count opportunities excluded
    opportunities_count = len(df[df['Initial_EV'] <= 0])
    opportunities_note = f" | Opportunities excluded: {opportunities_count}" if opportunities_count > 0 else ""

    fig.update_layout(
        title={
            'text': f'<b>Risk Flow by Schedule Impact</b><br><sup>Breakdown by schedule impact category{opportunities_note}</sup>',
            'font': {'size': 18, 'color': '#2C3E50'},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=14, color='#2C3E50', family='Arial'),
        height=600,
        paper_bgcolor='#FAFAFA',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig

# ============================================================================
# DOCX Report Generation Functions
# ============================================================================
def set_cell_background(cell, fill_color):
    """Set background color for table cell"""
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
    cell._element.get_or_add_tcPr().append(shading_elm)

def set_cell_border(cell, **kwargs):
    """
    Set cell border
    kwargs: top, bottom, left, right, insideH, insideV
    values: {'sz': 24, 'val': 'single', 'color': '#000000'}
    """
    from docx.oxml import OxmlElement, parse_xml
    from docx.oxml.ns import qn

    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')

    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            edge_data = kwargs.get(edge)
            edge_el = OxmlElement(f'w:{edge}')
            edge_el.set(qn('w:val'), edge_data.get('val', 'single'))
            edge_el.set(qn('w:sz'), str(edge_data.get('sz', 4)))
            edge_el.set(qn('w:color'), edge_data.get('color', '000000'))
            tcBorders.append(edge_el)

    tcPr.append(tcBorders)

def format_table_professional(table, has_header=True):
    """Apply professional formatting to table"""
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Set column widths to auto
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = 1  # Center vertically

    if has_header and len(table.rows) > 0:
        # Format header row
        for cell in table.rows[0].cells:
            # Dark blue background
            set_cell_background(cell, "1F4E78")
            # White text
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.font.name = 'Calibri'
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Format data rows with alternating colors
    for i, row in enumerate(table.rows[1:], 1):
        # Alternating row colors
        if i % 2 == 0:
            bg_color = "F2F2F2"  # Light gray
        else:
            bg_color = "FFFFFF"  # White

        for cell in row.cells:
            set_cell_background(cell, bg_color)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Calibri'
                    run.font.color.rgb = RGBColor(0, 0, 0)

def format_table_executive(table, has_header=True, highlight_rows=None):
    """Apply executive-level professional formatting to summary tables with enhanced styling"""
    from docx.shared import Pt, RGBColor, Inches, Twips
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    highlight_rows = highlight_rows or []

    # Set table alignment to center
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Apply table borders
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement('w:tblPr')
    tblBorders = OxmlElement('w:tblBorders')

    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:color'), '1F4E78')
        tblBorders.append(border)

    tblPr.append(tblBorders)
    if tbl.tblPr is None:
        tbl.insert(0, tblPr)

    # Format cells
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = 1  # Center vertically

    if has_header and len(table.rows) > 0:
        # Format header row with gradient-style professional look
        for cell in table.rows[0].cells:
            set_cell_background(cell, "1F4E78")  # Dark blue
            # Add bottom border emphasis
            set_cell_border(cell, bottom={'sz': 12, 'val': 'single', 'color': '0D2E4D'})
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.font.name = 'Calibri'

    # Format data rows
    for i, row in enumerate(table.rows[1:], 1):
        is_highlight = i in highlight_rows

        if is_highlight:
            bg_color = "D4EDFC"  # Light blue highlight
        elif i % 2 == 0:
            bg_color = "F8F9FA"  # Very light gray
        else:
            bg_color = "FFFFFF"  # White

        for j, cell in enumerate(row.cells):
            set_cell_background(cell, bg_color)
            for paragraph in cell.paragraphs:
                # Right-align numeric columns (typically not the first column)
                if j > 0:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Calibri'
                    run.font.color.rgb = RGBColor(0, 0, 0)
                    if is_highlight:
                        run.font.bold = True

def format_table_contingency(table):
    """Apply special formatting for contingency allocation summary table"""
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    # Set table alignment
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Apply thick table borders
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement('w:tblPr')
    tblBorders = OxmlElement('w:tblBorders')

    for border_name in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '12')  # Thicker outer border
        border.set(qn('w:color'), '1F4E78')
        tblBorders.append(border)

    for border_name in ['insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:color'), 'B4C6E7')  # Lighter internal borders
        tblBorders.append(border)

    tblPr.append(tblBorders)
    if tbl.tblPr is None:
        tbl.insert(0, tblPr)

    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = 1

    # Format header row
    if len(table.rows) > 0:
        for cell in table.rows[0].cells:
            set_cell_background(cell, "1F4E78")
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = Pt(12)
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.font.name = 'Calibri'

    # Format data rows with special styling
    row_styles = [
        ("E8F4FD", False),   # Row 1: Light blue - Residual Risk Reserve
        ("E8F4FD", False),   # Row 2: Light blue - Mitigation Investment
        ("1F4E78", True),    # Row 3: Dark blue - Total Contingency (header-style)
    ]

    for i, row in enumerate(table.rows[1:], 0):
        if i < len(row_styles):
            bg_color, is_bold = row_styles[i]
        else:
            bg_color, is_bold = ("FFFFFF", False)

        for j, cell in enumerate(row.cells):
            set_cell_background(cell, bg_color)
            for paragraph in cell.paragraphs:
                if j > 0:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for run in paragraph.runs:
                    run.font.size = Pt(11)
                    run.font.name = 'Calibri'
                    run.font.bold = is_bold
                    # White text for dark row
                    if bg_color == "1F4E78":
                        run.font.color.rgb = RGBColor(255, 255, 255)
                    else:
                        run.font.color.rgb = RGBColor(0, 0, 0)

def plotly_to_image_bytes(fig, width=1600, height=800):
    """Convert Plotly figure to PNG image bytes using kaleido"""
    try:
        import io
        # Use kaleido for server-side rendering (lightweight, no browser needed)
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return io.BytesIO(img_bytes)
    except Exception as e:
        # Fall back gracefully if kaleido is not available
        return None

# ============= MATPLOTLIB CHART FUNCTIONS FOR DOCX EXPORT =============

def create_matplotlib_risk_matrix_combined(df):
    """Create side-by-side risk matrix comparison (initial vs residual) for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Determine common axis limits for both charts
    max_likelihood = max(df['Initial_Likelihood'].max(), df['Residual_Likelihood'].max()) * 100 * 1.1
    max_impact = max(df['Initial risk_Value'].max(), df['Residual risk_Value'].max()) / 1e6 * 1.1
    max_ev = max(
        (df['Initial_Likelihood'] * df['Initial risk_Value']).max(),
        (df['Residual_Likelihood'] * df['Residual risk_Value']).max()
    ) / 1e6

    for idx, (ax, risk_type) in enumerate(zip(axes, ['initial', 'residual'])):
        if risk_type == 'initial':
            x = df['Initial_Likelihood'] * 100
            y = df['Initial risk_Value'] / 1e6
            title = 'Initial Risk Assessment'
            color_label = 'Before Mitigation'
        else:
            x = df['Residual_Likelihood'] * 100
            y = df['Residual risk_Value'] / 1e6
            title = 'Residual Risk Assessment'
            color_label = 'After Mitigation'

        # Calculate expected values for color
        ev = x * y / 100

        # Create scatter plot with consistent color scale
        scatter = ax.scatter(x, y, c=ev, cmap='RdYlGn_r', s=120, alpha=0.7,
                            edgecolors='black', linewidth=0.5, vmin=0, vmax=max_ev)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Expected Value (M CHF)', fontsize=9)

        # Set consistent axis limits
        ax.set_xlim(0, max_likelihood)
        ax.set_ylim(0, max_impact)

        # Add quadrant lines using overall median
        ax.axhline(y=df['Initial risk_Value'].median() / 1e6, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=df['Initial_Likelihood'].median() * 100, color='gray', linestyle='--', alpha=0.5)

        # Labels
        ax.set_xlabel('Likelihood (%)', fontsize=11)
        ax.set_ylabel('Impact (Million CHF)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', color='#1F4E78')

        # Add risk labels
        for _, row in df.iterrows():
            if risk_type == 'initial':
                xi, yi = row['Initial_Likelihood'] * 100, row['Initial risk_Value'] / 1e6
            else:
                xi, yi = row['Residual_Likelihood'] * 100, row['Residual risk_Value'] / 1e6
            ax.annotate(row['Risk ID'], (xi, yi), fontsize=6, alpha=0.7,
                       xytext=(2, 2), textcoords='offset points')

        ax.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle('Risk Matrix Comparison: Before vs After Mitigation', fontsize=14, fontweight='bold', color='#1F4E78', y=1.02)
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_risk_matrix(df, risk_type='initial'):
    """Create risk matrix using matplotlib for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    if risk_type == 'initial':
        x = df['Initial_Likelihood'] * 100
        y = df['Initial risk_Value'] / 1e6
        title = 'Risk Matrix - Initial Risk Assessment'
    else:
        x = df['Residual_Likelihood'] * 100
        y = df['Residual risk_Value'] / 1e6
        title = 'Risk Matrix - Residual Risk Assessment'

    # Calculate expected values for color
    ev = x * y / 100

    # Create scatter plot
    scatter = ax.scatter(x, y, c=ev, cmap='RdYlGn_r', s=150, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Expected Value (M CHF)', fontsize=10)

    # Add quadrant lines
    ax.axhline(y=y.median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=x.median(), color='gray', linestyle='--', alpha=0.5)

    # Labels
    ax.set_xlabel('Likelihood (%)', fontsize=12)
    ax.set_ylabel('Impact (Million CHF)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1F4E78')

    # Add risk labels
    for idx, row in df.iterrows():
        if risk_type == 'initial':
            xi, yi = row['Initial_Likelihood'] * 100, row['Initial risk_Value'] / 1e6
        else:
            xi, yi = row['Residual_Likelihood'] * 100, row['Residual risk_Value'] / 1e6
        ax.annotate(row['Risk ID'], (xi, yi), fontsize=7, alpha=0.7,
                   xytext=(3, 3), textcoords='offset points')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_cdf_combined(initial_results, initial_stats, residual_results, residual_stats, confidence_level='P95'):
    """Create side-by-side CDF comparison (initial vs residual) for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Determine common x-axis limit
    max_value = max(initial_results.max(), residual_results.max()) / 1e6 * 1.05

    for idx, (ax, results, stats, title) in enumerate(zip(
        axes,
        [initial_results, residual_results],
        [initial_stats, residual_stats],
        ['Initial Risk Exposure', 'Residual Risk Exposure']
    )):
        # Sort results for CDF
        sorted_results = np.sort(results)
        cumulative = np.arange(1, len(sorted_results) + 1) / len(sorted_results)

        # Plot CDF
        ax.plot(sorted_results / 1e6, cumulative * 100, color='#1F4E78', linewidth=2)
        ax.fill_between(sorted_results / 1e6, cumulative * 100, alpha=0.3, color='#1F4E78')

        # Add percentile lines
        percentiles = {'P50': (stats['p50'], '#2ecc71'), 'P80': (stats['p80'], '#f39c12'), 'P95': (stats['p95'], '#e74c3c')}
        for label, (value, color) in percentiles.items():
            ax.axvline(x=value / 1e6, color=color, linestyle='--', linewidth=2, alpha=0.8)
            pct = int(label[1:])
            ax.axhline(y=pct, color=color, linestyle=':', linewidth=1, alpha=0.5)

            # Highlight selected confidence level
            if label == confidence_level:
                ax.annotate(f'{label}: {value/1e6:.2f}M CHF',
                           xy=(value / 1e6, pct), xytext=(10, 10),
                           textcoords='offset points', fontsize=9, fontweight='bold',
                           color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color))
            else:
                ax.annotate(f'{label}: {value/1e6:.2f}M',
                           xy=(value / 1e6, pct), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, color=color)

        ax.set_title(title, fontsize=13, fontweight='bold', color='#1F4E78')
        ax.set_xlabel('Total Risk Exposure (Million CHF)', fontsize=11)
        ax.set_ylabel('Cumulative Probability (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_value)

    # Add overall title
    fig.suptitle('CDF Comparison: Before vs After Mitigation', fontsize=14, fontweight='bold', color='#1F4E78', y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_cdf(results, stats, risk_type='initial', confidence_level='P95'):
    """Create CDF plot using matplotlib for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort results for CDF
    sorted_results = np.sort(results)
    cumulative = np.arange(1, len(sorted_results) + 1) / len(sorted_results)

    # Plot CDF
    ax.plot(sorted_results / 1e6, cumulative * 100, color='#1F4E78', linewidth=2)
    ax.fill_between(sorted_results / 1e6, cumulative * 100, alpha=0.3, color='#1F4E78')

    # Add percentile lines
    percentiles = {'P50': (stats['p50'], '#2ecc71'), 'P80': (stats['p80'], '#f39c12'), 'P95': (stats['p95'], '#e74c3c')}
    for label, (value, color) in percentiles.items():
        ax.axvline(x=value / 1e6, color=color, linestyle='--', linewidth=2, alpha=0.8)
        pct = int(label[1:])
        ax.axhline(y=pct, color=color, linestyle=':', linewidth=1, alpha=0.5)

        # Highlight selected confidence level
        if label == confidence_level:
            ax.annotate(f'{label}: {value/1e6:.2f}M CHF',
                       xy=(value / 1e6, pct), xytext=(10, 10),
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color))
        else:
            ax.annotate(f'{label}: {value/1e6:.2f}M',
                       xy=(value / 1e6, pct), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, color=color)

    title = 'Cumulative Distribution Function - ' + ('Initial' if risk_type == 'initial' else 'Residual') + ' Risk'
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1F4E78')
    ax.set_xlabel('Total Risk Exposure (Million CHF)', fontsize=12)
    ax.set_ylabel('Cumulative Probability (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_histogram_combined(initial_results, initial_stats, residual_results, residual_stats):
    """Create side-by-side histogram comparison (initial vs residual) for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Determine common axis limits
    all_results = np.concatenate([initial_results, residual_results])
    min_val = all_results.min() / 1e6
    max_val = all_results.max() / 1e6
    bins = np.linspace(min_val, max_val, 51)

    for idx, (ax, results, stats, title) in enumerate(zip(
        axes,
        [initial_results, residual_results],
        [initial_stats, residual_stats],
        ['Initial Risk Distribution', 'Residual Risk Distribution']
    )):
        # Create histogram
        n, _, patches = ax.hist(results / 1e6, bins=bins, color='#1F4E78', alpha=0.7, edgecolor='white')

        # Add mean and median lines
        ax.axvline(x=stats['mean'] / 1e6, color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean: {stats["mean"]/1e6:.2f}M')
        ax.axvline(x=stats['p50'] / 1e6, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: {stats["p50"]/1e6:.2f}M')
        ax.axvline(x=stats['p95'] / 1e6, color='#f39c12', linestyle=':', linewidth=2, label=f'P95: {stats["p95"]/1e6:.2f}M')

        ax.set_title(title, fontsize=13, fontweight='bold', color='#1F4E78')
        ax.set_xlabel('Total Risk Exposure (Million CHF)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(min_val, max_val)

    # Add overall title
    fig.suptitle('Risk Distribution Comparison: Before vs After Mitigation', fontsize=14, fontweight='bold', color='#1F4E78', y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_histogram(results, stats, risk_type='initial'):
    """Create histogram using matplotlib for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins, patches = ax.hist(results / 1e6, bins=50, color='#1F4E78', alpha=0.7, edgecolor='white')

    # Add mean and median lines
    ax.axvline(x=stats['mean'] / 1e6, color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean: {stats["mean"]/1e6:.2f}M')
    ax.axvline(x=stats['p50'] / 1e6, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: {stats["p50"]/1e6:.2f}M')
    ax.axvline(x=stats['p95'] / 1e6, color='#f39c12', linestyle=':', linewidth=2, label=f'P95: {stats["p95"]/1e6:.2f}M')

    title = 'Risk Exposure Distribution - ' + ('Initial' if risk_type == 'initial' else 'Residual') + ' Risk'
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1F4E78')
    ax.set_xlabel('Total Risk Exposure (Million CHF)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_pareto(sensitivity_df, top_n=20):
    """Create Pareto chart using matplotlib for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Get top N risks
    df_plot = sensitivity_df.head(top_n).copy()

    # Bar chart for variance contribution
    bars = ax1.barh(range(len(df_plot)), df_plot['Variance %'], color='#1F4E78', alpha=0.8)
    ax1.set_yticks(range(len(df_plot)))
    ax1.set_yticklabels([f"{row['Risk ID']}: {row['Risk Description'][:30]}..."
                         if len(row['Risk Description']) > 30 else f"{row['Risk ID']}: {row['Risk Description']}"
                         for _, row in df_plot.iterrows()], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Variance Contribution (%)', fontsize=12, color='#1F4E78')
    ax1.tick_params(axis='x', labelcolor='#1F4E78')

    # Cumulative line on secondary axis
    ax2 = ax1.twiny()
    ax2.plot(df_plot['Cumulative %'], range(len(df_plot)), color='#e74c3c', linewidth=2, marker='o', markersize=5)
    ax2.set_xlabel('Cumulative %', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='x', labelcolor='#e74c3c')

    # Add 80% threshold line
    ax2.axvline(x=80, color='#e74c3c', linestyle='--', alpha=0.5)
    ax2.annotate('80% threshold', xy=(80, len(df_plot)-1), fontsize=9, color='#e74c3c')

    ax1.set_title('Pareto Analysis - Risk Variance Contribution', fontsize=14, fontweight='bold', color='#1F4E78', pad=20)
    ax1.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_matplotlib_roi_chart(df_with_roi, top_n=20):
    """Create ROI bar chart using matplotlib for DOCX export"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df_with_measures = df_with_roi[df_with_roi['Cost of Measures_Value'] > 0].copy()
    if len(df_with_measures) == 0:
        return None

    df_plot = df_with_measures.nlargest(top_n, 'ROI')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color based on ROI value (green for positive, red for negative)
    colors = ['#27ae60' if roi > 0 else '#e74c3c' for roi in df_plot['ROI']]

    bars = ax.barh(range(len(df_plot)), df_plot['ROI'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{row['Risk ID']}: {row['Risk Description'][:35]}..."
                        if len(row['Risk Description']) > 35 else f"{row['Risk ID']}: {row['Risk Description']}"
                        for _, row in df_plot.iterrows()], fontsize=9)
    ax.invert_yaxis()

    # Add value labels on bars
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        ax.annotate(f'{row["ROI"]:.0f}%', xy=(row['ROI'], i), va='center',
                   fontsize=9, fontweight='bold',
                   xytext=(5 if row['ROI'] >= 0 else -5, 0), textcoords='offset points',
                   ha='left' if row['ROI'] >= 0 else 'right')

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Return on Investment (%)', fontsize=12)
    ax.set_title('Top Mitigation Measures by ROI', fontsize=14, fontweight='bold', color='#1F4E78')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def add_docx_cover_page(doc, confidence_level):
    """Add professional cover page to DOCX report"""
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Add spacing at top
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run('Risk Assessment Report')
    title_run.font.size = Pt(44)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(31, 78, 120)  # Professional dark blue
    title_run.font.name = 'Calibri'

    # Spacing
    doc.add_paragraph()

    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run('Monte Carlo Simulation & Probabilistic Risk Analysis')
    subtitle_run.font.size = Pt(18)
    subtitle_run.font.color.rgb = RGBColor(89, 89, 89)  # Gray
    subtitle_run.font.name = 'Calibri'

    # More spacing
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()

    # Horizontal line
    line = doc.add_paragraph('_' * 80)
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Report metadata in a nice box
    metadata = doc.add_paragraph()
    metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta_run = metadata.add_run(f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}\n')
    meta_run.font.size = Pt(12)
    meta_run.font.name = 'Calibri'
    meta_run.font.color.rgb = RGBColor(89, 89, 89)

    conf_run = metadata.add_run(f'Confidence Level: {confidence_level}')
    conf_run.font.size = Pt(14)
    conf_run.font.bold = True
    conf_run.font.name = 'Calibri'
    conf_run.font.color.rgb = RGBColor(31, 78, 120)

    # Page break
    doc.add_page_break()

def add_docx_executive_summary(doc, initial_stats, residual_stats, df, confidence_level):
    """Add executive summary section to DOCX report"""
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Section heading
    heading = doc.add_heading('Executive Summary', 1)
    for run in heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    # Get selected confidence values (probabilistic)
    initial_selected = get_confidence_value(initial_stats, confidence_level)
    residual_selected = get_confidence_value(residual_stats, confidence_level)
    risk_reduction_selected = initial_selected - residual_selected
    risk_reduction_pct = (risk_reduction_selected / initial_selected * 100) if initial_selected > 0 else 0

    # Calculate deterministic values (sum of expected values)
    initial_ev_total = df['Initial_EV'].sum()
    residual_ev_total = df['Residual_EV'].sum()
    deterministic_reduction = initial_ev_total - residual_ev_total
    deterministic_reduction_pct = (deterministic_reduction / initial_ev_total * 100) if initial_ev_total > 0 else 0
    total_mitigation_cost = df['Cost of Measures_Value'].sum()

    # Key findings paragraph
    intro = doc.add_paragraph()
    intro_bold = intro.add_run('Overview: ')
    intro_bold.bold = True
    intro_bold.font.name = 'Calibri'
    intro_bold.font.size = Pt(11)

    intro_text = intro.add_run(
        f'This risk assessment analyzes {len(df)} identified risks using Monte Carlo simulation '
        f'with {confidence_level} confidence level. The analysis provides both deterministic (expected value) '
        f'and probabilistic estimates of total risk exposure, evaluating the effectiveness of planned mitigation measures.'
    )
    intro_text.font.name = 'Calibri'
    intro_text.font.size = Pt(11)

    doc.add_paragraph()  # Spacing

    # ========== THREAT VS OPPORTUNITY ANALYSIS SECTION ==========
    # Calculate threat/opportunity metrics
    to_metrics = calculate_threat_opportunity_metrics(df)

    to_heading = doc.add_heading('Threat vs Opportunity Analysis (Expected Values)', 2)
    for run in to_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    to_intro = doc.add_paragraph()
    to_intro_run = to_intro.add_run(
        'In quantitative risk analysis, threats and opportunities are calculated separately. '
        'Threats represent potential costs (positive EV), while opportunities represent potential benefits (negative EV). '
        'Net Exposure = Threat EV + Opportunity EV.'
    )
    to_intro_run.font.name = 'Calibri'
    to_intro_run.font.size = Pt(10)
    to_intro_run.font.italic = True
    to_intro_run.font.color.rgb = RGBColor(89, 89, 89)

    # Create table for threat/opportunity metrics
    to_table = doc.add_table(rows=7, cols=3)

    to_table.rows[0].cells[0].text = 'Category'
    to_table.rows[0].cells[1].text = 'Initial (M CHF)'
    to_table.rows[0].cells[2].text = 'Residual (M CHF)'

    to_table_data = [
        (f'Threats ({to_metrics["threat_count"]} risks)',
         f'{to_metrics["threat_initial_ev"]/1e6:.2f}',
         f'{to_metrics["threat_residual_ev"]/1e6:.2f}'),
        (f'Opportunities ({to_metrics["opportunity_count"]} risks)',
         f'{to_metrics["opportunity_initial_ev"]/1e6:.2f}',
         f'{to_metrics["opportunity_residual_ev"]/1e6:.2f}'),
        ('Net Exposure',
         f'{to_metrics["net_initial_exposure"]/1e6:.2f}',
         f'{to_metrics["net_residual_exposure"]/1e6:.2f}'),
        ('Threat Reduction',
         f'{to_metrics["threat_reduction"]/1e6:.2f}',
         f'{(to_metrics["threat_reduction"]/to_metrics["threat_initial_ev"]*100) if to_metrics["threat_initial_ev"] > 0 else 0:.1f}%'),
        ('Opportunity Change',
         f'{to_metrics["opportunity_change"]/1e6:.2f}',
         'Enhanced' if to_metrics["opportunity_change"] < 0 else 'Unchanged'),
        ('Net Reduction',
         f'{to_metrics["net_reduction"]/1e6:.2f}',
         f'{(to_metrics["net_reduction"]/to_metrics["net_initial_exposure"]*100) if to_metrics["net_initial_exposure"] > 0 else 0:.1f}%'),
    ]

    for i, (cat, initial, residual) in enumerate(to_table_data, 1):
        to_table.rows[i].cells[0].text = cat
        to_table.rows[i].cells[1].text = initial
        to_table.rows[i].cells[2].text = residual

    format_table_executive(to_table, has_header=True, highlight_rows=[3, 6])

    doc.add_paragraph()  # Spacing

    # ========== DETERMINISTIC METRICS SECTION ==========
    det_heading = doc.add_heading('Overall Risk Metrics (Expected Values)', 2)
    for run in det_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    det_intro = doc.add_paragraph()
    det_intro_run = det_intro.add_run(
        'The deterministic analysis calculates risk exposure as the sum of individual expected values '
        '(Impact × Likelihood) for each risk. This provides a baseline estimate assuming average outcomes.'
    )
    det_intro_run.font.name = 'Calibri'
    det_intro_run.font.size = Pt(10)
    det_intro_run.font.italic = True
    det_intro_run.font.color.rgb = RGBColor(89, 89, 89)

    # Create table for deterministic metrics
    det_table = doc.add_table(rows=5, cols=2)

    det_table.rows[0].cells[0].text = 'Deterministic Metric'
    det_table.rows[0].cells[1].text = 'Value (Million CHF)'

    det_metrics = [
        ('Initial Risk Exposure (Σ EV)', f'{initial_ev_total/1e6:.2f}'),
        ('Residual Risk Exposure (Σ EV)', f'{residual_ev_total/1e6:.2f}'),
        ('Risk Reduction (Σ EV)', f'{deterministic_reduction/1e6:.2f} ({deterministic_reduction_pct:.1f}%)'),
        ('Total Mitigation Investment', f'{total_mitigation_cost/1e6:.2f}'),
    ]

    for i, (metric, value) in enumerate(det_metrics, 1):
        det_table.rows[i].cells[0].text = metric
        det_table.rows[i].cells[1].text = value

    format_table_executive(det_table, has_header=True, highlight_rows=[3])

    doc.add_paragraph()  # Spacing

    # ========== PROBABILISTIC METRICS SECTION ==========
    prob_heading = doc.add_heading(f'Probabilistic Risk Metrics ({confidence_level} Confidence)', 2)
    for run in prob_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    prob_intro = doc.add_paragraph()
    prob_intro_run = prob_intro.add_run(
        f'The probabilistic analysis uses Monte Carlo simulation to model portfolio-level risk exposure. '
        f'The {confidence_level} percentile indicates {confidence_level[1:]}% probability that actual exposure '
        f'will not exceed this value, accounting for risk correlations and uncertainty.'
    )
    prob_intro_run.font.name = 'Calibri'
    prob_intro_run.font.size = Pt(10)
    prob_intro_run.font.italic = True
    prob_intro_run.font.color.rgb = RGBColor(89, 89, 89)

    # Create table for probabilistic metrics
    prob_table = doc.add_table(rows=6, cols=2)

    prob_table.rows[0].cells[0].text = f'Probabilistic Metric ({confidence_level})'
    prob_table.rows[0].cells[1].text = 'Value (Million CHF)'

    prob_metrics = [
        (f'Initial Risk Exposure ({confidence_level})', f'{initial_selected/1e6:.2f}'),
        (f'Residual Risk Exposure ({confidence_level})', f'{residual_selected/1e6:.2f}'),
        (f'Portfolio Risk Reduction', f'{risk_reduction_selected/1e6:.2f} ({risk_reduction_pct:.1f}%)'),
        ('Total Mitigation Investment', f'{total_mitigation_cost/1e6:.2f}'),
        (f'Net Benefit ({confidence_level})', f'{(risk_reduction_selected - total_mitigation_cost)/1e6:.2f}'),
    ]

    for i, (metric, value) in enumerate(prob_metrics, 1):
        prob_table.rows[i].cells[0].text = metric
        prob_table.rows[i].cells[1].text = value

    format_table_executive(prob_table, has_header=True, highlight_rows=[3, 5])

    doc.add_paragraph()  # Spacing

    # Top risks section
    top_risks_heading = doc.add_heading('Critical Risks', 2)
    for run in top_risks_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    top_risks = df.nlargest(5, 'Initial_EV')
    top_risks_para = doc.add_paragraph()
    top_risks_para.add_run(
        f'The top 5 risks by expected value account for '
        f'{top_risks["Initial_EV"].sum() / df["Initial_EV"].sum() * 100:.1f}% '
        f'of total portfolio exposure:\n'
    )

    for idx, (_, risk) in enumerate(top_risks.iterrows(), 1):
        risk_para = doc.add_paragraph(style='List Number')
        risk_para.add_run(f'{risk["Risk ID"]}: ').bold = True
        risk_para.add_run(
            f'{risk["Risk Description"]} '
            f'(Expected Value: {risk["Initial_EV"]/1e6:.2f}M CHF)'
        )

    doc.add_page_break()

def add_docx_contingency_section(doc, initial_stats, residual_stats, df, confidence_level):
    """Add contingency allocation section with professional summary table and narrative"""
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Section heading
    heading = doc.add_heading('Project Risk Contingency Allocation', 1)
    for run in heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    # Get values
    residual_selected = get_confidence_value(residual_stats, confidence_level)
    total_mitigation_cost = df['Cost of Measures_Value'].sum()
    total_contingency = residual_selected + total_mitigation_cost

    # Introductory narrative
    intro = doc.add_paragraph()
    intro_bold = intro.add_run('Contingency Planning Framework: ')
    intro_bold.bold = True
    intro_bold.font.name = 'Calibri'
    intro_bold.font.size = Pt(11)

    intro_text = intro.add_run(
        f'Based on the {confidence_level} confidence level analysis, the following contingency allocation '
        f'is recommended to adequately cover identified project risks. This allocation should be reviewed '
        f'and approved by the Board of Directors before project commitment.'
    )
    intro_text.font.name = 'Calibri'
    intro_text.font.size = Pt(11)

    doc.add_paragraph()

    # ========== CONTINGENCY ALLOCATION TABLE ==========
    table_heading = doc.add_heading('Recommended Contingency Allocation', 2)
    for run in table_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    # Create the professional contingency table
    table = doc.add_table(rows=4, cols=3)

    # Headers
    table.rows[0].cells[0].text = 'Component'
    table.rows[0].cells[1].text = 'Amount (M CHF)'
    table.rows[0].cells[2].text = 'Description'

    # Data rows
    table_data = [
        ('Residual Risk Reserve', f'{residual_selected/1e6:.2f}',
         f'{confidence_level} probabilistic exposure after mitigation'),
        ('Mitigation Investment', f'{total_mitigation_cost/1e6:.2f}',
         'Total cost of planned risk mitigation measures'),
        ('TOTAL PROJECT CONTINGENCY', f'{total_contingency/1e6:.2f}',
         'Recommended allocation for BoD approval'),
    ]

    for i, (component, amount, description) in enumerate(table_data, 1):
        table.rows[i].cells[0].text = component
        table.rows[i].cells[1].text = amount
        table.rows[i].cells[2].text = description

    # Apply special contingency formatting
    format_table_contingency(table)

    doc.add_paragraph()

    # ========== PERCENTILE SELECTION NARRATIVE ==========
    narrative_heading = doc.add_heading('Understanding the Selected Confidence Level', 2)
    for run in narrative_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    # Confidence level explanation
    if confidence_level == 'P50':
        conf_explanation = (
            'The P50 (median) confidence level represents a 50% probability that actual risk exposure '
            'will not exceed this value. This is considered an optimistic estimate suitable for projects '
            'with high risk tolerance, strong risk management capabilities, or where cost constraints '
            'require accepting higher uncertainty. Organizations selecting P50 should maintain robust '
            'contingency access mechanisms and be prepared for potential cost overruns.'
        )
        profile_assessment = 'Optimistic / High Risk Tolerance'
        profile_color = RGBColor(230, 126, 34)  # Orange
    elif confidence_level == 'P80':
        conf_explanation = (
            'The P80 confidence level represents an 80% probability that actual risk exposure '
            'will not exceed this value. This is considered a balanced, moderately conservative estimate '
            'suitable for most commercial and infrastructure projects. P80 provides reasonable protection '
            'against adverse outcomes while avoiding excessive contingency that could impact project viability. '
            'This level is commonly adopted as industry best practice for major capital projects.'
        )
        profile_assessment = 'Balanced / Moderate Risk Profile'
        profile_color = RGBColor(41, 128, 185)  # Blue
    else:  # P95
        conf_explanation = (
            'The P95 confidence level represents a 95% probability that actual risk exposure '
            'will not exceed this value. This is a conservative estimate suitable for projects with '
            'low risk tolerance, regulatory constraints, fixed-price contracts, or critical infrastructure. '
            'While P95 provides high confidence in budget adequacy, it may result in higher contingency '
            'allocation that could affect project economics or competitiveness.'
        )
        profile_assessment = 'Conservative / Low Risk Tolerance'
        profile_color = RGBColor(39, 174, 96)  # Green

    conf_para = doc.add_paragraph()
    conf_para.add_run(conf_explanation).font.name = 'Calibri'

    doc.add_paragraph()

    # Risk profile box
    profile_para = doc.add_paragraph()
    profile_label = profile_para.add_run('Project Risk Profile Assessment: ')
    profile_label.bold = True
    profile_label.font.name = 'Calibri'
    profile_label.font.size = Pt(11)

    profile_value = profile_para.add_run(profile_assessment)
    profile_value.bold = True
    profile_value.font.name = 'Calibri'
    profile_value.font.size = Pt(12)
    profile_value.font.color.rgb = profile_color

    doc.add_paragraph()

    # ========== CONTINGENCY ALLOCATION GUIDANCE ==========
    guidance_heading = doc.add_heading('Contingency Allocation Guidance', 2)
    for run in guidance_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 120)

    guidance_intro = doc.add_paragraph()
    guidance_intro.add_run(
        'The recommended contingency should be allocated and managed according to the following principles:'
    ).font.name = 'Calibri'

    guidance_items = [
        ('Governance: ', 'Contingency release should require formal change control approval, with clear '
         'thresholds for project manager, steering committee, and Board authorization levels.'),
        ('Monitoring: ', 'Track contingency drawdown against risk realization. If drawdown exceeds '
         'forecast early in the project, escalate for review and potential replenishment.'),
        ('Risk-Based Release: ', 'Link contingency release to specific risk events rather than '
         'general cost growth. Maintain traceability between risks and contingency utilization.'),
        ('Periodic Review: ', 'Reassess risk exposure quarterly. As risks are retired or new risks '
         'emerge, adjust the contingency forecast and communicate changes to stakeholders.'),
        ('Residual Contingency: ', 'Unused contingency at project completion may be returned to '
         'corporate reserves or reallocated per organizational policy.')
    ]

    for title, content in guidance_items:
        item_para = doc.add_paragraph(style='List Bullet')
        item_title = item_para.add_run(title)
        item_title.bold = True
        item_title.font.name = 'Calibri'
        item_content = item_para.add_run(content)
        item_content.font.name = 'Calibri'

    doc.add_page_break()

def add_docx_risk_portfolio_section(doc, initial_stats, residual_stats, confidence_level, risk_matrix_img=None):
    """Add risk portfolio overview section with statistics and charts"""
    from docx.shared import Inches

    doc.add_heading('Risk Portfolio Overview', 1)

    # Statistical summary
    doc.add_heading('Statistical Summary', 2)

    # Create statistics table
    table = doc.add_table(rows=8, cols=3)

    # Headers
    table.rows[0].cells[0].text = 'Metric'
    table.rows[0].cells[1].text = 'Initial Risk (M CHF)'
    table.rows[0].cells[2].text = 'Residual Risk (M CHF)'

    # Data rows
    metrics = ['Mean', 'Median (P50)', 'P80', 'P95', 'Std Dev', 'Min', 'Max']
    stat_keys = ['mean', 'p50', 'p80', 'p95', 'std', 'min', 'max']

    highlight_row = None
    for i, (metric, key) in enumerate(zip(metrics, stat_keys), 1):
        # Highlight selected confidence level
        if (key == 'p50' and confidence_level == 'P50') or \
           (key == 'p80' and confidence_level == 'P80') or \
           (key == 'p95' and confidence_level == 'P95'):
            table.rows[i].cells[0].text = f'★ {metric} (SELECTED)'
            highlight_row = i
        else:
            table.rows[i].cells[0].text = metric

        table.rows[i].cells[1].text = f'{initial_stats[key]/1e6:.2f}'
        table.rows[i].cells[2].text = f'{residual_stats[key]/1e6:.2f}'

    # Apply professional formatting
    format_table_executive(table, has_header=True, highlight_rows=[highlight_row] if highlight_row else [])

    # Add risk matrix chart if provided
    if risk_matrix_img:
        doc.add_heading('Risk Matrix Visualization', 2)
        doc.add_picture(risk_matrix_img, width=Inches(6))

    doc.add_page_break()

def add_docx_monte_carlo_section(doc, n_simulations, cdf_img=None, histogram_img=None):
    """Add Monte Carlo simulation results section"""
    from docx.shared import Inches

    doc.add_heading('Monte Carlo Simulation Results', 1)

    intro = doc.add_paragraph()
    intro.add_run(
        f'The risk assessment used Monte Carlo simulation with {n_simulations:,} iterations '
        f'to model the probabilistic distribution of total risk exposure. This approach accounts '
        f'for the uncertainty in both risk occurrence and impact.'
    )

    # CDF plot
    if cdf_img:
        doc.add_heading('Cumulative Distribution Function', 2)
        desc = doc.add_paragraph()
        desc.add_run(
            'The CDF shows the probability that total risk exposure will be at or below a given value. '
            'The selected confidence level is highlighted in the chart.'
        )
        doc.add_picture(cdf_img, width=Inches(6))

    # Histogram
    if histogram_img:
        doc.add_heading('Risk Distribution', 2)
        desc = doc.add_paragraph()
        desc.add_run(
            'The histogram shows the frequency distribution of simulated total risk exposures, '
            'providing insight into the most likely outcomes.'
        )
        doc.add_picture(histogram_img, width=Inches(6))

    doc.add_page_break()

def add_docx_sensitivity_section(doc, sensitivity_df, pareto_img=None):
    """Add sensitivity analysis section"""
    from docx.shared import Inches

    doc.add_heading('Sensitivity Analysis', 1)

    intro = doc.add_paragraph()
    intro.add_run(
        'Sensitivity analysis identifies which risks contribute most to overall portfolio uncertainty. '
        'This helps prioritize risk management efforts on the risks that matter most.'
    )

    # Key findings
    top_risk = sensitivity_df.iloc[0]
    risks_80 = (sensitivity_df['Cumulative %'] <= 80).sum()

    findings = doc.add_paragraph()
    findings.add_run('Key Findings:\n').bold = True
    findings.add_run(
        f'• The top risk driver (Risk {top_risk["Risk ID"]}) accounts for {top_risk["Variance %"]:.1f}% of total variance\n'
        f'• Just {risks_80} risks drive 80% of the portfolio uncertainty\n'
        f'• This follows the Pareto principle (80/20 rule) for risk concentration\n'
    )

    # Pareto chart
    if pareto_img:
        doc.add_heading('Pareto Chart', 2)
        desc = doc.add_paragraph()
        desc.add_run(
            'The Pareto chart shows the cumulative contribution of each risk to total variance, '
            'helping identify the "vital few" risks that require focused attention.'
        )
        doc.add_picture(pareto_img, width=Inches(6.5))

    # Top 10 risks table
    doc.add_heading('Top 10 Risk Drivers', 2)

    table = doc.add_table(rows=11, cols=4)

    # Headers
    headers = ['Risk ID', 'Description', 'Variance %', 'Cumulative %']
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header

    # Data
    for i, (_, risk) in enumerate(sensitivity_df.head(10).iterrows(), 1):
        table.rows[i].cells[0].text = str(risk['Risk ID'])
        table.rows[i].cells[1].text = risk['Risk Description'][:50] + ('...' if len(risk['Risk Description']) > 50 else '')
        table.rows[i].cells[2].text = f'{risk["Variance %"]:.2f}%'
        table.rows[i].cells[3].text = f'{risk["Cumulative %"]:.1f}%'

    # Apply professional formatting
    format_table_executive(table, has_header=True)

    doc.add_page_break()

def add_docx_mitigation_section(doc, df_with_roi, roi_chart_img=None):
    """Add mitigation cost-benefit analysis section"""
    from docx.shared import Inches, Pt, RGBColor

    doc.add_heading('Mitigation Cost-Benefit Analysis', 1)

    # Calculate overall portfolio metrics (all risks where reduction occurred)
    df_with_reduction = df_with_roi[df_with_roi['Risk_Reduction'] > 0].copy()

    # Separate into risks with and without mitigation costs
    df_with_measures = df_with_reduction[df_with_reduction['Cost of Measures_Value'] > 0]
    df_without_measures = df_with_reduction[df_with_reduction['Cost of Measures_Value'] == 0]

    if len(df_with_reduction) > 0:
        # Overall metrics (all risks with reduction)
        total_reduction_all = df_with_reduction['Risk_Reduction'].sum()

        # Metrics for risks with mitigation measures
        total_reduction_with_cost = df_with_measures['Risk_Reduction'].sum() if len(df_with_measures) > 0 else 0
        total_cost = df_with_measures['Cost of Measures_Value'].sum() if len(df_with_measures) > 0 else 0

        # Metrics for risks without mitigation cost but still reduced
        total_reduction_no_cost = df_without_measures['Risk_Reduction'].sum() if len(df_without_measures) > 0 else 0

        net_benefit = total_reduction_all - total_cost

        summary = doc.add_paragraph()
        summary_title = summary.add_run('Overall Risk Reduction Summary:\n')
        summary_title.bold = True
        summary_title.font.name = 'Calibri'
        summary_title.font.size = Pt(11)

        summary_text = summary.add_run(
            f'• Total Risk Reduction (All Risks): {total_reduction_all/1e6:.2f} Million CHF\n'
            f'  - With Mitigation Measures ({len(df_with_measures)} risks): {total_reduction_with_cost/1e6:.2f} Million CHF\n'
            f'  - Natural Reduction ({len(df_without_measures)} risks): {total_reduction_no_cost/1e6:.2f} Million CHF\n'
            f'• Total Mitigation Cost: {total_cost/1e6:.2f} Million CHF\n'
            f'• Net Benefit: {net_benefit/1e6:.2f} Million CHF\n'
        )
        summary_text.font.name = 'Calibri'
        summary_text.font.size = Pt(10)

        if total_cost > 0:
            bc_ratio_text = summary.add_run(f'• Benefit/Cost Ratio: {total_reduction_with_cost/total_cost:.2f}\n')
            bc_ratio_text.font.name = 'Calibri'
            bc_ratio_text.font.size = Pt(10)

        # Add explanation note
        doc.add_paragraph()
        note = doc.add_paragraph()
        note_run = note.add_run(
            'Note: "Total Risk Reduction" includes all risks where reduction occurred. '
            '"Natural Reduction" refers to risks that reduced without explicit mitigation costs '
            '(e.g., due to changed circumstances, passive controls, or re-assessment). '
            'The Benefit/Cost Ratio only considers risks with active mitigation measures. '
            'These values represent the sum of individual risk Expected Values (Impact × Likelihood), '
            'which differs from portfolio-level Monte Carlo percentiles in the Executive Summary.'
        )
        note_run.font.size = Pt(9)
        note_run.font.name = 'Calibri'
        note_run.font.italic = True
        note_run.font.color.rgb = RGBColor(89, 89, 89)

        # ROI chart
        if roi_chart_img:
            doc.add_heading('Return on Investment', 2)
            doc.add_picture(roi_chart_img, width=Inches(6))

        # Top ROI opportunities table (only for risks with mitigation costs)
        if len(df_with_measures) > 0:
            doc.add_heading('Top 10 Mitigation Opportunities (Active Measures)', 2)

            # Add clarifying note
            roi_note = doc.add_paragraph()
            roi_note_run = roi_note.add_run('This table shows risks with active mitigation measures (Cost > 0). ')
            roi_note_run.font.size = Pt(9)
            roi_note_run.font.name = 'Calibri'
            roi_note_run.font.italic = True
            roi_note_run.font.color.rgb = RGBColor(89, 89, 89)

            top_roi = df_with_measures.nlargest(10, 'ROI')

            table = doc.add_table(rows=min(len(top_roi) + 1, 11), cols=5)

            # Headers
            headers = ['Risk ID', 'Description', 'Risk Reduction (M CHF)', 'Cost (M CHF)', 'ROI %']
            for i, header in enumerate(headers):
                table.rows[0].cells[i].text = header

            # Data
            for i, (_, risk) in enumerate(top_roi.iterrows(), 1):
                table.rows[i].cells[0].text = str(risk['Risk ID'])
                table.rows[i].cells[1].text = risk['Risk Description'][:40] + ('...' if len(risk['Risk Description']) > 40 else '')
                table.rows[i].cells[2].text = f'{risk["Risk_Reduction"]/1e6:.2f}'
                table.rows[i].cells[3].text = f'{risk["Cost of Measures_Value"]/1e6:.2f}'
                table.rows[i].cells[4].text = f'{risk["ROI"]:.1f}%'

            # Apply professional formatting
            format_table_executive(table, has_header=True)
    else:
        no_reduction_para = doc.add_paragraph()
        no_reduction_text = no_reduction_para.add_run('No risk reduction occurred in the portfolio. All risks maintained the same expected values.')
        no_reduction_text.font.name = 'Calibri'
        no_reduction_text.font.size = Pt(11)
        no_reduction_text.font.italic = True

    doc.add_page_break()

def add_docx_risk_register_appendix(doc, df):
    """Add full risk register as appendix"""
    from docx.shared import Pt

    doc.add_heading('Appendix A: Risk Register', 1)

    # Create table
    display_cols = ['Risk ID', 'Risk Description', 'Initial risk_Value', 'Initial_Likelihood',
                   'Residual risk_Value', 'Residual_Likelihood', 'Cost of Measures_Value']

    table = doc.add_table(rows=len(df) + 1, cols=len(display_cols))

    # Headers
    headers = ['Risk ID', 'Description', 'Initial Risk (CHF)', 'Initial Likelihood',
               'Residual Risk (CHF)', 'Residual Likelihood', 'Mitigation Cost (CHF)']
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header

    # Data
    for i, (_, risk) in enumerate(df.iterrows(), 1):
        table.rows[i].cells[0].text = str(risk['Risk ID'])
        table.rows[i].cells[1].text = risk['Risk Description'][:60] + ('...' if len(risk['Risk Description']) > 60 else '')
        table.rows[i].cells[2].text = f'{risk["Initial risk_Value"]:,.0f}'
        table.rows[i].cells[3].text = f'{risk["Initial_Likelihood"]:.1%}'
        table.rows[i].cells[4].text = f'{risk["Residual risk_Value"]:,.0f}'
        table.rows[i].cells[5].text = f'{risk["Residual_Likelihood"]:.1%}'
        table.rows[i].cells[6].text = f'{risk["Cost of Measures_Value"]:,.0f}'

    # Apply professional formatting
    format_table_executive(table, has_header=True)

    doc.add_paragraph()  # Spacing
    note = doc.add_paragraph()
    note_run = note.add_run('Note: This table contains the complete risk register with all identified risks.')
    note_run.font.size = Pt(9)
    note_run.font.name = 'Calibri'
    note_run.font.italic = True

    doc.add_page_break()

def add_docx_methodology_appendix(doc, n_simulations, confidence_level):
    """Add methodology explanation appendix"""

    doc.add_heading('Appendix B: Methodology', 1)

    # Monte Carlo explanation
    doc.add_heading('Monte Carlo Simulation', 2)
    mc_para = doc.add_paragraph()
    mc_para.add_run(
        f'This analysis uses Monte Carlo simulation with {n_simulations:,} iterations to model '
        f'the probabilistic distribution of total risk exposure. For each iteration:\n'
    )

    steps = [
        'Each risk is evaluated independently based on its likelihood of occurrence',
        'Risks that "occur" in the simulation contribute their impact to the total',
        'The simulation aggregates all occurring risks to calculate total portfolio exposure',
        'This process repeats thousands of times to build a statistical distribution'
    ]

    for step in steps:
        doc.add_paragraph(step, style='List Bullet')

    # Confidence level explanation
    doc.add_heading('Confidence Levels', 2)
    conf_para = doc.add_paragraph()
    conf_para.add_run(
        f'The selected {confidence_level} confidence level represents a percentile of the risk distribution:\n'
    )

    conf_meanings = {
        'P50': 'There is a 50% probability that actual risk exposure will be at or below this value (median/expected value)',
        'P80': 'There is an 80% probability that actual risk exposure will be at or below this value (moderately conservative)',
        'P95': 'There is a 95% probability that actual risk exposure will be at or below this value (very conservative, suitable for contingency planning)'
    }

    for level, meaning in conf_meanings.items():
        marker = ' ★ (Selected)' if level == confidence_level else ''
        para = doc.add_paragraph(style='List Bullet')
        para.add_run(f'{level}{marker}: ').bold = True
        para.add_run(meaning)

    # Sensitivity analysis explanation
    doc.add_heading('Sensitivity Analysis', 2)
    sens_para = doc.add_paragraph()
    sens_para.add_run(
        'Sensitivity analysis uses variance decomposition to identify which risks contribute most '
        'to overall portfolio uncertainty. The methodology:\n'
    )

    sens_steps = [
        'Run baseline simulation with all risks',
        'For each risk, run simulation with that risk removed',
        'Calculate variance reduction when the risk is removed',
        'Risks with higher variance contribution are the main drivers of uncertainty'
    ]

    for step in sens_steps:
        doc.add_paragraph(step, style='List Bullet')

def generate_docx_report(initial_stats, residual_stats, df, df_with_roi, sensitivity_df,
                        initial_results, residual_results, n_simulations, confidence_level):
    """
    Generate comprehensive DOCX report with embedded charts

    Returns:
        BytesIO object containing the Word document
    """
    from docx import Document

    doc = Document()

    # Add cover page
    add_docx_cover_page(doc, confidence_level)

    # Add executive summary
    add_docx_executive_summary(doc, initial_stats, residual_stats, df, confidence_level)

    # Add contingency allocation section
    add_docx_contingency_section(doc, initial_stats, residual_stats, df, confidence_level)

    # Generate and embed charts using matplotlib (reliable, no external dependencies)
    # Side-by-side comparison charts showing initial vs residual
    with st.spinner("Generating risk matrix comparison chart..."):
        risk_matrix_img = create_matplotlib_risk_matrix_combined(df)

    add_docx_risk_portfolio_section(doc, initial_stats, residual_stats, confidence_level, risk_matrix_img)

    # Monte Carlo section with side-by-side comparison charts
    with st.spinner("Generating Monte Carlo comparison charts..."):
        cdf_img = create_matplotlib_cdf_combined(initial_results, initial_stats, residual_results, residual_stats, confidence_level)
        hist_img = create_matplotlib_histogram_combined(initial_results, initial_stats, residual_results, residual_stats)

    add_docx_monte_carlo_section(doc, n_simulations, cdf_img, hist_img)

    # Sensitivity section with Pareto chart
    with st.spinner("Generating sensitivity analysis chart..."):
        pareto_img = create_matplotlib_pareto(sensitivity_df, top_n=20)

    add_docx_sensitivity_section(doc, sensitivity_df, pareto_img)

    # Mitigation section with ROI chart
    with st.spinner("Generating mitigation analysis chart..."):
        roi_img = create_matplotlib_roi_chart(df_with_roi, top_n=20)

    add_docx_mitigation_section(doc, df_with_roi, roi_img)

    # Appendices
    add_docx_risk_register_appendix(doc, df)
    add_docx_methodology_appendix(doc, n_simulations, confidence_level)

    # Save to BytesIO
    docx_bytes = io.BytesIO()
    doc.save(docx_bytes)
    docx_bytes.seek(0)

    return docx_bytes

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
            st.session_state['confidence_level'] = confidence_level
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
        confidence_level = st.session_state.get('confidence_level', 'P95')
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Dashboard",
            "📈 Risk Matrix & Heatmap",
            "🔍 Sensitivity Analysis",
            "🎲 Monte Carlo Results",
            "💰 Cost-Benefit Analysis",
            "🔮 What-If Scenarios",
            "📋 Risk Register",
            "📥 Export"
        ])
        
        with tab1:
            st.header("Risk Portfolio Dashboard")

            # Display active confidence level
            st.info(f"📊 **Active Confidence Level: {confidence_level}** | Showing risk exposure at {confidence_level} percentile")

            # Calculate threat/opportunity metrics
            to_metrics = calculate_threat_opportunity_metrics(df_with_roi)

            # Primary metrics row - Threat EV, Opportunity EV, Net Exposure
            st.subheader("📊 Risk Exposure Summary (Expected Value)")

            primary_col1, primary_col2, primary_col3, primary_col4 = st.columns(4)

            with primary_col1:
                st.metric(
                    "Threat EV (Initial)",
                    f"{to_metrics['threat_initial_ev']/1e6:.2f}M CHF",
                    help=f"{to_metrics['threat_count']} threats identified"
                )
                st.metric(
                    "Threat EV (Residual)",
                    f"{to_metrics['threat_residual_ev']/1e6:.2f}M CHF",
                    delta=f"{-to_metrics['threat_reduction']/1e6:.2f}M" if to_metrics['threat_reduction'] > 0 else None
                )

            with primary_col2:
                # Opportunity EV is negative, so we show absolute value with a minus sign for clarity
                st.metric(
                    "Opportunity EV (Initial)",
                    f"{to_metrics['opportunity_initial_ev']/1e6:.2f}M CHF",
                    help=f"{to_metrics['opportunity_count']} opportunities identified (negative = benefit)"
                )
                st.metric(
                    "Opportunity EV (Residual)",
                    f"{to_metrics['opportunity_residual_ev']/1e6:.2f}M CHF",
                    delta=f"{-to_metrics['opportunity_change']/1e6:.2f}M" if to_metrics['opportunity_change'] != 0 else None
                )

            with primary_col3:
                # Net Exposure = Threat EV + Opportunity EV (where Opportunity is negative)
                st.metric(
                    "Net Exposure (Initial)",
                    f"{to_metrics['net_initial_exposure']/1e6:.2f}M CHF",
                    help="Threat EV + Opportunity EV"
                )
                st.metric(
                    "Net Exposure (Residual)",
                    f"{to_metrics['net_residual_exposure']/1e6:.2f}M CHF",
                    delta=f"{-to_metrics['net_reduction']/1e6:.2f}M" if to_metrics['net_reduction'] != 0 else None
                )

            with primary_col4:
                st.metric("Total Risks", len(df))
                st.metric(
                    "Threats / Opportunities",
                    f"{to_metrics['threat_count']} / {to_metrics['opportunity_count']}"
                )

            st.markdown("---")

            # Monte Carlo Results - Key metrics
            st.subheader("🎲 Monte Carlo Simulation Results")

            col1, col2, col3, col4 = st.columns(4)

            # Get confidence-specific values
            initial_confidence_value = get_confidence_value(initial_stats, confidence_level)
            residual_confidence_value = get_confidence_value(residual_stats, confidence_level)

            with col1:
                st.metric(f"Initial Risk ({confidence_level})", f"{initial_confidence_value/1e6:.2f}M CHF")
                st.metric("Initial Risk (Mean)", f"{initial_stats['mean']/1e6:.2f}M CHF")

            with col2:
                st.metric(f"Residual Risk ({confidence_level})", f"{residual_confidence_value/1e6:.2f}M CHF")
                st.metric("Residual Risk (Mean)", f"{residual_stats['mean']/1e6:.2f}M CHF")

            with col3:
                risk_reduction = ((initial_confidence_value - residual_confidence_value) / initial_confidence_value * 100) if initial_confidence_value != 0 else 0
                st.metric(f"Risk Reduction ({confidence_level})", f"{risk_reduction:.1f}%")
                total_mitigation_cost = df['Cost of Measures_Value'].sum()
                st.metric("Total Mitigation Cost", f"{total_mitigation_cost/1e6:.2f}M CHF")

            with col4:
                st.metric("Risks with Schedule Impact", df['Schedule_Impact'].sum())
                net_benefit = (initial_confidence_value - residual_confidence_value) - total_mitigation_cost
                st.metric("Net Benefit", f"{net_benefit/1e6:.2f}M CHF")

            st.markdown("---")
            
            # Tornado chart
            st.subheader("Top Risks by Expected Value")
            tornado_fig = create_tornado_chart(df_with_roi, top_n=15)
            st.plotly_chart(tornado_fig, use_container_width=True)
            
            # Statistics comparison table
            st.subheader("Statistical Summary")

            # Create metric names with indicator for selected confidence level
            metric_labels = {
                'Mean': 'Mean',
                'P50': 'Median (P50)',
                'P80': 'P80',
                'P95': 'P95',
                'Std Dev': 'Std Dev',
                'Min': 'Min',
                'Max': 'Max'
            }

            # Add visual indicator (★) to the active confidence level
            metrics_list = ['Mean', 'P50', 'P80', 'P95', 'Std Dev', 'Min', 'Max']
            display_metrics = []
            for metric in metrics_list:
                if metric == confidence_level:
                    display_metrics.append(f"★ {metric_labels.get(metric, metric)} (ACTIVE)")
                else:
                    display_metrics.append(metric_labels.get(metric, metric))

            stats_df = pd.DataFrame({
                'Metric': display_metrics,
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
                cdf_initial = create_cdf_plot(initial_results, initial_stats, 'initial', confidence_level)
                st.plotly_chart(cdf_initial, use_container_width=True)

            with col2:
                st.subheader("Residual Risk CDF")
                cdf_residual = create_cdf_plot(residual_results, residual_stats, 'residual', confidence_level)
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

                # Sankey Diagram Section
                st.markdown("---")
                st.subheader("🔄 Risk Mitigation Flow (Sankey Diagram)")

                st.markdown("""
                The Sankey diagram below visualizes how risk flows through the mitigation process:
                - **Threats** (red tones): Potential costs/losses that flow from initial → reduced → residual
                - **Opportunities** (green/teal tones): Potential benefits that offset the threat exposure
                - **Net Exposure**: Threat EV minus Opportunity EV (shown in purple)
                """)

                # Sankey view selector
                sankey_view = st.radio(
                    "Select View:",
                    options=["Threats & Opportunities", "By Schedule Impact"],
                    horizontal=True,
                    help="Choose how to visualize the risk flow breakdown"
                )

                if sankey_view == "Threats & Opportunities":
                    sankey_fig = create_risk_sankey(df_with_roi, initial_stats, residual_stats, confidence_level)
                else:
                    sankey_fig = create_risk_sankey_detailed(df_with_roi)

                st.plotly_chart(sankey_fig, use_container_width=True)

                # Summary statistics below Sankey - now using threat/opportunity breakdown
                st.markdown("##### Flow Summary: Threats vs Opportunities")

                # Calculate threat/opportunity metrics
                to_flow_metrics = calculate_threat_opportunity_metrics(df_with_roi)

                flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)

                with flow_col1:
                    st.metric(
                        "Threat EV (Initial)",
                        f"{to_flow_metrics['threat_initial_ev']/1e6:.2f}M CHF",
                        help=f"{to_flow_metrics['threat_count']} threats"
                    )
                    st.metric(
                        "Threat EV (Residual)",
                        f"{to_flow_metrics['threat_residual_ev']/1e6:.2f}M CHF"
                    )

                with flow_col2:
                    st.metric(
                        "Opportunity EV (Initial)",
                        f"{to_flow_metrics['opportunity_initial_ev']/1e6:.2f}M CHF",
                        help=f"{to_flow_metrics['opportunity_count']} opportunities (negative = benefit)"
                    )
                    st.metric(
                        "Opportunity EV (Residual)",
                        f"{to_flow_metrics['opportunity_residual_ev']/1e6:.2f}M CHF"
                    )

                with flow_col3:
                    st.metric(
                        "Net Exposure (Initial)",
                        f"{to_flow_metrics['net_initial_exposure']/1e6:.2f}M CHF",
                        help="Threat EV + Opportunity EV"
                    )
                    st.metric(
                        "Net Exposure (Residual)",
                        f"{to_flow_metrics['net_residual_exposure']/1e6:.2f}M CHF"
                    )

                with flow_col4:
                    threat_reduction_pct = (to_flow_metrics['threat_reduction'] / to_flow_metrics['threat_initial_ev'] * 100) if to_flow_metrics['threat_initial_ev'] > 0 else 0
                    st.metric(
                        "Threat Reduction",
                        f"{to_flow_metrics['threat_reduction']/1e6:.2f}M CHF",
                        delta=f"-{threat_reduction_pct:.1f}%"
                    )
                    mitigation_efficiency = (to_flow_metrics['threat_reduction'] / total_cost * 100) if total_cost > 0 else 0
                    st.metric("Mitigation Efficiency", f"{mitigation_efficiency:.0f}%",
                             help="Threat reduced per CHF spent")

            else:
                st.warning("No risks with mitigation measures found in the register.")

        with tab6:
            st.header("🔮 What-If Scenario Analysis")

            st.markdown("""
            Explore how changes to individual risks affect your overall portfolio exposure.
            Create scenarios by adjusting risk likelihood and/or impact values, then compare against the baseline.
            """)

            # Initialize scenario adjustments in session state
            if 'scenario_adjustments' not in st.session_state:
                st.session_state.scenario_adjustments = {}
            if 'scenario_results' not in st.session_state:
                st.session_state.scenario_results = None

            # Scenario configuration
            col_name, col_type = st.columns([2, 1])

            with col_name:
                scenario_name = st.text_input(
                    "Scenario Name",
                    value="Custom Scenario",
                    help="Give your scenario a descriptive name"
                )

            with col_type:
                scenario_type = st.selectbox(
                    "Scenario Type",
                    options=["Custom", "Optimistic (-20%)", "Pessimistic (+20%)", "Best Case (-50%)", "Worst Case (+50%)"],
                    help="Quick presets or create a custom scenario"
                )

            st.markdown("---")

            # Handle preset scenarios
            if scenario_type != "Custom":
                st.subheader("📋 Preset Scenario")

                preset_adjustments = {}
                if scenario_type == "Optimistic (-20%)":
                    factor = 0.8
                    description = "All risks reduced by 20%"
                elif scenario_type == "Pessimistic (+20%)":
                    factor = 1.2
                    description = "All risks increased by 20%"
                elif scenario_type == "Best Case (-50%)":
                    factor = 0.5
                    description = "All risks reduced by 50%"
                else:  # Worst Case
                    factor = 1.5
                    description = "All risks increased by 50%"

                st.info(f"**{scenario_type}**: {description}")

                # Apply factor to all risks
                for _, row in df.iterrows():
                    risk_id = row['Risk ID']
                    preset_adjustments[risk_id] = {
                        'likelihood': min(row['Initial_Likelihood'] * factor, 1.0),
                        'impact': row['Initial risk_Value'] * factor
                    }

                if st.button("Apply Preset & Run Simulation", type="primary", key="preset_btn"):
                    with st.spinner("Running scenario simulation..."):
                        scenario_results, _ = run_scenario_monte_carlo(
                            df, preset_adjustments, n_simulations, 'initial'
                        )
                        scenario_stats = calculate_statistics(scenario_results)

                        st.session_state.scenario_results = {
                            'name': f"{scenario_type}",
                            'results': scenario_results,
                            'stats': scenario_stats,
                            'adjustments': preset_adjustments
                        }
                        st.success("Scenario simulation complete!")
                        st.rerun()

            else:
                # Custom scenario builder
                st.subheader("📝 Custom Scenario Builder")

                # Risk selection
                st.markdown("**Select risks to modify:**")

                # Get top risks by expected value for quick selection
                top_risks = df.nlargest(10, 'Initial_EV')[['Risk ID', 'Risk Description', 'Initial risk_Value', 'Initial_Likelihood']].copy()

                selected_risks = st.multiselect(
                    "Choose risks to adjust",
                    options=df['Risk ID'].tolist(),
                    default=list(st.session_state.scenario_adjustments.keys()),
                    help="Select one or more risks to modify in this scenario"
                )

                # Display adjustment controls for selected risks
                if selected_risks:
                    st.markdown("**Adjust risk parameters:**")

                    new_adjustments = {}

                    for risk_id in selected_risks:
                        risk_row = df[df['Risk ID'] == risk_id].iloc[0]

                        with st.expander(f"📌 {risk_id}: {risk_row['Risk Description'][:60]}...", expanded=True):
                            col1, col2, col3 = st.columns([1, 1, 1])

                            # Get current adjustment or original value
                            current_adjustment = st.session_state.scenario_adjustments.get(risk_id, {})
                            orig_likelihood = risk_row['Initial_Likelihood']
                            orig_impact = risk_row['Initial risk_Value']

                            with col1:
                                st.markdown("**Original Values:**")
                                st.write(f"Likelihood: {orig_likelihood*100:.1f}%")
                                st.write(f"Impact: {orig_impact/1e6:.2f}M CHF")
                                st.write(f"EV: {(orig_likelihood * orig_impact)/1e6:.2f}M CHF")

                            with col2:
                                new_likelihood = st.slider(
                                    f"New Likelihood (%)",
                                    min_value=0,
                                    max_value=100,
                                    value=int(current_adjustment.get('likelihood', orig_likelihood) * 100),
                                    key=f"lik_{risk_id}"
                                ) / 100

                            with col3:
                                new_impact = st.number_input(
                                    f"New Impact (M CHF)",
                                    min_value=0.0,
                                    max_value=float(df['Initial risk_Value'].max() / 1e6 * 2),
                                    value=float(current_adjustment.get('impact', orig_impact) / 1e6),
                                    step=0.1,
                                    key=f"imp_{risk_id}"
                                ) * 1e6

                            # Store adjustment
                            new_adjustments[risk_id] = {
                                'likelihood': new_likelihood,
                                'impact': new_impact
                            }

                            # Show change summary
                            new_ev = new_likelihood * new_impact
                            orig_ev = orig_likelihood * orig_impact
                            ev_change = new_ev - orig_ev
                            ev_change_pct = (ev_change / orig_ev * 100) if orig_ev > 0 else 0

                            if ev_change < 0:
                                st.success(f"📉 EV Change: {ev_change/1e6:+.2f}M CHF ({ev_change_pct:+.1f}%)")
                            elif ev_change > 0:
                                st.error(f"📈 EV Change: {ev_change/1e6:+.2f}M CHF ({ev_change_pct:+.1f}%)")
                            else:
                                st.info("No change in Expected Value")

                    # Update session state
                    st.session_state.scenario_adjustments = new_adjustments

                    # Run simulation button
                    st.markdown("---")
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

                    with col_btn2:
                        if st.button("🚀 Run Scenario Simulation", type="primary", use_container_width=True):
                            with st.spinner("Running Monte Carlo simulation for scenario..."):
                                scenario_results, _ = run_scenario_monte_carlo(
                                    df, new_adjustments, n_simulations, 'initial'
                                )
                                scenario_stats = calculate_statistics(scenario_results)

                                st.session_state.scenario_results = {
                                    'name': scenario_name,
                                    'results': scenario_results,
                                    'stats': scenario_stats,
                                    'adjustments': new_adjustments
                                }
                                st.success("Scenario simulation complete!")
                                st.rerun()

                else:
                    st.info("👆 Select one or more risks above to begin building your scenario.")

                    # Show top risks as suggestions
                    st.markdown("**💡 Suggested risks to explore (Top 10 by Expected Value):**")
                    suggestion_df = top_risks.copy()
                    suggestion_df['Initial risk_Value'] = suggestion_df['Initial risk_Value'].apply(lambda x: f"{x/1e6:.2f}M CHF")
                    suggestion_df['Initial_Likelihood'] = suggestion_df['Initial_Likelihood'].apply(lambda x: f"{x*100:.1f}%")
                    suggestion_df.columns = ['Risk ID', 'Description', 'Impact', 'Likelihood']
                    st.dataframe(suggestion_df, use_container_width=True, hide_index=True)

            # Display results if available
            st.markdown("---")
            st.subheader("📊 Scenario Comparison Results")

            if st.session_state.scenario_results is not None:
                scenario_data = st.session_state.scenario_results
                scenario_stats = scenario_data['stats']
                scenario_results = scenario_data['results']
                scenario_display_name = scenario_data['name']

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                baseline_conf_val = get_confidence_value(initial_stats, confidence_level)
                scenario_conf_val = get_confidence_value(scenario_stats, confidence_level)
                change = scenario_conf_val - baseline_conf_val
                change_pct = (change / baseline_conf_val * 100) if baseline_conf_val > 0 else 0

                with col1:
                    st.metric(
                        f"Baseline ({confidence_level})",
                        f"{baseline_conf_val/1e6:.2f}M CHF"
                    )

                with col2:
                    st.metric(
                        f"Scenario ({confidence_level})",
                        f"{scenario_conf_val/1e6:.2f}M CHF",
                        delta=f"{change/1e6:+.2f}M CHF"
                    )

                with col3:
                    st.metric(
                        "Change",
                        f"{change_pct:+.1f}%",
                        delta="Reduced" if change < 0 else "Increased",
                        delta_color="normal" if change < 0 else "inverse"
                    )

                with col4:
                    risks_modified = len(scenario_data.get('adjustments', {}))
                    st.metric("Risks Modified", risks_modified)

                # Comparison charts
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    comparison_fig = create_scenario_comparison_chart(
                        initial_stats, scenario_stats, scenario_display_name, confidence_level
                    )
                    st.plotly_chart(comparison_fig, use_container_width=True)

                with chart_col2:
                    cdf_fig = create_scenario_cdf_comparison(
                        initial_results, scenario_results, initial_stats, scenario_stats,
                        scenario_display_name, confidence_level
                    )
                    st.plotly_chart(cdf_fig, use_container_width=True)

                # Modified risks table
                if scenario_data.get('adjustments'):
                    st.subheader("📋 Modified Risks Summary")
                    impact_table = create_scenario_impact_table(df, scenario_data['adjustments'], 'initial')
                    if impact_table is not None:
                        st.dataframe(impact_table, use_container_width=True, hide_index=True)

                # Clear results button
                col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
                with col_clear2:
                    if st.button("🗑️ Clear Scenario Results", use_container_width=True):
                        st.session_state.scenario_results = None
                        st.session_state.scenario_adjustments = {}
                        st.rerun()

            else:
                st.info("👆 Configure and run a scenario above to see comparison results here.")

                # Show quick summary of baseline
                st.markdown("**Current Baseline Summary:**")
                baseline_summary = pd.DataFrame({
                    'Metric': ['Mean', 'P50 (Median)', 'P80', 'P95'],
                    'Initial Risk (M CHF)': [
                        f"{initial_stats['mean']/1e6:.2f}",
                        f"{initial_stats['p50']/1e6:.2f}",
                        f"{initial_stats['p80']/1e6:.2f}",
                        f"{initial_stats['p95']/1e6:.2f}"
                    ]
                })
                st.dataframe(baseline_summary, use_container_width=True, hide_index=True)

        with tab7:
            st.header("Risk Register")

            # Display portfolio-level confidence information
            st.info(f"📊 **Portfolio Analysis using {confidence_level} confidence level** | "
                   f"Total Portfolio Risk ({confidence_level}): Initial = {initial_confidence_value/1e6:.2f}M CHF, "
                   f"Residual = {residual_confidence_value/1e6:.2f}M CHF")

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

        with tab8:
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

            # Professional DOCX Report
            st.subheader("📄 Professional DOCX Report")
            st.write("Generate a comprehensive Word document with executive summary, embedded charts, and detailed analysis.")

            col_docx1, col_docx2, col_docx3 = st.columns([1, 2, 1])

            with col_docx2:
                if st.button("🎨 Generate Professional DOCX Report", type="primary", use_container_width=True):
                    with st.spinner("📄 Generating comprehensive DOCX report... This may take 30-60 seconds"):
                        try:
                            # Get sensitivity data
                            sensitivity_df = st.session_state.get('sensitivity_df')

                            if sensitivity_df is None:
                                st.error("❌ Sensitivity analysis data not found. Please run the simulation first.")
                            else:
                                # Generate the DOCX report
                                docx_report = generate_docx_report(
                                    initial_stats=initial_stats,
                                    residual_stats=residual_stats,
                                    df=df,
                                    df_with_roi=df_with_roi,
                                    sensitivity_df=sensitivity_df,
                                    initial_results=initial_results,
                                    residual_results=residual_results,
                                    n_simulations=n_simulations,
                                    confidence_level=confidence_level
                                )

                                st.success("✅ DOCX report generated successfully!")

                                st.download_button(
                                    label="📥 Download Professional Report (DOCX)",
                                    data=docx_report,
                                    file_name=f"risk_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True
                                )

                        except Exception as e:
                            st.error(f"❌ Error generating DOCX report: {str(e)}")
                            st.write("Please ensure all required libraries are installed: `pip install python-docx`")
                            import traceback
                            st.code(traceback.format_exc())

            st.info("""
                **📋 Report Contents:**
                - ✅ Professional cover page with branding
                - ✅ Executive summary with key metrics and top risks
                - ✅ Risk portfolio overview with embedded charts
                - ✅ Monte Carlo simulation results with CDF and histograms
                - ✅ Sensitivity analysis with Pareto chart
                - ✅ Mitigation cost-benefit analysis with ROI charts
                - ✅ Full risk register appendix
                - ✅ Methodology explanation appendix
            """)

            st.markdown("---")

            # Summary report
            st.subheader("📋 Summary Report")

            # Get selected confidence values
            initial_selected = get_confidence_value(initial_stats, confidence_level)
            residual_selected = get_confidence_value(residual_stats, confidence_level)
            risk_reduction_selected = initial_selected - residual_selected
            risk_reduction_pct = (risk_reduction_selected / initial_selected * 100)
            net_benefit_selected = risk_reduction_selected - df['Cost of Measures_Value'].sum()

            # Mark selected confidence level in the report
            p50_marker = " ★ (SELECTED)" if confidence_level == "P50" else ""
            p80_marker = " ★ (SELECTED)" if confidence_level == "P80" else ""
            p95_marker = " ★ (SELECTED)" if confidence_level == "P95" else ""

            report = f"""
# Risk Assessment Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Monte Carlo Simulations:** {n_simulations:,}
**Selected Confidence Level:** {confidence_level}

## Executive Summary

### Primary Risk Assessment ({confidence_level} - Selected Confidence Level)
- **Initial Risk Exposure:** {initial_selected/1e6:.2f} Million CHF
- **Residual Risk Exposure (After Mitigation):** {residual_selected/1e6:.2f} Million CHF
- **Total Risk Reduction:** {risk_reduction_selected/1e6:.2f} Million CHF ({risk_reduction_pct:.1f}%)
- **Total Mitigation Investment:** {df['Cost of Measures_Value'].sum()/1e6:.2f} Million CHF
- **Net Benefit:** {net_benefit_selected/1e6:.2f} Million CHF

### Initial Risk Exposure (All Confidence Levels)
- **P50 (Median):** {initial_stats['p50']/1e6:.2f} Million CHF{p50_marker}
- **P80:** {initial_stats['p80']/1e6:.2f} Million CHF{p80_marker}
- **P95:** {initial_stats['p95']/1e6:.2f} Million CHF{p95_marker}

### Residual Risk Exposure (After Mitigation)
- **P50 (Median):** {residual_stats['p50']/1e6:.2f} Million CHF{p50_marker}
- **P80:** {residual_stats['p80']/1e6:.2f} Million CHF{p80_marker}
- **P95:** {residual_stats['p95']/1e6:.2f} Million CHF{p95_marker}

### Key Findings
- Total number of identified risks: {len(df)}
- Risks with schedule impact: {df['Schedule_Impact'].sum()}
- Risks with mitigation measures: {len(df[df['Cost of Measures_Value'] > 0])}

## Recommendations
Based on the Monte Carlo simulation with **{confidence_level}** confidence level, the project should reserve at least **{residual_selected/1e6:.2f} Million CHF**
for residual risk exposure after implementing all planned mitigation measures.

The selected {confidence_level} confidence level indicates that there is a {confidence_level.replace('P', '')}% probability that
the actual risk exposure will be at or below this amount.
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
