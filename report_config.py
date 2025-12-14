# report_config.py
"""
Configuration module for KVI Risk Assessment Report Generation.

Contains KVI brand colors, fonts, document settings, and thresholds
for consistent professional report styling.
"""

# KVI Brand Color Palette
KVI_COLORS = {
    'primary': '#0077B6',       # Corporate blue - headers, accents
    'secondary': '#90E0EF',     # Light blue - backgrounds, highlights
    'accent': '#00B4D8',        # Teal - charts, emphasis
    'dark': '#023E8A',          # Dark blue - titles, key text
    'light': '#CAF0F8',         # Pale blue - subtle backgrounds
    'neutral': '#495057',       # Charcoal - body text
    'success': '#40916C',       # Green - positive indicators, opportunities
    'warning': '#F4A261',       # Amber - caution, medium risk
    'danger': '#E63946',        # Red - critical, threats, high risk
    'surface': '#FFFFFF',       # White - default background
    'table_header': '#1F4E79',  # Dark blue for table headers
    'table_alt_row': '#F2F2F2', # Light gray alternating rows
    'table_border': '#D9D9D9',  # Gray borders
    'total_row_bg': '#D6EAF8',  # Light blue for totals
    'kpi_box_bg': '#E8F4FD',    # Light blue for KPI summary box
    'kpi_box_border': '#0077B6', # Blue border for KPI box
}

# Font Configuration
FONTS = {
    'heading': 'Calibri',
    'body': 'Calibri',
    'code': 'Consolas',
}

# Font Sizes (in points)
FONT_SIZES = {
    'title': 44,
    'subtitle': 18,
    'heading1': 16,
    'heading2': 14,
    'heading3': 12,
    'body': 11,
    'table_header': 10,
    'table_data': 10,
    'caption': 9,
    'footer': 8,
    'header': 9,
    'metadata': 12,
}

# Document Settings
DOCUMENT_SETTINGS = {
    'page_size': 'A4',
    'chart_width_inches': 6.0,
    'chart_dpi': 150,
    'logo_width_cm': 4.0,
    'default_classification': 'CONFIDENTIAL',
    'company_name': 'Kanadevia INOVA',
    'default_currency_suffix': 'M CHF',
}

# Valid crystallization phases
VALID_PHASES = [
    'Engineering',
    'Procurement',
    'Fabrication',
    'Construction',
    'Commissioning',
    'Warranty'
]

# Phase abbreviation mapping
PHASE_ABBREVIATIONS = {
    'ENG': 'Engineering',
    'PROC': 'Procurement',
    'FAB': 'Fabrication',
    'CON': 'Construction',
    'COM': 'Commissioning',
    'WAR': 'Warranty',
    'Engineering': 'Engineering',
    'Procurement': 'Procurement',
    'Fabrication': 'Fabrication',
    'Construction': 'Construction',
    'Commissioning': 'Commissioning',
    'Warranty': 'Warranty',
}

# Thresholds for conditional formatting
THRESHOLDS = {
    'high_likelihood': 0.50,        # 50%
    'very_high_likelihood': 0.70,   # 70%
    'high_impact': 5_000_000,       # 5M CHF
    'critical_impact': 10_000_000,  # 10M CHF
}

# Risk Profile Classifications
RISK_PROFILES = {
    'low': {
        'label': 'Low',
        'color': '#40916C',
        'emoji': '🟢',
        'threshold_pct': 25,
    },
    'moderate': {
        'label': 'Moderate',
        'color': '#F4A261',
        'emoji': '🟡',
        'threshold_pct': 50,
    },
    'high': {
        'label': 'High',
        'color': '#E67E22',
        'emoji': '🟠',
        'threshold_pct': 75,
    },
    'very_high': {
        'label': 'Very High',
        'color': '#E63946',
        'emoji': '🔴',
        'threshold_pct': 100,
    },
}

# Glossary of Terms
GLOSSARY_TERMS = [
    ('P50', 'The 50th percentile (median) - 50% probability that actual exposure will be at or below this value.'),
    ('P80', 'The 80th percentile - 80% probability that actual exposure will be at or below this value. Industry standard for contingency planning.'),
    ('P95', 'The 95th percentile - 95% probability that actual exposure will be at or below this value. Conservative estimate.'),
    ('Monte Carlo Simulation', 'A computational technique using repeated random sampling to model probability distributions of outcomes.'),
    ('Expected Value (EV)', 'The probability-weighted average outcome, calculated as Impact × Likelihood for each risk.'),
    ('Variance Contribution', 'The proportion of total portfolio uncertainty attributable to a specific risk.'),
    ('Pareto Analysis', 'The 80/20 principle identifying the vital few risks that drive the majority of portfolio uncertainty.'),
    ('Residual Risk', 'The remaining risk exposure after mitigation measures have been applied.'),
    ('Mitigation Cost', 'The investment required to implement risk reduction measures.'),
    ('ROI (Return on Investment)', 'Risk reduction achieved divided by mitigation cost invested.'),
    ('Threat', 'A risk with negative impact (potential cost) represented as positive Expected Value.'),
    ('Opportunity', 'A risk with positive impact (potential benefit) represented as negative Expected Value.'),
    ('Contingency', 'Financial reserve allocated to cover potential risk materialisation.'),
    ('Crystallization Phase', 'The project phase during which a risk is most likely to materialise (Engineering, Procurement, Fabrication, Construction, Commissioning, Warranty).'),
]

# Key Assumptions for reports
DEFAULT_ASSUMPTIONS = [
    "Risk impacts and likelihoods are based on expert judgement and available project data.",
    "Risks are modelled as independent events (correlations between risks are not considered).",
    "Impact values represent the most likely cost if the risk materialises.",
    "Mitigation measures are assumed to be implemented as planned.",
    "Market conditions and external factors remain within reasonable bounds.",
]

# Limitations for reports
DEFAULT_LIMITATIONS = [
    "This analysis is based on information available at the time of assessment.",
    "Actual outcomes may differ from probabilistic estimates due to unforeseen events.",
    "The model does not account for potential risk correlations or cascade effects.",
    "Currency fluctuations and inflation beyond baseline assumptions are not modelled.",
]

# Default Disclaimer
DEFAULT_DISCLAIMER = """This risk assessment report is provided for project planning and decision-making purposes.
The probabilistic estimates contained herein represent a range of possible outcomes based on the
input data and assumptions stated. They should not be interpreted as guarantees or precise predictions.
Users should exercise professional judgement when applying these results to project decisions."""

# Default Approval Roles
DEFAULT_APPROVAL_ROLES = [
    ('Prepared by', 'Risk Analyst', ''),
    ('Reviewed by', 'Project Controls Manager', ''),
    ('Approved by', 'Project Director', ''),
]

# Chart Captions (mapping chart names to caption text)
CHART_CAPTIONS = {
    'risk_heatmap': 'Risk Heatmap Comparison: Initial vs. Residual Risk Exposure',
    'cdf_comparison': 'Cumulative Distribution Function (CDF) Comparison',
    'histogram_comparison': 'Monte Carlo Distribution Comparison: Initial vs. Residual',
    'pareto_chart': 'Pareto Analysis: Top Risk Contributors by Variance',
    'roi_chart': 'Mitigation ROI Analysis: Top Cost-Effective Opportunities',
    'confidence_comparison': 'Confidence Level Comparison: P50, P80, P95',
    'phase_allocation': 'Time-Phased Contingency Allocation by Project Phase',
    'phase_waterfall': 'Waterfall Analysis: Phase-by-Phase Risk Exposure',
    '3d_comparison': '3D Risk Landscape: Initial vs. Residual Risk Surface',
    'risk_matrix': 'Risk Matrix: Impact vs. Likelihood Assessment',
    'box_plot': 'Box Plot: Distribution Comparison',
}

# Report section order
REPORT_SECTIONS = [
    'cover_page',
    'table_of_contents',
    'executive_summary',
    'contingency_allocation',
    'risk_portfolio',
    'narrative',
    '3d_visualization',
    'monte_carlo_results',
    'sensitivity_analysis',
    'mitigation_analysis',
    'confidence_comparison',
    'time_phased_profile',
    'appendix_risk_register',
    'appendix_methodology',
    'appendix_glossary',
    'appendix_assumptions',
    'appendix_approval',
]
