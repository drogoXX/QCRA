# utils.py
"""
Utility functions for KVI Risk Assessment Report Generation.

Contains formatting utilities for currency, percentages, and grammatical pluralization.
"""

from typing import Optional, Union


def format_currency(
    value: Optional[float],
    decimals: int = 2,
    suffix: str = 'M CHF',
    include_sign: bool = False
) -> str:
    """
    Format currency values consistently.

    Args:
        value: The numeric value to format
        decimals: Number of decimal places
        suffix: Currency suffix (e.g., 'M CHF', 'CHF')
        include_sign: If True, prefix positive values with '+'

    Returns:
        Formatted currency string

    Examples:
        format_currency(28.61) -> '28.61 M CHF'
        format_currency(-1.57) -> '-1.57 M CHF'
        format_currency(39.50, include_sign=True) -> '+39.50 M CHF'
        format_currency(None) -> '-'
    """
    if value is None:
        return '-'

    sign = ''
    if include_sign and value > 0:
        sign = '+'

    formatted = f'{sign}{value:,.{decimals}f}'
    if suffix:
        formatted += f' {suffix}'
    return formatted


def format_currency_raw(
    value: Optional[float],
    decimals: int = 2,
    include_sign: bool = False
) -> str:
    """
    Format currency value without suffix (for table cells).

    Args:
        value: The numeric value to format
        decimals: Number of decimal places
        include_sign: If True, prefix positive values with '+'

    Returns:
        Formatted number string without currency suffix

    Examples:
        format_currency_raw(28.61) -> '28.61'
        format_currency_raw(-1.57) -> '-1.57'
        format_currency_raw(None) -> '-'
    """
    return format_currency(value, decimals=decimals, suffix='', include_sign=include_sign)


def format_percentage(
    value: Optional[float],
    decimals: int = 1,
    include_sign: bool = False
) -> str:
    """
    Format percentage values consistently.

    Args:
        value: The percentage value (e.g., 25.5 for 25.5%)
        decimals: Number of decimal places
        include_sign: If True, prefix positive values with '+'

    Returns:
        Formatted percentage string

    Examples:
        format_percentage(25.5) -> '25.5%'
        format_percentage(-10.0, include_sign=True) -> '-10.0%'
        format_percentage(15.0, include_sign=True) -> '+15.0%'
        format_percentage(None) -> '-'
    """
    if value is None:
        return '-'

    sign = ''
    if include_sign and value > 0:
        sign = '+'

    return f'{sign}{value:,.{decimals}f}%'


def format_number(
    value: Optional[float],
    decimals: int = 0,
    include_sign: bool = False
) -> str:
    """
    Format numeric values with thousand separators.

    Args:
        value: The numeric value to format
        decimals: Number of decimal places
        include_sign: If True, prefix positive values with '+'

    Returns:
        Formatted number string

    Examples:
        format_number(1500000) -> '1,500,000'
        format_number(1234.567, decimals=2) -> '1,234.57'
        format_number(None) -> '-'
    """
    if value is None:
        return '-'

    sign = ''
    if include_sign and value > 0:
        sign = '+'

    if decimals == 0:
        return f'{sign}{int(value):,}'
    return f'{sign}{value:,.{decimals}f}'


def pluralize(count: Union[int, float], singular: str, plural: str) -> str:
    """
    Return grammatically correct singular or plural form with count.

    Args:
        count: The numeric count
        singular: Singular form of the word
        plural: Plural form of the word

    Returns:
        String with count and correct noun form

    Examples:
        pluralize(1, 'risk', 'risks') -> '1 risk'
        pluralize(5, 'opportunity', 'opportunities') -> '5 opportunities'
        pluralize(0, 'threat', 'threats') -> '0 threats'
    """
    count_int = int(count)
    if count_int == 1:
        return f'{count_int} {singular}'
    return f'{count_int} {plural}'


def pluralize_noun(count: Union[int, float], singular: str, plural: str) -> str:
    """
    Return just the correct singular or plural noun form (no count).

    Args:
        count: The numeric count
        singular: Singular form of the word
        plural: Plural form of the word

    Returns:
        The correct noun form

    Examples:
        pluralize_noun(1, 'risk', 'risks') -> 'risk'
        pluralize_noun(5, 'opportunity', 'opportunities') -> 'opportunities'
    """
    count_int = int(count)
    return singular if count_int == 1 else plural


def pluralize_verb(count: Union[int, float], singular_verb: str, plural_verb: str) -> str:
    """
    Return correct verb form based on count.

    Args:
        count: The numeric count
        singular_verb: Singular verb form (e.g., 'has', 'is')
        plural_verb: Plural verb form (e.g., 'have', 'are')

    Returns:
        The correct verb form

    Examples:
        pluralize_verb(1, 'has', 'have') -> 'has'
        pluralize_verb(5, 'has', 'have') -> 'have'
        pluralize_verb(1, 'is', 'are') -> 'is'
    """
    count_int = int(count)
    return singular_verb if count_int == 1 else plural_verb


def format_risk_statement(
    count: int,
    risk_type: str = 'risk',
    initial_value: Optional[float] = None,
    residual_value: Optional[float] = None
) -> str:
    """
    Generate a grammatically correct risk statement.

    Args:
        count: Number of risks
        risk_type: Type of risk ('threat', 'opportunity', 'risk')
        initial_value: Initial exposure value (in millions)
        residual_value: Residual exposure value (in millions)

    Returns:
        Formatted risk statement

    Examples:
        format_risk_statement(5, 'threat', 28.61, 19.23)
        -> '5 threats have been identified with initial exposure of 28.61 M CHF reducing to 19.23 M CHF'

        format_risk_statement(1, 'opportunity', -1.57, -2.05)
        -> '1 opportunity has been identified with initial exposure of -1.57 M CHF reducing to -2.05 M CHF'
    """
    # Pluralization
    plural_map = {
        'risk': 'risks',
        'threat': 'threats',
        'opportunity': 'opportunities',
    }

    noun = pluralize_noun(count, risk_type, plural_map.get(risk_type, f'{risk_type}s'))
    verb = pluralize_verb(count, 'has', 'have')

    base_statement = f"{count} {noun} {verb} been identified"

    if initial_value is not None and residual_value is not None:
        base_statement += (
            f" with initial exposure of {format_currency(initial_value)} "
            f"reducing to {format_currency(residual_value)}"
        )
    elif initial_value is not None:
        base_statement += f" with exposure of {format_currency(initial_value)}"

    return base_statement


def calculate_reduction_percentage(initial: float, residual: float) -> float:
    """
    Calculate percentage reduction from initial to residual value.

    Args:
        initial: Initial value
        residual: Residual value

    Returns:
        Percentage reduction (positive means reduction)

    Examples:
        calculate_reduction_percentage(100, 70) -> 30.0
        calculate_reduction_percentage(50, 75) -> -50.0 (increase)
    """
    if initial == 0:
        return 0.0
    return ((initial - residual) / abs(initial)) * 100


def get_risk_profile(reduction_pct: float) -> dict:
    """
    Get risk profile based on reduction percentage.

    Args:
        reduction_pct: Risk reduction percentage

    Returns:
        Dictionary with profile info (label, color, emoji)
    """
    from report_config import RISK_PROFILES

    # Invert for assessment: higher reduction = lower risk
    if reduction_pct >= 50:
        return RISK_PROFILES['low']
    elif reduction_pct >= 25:
        return RISK_PROFILES['moderate']
    elif reduction_pct >= 10:
        return RISK_PROFILES['high']
    else:
        return RISK_PROFILES['very_high']


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., '#1F4E79' or '1F4E79')

    Returns:
        Tuple of (R, G, B) values

    Examples:
        hex_to_rgb('#1F4E79') -> (31, 78, 121)
        hex_to_rgb('FFFFFF') -> (255, 255, 255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color string.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        Hex color string without '#' prefix

    Examples:
        rgb_to_hex(31, 78, 121) -> '1F4E79'
    """
    return f'{r:02X}{g:02X}{b:02X}'


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division by zero

    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator
