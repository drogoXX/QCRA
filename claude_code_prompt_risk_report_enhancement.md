# Claude Code Prompt: Professional Risk Assessment Report Enhancement

## Context

I have a Streamlit/Python application that generates risk assessment reports as Word documents (.docx) using the `python-docx` library. The reports include Monte Carlo simulation results, tables, Matplotlib/Plotly charts (saved as images), and various risk analysis sections.

The application is used for the PROTOS CC Carbon Capture project at Kanadevia INOVA (KVI). Reports need to be board-ready and professional.

**Current report sections include:**
- Executive Summary with threat/opportunity analysis
- Risk contingency allocation
- Risk portfolio overview with heatmaps
- Executive risk narrative
- 3D risk landscape visualization
- Monte Carlo simulation results (CDF, histograms)
- Sensitivity/Pareto analysis
- Mitigation cost-benefit analysis
- Confidence level comparison
- Time-phased contingency profile
- Appendix A: Risk Register
- Appendix B: Methodology

**Data available includes:**
- Risk ID, description, initial/residual impact and likelihood
- Mitigation costs
- Optional: Crystallization phase (Engineering, Procurement, Fabrication, Construction, Commissioning, Warranty)
- Monte Carlo simulation results at various percentiles

---

## Enhancement Requirements

Please implement the following improvements to make the document generation professional and board-ready.

---

### 1. COVER PAGE ENHANCEMENTS

Create a professional cover page with:

```
┌─────────────────────────────────────────────────────────────────┐
│  [KVI LOGO - top left]                                          │
│                                                                 │
│                                                                 │
│                    RISK ASSESSMENT REPORT                       │
│                                                                 │
│              Monte Carlo Simulation & Probabilistic             │
│                       Risk Analysis                             │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Project:        [PROJECT_NAME]                                 │
│  Contract Ref:   [CONTRACT_REF]                                 │
│  Document No:    [DOC_NUMBER] Rev [REVISION]                    │
│  Confidence:     P80                                            │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Classification: [CONFIDENTIAL / INTERNAL]                      │
│  Generated:      [DATE TIME]                                    │
│                                                                 │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ DOCUMENT CONTROL                                          │  │
│  ├──────────┬────────┬─────────────────┬────────────────────┤  │
│  │ Rev      │ Date   │ Author          │ Description        │  │
│  ├──────────┼────────┼─────────────────┼────────────────────┤  │
│  │ A        │ [DATE] │ [AUTHOR]        │ Initial Issue      │  │
│  └──────────┴────────┴─────────────────┴────────────────────┘  │
│                                                                 │
│  Prepared by:  ________________  Date: ________                 │
│  Reviewed by:  ________________  Date: ________                 │
│  Approved by:  ________________  Date: ________                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation details:**
- Logo: Download from `https://www.kanadevia-inova.com/wp-content/uploads/2024/09/kanadevia.svg` or use a provided PNG. If SVG is problematic, convert to PNG first. Place top-left, approximately 4cm width.
- Use configurable parameters for project name, contract ref, document number, revision, classification, and author
- Document control table should have subtle borders and alternating row colors
- Add page break after cover page

---

### 2. KVI COLOR PALETTE (Use Throughout)

Define a configuration dictionary with KVI brand colors:

```python
KVI_COLORS = {
    'primary': '#0077B6',      # Corporate blue - headers, accents
    'secondary': '#90E0EF',    # Light blue - backgrounds, highlights
    'accent': '#00B4D8',       # Teal - charts, emphasis
    'dark': '#023E8A',         # Dark blue - titles, key text
    'light': '#CAF0F8',        # Pale blue - subtle backgrounds
    'neutral': '#495057',      # Charcoal - body text
    'success': '#40916C',      # Green - positive indicators, opportunities
    'warning': '#F4A261',      # Amber - caution, medium risk
    'danger': '#E63946',       # Red - critical, threats, high risk
    'surface': '#FFFFFF',      # White - default background
    'table_header': '#1F4E79', # Dark blue for table headers
    'table_alt_row': '#F2F2F2',# Light gray alternating rows
    'table_border': '#D9D9D9', # Gray borders
    'total_row_bg': '#D6EAF8', # Light blue for totals
}
```

---

### 3. TABLE OF CONTENTS

After the cover page, add an auto-updating Table of Contents:

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_table_of_contents(document):
    """Add a Table of Contents that updates when document is opened in Word."""
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    
    # Create TOC field
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'  # 3 levels, hyperlinks, no page numbers until update
    
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)
    
    # Add instruction text
    document.add_paragraph("Right-click and select 'Update Field' to refresh Table of Contents", 
                          style='Caption')
    document.add_page_break()
```

---

### 4. HEADERS AND FOOTERS

Add professional running headers and footers:

```python
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_header_footer(document, project_name, doc_ref, report_date):
    """Add running headers and footers to all sections."""
    for section in document.sections:
        # Different first page (no header/footer on cover)
        section.different_first_page_header_footer = True
        
        # Header
        header = section.header
        header_para = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        header_para.clear()
        
        # Left-aligned project name
        run_left = header_para.add_run(project_name)
        run_left.font.size = Pt(9)
        run_left.font.color.rgb = RGBColor.from_string('495057')
        
        # Tab to right
        header_para.add_run('\t\t')
        
        # Right-aligned document title
        run_right = header_para.add_run('Risk Assessment Report')
        run_right.font.size = Pt(9)
        run_right.font.color.rgb = RGBColor.from_string('495057')
        run_right.font.italic = True
        
        # Footer
        footer = section.footer
        footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        footer_para.clear()
        
        # Left: Doc reference
        run_left = footer_para.add_run(doc_ref)
        run_left.font.size = Pt(8)
        run_left.font.color.rgb = RGBColor.from_string('495057')
        
        # Center: Page X of Y (using fields)
        footer_para.add_run('\t')
        add_page_number(footer_para)
        
        # Right: Date
        footer_para.add_run('\t')
        run_date = footer_para.add_run(report_date)
        run_date.font.size = Pt(8)
        run_date.font.color.rgb = RGBColor.from_string('495057')

def add_page_number(paragraph):
    """Add 'Page X of Y' field to paragraph."""
    run = paragraph.add_run()
    run.font.size = Pt(8)
    
    # PAGE field
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText1 = OxmlElement('w:instrText')
    instrText1.text = "PAGE"
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    
    run._r.append(fldChar1)
    run._r.append(instrText1)
    run._r.append(fldChar2)
    
    paragraph.add_run(' of ')
    
    # NUMPAGES field
    run2 = paragraph.add_run()
    run2.font.size = Pt(8)
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'begin')
    instrText2 = OxmlElement('w:instrText')
    instrText2.text = "NUMPAGES"
    fldChar4 = OxmlElement('w:fldChar')
    fldChar4.set(qn('w:fldCharType'), 'end')
    
    run2._r.append(fldChar3)
    run2._r.append(instrText2)
    run2._r.append(fldChar4)
```

---

### 5. HEADING STYLES

Configure consistent heading styles with KVI colors:

```python
from docx.shared import Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE

def configure_heading_styles(document):
    """Configure heading styles with KVI branding."""
    styles = document.styles
    
    # Heading 1: Main sections
    h1 = styles['Heading 1']
    h1.font.name = 'Calibri'
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor.from_string('1F4E79')  # Dark blue
    h1.paragraph_format.space_before = Pt(24)
    h1.paragraph_format.space_after = Pt(12)
    h1.paragraph_format.page_break_before = True  # New page for major sections
    
    # Heading 2: Subsections
    h2 = styles['Heading 2']
    h2.font.name = 'Calibri'
    h2.font.size = Pt(14)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor.from_string('2E75B6')  # Medium blue
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)
    
    # Heading 3: Sub-subsections
    h3 = styles['Heading 3']
    h3.font.name = 'Calibri'
    h3.font.size = Pt(12)
    h3.font.bold = True
    h3.font.color.rgb = RGBColor.from_string('404040')  # Dark gray
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)
    
    # Body text style
    normal = styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string('495057')
    normal.paragraph_format.line_spacing = 1.15
```

---

### 6. PROFESSIONAL TABLE FORMATTING

Create a reusable function for consistent table styling:

```python
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

def apply_professional_table_style(table, has_total_row=False, highlight_conditions=None):
    """
    Apply professional KVI table styling.
    
    Args:
        table: python-docx Table object
        has_total_row: If True, style the last row as a total/summary row
        highlight_conditions: Dict with column index and condition functions for conditional formatting
    """
    # Table-level formatting
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            # Set cell margins
            set_cell_margins(cell, top=50, bottom=50, left=100, right=100)
            
            # Header row (first row)
            if i == 0:
                set_cell_shading(cell, '1F4E79')  # Dark blue
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)  # White
                        run.font.size = Pt(10)
            
            # Total row (last row if has_total_row)
            elif has_total_row and i == len(table.rows) - 1:
                set_cell_shading(cell, 'D6EAF8')  # Light blue
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
            
            # Alternating rows
            elif i % 2 == 0:
                set_cell_shading(cell, 'F2F2F2')  # Light gray
            else:
                set_cell_shading(cell, 'FFFFFF')  # White
            
            # Apply conditional formatting if specified
            if highlight_conditions and j in highlight_conditions:
                condition_func, bg_color = highlight_conditions[j]
                cell_text = cell.text.strip()
                if condition_func(cell_text):
                    set_cell_shading(cell, bg_color)
            
            # Add thin borders
            set_cell_border(cell)

def set_cell_shading(cell, color_hex):
    """Set cell background color."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_cell_margins(cell, top=0, bottom=0, left=0, right=0):
    """Set cell margins in twips (1/20 of a point)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, margin_value in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(margin_value))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_border(cell, color='D9D9D9', size='4'):
    """Add thin borders to cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for border_name in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), size)
        border.set(qn('w:color'), color)
        tcBorders.append(border)
    tcPr.append(tcBorders)
```

---

### 7. NUMBER FORMATTING UTILITIES

Create helper functions for consistent number formatting:

```python
def format_currency(value, decimals=2, suffix='M CHF', include_sign=False):
    """
    Format currency values consistently.
    
    Examples:
        format_currency(28.61) -> '28.61 M CHF'
        format_currency(-1.57) -> '-1.57 M CHF'
        format_currency(39.50, include_sign=True) -> '+39.50 M CHF'
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

def format_percentage(value, decimals=1, include_sign=False):
    """Format percentage values consistently."""
    if value is None:
        return '-'
    
    sign = ''
    if include_sign and value > 0:
        sign = '+'
    
    return f'{sign}{value:,.{decimals}f}%'

def pluralize(count, singular, plural):
    """
    Return grammatically correct singular or plural form.
    
    Examples:
        pluralize(1, 'risk', 'risks') -> '1 risk'
        pluralize(5, 'opportunity', 'opportunities') -> '5 opportunities'
        pluralize(1, 'has', 'have') -> 'has'
    """
    if count == 1:
        return singular if singular in ['has', 'have', 'is', 'are'] else f'{count} {singular}'
    return plural if plural in ['has', 'have', 'is', 'are'] else f'{count} {plural}'

def pluralize_verb(count, singular_verb, plural_verb):
    """Return correct verb form based on count."""
    return singular_verb if count == 1 else plural_verb
```

**Apply these in dynamic text generation:**
```python
# Before (incorrect):
text = f"{opp_count} opportunities have been identified"

# After (correct):
text = f"{pluralize(opp_count, 'opportunity has', 'opportunities have')} been identified"

# Or more explicitly:
verb = pluralize_verb(opp_count, 'has', 'have')
noun = pluralize(opp_count, 'opportunity', 'opportunities')  
text = f"{noun} {verb} been identified"
```

---

### 8. KPI SUMMARY BOX (Executive Summary)

Add an "At a Glance" summary box at the start of Executive Summary:

```python
def add_kpi_summary_box(document, metrics):
    """
    Add a KPI summary callout box.
    
    Args:
        metrics: dict with keys:
            - total_risks, threats, opportunities
            - initial_p80, residual_p80
            - risk_reduction_pct
            - risk_profile ('Low', 'Moderate', 'High', 'Very High')
    """
    # Create a 1-row, 1-column table as a container
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    cell = table.cell(0, 0)
    
    # Style the cell as a callout box
    set_cell_shading(cell, 'E8F4FD')  # Light blue background
    set_cell_border(cell, color='0077B6', size='12')  # Blue border
    set_cell_margins(cell, top=150, bottom=150, left=200, right=200)
    
    # Title
    p1 = cell.paragraphs[0]
    p1.clear()
    title_run = p1.add_run('📊 KEY METRICS AT A GLANCE')
    title_run.font.bold = True
    title_run.font.size = Pt(12)
    title_run.font.color.rgb = RGBColor.from_string('023E8A')
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Metrics line 1
    p2 = cell.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = p2.add_run(
        f"Total Risks: {metrics['total_risks']}  │  "
        f"Threats: {metrics['threats']}  │  "
        f"Opportunities: {metrics['opportunities']}"
    )
    run2.font.size = Pt(10)
    
    # Metrics line 2 - P80 exposure
    p3 = cell.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run3 = p3.add_run(
        f"P80 Exposure: {format_currency(metrics['initial_p80'])} → {format_currency(metrics['residual_p80'])}"
    )
    run3.font.size = Pt(11)
    run3.font.bold = True
    
    # Metrics line 3 - Risk reduction with color
    p4 = cell.add_paragraph()
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run4a = p4.add_run(f"Risk Reduction: ")
    run4a.font.size = Pt(11)
    
    run4b = p4.add_run(f"{format_percentage(metrics['risk_reduction_pct'])}")
    run4b.font.size = Pt(11)
    run4b.font.bold = True
    run4b.font.color.rgb = RGBColor.from_string('40916C')  # Green
    
    # Risk profile indicator
    p5 = cell.add_paragraph()
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    profile = metrics.get('risk_profile', 'Moderate')
    profile_colors = {
        'Low': ('40916C', '🟢'),
        'Moderate': ('F4A261', '🟡'),
        'High': ('E63946', '🟠'),
        'Very High': ('E63946', '🔴')
    }
    color, emoji = profile_colors.get(profile, ('495057', '⚪'))
    
    run5 = p5.add_run(f"Risk Profile: {emoji} {profile}")
    run5.font.size = Pt(10)
    run5.font.bold = True
    run5.font.color.rgb = RGBColor.from_string(color)
    
    document.add_paragraph()  # Spacer
```

---

### 9. FIGURE CAPTIONS AND NUMBERING

Add automatic figure numbering for charts:

```python
class FigureCounter:
    """Track figure numbers for captioning."""
    def __init__(self):
        self.count = 0
    
    def next(self):
        self.count += 1
        return self.count

# Global counter
figure_counter = FigureCounter()

def add_captioned_image(document, image_path, caption_text, width=Inches(6)):
    """
    Add an image with a numbered caption.
    
    Example output: "Figure 3: Risk Heatmap Comparison Before vs After Mitigation"
    """
    # Add image centered
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(image_path, width=width)
    
    # Add caption
    caption_para = document.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_para.style = 'Caption'
    
    fig_num = figure_counter.next()
    caption_run = caption_para.add_run(f'Figure {fig_num}: {caption_text}')
    caption_run.font.size = Pt(9)
    caption_run.font.italic = True
    caption_run.font.color.rgb = RGBColor.from_string('495057')
    
    return fig_num
```

---

### 10. RISK REGISTER ENHANCEMENTS

Add conditional formatting to the Risk Register table:

```python
def format_risk_register_table(table, risk_data):
    """
    Apply professional formatting to risk register with conditional highlighting.
    
    Args:
        table: The risk register table
        risk_data: List of dicts with risk data including likelihood and impact values
    """
    # Define thresholds
    HIGH_LIKELIHOOD = 0.50  # 50%
    HIGH_IMPACT = 5_000_000  # 5M CHF
    VERY_HIGH_LIKELIHOOD = 0.70
    
    # Apply base styling first
    apply_professional_table_style(table, has_total_row=False)
    
    # Additional conditional formatting
    for i, row in enumerate(table.rows):
        if i == 0:  # Skip header
            continue
        
        try:
            # Get data from the row (adjust indices based on your table structure)
            risk_id = risk_data[i-1].get('risk_id')
            initial_likelihood = risk_data[i-1].get('initial_likelihood', 0)
            initial_impact = risk_data[i-1].get('initial_impact', 0)
            mitigation_cost = risk_data[i-1].get('mitigation_cost', 0)
            
            # High-priority risk: highlight entire row
            if initial_likelihood >= HIGH_LIKELIHOOD and initial_impact >= HIGH_IMPACT:
                for cell in row.cells:
                    set_cell_shading(cell, 'FADBD8')  # Light red
            
            # Very high likelihood: orange text
            if initial_likelihood >= VERY_HIGH_LIKELIHOOD:
                likelihood_cell = row.cells[3]  # Adjust index as needed
                for paragraph in likelihood_cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = RGBColor.from_string('E67E22')  # Orange
                        run.font.bold = True
            
            # Zero mitigation cost: gray italic
            if mitigation_cost == 0:
                cost_cell = row.cells[-1]  # Last column
                for paragraph in cost_cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = RGBColor.from_string('808080')  # Gray
                        run.font.italic = True
        except (IndexError, KeyError):
            continue

def add_expected_value_column(table, risk_data):
    """Add Expected Value (EV) column to risk register."""
    # Add column header
    header_row = table.rows[0]
    # You'll need to add the column during table creation
    # EV = Impact × Likelihood
    pass
```

---

### 11. PHASE-BASED FILTERING (Optional)

If crystallization phase data is available, add phase breakdowns:

```python
VALID_PHASES = ['Engineering', 'Procurement', 'Fabrication', 'Construction', 'Commissioning', 'Warranty']

def add_phase_summary_table(document, risk_data):
    """
    Add a summary table showing risks by crystallization phase.
    Only include if phase data is available.
    """
    # Check if phase data exists
    phases_with_data = [r.get('phase') for r in risk_data if r.get('phase') in VALID_PHASES]
    
    if not phases_with_data:
        return  # Skip if no phase data
    
    document.add_heading('Risk Distribution by Project Phase', level=2)
    
    # Count risks per phase
    phase_counts = {phase: 0 for phase in VALID_PHASES}
    phase_exposure = {phase: 0.0 for phase in VALID_PHASES}
    
    for risk in risk_data:
        phase = risk.get('phase')
        if phase in VALID_PHASES:
            phase_counts[phase] += 1
            ev = risk.get('initial_impact', 0) * risk.get('initial_likelihood', 0)
            phase_exposure[phase] += ev
    
    # Create table
    table = document.add_table(rows=len(VALID_PHASES) + 1, cols=3)
    
    # Header
    headers = ['Phase', 'Risk Count', 'Expected Exposure (M CHF)']
    for j, header in enumerate(headers):
        table.cell(0, j).text = header
    
    # Data rows
    for i, phase in enumerate(VALID_PHASES, start=1):
        table.cell(i, 0).text = phase
        table.cell(i, 1).text = str(phase_counts[phase])
        table.cell(i, 2).text = format_currency(phase_exposure[phase] / 1_000_000, decimals=2, suffix='')
    
    apply_professional_table_style(table)
```

---

### 12. ADDITIONAL APPENDICES

Add the following new appendices:

#### Appendix C: Glossary of Terms

```python
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

def add_glossary_appendix(document):
    """Add Glossary of Terms appendix."""
    document.add_heading('Appendix C: Glossary of Terms', level=1)
    
    table = document.add_table(rows=len(GLOSSARY_TERMS) + 1, cols=2)
    
    # Header
    table.cell(0, 0).text = 'Term'
    table.cell(0, 1).text = 'Definition'
    
    # Terms
    for i, (term, definition) in enumerate(GLOSSARY_TERMS, start=1):
        table.cell(i, 0).text = term
        table.cell(i, 1).text = definition
    
    apply_professional_table_style(table)
    
    # Set column widths
    table.columns[0].width = Inches(2)
    table.columns[1].width = Inches(4.5)
```

#### Appendix D: Assumptions and Limitations

```python
def add_assumptions_appendix(document, simulation_params):
    """
    Add Assumptions and Limitations appendix.
    
    Args:
        simulation_params: dict with 'iterations', 'confidence_level', 'data_date', etc.
    """
    document.add_heading('Appendix D: Assumptions and Limitations', level=1)
    
    # Simulation Parameters
    document.add_heading('Simulation Parameters', level=2)
    params_text = f"""
    • Monte Carlo Iterations: {simulation_params.get('iterations', 15000):,}
    • Selected Confidence Level: P{simulation_params.get('confidence_level', 80)}
    • Risk Data Date: {simulation_params.get('data_date', 'As per Risk Register')}
    • Correlation Model: Independent risks (no correlation modelled)
    """
    document.add_paragraph(params_text.strip())
    
    # Key Assumptions
    document.add_heading('Key Assumptions', level=2)
    assumptions = [
        "Risk impacts and likelihoods are based on expert judgement and available project data.",
        "Risks are modelled as independent events (correlations between risks are not considered).",
        "Impact values represent the most likely cost if the risk materialises.",
        "Mitigation measures are assumed to be implemented as planned.",
        "Market conditions and external factors remain within reasonable bounds.",
    ]
    for assumption in assumptions:
        p = document.add_paragraph(assumption, style='List Bullet')
    
    # Limitations
    document.add_heading('Limitations', level=2)
    limitations = [
        "This analysis is based on information available at the time of assessment.",
        "Actual outcomes may differ from probabilistic estimates due to unforeseen events.",
        "The model does not account for potential risk correlations or cascade effects.",
        "Currency fluctuations and inflation beyond baseline assumptions are not modelled.",
    ]
    for limitation in limitations:
        p = document.add_paragraph(limitation, style='List Bullet')
    
    # Disclaimer
    document.add_heading('Disclaimer', level=2)
    disclaimer = """This risk assessment report is provided for project planning and decision-making purposes. 
    The probabilistic estimates contained herein represent a range of possible outcomes based on the 
    input data and assumptions stated. They should not be interpreted as guarantees or precise predictions. 
    Users should exercise professional judgement when applying these results to project decisions."""
    document.add_paragraph(disclaimer.strip())
```

#### Appendix E: Document Approval

```python
def add_approval_appendix(document, approval_roles=None):
    """Add Document Approval sign-off page."""
    document.add_heading('Appendix E: Document Approval', level=1)
    
    document.add_paragraph(
        'This document has been reviewed and approved by the following personnel:',
        style='Normal'
    )
    document.add_paragraph()  # Spacer
    
    if approval_roles is None:
        approval_roles = [
            ('Prepared by', 'Risk Analyst', ''),
            ('Reviewed by', 'Project Controls Manager', ''),
            ('Approved by', 'Project Director', ''),
        ]
    
    table = document.add_table(rows=len(approval_roles) + 1, cols=4)
    
    # Header
    headers = ['Role', 'Name', 'Signature', 'Date']
    for j, header in enumerate(headers):
        table.cell(0, j).text = header
    
    # Data rows
    for i, (role, title, name) in enumerate(approval_roles, start=1):
        table.cell(i, 0).text = f'{role}\n({title})'
        table.cell(i, 1).text = name if name else ''
        table.cell(i, 2).text = ''  # Signature space
        table.cell(i, 3).text = ''  # Date space
    
    apply_professional_table_style(table)
    
    # Set row heights for signature space
    for row in table.rows[1:]:
        row.height = Cm(1.5)
```

---

### 13. DOCUMENT PROPERTIES

Set document metadata:

```python
from docx import Document
from docx.opc.coreprops import CoreProperties

def set_document_properties(document, project_name, author='Kanadevia INOVA'):
    """Set document metadata properties."""
    core_props = document.core_properties
    core_props.title = f'Risk Assessment Report - {project_name}'
    core_props.author = author
    core_props.subject = 'Monte Carlo Risk Analysis'
    core_props.keywords = f'Risk Assessment, Monte Carlo, P80, {project_name}, Kanadevia INOVA'
    core_props.category = 'Project Risk Management'
    core_props.comments = 'Generated by KVI Risk Assessment Application'
```

---

### 14. MAIN REPORT GENERATION STRUCTURE

Here's how to structure the main report generation function:

```python
def generate_risk_report(
    risk_data: list,
    simulation_results: dict,
    chart_paths: dict,
    config: dict
) -> Document:
    """
    Generate professional risk assessment report.
    
    Args:
        risk_data: List of risk dictionaries
        simulation_results: Monte Carlo simulation outputs
        chart_paths: Dict mapping chart names to file paths
        config: Report configuration (project_name, doc_ref, author, etc.)
    
    Returns:
        python-docx Document object
    """
    document = Document()
    
    # 1. Configure styles
    configure_heading_styles(document)
    
    # 2. Set document properties
    set_document_properties(document, config['project_name'], config.get('author'))
    
    # 3. Add cover page
    add_cover_page(document, config)
    
    # 4. Add Table of Contents
    add_table_of_contents(document)
    
    # 5. Executive Summary with KPI box
    document.add_heading('Executive Summary', level=1)
    add_kpi_summary_box(document, calculate_kpi_metrics(risk_data, simulation_results))
    # ... rest of executive summary content
    
    # 6. Add remaining sections with proper formatting
    # ... (Risk Contingency Allocation, Portfolio Overview, etc.)
    
    # 7. Add charts with captions
    for chart_name, chart_path in chart_paths.items():
        add_captioned_image(document, chart_path, CHART_CAPTIONS[chart_name])
    
    # 8. Add appendices
    add_risk_register_appendix(document, risk_data)
    add_methodology_appendix(document)
    add_glossary_appendix(document)
    add_assumptions_appendix(document, simulation_results.get('params', {}))
    add_approval_appendix(document)
    
    # 9. Add headers/footers (after all content)
    add_header_footer(document, config['project_name'], config['doc_ref'], config['report_date'])
    
    return document
```

---

### 15. CONFIGURATION FILE

Create a configuration module for easy customization:

```python
# config.py

REPORT_CONFIG = {
    # KVI Branding
    'logo_path': 'assets/kanadevia_logo.png',  # Download from KVI website
    'company_name': 'Kanadevia INOVA',
    
    # Colors (KVI palette)
    'colors': {
        'primary': '#0077B6',
        'secondary': '#90E0EF',
        'accent': '#00B4D8',
        'dark': '#023E8A',
        'light': '#CAF0F8',
        'neutral': '#495057',
        'success': '#40916C',
        'warning': '#F4A261',
        'danger': '#E63946',
        'table_header': '#1F4E79',
        'table_alt_row': '#F2F2F2',
        'table_border': '#D9D9D9',
        'total_row_bg': '#D6EAF8',
    },
    
    # Fonts
    'fonts': {
        'heading': 'Calibri',
        'body': 'Calibri',
        'code': 'Consolas',
    },
    
    # Document settings
    'classification': 'CONFIDENTIAL',  # or 'INTERNAL', 'PUBLIC'
    'page_size': 'A4',
    
    # Chart settings
    'chart_width_inches': 6.0,
    'chart_dpi': 150,
    
    # Thresholds for conditional formatting
    'thresholds': {
        'high_likelihood': 0.50,
        'high_impact': 5_000_000,
        'very_high_likelihood': 0.70,
    },
    
    # Valid crystallization phases
    'phases': ['Engineering', 'Procurement', 'Fabrication', 'Construction', 'Commissioning', 'Warranty'],
}
```

---

## Testing Checklist

After implementing, verify:

1. ☐ Document opens correctly in Microsoft Word
2. ☐ Table of Contents updates when right-clicking and selecting "Update Field"
3. ☐ Cover page has no header/footer, subsequent pages do
4. ☐ Page numbers show "Page X of Y" format
5. ☐ All tables have consistent professional styling
6. ☐ Monetary values have thousand separators
7. ☐ Grammar is correct (singular/plural agreement)
8. ☐ All charts render at proper resolution
9. ☐ Figure captions are numbered sequentially
10. ☐ Conditional formatting highlights high-risk items in Risk Register
11. ☐ All appendices are present and properly formatted
12. ☐ Document properties are set correctly (check File > Properties)
13. ☐ KVI logo appears on cover page
14. ☐ Colors match KVI brand palette

---

## Files to Create/Modify

1. `report_formatter.py` - New module with all formatting functions
2. `report_config.py` - Configuration constants
3. `report_generator.py` - Main report generation logic (modify existing)
4. `utils.py` - Add pluralize, format_currency, format_percentage functions
5. `assets/kanadevia_logo.png` - Download KVI logo

---

## Logo Download Instructions

Download the KVI logo from:
- SVG (preferred): `https://www.kanadevia-inova.com/wp-content/uploads/2024/09/kanadevia.svg`
- Convert to PNG using: `cairosvg kanadevia.svg -o kanadevia_logo.png -d 300`

Or use the visual from press release:
- `https://www.kanadevia-inova.com/wp-content/uploads/2024/09/KVI_Pressemitteilung_Motiv3-scaled.jpg`
