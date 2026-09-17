"""
Corporate Financial Anomaly Detection Dashboard
Styled with High-Contrast Light & Dark Theme Variations
Featured with an Expanded 24-Color High-Contrast Palette for Multidimensional Feature Clustering
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

anomaly_df = pd.read_csv(os.path.join(DATA_DIR, "anomaly_scores.csv"))
panel_df   = pd.read_csv(os.path.join(DATA_DIR, "raw_panel_data.csv"))

# Normalise column names
anomaly_df.columns = anomaly_df.columns.str.strip()
panel_df.columns   = panel_df.columns.str.strip()

# Ensure numeric types
for col in ["Anomaly_Score", "Accruals_Ratio", "Leverage_Change_YoY",
            "Leverage_Ratio", "Operating_Margin", "Margin_Volatility_3yr"]:
    if col in anomaly_df.columns:
        anomaly_df[col] = pd.to_numeric(anomaly_df[col], errors="coerce")

for col in ["Sales", "Net profit", "Cash from Operating Activity",
            "Borrowings", "Operating Profit"]:
    if col in panel_df.columns:
        panel_df[col] = pd.to_numeric(panel_df[col], errors="coerce")

# Year as int
anomaly_df["Year"] = pd.to_numeric(anomaly_df["Year"], errors="coerce").astype("Int64")
panel_df["Year"]   = pd.to_numeric(panel_df["Year"],   errors="coerce").astype("Int64")

# Merge anomaly scores into panel data for highlighting
merged_df = panel_df.merge(
    anomaly_df[["Company", "Year", "Anomaly_Score", "Is_Anomaly"]],
    on=["Company", "Year"], how="left"
)

# Sorted lists
all_industries = sorted(anomaly_df["Industry"].dropna().unique().tolist())
all_companies  = sorted(anomaly_df["Company"].dropna().unique().tolist())


# ─────────────────────────────────────────────
# 2. COLOR PALETTES & THEME CONFIGURATION
# ─────────────────────────────────────────────

# High-contrast 24-color vibrant palette for Multidimensional Feature Canvas & Sector Clusters
HIGH_CONTRAST_PALETTE = [
    "#E53926",  # Vibrant Red
    "#1D3557",  # Cobalt Blue
    "#2A9D8F",  # Viridian Teal
    "#F4A261",  # Cadmium Gold
    "#9D4EDD",  # Vivid Purple
    "#00F5D4",  # Neon Cyan
    "#FB5607",  # Electric Orange
    "#3A86FF",  # Royal Blue
    "#FF007F",  # Hot Pink
    "#38B000",  # Emerald Green
    "#7209B7",  # Deep Violet
    "#FFBE0B",  # Vivid Yellow
    "#4361EE",  # Indigo Blue
    "#F72585",  # Neon Crimson
    "#4CC9F0",  # Bright Sky Blue
    "#D4A373",  # Ochre Gold
    "#10B981",  # Mint Emerald
    "#FF9F1C",  # Bright Tangerine
    "#8338EC",  # Electric Purple
    "#00B4D8",  # Deep Cyan
    "#E63946",  # Signal Red
    "#FFB703",  # Amber Marigold
    "#52B788",  # Forest Sage
    "#D90429",  # Carmine Red
]


def get_theme_config(theme="light"):
    is_dark = (theme == "dark")
    bg_page      = "#0B0C10" if is_dark else "#F4F3EF"
    card_bg      = "#14171F" if is_dark else "#FFFFFF"
    border       = "#242936" if is_dark else "#E2DFD6"
    text_main    = "#F8F9FA" if is_dark else "#111111"
    text_dim     = "#A0AAB8" if is_dark else "#666666"
    grid_color   = "#1E2330" if is_dark else "#EAE7DE"
    accent_red   = "#FF4D4D" if is_dark else "#E53926"
    accent_blue  = "#3A86FF" if is_dark else "#1D3557"
    accent_teal  = "#00F5D4" if is_dark else "#2A9D8F"
    accent_gold  = "#FFBE0B" if is_dark else "#F4A261"

    colorscale = [
        [0.00, "#3A86FF" if is_dark else "#1D3557"],
        [0.35, "#00F5D4" if is_dark else "#2A9D8F"],
        [0.65, "#FFBE0B" if is_dark else "#F4A261"],
        [1.00, "#FF4D4D" if is_dark else "#E53926"],
    ]

    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=text_main, size=12),
        xaxis=dict(gridcolor=grid_color, linecolor=border, tickfont=dict(color=text_dim)),
        yaxis=dict(gridcolor=grid_color, linecolor=border, tickfont=dict(color=text_dim)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_dim)),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return {
        "theme": theme,
        "is_dark": is_dark,
        "bg_page": bg_page,
        "card_bg": card_bg,
        "border": border,
        "text_main": text_main,
        "text_dim": text_dim,
        "grid_color": grid_color,
        "accent_red": accent_red,
        "accent_blue": accent_blue,
        "accent_teal": accent_teal,
        "accent_gold": accent_gold,
        "colorscale": colorscale,
        "layout": layout,
    }


def card(children, className="", style=None, theme_cfg=None):
    if theme_cfg is None:
        theme_cfg = get_theme_config("light")
    base = {
        "background": theme_cfg["card_bg"],
        "border": f"1px solid {theme_cfg['border']}",
        "borderRadius": "8px",
        "padding": "24px",
        "boxShadow": "0 4px 16px rgba(0,0,0,0.3)" if theme_cfg["is_dark"] else "0 2px 8px rgba(0,0,0,0.02)",
        "transition": "all 0.2s ease",
    }
    if style:
        base.update(style)
    return html.Div(children, className=className, style=base)


def info_icon(icon_id, text, placement="top", theme_cfg=None):
    if theme_cfg is None:
        theme_cfg = get_theme_config("light")
    return html.Span(
        [
            html.Span(
                " ℹ️",
                id=icon_id,
                style={
                    "cursor": "pointer",
                    "fontSize": "11px",
                    "opacity": "0.7",
                    "marginLeft": "4px",
                    "transition": "opacity 0.2s",
                },
            ),
            dbc.Tooltip(
                text,
                target=icon_id,
                placement=placement,
                style={
                    "fontSize": "12px",
                    "maxWidth": "280px",
                    "backgroundColor": "#1A1D24" if theme_cfg["is_dark"] else "#111111",
                    "color": "#FFFFFF",
                    "padding": "8px 12px",
                    "borderRadius": "6px",
                    "boxShadow": "0 4px 16px rgba(0,0,0,0.4)",
                    "border": f"1px solid {theme_cfg['border']}" if theme_cfg["is_dark"] else "none",
                },
            ),
        ],
        style={"display": "inline-flex", "alignItems": "center"},
    )


def stat_card(label, value_id, icon, color=None, tooltip_text=None, tooltip_id=None, theme_cfg=None):
    if theme_cfg is None:
        theme_cfg = get_theme_config("light")
    if color is None:
        color = theme_cfg["accent_red"]

    label_children = [html.Span(label)]
    if tooltip_text and tooltip_id:
        label_children.append(info_icon(tooltip_id, tooltip_text, theme_cfg=theme_cfg))

    val_kwargs = {"style": {"color": color, "fontSize": "34px", "fontFamily": "'Space Grotesk', sans-serif",
                            "fontWeight": "700", "marginTop": "4px"}}
    if value_id is not None:
        val_kwargs["id"] = value_id

    return card(
        [
            html.Div(icon, style={"fontSize": "26px", "marginBottom": "6px"}),
            html.Div(label_children, style={"color": theme_cfg["text_dim"], "fontSize": "11px",
                                            "textTransform": "uppercase", "letterSpacing": "0.5px",
                                            "fontWeight": "600",
                                            "display": "flex", "alignItems": "center", "justifyContent": "center"}),
            html.Div(**val_kwargs),
        ],
        style={
            "background": theme_cfg["card_bg"],
            "border": f"1px solid {theme_cfg['border']}",
            "borderRadius": "8px",
            "padding": "20px",
            "textAlign": "center",
        },
        theme_cfg=theme_cfg,
    )


# ─────────────────────────────────────────────
# 3. DASH INITIALIZATION
# ─────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="AnomalyIQ · Financial Risk Intelligence",
    suppress_callback_exceptions=True,
)
server = app.server  # Exposed Flask server for production WSGI deployment

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        html.Div(id="theme-dummy-output", style={"display": "none"}),
        html.Div(id="app-container"),
    ]
)

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: 'Inter', sans-serif; }
        
        body.light-theme, #app-container.light-theme { background: #F4F3EF; color: #111111; }
        body.dark-theme,  #app-container.dark-theme  { background: #0B0C10; color: #F8F9FA; }
        
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        .light-theme ::-webkit-scrollbar-thumb { background: #E2DFD6; border-radius: 3px; }
        .dark-theme  ::-webkit-scrollbar-thumb { background: #242936; border-radius: 3px; }
        
        h1, h2, h3, h4, h5, h6 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.5px; }
        
        /* ── TABLE THEME STYLING (LIGHT & DARK MODE) ── */
        .light-theme table,
        .light-theme table th,
        .light-theme table td {
            color: #111111 !important;
        }
        .light-theme table th,
        .light-theme table .td-dim {
            color: #666666 !important;
        }
        .light-theme table .td-company {
            color: #111111 !important;
            font-weight: 600 !important;
        }
        
        .dark-theme table,
        .dark-theme table th,
        .dark-theme table td {
            color: #F8F9FA !important;
        }
        .dark-theme table th,
        .dark-theme table .td-dim {
            color: #A0AAB8 !important;
        }
        .dark-theme table .td-company {
            color: #F8F9FA !important;
            font-weight: 600 !important;
        }
        
        /* ── LIGHT THEME DROPDOWNS ── */
        body.light-theme .Select-control,
        .light-theme .Select-control,
        body.light-theme div[class*="-control"],
        .light-theme div[class*="-control"] {
            background-color: #FFFFFF !important;
            border-color: #E2DFD6 !important;
            color: #111111 !important;
        }
        body.light-theme .Select-menu-outer,
        .light-theme .Select-menu-outer,
        body.light-theme div[class*="-menu"],
        .light-theme div[class*="-menu"] {
            background-color: #FFFFFF !important;
            border-color: #E2DFD6 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
            color: #111111 !important;
        }
        body.light-theme .Select-option,
        .light-theme .Select-option,
        body.light-theme div[class*="-option"],
        .light-theme div[class*="-option"] {
            background-color: #FFFFFF !important;
            color: #111111 !important;
        }
        body.light-theme .Select-option:hover,
        .light-theme .Select-option:hover,
        body.light-theme .Select-option.is-focused,
        .light-theme .Select-option.is-focused,
        body.light-theme div[class*="-option"]:hover,
        .light-theme div[class*="-option"]:hover {
            background-color: #F4F3EF !important;
            color: #E53926 !important;
            font-weight: 600 !important;
        }
        body.light-theme .Select-value-label,
        .light-theme .Select-value-label,
        body.light-theme .Select-single-value,
        .light-theme .Select-single-value,
        body.light-theme div[class*="-singleValue"],
        .light-theme div[class*="-singleValue"],
        body.light-theme .Select-input input,
        .light-theme .Select-input input,
        body.light-theme div[class*="-Input"] input,
        .light-theme div[class*="-Input"] input {
            color: #111111 !important;
        }
        body.light-theme .Select-placeholder,
        .light-theme .Select-placeholder,
        body.light-theme div[class*="-placeholder"],
        .light-theme div[class*="-placeholder"] {
            color: #666666 !important;
        }

        /* ── DARK THEME DROPDOWNS ── */
        body.dark-theme .Select-control,
        .dark-theme .Select-control,
        body.dark-theme div[class*="-control"],
        .dark-theme div[class*="-control"] {
            background-color: #14171F !important;
            border-color: #242936 !important;
            color: #F8F9FA !important;
        }
        body.dark-theme .Select-menu-outer,
        .dark-theme .Select-menu-outer,
        body.dark-theme div[class*="-menu"],
        .dark-theme div[class*="-menu"] {
            background-color: #14171F !important;
            border-color: #242936 !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6) !important;
            color: #F8F9FA !important;
        }
        body.dark-theme .Select-option,
        .dark-theme .Select-option,
        body.dark-theme div[class*="-option"],
        .dark-theme div[class*="-option"] {
            background-color: #14171F !important;
            color: #F8F9FA !important;
        }
        body.dark-theme .Select-option:hover,
        .dark-theme .Select-option:hover,
        body.dark-theme .Select-option.is-focused,
        .dark-theme .Select-option.is-focused,
        body.dark-theme div[class*="-option"]:hover,
        .dark-theme div[class*="-option"]:hover {
            background-color: #242936 !important;
            color: #FF4D4D !important;
            font-weight: 600 !important;
        }
        body.dark-theme .Select-value-label,
        .dark-theme .Select-value-label,
        body.dark-theme .Select-single-value,
        .dark-theme .Select-single-value,
        body.dark-theme div[class*="-singleValue"],
        .dark-theme div[class*="-singleValue"],
        body.dark-theme .Select-input input,
        .dark-theme .Select-input input,
        body.dark-theme div[class*="-Input"] input,
        .dark-theme div[class*="-Input"] input {
            color: #F8F9FA !important;
        }
        body.dark-theme .Select-placeholder,
        .dark-theme .Select-placeholder,
        body.dark-theme div[class*="-placeholder"],
        .dark-theme div[class*="-placeholder"] {
            color: #A0AAB8 !important;
        }
        body.dark-theme .Select-arrow,
        .dark-theme .Select-arrow {
            border-color: #A0AAB8 transparent transparent !important;
        }
        body.dark-theme .Select-value,
        .dark-theme .Select-value,
        body.dark-theme div[class*="-multiValue"],
        .dark-theme div[class*="-multiValue"] {
            background-color: #242936 !important;
            border-color: #3D4852 !important;
            color: #F8F9FA !important;
        }
        body.dark-theme .Select-value-icon,
        .dark-theme .Select-value-icon {
            border-right-color: #3D4852 !important;
            color: #F8F9FA !important;
        }
        body.dark-theme div[class*="-Input"] {
            color: #F8F9FA !important;
        }
        
        /* Slider Theme Support */
        .light-theme .rc-slider-rail { background-color: #E2DFD6 !important; }
        .dark-theme  .rc-slider-rail { background-color: #242936 !important; }
        
        .light-theme .rc-slider-track { background-color: #E53926 !important; }
        .dark-theme  .rc-slider-track { background-color: #FF4D4D !important; }
        
        .light-theme .rc-slider-handle { border-color: #E53926 !important; background-color: #FFFFFF !important; }
        .dark-theme  .rc-slider-handle { border-color: #FF4D4D !important; background-color: #14171F !important; }

        .rc-slider-mark-text {
            font-size: 10px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            white-space: nowrap !important;
        }
        .light-theme .rc-slider-mark-text { color: #666666 !important; }
        .dark-theme  .rc-slider-mark-text { color: #A0AAB8 !important; }
    </style>
    <script>
        window.dash_clientside = Object.assign({}, window.dash_clientside, {
            clientside: {
                toggleTheme: function(theme) {
                    if (theme === 'dark') {
                        document.body.className = 'dark-theme';
                    } else {
                        document.body.className = 'light-theme';
                    }
                    return '';
                }
            }
        });
    </script>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""


# ─────────────────────────────────────────────
# 4. THEME SWITCHING CALLBACKS
# ─────────────────────────────────────────────

@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme_store(n_clicks, current_theme):
    return "dark" if current_theme == "light" else "light"


app.clientside_callback(
    dash.ClientsideFunction(namespace="clientside", function_name="toggleTheme"),
    Output("theme-dummy-output", "children"),
    Input("theme-store", "data"),
)


# ─────────────────────────────────────────────
# 5. PAGE HEADERS & NAVBAR BUILDERS
# ─────────────────────────────────────────────

def render_navbar_links(pathname, cfg):
    def get_link_style(path):
        is_active = (pathname == path) or (path == "/industry" and pathname in ("/", "", "/industry", "/industry/"))
        if is_active:
            return {
                "color": cfg["text_main"],
                "borderBottom": f"2.5px solid {cfg['accent_red']}",
                "fontWeight": "700",
                "paddingBottom": "4px",
                "textDecoration": "none",
                "fontSize": "12px",
                "letterSpacing": "0.5px",
            }
        else:
            return {
                "color": cfg["text_dim"],
                "borderBottom": "2.5px solid transparent",
                "fontWeight": "500",
                "paddingBottom": "4px",
                "textDecoration": "none",
                "fontSize": "12px",
                "letterSpacing": "0.5px",
            }

    return html.Div(
        [
            dcc.Link("01  INDUSTRY OVERVIEW", href="/industry", style=get_link_style("/industry")),
            dcc.Link("02  ANOMALY EXPLORER", href="/anomaly", style=get_link_style("/anomaly")),
            dcc.Link("03  COMPANY DEEP DIVE", href="/company", style=get_link_style("/company")),
            dcc.Link("04  METHODOLOGY", href="/methodology", style=get_link_style("/methodology")),
        ],
        style={"display": "flex", "gap": "28px", "alignItems": "center"}
    )


def page_header(section_num, title, subtitle, cfg):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(f"{section_num} —— ", style={"color": cfg["accent_red"], "fontWeight": "700", "fontSize": "12px", "letterSpacing": "0.5px"}),
                            html.Span(subtitle.upper(), style={"color": cfg["text_dim"], "fontWeight": "600", "fontSize": "11px", "letterSpacing": "0.5px"}),
                        ],
                        style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}
                    ),
                    html.H1(title, style={"color": cfg["text_main"], "fontWeight": "700", "fontSize": "38px",
                                          "letterSpacing": "-0.5px", "margin": "0", "lineHeight": "1.1", "fontFamily": "'Space Grotesk', sans-serif"}),
                ]
            ),
            html.Div(
                [
                    html.Span("● ", style={"color": cfg["accent_red"], "fontSize": "14px"}),
                    html.Span("NIFTY 500 MODEL LIVE", style={"color": cfg["text_main"], "fontWeight": "700", "fontSize": "11px", "letterSpacing": "0.5px"}),
                ],
                style={"background": cfg["card_bg"], "border": f"1px solid {cfg['border']}", "padding": "8px 16px",
                       "borderRadius": "20px", "display": "flex", "alignItems": "center"}
            )
        ],
        style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end",
               "marginBottom": "36px", "borderBottom": f"1px solid {cfg['border']}", "paddingBottom": "24px"}
    )


# ─────────────────────────────────────────────
# 6. PAGE LAYOUT BUILDERS
# ─────────────────────────────────────────────

def layout_industry(cfg):
    return html.Div(
        [
            page_header("01", "Industry Risk Overview", "Aggregate financial health & anomaly distribution across Nifty 500 sectors", cfg),
            
            # KPI row
            html.Div(
                [
                    stat_card("Total Companies", "kpi-companies", "🏢", cfg["accent_blue"], "Total active Nifty 500 companies tracked across 10 fiscal years.", "tt-kpi-comp", theme_cfg=cfg),
                    stat_card("Anomalous Years", "kpi-anomalies", "⚠️", cfg["accent_red"], "Total company-years flagged as mathematical outliers (Anomaly Score > 0.2).", "tt-kpi-anom", theme_cfg=cfg),
                    stat_card("Industries Covered", "kpi-industries", "🏭", cfg["accent_teal"], "Sectoral breakdown derived from Screen.in / NSE classifications.", "tt-kpi-ind", theme_cfg=cfg),
                    stat_card("Years of Data", "kpi-years", "📅", cfg["accent_gold"], "Historical financial panel dataset spanning FY2015 to FY2024.", "tt-kpi-years", theme_cfg=cfg),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "24px"},
            ),

            # Filter bar
            card(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(["Filter Industry", info_icon("tt-ind-filter", "Select specific sectors to filter all charts and metrics on this page.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "letterSpacing": "0.5px", "fontWeight": "600", "textTransform": "uppercase", "marginBottom": "6px", "display": "flex", "alignItems": "center"}),
                                dcc.Dropdown(
                                    id="ind-filter",
                                    options=[{"label": i, "value": i} for i in all_industries],
                                    multi=True,
                                    placeholder="All industries…",
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label(["Min. Anomaly Score Threshold", info_icon("tt-score-slider", "Minimum Isolation Forest outlier threshold. Higher scores isolate extreme financial anomalies.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "letterSpacing": "0.5px", "fontWeight": "600", "textTransform": "uppercase", "marginBottom": "6px", "display": "flex", "alignItems": "center"}),
                                dcc.Slider(
                                    id="score-threshold",
                                    min=0, max=0.5, step=0.01, value=0.1,
                                    marks={0: "0", 0.25: "0.25", 0.5: "0.5"},
                                    tooltip={"placement": "bottom"},
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                    ],
                    style={"display": "flex", "gap": "32px", "alignItems": "flex-start"},
                ),
                style={"marginBottom": "24px"},
                theme_cfg=cfg,
            ),

            # Charts row 1
            html.Div(
                [
                    card(
                        [
                            html.H6(["Anomaly Rate by Industry", info_icon("tt-h-rate", "Percentage of company-years within each industry flagged as financial outliers.", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-anomaly-rate", config={"displayModeBar": False}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.H6(["Anomaly Score Distribution", info_icon("tt-h-dist", "Box plot displaying score range, median, and outliers across sectors.", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-score-dist", config={"displayModeBar": False}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
            ),

            # Charts row 2
            html.Div(
                [
                    card(
                        [
                            html.H6(["Average Operating Margin by Sector", info_icon("tt-h-margin", "Sector-wide average operating profit margin (Operating Profit / Sales).", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-margin-sector", config={"displayModeBar": False}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.H6(["Anomaly Trend Over Years", info_icon("tt-h-trend", "Annual count and percentage of anomalous companies from 2015 to 2024.", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-year-trend", config={"displayModeBar": False}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                ],
                style={"display": "flex", "gap": "16px"},
            ),
        ]
    )


def layout_anomaly(cfg):
    return html.Div(
        [
            page_header("02", "Anomaly Explorer", "Interactive scatter plot — hover any dot to inspect a company's risk profile", cfg),

            # Controls
            card(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(["X Axis Metric", info_icon("tt-scat-x", "Select financial feature mapped to horizontal axis.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px", "textTransform": "uppercase", "display": "flex", "alignItems": "center"}),
                                dcc.Dropdown(
                                    id="scatter-x",
                                    options=[
                                        {"label": "Accruals Ratio", "value": "Accruals_Ratio"},
                                        {"label": "Leverage Change YoY", "value": "Leverage_Change_YoY"},
                                        {"label": "Leverage Ratio", "value": "Leverage_Ratio"},
                                        {"label": "Operating Margin", "value": "Operating_Margin"},
                                        {"label": "Margin Volatility 3yr", "value": "Margin_Volatility_3yr"},
                                    ],
                                    value="Accruals_Ratio",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label(["Y Axis Metric", info_icon("tt-scat-y", "Select financial feature mapped to vertical axis.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px", "textTransform": "uppercase", "display": "flex", "alignItems": "center"}),
                                dcc.Dropdown(
                                    id="scatter-y",
                                    options=[
                                        {"label": "Accruals Ratio", "value": "Accruals_Ratio"},
                                        {"label": "Leverage Change YoY", "value": "Leverage_Change_YoY"},
                                        {"label": "Leverage Ratio", "value": "Leverage_Ratio"},
                                        {"label": "Operating Margin", "value": "Operating_Margin"},
                                        {"label": "Margin Volatility 3yr", "value": "Margin_Volatility_3yr"},
                                    ],
                                    value="Leverage_Change_YoY",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label(["Color Scheme Mode", info_icon("tt-scat-colmode", "Choose visual mode: Risk Heatmap or Multi-Color Sector Clusters.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px", "textTransform": "uppercase", "display": "flex", "alignItems": "center"}),
                                dcc.Dropdown(
                                    id="scatter-colmode",
                                    options=[
                                        {"label": "Risk Score Heatmap (High Contrast)", "value": "score"},
                                        {"label": "Sector Clusters (24-Color Palette)", "value": "industry"},
                                    ],
                                    value="score",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1.2"},
                        ),
                        html.Div(
                            [
                                html.Label(["Filter Industry", info_icon("tt-scat-ind", "Isolate specific sectors in the scatter plot.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px", "textTransform": "uppercase", "display": "flex", "alignItems": "center"}),
                                dcc.Dropdown(
                                    id="scatter-industry",
                                    options=[{"label": i, "value": i} for i in all_industries],
                                    multi=True,
                                    placeholder="All…",
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label(["Year Range", info_icon("tt-scat-yr", "Filter financial data across specific years.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px", "textTransform": "uppercase", "display": "flex", "alignItems": "center"}),
                                dcc.RangeSlider(
                                    id="scatter-years",
                                    min=int(anomaly_df["Year"].min()),
                                    max=int(anomaly_df["Year"].max()),
                                    step=1,
                                    value=[int(anomaly_df["Year"].min()), int(anomaly_df["Year"].max())],
                                    marks={y: str(y) for y in list(range(int(anomaly_df["Year"].min()), int(anomaly_df["Year"].max()) + 1, 5)) + [int(anomaly_df["Year"].max())]},
                                    tooltip={"placement": "bottom"},
                                ),
                            ],
                            style={"flex": "2", "minWidth": "260px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "alignItems": "flex-start", "flexWrap": "wrap"},
                ),
                style={"marginBottom": "20px"},
                theme_cfg=cfg,
            ),

            # Main scatter
            card(
                [
                    html.Div(
                        [
                            html.Span("Multi-Dimensional Feature Canvas (High-Contrast Cluster View)", style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                            info_icon("tt-scat-canvas", "Each dot represents a company-year. High-contrast colors highlight risk clusters.", theme_cfg=cfg),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginBottom": "12px"},
                    ),
                    dcc.Graph(id="scatter-main",
                               config={"displayModeBar": True,
                                       "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                               style={"height": "540px"}),
                ],
                style={"marginBottom": "16px"},
                theme_cfg=cfg,
            ),

            # Top anomalies table
            card(
                [
                    html.H6(["🚨 Top 20 Most Anomalous Company-Years", info_icon("tt-top20", "Ranked list of company-years with highest anomaly scores.", theme_cfg=cfg)],
                             style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                    html.Div(id="top-anomalies-table"),
                ],
                theme_cfg=cfg,
            ),
        ]
    )


def layout_company(cfg):
    return html.Div(
        [
            page_header("03", "Company Deep Dive", "Select any company to explore its 10-year financial trajectory and flagged anomaly years", cfg),

            # Company selector
            card(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(["Select Company", info_icon("tt-comp-sel", "Search & select any company to view its 10-year financial health trajectory.", theme_cfg=cfg)],
                                           style={"color": cfg["text_dim"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px", "textTransform": "uppercase", "display": "flex", "alignItems": "center"}),
                                dcc.Dropdown(
                                    id="company-select",
                                    options=[{"label": c, "value": c} for c in all_companies],
                                    value=all_companies[0] if all_companies else None,
                                    clearable=False,
                                    style={"minWidth": "320px"},
                                ),
                            ]
                        ),
                        html.Div(id="company-badges",
                                 style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"}),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end", "flexWrap": "wrap", "gap": "16px"},
                ),
                style={"marginBottom": "20px"},
                theme_cfg=cfg,
            ),

            # KPI row
            html.Div(id="company-kpis",
                     style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "20px"}),

            # Charts
            html.Div(
                [
                    card(
                        [
                            html.H6(["Revenue & Net Profit (₹ Cr)", info_icon("tt-rev-prof", "Annual top-line sales (bars) vs bottom-line profit (line).", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "12px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-revenue", config={"displayModeBar": False}, style={"height": "300px"}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.H6(["Cash from Operations vs Net Profit", info_icon("tt-cfo-prof", "Operating Cash Flow vs Net Profit. A persistent gap signals accruals risk.", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "12px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-cashflow", config={"displayModeBar": False}, style={"height": "300px"}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
            ),

            html.Div(
                [
                    card(
                        [
                            html.H6(["Debt (Borrowings) Over Time", info_icon("tt-debt-trend", "Total borrowings per year. Highlights rapid leverage growth.", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "12px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-debt", config={"displayModeBar": False}, style={"height": "280px"}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.H6(["Anomaly Score Timeline", info_icon("tt-score-timeline", "Historical anomaly score curve. Shaded vertical bands highlight flagged anomaly years.", theme_cfg=cfg)],
                                    style={"color": cfg["text_main"], "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "12px", "display": "flex", "alignItems": "center"}),
                            dcc.Graph(id="chart-anomaly-timeline", config={"displayModeBar": False}, style={"height": "280px"}),
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                ],
                style={"display": "flex", "gap": "16px"},
            ),
        ]
    )


def layout_methodology(cfg):
    return html.Div(
        [
            page_header("04", "Methodology & Financial Theory", "Why we built this platform, why each chart is necessary, and the mathematical properties driving our ML pipeline", cfg),
            
            # Mission Card
            card(
                [
                    html.H4("🎯 Why Build an Accounting Anomaly Detection System?", style={"color": cfg["text_main"], "fontWeight": "700", "marginBottom": "12px"}),
                    html.P(
                        "Traditional equity analysis relies heavily on stock price charts or simple P/E valuation ratios. "
                        "However, stock price fluctuations often obscure fundamental financial health. Companies undergoing "
                        "earnings manipulation, unsustainable debt expansion, or operational decay frequently mask these issues "
                        "in published financial statements until catastrophic defaults or write-downs occur.",
                        style={"color": cfg["text_dim"], "lineHeight": "1.6", "fontSize": "14px"}
                    ),
                    html.P(
                        "AnomalyIQ addresses this by analyzing full 10-year financial disclosures (Profit & Loss, Balance Sheet, Cash Flows) "
                        "for 492 Nifty 500 companies. Using Unsupervised Machine Learning (Isolation Forests), it automatically detects "
                        "multivariate statistical outliers without relying on hindsight bias or subjective rules.",
                        style={"color": cfg["text_dim"], "lineHeight": "1.6", "fontSize": "14px", "margin": "0"}
                    ),
                ],
                style={"marginBottom": "24px"},
                theme_cfg=cfg,
            ),

            # Financial Ratios & Properties Grid
            html.H5("🧮 Core Financial Ratios & Mathematical Properties", style={"color": cfg["text_main"], "fontWeight": "700", "marginBottom": "16px"}),
            html.Div(
                [
                    card(
                        [
                            html.Div("1️⃣ Accruals Ratio (Sloan's Anomaly)", style={"color": cfg["accent_red"], "fontWeight": "700", "fontSize": "15px", "marginBottom": "6px"}),
                            html.Div("Formula: (Net Profit - Operating Cash Flow) / Total Assets", style={"color": cfg["text_main"], "fontSize": "12px", "fontFamily": "monospace", "background": cfg["bg_page"], "padding": "4px 8px", "borderRadius": "4px", "marginBottom": "12px", "display": "inline-block"}),
                            html.P(
                                "Accounting profit is based on accrual accounting, which includes unpaid invoices and uncollected revenues. "
                                "When reported Net Profit consistently exceeds Cash from Operating Activities, the Accruals Ratio spikes. "
                                "Historically (Sloan, 1996), high positive accruals signal low-quality earnings vulnerable to future earnings restatements.",
                                style={"color": cfg["text_dim"], "fontSize": "13px", "lineHeight": "1.5", "margin": "0"}
                            )
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.Div("2️⃣ Leverage Change YoY", style={"color": cfg["accent_red"], "fontWeight": "700", "fontSize": "15px", "marginBottom": "6px"}),
                            html.Div("Formula: (Debt_t - Debt_t-1) / Total Assets_t-1", style={"color": cfg["text_main"], "fontSize": "12px", "fontFamily": "monospace", "background": cfg["bg_page"], "padding": "4px 8px", "borderRadius": "4px", "marginBottom": "12px", "display": "inline-block"}),
                            html.P(
                                "Tracks rapid debt expansion relative to capital structure. "
                                "Sudden jumps in leverage indicate aggressive debt-fueled expansion, refinancing distress, or reliance on external borrowing to maintain operations.",
                                style={"color": cfg["text_dim"], "fontSize": "13px", "lineHeight": "1.5", "margin": "0"}
                            )
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.Div("3️⃣ Margin Volatility (3-Year Rolling Std Dev)", style={"color": cfg["accent_red"], "fontWeight": "700", "fontSize": "15px", "marginBottom": "6px"}),
                            html.Div("Formula: StdDev(Operating Margin_t-2..t)", style={"color": cfg["text_main"], "fontSize": "12px", "fontFamily": "monospace", "background": cfg["bg_page"], "padding": "4px 8px", "borderRadius": "4px", "marginBottom": "12px", "display": "inline-block"}),
                            html.P(
                                "Measures the operational stability of core business margins. "
                                "High volatility highlights erratic pricing power, vulnerability to commodity swings, or inconsistent expense recognition.",
                                style={"color": cfg["text_dim"], "fontSize": "13px", "lineHeight": "1.5", "margin": "0"}
                            )
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "24px"}
            ),

            # Machine Learning Section
            html.Div(
                [
                    card(
                        [
                            html.H5("🤖 Isolation Forest Model Architecture", style={"color": cfg["text_main"], "fontWeight": "700", "marginBottom": "12px"}),
                            html.P(
                                "Financial anomalies are rare, diverse, and unlabelled. Supervised models fail because true fraud labels are scarce. "
                                "Isolation Forests build decision trees by randomly selecting features and split values. "
                                "Anomalous data points require far fewer splits to isolate compared to normal clusters, giving them high anomaly scores.",
                                style={"color": cfg["text_dim"], "fontSize": "13px", "lineHeight": "1.5"}
                            ),
                            html.P(
                                "Our pipeline runs Isolation Forests grouped per Industry to evaluate each company relative to its sector peers.",
                                style={"color": cfg["text_dim"], "fontSize": "13px", "lineHeight": "1.5", "margin": "0"}
                            )
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                    card(
                        [
                            html.H5("📊 Why Each Visualization View is Necessary", style={"color": cfg["text_main"], "fontWeight": "700", "marginBottom": "12px"}),
                            html.Ul(
                                [
                                    html.Li("Industry Overview: Uncovers systemic risk concentration across sectors (e.g. Real Estate vs FMCG).", style={"marginBottom": "8px"}),
                                    html.Li("Anomaly Explorer: 2D feature scatter mapping multi-dimensional risk interaction (e.g. Accruals vs Leverage Δ).", style={"marginBottom": "8px"}),
                                    html.Li("Company Deep Dive: 10-year longitudinal drilldown to verify profit-cashflow gaps & anomaly years.", style={"margin": "0"}),
                                ],
                                style={"color": cfg["text_dim"], "fontSize": "13px", "lineHeight": "1.5", "paddingLeft": "18px", "margin": "0"}
                            )
                        ],
                        style={"flex": "1"},
                        theme_cfg=cfg,
                    ),
                ],
                style={"display": "flex", "gap": "16px"}
            ),
        ]
    )


# ─────────────────────────────────────────────
# 7. MAIN APP CONTAINER RENDER CALLBACK
# ─────────────────────────────────────────────

@app.callback(
    Output("app-container", "children"),
    Output("app-container", "style"),
    Output("app-container", "className"),
    Input("url", "pathname"),
    Input("theme-store", "data"),
)
def render_app_container(pathname, theme):
    cfg = get_theme_config(theme)

    btn_text = "🌙 DARK MODE" if theme == "light" else "☀️ LIGHT MODE"
    btn_bg   = cfg["card_bg"]
    btn_fg   = cfg["text_main"]

    navbar = html.Div(
        [
            html.Div(
                [
                    # Brand title left
                    html.Div(
                        [
                            html.Span("ANOMALY", style={"fontFamily": "'Space Grotesk', sans-serif",
                                                        "fontWeight": "700", "fontSize": "22px",
                                                        "letterSpacing": "0.5px", "color": cfg["text_main"]}),
                            html.Span("IQ", style={"fontFamily": "'Space Grotesk', sans-serif",
                                                   "fontWeight": "700", "fontSize": "22px",
                                                   "letterSpacing": "0.5px", "color": cfg["accent_red"]}),
                            html.Span("FINANCIAL RISK & ANOMALY SYSTEM",
                                      style={"fontSize": "10px", "letterSpacing": "0.5px",
                                             "color": cfg["text_dim"], "display": "block", "marginTop": "2px", "fontWeight": "600"}),
                        ]
                    ),
                    # Nav links right & theme button
                    html.Div(
                        [
                            render_navbar_links(pathname, cfg),
                            html.Button(
                                btn_text,
                                id="theme-toggle-btn",
                                n_clicks=0,
                                style={
                                    "background": btn_bg,
                                    "color": btn_fg,
                                    "border": f"1px solid {cfg['border']}",
                                    "borderRadius": "20px",
                                    "padding": "6px 16px",
                                    "fontSize": "11px",
                                    "fontWeight": "700",
                                    "cursor": "pointer",
                                    "letterSpacing": "0.5px",
                                    "transition": "all 0.2s ease",
                                },
                            ),
                        ],
                        style={"display": "flex", "gap": "24px", "alignItems": "center"},
                    ),
                ],
                style={"maxWidth": "1400px", "margin": "0 auto", "display": "flex",
                       "justifyContent": "space-between", "alignItems": "center", "padding": "0 24px"},
            )
        ],
        style={
            "height": "76px",
            "background": cfg["bg_page"],
            "borderBottom": f"1px solid {cfg['border']}",
            "position": "sticky",
            "top": "0",
            "zIndex": "1000",
            "display": "flex",
            "alignItems": "center",
        },
    )

    if pathname in ("/anomaly", "/anomaly/"):
        page_layout = layout_anomaly(cfg)
    elif pathname in ("/company", "/company/"):
        page_layout = layout_company(cfg)
    elif pathname in ("/methodology", "/methodology/"):
        page_layout = layout_methodology(cfg)
    else:
        page_layout = layout_industry(cfg)

    content = html.Div(
        [
            html.Div(page_layout, id="page-content", style={"maxWidth": "1400px", "margin": "0 auto", "padding": "36px 24px"}),
        ],
        style={"background": cfg["bg_page"], "minHeight": "calc(100vh - 76px)"},
    )

    container_style = {
        "background": cfg["bg_page"],
        "color": cfg["text_main"],
        "minHeight": "100vh",
        "fontFamily": "'Inter', sans-serif",
    }

    container_className = "dark-theme" if theme == "dark" else "light-theme"

    return [navbar, content], container_style, container_className


# ─────────────────────────────────────────────
# 8. INDUSTRY PAGE CALLBACKS
# ─────────────────────────────────────────────

@app.callback(
    Output("kpi-companies", "children"),
    Output("kpi-anomalies", "children"),
    Output("kpi-industries", "children"),
    Output("kpi-years", "children"),
    Input("url", "pathname"),
)
def update_kpis(_):
    n_companies  = anomaly_df["Company"].nunique()
    n_anomalies  = int(anomaly_df["Is_Anomaly"].sum())
    n_industries = anomaly_df["Industry"].nunique()
    n_years      = anomaly_df["Year"].nunique()
    return str(n_companies), str(n_anomalies), str(n_industries), str(n_years)


def _filter_industry_df(industries, score_threshold):
    df = anomaly_df.copy()
    if industries:
        df = df[df["Industry"].isin(industries)]
    return df


@app.callback(
    Output("chart-anomaly-rate", "figure"),
    Output("chart-score-dist", "figure"),
    Output("chart-margin-sector", "figure"),
    Output("chart-year-trend", "figure"),
    Input("ind-filter", "value"),
    Input("score-threshold", "value"),
    Input("theme-store", "data"),
)
def update_industry_charts(industries, score_threshold, theme):
    cfg = get_theme_config(theme)
    df = _filter_industry_df(industries, score_threshold)

    # --- Chart 1: Anomaly rate by industry ---
    grp = df.groupby("Industry").agg(
        total=("Company", "count"),
        anomalies=("Is_Anomaly", "sum"),
    ).reset_index()
    grp["rate"] = (grp["anomalies"] / grp["total"] * 100).round(1)
    grp = grp.sort_values("rate", ascending=True).tail(15)

    line_outline = "#14171F" if cfg["is_dark"] else "#FFFFFF"

    fig1 = go.Figure(go.Bar(
        y=grp["Industry"],
        x=grp["rate"],
        orientation="h",
        marker=dict(
            color=grp["rate"],
            colorscale=cfg["colorscale"],
            showscale=False,
            line=dict(width=1, color=line_outline),
        ),
        text=grp["rate"].astype(str) + "%",
        textposition="outside",
        textfont=dict(color=cfg["text_dim"], size=11, family="Inter, sans-serif"),
    ))
    fig1.update_layout(**cfg["layout"], height=380, xaxis_title="Anomaly Rate (%)", yaxis_title="")

    # --- Chart 2: Score distribution ---
    top_ind = (df.groupby("Industry")["Anomaly_Score"].median()
                 .sort_values(ascending=False).head(12).index.tolist())
    df2 = df[df["Industry"].isin(top_ind)]
    fig2 = px.box(
        df2, x="Industry", y="Anomaly_Score",
        color_discrete_sequence=[cfg["accent_blue"]],
        points="outliers",
    )
    fig2.update_layout(**cfg["layout"], height=380, xaxis_tickangle=-30, xaxis_title="", yaxis_title="Anomaly Score")
    fig2.update_traces(marker_color=cfg["accent_red"], marker_size=6,
                       line_color=cfg["accent_blue"],
                       fillcolor="rgba(58,134,255,0.15)" if cfg["is_dark"] else "rgba(29,53,87,0.15)")

    # --- Chart 3: Avg operating margin ---
    margin_grp = (anomaly_df.dropna(subset=["Operating_Margin"])
                  .groupby("Industry")["Operating_Margin"]
                  .mean().reset_index()
                  .sort_values("Operating_Margin", ascending=True).tail(15))
    fig3 = go.Figure(go.Bar(
        y=margin_grp["Industry"],
        x=margin_grp["Operating_Margin"] * 100,
        orientation="h",
        marker=dict(
            color=margin_grp["Operating_Margin"],
            colorscale=[[0, cfg["accent_red"]], [0.5, cfg["accent_gold"]], [1, cfg["accent_teal"]]],
            showscale=False,
            line=dict(width=1, color=line_outline),
        ),
        text=(margin_grp["Operating_Margin"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(color=cfg["text_dim"], size=11),
    ))
    fig3.update_layout(**cfg["layout"], height=380, xaxis_title="Avg Operating Margin (%)", yaxis_title="")

    # --- Chart 4: Anomaly trend ---
    year_grp = df.groupby("Year").agg(
        anomalies=("Is_Anomaly", "sum"),
        total=("Company", "count"),
    ).reset_index()
    year_grp["rate"] = year_grp["anomalies"] / year_grp["total"] * 100

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=year_grp["Year"].astype(str), y=year_grp["anomalies"],
        name="# Anomalies", mode="lines+markers",
        line=dict(color=cfg["accent_red"], width=3),
        marker=dict(size=8, color=cfg["accent_red"], symbol="circle", line=dict(width=1.5, color=line_outline)),
        fill="tozeroy", fillcolor="rgba(255,77,77,0.12)" if cfg["is_dark"] else "rgba(229,57,38,0.08)",
    ))
    fig4.add_trace(go.Scatter(
        x=year_grp["Year"].astype(str), y=year_grp["rate"],
        name="Rate (%)", mode="lines+markers",
        line=dict(color=cfg["accent_blue"], width=2.5, dash="dot"),
        marker=dict(size=7, color=cfg["accent_blue"], symbol="diamond", line=dict(width=1.5, color=line_outline)),
        yaxis="y2",
    ))
    fig4.update_layout(
        **cfg["layout"],
        height=380,
        yaxis2=dict(overlaying="y", side="right", gridcolor=cfg["grid_color"],
                    tickfont=dict(color=cfg["text_dim"])),
    )
    fig4.update_layout(legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font=dict(color=cfg["text_dim"])))
    return fig1, fig2, fig3, fig4


# ─────────────────────────────────────────────
# 9. ANOMALY EXPLORER CALLBACKS (24-Color Canvas)
# ─────────────────────────────────────────────

@app.callback(
    Output("scatter-main", "figure"),
    Output("top-anomalies-table", "children"),
    Input("scatter-x", "value"),
    Input("scatter-y", "value"),
    Input("scatter-colmode", "value"),
    Input("scatter-industry", "value"),
    Input("scatter-years", "value"),
    Input("theme-store", "data"),
)
def update_scatter(x_col, y_col, col_mode, industries, year_range, theme):
    cfg = get_theme_config(theme)
    df = anomaly_df.copy()
    if industries:
        df = df[df["Industry"].isin(industries)]
    if year_range:
        df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    df = df.dropna(subset=[x_col, y_col, "Anomaly_Score"])

    for col in [x_col, y_col]:
        q1, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q99)

    df["hover_text"] = (
        "<b>" + df["Company"] + "</b><br>" +
        "Year: " + df["Year"].astype(str) + "<br>" +
        "Industry: " + df["Industry"].fillna("Unknown") + "<br>" +
        x_col.replace("_", " ") + ": " + df[x_col].round(3).astype(str) + "<br>" +
        y_col.replace("_", " ") + ": " + df[y_col].round(3).astype(str) + "<br>" +
        "Anomaly Score: " + df["Anomaly_Score"].round(3).astype(str) + "<br>" +
        "🚨 Flagged: " + df["Is_Anomaly"].map({1: "Yes", 0: "No"}).astype(str)
    )

    fig = go.Figure()
    line_outline = "#14171F" if cfg["is_dark"] else "#FFFFFF"

    if col_mode == "industry":
        # --- Multi-Color Sector Clusters Mode (24-Color High Contrast) ---
        present_industries = sorted(df["Industry"].unique().tolist())
        for idx, ind in enumerate(present_industries):
            ind_df = df[df["Industry"] == ind]
            color_hex = HIGH_CONTRAST_PALETTE[idx % len(HIGH_CONTRAST_PALETTE)]

            fig.add_trace(go.Scatter(
                x=ind_df[x_col], y=ind_df[y_col],
                mode="markers",
                name=ind,
                hovertext=ind_df["hover_text"],
                hoverinfo="text",
                marker=dict(
                    size=np.maximum(0, ind_df["Anomaly_Score"].fillna(0).clip(0, None) * 28 + 7),
                    color=color_hex,
                    opacity=0.88,
                    line=dict(width=1, color=line_outline),
                ),
            ))
    else:
        # --- High-Contrast Risk Heatmap Mode ---
        normal = df[df["Is_Anomaly"] == 0]
        fig.add_trace(go.Scatter(
            x=normal[x_col], y=normal[y_col],
            mode="markers",
            name="Normal Baseline",
            hovertext=normal["hover_text"],
            hoverinfo="text",
            marker=dict(
                size=6,
                color=cfg["accent_blue"],
                opacity=0.55,
                line=dict(width=0.8, color=line_outline),
            ),
        ))

        anom = df[df["Is_Anomaly"] == 1]
        fig.add_trace(go.Scatter(
            x=anom[x_col], y=anom[y_col],
            mode="markers",
            name="Anomaly Outlier",
            hovertext=anom["hover_text"],
            hoverinfo="text",
            marker=dict(
                size=np.maximum(0, anom["Anomaly_Score"].fillna(0).clip(0, None) * 36 + 9),
                color=anom["Anomaly_Score"],
                colorscale=cfg["colorscale"],
                opacity=0.95,
                line=dict(width=1.5, color=line_outline),
                showscale=True,
                colorbar=dict(
                    title=dict(text="Risk Score", font=dict(color=cfg["text_dim"], family="Inter, sans-serif")),
                    tickfont=dict(color=cfg["text_dim"]),
                    thickness=14,
                ),
            ),
        ))

    fig.update_layout(
        **cfg["layout"],
        height=540,
        xaxis_title=x_col.replace("_", " "),
        yaxis_title=y_col.replace("_", " "),
        hovermode="closest",
    )
    fig.update_layout(legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)", font=dict(color=cfg["text_dim"])))

    # Table styling for Dark / Light mode
    top20 = (anomaly_df[anomaly_df["Is_Anomaly"] == 1]
             .sort_values("Anomaly_Score", ascending=False)
             .head(20)[["Company", "Year", "Industry", "Anomaly_Score",
                         "Accruals_Ratio", "Leverage_Change_YoY"]]
             .reset_index(drop=True))
    top20.index += 1

    rows = []
    for _, row in top20.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(str(int(row.name)), className="td-dim", style={"padding": "12px 10px", "fontSize": "12px"}),
                    html.Td(row["Company"], className="td-company", style={"padding": "12px 10px", "fontWeight": "600"}),
                    html.Td(str(int(row["Year"])) if pd.notna(row["Year"]) else "—", className="td-dim", style={"padding": "12px 10px"}),
                    html.Td(row["Industry"] if pd.notna(row["Industry"]) else "—", className="td-dim", style={"padding": "12px 10px", "fontSize": "12px"}),
                    html.Td(
                        html.Span(f"{row['Anomaly_Score']:.3f}",
                                  style={"background": "rgba(255,77,77,0.18)" if cfg["is_dark"] else "rgba(229,57,38,0.12)",
                                         "color": cfg["accent_red"], "padding": "4px 10px",
                                         "borderRadius": "4px", "fontWeight": "700", "fontSize": "12px"}),
                        style={"padding": "12px 10px"},
                    ),
                    html.Td(f"{row['Accruals_Ratio']:.3f}" if pd.notna(row["Accruals_Ratio"]) else "—", className="td-dim", style={"padding": "12px 10px"}),
                    html.Td(f"{row['Leverage_Change_YoY']:.3f}" if pd.notna(row["Leverage_Change_YoY"]) else "—", className="td-dim", style={"padding": "12px 10px"}),
                ],
                style={"borderBottom": f"1px solid {cfg['border']}"},
            )
        )

    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(h, className="td-dim", style={"fontSize": "11px",
                                          "textTransform": "uppercase", "letterSpacing": "0.5px",
                                          "padding": "10px", "borderBottom": f"1px solid {cfg['border']}"})
                        for h in ["#", "Company", "Year", "Industry", "Score", "Accruals Ratio", "Leverage Δ"]
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
    return fig, table


# ─────────────────────────────────────────────
# 10. COMPANY DEEP DIVE CALLBACKS
# ─────────────────────────────────────────────

@app.callback(
    Output("company-badges", "children"),
    Output("company-kpis", "children"),
    Output("chart-revenue", "figure"),
    Output("chart-cashflow", "figure"),
    Output("chart-debt", "figure"),
    Output("chart-anomaly-timeline", "figure"),
    Input("company-select", "value"),
    Input("theme-store", "data"),
)
def update_company(company, theme):
    cfg = get_theme_config(theme)
    if not company:
        empty = go.Figure()
        empty.update_layout(**cfg["layout"])
        return [], [], empty, empty, empty, empty

    cdf   = merged_df[merged_df["Company"] == company].sort_values("Year")
    adf   = anomaly_df[anomaly_df["Company"] == company].sort_values("Year")
    years = cdf["Year"].astype(str).tolist()

    anomaly_years = adf[adf["Is_Anomaly"] == 1]["Year"].tolist()

    shapes = []
    for y in anomaly_years:
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=str(int(y) - 0.4), x1=str(int(y) + 0.4),
            y0=0, y1=1,
            fillcolor="rgba(255,77,77,0.15)" if cfg["is_dark"] else "rgba(229,57,38,0.08)",
            line_width=0,
            layer="below",
        ))

    # Badges & KPIs
    industry = adf["Industry"].dropna().iloc[0] if not adf["Industry"].dropna().empty else "Unknown"
    badges = [
        html.Span(industry, style={"background": cfg["bg_page"], "color": cfg["text_main"],
                                    "padding": "6px 14px", "borderRadius": "20px",
                                    "fontSize": "12px", "fontWeight": "600",
                                    "border": f"1px solid {cfg['border']}"}),
    ]
    if anomaly_years:
        badges.append(
            html.Span(f"⚠️ {len(anomaly_years)} Anomalous Year(s)",
                      style={"background": "rgba(255,77,77,0.15)" if cfg["is_dark"] else "rgba(229,57,38,0.1)",
                             "color": cfg["accent_red"], "padding": "6px 14px",
                             "borderRadius": "20px", "fontWeight": "700", "fontSize": "12px",
                             "border": f"1px solid {cfg['accent_red']}"}),
        )

    latest = cdf.iloc[-1] if not cdf.empty else None

    if latest is not None:
        sales_val   = f"₹{latest['Sales']:,.0f} Cr" if pd.notna(latest.get("Sales")) else "—"
        profit_val  = f"₹{latest['Net profit']:,.0f} Cr" if pd.notna(latest.get("Net profit")) else "—"
        debt_val    = f"₹{latest['Borrowings']:,.0f} Cr" if pd.notna(latest.get("Borrowings")) else "—"
        max_score   = adf["Anomaly_Score"].max() if not adf.empty else 0
        score_color = cfg["accent_red"] if max_score > 0.2 else cfg["accent_teal"]
    else:
        sales_val = profit_val = debt_val = "—"
        max_score = 0
        score_color = cfg["accent_teal"]

    kpis = [
        stat_card("Latest Revenue", None, "📈", cfg["accent_blue"], "Top-line revenue reported in recent fiscal year.", "tt-mini-rev", theme_cfg=cfg),
        stat_card("Latest Net Profit", None, "💰",
                  cfg["accent_teal"] if latest is not None and pd.notna(latest.get("Net profit"))
                          and latest.get("Net profit", 0) > 0 else cfg["accent_red"],
                  "Bottom-line profit reported in recent fiscal year.", "tt-mini-prof", theme_cfg=cfg),
        stat_card("Latest Debt", None, "🏦", cfg["accent_red"], "Total borrowings in recent fiscal year.", "tt-mini-debt", theme_cfg=cfg),
        stat_card("Peak Anomaly Score", None, "⚡", score_color, "Maximum Isolation Forest score recorded across all 10 years.", "tt-mini-peak", theme_cfg=cfg),
    ]
    kpis[0].children[2].children = sales_val
    kpis[1].children[2].children = profit_val
    kpis[2].children[2].children = debt_val
    kpis[3].children[2].children = f"{max_score:.3f}"

    common = dict(cfg["layout"])
    common.update({"height": 300, "shapes": shapes})

    # Revenue chart
    fig_rev = go.Figure()
    if "Sales" in cdf.columns:
        fig_rev.add_trace(go.Bar(x=years, y=cdf["Sales"], name="Revenue", marker_color=cfg["accent_blue"]))
    if "Net profit" in cdf.columns:
        fig_rev.add_trace(go.Scatter(x=years, y=cdf["Net profit"], name="Net Profit", mode="lines+markers",
                                      line=dict(color=cfg["accent_red"], width=2.5),
                                      marker=dict(size=7, color=cfg["accent_red"])))
    fig_rev.update_layout(**common)

    # Cashflow chart
    fig_cf = go.Figure()
    if "Cash from Operating Activity" in cdf.columns:
        fig_cf.add_trace(go.Scatter(x=years, y=cdf["Cash from Operating Activity"], name="Operating Cash Flow",
                                     mode="lines+markers", line=dict(color=cfg["accent_teal"], width=2.5),
                                     marker=dict(size=7, color=cfg["accent_teal"]),
                                     fill="tozeroy", fillcolor="rgba(0,245,212,0.1)" if cfg["is_dark"] else "rgba(42,157,143,0.08)"))
    if "Net profit" in cdf.columns:
        fig_cf.add_trace(go.Scatter(x=years, y=cdf["Net profit"], name="Net Profit", mode="lines+markers",
                                     line=dict(color=cfg["accent_red"], width=2, dash="dot"),
                                     marker=dict(size=7, color=cfg["accent_red"])))
    fig_cf.update_layout(**common)

    # Debt chart
    fig_debt = go.Figure()
    if "Borrowings" in cdf.columns:
        fig_debt.add_trace(go.Bar(
            x=years, y=cdf["Borrowings"], name="Borrowings",
            marker=dict(
                color=cdf["Borrowings"],
                colorscale=[[0, "rgba(58,134,255,0.7)" if cfg["is_dark"] else "rgba(29,53,87,0.7)"], [1, cfg["accent_red"]]],
                showscale=False,
            ),
        ))
    fig_debt.update_layout(**common)

    # Anomaly timeline chart
    fig_anom = go.Figure()
    fig_anom.add_hline(y=0.2, line_dash="dot", line_color=cfg["accent_red"],
                       annotation_text="Anomaly Threshold",
                       annotation_font_color=cfg["accent_red"],
                       annotation_position="bottom right")
    fig_anom.add_trace(go.Scatter(
        x=adf["Year"].astype(str), y=adf["Anomaly_Score"],
        name="Anomaly Score", mode="lines+markers",
        line=dict(color=cfg["accent_blue"], width=2.5),
        marker=dict(
            size=np.maximum(0, adf["Anomaly_Score"].fillna(0).clip(0, None) * 30 + 6),
            color=adf["Is_Anomaly"].map({1: cfg["accent_red"], 0: cfg["accent_teal"]}),
            line=dict(width=1, color="rgba(255,255,255,0.9)"),
        ),
        fill="tozeroy", fillcolor="rgba(255,77,77,0.12)" if cfg["is_dark"] else "rgba(229,57,38,0.06)",
    ))
    common_no_shapes = {k: v for k, v in common.items() if k != "shapes"}
    fig_anom.update_layout(**common_no_shapes)

    return badges, kpis, fig_rev, fig_cf, fig_debt, fig_anom


# ─────────────────────────────────────────────
# 11. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
