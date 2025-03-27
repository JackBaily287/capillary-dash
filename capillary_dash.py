import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
import os

app = dash.Dash(__name__)
app.title = "Water vs Mercury Meniscus"

# Physical constants
g = 9.81       # gravity (m/s²)
r = 0.0005     # capillary radius (m)

# Fluid properties
fluids = {
    'Water':   {'rho': 1000,  'gamma': 0.0728, 'color': 'blue'},
    'Mercury': {'rho': 13560, 'gamma': 0.485,  'color': 'gray'}
}

# Convert capillary height (cm) to contact angle (degrees)
def height_to_theta(h_cm, rho, gamma):
    h = h_cm / 100  # convert to meters
    cos_theta = (h * rho * g * r) / (2 * gamma)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)
    return np.degrees(theta_rad)

# App layout
app.layout = html.Div([
    html.H2(
        "Water vs Mercury Meniscus – How the contact angle of the liquid with the tube changes with height",
        style={"textAlign": "center", "lineHeight": "1.4"}
    ),

    html.Div([
        html.Label("Capillary Rise/Depression Height (cm):"),
        dcc.Slider(
            id='height-slider-live',
            min=0,
            max=3.0,
            step=0.005,
            value=1.0,
            marks={i: f"{i} cm" for i in np.arange(0, 3.1, 0.5)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
        dcc.Slider(
            id='height-slider-release',
            min=0,
            max=3.0,
            step=0.005,
            value=1.0,
            style={'display': 'none'}
        )
    ], style={"width": "70%", "margin": "auto"}),

    html.Div([
        dcc.Graph(id='meniscus-graph', style={"width": "50%", "display": "inline-block"}),
        dcc.Graph(id='theta-vs-height-graph', style={"width": "48%", "display": "inline-block"})
    ])
])

# Sync release-slider with live-slider (only updates on release)
@app.callback(
    Output('height-slider-release', 'value'),
    Input('height-slider-live', 'value')
)
def sync_on_release(val):
    return val

# Update meniscus graph live
@app.callback(
    Output('meniscus-graph', 'figure'),
    Input('height-slider-live', 'value')
)
def update_meniscus_graph(h_cm):
    fig_meniscus = go.Figure()
    tube_width = 0.10
    tube_height = 2.5
    liquid_top_y = 1.5
    x = np.linspace(-1, 1, 200)
    tube_spacing = 0.15
    base_shift = 0.45

    for i, (name, props) in enumerate(fluids.items()):
        rho, gamma, color = props['rho'], props['gamma'], props['color']
        theta_deg = height_to_theta(h_cm, rho, gamma)
        theta_rad = np.radians(theta_deg)
        curvature = np.cos(theta_rad)
        sign = -1 if name == 'Water' else 1
        amplitude = 0.2 * curvature * sign

        tube_left = base_shift + i * (tube_width + tube_spacing)
        tube_right = tube_left + tube_width

        x_scaled = tube_left + (tube_width / 2) + (tube_width / 2) * x
        y_curve = liquid_top_y + amplitude * np.cos(np.pi * x / 2)

        fig_meniscus.add_shape(type="rect",
            x0=tube_left, x1=tube_right, y0=0, y1=tube_height,
            line=dict(color="black"))

        fig_meniscus.add_trace(go.Scatter(
            x=np.concatenate([x_scaled, x_scaled[::-1]]),
            y=np.concatenate([y_curve, np.zeros_like(x_scaled)]),
            fill='toself',
            fillcolor=color,
            line=dict(color=color),
            mode='lines',
            name=f"{name} θ = {theta_deg:.1f}°"
        ))

    fig_meniscus.update_layout(
        title=f"Capillary Height: {h_cm:.2f} cm",
        xaxis=dict(showticklabels=False, fixedrange=True),
        yaxis=dict(showticklabels=False, fixedrange=True),
        width=600, height=600,
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(x=1.05, y=1.05)
    )

    return fig_meniscus

# Update theta graph only on release
@app.callback(
    Output('theta-vs-height-graph', 'figure'),
    Input('height-slider-release', 'value')
)
def update_theta_graph(h_cm):
    h_vals = np.linspace(0, 3.0, 200)
    fig_theta = go.Figure()

    for name, props in fluids.items():
        theta_vals = [height_to_theta(h, props['rho'], props['gamma']) for h in h_vals]
        fig_theta.add_trace(go.Scatter(
            x=h_vals, y=theta_vals, mode='lines', name=name, line=dict(color=props['color'])
        ))

    fig_theta.add_trace(go.Scatter(
        x=[h_cm],
        y=[height_to_theta(h_cm, fluids['Water']['rho'], fluids['Water']['gamma'])],
        mode='markers+text', name='Water θ',
        marker=dict(color='blue', size=10),
        text=[f"θ = {height_to_theta(h_cm, fluids['Water']['rho'], fluids['Water']['gamma']):.1f}°"],
        textposition="top center"
    ))

    fig_theta.add_trace(go.Scatter(
        x=[h_cm],
        y=[height_to_theta(h_cm, fluids['Mercury']['rho'], fluids['Mercury']['gamma'])],
        mode='markers+text', name='Mercury θ',
        marker=dict(color='gray', size=10),
        text=[f"θ = {height_to_theta(h_cm, fluids['Mercury']['rho'], fluids['Mercury']['gamma']):.1f}°"],
        textposition="top center"
    ))

    fig_theta.update_layout(
        xaxis_title="Capillary Height (cm)",
        yaxis_title="Contact Angle θ (degrees)",
        height=600, width=600,
        margin=dict(l=20, r=20, t=80, b=40)
    )

    return fig_theta

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)

