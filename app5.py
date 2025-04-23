import dash
import pandas as pd
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# Load biometric data
df = pd.read_csv("biometric_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Mock height mapping
height_map = {
    "Alice Smith": 165,
    "Bob Johnson": 178,
    "Carlos Diaz": 172,
}
df["Height (cm)"] = df["Patient"].map(height_map)
df["Height (m)"] = df["Height (cm)"] / 100
df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2)

# Add mock hydration and sleep quality data (for heatmap)
df["Hydration (%)"] = np.clip(np.random.normal(60, 5, size=len(df)), 50, 70)
df["Sleep Quality"] = np.random.randint(1, 10, size=len(df))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Enhanced Biometric Dashboard"

# Layout
app.layout = dbc.Container([
    html.H2("Biometric Progress Tracker", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id="patient-dropdown",
                options=[{"label": p, "value": p} for p in df["Patient"].unique()],
                value=df["Patient"].unique()[0],
                clearable=False,
                className="mb-3"
            )
        ], width=6)
    ], justify="center"),

    dbc.Row([
        dbc.Col(html.Div(id="summary-table"), width=12)
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="radar-graph"), width=12)
    ], justify="center", className="mb-4"),

    dbc.Tabs([
        dbc.Tab(dcc.Graph(id="weight-graph"), label="Weight"),
        dbc.Tab(dcc.Graph(id="bodyfat-graph"), label="Body Fat %"),
        dbc.Tab(dcc.Graph(id="muscle-graph"), label="Muscle Mass %")
    ]),

    html.Br(),
    html.H4("Recovery & Readiness Heatmap", className="text-center my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="heatmap"), width=12)
    ])
])

@app.callback(
    [Output("weight-graph", "figure"),
     Output("radar-graph", "figure"),
     Output("bodyfat-graph", "figure"),
     Output("muscle-graph", "figure"),
     Output("summary-table", "children"),
     Output("heatmap", "figure")],
    Input("patient-dropdown", "value")
)
def update_figures(patient):
    dff = df[df["Patient"] == patient].sort_values("Date")
    latest = dff.iloc[-1]

    # Summary Table
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
        html.Tbody([
            html.Tr([html.Td("Height (cm)"), html.Td(f"{latest['Height (cm)']:.1f}")]),
            html.Tr([html.Td("Weight (kg)"), html.Td(f"{latest['Weight (kg)']:.1f}")]),
            html.Tr([html.Td("BMI"), html.Td(f"{latest['BMI']:.1f}")]),
            html.Tr([html.Td("Body Fat (%)"), html.Td(f"{latest['Body Fat (%)']:.1f}")]),
            html.Tr([html.Td("Muscle Mass (%)"), html.Td(f"{latest['Muscle Mass (%)']:.1f}")]),
        ])
    ], bordered=True, hover=True, responsive=True, striped=True)

    # Forecasting setup for weight
    dff["Days"] = (dff["Date"] - dff["Date"].min()).dt.days
    model = LinearRegression()
    model.fit(dff[["Days"]], dff["Weight (kg)"])
    future_days = np.arange(dff["Days"].max() + 7, dff["Days"].max() + 35, 7)
    future_dates = [dff["Date"].max() + timedelta(days=int(d - dff["Days"].max())) for d in future_days]
    future_weights = model.predict(future_days.reshape(-1, 1))

    # Graphs
    weight_fig = go.Figure()
    weight_fig.add_trace(go.Scatter(x=dff["Date"], y=dff["Weight (kg)"], mode="lines+markers", name="Actual"))
    weight_fig.add_trace(go.Scatter(x=future_dates, y=future_weights, mode="lines+markers", name="Forecast", line=dict(dash="dot")))
    weight_fig.update_layout(title=f"Weight Trend with Forecast - {patient}", template="plotly_white")

    bodyfat_fig = go.Figure()
    bodyfat_fig.add_trace(go.Scatter(x=dff["Date"], y=dff["Body Fat (%)"], mode="lines+markers", name="Body Fat %"))
    bodyfat_fig.update_layout(title=f"Body Fat % Trend - {patient}", template="plotly_white")

    muscle_fig = go.Figure()
    muscle_fig.add_trace(go.Scatter(x=dff["Date"], y=dff["Muscle Mass (%)"], mode="lines+markers", name="Muscle Mass %"))
    muscle_fig.update_layout(title=f"Muscle Mass % Trend - {patient}", template="plotly_white")

    # Heatmap
    heatmap_data = dff[["Date", "Hydration (%)", "Sleep Quality"]].set_index("Date")
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.T.values,
        x=heatmap_data.index,
        y=heatmap_data.T.index,
        colorscale="YlGnBu",
        colorbar=dict(title="Score")
    ))
    heatmap_fig.update_layout(title="Hydration & Sleep Quality", template="plotly_white")

    # Radar Chart
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[latest["Body Fat (%)"], latest["Muscle Mass (%)"], latest["BMI"], latest["Hydration (%)"], latest["Sleep Quality"]],
        theta=["Body Fat %", "Muscle Mass %", "BMI", "Hydration %", "Sleep Quality"],
        fill='toself',
        name='Current Profile'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12, color="gray")
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="black")
            )
        ),
        title={"text": "Current Body Composition Snapshot", "x": 0.5},
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=30),
        height=450
    )

    return weight_fig, radar_fig, bodyfat_fig, muscle_fig, table, heatmap_fig

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)