# bhe_digital_twin_dashboard_vscode.py
# Run in VS Code terminal with:
#   streamlit run bhe_digital_twin_dashboard_vscode.py

from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# DASHBOARD BRANDING / LOGOS
# Keep the four PNG logo files in the same folder as this script.
# ------------------------------------------------------------
ASSET_DIR = Path(__file__).resolve().parent
SITE_LOGO_PATH = ASSET_DIR / 'Site Logo.png'
FOOTER1_PATH = ASSET_DIR / 'footer1.png'
FOOTER2_PATH = ASSET_DIR / 'footer2.png'
RIGHT_LOGO_PATH = ASSET_DIR / 'right logo.png'

st.set_page_config(page_title='Oulu BHE Digital Twin Demonstrator', page_icon='🌍', layout='wide')

# A little spacing/alignment for the branded header and footer.
st.markdown(
    '''
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }
    .bhe-title {
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1.15;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
    }
    .bhe-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 0.6rem;
    }
    .dev-badge {
        display: inline-block;
        margin-top: 0.25rem;
        margin-bottom: 0.35rem;
        padding: 0.22rem 0.60rem;
        border-radius: 999px;
        background: #fff3cd;
        color: #7a5600;
        border: 1px solid #f2d675;
        font-size: 0.82rem;
        font-weight: 650;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Header shown above every dashboard tab.
header_left, header_center, header_right = st.columns([1.0, 2.7, 1.35], vertical_alignment='center')

with header_left:
    if SITE_LOGO_PATH.exists():
        st.image(str(SITE_LOGO_PATH), width=180)
    else:
        st.warning('Site Logo.png not found')

with header_center:
    st.markdown(
        '<div class="bhe-title">Oulu Borehole Heat Exchanger Digital Twin Demonstrator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="dev-badge">Under development</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="bhe-subtitle">Research demonstrator for BHE state replay, short-term forecasting and simple what-if analysis.</div>',
        unsafe_allow_html=True,
    )

with header_right:
    if RIGHT_LOGO_PATH.exists():
        st.image(str(RIGHT_LOGO_PATH), use_container_width=True)
    else:
        st.warning('right logo.png not found')

st.divider()

# ------------------------------------------------------------
# DEFAULT FILE LOCATIONS — EDIT THESE IF NEEDED
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FOOTER2 = BASE_DIR / "footer2.png"
FOOTER1 = BASE_DIR / "footer1.png"
RIGHT_LOGO = BASE_DIR / "right logo.png"
SITE_LOGO = BASE_DIR / "Site Logo.png"
DEFAULT_DATA_PATH = BASE_DIR / "Oulu_all_sets.xlsx"
DEFAULT_MODEL_PATH = BASE_DIR / "Oulu_BHE_Digital_Twin_model.pkl"
ARX_FFNN_FIGURE_PATH = BASE_DIR / "arx_ffnn_structure_15_3_1.png"

# Minimum vertical temperature range shown in plots.
# This prevents very small temperature changes from looking visually exaggerated.
MIN_Y_AXIS_SPAN_DEGC = 1.0


def find_latest_model(folder: Path):
    candidates = list(folder.rglob('*_BHE_model.pkl'))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


@st.cache_resource
def load_model_package(path_str):
    return joblib.load(path_str)


@st.cache_data
def load_excel(path_str):
    return pd.read_excel(path_str)


def detect_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def prepare_data(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    scenario_col = detect_column(df, ['scenario', 'scenario id', 'scenario_id'])
    time_col = detect_column(df, ['time', 'time [min]', 'time (min)', 'time_min'])
    tout_col = detect_column(df, ['bhe outlet temperature', 'outlet temperature', 't_out', 'tout'])
    tin_col = detect_column(df, ['bhe inlet temperature', 'inlet temperature', 't_in', 'tin'])
    q_col = detect_column(df, ['flow rate', 'flow rate [l/s]', 'flow rate (l/s)', 'flow_rate', 'flowrate', 'q'])

    required = {'Scenario': scenario_col, 'Time': time_col, 'T_out': tout_col, 'T_in': tin_col, 'Q': q_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError('Could not identify columns: ' + ', '.join(missing))

    df = df.rename(columns={old: new for new, old in required.items()})
    df = df[['Scenario', 'Time', 'T_out', 'T_in', 'Q']].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna().copy()
    df['Scenario'] = df['Scenario'].astype(int)
    df = df.sort_values(['Scenario', 'Time']).reset_index(drop=True)
    df['Time_step'] = df.groupby('Scenario').cumcount()
    return df


def get_max_lag(input_order, prefix):
    vals = []
    for feature in input_order:
        if feature.startswith(prefix):
            try:
                vals.append(int(feature.split('_')[-1]))
            except ValueError:
                pass
    return max(vals) if vals else 0


def build_input_vector(run, index, recursive_tout, input_order, q_override=None, tin_override=None):
    values = []
    for feature in input_order:
        if feature == 'Q':
            value = q_override[index] if q_override is not None else run.loc[index, 'Q']
        elif feature.startswith('Q_lag_'):
            lag = int(feature.split('_')[-1])
            src = index - lag
            value = q_override[src] if q_override is not None else run.loc[src, 'Q']
        elif feature == 'T_in':
            value = tin_override[index] if tin_override is not None else run.loc[index, 'T_in']
        elif feature.startswith('T_in_lag_'):
            lag = int(feature.split('_')[-1])
            src = index - lag
            value = tin_override[src] if tin_override is not None else run.loc[src, 'T_in']
        elif feature.startswith('T_out_lag_'):
            lag = int(feature.split('_')[-1])
            value = recursive_tout[index - lag]
        elif feature == 'Time_step':
            value = run.loc[index, 'Time_step']
        else:
            raise ValueError(f'Unsupported model input: {feature}')
        values.append(float(value))
    return np.asarray(values, dtype=float).reshape(1, -1)


def predict_delta(package, x):
    model = package['model']
    scaler_x = package['scaler_X']
    scaler_y = package['scaler_Y']
    x_scaled = scaler_x.transform(x)
    delta_scaled = model.predict(x_scaled).reshape(-1, 1)
    return float(scaler_y.inverse_transform(delta_scaled).ravel()[0])


def recursive_forecast(run, start_index, forecast_steps, package, q_override=None, tin_override=None):
    input_order = list(package['input_order'])
    n_tout_lags = get_max_lag(input_order, 'T_out_lag_')
    final_index = min(start_index + forecast_steps - 1, len(run) - 1)

    measured = run['T_out'].to_numpy(dtype=float)
    recursive = np.full(len(run), np.nan, dtype=float)
    recursive[start_index-n_tout_lags:start_index] = measured[start_index-n_tout_lags:start_index]

    for i in range(start_index, final_index + 1):
        x = build_input_vector(run, i, recursive, input_order, q_override, tin_override)
        delta = predict_delta(package, x)
        recursive[i] = recursive[i-1] + delta

    idx = np.arange(start_index, final_index + 1)
    return pd.DataFrame({
        'index': idx,
        'Time': run.loc[idx, 'Time'].to_numpy(dtype=float),
        'Measured': run.loc[idx, 'T_out'].to_numpy(dtype=float),
        'Predicted': recursive[idx],
    })


def calc_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    return rmse, mae, r2

def summarize_model_inputs(input_order):
    q_lags = sorted(int(x.split('_')[-1]) for x in input_order if x.startswith('Q_lag_'))
    tin_lags = sorted(int(x.split('_')[-1]) for x in input_order if x.startswith('T_in_lag_'))
    tout_lags = sorted(int(x.split('_')[-1]) for x in input_order if x.startswith('T_out_lag_'))

    groups = []
    if 'Q' in input_order:
        groups.append(
            f"Flow rate Q\nQ(t) ... Q(t-{max(q_lags)})\n{1 + len(q_lags)} inputs"
            if q_lags else "Flow rate Q\nQ(t)\n1 input"
        )
    if 'T_in' in input_order:
        groups.append(
            f"Inlet temperature Tin\nTin(t) ... Tin(t-{max(tin_lags)})\n{1 + len(tin_lags)} inputs"
            if tin_lags else "Inlet temperature Tin\nTin(t)\n1 input"
        )
    if tout_lags:
        groups.append(
            f"Previous outlet temperature\nTout(t-1) ... Tout(t-{max(tout_lags)})\n{len(tout_lags)} inputs"
        )
    if 'Time_step' in input_order:
        groups.append("Scenario time step\nTime_step(t)\n1 input")
    return groups


def make_arx_ffnn_figure(package):
    model = package['model']
    local_inputs = list(package['input_order'])
    input_dim = len(local_inputs)

    architecture = package.get('architecture', {})
    hidden = architecture.get('hidden_layer_sizes', getattr(model, 'hidden_layer_sizes', ()))
    if isinstance(hidden, int):
        hidden = (hidden,)
    hidden = tuple(hidden)

    activation = architecture.get('activation', getattr(model, 'activation', 'unknown'))
    output_dim = int(architecture.get('output_dim', 1))
    groups = summarize_model_inputs(local_inputs)

    fig = go.Figure()
    x_inputs, x_hidden, x_output, x_update = 0.8, 2.6, 4.4, 6.2
    ys = [2.5] if len(groups) == 1 else np.linspace(4.0, 1.0, len(groups))

    for y, label in zip(ys, groups):
        fig.add_shape(
            type='rect',
            x0=x_inputs-0.55, x1=x_inputs+0.55,
            y0=y-0.38, y1=y+0.38,
            line=dict(width=1.5),
            fillcolor='rgba(230,240,255,0.65)'
        )
        fig.add_annotation(
            x=x_inputs, y=float(y),
            text=label.replace('\\n', '<br>'),
            showarrow=False, align='center', font=dict(size=12)
        )

    hidden_text = '<br>'.join(
        f"Hidden layer {i+1}: {n} neuron{'s' if n != 1 else ''}"
        for i, n in enumerate(hidden)
    ) or 'No hidden layer'

    fig.add_shape(
        type='rect',
        x0=x_hidden-0.68, x1=x_hidden+0.68,
        y0=1.75, y1=3.25,
        line=dict(width=1.8),
        fillcolor='rgba(255,241,204,0.75)'
    )
    fig.add_annotation(
        x=x_hidden, y=2.5,
        text=f"<b>FFNN</b><br>{hidden_text}<br>Activation: {activation}",
        showarrow=False, align='center', font=dict(size=13)
    )

    fig.add_shape(
        type='rect',
        x0=x_output-0.62, x1=x_output+0.62,
        y0=2.05, y1=2.95,
        line=dict(width=1.8),
        fillcolor='rgba(220,252,231,0.75)'
    )
    fig.add_annotation(
        x=x_output, y=2.5,
        text=f"<b>Network output</b><br>ΔTout(t)<br>{output_dim} output",
        showarrow=False, align='center', font=dict(size=13)
    )

    fig.add_shape(
        type='rect',
        x0=x_update-0.78, x1=x_update+0.78,
        y0=1.90, y1=3.10,
        line=dict(width=1.8),
        fillcolor='rgba(243,232,255,0.75)'
    )
    fig.add_annotation(
        x=x_update, y=2.5,
        text="<b>Recursive update</b><br>Tout(t) = Tout(t−1)<br>+ ΔTout(t)",
        showarrow=False, align='center', font=dict(size=13)
    )

    for y in ys:
        fig.add_annotation(
            x=x_hidden-0.72, y=2.5,
            ax=x_inputs+0.58, ay=float(y),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowwidth=1.4, text=''
        )

    fig.add_annotation(
        x=x_output-0.66, y=2.5,
        ax=x_hidden+0.70, ay=2.5,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowwidth=1.6, text=''
    )
    fig.add_annotation(
        x=x_update-0.82, y=2.5,
        ax=x_output+0.66, ay=2.5,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowwidth=1.6, text=''
    )

    architecture_label = (
        f"{input_dim}-" + "-".join(str(n) for n in hidden) + f"-{output_dim}"
        if hidden else f"{input_dim}-{output_dim}"
    )

    fig.update_layout(
        title=f"ARX-FFNN structure used in the demonstrator ({architecture_label})",
        xaxis=dict(visible=False, range=[0, 7.1], fixedrange=True),
        yaxis=dict(visible=False, range=[0.35, 4.65], fixedrange=True),
        height=500,
        margin=dict(l=20, r=20, t=70, b=25),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    return fig, architecture_label, activation



# ------------------------------------------------------------
# SIDEBAR: MODEL + DATA PATHS
# ------------------------------------------------------------
st.sidebar.header("Demonstrator controls")

model_path = DEFAULT_MODEL_PATH
data_path = DEFAULT_DATA_PATH

if not model_path.exists():
    st.error('Model file not found. Make sure Oulu_BHE_Digital_Twin_model.pkl is in the same folder as this dashboard.')
    st.stop()
if not data_path.exists():
    st.error('Dataset not found. Make sure Oulu_all_sets.xlsx is in the same folder as this dashboard.')
    st.stop()

try:
    package = load_model_package(str(model_path))
    data = prepare_data(load_excel(str(data_path)))
except Exception as exc:
    st.exception(exc)
    st.stop()

input_order = list(package['input_order'])
n_tout_lags = get_max_lag(input_order, 'T_out_lag_')

# ------------------------------------------------------------
# STAKEHOLDER STORY
# ------------------------------------------------------------
a, b, c = st.columns(3)
with a:
    st.info('🌡️ **Physical BHE**\n\nSensors measure flow and temperatures.')
with b:
    st.info('🧠 **Digital Twin Demonstrator**\n\nThe ARX-FFNN estimates and predicts the BHE thermal response using recorded operational data.')
with c:
    st.info('⚡ **Energy Management**\n\nThe forecast can support better operating decisions.')

scenario_ids = sorted(data['Scenario'].unique())
scenario = st.sidebar.selectbox(
    'Recorded scenario',
    scenario_ids,
    help=(
        'A scenario is one continuous recorded BHE operating period/test case. '
        'Each scenario contains its own chronological sequence of flow rate, '
        'inlet temperature and measured outlet temperature.'
    )
)
st.sidebar.caption(
    '**Scenario:** one continuous recorded operating period/test case from the '
    'Oulu BHE dataset. Selecting a scenario chooses which recorded history is replayed.'
)
run = data[data['Scenario'] == scenario].sort_values('Time').reset_index(drop=True)

times = run['Time'].to_numpy(dtype=float)
dt_minutes = float(np.median(np.diff(times)))

default_index = min(max(n_tout_lags + 1, int(round(24*60/dt_minutes))), len(run)-2)
current_index = st.sidebar.slider('Current position in scenario', n_tout_lags+1, len(run)-2, default_index)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(['📈 Digital twin demonstrator', '🎛️ What-if', 'ℹ️ Explanation'])

with tab1:
    current_q = float(run.loc[current_index-1, 'Q'])
    current_tin = float(run.loc[current_index-1, 'T_in'])
    current_tout = float(run.loc[current_index-1, 'T_out'])

    c1, c2, c3 = st.columns(3)
    c1.metric('Flow rate', f'{current_q:.2f} L/s')
    c2.metric('BHE inlet temperature', f'{current_tin:.2f} °C')
    c3.metric('BHE outlet temperature', f'{current_tout:.2f} °C')

    horizon_h = st.select_slider('Forecast horizon', options=[1,2,3,4,5,6,12,24], value=6)
    steps = min(int(round(horizon_h*60/dt_minutes)), len(run)-current_index)
    forecast = recursive_forecast(run, current_index, steps, package)

    t0 = float(run.loc[0, 'Time'])
    history_start = max(0, current_index-int(round(12*60/dt_minutes)))
    history_x = (run.loc[history_start:current_index-1, 'Time'].to_numpy()-t0)/60
    forecast_x = (forecast['Time'].to_numpy()-t0)/60

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_x, y=run.loc[history_start:current_index-1, 'T_out'], mode='lines', name='Measured history', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=forecast_x, y=forecast['Measured'], mode='lines', name='Measured future (validation only)', line=dict(width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_x, y=forecast['Predicted'], mode='lines', name='Demonstrator forecast', line=dict(width=3)))
    # Keep the visible plot domain tied to the data actually being shown.
    # Plotly zooming changes only the camera/view; it does not create additional
    # forecast data. For this stakeholder dashboard we therefore lock the axes
    # to the selected history + forecast window.
    digital_x_all = np.concatenate([history_x, forecast_x])
    digital_y_all = np.concatenate([
        run.loc[history_start:current_index-1, 'T_out'].to_numpy(dtype=float),
        forecast['Measured'].to_numpy(dtype=float),
        forecast['Predicted'].to_numpy(dtype=float),
    ])

    digital_x_span = max(float(np.ptp(digital_x_all)), 1e-6)
    digital_y_span = max(float(np.ptp(digital_y_all)), MIN_Y_AXIS_SPAN_DEGC)

    digital_x_pad = 0.03 * digital_x_span
    digital_y_pad = 0.10 * digital_y_span

    fig.update_layout(
        title=f'Scenario {scenario}: BHE outlet-temperature forecast',
        xaxis_title='Elapsed time [h]',
        yaxis_title='BHE outlet temperature [°C]',
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h'),
        xaxis=dict(
            range=[
                float(np.min(digital_x_all) - digital_x_pad),
                float(np.max(digital_x_all) + digital_x_pad),
            ],
            fixedrange=True,
        ),
        yaxis=dict(
            range=[
                float(np.min(digital_y_all) - digital_y_pad),
                float(np.max(digital_y_all) + digital_y_pad),
            ],
            fixedrange=True,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'
            ],
        },
    )

    rmse, mae, _ = calc_metrics(forecast['Measured'], forecast['Predicted'])

    max_abs_error = float(
        np.max(
            np.abs(
                forecast['Measured'].to_numpy(dtype=float)
                - forecast['Predicted'].to_numpy(dtype=float)
            )
        )
    )

    m1, m2, m3 = st.columns(3)
    m1.metric('RMSE over shown forecast', f'{rmse:.3f} °C')
    m2.metric('MAE over shown forecast', f'{mae:.3f} °C')
    m3.metric('Maximum error over shown forecast', f'{max_abs_error:.3f} °C')

    with st.expander("What do RMSE and MAE mean?"):
        st.markdown(
            """
**RMSE — Root Mean Squared Error**

RMSE measures the typical difference between the measured and predicted
BHE outlet temperature, while giving more weight to larger errors.
"""
        )

        st.latex(
            r"""
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(
T_{out,i}^{measured}
-
T_{out,i}^{predicted}
\right)^2
}
"""
        )

        st.markdown(
            """
A **smaller RMSE is better**.  
For example, an RMSE of **0.20 °C** means that the model predictions are
typically within a few tenths of a degree of the measured values.

---

**MAE — Mean Absolute Error**

MAE is the average absolute difference between the measured and predicted
BHE outlet temperature.
"""
        )

        st.latex(
            r"""
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
\left|
T_{out,i}^{measured}
-
T_{out,i}^{predicted}
\right|
"""
        )

        st.markdown(
            """
A **smaller MAE is better**.  
MAE is easy to interpret because it gives the average prediction error
directly in **degrees Celsius**.

---

**Maximum error**

This is the largest absolute temperature difference anywhere in the
displayed forecast period:
"""
        )

        st.latex(
            r"""
Max\ Error =
\max_i
\left|
T_{out,i}^{measured}
-
T_{out,i}^{predicted}
\right|
"""
        )

        st.markdown(
            "It shows the **worst single prediction error** in the selected forecast window."
        )

    st.caption(
        'The measured future is shown only for replay/validation. '
        'In real operation, only the digital-twin forecast would be available. '
        'R² is intentionally not shown for this short local forecast window '
        'because it can be misleading when the measured outlet temperature '
        'changes only slightly.'
    )

with tab2:
    st.subheader('What happens if operation changes?')
    st.write('Select the future flow rate and BHE inlet temperature and the demonstrator predicts the corresponding outlet-temperature trajectory.')

    c1, c2, c3 = st.columns(3)
    with c1:
        whatif_h = st.select_slider('Prediction horizon [h]', options=[1,2,3,4,5,6], value=6)
    current_q_whatif = float(run.loc[current_index - 1, 'Q'])
    current_tin_whatif = float(run.loc[current_index - 1, 'T_in'])

    with c2:
        q_whatif = st.slider(
            'Future flow rate [L/s]',
            min_value=0.0,
            max_value=2.0,
            value=float(np.clip(current_q_whatif, 0.0, 2.0)),
            step=0.05
        )
    with c3:
        tin_whatif = st.slider(
            'Future inlet temperature [°C]',
            min_value=-5.0,
            max_value=5.0,
            value=float(np.clip(current_tin_whatif, -5.0, 5.0)),
            step=0.1
        )

    st.info(
        "The selected flow rate and inlet temperature are assumed to remain "
        "constant over the entire prediction horizon."
    )

    steps = min(int(round(whatif_h*60/dt_minutes)), len(run)-current_index)
    q_override = run['Q'].to_numpy(dtype=float).copy()
    tin_override = run['T_in'].to_numpy(dtype=float).copy()

    # Apply selected absolute operating values from the current
    # forecast point onward.
    q_override[current_index:] = q_whatif
    tin_override[current_index:] = tin_whatif

    changed = recursive_forecast(run, current_index, steps, package, q_override, tin_override)
    x = (changed['Time'].to_numpy()-t0)/60

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=x,
            y=changed['Predicted'],
            mode='lines',
            name='Predicted outlet temperature',
            line=dict(width=3),
        )
    )

    # The x-domain is the selected what-if forecast horizon.
    # The y-axis is fitted only to the selected operating-condition forecast.
    whatif_y_all = changed['Predicted'].to_numpy(dtype=float)

    whatif_x_span = max(float(np.ptp(x)), 1e-6)
    whatif_y_span = max(float(np.ptp(whatif_y_all)), MIN_Y_AXIS_SPAN_DEGC)

    whatif_x_pad = 0.03 * whatif_x_span
    whatif_y_pad = 0.10 * whatif_y_span

    fig2.update_layout(
        title='Digital twin demonstrator prediction for selected operating conditions',
        xaxis_title='Elapsed time [h]',
        yaxis_title='Predicted BHE outlet temperature [°C]',
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h'),
        xaxis=dict(
            range=[
                float(np.min(x) - whatif_x_pad),
                float(np.max(x) + whatif_x_pad),
            ],
            fixedrange=True,
        ),
        yaxis=dict(
            range=[
                float(np.min(whatif_y_all) - whatif_y_pad),
                float(np.max(whatif_y_all) + whatif_y_pad),
            ],
            fixedrange=True,
        ),
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'
            ],
        },
    )

    changed_end = float(changed['Predicted'].iloc[-1])
    changed_min = float(changed['Predicted'].min())
    changed_max = float(changed['Predicted'].max())

    d1, d2, d3 = st.columns(3)
    d1.metric('Outlet temperature at horizon', f'{changed_end:.2f} °C')
    d2.metric('Minimum predicted outlet', f'{changed_min:.2f} °C')
    d3.metric('Maximum predicted outlet', f'{changed_max:.2f} °C')

    st.warning('Keep what-if changes within or close to the operating conditions represented in the training data.')

with tab3:
    st.subheader('About this digital twin demonstrator')

    st.info(
        'This application is called a **digital twin demonstrator** because it '
        'currently replays previously collected operational data rather than '
        'receiving continuously updated live measurements from the physical BHE. '
        'It demonstrates how a future online digital twin could operate once '
        'a live data connection is available.'
    )

    st.markdown('''
### What does “scenario” mean?

A **scenario** is one continuous recorded operating period/test case from the
Oulu BHE dataset. Each scenario contains its own chronological sequence of
flow rate, BHE inlet temperature, measured BHE outlet temperature and elapsed time.

Selecting a scenario therefore chooses which recorded operating period is
replayed and used to initialize the demonstrator.
''')

    scenario_summary = (
        data.groupby('Scenario')
        .agg(
            Points=('Time', 'size'),
            Start_time_min=('Time', 'min'),
            End_time_min=('Time', 'max'),
            Q_min=('Q', 'min'),
            Q_max=('Q', 'max'),
            Tin_min=('T_in', 'min'),
            Tin_max=('T_in', 'max')
        )
        .reset_index()
    )
    # Round scenario duration UP to the next whole hour for stakeholder display.
    # Example: 71.2 h -> 72 h
    scenario_summary['Duration [h]'] = np.ceil(
        (
            scenario_summary['End_time_min']
            - scenario_summary['Start_time_min']
        ) / 60.0
    ).astype(int)

    with st.expander('Show recorded scenario summary'):
        st.dataframe(
            scenario_summary[
                ['Scenario', 'Duration [h]', 'Points', 'Q_min', 'Q_max', 'Tin_min', 'Tin_max']
            ].rename(
                columns={
                    'Q_min': 'Flow min [L/s]',
                    'Q_max': 'Flow max [L/s]',
                    'Tin_min': 'Tin min [°C]',
                    'Tin_max': 'Tin max [°C]'
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    st.subheader('ARX-FFNN model used for the BHE')

    st.markdown('''
The BHE surrogate is an **ARX-FFNN**: an **autoregressive model with exogenous
inputs implemented using a feed-forward neural network**.

- **Autoregressive (AR):** previous BHE outlet temperatures are used as inputs,
  so the model carries information about the recent thermal state.
- **Exogenous inputs (X):** the model also uses BHE flow rate and inlet
  temperature, including their recent history.
- **FFNN:** these inputs are passed through a compact feed-forward neural network
  to represent the nonlinear relation between operation and BHE thermal response.
- **Delta formulation:** the network predicts the temperature change
  ΔTout(t), rather than the absolute outlet temperature directly.
''')

    if ARX_FFNN_FIGURE_PATH.exists():
        st.image(
            str(ARX_FFNN_FIGURE_PATH),
            caption='Illustrative ARX-FFNN structure used in the demonstrator (15-3-1).',
            use_container_width=True,
        )
    else:
        st.warning(
            'The ARX-FFNN figure file was not found. Add '
            '"arx_ffnn_structure_15_3_1.png" to the same folder as this dashboard.'
        )

    st.caption(
        'This is a static explanatory figure prepared for stakeholder communication. '
        'It illustrates the selected 15-3-1 ARX-FFNN structure: input features → FFNN → '
        'predicted temperature change ΔTout(t) → recursive update of Tout(t).'
    )

    st.markdown('''
### How the forecast is generated

1. The selected scenario provides the recent measured operating history.
2. The ARX-FFNN receives the current/recent flow rate, inlet temperature,
   previous outlet temperatures and scenario time-step input.
3. The network predicts the next outlet-temperature change.
''')

    st.latex(
        r'\hat{T}_{out}(t)=\hat{T}_{out}(t-1)+\widehat{\Delta T}_{out}(t)'
    )

    st.markdown('''
4. For a multi-step forecast, each predicted outlet temperature is fed back
   into the next prediction. This is the **recursive prediction**.
5. In What-if mode, the selected flow rate and inlet temperature are assumed
   constant over the chosen prediction horizon.
''')

    st.success(
        'Stakeholder message: this demonstrator shows how a compact machine-learning '
        'surrogate can reproduce and forecast the thermal behaviour of the Oulu '
        'borehole heat exchanger using recorded operating data.'
    )

# ------------------------------------------------------------
# FOOTER BRANDING — visible at the bottom of every dashboard tab
# ------------------------------------------------------------
st.divider()
footer_left, footer_right = st.columns([1, 1], vertical_alignment='center')

with footer_left:
    if FOOTER1_PATH.exists():
        st.image(str(FOOTER1_PATH), use_container_width=True)
    else:
        st.warning('footer1.png not found')

with footer_right:
    if FOOTER2_PATH.exists():
        st.image(str(FOOTER2_PATH), use_container_width=True)
    else:
        st.warning('footer2.png not found')

st.markdown(
    '<div class="footer-note">Oulu BHE Digital Twin Demonstrator — research prototype under development</div>',
    unsafe_allow_html=True,
)
