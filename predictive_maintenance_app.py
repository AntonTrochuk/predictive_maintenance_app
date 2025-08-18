import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ------------------------------
# Конфигурация страницы
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    page_icon="🔧"
)
st.title("🔧 Система предиктивного обслуживания: Live Dashboard")

# ------------------------------
# Константы и настройки
COMPONENTS = ["Насосы", "Подшипники", "Вентиляторы"]
WARNING_THRESHOLD = 0.7
CRITICAL_THRESHOLD = 0.95  # новый верхний порог
DAY_LENGTH = 24  # часы
SIMULATION_DAYS = 7  # дней для симуляции
np.random.seed(42)

COLORS = {
    "Насосы": "blue",
    "Подшипники": "orange",
    "Вентиляторы": "green"
}

# ------------------------------
# Функции обработки данных
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 0.001)

def compute_risk(vib, temp):
    risk_raw = 0.6*vib + 0.4*temp
    return 1 / (1 + np.exp(-5*(risk_raw - 0.5)))

def predict_days_to_maintenance(risk_arr):
    if len(risk_arr) == 0:
        return np.nan
    window_size = min(5, len(risk_arr))
    growth_rate = np.mean(np.diff(risk_arr[-window_size:])) if window_size > 1 else 0.01
    if growth_rate <= 0:
        return np.nan
    days = (CRITICAL_THRESHOLD - risk_arr[-1]) / growth_rate
    return max(0, int(np.ceil(days)))

def get_recommendations(risk, vib, temp):
    recs = []
    if vib[-1] > 0.6:
        recs.append("Проверить балансировку и крепления")
    if temp[-1] > 55:
        recs.append("Проверить систему охлаждения и смазку")
    if risk[-1] >= WARNING_THRESHOLD:
        recs.append("❗ Требуется срочное обслуживание")
    elif risk[-1] > 0.5:
        recs.append("Запланировать обслуживание в ближайшее время")
    else:
        recs.append("Нормальная работа - продолжать мониторинг")
    return recs

# ------------------------------
# Инициализация и симуляция данных
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {comp: {"vibration": [], "temperature": []} for comp in COMPONENTS}
    st.session_state.risk_data = {comp: [] for comp in COMPONENTS}
    st.session_state.timestamps = []

    for day in range(SIMULATION_DAYS):
        for hour in range(DAY_LENGTH):
            timestamp = datetime.now() - timedelta(days=SIMULATION_DAYS-day-1, hours=DAY_LENGTH-hour-1)
            st.session_state.timestamps.append(timestamp)

            for comp in COMPONENTS:
                degradation = 1 + 0.02*day
                new_vib = 0.5*degradation + 0.05*np.random.rand() + np.random.normal(0, 0.1)
                new_temp = 50*degradation + 0.02*np.random.rand() + np.random.normal(0, 0.2)

                st.session_state.sensor_data[comp]["vibration"].append(new_vib)
                st.session_state.sensor_data[comp]["temperature"].append(new_temp)

                vib_norm = normalize(st.session_state.sensor_data[comp]["vibration"])
                temp_norm = normalize(st.session_state.sensor_data[comp]["temperature"])
                risk = compute_risk(vib_norm[-1], temp_norm[-1])
                st.session_state.risk_data[comp].append(risk)

# ------------------------------
# Панель управления
st.sidebar.header("⚙️ Настройки")
show_components = st.sidebar.multiselect(
    "Выберите компоненты для отображения",
    COMPONENTS,
    default=COMPONENTS
)

# ------------------------------
# Информационные карточки
st.sidebar.header("📊 Текущие показатели")
cols = st.sidebar.columns(2)
for i, comp in enumerate(COMPONENTS):
    risk = st.session_state.risk_data[comp][-1]
    color = "green" if risk < 0.4 else "orange" if risk < 0.7 else "red"

    cols[i%2].metric(
        label=f"{comp}",
        value=f"{risk:.0%}",
        delta_color="off",
        help=f"Вибрация: {st.session_state.sensor_data[comp]['vibration'][-1]:.2f}\n"
             f"Температура: {st.session_state.sensor_data[comp]['temperature'][-1]:.2f}"
    )
    cols[i%2].progress(risk)

# ------------------------------
# Основная панель
tab1, tab2 = st.tabs(["📈 Графики показателей", "📋 Рекомендации"])

with tab1:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Риск отказа", "Вибрация", "Температура")
    )

    for comp in show_components:
        # Риск с цветами
        fig.add_trace(go.Scatter(
            x=st.session_state.timestamps,
            y=st.session_state.risk_data[comp],
            mode='lines+markers',
            name=f"Риск {comp}",
            line=dict(color=COLORS.get(comp, "gray"), width=2)
        ), row=1, col=1)

        # Вибрация
        fig.add_trace(go.Scatter(
            x=st.session_state.timestamps,
            y=st.session_state.sensor_data[comp]["vibration"],
            mode='lines',
            name=f"Вибрация {comp}"
        ), row=2, col=1)

        # Температура
        fig.add_trace(go.Scatter(
            x=st.session_state.timestamps,
            y=st.session_state.sensor_data[comp]["temperature"],
            mode='lines',
            name=f"Температура {comp}"
        ), row=3, col=1)

    fig.add_hline(y=WARNING_THRESHOLD, line_dash="dot", line_color="red", row=1, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=55, line_dash="dot", line_color="orange", row=3, col=1)

    fig.update_layout(
        height=800,
        title_text="История показателей за последние 7 дней",
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Время", row=3, col=1)
    fig.update_yaxes(title_text="Уровень риска", row=1, col=1)
    fig.update_yaxes(title_text="Вибрация (мм/с)", row=2, col=1)
    fig.update_yaxes(title_text="Температура (°C)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Рекомендации по обслуживанию")

    if st.button("🔄 Обновить анализ"):
        for comp in show_components:
            with st.expander(f"🔧 {comp}", expanded=True):
                days_to_maintenance = predict_days_to_maintenance(st.session_state.risk_data[comp])
                recs = get_recommendations(
                    st.session_state.risk_data[comp],
                    st.session_state.sensor_data[comp]["vibration"],
                    st.session_state.sensor_data[comp]["temperature"]
                )

                col1, col2 = st.columns(2)
                risk = st.session_state.risk_data[comp][-1]

                col1.metric(
                    "Текущий уровень риска",
                    f"{risk:.0%}",
                    delta=f"{risk - st.session_state.risk_data[comp][-2]:+.1%}" if len(st.session_state.risk_data[comp]) > 1 else "N/A"
                )

                if not np.isnan(days_to_maintenance):
                    col2.metric(
                        "Прогноз до обслуживания",
                        f"{days_to_maintenance} дней",
                        delta=None,
                        help="На основе тренда изменения риска"
                    )

                st.markdown("**Рекомендации:**")
                for rec in recs:
                    st.markdown(f"- {rec}")

                st.markdown("---")
