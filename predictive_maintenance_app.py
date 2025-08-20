import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)
st.title("Система предиктивного обслуживания судов")

COMPONENTS = ["Насосы", "Подшипники", "Вентиляторы"]
WARNING_THRESHOLD, CRITICAL_THRESHOLD, FAILURE_THRESHOLD = 0.7, 0.85, 0.95
DAY_LENGTH, SIMULATION_DAYS = 24, 7
np.random.seed(42)

def normalize(x):
    """Нормализация данных с защитой от деления на ноль"""
    if len(x) == 0:
        return np.array([])
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min == 0:
        return np.ones_like(x) * 0.5
    return (x - x_min) / (x_max - x_min + 1e-10)

def compute_risk(vib, temp):
    """Вычисление риска на основе вибрации и температуры"""
    return 1 / (1 + np.exp(-5*(0.6*vib + 0.4*temp - 0.5)))

def calculate_rul(risk_data, method='linear'):
    """Расчет оставшегося срока службы (RUL) - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    if len(risk_data) < 10:
        return np.nan, "Недостаточно данных для прогноза RUL", 0.0

    try:
        # Используем последние 20 точек для анализа
        window_size = min(30, len(risk_data))
        recent_data = risk_data[-window_size:]
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = np.array(recent_data)

        if method == 'linear':
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
        else:
            model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            model.fit(X, y)
            # Прогнозируем следующие значения для определения тренда
            y_pred_current = model.predict([[len(recent_data)-1]])[0]
            y_pred_next = model.predict([[len(recent_data)]])[0]
            slope = y_pred_next - y_pred_current

        if slope <= 1e-10:
            return np.nan, "Тренд стабильный или улучшающийся", 0.0

        current_risk = risk_data[-1]
        if current_risk >= FAILURE_THRESHOLD:
            return 0.0, "❌ ОТКАЗ! Немедленная остановка", 1.0

        # Прогнозируем, когда риск достигнет порога отказа
        # Более консервативный подход - используем линейную экстраполяцию
        time_to_failure = max(0, (FAILURE_THRESHOLD - current_risk) / slope)

        # Переводим из "интервалов измерения" в дни (24 измерения в день)
        time_to_failure_days = time_to_failure / 24.0

        # Расчет достоверности на основе R²
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        confidence = min(0.95, max(0.3, r_squared))  # Ограничиваем достоверность

        # Сообщения в зависимости от оставшегося времени
        if time_to_failure_days <= 0.5:
            message = f"🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ! Осталось менее {time_to_failure_days*24:.1f} часов"
        elif time_to_failure_days <= 1:
            message = f"⚠️ Высокий риск! Осталось {time_to_failure_days:.1f} дней"
        elif time_to_failure_days <= 3:
            message = f"🔶 Средний риск! Осталось {time_to_failure_days:.1f} дней"
        elif time_to_failure_days <= 7:
            message = f"🟢 Низкий риск! Осталось {time_to_failure_days:.1f} дней"
        else:
            message = f"✅ Нормальное состояние! Осталось более {time_to_failure_days:.1f} дней"

        return time_to_failure_days, message, confidence

    except Exception as e:
        return np.nan, f"Ошибка прогноза: {str(e)}", 0.0

def get_recommendations(risk_data, vibration_data, temperature_data, rul_days):
    """Генерация рекомендаций на основе текущего состояния"""
    recs = []

    if not risk_data or len(risk_data) == 0:
        return ["📊 Недостаточно данных для анализа"]

    current_risk = risk_data[-1]
    current_vib = vibration_data[-1] if vibration_data and len(vibration_data) > 0 else 0
    current_temp = temperature_data[-1] if temperature_data and len(temperature_data) > 0 else 0

    # Рекомендации по вибрации
    if current_vib > 0.8:
        recs.append("🔴 ВЫСОКАЯ ВИБРАЦИЯ! Проверить балансировку, крепления и износ подшипников")
    elif current_vib > 0.6:
        recs.append("🟠 Повышенная вибрация. Рекомендуется диагностика механической части")
    elif current_vib > 0.4:
        recs.append("🟡 Умеренная вибрация. Контролировать состояние")

    # Рекомендации по температуре
    if current_temp > 70:
        recs.append("🔴 КРИТИЧЕСКАЯ ТЕМПЕРАТУРА! Проверить систему охлаждения и смазку")
    elif current_temp > 60:
        recs.append("🟠 Высокая температура. Проверить теплоотвод и циркуляцию смазки")
    elif current_temp > 55:
        recs.append("🟡 Повышенная температура. Увеличить частоту мониторинга")

    # Рекомендации по риску
    if current_risk >= FAILURE_THRESHOLD:
        recs.append("❌ АВАРИЙНАЯ ОСТАНОВКА! Немедленный ремонт требуется")
    elif current_risk >= CRITICAL_THRESHOLD:
        recs.append("🚨 КРИТИЧЕСКИЙ РИСК! Немедленная остановка и обслуживание")
    elif current_risk >= WARNING_THRESHOLD:
        recs.append("⚠️ ВЫСОКИЙ РИСК! Срочное плановое обслуживание")
    elif current_risk >= 0.6:
        recs.append("🔶 Средний риск. Запланировать обслуживание в ближайшее время")
    elif current_risk >= 0.4:
        recs.append("🔶 Низкий риск. Рекомендуется профилактический осмотр")
    else:
        recs.append("✅ Нормальная работа. Продолжать регулярный мониторинг")

    # Рекомендации по RUL
    if not np.isnan(rul_days):
        if rul_days <= 1:
            recs.append(f"🚨 НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ! Прогноз отказа через {rul_days*24:.1f} часов")
        elif rul_days <= 3:
            recs.append(f"⚠️ СРОЧНЫЙ РЕМОНТ! Прогноз отказа через {rul_days:.1f} дней")
        elif rul_days <= 7:
            recs.append(f"🔶 ПЛАНОВЫЙ РЕМОНТ! Прогноз отказа через {rul_days:.1f} дней")
        elif rul_days <= 30:
            recs.append(f"🟢 ПЛАНОВОЕ ОБСЛУЖИВАНИЕ! Прогноз отказа через {rul_days:.1f} дней")

    # Общие рекомендации
    recs.extend([
        "📋 Провести визуальный осмотр оборудования",
        "📊 Проанализировать исторические данные трендов",
        "🔧 Проверить затяжку всех соединений",
        "⛽ Проверить уровень смазки и качество масла"
    ])

    if not np.isnan(rul_days) and rul_days <= 7:
        recs.extend([
            "🛠️ Подготовить запасные части для ремонта",
            "👥 Уведомить техническую службу о предстоящем ремонте",
            "📅 Запланировать остановку оборудования"
        ])

    return recs

def get_risk_status(risk_value):
    """Получение статуса риска"""
    if risk_value >= FAILURE_THRESHOLD:
        return "❌ АВАРИЙНЫЙ", "red"
    elif risk_value >= CRITICAL_THRESHOLD:
        return "🔴 КРИТИЧЕСКИЙ", "red"
    elif risk_value >= WARNING_THRESHOLD:
        return "🟠 ВЫСОКИЙ", "orange"
    elif risk_value >= 0.5:
        return "🟡 ПОВЫШЕННЫЙ", "yellow"
    else:
        return "🟢 НОРМАЛЬНЫЙ", "green"

def get_component_status(risk, rul_days):
    """Получение общего статуса компонента"""
    if risk >= FAILURE_THRESHOLD:
        return "❌ АВАРИЙНЫЙ", "red"
    elif risk >= CRITICAL_THRESHOLD:
        return "🔴 КРИТИЧЕСКИЙ", "red"
    elif risk >= WARNING_THRESHOLD:
        return "🟠 ВЫСОКИЙ", "orange"
    elif not np.isnan(rul_days) and rul_days <= 1:
        return "⚠️ КРИТИЧЕСКИЙ RUL", "red"
    elif not np.isnan(rul_days) and rul_days <= 3:
        return "🔶 ВЫСОКИЙ RUL", "orange"
    elif not np.isnan(rul_days) and rul_days <= 7:
        return "🟡 СРЕДНИЙ RUL", "yellow"
    else:
        return "🟢 НОРМАЛЬНЫЙ", "green"

def generate_initial_data():
    """Генерация начальных данных (нормальное состояние)"""
    timestamps = []
    sensor_data = {comp: {"vibration": [], "temperature": []} for comp in COMPONENTS}
    risk_data = {comp: [] for comp in COMPONENTS}

    # Генерация 24 часов нормальных данных
    for hour in range(24):
        timestamp = datetime.now() - timedelta(hours=23-hour)
        timestamps.append(timestamp)

        for comp in COMPONENTS:
            # Нормальные значения
            vib = 0.3 + 0.1 * np.random.rand() + np.random.normal(0, 0.03)
            temp = 45 + 3 * np.random.rand() + np.random.normal(0, 0.5)

            sensor_data[comp]["vibration"].append(vib)
            sensor_data[comp]["temperature"].append(temp)

            # Нормализация и расчет риска
            vib_norm = normalize(sensor_data[comp]["vibration"])[-1] if len(sensor_data[comp]["vibration"]) > 0 else 0.5
            temp_norm = normalize(sensor_data[comp]["temperature"])[-1] if len(sensor_data[comp]["temperature"]) > 0 else 0.5

            risk = compute_risk(vib_norm, temp_norm)
            risk_data[comp].append(risk)

    return timestamps, sensor_data, risk_data

def generate_test_data(days=7, failure_scenario=False):
    """Генерация тестовых данных с различными сценариями"""
    timestamps = []
    sensor_data = {comp: {"vibration": [], "temperature": []} for comp in COMPONENTS}
    risk_data = {comp: [] for comp in COMPONENTS}

    for day in range(days):
        for hour in range(DAY_LENGTH):
            timestamp = datetime.now() - timedelta(days=days-day-1, hours=DAY_LENGTH-hour-1)
            timestamps.append(timestamp)

            for comp in COMPONENTS:
                if failure_scenario:
                    # Сценарий с отказами
                    if comp == "Насосы":
                        degradation = 1 + 0.08*day + 0.004*hour
                    elif comp == "Подшипники":
                        degradation = 1 + 0.05*day + 0.002*hour
                    else:
                        degradation = 1 + 0.03*day + 0.001*hour
                else:
                    # Нормальный сценарий
                    degradation = 1 + 0.01*day + 0.0005*hour

                vib = 0.3 * degradation + 0.1 * np.random.rand() + np.random.normal(0, 0.05)
                temp = 40 + 8 * degradation + 3 * np.random.rand() + np.random.normal(0, 0.8)

                sensor_data[comp]["vibration"].append(vib)
                sensor_data[comp]["temperature"].append(temp)

                # Нормализация и расчет риска
                window_size = min(24, len(sensor_data[comp]["vibration"]))
                vib_window = sensor_data[comp]["vibration"][-window_size:]
                temp_window = sensor_data[comp]["temperature"][-window_size:]

                vib_norm = normalize(vib_window)[-1] if len(vib_window) > 0 else 0.5
                temp_norm = normalize(temp_window)[-1] if len(temp_window) > 0 else 0.5

                risk = compute_risk(vib_norm, temp_norm)
                risk_data[comp].append(risk)

    return timestamps, sensor_data, risk_data

# Инициализация session_state
if 'sensor_data' not in st.session_state:
    # Генерация начальных данных
    timestamps, sensor_data, risk_data = generate_initial_data()

    st.session_state.update({
        'sensor_data': sensor_data,
        'risk_data': risk_data,
        'timestamps': timestamps,
        'rul_data': {comp: {"days": np.nan, "confidence": 0.0} for comp in COMPONENTS},
        'maintenance_history': [],
        'alerts': [],
        'data_loaded': True,
        'test_data_loaded': False
    })

# Обновление RUL данных
for comp in COMPONENTS:
    if len(st.session_state.risk_data[comp]) >= 10:
        rul_days, _, confidence = calculate_rul(st.session_state.risk_data[comp])
        st.session_state.rul_data[comp] = {"days": rul_days, "confidence": confidence}

# Сайдбар с управлением
st.sidebar.header("🧪 Управление данными")
if st.sidebar.button("🔄 Загрузить нормальные тестовые данные", use_container_width=True):
    timestamps, sensor_data, risk_data = generate_test_data(failure_scenario=False)
    st.session_state.update({
        'sensor_data': sensor_data,
        'risk_data': risk_data,
        'timestamps': timestamps,
        'test_data_loaded': True
    })
    st.rerun()

if st.sidebar.button("🔥 Загрузить критические тестовые данные", use_container_width=True):
    timestamps, sensor_data, risk_data = generate_test_data(failure_scenario=True)
    st.session_state.update({
        'sensor_data': sensor_data,
        'risk_data': risk_data,
        'timestamps': timestamps,
        'test_data_loaded': True
    })
    st.rerun()

if st.sidebar.button("🗑️ Сбросить к начальным данным", use_container_width=True):
    timestamps, sensor_data, risk_data = generate_initial_data()
    st.session_state.update({
        'sensor_data': sensor_data,
        'risk_data': risk_data,
        'timestamps': timestamps,
        'test_data_loaded': False
    })
    st.rerun()

st.sidebar.header("⚙️ Настройки")
show_components = st.sidebar.multiselect("Компоненты для отображения", COMPONENTS, default=COMPONENTS)
risk_threshold = st.sidebar.slider("Порог оповещений", 0.1, 1.0, WARNING_THRESHOLD, 0.05)

# Основной интерфейс
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Обзор", "📈 Графики", "📋 Рекомендации", "🔍 Диагностика", "🛠️ Обслуживание"])

with tab1:
    st.header("🏠 Обзор системы")

    # Статус данных
    if st.session_state.get('test_data_loaded', False):
        st.success("✅ Загружены тестовые данные")
    else:
        st.info("ℹ️ Отображаются начальные данные. Загрузите тестовые данные для анализа.")

    # Ключевые метрики
    cols = st.columns(4)
    with cols[0]:
        total_risk = np.mean([st.session_state.risk_data[comp][-1] for comp in COMPONENTS])
        delta = total_risk - 0.5 if total_risk > 0.5 else None
        st.metric("Общий риск системы", f"{total_risk:.0%}",
                 delta=f"{delta:+.1%}" if delta else None,
                 delta_color="inverse")

    with cols[1]:
        rul_values = [st.session_state.rul_data[comp]["days"] for comp in COMPONENTS
                     if not np.isnan(st.session_state.rul_data[comp]["days"])]
        avg_rul = np.mean(rul_values) if rul_values else np.nan
        st.metric("Средний RUL", f"{avg_rul:.1f} дней" if not np.isnan(avg_rul) else "N/A")

    with cols[2]:
        avg_vib = np.mean([st.session_state.sensor_data[comp]["vibration"][-1] for comp in COMPONENTS])
        st.metric("Средняя вибрация", f"{avg_vib:.2f} мм/с")

    with cols[3]:
        avg_temp = np.mean([st.session_state.sensor_data[comp]["temperature"][-1] for comp in COMPONENTS])
        st.metric("Средняя температура", f"{avg_temp:.1f}°C")

    # Показатели оборудования
    st.subheader("📊 Текущие показатели оборудования")

    for comp in COMPONENTS:
        risk = st.session_state.risk_data[comp][-1]
        rul_info = st.session_state.rul_data[comp]
        vib = st.session_state.sensor_data[comp]["vibration"][-1]
        temp = st.session_state.sensor_data[comp]["temperature"][-1]

        status_text, status_color = get_component_status(risk, rul_info["days"])
        risk_status, _ = get_risk_status(risk)

        with st.container():
            st.markdown(f"""
            <div style='border-left: 4px solid {status_color}; padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px;'>
                <h4 style='color: {status_color}; margin: 0;'>{comp} - {status_text}</h4>
                <p style='margin: 5px 0;'>
                    <b>Риск:</b> {risk:.0%} ({risk_status}) |
                    <b>RUL:</b> {rul_info['days']:.1f} д. |
                    <b>Вибр:</b> {vib:.2f} мм/с |
                    <b>Темп:</b> {temp:.1f}°C
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(risk))

with tab2:
    st.header("📈 Графики показателей")

    selected_comp = st.selectbox("Выберите компонент", COMPONENTS, key="graph_component")

    if selected_comp:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Риск отказа", "Вибрация", "Температура", "Тренд RUL"),
            vertical_spacing=0.1
        )

        # Данные для графиков
        risk_data = st.session_state.risk_data[selected_comp]
        vib_data = st.session_state.sensor_data[selected_comp]["vibration"]
        temp_data = st.session_state.sensor_data[selected_comp]["temperature"]

        # Риск
        fig.add_trace(go.Scatter(x=st.session_state.timestamps, y=risk_data, name='Риск', line=dict(color='red')), 1, 1)

        # Вибрация
        fig.add_trace(go.Scatter(x=st.session_state.timestamps, y=vib_data, name='Вибрация', line=dict(color='blue')), 1, 2)

        # Температура
        fig.add_trace(go.Scatter(x=st.session_state.timestamps, y=temp_data, name='Температура', line=dict(color='orange')), 2, 1)

        # RUL
        rul_values = []
        for i in range(len(risk_data)):
            if i >= 10 and i % 6 == 0:
                rul_days, _, _ = calculate_rul(risk_data[:i+1])
                rul_values.append(rul_days if not np.isnan(rul_days) else None)
            else:
                rul_values.append(None)

        fig.add_trace(go.Scatter(x=st.session_state.timestamps[:len(rul_values)], y=rul_values,
                               name='RUL', line=dict(color='green')), 2, 2)

        # Пороги
        fig.add_hline(y=WARNING_THRESHOLD, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=CRITICAL_THRESHOLD, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=FAILURE_THRESHOLD, line_dash="dash", line_color="black", row=1, col=1)

        fig.update_layout(height=600, showlegend=True, title_text=f"Показатели {selected_comp}")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📋 Рекомендации по обслуживанию")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Данные не загружены.")
    else:
        for comp in COMPONENTS:
            with st.expander(f"🔧 {comp} - Рекомендации", expanded=True):
                risk_data = st.session_state.risk_data[comp]
                vib_data = st.session_state.sensor_data[comp]["vibration"]
                temp_data = st.session_state.sensor_data[comp]["temperature"]
                rul_info = st.session_state.rul_data[comp]

                # Получаем рекомендации
                recommendations = get_recommendations(risk_data, vib_data, temp_data, rul_info["days"])

                # Отображаем текущие показатели
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Текущий риск", f"{risk_data[-1]:.0%}")
                with col2:
                    st.metric("Вибрация", f"{vib_data[-1]:.2f} мм/с")
                with col3:
                    st.metric("Температура", f"{temp_data[-1]:.1f}°C")

                # Отображаем RUL если доступен
                if not np.isnan(rul_info["days"]):
                    st.info(f"**Прогноз RUL:** {rul_info['days']:.1f} дней (достоверность: {rul_info['confidence']:.0%})")

                # Отображаем рекомендации
                st.subheader("Рекомендуемые действия:")
                for i, rec in enumerate(recommendations, 1):
                    if "❌" in rec or "🚨" in rec or "🔴" in rec:
                        st.error(f"{i}. {rec}")
                    elif "⚠️" in rec or "🟠" in rec:
                        st.warning(f"{i}. {rec}")
                    elif "🔶" in rec or "🟡" in rec:
                        st.info(f"{i}. {rec}")
                    else:
                        st.success(f"{i}. {rec}")

with tab4:
    st.header("🔍 Диагностика оборудования")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Данные не загружены.")
    else:
        selected_component = st.selectbox("Выберите оборудование для диагностики", COMPONENTS, key="diagnostics_component")

        if selected_component:
            risk_data = st.session_state.risk_data[selected_component]
            vib_data = st.session_state.sensor_data[selected_component]["vibration"]
            temp_data = st.session_state.sensor_data[selected_component]["temperature"]
            rul_info = st.session_state.rul_data[selected_component]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Анализ риска")

                st.metric("Текущий риск", f"{risk_data[-1]:.0%}")
                st.metric("Максимальный риск", f"{np.max(risk_data):.0%}")
                st.metric("Средний риск", f"{np.mean(risk_data):.0%}")

                if not np.isnan(rul_info["days"]):
                    st.metric("Прогноз RUL", f"{rul_info['days']:.1f} дней")
                    st.progress(min(1.0, rul_info["confidence"]))
                    st.caption(f"Достоверность прогноза: {rul_info['confidence']:.0%}")

                # График риска
                risk_fig = go.Figure()
                risk_fig.add_trace(go.Scatter(
                    x=st.session_state.timestamps[:len(risk_data)],
                    y=risk_data,
                    name='Риск',
                    line=dict(color='red', width=2)
                ))
                risk_fig.add_hline(y=WARNING_THRESHOLD, line_dash="dot", line_color="orange")
                risk_fig.add_hline(y=CRITICAL_THRESHOLD, line_dash="dot", line_color="red")
                risk_fig.add_hline(y=FAILURE_THRESHOLD, line_dash="dot", line_color="black")
                risk_fig.update_layout(height=300, title="Динамика риска")
                st.plotly_chart(risk_fig, use_container_width=True)

            with col2:
                st.subheader("📈 Статистика вибрации")

                st.metric("Текущая вибрация", f"{vib_data[-1]:.2f} мм/с")
                st.metric("Максимальная", f"{np.max(vib_data):.2f} мм/с")
                st.metric("Средняя", f"{np.mean(vib_data):.2f} мм/с")
                st.metric("Стандартное отклонение", f"{np.std(vib_data):.3f} мм/с")

                # Гистограмма вибрации
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Histogram(
                    x=vib_data,
                    name='Вибрация',
                    marker_color='blue',
                    nbinsx=20
                ))
                hist_fig.update_layout(height=300, title="Распределение вибрации")
                st.plotly_chart(hist_fig, use_container_width=True)

            # Дополнительная диагностика
            st.subheader("🌡️ Анализ температуры")
            col3, col4 = st.columns(2)

            with col3:
                st.metric("Текущая температура", f"{temp_data[-1]:.1f}°C")
                st.metric("Максимальная", f"{np.max(temp_data):.1f}°C")
                st.metric("Средняя", f"{np.mean(temp_data):.1f}°C")

                if len(temp_data) > 1:
                    trend = "↗️ Растет" if temp_data[-1] > temp_data[-2] else "↘️ Падает" if temp_data[-1] < temp_data[-2] else "➡️ Стабильна"
                    st.metric("Тренд", trend)

            with col4:
                if len(risk_data) > 10:
                    # Корреляционный анализ
                    min_len = min(50, len(risk_data), len(vib_data))
                    correlation = np.corrcoef(risk_data[-min_len:], vib_data[-min_len:])[0, 1]

                    st.metric("Корреляция риск-вибрация", f"{correlation:.2f}")

                    if correlation > 0.7:
                        st.info("✅ Сильная корреляция")
                    elif correlation > 0.3:
                        st.info("🔶 Умеренная корреляция")
                    else:
                        st.info("🔴 Слабая корреляция")

with tab5:
    st.header("🛠️ Техническое обслуживание")

    st.subheader("📋 История обслуживания")
    if st.session_state.maintenance_history:
        maintenance_df = pd.DataFrame(st.session_state.maintenance_history)
        st.dataframe(maintenance_df, use_container_width=True, height=300)
    else:
        st.info("История обслуживания отсутствует")

    st.subheader("📅 Планирование обслуживания")

    with st.form("maintenance_form"):
        col1, col2 = st.columns(2)

        with col1:
            maintenance_component = st.selectbox("Оборудование", COMPONENTS, key="maintenance_component")
            maintenance_type = st.selectbox("Тип обслуживания",
                                          ["Плановое", "Внеплановое", "Экстренное", "Профилактическое"],
                                          key="maintenance_type")

            # Рекомендация на основе RUL
            rul_info = st.session_state.rul_data[maintenance_component]
            if not np.isnan(rul_info["days"]):
                suggested_date = datetime.now() + timedelta(days=max(1, min(14, int(rul_info["days"]))))
                st.info(f"💡 Рекомендуемая дата: {suggested_date.strftime('%d.%m.%Y')}")

        with col2:
            planned_date = st.date_input("Планируемая дата", datetime.now() + timedelta(days=7))
            maintenance_status = st.selectbox("Статус", ["Запланировано", "Выполнено", "Отменено"], key="maintenance_status")
            assigned_technician = st.text_input("Ответственный техник", "Иванов И.И.")

        description = st.text_area("Описание работ", placeholder="Опишите необходимые работы...")
        required_parts = st.text_input("Необходимые запчасти", placeholder="Перечислите необходимые запчасти...")

        submitted = st.form_submit_button("💾 Сохранить обслуживание", type="primary")

        if submitted:
            if not description.strip():
                st.error("Пожалуйста, заполните описание работ")
            else:
                new_maintenance = {
                    "Оборудование": maintenance_component,
                    "Тип": maintenance_type,
                    "Дата": planned_date.strftime("%Y-%m-%d"),
                    "Статус": maintenance_status,
                    "Ответственный": assigned_technician,
                    "Описание": description,
                    "Запчасти": required_parts,
                    "Время создания": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.maintenance_history.append(new_maintenance)
                st.success("✅ Обслуживание успешно запланировано!")

                alert = {
                    "type": "maintenance_planned",
                    "component": maintenance_component,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": f"📅 Запланировано обслуживание {maintenance_component}"
                }
                st.session_state.alerts.append(alert)
                st.rerun()

    # Статистика обслуживания
    if st.session_state.maintenance_history:
        st.subheader("📊 Статистика обслуживания")

        stats_col1, stats_col2, stats_col3 = st.columns(3)

        status_counts = {"Выполнено": 0, "Запланировано": 0, "Отменено": 0}
        for m in st.session_state.maintenance_history:
            status = m.get("Статус", "")
            if status in status_counts:
                status_counts[status] += 1

        with stats_col1:
            st.metric("Выполнено", status_counts["Выполнено"])
        with stats_col2:
            st.metric("Запланировано", status_counts["Запланировано"])
        with stats_col3:
            st.metric("Отменено", status_counts["Отменено"])

        # График по компонентам
        maintenance_by_component = {}
        for maintenance in st.session_state.maintenance_history:
            comp = maintenance["Оборудование"]
            maintenance_by_component[comp] = maintenance_by_component.get(comp, 0) + 1

        if maintenance_by_component:
            maintenance_fig = go.Figure()
            maintenance_fig.add_trace(go.Bar(
                x=list(maintenance_by_component.keys()),
                y=list(maintenance_by_component.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ))
            maintenance_fig.update_layout(
                title="Количество обслуживаний по компонентам",
                height=300
            )
            st.plotly_chart(maintenance_fig, use_container_width=True)

# Оповещения
if st.session_state.data_loaded:
    for comp in COMPONENTS:
        risk = st.session_state.risk_data[comp][-1]
        rul_info = st.session_state.rul_data[comp]

        if risk > risk_threshold or (not np.isnan(rul_info["days"]) and rul_info["days"] <= 3):
            alert_msg = f"{comp}: риск {risk:.0%}, RUL {rul_info['days']:.1f}д"

            if not any(a.get("component") == comp for a in st.session_state.alerts[-3:]):
                st.session_state.alerts.append({
                    "component": comp,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": alert_msg
                })

# Отображение оповещений
if st.session_state.alerts:
    st.sidebar.header("🚨 Оповещения")
    for alert in st.session_state.alerts[-3:]:
        st.sidebar.warning(f"{alert['timestamp']} - {alert['message']}")

# Очистка старых оповещений
if len(st.session_state.alerts) > 20:
    st.session_state.alerts = st.session_state.alerts[-20:]

# Стили
st.markdown("""
<style>
    .stMetric {
        padding: 10px;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div {
        height: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
