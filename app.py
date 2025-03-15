import streamlit as st
import pandas as pd
import altair as alt
import requests
from datetime import datetime, timedelta
import plotly.express as px
import numpy as np
import time
import concurrent.futures
import asyncio
import httpx

st.set_page_config(
    page_title="Анализ температур",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили для оформления заголовков и карточек
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        color: #212121;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Панель управления")
    
    uploaded_file = st.file_uploader("Загрузите собственный датасет (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['season'] = df['timestamp'].dt.month.apply(
            lambda m: 'winter' if m in [12, 1, 2] else 
                      'spring' if m in [3, 4, 5] else
                      'summer' if m in [6, 7, 8] else 'autumn'
        )
    else:
        @st.cache_data
        def load_data():
            df = pd.read_csv("temperature_data.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['season'] = df['timestamp'].dt.month.apply(
                lambda m: 'winter' if m in [12, 1, 2] else 
                          'spring' if m in [3, 4, 5] else
                          'summer' if m in [6, 7, 8] else 'autumn'
            )
            return df
        df = load_data()
    
    st.subheader("Выбор города")
    cities = df["city"].unique()
    selected_city = st.selectbox("Город для анализа", cities)
    
    st.subheader("Выбор периода анализа")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Начало периода", min_date)
    with col2:
        end_date = st.date_input("Конец периода", max_date)
    
    st.subheader("API для текущих данных")
    api_key = st.text_input("OpenWeatherMap API ключ", type="password")
    if api_key:
        st.info("API ключ введен ✓")
    
    st.subheader("Метод получения текущей температуры")
    fetch_method = st.radio("Выберите метод", ("Синхронный", "Асинхронный"))
    
    st.subheader("Сравнение производительности анализа")
    parallel_compare = st.checkbox("Сравнить последовательный и параллельный расчёт для всех городов")
    
    if parallel_compare: 
        parallel_method = st.radio(
            "Выберите метод параллельных вычислений", 
            ("ThreadPoolExecutor", "multiprocessing")
        )
        
        def compute_rolling_stats(df_city):
            # Расчёт скользящего среднего, стандартного отклонения и границ
            df_city = df_city.sort_values("timestamp")
            df_city['скользящее_среднее'] = df_city['temperature'].rolling(window=30, min_periods=1).mean()
            df_city['скользящее_стандартноеОтклонение'] = df_city['temperature'].rolling(window=30, min_periods=1).std()
            df_city['верхняя_граница'] = df_city['скользящее_среднее'] + 2 * df_city['скользящее_стандартноеОтклонение']
            df_city['нижняя_граница'] = df_city['скользящее_среднее'] - 2 * df_city['скользящее_стандартноеОтклонение']
            df_city['аномалия'] = (df_city['temperature'] > df_city['верхняя_граница']) | (df_city['temperature'] < df_city['нижняя_граница'])
            return df_city

        start_seq = time.time()
        sequential_result = df.groupby('city').apply(lambda group: compute_rolling_stats(group.copy()))
        time_seq = time.time() - start_seq
        
        if parallel_method == "ThreadPoolExecutor":
            from concurrent.futures import ThreadPoolExecutor
            start_par = time.time()
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(compute_rolling_stats, group.copy()) for _, group in df.groupby('city')]
                parallel_result = [f.result() for f in futures]
            parallel_df = pd.concat(parallel_result)
            time_par = time.time() - start_par
        else:
            # На Windows multiprocessing в Streamlit может работать нестабильно.
            st.warning("Multiprocessing может работать нестабильно в данной среде. Рекомендуется использовать ThreadPoolExecutor.")
            parallel_df = sequential_result.copy()
            time_par = 0
        
        st.write(f"Последовательный расчёт: {time_seq:.3f} сек")
        st.write(f"Параллельный расчёт (метод {parallel_method}): {time_par:.3f} сек")

# Отбор данных по выбранному городу и периоду
city_data = df[(df["city"] == selected_city) & (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)].copy()
city_data.sort_values("timestamp", inplace=True)

#Скользящие статистики и границы определения аномалий.
rolling_window = 30
city_data['скользящее_среднее'] = city_data['temperature'].rolling(window=rolling_window, min_periods = 1).mean()
city_data['скользящее_стандартноеОтклонение'] = city_data['temperature'].rolling(window=rolling_window, min_periods = 1).std()
city_data['верхняя_граница'] = city_data['скользящее_среднее'] + 2 * city_data['скользящее_стандартноеОтклонение']
city_data['нижняя_граница'] = city_data['скользящее_среднее'] - 2 * city_data['скользящее_стандартноеОтклонение']
city_data['аномалия'] = (city_data['temperature'] > city_data['верхняя_граница']) | (city_data['temperature'] < city_data['нижняя_граница'])

# Группировка данных для получения сезонной статистики
seasonal_stats = df.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

# Построение линейного тренда через полиномиальную аппроксимацию
if not city_data.empty:
    city_data = city_data.sort_values("timestamp")
    city_data['timestamp_ordinal'] = city_data['timestamp'].apply(lambda x: x.toordinal())
    coeffs = np.polyfit(city_data['timestamp_ordinal'], city_data['temperature'], 1)
    city_data['тренд'] = np.polyval(coeffs, city_data['timestamp_ordinal'])
    # Тренд рассчитывается как линейная аппроксимация температур по времени.

# Создание вкладок для различных визуализаций
tab1, tab2, tab3 = st.tabs(["Временной ряд", "Аномалии", "Текущий мониторинг"])

with tab1:
    st.markdown("<h2 class='subheader'>Временной ряд температур</h2>", unsafe_allow_html=True)
    
    # Построение графика временного ряда с линиями для температуры, скользящего среднего и границ
    fig = px.line(city_data, x='timestamp', y=['temperature', 'скользящее_среднее', 'верхняя_граница', 'нижняя_граница'],
                  title=f"Температура в {selected_city} с {start_date} по {end_date}",
                  labels={'timestamp': 'Дата', 'value': 'Температура (°C)'},
                  color_discrete_map={
                      'temperature': '#1E88E5',
                      'скользящее_среднее': '#FFA000',
                      'верхняя_граница': '#EF5350',
                      'нижняя_граница': '#EF5350'
                  })
    
    anomaly_points = city_data[city_data['аномалия']]
    if not anomaly_points.empty:
        # Отмечаем аномальные точки на графике
        fig.add_scatter(x=anomaly_points['timestamp'], y=anomaly_points['temperature'],
                        mode='markers', marker=dict(size=12, color='red', symbol='circle'),
                        name='Аномалии')
    # Добавляем линию тренда на график
    fig.add_scatter(x=city_data['timestamp'], y=city_data['тренд'],
                    mode='lines', line=dict(color='green', dash='dash'),
                    name='Долгосрочный тренд')
    
    fig.update_layout(legend_title_text='', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Сезонные статистики для каждого города")
    st.dataframe(seasonal_stats)
    
    # Добавляем диаграмму сезонных профилей для выбранного города с переупорядочиванием сезонов
    city_seasonal_stats = seasonal_stats[seasonal_stats["city"] == selected_city].copy()
    if not city_seasonal_stats.empty:
        # Преобразуем названия сезонов на русский вариант
        season_map = {'winter': 'зима', 'spring': 'весна', 'summer': 'лето', 'autumn': 'осень'}
        city_seasonal_stats['season_rus'] = city_seasonal_stats['season'].map(season_map)
        
        fig_season = px.bar(
            city_seasonal_stats,
            x="season_rus",
            y="mean",
            error_y="std",
            labels={"season_rus": "Сезон", "mean": "Средняя температура", "std": "Отклонение (σ)"},
            title=f"Сезонные профили температуры для {selected_city}",
            text="mean",
            category_orders={"season_rus": ["зима", "весна", "осень", "лето"]}
        )
        fig_season.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig_season, use_container_width=True)

with tab2:
    st.markdown("<h2 class='subheader'>Обнаруженные аномалии</h2>", unsafe_allow_html=True)
    
    anomalies = city_data[city_data['аномалия']][['timestamp', 'temperature']]
    if not anomalies.empty:
        anomalies = anomalies.sort_values('timestamp', ascending=False)
        
        anomalies['дата'] = anomalies['timestamp'].dt.strftime('%d.%m.%Y')
        anomalies['время'] = anomalies['timestamp'].dt.strftime('%H:%M')
        anomalies['температура'] = anomalies['temperature'].round(1).astype(str) + ' °C'
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.dataframe(anomalies[['дата', 'время', 'температура']], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.info(f"Всего обнаружено {len(anomalies)} аномальных измерений")
    else:
        st.success("В выбранном периоде аномалии не обнаружены")

# Асинхронная функция для запроса текущей температуры через httpx
async def fetch_current_temp_async(city, api_key):
    async with httpx.AsyncClient() as client:
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = await client.get("https://api.openweathermap.org/data/2.5/weather", params=params)
        return response

with tab3:
    st.markdown("<h2 class='subheader'>Мониторинг текущей температуры</h2>", unsafe_allow_html=True)
    
    if api_key:
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": selected_city,
            "appid": api_key,
            "units": "metric"
        }
        
        try:
            with st.spinner("Получение текущих данных..."):
               
                # При одном запросе преимущество асинхронного метода не столь заметно, 
                # однако в сценариях с несколькими одновременными запросами асинхронность может дать прирост производительности.
                
                if fetch_method == "Синхронный":
                    response = requests.get(base_url, params=params)
                else:
                    response = asyncio.run(fetch_current_temp_async(selected_city, api_key))
                    
            if response.status_code == 200:
                current_data = response.json()
                current_temp = current_data['main']['temp']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.metric("Текущая температура", f"{current_temp:.1f} °C")
                    
                    today = datetime.now()
                    month = today.month
                    if month in [12, 1, 2]:
                        current_season = "зима"
                    elif month in [3, 4, 5]:
                        current_season = "весна"
                    elif month in [6, 7, 8]:
                        current_season = "лето"
                    else:
                        current_season = "осень"
                        
                    st.write(f"Текущий сезон: {current_season}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    season_map = {'winter': 'зима', 'spring': 'весна', 'summer': 'лето', 'autumn': 'осень'}
                    # Преобразование текущего сезона для сравнения с историческими данными
                    season_key = next((k for k, v in season_map.items() if v == current_season), None)
                    
                    seasonal_data = city_data[city_data['season'] == season_key]
                    if not seasonal_data.empty:
                        seasonal_mean = seasonal_data['temperature'].mean()
                        seasonal_std = seasonal_data['temperature'].std()
                        
                        st.write(f"Историческая норма для сезона '{current_season}':")
                        st.write(f"{seasonal_mean:.1f} °C ± {seasonal_std:.1f} °C")
                        
                        lower_bound_season = seasonal_mean - 2 * seasonal_std
                        upper_bound_season = seasonal_mean + 2 * seasonal_std
                        
                        if current_temp < lower_bound_season:
                            st.error("Текущая температура НИЖЕ исторической нормы!")
                        elif current_temp > upper_bound_season:
                            st.error("Текущая температура ВЫШЕ исторической нормы!")
                        else:
                            st.success("Температура в пределах исторической нормы")
                    else:
                        st.warning("Нет исторических данных для текущего сезона")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                try:
                    err_data = response.json()
                    if response.status_code == 401 and "Invalid API key" in err_data.get("message", ""):
                        st.error(f"401: {err_data.get('message', 'Неверный API ключ. Подробнее: https://openweathermap.org/faq#error401')}") 
                    else:
                        st.error(f"Ошибка API: {response.status_code}. {err_data.get('message','')}")
                except Exception:
                    st.error(f"Ошибка API: {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка при получении данных: {str(e)}")
    else:
        st.info("Введите API ключ в боковой панели для просмотра текущей температуры")
