"""
Анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API.
"""
import sys
import time
from datetime import timedelta, date, datetime
import asyncio
import httpx
import requests
import numpy as np
import pandas as pd
import plotly.express as px
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import streamlit as st

st.set_page_config(
    page_title="Анализ температур",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_season(month: int) -> str:
    """
    Возвращение названий времен года по числовому значению.
    """
    if month in [12, 1, 2]:
        return 'winter'
    if month in [3, 4, 5]:
        return 'spring'
    if month in [6, 7, 8]:
        return 'summer'
    return 'autumn'

def load_csv_data(path: str) -> pd.DataFrame:
    """
    Загрузка CSV из указанного пути.
    Преобразует столбец 'timestamp' в формат datetime и добавляет столбец 'season'.
    """
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['season'] = df['timestamp'].dt.month.apply(get_season)
    return df

@st.cache_data
def load_default_data() -> pd.DataFrame:
    """
    Загрузка стандартного датасета при отсутствии загрузки пользовательского файла.
    Результаты кэшируются для ускорения работы.
    """
    return load_csv_data("temperature_data.csv")

def compute_rolling_stats(group: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление скользящей статистики (среднее, std, верхняя и нижняя граница)
    и флаг аномалии (вне границ ±2σ).
    """
    rolling_window = 30
    group = group.sort_values('timestamp').copy()
    group['скользящее_среднее'] = group['temperature'].rolling(window=rolling_window, min_periods=1).mean()
    group['скользящее_стандартноеОтклонение'] = group['temperature'].rolling(window=rolling_window, min_periods=1).std()
    group['верхняя_граница'] = group['скользящее_среднее'] + 2 * group['скользящее_стандартноеОтклонение']
    group['нижняя_граница'] = group['скользящее_среднее'] - 2 * group['скользящее_стандартноеОтклонение']
    group['аномалия'] = (group['temperature'] > group['верхняя_граница']) | (
        group['temperature'] < group['нижняя_граница']
    )
    return group

async def compare_fetch_methods(cities: list, api_key: str):
    """
    Сравнение производительности синхронного и асинхронного методов получения данных через API.
    Возвращает время выполнения каждого метода и отношение ускорения.
    """
    start = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": city,
                    "appid": api_key,
                    "units": "metric"
                }
            )
            for city in cities
        ]
        await asyncio.gather(*tasks)
    async_time = time.time() - start
    start = time.time()
    for city in cities:
        requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": api_key,
                "units": "metric"
            },
            timeout=10
        )
    sync_time = time.time() - start
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Асинхронный метод: {async_time:.3f} сек")
    with col2:
        st.write(f"Синхронный метод: {sync_time:.3f} сек")
    with col3:
        st.write(f"Ускорение: x{sync_time / max(async_time, 0.001):.1f}")
    return async_time, sync_time

async def fetch_current_temp_async(city: str, api_key: str):
    """
    Асинхронное получение текущей температуры для города через OpenWeatherMap API.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        try:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params=params
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            st.error(f"Ошибка API: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            st.error(f"Ошибка соединения: {exc}")
    return None

def test_api_key(api_key: str) -> bool:
    """
    Проверка корректности API ключа путем запроса для города "Moscow".
    """
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": "Moscow",
                "appid": api_key,
                "units": "metric"
            },
            timeout=5
        )
        if response.status_code == 200:
            st.info("API ключ введен ✓")
            return True
        st.error("Неверный API ключ! Проверьте корректность ключа "
                 "[подробнее](https://openweathermap.org/faq#error401)")
    except Exception as exc:
        st.error(f"Ошибка при проверке API ключа: {exc}")
    return False

def map_season_to_rus(season: str) -> str:
    """
    Преобразует название сезона с английского на русский.
    """
    season_map = {'winter': 'зима', 'spring': 'весна', 'summer': 'лето', 'autumn': 'осень'}
    return season_map.get(season, season)

st.markdown(
    """
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
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 class='main-header'>Анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API</h1>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Панель управления")
    uploaded_file = st.file_uploader("Загрузите собственный датасет (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['season'] = df['timestamp'].dt.month.apply(get_season)
    else:
        df = load_default_data()

    st.subheader("Выбор города")
    cities_list = df["city"].unique()
    selected_city = st.selectbox("Город для анализа", cities_list)

    st.subheader("Выбор периода анализа")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    col1_side, col2_side = st.columns(2)
    with col1_side:
        start_date = st.date_input("Начало периода", min_date)
    with col2_side:
        end_date = st.date_input("Конец периода", max_date)

    st.subheader("API для текущих данных")
    api_key = st.text_input("OpenWeatherMap API ключ", type="password")
    if api_key:
        valid_key = test_api_key(api_key)

    st.subheader("Метод получения текущей температуры")
    fetch_method = st.radio("Выберите метод", ("Синхронный", "Асинхронный"))

    if st.checkbox("Сравнить производительность методов"):
        if not api_key or not valid_key:
            st.error("Для сравнения методов требуется корректный API-ключ!")
        else:
            test_cities = ["Berlin", "Moscow", "Beijing", "Dubai", "Cairo"]
            with st.spinner("Тестируем методы..."):
                try:
                    asyncio.run(compare_fetch_methods(test_cities, api_key))
                except Exception as exc:
                    if "401" in str(exc):
                        st.error(
                            "Неверный API-ключ! "
                            "Проверьте корректность ключа [подробнее](https://openweathermap.org/faq#error401)"
                        )
                    else:
                        st.error(f"Ошибка тестирования: {str(exc)}")

    st.subheader("Сравнение производительности анализа")
    parallel_compare = st.checkbox("Сравнить последовательный и параллельный расчёт для всех городов")
    if parallel_compare:
        parallel_method = st.radio("Выберите метод параллельных вычислений", ("ThreadPoolExecutor", "multiprocessing"))
        start_seq = time.time()
        sequential_result = df.groupby('city').apply(lambda group: compute_rolling_stats(group.copy()))
        time_seq = time.time() - start_seq
        if parallel_method == "ThreadPoolExecutor":
            start_par = time.time()
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(compute_rolling_stats, group.copy()) for _, group in df.groupby('city')]
                parallel_result = [f.result() for f in futures]
            parallel_df = pd.concat(parallel_result)
            time_par = time.time() - start_par
        else:
            start_par = time.time()
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(compute_rolling_stats, group.copy()) for _, group in df.groupby('city')]
                parallel_result = [f.result() for f in futures]
            parallel_df = pd.concat(parallel_result)
            time_par = time.time() - start_par
            if sys.platform.startswith('win'):
                st.warning("На Windows multiprocessing может иметь большие расходы для небольших данных.")
        st.write(f"Последовательный расчёт: {time_seq:.3f} сек")
        st.write(f"Параллельный расчёт ({parallel_method}): {time_par:.3f} сек")
        st.write(f"Ускорение: x{time_seq / max(time_par, 0.001):.1f}")

mask = (
    (df["city"] == selected_city)
    & (df["timestamp"].dt.date >= start_date)
    & (df["timestamp"].dt.date <= end_date)
)
city_data = df.loc[mask].sort_values("timestamp").copy()
city_data = compute_rolling_stats(city_data)

seasonal_stats = df.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

if not city_data.empty:
    city_data = city_data.sort_values("timestamp")
    city_data['timestamp_ordinal'] = city_data['timestamp'].apply(lambda x: x.toordinal())
    coeffs = np.polyfit(city_data['timestamp_ordinal'], city_data['temperature'], 1)
    city_data['тренд'] = np.polyval(coeffs, city_data['timestamp_ordinal'])

def time_series_tab():
    """
    Отображение графика временного ряда со скользящими статистиками, трендом и аномалиями.
    """
    st.markdown("<h2 class='subheader'>Временной ряд температур</h2>",
                unsafe_allow_html=True)
    fig = px.line(
        city_data,
        x='timestamp',
        y=['temperature', 'скользящее_среднее', 'верхняя_граница', 'нижняя_граница'],
        title=f"Температура в {selected_city} с {start_date} по {end_date}",
        labels={'timestamp': 'Дата', 'value': 'Температура (°C)'},
        color_discrete_map={
            'temperature': '#1E88E5',
            'скользящее_среднее': '#FFA000',
            'верхняя_граница': '#EF5350',
            'нижняя_граница': '#EF5350'
        }
    )
    anomaly_points = city_data[city_data['аномалия']]
    if not anomaly_points.empty:
        fig.add_scatter(
            x=anomaly_points['timestamp'],
            y=anomaly_points['temperature'],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            name='Аномалии'
        )
    fig.add_scatter(
        x=city_data['timestamp'],
        y=city_data['тренд'],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='Долгосрочный тренд'
    )
    fig.update_layout(
        legend_title_text='',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Сезонные статистики для каждого города")
    st.dataframe(seasonal_stats)

    city_seasonal_stats = seasonal_stats[seasonal_stats["city"] == selected_city].copy()
    if not city_seasonal_stats.empty:
        city_seasonal_stats['season_rus'] = city_seasonal_stats['season'].apply(map_season_to_rus)
        fig_season = px.bar(
            city_seasonal_stats,
            x="season_rus",
            y="mean",
            error_y="std",
            labels={"season_rus": "Сезон", "mean": "Средняя температура", "std": "Отклонение (σ)"},
            title=f"Сезонные профили температуры для {selected_city}",
            text="mean",
            category_orders={"season_rus": ["зима", "весна", "лето", "осень"]}
        )
        fig_season.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig_season, use_container_width=True)

def anomalies_tab():
    """
    Отображение таблицы обнаруженных температурных аномалий для выбранного города.
    """
    st.markdown("<h2 class='subheader'>Обнаруженные аномалии</h2>", unsafe_allow_html=True)
    anomalies = city_data[city_data['аномалия']][['timestamp', 'temperature']]
    if not anomalies.empty:
        anomalies = anomalies.sort_values('timestamp', ascending=False)
        anomalies['дата'] = anomalies['timestamp'].dt.strftime('%d.%m.%Y')
        has_time_data = not all(anomalies['timestamp'].dt.time == pd.Timestamp('00:00:00').time())
        if has_time_data:
            anomalies['время'] = anomalies['timestamp'].dt.strftime('%H:%M')
            anomalies['температура'] = anomalies['temperature'].round(1).astype(str) + ' °C'
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.dataframe(anomalies[['дата', 'время', 'температура']], use_container_width=True)
        else:
            anomalies['температура'] = anomalies['temperature'].round(1).astype(str) + ' °C'
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.dataframe(anomalies[['дата', 'температура']], use_container_width=True)
            st.info("Примечание: временная метка недоступна в данных, отображены только даты")
        st.markdown("</div>", unsafe_allow_html=True)
        st.info(f"Всего обнаружено {len(anomalies)} аномальных измерений")
    else:
        st.success("В выбранном периоде аномалии не обнаружены")

def monitoring_tab():
    """
    Получение и отображение текущей температуры для выбранного города.
    Сравнение с климатической сезонной нормой.
    """
    st.markdown("<h2 class='subheader'>Мониторинг текущей температуры</h2>", unsafe_allow_html=True)
    if not api_key or not test_api_key(api_key):
        st.info("Введите корректный API ключ в боковой панели для просмотра текущей температуры")
        return
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": selected_city, "appid": api_key, "units": "metric"}
    try:
        with st.spinner("Получение текущих данных..."):
            if fetch_method == "Синхронный":
                response = requests.get(base_url, params=params)
            else:
                response = asyncio.run(fetch_current_temp_async(selected_city, api_key))
        if response and response.status_code == 200:
            current_data = response.json()
            current_temp = current_data['main']['temp']
            col_mon_1, col_mon_2 = st.columns(2)
            with col_mon_1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("Текущая температура", f"{current_temp:.1f} °C")
                current_season = map_season_to_rus(get_season(datetime.now().month))
                st.write(f"Текущий сезон: {current_season}")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_mon_2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                current_season_eng = get_season(datetime.now().month)
                current_city_data = df[df['city'] == selected_city]
                seasonal_data = current_city_data[current_city_data['season'] == current_season_eng]
                if not seasonal_data.empty:
                    seasonal_mean = seasonal_data['temperature'].mean()
                    seasonal_std = seasonal_data['temperature'].std()
                    st.write(f"Норма для сезона '{current_season}':")
                    st.write(f"{seasonal_mean:.1f} °C ± {seasonal_std:.1f} °C")
                    lower_bound, upper_bound = seasonal_mean - 2 * seasonal_std, seasonal_mean + 2 * seasonal_std
                    if current_temp < lower_bound:
                        st.error("Текущая температура НИЖЕ климатической нормы!")
                    elif current_temp > upper_bound:
                        st.error("Текущая температура ВЫШЕ климатической нормы!")
                    else:
                        st.success("Температура в пределах климатической нормы")
                else:
                    st.warning("Недостаточно данных для сравнения с текущим сезоном")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            error_handled = True
            if response is None:
                st.error("Не удалось получить ответ от API")
            else:
                try:
                    err_data = response.json()
                    if response.status_code == 401 and "Invalid API key" in err_data.get("message", ""):
                        st.error("401. Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")
                    elif response.status_code == 404:
                        st.error("Город не найден. Проверьте корректность названия города.")
                    else:
                        st.error(f"Ошибка API: {response.status_code}. {err_data.get('message','')}")
                except Exception:
                    st.error(f"Ошибка API: {response.status_code}")
    except Exception as exc:
        if 'error_handled' not in locals() or not error_handled:
            st.error(f"Ошибка при получении данных: {str(exc)}")

def heatmap_tab():
    """
    Отображение тепловой карты температур по городам за выбранный период
    и дополнительная тепловая карта только для выбранного города.
    """
    st.markdown("<h2 class='subheader'>Тепловая карта</h2>", unsafe_allow_html=True)
    
    # Добавляем настройки агрегации данных
    aggregation_options = {
        "День": "D", 
        "Неделя": "W-MON", 
        "Месяц": "MS"
    }
    time_aggregation = st.radio(
        "Агрегация данных по временному периоду:",
        options=list(aggregation_options.keys()),
        horizontal=True
    )
    agg_frequency = aggregation_options[time_aggregation]
    
    period_start = st.date_input("Начало периода для тепловой карты (все города)", value=start_date, key="all_start")
    period_end = st.date_input("Конец периода для тепловой карты (все города)", value=end_date, key="all_end")
    
    # Кэширование обработки данных для тепловой карты
    @st.cache_data(ttl=3600, show_spinner=False)
    def prepare_heatmap_data(df_input, p_start, p_end, freq):
        # Используем ссылку на исходный DataFrame вместо копирования
        filtered_df = df_input[(df_input['timestamp'].dt.date >= p_start) & 
                              (df_input['timestamp'].dt.date <= p_end)]
        
        # Добавляем период для агрегации
        if freq != 'D':
            filtered_df['period'] = pd.to_datetime(filtered_df['timestamp']).dt.to_period(freq)
            # Группируем по городу и периоду
            agg_df = filtered_df.groupby(['city', 'period'])['temperature'].mean().reset_index()
            # Преобразуем период в дату (начало периода)
            agg_df['date'] = agg_df['period'].dt.to_timestamp()
        else:
            filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
            agg_df = filtered_df.groupby(['city', 'date'])['temperature'].mean().reset_index()
        
        # Создаем сводную таблицу
        pivot = agg_df.pivot_table(
            index='city',
            columns='date',
            values='temperature',
            aggfunc='mean'
        ).round(1)
        
        if not pivot.empty:
            # Сортируем индексы и столбцы один раз
            pivot = pivot.reindex(sorted(pivot.index), axis=0)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            
        return pivot, agg_df
    
    # Отображаем индикатор загрузки
    with st.spinner("Подготовка данных для тепловой карты..."):
        pivot_data, aggregated_data = prepare_heatmap_data(df, period_start, period_end, agg_frequency)
    
    if pivot_data.empty:
        st.warning("Недостаточно данных для построения тепловой карты в выбранном периоде")
    else:
        # Оптимизируем создание меток времени
        if agg_frequency == 'D':
            x_labels = [dt.strftime("%d %b %Y") for dt in pivot_data.columns]
        elif agg_frequency == 'W-MON':
            x_labels = [f"Week {dt.strftime('%d %b %Y')}" for dt in pivot_data.columns]
        else:
            x_labels = [dt.strftime("%b %Y") for dt in pivot_data.columns]
        
        try:
            selected_index = list(pivot_data.index).index(selected_city)
        except ValueError:
            selected_index = None
        
        with st.spinner("Создание тепловой карты..."):
            fig_heat = px.imshow(
                pivot_data.values,
                labels=dict(x="Дата", y="Город", color="Температура (°C)"),
                x=x_labels,
                y=pivot_data.index,
                color_continuous_scale='RdBu_r',
                zmin=-30,
                zmax=40,
                aspect="auto"
            )
            
            start_zoom = date(2010, 1, 1)
            end_zoom = date(2020, 12, 31)
            dates_list = list(pivot_data.columns)
            start_index = next((i for i, d in enumerate(dates_list) if d >= start_zoom), 0)
            end_index = next((i for i, d in enumerate(dates_list) if d > end_zoom), len(dates_list)) - 1
            
            fig_heat.update_layout(
                title="Тепловая карта температур по городам",
                height=600,
                xaxis=dict(
                    tickangle=-45,
                    tickmode="array",
                    tickvals=list(range(len(x_labels))),
                    ticktext=x_labels,
                    range=[start_index - 0.5, end_index + 0.5] if start_index < end_index else None
                ),
                coloraxis_colorbar=dict(
                    title="°C",
                    thicknessmode="pixels",
                    thickness=20,
                    lenmode="pixels",
                    len=500
                )
            )
            
            # Оптимизируем добавление разделительных линий
            if len(pivot_data.index) > 1:
                for i in range(1, len(pivot_data.index)):
                    fig_heat.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(x_labels) - 0.5,
                        y0=i - 0.5,
                        y1=i - 0.5,
                        line=dict(color="black", width=1, dash="dash")
                    )
            
            # Выделение выбранного города
            if selected_index is not None:
                fig_heat.add_shape(
                    type="rect",
                    x0=-0.5,
                    x1=len(x_labels) - 0.5,
                    y0=selected_index - 0.5,
                    y1=selected_index + 0.5,
                    line=dict(color="black", width=2),
                    opacity=1
                )
        
        st.plotly_chart(fig_heat, use_container_width=True)

    # Оптимизируем расчет статистики
    with st.spinner("Расчет статистики..."):
        @st.cache_data
        def calculate_stats(agg_df):
            return agg_df.groupby('city')['temperature'].agg(['min', 'mean', 'max']).round(1)
        
        stats_df = calculate_stats(aggregated_data)
        stats_df.columns = ['Минимум (°C)', 'Среднее (°C)', 'Максимум (°C)']
    
    st.markdown("### Статистика температур по городам")
    st.dataframe(stats_df, use_container_width=True)

    # Тепловая карта для выбранного города
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>Тепловая карта для выбранного города</h2>", unsafe_allow_html=True)
    
    # Кэшируем подготовку данных для отдельного города
    @st.cache_data(ttl=3600, show_spinner=False)
    def prepare_city_data(df_input, city_name, p_start, p_end, freq):
        city_data = df_input[df_input['city'] == city_name]
        
        if city_data.empty:
            return None, None, p_start, p_end
        
        if freq != 'D':
            city_data['period'] = pd.to_datetime(city_data['timestamp']).dt.to_period(freq)
            city_agg = city_data.groupby(['city', 'period'])['temperature'].mean().reset_index()
            city_agg['date'] = city_agg['period'].dt.to_timestamp()
        else:
            city_data['date'] = pd.to_datetime(city_data['timestamp']).dt.date
            city_agg = city_data.groupby(['city', 'date'])['temperature'].mean().reset_index()
        
        if not city_agg.empty:
            min_date = city_agg['date'].min()
            max_date = city_agg['date'].max()
            
            # Ensure min_date and max_date are converted to date objects safely
            min_date_obj = min_date.date() if hasattr(min_date, 'date') else min_date
            max_date_obj = max_date.date() if hasattr(max_date, 'date') else max_date
            
            # Фильтруем по указанному периоду
            date_mask = (city_agg['date'] >= p_start) & (city_agg['date'] <= p_end)
            city_filtered = city_agg[date_mask]
            
            if city_filtered.empty:
                return None, None, min_date_obj, max_date_obj
            
            # Создаем сводную таблицу
            pivot = city_filtered.pivot_table(
                index='city',
                columns='date',
                values='temperature',
                aggfunc='mean'
            ).round(1)
            
            if not pivot.empty:
                pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                
            return pivot, city_filtered, min_date_obj, max_date_obj
        
        return None, None, p_start, p_end
    
    # Определяем даты для выбранного города
    city_period_start = st.date_input("Начало периода для выбранного города", value=start_date, key="city_start")
    city_period_end = st.date_input("Конец периода для выбранного города", value=end_date, key="city_end")
    
    with st.spinner("Подготовка данных для выбранного города..."):
        pivot_city, city_filtered_data, _, _ = prepare_city_data(df, selected_city, city_period_start, city_period_end, agg_frequency)
    
    if pivot_city is None or pivot_city.empty:
        st.warning("Недостаточно данных для выбранного города в выбранном периоде")
    else:
        # Оптимизируем создание меток времени
        if agg_frequency == 'D':
            x_labels_city = [dt.strftime("%d %b %Y") for dt in pivot_city.columns]
        elif agg_frequency == 'W-MON':
            x_labels_city = [f"Week {dt.strftime('%d %b %Y')}" for dt in pivot_city.columns]
        else:
            x_labels_city = [dt.strftime("%b %Y") for dt in pivot_city.columns]
        
        with st.spinner("Создание тепловой карты для города..."):
            fig_city = px.imshow(
                pivot_city.values,
                labels=dict(x="Дата", y="Город", color="Температура (°C)"),
                x=x_labels_city,
                y=pivot_city.index,
                color_continuous_scale='RdBu_r',
                zmin=-30,
                zmax=40,
                aspect="auto"
            )
            
            fig_city.update_layout(
                title=f"Тепловая карта для города {selected_city}",
                height=300,
                xaxis=dict(
                    tickangle=-45,
                    tickmode="array",
                    tickvals=list(range(len(x_labels_city))),
                    ticktext=x_labels_city
                ),
                coloraxis_colorbar=dict(
                    title="°C",
                    thicknessmode="pixels",
                    thickness=20,
                    lenmode="pixels",
                    len=300
                )
            )
        
        st.plotly_chart(fig_city, use_container_width=True)

        # Оптимизируем расчет статистики для города
        st.markdown("### Статистика температур для выбранного города")
        with st.spinner("Расчет статистики для города..."):
            @st.cache_data
            def calculate_city_stats(city_df):
                stats = city_df['temperature'].agg(['min', 'mean', 'max']).round(1)
                return pd.DataFrame(stats).transpose()
            
            stats_city_df = calculate_city_stats(city_filtered_data)
            stats_city_df.columns = ['Минимум (°C)', 'Среднее (°C)', 'Максимум (°C)']
        
        st.dataframe(stats_city_df, use_container_width=True)

def export_data_tab(full_df: pd.DataFrame, city_df_: pd.DataFrame, seasonal_df: pd.DataFrame, current_city: str) -> None:
    """
    Вкладка экспорта данных:
    - Полный датасет,
    - Данные по выбранному городу,
    - Сезонные статистики.
    """
    st.markdown("<h2 class='subheader'>Экспорт данных</h2>", unsafe_allow_html=True)
    def convert_df_to_csv(df_to_convert: pd.DataFrame) -> bytes:
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv_city = convert_df_to_csv(city_df_)
    st.download_button(
        label=f"Скачать данные по городу {current_city} (CSV)",
        data=csv_city,
        file_name=f"temperature_data_{current_city}.csv",
        mime="text/csv"
    )

    csv_seasonal = convert_df_to_csv(seasonal_df)
    st.download_button(
        label="Скачать сезонные статистики (CSV)",
        data=csv_seasonal,
        file_name="seasonal_statistics.csv",
        mime="text/csv"
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Временной ряд", "Аномалии", "Текущий мониторинг", "Тепловая карта", "Экспорт данных"])

with tab1:
    time_series_tab()

with tab2:
    anomalies_tab()

with tab3:
    monitoring_tab()

with tab4:
    heatmap_tab()

with tab5:
    export_data_tab(df, city_data, seasonal_stats, selected_city)