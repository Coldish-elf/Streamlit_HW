import streamlit as st
import pandas as pd
import asyncio
from .helper_functions import load_default_data, get_season, test_api_key, compare_fetch_methods
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import sys

def render_sidebar():
    st.sidebar.header("Панель управления")
    uploaded_file = st.sidebar.file_uploader("Загрузите собственный датасет (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['season'] = df['timestamp'].dt.month.apply(get_season)
    else:
        df = load_default_data()

    st.sidebar.subheader("Выбор города")
    cities_list = df["city"].unique()
    selected_city = st.sidebar.selectbox("Город для анализа", cities_list)

    st.sidebar.subheader("Выбор периода анализа")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    col1_side, col2_side = st.sidebar.columns(2)
    with col1_side:
        start_date = st.date_input("Начало периода", min_date)
    with col2_side:
        end_date = st.date_input("Конец периода", max_date)

    st.subheader("API для текущих данных")
    api_key = st.text_input("OpenWeatherMap API ключ", type="password")
    valid_key = False
    if api_key:
        valid_key = test_api_key(api_key)

    st.sidebar.subheader("Метод получения текущей температуры")
    fetch_method = st.sidebar.radio("Выберите метод", ("Синхронный", "Асинхронный"))
    if st.sidebar.checkbox("Сравнить производительность методов"):
        if not api_key or not valid_key:
            st.sidebar.error("Для сравнения методов требуется корректный API-ключ!")
        else:
            test_cities = ["Berlin", "Moscow", "Beijing", "Dubai", "Cairo"]
            with st.sidebar.spinner("Тестируем методы..."):
                try:
                    asyncio.run(compare_fetch_methods(test_cities, api_key))
                except Exception as exc:
                    if "401" in str(exc):
                        st.sidebar.error(
                            "Неверный API-ключ! "
                            "Проверьте корректность ключа [подробнее](https://openweathermap.org/faq#error401)"
                        )
                    else:
                        st.sidebar.error(f"Ошибка тестирования: {str(exc)}")

    st.sidebar.subheader("Сравнение производительности анализа")
    parallel_compare = st.sidebar.checkbox("Сравнить последовательный и параллельный расчёт для всех городов")
    parallel_method = None
    if parallel_compare:
        parallel_method = st.sidebar.radio("Выберите метод параллельных вычислений", ("ThreadPoolExecutor", "multiprocessing"))

        start_seq = time.time()
        _ = df.groupby('city').apply(lambda group: group.copy())  # placeholder
        time_seq = time.time() - start_seq

        if parallel_method == "ThreadPoolExecutor":
            start_par = time.time()
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(lambda g: g.copy(), group) for _, group in df.groupby('city')]
                _ = [f.result() for f in futures]
            time_par = time.time() - start_par
        else:
            start_par = time.time()
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(lambda g: g.copy(), group) for _, group in df.groupby('city')]
                _ = [f.result() for f in futures]
            time_par = time.time() - start_par
            if sys.platform.startswith('win'):
                st.sidebar.warning("На Windows multiprocessing может иметь большие расходы для меньших данных.")

        st.sidebar.write(f"Последовательный расчёт: {time_seq:.3f} сек")
        st.sidebar.write(f"Параллельный расчёт ({parallel_method}): {time_par:.3f} сек")
        st.sidebar.write(f"Ускорение: x{time_seq / max(time_par, 0.001):.1f}")

    # Возвращаем нужные объекты для дальнейшего использования
    return {
        "df": df,
        "selected_city": selected_city,
        "start_date": start_date,
        "end_date": end_date,
        "api_key": api_key,
        "valid_key": valid_key,
        "fetch_method": fetch_method
    }
