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

from .sidebar import render_sidebar
from .helper_functions import compute_rolling_stats
from .time_series_tab import time_series_tab
from .anomalies_tab import anomalies_tab
from .monitoring_tab import monitoring_tab
from .heatmap_tab import heatmap_tab

def run_app():
    st.set_page_config(
        page_title="Анализ температур",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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

    # Получаем данные из боковой панели
    sidebar_data = render_sidebar()
    df = sidebar_data["df"]
    selected_city = sidebar_data["selected_city"]
    start_date = sidebar_data["start_date"]
    end_date = sidebar_data["end_date"]
    api_key = sidebar_data["api_key"]
    valid_key = sidebar_data["valid_key"]
    fetch_method = sidebar_data["fetch_method"]

    # Фильтруем и вычисляем статистики
    mask = (
        (df["city"] == selected_city)
        & (df["timestamp"].dt.date >= start_date)
        & (df["timestamp"].dt.date <= end_date)
    )
    city_data = df.loc[mask].sort_values("timestamp").copy()
    city_data = compute_rolling_stats(city_data)

    seasonal_stats = df.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

    if not city_data.empty:
        city_data['timestamp_ordinal'] = city_data['timestamp'].apply(lambda x: x.toordinal())
        coeffs = np.polyfit(city_data['timestamp_ordinal'], city_data['temperature'], 1)
        city_data['тренд'] = np.polyval(coeffs, city_data['timestamp_ordinal'])

    # Создаем вкладки и отслеживаем их состояние
    tab_names = ["Временной ряд", "Аномалии", "Текущий мониторинг", "Тепловая карта"]
    tabs = st.tabs(tab_names)
    
    # Определяем индекс активной вкладки
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    
    # Отрисовываем содержимое вкладок
    with tabs[0]:
        time_series_tab(city_data, selected_city, start_date, end_date, seasonal_stats)
        st.session_state.active_tab = 0
    
    with tabs[1]:
        anomalies_tab(city_data)
        st.session_state.active_tab = 1
    
    with tabs[2]:
        monitoring_tab(api_key, selected_city, df, fetch_method)
        st.session_state.active_tab = 2
    
    with tabs[3]:
        st.session_state.active_tab = 3
        # Теперь heatmap загружается только когда вкладка активна
        heatmap_tab(df, selected_city, start_date, end_date)

if __name__ == "__main__":
    run_app()
