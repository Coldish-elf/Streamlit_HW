import sys
import time
import asyncio
import requests
import httpx
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

def get_season(month: int) -> str:
    """
    Определяет сезон по номеру месяца.
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
    Загружает данные о температуре из CSV файла и добавляет информацию о сезоне.
    """
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['season'] = df['timestamp'].dt.month.apply(get_season)
    return df

@st.cache_data
def load_default_data() -> pd.DataFrame:
    """
    Загружает набор данных по умолчанию из файла temperature_data.csv с предобработкой дат и сезона.
    """
    return load_csv_data("temperature_data.csv")

@st.cache_data
def compute_rolling_stats(group: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет скользящее среднее и стандартное отклонение за 30 дней и отмечает аномалии.
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
    Сравнивает асинхронный и синхронный методы получения данных о погоде для списка городов.

    Аргументы:
        cities (list): список городов.
        api_key (str): API ключ OpenWeatherMap.

    Возвращает:
        tuple: время выполнения асинхронного и синхронного запросов.
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
    Получает текущую температуру для города асинхронно.

    Аргументы:
        city (str): название города.
        api_key (str): API ключ OpenWeatherMap.

    Возвращает:
        dict: словарь с данными ответа (status_code, data, error)
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
            return {
                "status_code": response.status_code,
                "data": response.json(),
                "error": None
            }
        except httpx.HTTPStatusError as exc:
            return {
                "status_code": exc.response.status_code,
                "data": None,
                "error": f"Ошибка API: {exc.response.status_code} - {exc.response.text}"
            }
        except httpx.RequestError as exc:
            return {
                "status_code": None,
                "data": None,
                "error": f"Ошибка соединения: {exc}"
            }
    return {"status_code": None, "data": None, "error": "Не удалось получить ответ от API"}

def test_api_key(api_key: str) -> bool:
    """
    Проверяет корректность API ключа OpenWeatherMap путем запроса для города Москва.
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
        st.error("401. Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")
    except Exception as exc:
        st.error(f"Ошибка при проверке API ключа: {exc}")
    return False

def map_season_to_rus(season: str) -> str:
    """
    Преобразует название сезона с английского на русский.
    """
    season_map = {'winter': 'зима', 'spring': 'весна', 'summer': 'лето', 'autumn': 'осень'}
    return season_map.get(season, season)
