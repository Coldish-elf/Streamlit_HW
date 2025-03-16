import asyncio
import requests
import streamlit as st
from datetime import datetime
from .helper_functions import test_api_key, fetch_current_temp_async, get_season, map_season_to_rus

@st.cache_data(ttl=300)
def fetch_weather_sync(base_url, params):
    """
    Получает данные о погоде синхронно через библиотеку requests.
    """
    try:
        response = requests.get(base_url, params=params)
        return {
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": None
        }
    except Exception as e:
        return {"status_code": None, "data": None, "error": str(e)}

async def fetch_weather_async(city, api_key):
    """
    Получает данные о погоде асинхронно через API OpenWeatherMap.
    """
    response = await fetch_current_temp_async(city, api_key)
    if response is None:
        return {"status_code": None, "data": None, "error": "Не удалось получить ответ от API"}
    try:
        status_code = response.get("status_code")
        data = response.get("data") if status_code == 200 else None
        return {"status_code": status_code, "data": data, "error": None}
    except Exception as e:
        return {"status_code": response.get("status_code"), "data": None, "error": str(e)}

@st.cache_data(ttl=300)
def fetch_weather_async_cached(city, api_key):
    return asyncio.run(fetch_weather_async(city, api_key))

def monitoring_tab(api_key, selected_city, df, fetch_method):
    """
    Выводит текущую температуру для выбранного города и сравнивает с историческими данными.
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
                result = fetch_weather_sync(base_url, params)
            else:
                result = fetch_weather_async_cached(selected_city, api_key)

        if result["status_code"] == 200 and result["data"]:
            current_data = result["data"]
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
            if result["status_code"] is None:
                st.error(result["error"] or "Не удалось получить ответ от API")
            else:
                try:
                    if result["status_code"] == 401:
                        st.error("401. Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")
                    elif result["status_code"] == 404:
                        st.error("Город не найден. Проверьте корректность названия города.")
                    else:
                        err_msg = ""
                        if result["data"] and "message" in result["data"]:
                            err_msg = result["data"]["message"]
                        st.error(f"Ошибка API: {result['status_code']}. {err_msg}")
                except Exception:
                    st.error(f"Ошибка API: {result['status_code']}")
    except Exception as exc:
        if 'error_handled' not in locals() or not error_handled:
            st.error(f"Ошибка при получении данных: {str(exc)}")
