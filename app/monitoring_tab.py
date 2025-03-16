import asyncio
import requests
import streamlit as st
from datetime import datetime
from .helper_functions import test_api_key, fetch_current_temp_async, get_season, map_season_to_rus

def monitoring_tab(api_key, selected_city, df, fetch_method):
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
