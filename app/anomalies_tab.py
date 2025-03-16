import streamlit as st
import pandas as pd

def anomalies_tab(city_data):
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
