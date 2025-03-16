import streamlit as st
import plotly.express as px
from .helper_functions import map_season_to_rus

def time_series_tab(city_data, selected_city, start_date, end_date, seasonal_stats):
    st.markdown("<h2 class='subheader'>Временной ряд температур</h2>", unsafe_allow_html=True)
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
    if 'тренд' in city_data.columns:
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
