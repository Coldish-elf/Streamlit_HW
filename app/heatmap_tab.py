import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import date

def heatmap_tab(df, selected_city, start_date, end_date):
    st.markdown("<h2 class='subheader'>Тепловая карта</h2>", unsafe_allow_html=True)
    period_start = st.date_input("Начало периода для тепловой карты (все города)", value=start_date, key="all_start")
    period_end = st.date_input("Конец периода для тепловой карты (все города)", value=end_date, key="all_end")

    heatmap_data = df.copy()
    heatmap_data['date'] = pd.to_datetime(heatmap_data['timestamp']).dt.date
    date_mask = (heatmap_data['date'] >= period_start) & (heatmap_data['date'] <= period_end)
    heatmap_data = heatmap_data[date_mask]

    pivot_data = heatmap_data.pivot_table(
        index='city',
        columns='date',
        values='temperature',
        aggfunc='mean'
    ).round(1)
    if pivot_data.empty:
        st.warning("Недостаточно данных для построения тепловой карты в выбранном периоде")
    else:
        pivot_data = pivot_data.reindex(sorted(pivot_data.index), axis=0)
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        x_labels = [dt.strftime("%d %b %Y") for dt in pivot_data.columns]
        try:
            selected_index = list(pivot_data.index).index(selected_city)
        except ValueError:
            selected_index = None
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
        for i in range(1, len(pivot_data.index)):
            fig_heat.add_shape(
                type="line",
                x0=-0.5,
                x1=len(x_labels) - 0.5,
                y0=i - 0.5,
                y1=i - 0.5,
                line=dict(color="black", width=1, dash="dash")
            )
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

    st.markdown("### Статистика температур по городам")
    stats_df = heatmap_data.groupby('city')['temperature'].agg(['min', 'mean', 'max']).round(1)
    stats_df.columns = ['Минимум (°C)', 'Среднее (°C)', 'Максимум (°C)']
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>Тепловая карта для выбранного города</h2>", unsafe_allow_html=True)
    city_df = df[df['city'] == selected_city].copy()
    if not city_df.empty:
        city_df['date'] = city_df['timestamp'].dt.date
        city_min_date = city_df['date'].min()
        city_max_date = city_df['date'].max()
    else:
        city_min_date, city_max_date = start_date, end_date

    city_period_start = st.date_input("Начало периода для выбранного города", value=city_min_date, key="city_start")
    city_period_end = st.date_input("Конец периода для выбранного города", value=city_max_date, key="city_end")

    city_heat_data = city_df.copy()
    city_date_mask = (city_heat_data['date'] >= city_period_start) & (city_heat_data['date'] <= city_period_end)
    city_heat_data = city_heat_data[city_date_mask]

    if city_heat_data.empty:
        st.warning("Недостаточно данных для выбранного города в выбранном периоде")
        return

    pivot_city = city_heat_data.pivot_table(
        index='city',
        columns='date',
        values='temperature',
        aggfunc='mean'
    ).round(1)
    pivot_city = pivot_city.reindex(sorted(pivot_city.columns), axis=1)
    x_labels_city = [dt.strftime("%d %b %Y") for dt in pivot_city.columns]

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

    st.markdown("### Статистика температур для выбранного города")
    city_stats = city_heat_data['temperature'].agg(['min', 'mean', 'max']).round(1)
    stats_city_df = pd.DataFrame(city_stats).transpose()
    stats_city_df.columns = ['Минимум (°C)', 'Среднее (°C)', 'Максимум (°C)']
    st.dataframe(stats_city_df, use_container_width=True)
