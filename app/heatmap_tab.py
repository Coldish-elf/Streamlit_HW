import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import date, timedelta

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_heatmap_data(df, period_start, period_end, agg_frequency='D'):
    filtered_df = df[(df['timestamp'].dt.date >= period_start) & 
                     (df['timestamp'].dt.date <= period_end)]
    
    if agg_frequency != 'D':
        period_freq = 'M' if agg_frequency == 'MS' else agg_frequency
        filtered_df['period'] = pd.to_datetime(filtered_df['timestamp']).dt.to_period(period_freq)
        agg_df = filtered_df.groupby(['city', 'period'])['temperature'].mean().reset_index()
        if agg_frequency == 'MS':
            agg_df['date'] = agg_df['period'].dt.to_timestamp(how='start')
        else:
            agg_df['date'] = agg_df['period'].dt.to_timestamp()
    else:
        filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
        agg_df = filtered_df.groupby(['city', 'date'])['temperature'].mean().reset_index()
    
    return agg_df

@st.cache_data(ttl=3600, show_spinner=False)
def create_pivot_table(agg_df):
    pivot = agg_df.pivot_table(
        index='city',
        columns='date',
        values='temperature',
        aggfunc='mean'
    ).round(1)
    
    if not pivot.empty:
        pivot = pivot.reindex(sorted(pivot.index), axis=0)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    return pivot

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_stats(agg_df):
    return agg_df.groupby('city')['temperature'].agg(['min', 'mean', 'max']).round(1)

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_city_data(df, city_name, period_start, period_end, agg_frequency='D'):
    city_df = df[df['city'] == city_name]
    
    if city_df.empty:
        return None, None, period_start, period_end
    
    city_df['date'] = pd.to_datetime(city_df['timestamp']).dt.date
    min_date = city_df['date'].min()
    max_date = city_df['date'].max()
    
    if agg_frequency != 'D':
        period_freq = 'M' if agg_frequency == 'MS' else agg_frequency
        city_df['period'] = pd.to_datetime(city_df['timestamp']).dt.to_period(period_freq)
        city_agg = city_df.groupby(['city', 'period'])['temperature'].mean().reset_index()
        
        if agg_frequency == 'MS':
            city_agg['date'] = city_agg['period'].dt.to_timestamp(how='start')
        else:
            city_agg['date'] = city_agg['period'].dt.to_timestamp()
            
        date_mask = (city_agg['date'].dt.date >= period_start) & (city_agg['date'].dt.date <= period_end)
    else:
        city_agg = city_df.groupby(['city', 'date'])['temperature'].mean().reset_index()
        date_mask = (city_agg['date'] >= period_start) & (city_agg['date'] <= period_end)
    
    city_filtered = city_agg[date_mask]
    
    if city_filtered.empty:
        return None, None, min_date, max_date
    
    return city_filtered, min_date, max_date

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_city_data_by_day_month_year(df, city_name):
    """Prepare city data for showing day-month on x-axis and years on y-axis"""
    city_df = df[df['city'] == city_name]
    
    if city_df.empty:
        return None
    
    city_df['year'] = pd.to_datetime(city_df['timestamp']).dt.year
    city_df['day_month'] = pd.to_datetime(city_df['timestamp']).dt.strftime("%d %b")
    
    city_agg = city_df.groupby(['year', 'day_month'])['temperature'].mean().reset_index()
    
    return city_agg

def format_x_labels(dates, agg_frequency):
    if agg_frequency == 'D':
        return [dt.strftime("%d %b %Y") for dt in dates]
    elif agg_frequency == 'W-MON':
        return [f"Week {dt.strftime('%d %b %Y')}" for dt in dates]
    elif agg_frequency == 'MS':
        return [dt.strftime("%b %Y") for dt in dates]
    else:
        return [dt.strftime("%b %Y") for dt in dates]

def to_date(timestamp_or_date):
    if isinstance(timestamp_or_date, pd.Timestamp):
        return timestamp_or_date.date()
    return timestamp_or_date

def heatmap_tab(df, selected_city, start_date, end_date):
    st.markdown("<h2 class='subheader'>Тепловая карта</h2>", unsafe_allow_html=True)
    
    use_aggregation = st.radio(
        "Использовать агрегацию данных?",
        options=["Да", "Нет"],
        horizontal=True,
        index=0 
    ) == "Да"
    
    if use_aggregation:
        aggregation_options = {
            "День": "D", 
            "Неделя": "W-MON", 
            "Месяц": "MS"
        }
        time_aggregation = st.radio(
            "Агрегация данных по временному периоду:",
            options=list(aggregation_options.keys()),
            horizontal=True,
            index=2  
        )
        agg_frequency = aggregation_options[time_aggregation]
    else:
        agg_frequency = "D"  
    
    period_start = st.date_input("Начало периода для тепловой карты (все города)", value=start_date, key="all_start")
    period_end = st.date_input("Конец периода для тепловой карты (все города)", value=end_date, key="all_end")

    with st.spinner("Подготовка данных для тепловой карты..."):
        agg_data = prepare_heatmap_data(df, period_start, period_end, agg_frequency)
        pivot_data = create_pivot_table(agg_data)
    
    if pivot_data.empty:
        st.warning("Недостаточно данных для построения тепловой карты в выбранном периоде")
    else:
        x_labels = format_x_labels(pivot_data.columns, agg_frequency)
        
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
            
            dates_list = list(pivot_data.columns)
            if len(dates_list) > 50:
                start_zoom = date(2010, 1, 1)
                end_zoom = date(2020, 12, 31)
                start_index = next((i for i, d in enumerate(dates_list) if to_date(d) >= start_zoom), 0)
                end_index = next((i for i, d in enumerate(dates_list) if to_date(d) > end_zoom), len(dates_list)) - 1
                xaxis_range = [start_index - 0.5, end_index + 0.5] if start_index < end_index else None
            else:
                xaxis_range = None
            
            fig_heat.update_layout(
                title="Тепловая карта температур по городам",
                height=600,
                xaxis=dict(
                    tickangle=-45,
                    tickmode="array",
                    tickvals=list(range(len(x_labels))),
                    ticktext=x_labels,
                    range=xaxis_range
                ),
                coloraxis_colorbar=dict(
                    title="°C",
                    thicknessmode="pixels",
                    thickness=20,
                    lenmode="pixels",
                    len=500
                )
            )
            
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

    with st.spinner("Расчет статистики..."):
        stats_df = calculate_stats(agg_data)
        stats_df.columns = ['Минимум (°C)', 'Среднее (°C)', 'Максимум (°C)']
    
    st.markdown("### Статистика температур по городам")
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>Тепловая карта для выбранного города</h2>", unsafe_allow_html=True)
    
    with st.spinner("Подготовка данных для выбранного города..."):
        city_data = prepare_city_data_by_day_month_year(df, selected_city)
    
    if city_data is None or city_data.empty:
        st.warning("Недостаточно данных для выбранного города")
        return
    
    pivot_city = city_data.pivot_table(
        index='year',
        columns='day_month',
        values='temperature',
        aggfunc='mean'
    ).round(1)
    
    if pivot_city.empty:
        st.warning("Недостаточно данных для выбранного города")
        return

    month_day_order = pd.to_datetime(pivot_city.columns.map(lambda x: f"{x} 2000"), format="%d %b %Y").sort_values()
    column_order = month_day_order.strftime("%d %b").tolist()
    pivot_city = pivot_city[column_order]
    
    with st.spinner("Создание тепловой карты для города..."):
        fig_city = px.imshow(
            pivot_city.values,
            labels=dict(x="День-Месяц", y="Год", color="Температура (°C)"),
            x=pivot_city.columns,
            y=pivot_city.index,
            color_continuous_scale='RdBu_r',
            zmin=-30,
            zmax=40,
            aspect="auto"
        )
        
        fig_city.update_layout(
            title=f"Тепловая карта для города {selected_city} по годам",
            height=500,
            xaxis=dict(
                tickangle=-45,
                tickmode="array",
                tickvals=list(range(len(pivot_city.columns))),
                ticktext=pivot_city.columns
            ),
            coloraxis_colorbar=dict(
                title="°C",
                thicknessmode="pixels",
                thickness=20,
                lenmode="pixels",
                len=500
            )
        )
    
    st.plotly_chart(fig_city, use_container_width=True)

    st.markdown("### Статистика температур для выбранного города (все годы)")
    with st.spinner("Расчет статистики для города..."):
        city_stats = city_data['temperature'].agg(['min', 'mean', 'max']).round(1)
        stats_city_df = pd.DataFrame(city_stats).transpose()
        stats_city_df.columns = ['Минимум (°C)', 'Среднее (°C)', 'Максимум (°C)']
    
    st.dataframe(stats_city_df, use_container_width=True)
