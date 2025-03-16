import streamlit as st
import pandas as pd

def export_data_tab(full_df: pd.DataFrame, city_df_: pd.DataFrame, seasonal_df: pd.DataFrame, current_city: str) -> None:
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
