from data import data_loader

import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import io

# 1. 데이터 로딩 함수 캐시 처리 (3시간)
@st.cache_data(ttl=10800)
def load_all_data():
    df = data_loader.load_frequent_caller_data()
    df['connected_at'] = pd.to_datetime(df['connected_at'], errors='coerce')
    return df

df = load_all_data()

## sidebar 1 : 기간 선택 #################################################################
st.sidebar.header("기준일 선택")
today = datetime.now().date()
yesterday = today - timedelta(days=1)

# 기준일 선택 범위
month_ago = today - timedelta(days=30)
reference_date = st.sidebar.date_input(
    '기준일',
    value=yesterday,
    min_value=month_ago,
    max_value=yesterday,
    help='기준일은 최근 1개월까지만 설정 가능합니다.'
)

# data filtering : 기준일, 2달전
ref_datetime = datetime.combine(reference_date, datetime.min.time())
two_months_before_ref = ref_datetime - timedelta(days=60)

df_filtered = df[
(df['connected_at'] >= two_months_before_ref) & (df['connected_at'] < ref_datetime + timedelta(days=1))
].copy()

##########################################################################################


if df_filtered.empty:
    st.warning("선택된 기간의 데이터가 없습니다.")
else:
    st.subheader(f"'{reference_date}' 기준 가맹점, 정산 상담내역")
    st.markdown("""
        - 최근 일주일간 3회이상 상담한 고객의 최근 2개월 내역 중 "가맹점", "정산"이 포함된 내역    
    """)
    df_display_raw = df_filtered[df_filtered['connected_at'].dt.date == reference_date]
    # st.dataframe(df_filtered)
    st.dataframe(df_filtered)
    st.markdown("---")
    
    # pivot table
    st.subheader(f"{reference_date} 기준 주 3회이상 문의한 가맹점의 최근 2개월 통화내역")
    st.markdown("""
        - 최근 일주일간 3회이상 상담한 고객의 월별 문의 건수
    """)
    df_filtered['month'] = df_filtered['connected_at'].dt.month.astype(str) + '월'
    pivot_df_detailed_1 = pd.pivot_table(
        df_filtered,
        values = 'connected_at',
        index='caller_number',
        columns=['month'],
        aggfunc='count',
        fill_value=0
    )
    
    pivot_df_detailed_1['계'] = pivot_df_detailed_1.sum(axis=1)
    pivot_df_detailed_1 = pivot_df_detailed_1.sort_values(by='계', ascending=False)
    # 0 → '-'
    styled_pivot = pivot_df_detailed_1.style.format(lambda x: '-' if x == 0 else x)
    st.dataframe(styled_pivot, width='stretch', height=210)
    
    st.markdown("""
        - 최근 일주일간 3회이상 상담한 고객의 월, intent별 문의 건수
    """)
    df_filtered['month'] = df_filtered['connected_at'].dt.month.astype(str) + '월'
    pivot_df_detailed_2 = pd.pivot_table(
        df_filtered,
        values = 'connected_at',
        index=['caller_number', 'detail_intent_0523'],
        columns=['month'],
        aggfunc='count',
        fill_value=0
    )
    
    pivot_df_detailed_2['계'] = pivot_df_detailed_2.sum(axis=1)
    # 0 → '-'
    styled_pivot = pivot_df_detailed_2.style.format(lambda x: '-' if x == 0 else x)
    st.dataframe(styled_pivot, width='stretch', height=210)
    st.markdown("---")
