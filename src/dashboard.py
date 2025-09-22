
import streamlit as st
import pandas as pd
from datetime import datetime
import io

from data import data_loader
from analysis import analyzer
from charts import plotting

# 1. 데이터 로딩 함수 캐시 처리 (3시간)
@st.cache_data(ttl=7200)
def load_all_data():
    df = data_loader.load_and_preprocess_data()
    df_contact = data_loader.load_and_preprocess_contact_data()
    return df, df_contact

# 데이터프레임들을 엑셀 파일로 변환하는 함수
def generate_excel(df_cnt, df_ratio, df_mcnt, df_mratio):
    output = io.BytesIO()  # io.BytesIO() : 메모리 안에서 파일처럼 데이터를 다룸
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # styler객체는 .data의 속성을 가지고 있어 hasattr(attribute)가 있는 경우 아닌 경우도 고려   
        if hasattr(df_cnt, 'data'):
            df_cnt.data.to_excel(writer, sheet_name='df_cnt')
        else:
            df_cnt.to_excel(writer, sheet_name='df_cnt')
            
        if hasattr(df_ratio, 'data'):
            df_ratio.data.to_excel(writer, sheet_name='df_ratio')
        else:
            df_ratio.to_excel(writer, sheet_name='df_ratio')

        if hasattr(df_mcnt, 'data'):
            df_mcnt.data.to_excel(writer, sheet_name='df_mcnt(민생지원금)')
        else:
            df_mcnt.to_excel(writer, sheet_name='df_mcnt(민생지원금)')
            
        if hasattr(df_mratio, 'data'):
            df_mratio.data.to_excel(writer, sheet_name='df_mratio(민생지원금)')
        else:
            df_mratio.to_excel(writer, sheet_name='df_mratio(민생지원금)')
            
    return output.getvalue()  # 메모리에 있는 것을 엑셀 데이터(바이트 형태)로 반환 

###### Dashboard UI 구성 ###### 
st.set_page_config(layout="wide") 
st.title("고객센터 내역 대시보드")

# 데이터 로딩
try:
    df_raw, df_contact_raw = load_all_data() # df, df_contact
except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다. db.database 설정을 확인해주세요: {e}")
    st.stop()

# 데이터가 비어있는지 확인
if df_raw.empty:
    st.warning("조회된 데이터가 없습니다. 데이터베이스에 데이터가 있는지 확인해주세요.")
    st.stop()

# connected_at컬럼은 날짜 정보인데 pandas에서 날짜 정보로 인식되지 않을 경우 pd.to_datetime를 적용
if not pd.api.types.is_datetime64_any_dtype(df_contact_raw['connected_at']):
    df_contact_raw['connected_at'] = pd.to_datetime(df_contact_raw['connected_at'], errors='coerce')


# 2. 사이드바 필터 : 사용자가 날짜 지정 
st.sidebar.header("필터")
min_date = df_raw['connected_at'].min().date() # 설정 가능한 시작날짜
max_date = df_raw['connected_at'].max().date() # 설정 가능한 최근날짜

# !!날짜 선택!!
selected_start_date, selected_end_date = st.sidebar.date_input(
    "날짜 선택",
    value=(min_date, max_date), # 설정 가능한 날짜
    min_value=min_date, # 유저가 선택한 시작일
    max_value=max_date, # 유저가 선택한 마지막일
    key='date_range_picker' # 위젯명
)

#  !!주차 선택!! 
# 날짜 선택에 따라 주차 옵션이 동적으로 변경
temp_df = df_raw[(df_raw['connected_at'].dt.date >= selected_start_date) & (df_raw['connected_at'].dt.date <= selected_end_date)]
week_options = ['전체'] + sorted(temp_df['week'].unique().tolist()) # 주차선택항목 : 전체, week컬럼 유니크값
selected_week = st.sidebar.selectbox("주차 선택", options=week_options) # 유저가 선택한 주차

# !!ASP 및 Category 컬럼 선택!!
asp_labels = ("민생지원금 구분(asp_name)", "asp_id기준(asp_nm)")
asp_choice = st.sidebar.radio(
    "ASP 컬럼 선택",
    asp_labels,
    index=0,
    key='asp_choice'
)
asp_column = 'asp_name' if asp_choice == asp_labels[0] else 'asp_nm'

category_labels = ("IVR기준(depth_1~3)", "상담원입력(category_nm)")
category_choice = st.sidebar.radio(
    "Category 컬럼 선택",
    category_labels,
    index=0,
    key='category_choice'
)
category_columns = ['depth_1','depth_2','depth_3'] if category_choice == category_labels[0] else ["main_category_nm","middle_category_nm","small_category_nm"]


# 3. 필터링 로직
# 주차 선택이 우선순위를 가짐
if selected_week != '전체': 
    # 특정 주차를 선택하면 해당 주의 데이터만 필터링
    filtered_df = df_raw[df_raw['week'] == selected_week]
    df_contact_filtered = df_contact_raw[df_contact_raw['week'] == selected_week]
    
    # 선택한 week컬럼에 대한 connected_at의 날짜 정보인 시작/종료일 
    start_date = filtered_df['connected_at'].min().date()
    end_date = filtered_df['connected_at'].max().date()
    
    filter_type = "week" # 필터 타입을 '주차'로 설정
else:
    # '전체' 주차를 선택한(default값)인 경우
    start_date = selected_start_date
    end_date = selected_end_date
    
    filtered_df = df_raw[(df_raw['connected_at'].dt.date >= start_date) & (df_raw['connected_at'].dt.date <= end_date)]
    df_contact_filtered = df_contact_raw[(df_contact_raw['connected_at'].dt.date >= start_date) & (df_contact_raw['connected_at'].dt.date <= end_date)]
    
    filter_type = "date" # 필터 타입을 '날짜'로 설정

# 필터링된 데이터가 없는 경우 처리
if filtered_df.empty:
    st.warning("선택하신 기간에 해당하는 데이터가 없습니다.")
    st.stop()


# 설정된 기간에 대한 정보 노출
if filter_type == 'week':
    summary_title = f"{selected_week} 요약"
else:
    summary_title = f"설정 기간 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"

st.subheader(summary_title)
st.markdown("---")


# --- 필터된 데이터로 분석 및 차트 생성 ---
df_asp_age_cat = analyzer.cal_cnt_ratio(filtered_df, asp_column, 'age_cat')

# chart 제목에 날짜 정보 추가를 위함(side bar의 날짜 설정 연동)
date_range_str = f"({start_date.strftime('%m/%d')} ~ {end_date.strftime('%m/%d')})"

# 통계 데이터 생성
df_cnt, df_ratio, df_mcnt, df_mratio = analyzer.generate_split_stats(df_contact_filtered, asp_column, category_columns)

# 차트 생성
figs = {
    'fig1': plotting.plot_asp_age_heatmap(df_asp_age_cat, asp_column, 'age_cat', title=f"ASP별 연령대 비율 히트맵 {date_range_str}"),
    'fig2': plotting.plot_age_comparison(df_asp_age_cat, asp_column, 'age_cat', title=f"ASP별 연령대 그룹비율 {date_range_str}"),
    'fig3': plotting.plot_top_asp_by_age(df_asp_age_cat, asp_column, 'age_cat', top_n=5, title=f"연령대별 상위 5개 ASP {date_range_str}"),
    'fig4': plotting.plot_crosstab_heatmap(df_cnt.data, title=f'ASP별 문의 건수 {date_range_str}'),
    'fig5': plotting.plot_crosstab_heatmap(df_ratio.data, title=f'ASP별 문의 비율(%) {date_range_str}'),
    'fig6': plotting.plot_crosstab_heatmap(df_mcnt.data, title=f'민생지원금 문의 건수 {date_range_str}'),
    'fig7': plotting.plot_crosstab_heatmap(df_mratio.data, title=f'민생지원금 비율(%) {date_range_str}'),
}

# 5. 차트 표시
figure_order = ['fig8', 'fig4', 'fig6', 'fig1', 'fig2', 'fig3', 'fig5', 'fig7']

for fig_name in figure_order:
    # fig_name으로 동적 헤더 생성
    header_title = {
        'fig8': f"일자별 통화량 추이 {date_range_str}",
        'fig4': f"ASP별 문의 건수 {date_range_str}",
        'fig6': f"민생지원금 문의 건수 {date_range_str}",
        'fig1': f"ASP별 연령대 비율 히트맵 {date_range_str}",
        'fig2': f"ASP별 연령대 그룹비율 {date_range_str}",
        'fig3': f"연령대별 상위 5개 ASP {date_range_str}",
        'fig5': f"ASP별 문의 비율(%) {date_range_str}",
        'fig7': f"민생지원금 비율(%) {date_range_str}"
    }.get(fig_name, f"Figure: {fig_name}")
    
    st.header(header_title)

    if fig_name == 'fig8':
        show_all_fig8 = st.checkbox("전체 ASP 보기", key=f'fig8_show_all_{filter_type}_{start_date}_{end_date}', value=False)
        fig8 = plotting.plot_top_line_chart(filtered_df, asp_column, show_all=show_all_fig8)
        fig8.update_xaxes(tickformat="%y년 %m월 %d일")
            st.plotly_chart(fig, use_container_width=True)
    else:
        fig = figs.get(fig_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            # 상세내용 보기에 원본 데이터프레임 표시
            if fig_name == 'fig4':
                with st.expander("상세내용 보기"):
                    st.dataframe(df_cnt.data)
            elif fig_name == 'fig6':
                with st.expander("상세내용 보기"):
                    st.dataframe(df_mcnt.data)
            elif fig_name == 'fig7':
                with st.expander("상세내용 보기"):
                    st.dataframe(df_mratio.data)
        else:
            st.warning(f"{fig_name}을 생성할 수 없습니다.")
    st.markdown("---")

# 6. 엑셀 파일 다운로드 (b.txt 요청사항)
st.header("데이터 다운로드")
excel_file = generate_excel(df_cnt, df_ratio, df_mcnt, df_mratio)

# 동적 파일 이름 생성
download_filename = f"analysis_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"

st.download_button(
    label="분석 데이터 다운로드",
    data=excel_file,
    file_name=download_filename,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
