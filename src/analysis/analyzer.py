
import pandas as pd
import numpy as np

def cal_cnt_ratio(df, *cols):
    """ 
    컬럼에 null값도 반영하여 선택한 컬럼 기준 cnt, ratio를 계산하여 dataFrame으로 반환
    - df : 데이터프레임
    - *cols : 그룹화 대상 컬럼
    - dropna=False : NaN값 반영
    """
    df_cnt = df.groupby(list(cols), dropna=False).size().reset_index(name='cnt')
    group_cols = list(cols[:-1]) if len(cols) > 1 else list(cols) # asp_name 단위의 ratio를 구하기 위함
    # df_cnt['sum_cnt'] : ratio를 구할 분모
    if group_cols:
        df_cnt['sum_cnt'] = df_cnt.groupby(group_cols, dropna=False)['cnt'].transform('sum')
    else:
        df_cnt['sum_cnt'] = df_cnt['cnt'].sum() # 그룹이 1개인 경우

    df_cnt['ratio'] = (df_cnt['cnt'] / df_cnt['sum_cnt'] * 100).round(0)
    df_cnt = df_cnt.sort_values(by=list(cols) + ['cnt'], ascending=[True] * len(cols) + [False])

    return df_cnt[list(cols) + ['cnt', 'ratio']]         

def cal_cnt_ratio2(df, group_cols, ratio_base_col):
    """
    - df: 데이터프레임
    - group_cols: 그룹화 대상 컬럼 리스트
    - ratio_base_col: 비율 계산 기준이 되는 컬럼 (예: 'asp_nm')
    """
    df_cnt = df.groupby(group_cols, dropna=False).size().reset_index(name='cnt')
    
    # ratio_base_col 기준으로 총합 계산
    df_cnt['sum_cnt'] = df_cnt.groupby(ratio_base_col, dropna=False)['cnt'].transform('sum')
    
    df_cnt['ratio'] = (df_cnt['cnt'] / df_cnt['sum_cnt'] * 100).round(0)
    df_cnt = df_cnt.sort_values(by=group_cols + ['cnt'], ascending=[True] * len(group_cols) + [False])
    
    return df_cnt[group_cols + ['cnt', 'ratio']]

def generate_group_stats(df, group_col, category_col):
    """
    특정 컬럼(category_col)의 고유값을 기준으로 그룹화(group_col)하여 통계 계산

    Parameters:
    - df: pandas DataFrame
    - group_col: 그룹화 기준 컬럼명 (예: 'asp_nm')
    - category_col: 값 분류 기준 컬럼명 (예: 'main_category_nm')

    Returns:
    - group_stats: 그룹별 통계 DataFrame
    """
    # category_col의 고유값 추출 (NaN 제외)
    category_values = df[category_col].dropna().unique().tolist()

    # 각 category_value에 대해 새로운 컬럼 생성
    for val in category_values:
        df[f'{val}수'] = (df[category_col] == val).astype(int)

    # 결측값 처리
    df['결측값수'] = df[category_col].isna().astype(int)

    # 총건수 계산용 임시 컬럼
    df['총건수'] = 1

    # 집계할 컬럼 목록
    agg_dict = {f'{val}수': 'sum' for val in category_values}
    agg_dict['결측값수'] = 'sum'
    agg_dict['총건수'] = 'sum'

    # 그룹화 및 집계
    group_stats = df.groupby(group_col).agg(agg_dict)

    # 비율 계산
    for val in category_values:
        group_stats[f'{val}비율(%)'] = (group_stats[f'{val}수'] / group_stats['총건수'] * 100).round(2)

    return group_stats

def highlight_top_10_percent(col):
    """DataFrame의 열(column)기준 '총계'와 0을 제외한 상위 10% 빨간색 볼드체로 강조하는 함수"""
    col_data = col.iloc[:-1]
    col_data = col_data[col_data > 0]
   
    if col_data.empty:
        return [''] * len(col)
       
    threshold = col_data.quantile(0.9)
    styles = np.where(col.iloc[:-1] > threshold, 'color: red; font-weight: bold;', '')
   
    return list(styles) + ['']
    
def generate_split_stats(df, group_col, category_cols):
    """ crosstab 옵션 활용 - 계층적 인덱스 지원 """
    # 1. "민생지원금" 포함 여부로 데이터 분리
    mask_minsung = df[group_col].astype(str).str.contains('민생지원금', na=False)
    df_minsung = df[mask_minsung].copy()
    df_non_minsung = df[~mask_minsung].copy()
    
    def calculate_stats_optimized(data):
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # index에 category_cols 리스트를 직접 전달 (MultiIndex 자동 생성)
        count = pd.crosstab(
            index=[data[col].astype(str) for col in category_cols],  # 각 컬럼을 개별적으로 문자열 변환
            columns=data[group_col],
            margins=True,
            margins_name='총계',
            dropna=False
        )
        # 총계값 NaN 처리 및 정수 변환
        count = count.fillna(0).astype(int)
        
        # 비율 계산
        ratio = pd.crosstab(
            index=[data[col].astype(str) for col in category_cols],  # 동일하게 처리
            columns=data[group_col],
            normalize='columns',  # 열 기준 정규화
            dropna=False
        ) * 100
        # 비율도 NaN 처리 및 정수 변환
        ratio = ratio.fillna(0).round(0).astype(int)
        
        return count, ratio
    
    count_df, ratio_df = calculate_stats_optimized(df_non_minsung)
    m_count_df, m_ratio_df = calculate_stats_optimized(df_minsung)
    
    # 스타일 적용은 동일
    def apply_style(df):
        if not df.empty:
            subset_cols = df.columns[df.columns != '총계'] if '총계' in df.columns else df.columns
            return df.style.apply(highlight_top_10_percent, subset=subset_cols)
        return pd.DataFrame().style
    
    return (apply_style(count_df), apply_style(ratio_df), apply_style(m_count_df), apply_style(m_ratio_df))

def analyze_region_characteristics(df):
    """지역별 상세 특성 분석"""
    results = {}
    
    for region in df['asp_nm'].unique():
        region_data = df[df['asp_nm'] == region]
        
        # 기본 통계
        total_count = len(region_data)
        member_ratio = (region_data['main_category_nm'] == '회원').mean() * 100
        
        # 연령대 분포
        age_dist = region_data['age_cat'].value_counts(normalize=True) * 100
        most_common_age = age_dist.index[0] if len(age_dist) > 0 else 'N/A'
        
        # 서비스 이용 패턴
        service_dist = region_data['depth_1'].value_counts(normalize=True) * 100
        most_common_service = service_dist.index[0] if len(service_dist) > 0 else 'N/A'
        
        results[region] = {
            '총_이용건수': total_count,
            '회원비율(%)': round(member_ratio, 2),
            '주요_연령대': most_common_age,
            '주요_서비스': most_common_service,
            '연령대_분포': dict(age_dist.round(2)),
            '서비스_분포': dict(service_dist.round(2))
        }
    
    return results

def calculate_region_scores(df):
    """지역별 특성을 점수로 변환"""
    scores = {}
    
    for region in df['asp_nm'].unique():
        region_data = df[df['asp_nm'] == region]
        
        # 회원 집중도 점수 (회원 비율이 높을수록 높은 점수)
        member_score = (region_data['main_category_nm'] == '회원').mean() * 100
        
        # 서비스 다양성 점수 (서비스 종류가 많을수록 높은 점수)
        service_diversity = len(region_data['depth_1'].unique())
        
        # 연령대 다양성 점수 (연령대가 다양할수록 높은 점수)
        age_diversity = len(region_data['age_cat'].unique())
        
        scores[region] = {
            '회원_집중도': round(member_score, 2),
            '서비스_다양성': service_diversity,
            '연령대_다양성': age_diversity
        }
    
    return pd.DataFrame(scores).T
