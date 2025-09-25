import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np # For hover text

# Modified for fig1 hover text
def plot_asp_age_heatmap(df_asp_age_cat, col1, col2, title='ASP별 연령대 비율 히트맵'):
    """ASP별 연령대 비율 히트맵"""
    # col1: 'age_cat', col2: 'asp_name'
    pivot_df_ratio = df_asp_age_cat.pivot(index=col2, columns=col1, values='ratio').fillna(0)
    pivot_df_cnt = df_asp_age_cat.pivot(index=col2, columns=col1, values='cnt').fillna(0)

    # 70대 이상 비율이 높은 순으로 정렬
    if '70대 이상' in pivot_df_ratio.columns:
        pivot_df_ratio = pivot_df_ratio.sort_values('70대 이상', ascending=False)
        pivot_df_cnt = pivot_df_cnt.loc[pivot_df_ratio.index]

    # tooltip text: 히트맵에 커서를 올렸을 경우 보여주는 정보
    hover_text = []
    # r_idx : row_index
    for r_idx, r in pivot_df_ratio.iterrows():
        row_text = []
        # r_idx로 각행(r)을 가져온 후 그 열(c_idx)을 값(c)로 반환 (ex: [('20대, 10.0),(70대 이상, 33.0)]
        for c_idx, c in r.items():
            ratio_val = pivot_df_ratio.loc[r_idx, c_idx]
            cnt_val = pivot_df_cnt.loc[r_idx, c_idx]
            row_text.append(f"asp_name: {r_idx}<br>age_cat: {c_idx}<br>ratio: {ratio_val}%<br>cnt: {cnt_val}")
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df_ratio.values,
        x=pivot_df_ratio.columns,
        y=pivot_df_ratio.index,
        text=pivot_df_ratio.applymap(lambda x: f'{x:.1f}'),
        texttemplate="%{text}",
        hovertext=hover_text,
        hoverinfo='text',
        colorscale='Blues'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='연령대',
        yaxis_title='ASP',
        font=dict(size=10)
    )
    fig.update_xaxes(side='bottom')
    return fig

def plot_age_comparison(df_asp_age_cat, col1, col2, title='ASP별 연령대 그룹 비교'):
    """고령층 vs 젊은층 비율 비교"""
    # This function remains largely the same
    df_asp_age_cat['ratio'] = pd.to_numeric(df_asp_age_cat['ratio'], errors='coerce').fillna(0)
    # index : y축(열 기준), column : x축(행 기준)
    df_pivot = df_asp_age_cat.pivot(index=col1, columns=col2, values='ratio').fillna(0) 
    young_cols = ['10대 미만', '10대', '20대', '30대']
    middle_cols = ['40대', '50대']
    elderly_cols = ['60대', '70대 이상']
    unknown_cols = ['Unknown']
    existing_young_cols = [col for col in young_cols if col in df_pivot.columns]
    existing_middle_cols = [col for col in middle_cols if col in df_pivot.columns]
    existing_elderly_cols = [col for col in elderly_cols if col in df_pivot.columns]
    existing_unknown_cols = [col for col in unknown_cols if col in df_pivot.columns]
    young = df_pivot[existing_young_cols].sum(axis=1) if existing_young_cols else 0
    middle = df_pivot[existing_middle_cols].sum(axis=1) if existing_middle_cols else 0
    elderly = df_pivot[existing_elderly_cols].sum(axis=1) if existing_elderly_cols else 0
    unknown = df_pivot[existing_unknown_cols].sum(axis=1) if existing_unknown_cols else 0
    comparison_df = pd.DataFrame({
        'ASP': df_pivot.index,
        '젊은층(10-30대)': young,
        '중장년층(40-50대)': middle,
        '고령층(60대이상)': elderly,
        'Unknown': unknown
    }).sort_values('고령층(60대이상)', ascending=False) # 고령층 높은 ASP(df_pivot.index)순으로 정렬
    fig = px.bar(comparison_df, 
                 x='ASP', 
                 y=['고령층(60대이상)', '중장년층(40-50대)', '젊은층(10-30대)', 'Unknown'],
                 title=title, 
                 barmode='group', 
                 color_discrete_map={'젊은층(10-30대)': '#1f77b4', '중장년층(40-50대)': '#bcbd22', '고령층(60대이상)': '#ff7f0e', 'Unknown': '#2ca02c'}
                )
    fig.update_layout(xaxis=dict(title='ASP', tickangle=-45),
                      yaxis=dict(title='비율(%)'),
                      legend_title='연령대 그룹'
                     )
    return fig

# Modified for fig3 requirements
def plot_top_asp_by_age(df_asp_age_cat, col1, col2, top_n=5, title='연령대별 상위 5개 ASP'):
    """각 연령대별 상위 ASP 표시"""
    # Exclude '10대 미만' and 'Unknown'
    age_order = ['10대', '20대', '30대', '40대', '50대', '60대', '70대 이상', 'Unknown']
    age_categories = [cat for cat in age_order if cat in df_asp_age_cat[col2].unique()]

    fig = make_subplots(
        rows=2, cols=len(age_categories) // 2 + len(age_categories) % 2,
        subplot_titles=age_categories,
        specs=[[{"type": "bar"}]*(len(age_categories) // 2 + len(age_categories) % 2)]*2
    )

    colors = px.colors.qualitative.Pastel

    for i, age_cat in enumerate(age_categories):
        df_age = df_asp_age_cat[df_asp_age_cat[col2] == age_cat].nlargest(top_n, 'ratio')

        row = (i // (len(age_categories) // 2 + len(age_categories) % 2)) + 1
        col = (i % (len(age_categories) // 2 + len(age_categories) % 2)) + 1

        fig.add_trace(
            go.Bar(
                x=df_age[col1],
                y=df_age['ratio'],
                name=age_cat,
                marker_color=colors[i % len(colors)],
                customdata=df_age['cnt'],
                hovertemplate='<b>%{x}</b><br>비율(y): %{y:.2f}%<br>건수(cnt): %{customdata}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_xaxes(tickangle=45, row=row, col=col)

    fig.update_layout(
        title=title,
        height=600,
        font=dict(size=10)
    )
    return fig

# Modified for fig4 and fig6 requirements
def plot_crosstab_heatmap(df, title=""):
    """crosstab 결과 Plotly 히트맵으로 시각화"""
    df_for_plot = df.copy()

    # '총계' 행과 열 제거 (있는 경우)
    if '총계' in df_for_plot.index:
        df_for_plot = df_for_plot.drop(index='총계')
    if '총계' in df_for_plot.columns:
        df_for_plot = df_for_plot.drop(columns='총계')

    if isinstance(df_for_plot.index, pd.MultiIndex):
        df_for_plot.index = df_for_plot.index.get_level_values(0).astype(str)
        df_for_plot = df_for_plot.groupby(df_for_plot.index).sum()
    # IVR명이 A&B&C 인 경우 "&"기준 줄바꿈(<br>)적용, "ASP명 - 민생지원금"인 경우 "-"기준 줄바꿈(<br>)적용
    y_labels = [str(label).replace("&", "<br>") for label in df_for_plot.index]
    x_labels = [str(col).replace("-", "<br>") for col in df_for_plot.columns]
    z_data = df_for_plot.values

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        text=z_data,
        texttemplate="%{text}",
        textfont={"size":10},
        hoverinfo='x+y+z'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="asp명",
        yaxis_title="IVR유형(1단계)",
        xaxis=dict(tickangle=90),
        yaxis=dict(autorange='reversed'),
        autosize=False,
        width=800,
        height=700,
        margin=dict(l=150)
    )
    fig.update_yaxes(tickfont=dict(size=12))
    return fig

# Modified for fig8 requirements
def plot_top_line_chart(df, group_col, date_col='connected_at', cnt_col='cnt',
                        top_n=10, show_all=False, title=None, width=1000, height=600):
    """상위 N개 그룹의 시계열 라인 차트를 그리는 함수"""
    from analysis import analyzer # analyzer.cal_cnt_ratio 함수 사용을 위해
    # 모두 보이기 (show_all=False로 기본값 설정됨)
    if show_all: 
        group_top = df[group_col].unique() # group_col : asp_name
        if title is None:
            title = "일별 문의 (전체)"
    else:
        group_top_df = analyzer.cal_cnt_ratio(df, group_col).sort_values(by='cnt', ascending=False).reset_index(drop=True)
        # 상위 10(top_n=10)곳의 ASP만 보이게 설정
        group_top = group_top_df.head(top_n)[group_col]  
        if title is None:
            title = f"Top {top_n} 일별 문의"

    df_cnt = analyzer.cal_cnt_ratio(df, date_col, group_col)
    df_cnt_filtered = df_cnt[df_cnt[group_col].isin(group_top)]

    sorted_groups = df_cnt_filtered.groupby(group_col)['cnt'].sum().sort_values(ascending=False).index

    color_sequence = px.colors.qualitative.Plotly

    fig = px.line(
        data_frame=df_cnt_filtered,
        x=date_col,
        y=cnt_col,
        color=group_col,
        markers=True,
        category_orders={group_col: sorted_groups},
        color_discrete_sequence=color_sequence,
        title=title
    )
    fig.update_layout(
        xaxis_title="기간", 
        yaxis_title="문의 건수", 
        legend_title=group_col,
        width=width,
        height=height
    )
    # Let the dashboard handle the date formatting
    fig.update_xaxes(dtick="D1", tickangle=45) # dtick(눈금 간격) : D1(1일), D7(1주), M1(1달), Y1(1년) 
    return fig


def plot_asp_clustering(df_asp_age_cat, col1, col2):
    """ASP별 연령 분포 패턴 클러스터링"""
    # This function remains the same
    pivot_df = df_asp_age_cat.pivot(index=col1, columns=col2, values='ratio').fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    fig = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        color=[f'클러스터 {i}' for i in clusters],
        text=pivot_df.index, title='ASP 연령 분포 패턴 클러스터링',
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
    )
    fig.update_traces(textposition="middle right")
    return fig