
import pandas as pd
from db import database
from datetime import datetime, timedelta, date, time
from dateutil.relativedelta import relativedelta

def load_and_preprocess_data():
    
    # 데이터베이스 연결
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # 날짜 조건 설정
    today = date.today()
    end_datetime = datetime.combine(today, time.min)
    start_datetime = end_datetime - timedelta(days=7)
    # start_date_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    start_date_str = "2025-07-01 00:00:00"
    end_date_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # 데이터 로드 쿼리
    query = f"""
        SELECT
            asp_id,
            asp_name,
            method,
            DATE_FORMAT(connected_at, "%Y%m%d") AS connected_at,
            depth_1,
            depth_2,
            depth_3,
            birth,
            main_category_nm,
            middle_category_nm,
            small_category_nm,
            detail_intent_0523
        FROM ksb.contact_consult
        WHERE 1=1
        AND detail_intent_0523 IS NOT NULL
        AND connected_at >= '{start_date_str}' AND connected_at < '{end_date_str}'
        AND asp_name != "테스트"
    """
    
    # 데이터 로드
    cursor.execute(query)
    rows = cursor.fetchall()
    col_nm = [desc[0] for desc in cursor.description]
    # 데이터프레임으로 변환
    df = pd.DataFrame(rows, columns=col_nm)
    # 데이터 전처리
    df["connected_at"] = pd.to_datetime(df["connected_at"], format="%Y%m%d")
    df["week"] = df["connected_at"].dt.strftime("%Y-W%U")
    current_year = datetime.now().year
    df["age"] = current_year - pd.to_numeric(df["birth"], errors="coerce")
    df["age"] = df["age"].astype("Int64")
    df["age_tens"] = (df["age"] // 10).astype("Int64")
    df['age_cat'] = df['age_tens'].apply(lambda x: "Unknown" if pd.isna(x) else 
                                                 '70대 이상' if x >= 7 else
                                                 '10대 미만' if x == 0 else
                                                 f"{int(x)}0대"
                                                )
    df = df.drop(columns=["age_tens", "age", "birth"])

    df["asp_nm"] = df["asp_id"].replace(
            {
                "000137000000000": "인천e음",
                "000139000000000": "양산사랑카드",
                "000140000000000": "경기지역화폐",
                "000143000000000": "바구페이",
                "000144000000000": "그리고 ",
                "000145000000000": "청주페이",
                "000146000000000": "강릉페이",
                "000147000000000": "천안사랑카드",
                "000152000000000": "경주페이",
                "000154000000000": "오륙도페이",
                "000159000000000": "밀양사랑카드",
                "000161000000000": "김포페이",
                "000177000000000": "울산페이",
                "000188000000000": "월출페이",
            }
    )
    # 기준일자(Data_Date)
    dd = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    df_dd = df[df["connected_at"] == dd]
    # 연결 종료
    conn.close()
    print(f"데이터 로드 완료: {len(df)}건")
    
    return df

def load_and_preprocess_contact_data():
    """ initial_contact_id 기준으로 select하는 쿼리
            - contact_id, initial_contact_id 둘다 있는 경우 제외(initial_contact_id와 같은 contact_id만 추출)
            - contact_id, initial_contact_id 둘다 있는 경우 "CALLBACK"인 케이스라 depth_1, depth_2, depth_3가 NULL값임
        """
    # 데이터베이스 연결
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # 날짜 조건
    today = date.today()  # datetime.date -> date
    end_datetime = datetime.combine(today, time.min)  
    start_datetime = end_datetime + timedelta(days=-7)  

    # start_date_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    start_date_str = "2025-07-01 00:00:00"
    end_date_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # 데이터 로드 쿼리
    query = f"""
        SELECT
            sno,
            asp_id,
            asp_name,
            initial_contact_id,
            contact_id,
            DATE_FORMAT(connected_at, "%Y%m%d") AS connected_at,
            agent_connected_at,
            dequeue_at,
            depth_1,
            depth_2,
            depth_3,
            birth,
            main_category_nm,
            middle_category_nm,
            small_category_nm,
            detail_intent_0523
        FROM
            ksb.contact_consult c1
        WHERE
            c1.connected_at >= '{start_date_str}' AND c1.connected_at < '{end_date_str}'
            AND (
                c1.initial_contact_id IS NULL
                OR EXISTS (
                    SELECT 1
                    FROM ksb.contact_consult c2
                    WHERE c2.initial_contact_id = c1.contact_id
                    AND c2.connected_at >= '{start_date_str}' AND c2.connected_at < '{end_date_str}'
                )
            )
            AND TIMESTAMPDIFF(SECOND, c1.agent_connected_at, c1.disconnected_at) >= 20 # 연결시간 20초 미만 제거
            AND asp_name != '테스트'
            
    """
    
    # 데이터 로드
    cursor.execute(query)
    rows = cursor.fetchall()
    col_nm = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=col_nm)
    
    # preprocessing : 주차, 연령대, asp_id를 asp명으로 
    df['week'] = pd.to_datetime(df['connected_at'], format='%Y%m%d').dt.strftime("%Y-W%U")

    current_year = datetime.now().year # 이제 datetime은 datetime 클래스를 가리키므로 정상 동작
    df['age'] = current_year - pd.to_numeric(df['birth'], errors='coerce')
    df['age'] = df['age'].astype('Int64')
    df['age_tens'] = (df['age'] // 10).astype('Int64')
    df['age_cat'] = df['age_tens'].apply(lambda x: "Unknown" if pd.isna(x) else 
                                            '70대 이상' if x >= 7 else
                                            '10대 미만' if x == 0 else
                                            f"{int(x)}0대"
                                        )
    df = df.drop(columns=['age', 'age_tens','birth'])

    df['asp_nm'] = df['asp_id']
    df['asp_nm'] = df['asp_nm'].replace({
        "000137000000000":"인천e음",
        "000139000000000":"양산사랑카드",
        "000140000000000":"경기지역화폐",
        "000143000000000":"바구페이",
        "000144000000000":"그리고 ",
        "000145000000000":"청주페이",
        "000146000000000":"강릉페이",
        "000147000000000":"천안사랑카드",
        "000152000000000":"경주페이",
        "000154000000000":"오륙도페이",
        "000159000000000":"밀양사랑카드",
        "000161000000000":"김포페이",
        "000177000000000":"울산페이",
        "000188000000000":"월출페이"  
    })

    # 연결 종료
    conn.close()

    print(f"contact_id 데이터 로드 완료: {len(df)}건")
    
    return df

def load_frequent_caller_data():
    """
    최근 일주일 동인 3회 이상 문의 건 중 2개월간 데이터에서 '정산', '가맹점'이 포함된 경우 기준만 추출 
    """    
    # 데이터베이스 연결
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # 날짜 조건
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_months_ago = yesterday - relativedelta(months=3)  # 정확히 3개월 전
    one_week_ago = yesterday - timedelta(days=7)          # 1주일 전
    start_date_str = three_months_ago.strftime("%Y-%m-%d 00:00:00")
    end_date_str = yesterday.strftime("%Y-%m-%d 23:59:59")
    week_start_date_str = one_week_ago.strftime("%Y-%m-%d 00:00:00")

    query = f"""
            WITH frequent_callers AS (
            SELECT caller_number
            FROM ksb.contact_consult
            WHERE connected_at BETWEEN '{week_start_date_str}' AND '{end_date_str}'
              AND enqueue_at IS NOT NULL
              AND stt IS NOT NULL
            GROUP BY caller_number
            HAVING COUNT(*) >= 3
        )
        SELECT
            cc.caller_number,
            cc.connected_at,
            cc.disconnected_at,
            cc.contact_id,
            cc.asp_id,
            cc.asp_name,
            cc.method,
            cc.depth_1,
            cc.depth_2,
            cc.depth_3,
            cc.birth,
            cc.user_id,
            cc.oper_name,
            cc.contents,
            cc.stt,
            cc.detail_intent_0523
        FROM ksb.contact_consult cc
        WHERE cc.connected_at BETWEEN '{start_date_str}' AND '{end_date_str}'
          AND cc.enqueue_at IS NOT NULL
          AND cc.stt IS NOT NULL
          AND (
              (cc.contents LIKE '%가맹점%'
               AND cc.contents LIKE '%정산%')
              OR (cc.stt LIKE '%가맹점%'
               AND cc.stt LIKE '%정산%')
          )
          AND cc.caller_number IN (SELECT caller_number FROM frequent_callers)
        ORDER BY cc.connected_at DESC, cc.caller_number;
    """

    # 데이터 로드
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        col_nm = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=col_nm)
        
        print(f"가맹점 정산 문의 데이터 로드 완료: {len(df)}건")
        return df
    except mariadb.Error as e:
        print(f"DB 연결 오류: {e}")
        return pd.DataFrame()
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
