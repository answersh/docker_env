
import pandas as pd
import os
from dotenv import load_dotenv
import pymysql
import mariadb
from datetime import date, time, datetime, timedelta

# .env 파일에서 환경 변수 로드
load_dotenv()

def get_db_connection():
    """데이터베이스 연결을 생성하고 반환"""
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DATABASE"),
        port=int(os.getenv("MYSQL_PORT"))
    )
