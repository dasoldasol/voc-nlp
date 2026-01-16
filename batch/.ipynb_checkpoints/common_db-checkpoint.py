# /home/ssm-user/jupyter/batch/common_db.py
import os
import psycopg2


def load_dotenv(path: str) -> None:
    """
    아주 단순한 .env 로더.
    - 이미 환경변수에 설정된 값은 덮어쓰지 않습니다(os.environ.setdefault).
    """
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


def get_db_env() -> dict:
    """
    .env 로딩은 호출자가 책임지는 전제로, 현재 os.environ 기반으로 읽습니다.
    """
    return {
        "DB_HOST": os.getenv("DB_HOST", ""),
        "DB_PORT": os.getenv("DB_PORT", "5432"),
        "DB_NAME": os.getenv("DB_NAME", ""),
        "DB_USER": os.getenv("DB_USER", ""),
        "DB_PASSWORD": os.getenv("DB_PASSWORD", ""),
    }


def db_connect():
    env = get_db_env()
    if not (env["DB_HOST"] and env["DB_NAME"] and env["DB_USER"] and env["DB_PASSWORD"]):
        raise RuntimeError("DB 환경변수가 비어 있습니다. DB_HOST/DB_NAME/DB_USER/DB_PASSWORD를 설정해 주세요.")
    return psycopg2.connect(
        host=env["DB_HOST"],
        port=env["DB_PORT"],
        dbname=env["DB_NAME"],
        user=env["DB_USER"],
        password=env["DB_PASSWORD"],
    )


def load_text_dict_from_db() -> tuple[dict, set]:
    """
    stat_text_dict 기준:
      - type='compound' : word_key -> word_value
      - type='stop'     : word_key
    실패 시 빈 dict/set 반환(배치 안정성 우선)
    """
    conn = None
    cursor = None
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT word_key, word_value
            FROM stat_text_dict
            WHERE type = 'compound' AND is_active = TRUE
        """)
        compound_result = cursor.fetchall()
        compound_words = {
            row[0].strip(): row[1].strip()
            for row in compound_result if row and row[0] and row[1]
        }

        cursor.execute("""
            SELECT word_key
            FROM stat_text_dict
            WHERE type = 'stop' AND is_active = TRUE
        """)
        stop_result = cursor.fetchall()
        stop_words = {row[0].strip() for row in stop_result if row and row[0]}

        return compound_words, stop_words

    except Exception:
        return {}, set()

    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass