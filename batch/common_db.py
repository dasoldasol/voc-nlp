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


def fetch_active_building_ids() -> list[int]:
    """
    활성 상태(state='ONGOING_OPERATING')인 Building ID 목록을 조회합니다.
    - 'insite' 또는 '시연'이 포함된 건물명은 제외

    Returns:
        list[int]: 활성 Building ID 목록 (id 오름차순)
        실패 시 빈 리스트 반환 (배치 안정성 우선)
    """
    conn = None
    cursor = None
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id
            FROM building
            WHERE state = 'ONGOING_OPERATING'
              AND LOWER(name) NOT LIKE '%%insite%%'
              AND name NOT LIKE '%%시연%%'
            ORDER BY id ASC
        """)
        
        rows = cursor.fetchall()
        return [int(row[0]) for row in rows if row and row[0] is not None]
    
    except Exception as e:
        print(f"[WARN] fetch_active_building_ids 실패: {e}")
        return []
    
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


def fetch_active_buildings() -> list[dict]:
    """
    활성 상태(state='ONGOING_OPERATING')인 Building 정보를 조회합니다.
    - 'insite' 또는 '시연'이 포함된 건물명은 제외

    Returns:
        list[dict]: Building 정보 목록 (id, name 포함)
        실패 시 빈 리스트 반환 (배치 안정성 우선)
    """
    conn = None
    cursor = None
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name
            FROM building
            WHERE state = 'ONGOING_OPERATING'
              AND LOWER(name) NOT LIKE '%%insite%%'
              AND name NOT LIKE '%%시연%%'
            ORDER BY id ASC
        """)
        
        rows = cursor.fetchall()
        return [
            {"id": int(row[0]), "name": str(row[1] or "").strip()}
            for row in rows if row and row[0] is not None
        ]
    
    except Exception as e:
        print(f"[WARN] fetch_active_buildings 실패: {e}")
        return []
    
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


def fetch_voc_counts_by_buildings(building_ids: list[int], start_date: str, end_date: str) -> dict[int, int]:
    """
    여러 빌딩의 VOC 건수를 한 번에 조회합니다.

    Args:
        building_ids: 빌딩 ID 목록
        start_date: 시작일 (inclusive)
        end_date: 종료일 (exclusive)

    Returns:
        dict[int, int]: {building_id: voc_count, ...}
        조회되지 않은 빌딩은 0으로 채워서 반환
    """
    if not building_ids:
        return {}

    conn = None
    cursor = None
    try:
        conn = db_connect()
        cursor = conn.cursor()

        # IN 절을 위한 플레이스홀더 생성
        placeholders = ",".join(["%s"] * len(building_ids))

        query = f"""
            SELECT building_id, COUNT(*) as cnt
            FROM voc
            WHERE building_id IN ({placeholders})
              AND title NOT LIKE '%%테스트%%'
              AND voc_date >= %s
              AND voc_date < %s
            GROUP BY building_id
        """

        params = list(building_ids) + [start_date, end_date]
        cursor.execute(query, params)

        rows = cursor.fetchall()
        result = {int(row[0]): int(row[1]) for row in rows if row}

        # 조회되지 않은 빌딩은 0으로 채움
        for bid in building_ids:
            if bid not in result:
                result[bid] = 0

        return result

    except Exception as e:
        print(f"[WARN] fetch_voc_counts_by_buildings 실패: {e}")
        # 실패 시 모든 빌딩 0으로 반환 (배치 안정성)
        return {bid: 0 for bid in building_ids}

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