# /home/ssm-user/jupyter/batch/keyword_analysis.py
import os
import re
import io
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

from common_db import load_dotenv, load_text_dict_from_db, db_connect
from nlp_model_core import strip_html, TOKEN_RE, build_compound_replacers, apply_compound

from report_html import build_report_context, render_report_html, save_html

KST = timezone(timedelta(hours=9))


# -----------------------------
# 노트북 스타일/폰트 강제
# -----------------------------
def _apply_notebook_style():
    # rc 초기화 후 seaborn 스타일 적용 (노트북 톤)
    mpl.rcdefaults()

    import seaborn as sns
    sns.set(style="whitegrid", palette="deep")

    # 노트북에서 보이던 약간 푸른 축 배경
    mpl.rcParams["axes.facecolor"] = "#EAEAF2"
    mpl.rcParams["axes.unicode_minus"] = False

    # 한글 폰트 강제
    font_path = "/usr/share/fonts/nanum/NanumGothic-Regular.ttf"
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            mpl.rcParams["font.family"] = font_name
        except Exception:
            pass


# -----------------------------
# HTML 유틸
# -----------------------------
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

# -----------------------------
# 층 마스터(all_floors) 로딩/정규화 (중간층은 name 그대로)
# -----------------------------
def normalize_floor_label_basic(x: str) -> str:
    s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
    s = s.strip().upper().replace(" ", "")
    if not s:
        return ""

    # 옥탑/PH
    if s.startswith("PH") or "옥탑" in s or s.startswith("ROOF") or s.startswith("RF"):
        m = re.findall(r"\d+", s)
        if m:
            return f"PH{m[0]}F"
        return "PH"

    # 지하
    if s.startswith("B") or "지하" in s:
        m = re.findall(r"\d+", s)
        return f"B{m[0]}F" if m else ""

    # 일반층
    m = re.findall(r"\d+", s)
    if not m:
        return ""
    return f"{int(m[0])}F"


def normalize_raw(x: str) -> str:
    s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def fetch_building_name(building_id: int) -> str:
    """
    building_id로 building 테이블에서 현장명(name) 조회
    실패/없음이면 빈 문자열 반환
    """
    sql = """
    SELECT name
    FROM building
    WHERE id = %(building_id)s
    LIMIT 1
    """
    conn = db_connect()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, {"building_id": building_id})
        row = cur.fetchone()
        name = "" if not row else str(row[0] or "").strip()
        return name
    except Exception:
        return ""
    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass
        conn.close()


def fetch_building_floor_master_df(building_id: int) -> pd.DataFrame:
    sql = """
    SELECT
      building_id,
      floor_type,
      name,
      floor,
      middle_floor
    FROM building_floor
    WHERE building_id = %(building_id)s
    ORDER BY floor ASC, middle_floor ASC, name ASC
    """
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, {"building_id": building_id})
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        df_floor = pd.DataFrame(rows, columns=cols)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

    if df_floor is None or df_floor.empty:
        return pd.DataFrame(columns=["floor", "floor_label", "floor_norm", "name_raw", "middle_floor"])

    df_floor["name"] = df_floor["name"].fillna("").astype(str).str.strip()
    df_floor["middle_floor"] = df_floor["middle_floor"].fillna(False).astype(bool)

    mask_normal = ~df_floor["middle_floor"]

    df_floor["floor_norm"] = ""
    df_floor.loc[mask_normal, "floor_norm"] = df_floor.loc[mask_normal, "name"].map(normalize_floor_label_basic)

    df_floor["floor_label"] = df_floor["name"]
    df_floor.loc[mask_normal, "floor_label"] = df_floor.loc[mask_normal, "floor_norm"]

    df_floor["name_raw"] = df_floor["name"].map(normalize_raw)

    df_floor["floor"] = pd.to_numeric(df_floor["floor"], errors="coerce")
    df_floor = df_floor[df_floor["floor"].notna()].copy()
    df_floor["floor"] = df_floor["floor"].astype(int)

    df_floor = df_floor[df_floor["floor_label"].astype(str).str.strip() != ""].copy()
    df_floor = df_floor.drop_duplicates(subset=["floor_label"], keep="first")

    return df_floor[
        ["floor", "floor_type", "floor_label", "floor_norm", "name_raw", "middle_floor"]
    ].sort_values(["floor", "middle_floor", "floor_label"], ascending=[True, True, True]).reset_index(drop=True)


def map_voc_floor_to_master_label(df_voc: pd.DataFrame, floor_master: pd.DataFrame) -> pd.Series:
    """
    VOC 층명(building_floor_name)을 floor_master의 floor_label로 매핑합니다.
    - 일반층: 정규화키(voc_norm) == floor_norm
    - 중간층: 포함 관계 기반 보조 매칭 (name_raw)
    - 미매칭: 원문 그대로 (공백만 정리)
    """
    voc_raw = df_voc["building_floor_name"].map(normalize_raw)
    voc_norm = df_voc["building_floor_name"].map(normalize_floor_label_basic)

    normal = floor_master[floor_master["floor_norm"].astype(str).str.strip() != ""]
    norm_map = dict(zip(normal["floor_norm"], normal["floor_label"]))

    mapped = voc_norm.map(norm_map)

    middle = floor_master[floor_master["middle_floor"]].copy()
    if not middle.empty:
        # 긴 키부터 매칭 (충돌 방지) - 버그 수정: sort_values에 by 파라미터 사용
        middle = middle.assign(_len=middle["name_raw"].str.len()).sort_values("_len", ascending=False).drop(columns=["_len"])
        for _, r in middle.iterrows():
            key = r["name_raw"]
            label = r["floor_label"]
            if not key:
                continue
            # 너무 짧은 키는 오탐 가능성이 커서 제외
            if len(key) < 2:
                continue
            hit = mapped.isna() & voc_raw.str.contains(re.escape(key), na=False)
            mapped.loc[hit] = label

    mapped = mapped.fillna(df_voc["building_floor_name"].fillna("").astype(str).str.strip())
    mapped = mapped.replace({"": "미상"})
    return mapped


def build_location_counts_all(
    df_voc: pd.DataFrame,
    floor_master: pd.DataFrame,
    min_count: int = 2,
) -> pd.Series:
    """
    floor_master 축(중간층 포함)에 맞춰 누락층을 0으로 채운 시리즈를 만듭니다.
    노트북 코드 기준으로 min_count=2(건수 1 초과) 필터링을 적용합니다.
    """
    all_labels = floor_master["floor_label"].tolist()

    df_voc = df_voc.copy()
    df_voc["floor_label"] = map_voc_floor_to_master_label(df_voc, floor_master)

    counts = df_voc["floor_label"].value_counts()
    counts = counts[counts >= min_count]

    counts_all = pd.Series(0, index=all_labels, dtype=int)
    counts_all.update(counts)
    return counts_all


# -----------------------------
# 키워드 토큰화 (태깅 아님)
# -----------------------------
def remove_stopwords_tokens(tokens: List[str], stop_words: set) -> List[str]:
    if not stop_words:
        return tokens
    return [t for t in tokens if t not in stop_words]


def keyword_tokenize_series(
    texts: pd.Series,
    compound_words: Dict[str, str],
    stop_words: set,
) -> List[str]:
    replacers = build_compound_replacers(compound_words)

    use_komoran = False
    komoran = None
    try:
        from konlpy.tag import Komoran  # type: ignore
        komoran = Komoran()
        use_komoran = True
    except Exception:
        use_komoran = False

    tokens_all: List[str] = []
    for raw in texts.fillna("").astype(str).tolist():
        t = strip_html(raw)
        t = apply_compound(t, replacers)
        t = t.lower().strip()
        if not t:
            continue

        if use_komoran and komoran is not None:
            try:
                toks = komoran.nouns(t)
            except Exception:
                toks = TOKEN_RE.findall(t)
        else:
            toks = TOKEN_RE.findall(t)

        toks = [x.strip() for x in toks if x and x.strip()]
        toks = remove_stopwords_tokens(toks, stop_words)
        tokens_all.extend(toks)

    return tokens_all


# -----------------------------
# 시각화 생성
# -----------------------------
def plot_bar_daily_counts(df: pd.DataFrame, date_col: str, title: str):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d[d[date_col].notna()]
    if d.empty:
        fig = plt.figure(figsize=(14, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    start = d[date_col].min().normalize()
    end = d[date_col].max().normalize()
    all_days = pd.date_range(start=start, end=end, freq="D")

    daily_counts = d[date_col].dt.normalize().value_counts().sort_index()
    daily_counts = daily_counts.reindex(all_days, fill_value=0)

    # 노트북과 동일하게 YYYY-MM-DD
    daily_counts.index = daily_counts.index.strftime("%Y-%m-%d")

    fig = plt.figure(figsize=(14, 6))
    bars = plt.bar(daily_counts.index, daily_counts.values)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.title(title)
    plt.xlabel("날짜")
    plt.ylabel("건수")
    plt.xticks(rotation=90)
    plt.tight_layout()
    return fig


def plot_floor_counts_from_series(counts_all: pd.Series, title: str, show_zeros: bool = True, no_floor_master: bool = False):
    """
    노트북 '층별 VOC 발생 건수' 스타일:
    - figsize (14, 8)
    - barh, color='skyblue'
    - 라벨은 막대 오른쪽에 표시
    - 중간층 포함 축은 counts_all.index에 의해 결정
    """
    # 층 마스터가 없는 경우
    if no_floor_master:
        fig = plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "층 정보가 존재하지 않습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    vc = counts_all.copy()
    if not show_zeros:
        vc = vc[vc > 0]

    if vc.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()
    bars = ax.barh(vc.index.astype(str), vc.values, color="skyblue")

    for b in bars:
        w = b.get_width()
        ax.text(w, b.get_y() + b.get_height() / 2, f"{int(w)}", ha="left", va="center", fontsize=9, color="black")

    plt.title(title)
    plt.xlabel("건수")
    plt.ylabel("발생위치")
    plt.tight_layout()
    return fig


def fetch_building_user_group_names(building_id: int) -> List[str]:

    conn = db_connect()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
                """
                SELECT name
                FROM building_user_group
                WHERE building_id = %(building_id)s
                ORDER BY name ASC
                """,
                {"building_id": building_id},
            )

        rows = cur.fetchall()
        names = []
        for r in rows:
            n = "" if r is None else str(r[0] or "").strip()
            if n and n.lower() != "none":
                names.append(n)
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass
        conn.close()


def plot_team_processed_counts(df: pd.DataFrame, team_master: Optional[List[str]] = None):
    d = df.copy()
    d["reply_write_date"] = pd.to_datetime(d.get("reply_write_date"), errors="coerce")

    ug = d.get("user_group_name", pd.Series([""] * len(d))).fillna("").astype(str)
    mask = d["reply_write_date"].notna() & (ug.str.strip().str.lower() != "none") & (ug.str.strip() != "")
    d = d[mask].copy()

    vc = d["user_group_name"].astype(str).value_counts()

    if team_master:
        vc = vc.reindex(team_master, fill_value=0)
        vc = vc.sort_values(ascending=False, kind="mergesort")

    if vc.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.title("팀별 VOC 처리 건수 (답변 작성 건수)")
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    fig = plt.figure(figsize=(14, 6))
    ax = vc.plot(kind="bar")
    plt.title("팀별 VOC 처리 건수 (답변 작성 건수)")
    plt.xlabel("담당팀")
    plt.ylabel("건수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(p.get_x() + p.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color="black"
        )

    return fig


def plot_weekday_counts(df: pd.DataFrame):
    d = df.copy()
    d["write_date"] = pd.to_datetime(d.get("write_date"), errors="coerce")
    d = d[d["write_date"].notna()]
    if d.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.title("요일별 VOC 발생 건수")
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    # 1) 요일 컬럼 추가 (0=월, 6=일)
    d["요일"] = d["write_date"].dt.dayofweek

    # 2) 요일 이름 매핑
    weekday_map = {0: "월요일", 1: "화요일", 2: "수요일", 3: "목요일", 4: "금요일", 5: "토요일", 6: "일요일"}
    d["요일"] = d["요일"].map(weekday_map)

    # 3) 요일 순서 정의
    weekday_order = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

    # 4) groupby + size + reindex
    weekday_counts = d.groupby("요일").size().reindex(weekday_order, fill_value=0)

    # 5) 인덱스를 CategoricalIndex로 강제
    weekday_counts.index = pd.CategoricalIndex(
        weekday_counts.index,
        categories=weekday_order,
        ordered=True,
    )

    # 6) 그래프
    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()
    bars = ax.barh(weekday_counts.index.astype(str), weekday_counts.values, color="skyblue")

    # 7) 라벨
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
        )

    # 8) y축 뒤집기 (월 -> 일)
    ax.invert_yaxis()

    plt.title("요일별 VOC 발생 건수")
    plt.xlabel("건수")
    plt.ylabel("요일")
    plt.tight_layout()
    return fig


def compute_processing_time(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["write_date"] = pd.to_datetime(d["write_date"], errors="coerce")
    d["reply_write_date"] = pd.to_datetime(d["reply_write_date"], errors="coerce")
    d["처리소요시간"] = d["reply_write_date"] - d["write_date"]
    return d


def format_timedelta(td) -> str:
    if td is None or pd.isna(td):
        return ""
    if not isinstance(td, pd.Timedelta):
        try:
            td = pd.to_timedelta(td)
        except Exception:
            return ""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    days = total_seconds // 86400
    rem = total_seconds % 86400
    hours = rem // 3600
    rem2 = rem % 3600
    mins = rem2 // 60
    if days > 0:
        return f"{days}일 {hours}시간 {mins}분"
    if hours > 0:
        return f"{hours}시간 {mins}분"
    return f"{mins}분"


def table_processing_top5(df: pd.DataFrame) -> pd.DataFrame:
    d = compute_processing_time(df)
    d = d[d["write_date"].notna() & d["reply_write_date"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["VOC제목", "VOC내용", "답변", "접수일", "답변완료시간", "처리소요시간"])

    d["처리소요시간_분"] = d["처리소요시간"].dt.total_seconds() / 60.0
    d = d.sort_values("처리소요시간_분", ascending=False).head(5)

    out = pd.DataFrame({
        "VOC제목": d.get("title", "").map(strip_html),
        "VOC내용": d.get("request_contents", "").map(strip_html),
        "답변": d.get("reply", "").map(strip_html),
        "접수일": d["write_date"].dt.strftime("%Y-%m-%d %H:%M"),
        "답변완료시간": d["reply_write_date"].dt.strftime("%Y-%m-%d %H:%M"),
        "처리소요시간": d["처리소요시간"].map(format_timedelta),
    })
    return out


def plot_daily_avg_processing_minutes(df: pd.DataFrame):
    d = df.copy()
    d["write_date"] = pd.to_datetime(d.get("write_date"), errors="coerce")
    d["reply_write_date"] = pd.to_datetime(d.get("reply_write_date"), errors="coerce")

    # 처리소요시간 계산
    d["처리소요시간"] = d["reply_write_date"] - d["write_date"]

    # 처리소요시간이 있는 것만 (노트북과 동일한 의미)
    d = d[d["write_date"].notna() & d["reply_write_date"].notna()].copy()
    if d.empty:
        fig = plt.figure(figsize=(14, 6))
        plt.title("일별 평균 처리 소요 시간")
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    # 일자 단위
    d["접수일"] = d["write_date"].dt.date
    daily_processing_time = d.groupby("접수일")["처리소요시간"].mean().sort_index()

    # 전체 일자 범위 생성 + 누락일 0으로
    start_date = d["write_date"].min().date()
    end_date = d["write_date"].max().date()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    daily_processing_time = daily_processing_time.reindex(all_dates.date, fill_value=pd.Timedelta(0))
    processing_time_minutes = daily_processing_time.apply(lambda x: x.total_seconds() / 60.0)

    # x축 문자열
    x = [str(dt) for dt in daily_processing_time.index]
    y = processing_time_minutes.values

    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # 노트북처럼: skyblue 선 + 마커
    ax.plot(x, y, marker="o", linestyle="-", color="skyblue")

    # 전체 평균선(빨간 점선)
    avg = float(np.mean(y)) if len(y) > 0 else 0.0
    ax.axhline(avg, color="red", linestyle="--", label=f"평균 처리 소요 시간: {avg:.1f}분")

    plt.title("일별 평균 처리 소요 시간")
    plt.xlabel("일자")
    plt.ylabel("평균 처리 소요 시간 (분)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_heatmap_topic_work(
    df: pd.DataFrame,
    title: str,
    topic_col: str,
    work_col: str,
    cmap: str = "Blues",
):
    import seaborn as sns

    d = df.copy()
    t = d.get(topic_col, pd.Series([""] * len(d))).fillna("").astype(str)
    w = d.get(work_col, pd.Series([""] * len(d))).fillna("").astype(str)
    d = d[(t.str.strip() != "") & (w.str.strip() != "")].copy()
    if d.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    pivot = (
        d.groupby([topic_col, work_col])
         .size()
         .unstack(fill_value=0)
    )
    if pivot.to_numpy().sum() == 0:
        fig = plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        return fig

    # 노트북: 전체(Blues) figsize=(14,8)
    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        ax=ax,
        cbar=True,
    )

    ax.set_title(title)
    ax.set_ylabel(topic_col)
    ax.set_xlabel(work_col)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


# -----------------------------
# main
# -----------------------------
def main():
    env_path = os.getenv("ENV_PATH", "/home/ssm-user/jupyter/.env")
    load_dotenv(env_path)

    building_id = int(os.getenv("BUILDING_ID", "95"))
    start_date = os.getenv("START_DATE", "2025-12-01")
    end_date = os.getenv("END_DATE", "2026-01-01")
    run_id = os.getenv("RUN_ID", datetime.now(KST).strftime("%Y%m%dT%H%M%S"))

    base_dir = os.getenv("BASE_DIR", "/home/ssm-user/jupyter")
    yyyymm = start_date.replace("-", "")[:6]

    # OUT_DIR 루트 기준 하위 3개 폴더 사용
    out_root = os.getenv("OUT_DIR", f"{base_dir}/output")
    tagging_dir = os.getenv("TAGGING_DIR", f"{out_root}/tagging")
    html_dir = os.getenv("HTML_DIR", f"{out_root}/html")
    os.makedirs(html_dir, exist_ok=True)

    default_csv = f"{tagging_dir}/tagged_{building_id}_{yyyymm}_{run_id}.csv"
    tagged_csv_path = os.getenv("TAGGED_CSV_PATH", default_csv)

    out_html_path = os.getenv(
        "OUT_HTML_PATH",
        f"{html_dir}/dashboard_{building_id}_{yyyymm}_{run_id}.html",
    )

    print(f"[INFO] BUILDING_ID={building_id} START_DATE={start_date} END_DATE={end_date} RUN_ID={run_id}")
    print(f"[INFO] TAGGED_CSV_PATH={tagged_csv_path}")
    print(f"[INFO] OUT_HTML_PATH={out_html_path}")

    # CSV 파일 체크 - 없으면 빈 DataFrame으로 진행
    if not os.path.exists(tagged_csv_path):
        print(f"[WARN] CSV 파일 없음: {tagged_csv_path} - 빈 데이터로 보고서 생성")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(tagged_csv_path, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

    for c in ["title", "request_contents", "reply", "building_floor_name", "zone_name", "user_group_name", "reply_writer"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    for c in ["write_date", "voc_date", "reply_write_date", "tagged_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # write_date가 없다면 voc_date를 write_date로 사용
    if "write_date" not in df.columns or df["write_date"].isna().all():
        if "voc_date" in df.columns:
            df["write_date"] = pd.to_datetime(df["voc_date"], errors="coerce")
        else:
            df["write_date"] = pd.NaT

    # 날짜 범위 표
    date_range = {}
    for col in ["write_date", "reply_write_date"]:
        valid = df[col].dropna() if col in df.columns else pd.Series([], dtype="datetime64[ns]")
        if not valid.empty:
            date_range[col] = (valid.min(), valid.max())
        else:
            date_range[col] = (None, None)

    date_range_df = pd.DataFrame(date_range, index=["최초일시", "최종일시"]).rename(
        columns={"write_date": "발생일시", "reply_write_date": "처리일시"}
    )

    figs: Dict[str, str] = {}

    # 1) 일별 VOC 발생 건수
    figs["daily_counts"] = fig_to_base64(plot_bar_daily_counts(df, "write_date", "일별 VOC 발생 건수"))

    # 2) 층별 VOC 발생 건수 (DB 층 마스터 기반, 중간층 포함)
    try:
        floor_master = fetch_building_floor_master_df(building_id)
        if floor_master.empty:
            # 층 마스터가 없는 경우
            figs["floor_counts"] = fig_to_base64(plot_floor_counts_from_series(pd.Series(dtype=int), "층별 VOC 발생 건수", no_floor_master=True))
        else:
            counts_all = build_location_counts_all(df, floor_master, min_count=2)
            figs["floor_counts"] = fig_to_base64(plot_floor_counts_from_series(counts_all, "층별 VOC 발생 건수", show_zeros=True))
    except Exception as e:
        fig = plt.figure(figsize=(12, 4))
        plt.title("층별 VOC 발생 건수")
        plt.text(0.5, 0.5, f"층 마스터 로딩/집계 실패: {e}", ha="center", va="center", fontsize=10, color="#666")
        plt.axis("off")
        figs["floor_counts"] = fig_to_base64(fig)

    # 3) 팀별 처리 건수 / 4) 요일별 발생 건수
    try:
        team_master = fetch_building_user_group_names(building_id)
    except Exception:
        team_master = None
    figs["team_counts"] = fig_to_base64(plot_team_processed_counts(df, team_master=team_master))

    figs["weekday_counts"] = fig_to_base64(plot_weekday_counts(df))

    # 처리 소요 시간
    top5_df = table_processing_top5(df)
    figs["avg_processing"] = fig_to_base64(plot_daily_avg_processing_minutes(df))

    # 키워드 분석
    compound_words, stop_words = load_text_dict_from_db()

    df["keyword_text"] = (
        df["request_contents"].map(strip_html).fillna("") + " " +
        df["title"].map(strip_html).fillna("") + " " +
        df["reply"].map(strip_html).fillna("")
    ).str.strip()

    tokens = keyword_tokenize_series(df["keyword_text"], compound_words=compound_words, stop_words=stop_words)

    kw_imgs: List[str] = []
    if tokens:
        from collections import Counter
        c = Counter(tokens)
        top20 = c.most_common(20)
        kw, cnt = zip(*top20) if top20 else ([], [])
    
        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()
    
        bars = ax.bar(list(kw), list(cnt), color="salmon")

        ax.set_title("VOC 키워드 빈도 상위 20")
        ax.set_xticks(range(len(kw)))
        ax.set_xticklabels(list(kw), rotation=45, ha="right")
        ax.tick_params(axis="x", labelsize=9)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        kw_imgs.append(fig_to_base64(fig))
    else:
        fig = plt.figure(figsize=(12, 4))
        plt.title("VOC 키워드 빈도 상위 20")
        plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        kw_imgs.append(fig_to_base64(fig))

    # 워드클라우드
    wc_img = ""
    try:
        from wordcloud import WordCloud
        from collections import Counter

        if tokens:
            freq = Counter(tokens)
            wc = WordCloud(
                font_path=os.getenv("WORDCLOUD_FONT_PATH", "/usr/share/fonts/nanum/NanumGothic-Regular.ttf"),
                width=900,
                height=450,
                background_color="white",
            ).generate_from_frequencies(freq)

            fig = plt.figure(figsize=(14, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title("VOC 워드클라우드")
            plt.tight_layout()
            wc_img = fig_to_base64(fig)
        else:
            fig = plt.figure(figsize=(12, 4))
            plt.title("VOC 워드클라우드")
            plt.text(0.5, 0.5, "데이터가 없습니다.", ha="center", va="center", fontsize=12, color="#666")
            plt.axis("off")
            wc_img = fig_to_base64(fig)
    except Exception:
        fig = plt.figure(figsize=(12, 4))
        plt.title("VOC 워드클라우드")
        plt.text(0.5, 0.5, "wordcloud 패키지가 없거나 생성에 실패했습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        wc_img = fig_to_base64(fig)

    # 주제분석(태깅된 컬럼만 사용) - 노트북 방식(주제: 대-중 결합, 작업유형: 중분류만)
    hm_imgs: List[str] = []

    required_cols = ["주제 대분류", "주제 중분류", "작업유형 중분류"]
    if all(c in df.columns for c in required_cols):
        def combine_major_minor(major, minor):
            a = "" if pd.isna(major) else str(major).strip()
            b = "" if pd.isna(minor) else str(minor).strip()
            if a and b:
                return f"{a}-{b}"
            return a or b or "미분류"

        d_hm = df.copy()
        d_hm["주제"] = [combine_major_minor(mj, mn) for mj, mn in zip(d_hm["주제 대분류"], d_hm["주제 중분류"])]
        d_hm["작업유형"] = [combine_major_minor(None, mn) for mn in d_hm["작업유형 중분류"]]

        hm_imgs.append(fig_to_base64(plot_heatmap_topic_work(d_hm, "주제별 작업유형 분포", "주제", "작업유형", cmap="Blues")))

        # 팀별 히트맵(노트북은 전체 teams 순회였으나, 배치 안정성 위해 상위 팀만)
        team_col = "user_group_name"
        teams = d_hm[team_col].fillna("").astype(str).str.strip()
        teams = teams[(teams != "") & (teams.str.lower() != "none")]
        top_teams = teams.value_counts().head(10).index.tolist()

        for team in top_teams:
            df_team = d_hm[d_hm[team_col].fillna("").astype(str).str.strip() == team].copy()
            if df_team.empty:
                continue
            pivot = df_team.groupby(["주제", "작업유형"]).size().unstack(fill_value=0)
            if pivot.to_numpy().sum() == 0:
                continue
            hm_imgs.append(fig_to_base64(plot_heatmap_topic_work(df_team, f"[{team}] 주제별 작업유형 분포", "주제", "작업유형", cmap="Oranges")))
    else:
        fig = plt.figure(figsize=(12, 4))
        plt.title("주제분석")
        plt.text(0.5, 0.5, "태깅 컬럼(주제 대분류/주제 중분류/작업유형 중분류)이 없습니다.", ha="center", va="center", fontsize=12, color="#666")
        plt.axis("off")
        hm_imgs.append(fig_to_base64(fig))

    now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    # ---- 리포트 헤더 데이터 ----
    building_name = fetch_building_name(building_id) or f"BUILDING_ID={building_id}"

    ctx = build_report_context(
        building_id=building_id,
        building_name=building_name,
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        now_kst=now_kst,
        date_range_df=date_range_df,
        figs=figs,
        top5_df=top5_df,
        kw_imgs=kw_imgs,
        wc_img=wc_img,
        hm_imgs=hm_imgs,
    )

    final_html = render_report_html(ctx)
    save_html(final_html, out_html_path)
    print(f"[INFO] saved html: {out_html_path}")


if __name__ == "__main__":
    _apply_notebook_style()
    main()