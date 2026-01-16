# /home/ssm-user/jupyter/batch/nlp_model.py
# -*- coding: utf-8 -*-
from common_db import load_dotenv, db_connect, load_text_dict_from_db
from nlp_model_core import (
    whitelist_text,
    build_compound_replacers,
    build_taxonomy_db,
    find_tax_etc,
    classify_weighted_general,
)

import os
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

import pandas as pd

KST = timezone(timedelta(hours=9))


# =========================================================
# 1) Result1 조회 (VOC + 최신 reply + reply_writer)
# =========================================================
RESULT1_SQL = """
SELECT
    v.id AS voc_id,
    v.building_id,
    v.voc_date,
    v.title,
    v.request_contents,

    bf.name AS building_floor_name,
    z.name AS zone_name,

    r.reply,
    r.reply_write_date,
    bug.name AS user_group_name,
    a.name AS reply_writer

FROM voc v
LEFT JOIN building_floor bf
  ON v.building_floor_id = bf.id
LEFT JOIN building_floor_zone z
  ON v.building_floor_zone_id = z.id
LEFT JOIN (
    SELECT DISTINCT ON (voc_id) *
    FROM voc_reply
    ORDER BY voc_id, reply_write_date DESC
) r
  ON v.id = r.voc_id
LEFT JOIN account_group ag
  ON r.reply_writer_id = ag.account_id
LEFT JOIN building_user_group bug
  ON ag.group_id = bug.id
LEFT JOIN account a
  ON r.reply_writer_id = a.id
WHERE v.building_id = %(building_id)s
  AND v.title NOT LIKE '%%테스트%%'
  AND v.voc_date >= %(start_date)s
  AND v.voc_date <  %(end_date)s
ORDER BY v.voc_date ASC, v.id ASC;
"""


def fetch_result1_df(building_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    conn = db_connect()
    try:
        df = pd.read_sql_query(
            RESULT1_SQL,
            conn,
            params={"building_id": building_id, "start_date": start_date, "end_date": end_date},
        )
    finally:
        conn.close()

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "voc_id",
                "building_id",
                "voc_date",
                "title",
                "request_contents",
                "building_floor_name",
                "zone_name",
                "reply",
                "reply_write_date",
                "user_group_name",
                "reply_writer",
            ]
        )

    fill_cols = [
        "title",
        "request_contents",
        "reply",
        "reply_writer",
        "building_floor_name",
        "zone_name",
        "user_group_name",
    ]
    for c in fill_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    return df


# =========================================================
# 2) taxonomy 조회 (voc_taxonomy: major/minor/keywords)
# =========================================================
def fetch_taxonomy_dfs(version: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = db_connect()
    try:
        if version:
            tax_df = pd.read_sql_query(
                """
                SELECT taxonomy_type, major, minor, keywords, priority
                FROM voc_taxonomy
                WHERE is_active = true
                  AND version = %(version)s
                """,
                conn,
                params={"version": version},
            )
        else:
            tax_df = pd.read_sql_query(
                """
                SELECT taxonomy_type, major, minor, keywords, priority
                FROM voc_taxonomy
                WHERE is_active = true
                """,
                conn,
            )
    finally:
        conn.close()

    if tax_df is None or tax_df.empty:
        tax_df = pd.DataFrame(columns=["taxonomy_type", "major", "minor", "keywords", "priority"])

    for c in ["taxonomy_type", "major", "minor", "keywords"]:
        if c not in tax_df.columns:
            tax_df[c] = ""
        tax_df[c] = tax_df[c].fillna("").astype(str)

    subj = tax_df[tax_df["taxonomy_type"] == "SUBJECT"].copy()
    work = tax_df[tax_df["taxonomy_type"] == "WORK"].copy()

    subj = subj[(subj["major"] != "") & (subj["minor"] != "") & (subj["keywords"] != "")]
    work = work[(work["major"] != "") & (work["minor"] != "") & (work["keywords"] != "")]

    return subj, work


# =========================================================
# 3) 주제/작업유형 태깅 (방어 로직 포함)
# =========================================================
def run_retagging_df(
    result_df: pd.DataFrame,
    subject_tax: pd.DataFrame,
    work_tax: pd.DataFrame,
    pos_max: float = 1.5,
    pos_min: float = 1.0,
    title_boost: float = 2.0,
    title_overlap_ratio: float = 0.3,
    short_len: int = 5,
    long_len: int = 60,
    short_penalty: float = 0.9,
    long_penalty: float = 0.9,
    decay_steps: int = 5,
    length_bonus_scale: float = 0.1,
    compound_words: dict | None = None,
    stop_words: set | None = None,
) -> pd.DataFrame:
    compound_words = compound_words or {}
    stop_words = stop_words or set()
    compound_replacers = build_compound_replacers(compound_words)

    if result_df is None:
        result_df = pd.DataFrame()
    result_df = result_df.copy()

    # 필수 컬럼 방어
    for col in ["title", "request_contents", "reply"]:
        if col not in result_df.columns:
            result_df[col] = ""
        result_df[col] = result_df[col].fillna("").astype(str)

    # 빈 DF 방어
    if result_df.empty:
        result_df["title_clean"] = ""
        result_df["all_text"] = ""
        result_df["주제 대분류"] = ""
        result_df["주제 중분류"] = ""
        result_df["작업유형 대분류"] = ""
        result_df["작업유형 중분류"] = ""
        result_df["주제_score"] = 0.0
        result_df["작업유형_score"] = 0.0
        return result_df

    # all_text 구성
    result_df.loc[:, "title_clean"] = result_df["title"].map(whitelist_text).fillna("")
    result_df.loc[:, "all_text"] = (
        result_df["request_contents"].map(whitelist_text).fillna("")
        + " "
        + result_df["title_clean"]
        + " "
        + result_df["reply"].map(whitelist_text).fillna("")
    ).str.strip()

    subj_empty = (subject_tax is None) or subject_tax.empty
    work_empty = (work_tax is None) or work_tax.empty

    subject_items = []
    work_items = []
    subject_etc = ("기타", "기타")
    work_etc = ("기타", "기타")

    if not subj_empty:
        subject_items = build_taxonomy_db(subject_tax, compound_replacers, stop_words)
        found = find_tax_etc(subject_items)
        if found:
            subject_etc = found
    if not work_empty:
        work_items = build_taxonomy_db(work_tax, compound_replacers, stop_words)
        found = find_tax_etc(work_items)
        if found:
            work_etc = found

    # items가 실제로 비어버린 경우도 방어
    if not subject_items:
        subj_empty = True
    if not work_items:
        work_empty = True

    if subj_empty and work_empty:
        result_df.loc[:, "주제 대분류"] = subject_etc[0]
        result_df.loc[:, "주제 중분류"] = subject_etc[1]
        result_df.loc[:, "작업유형 대분류"] = work_etc[0]
        result_df.loc[:, "작업유형 중분류"] = work_etc[1]
        result_df.loc[:, "주제_score"] = 0.0
        result_df.loc[:, "작업유형_score"] = 0.0
        return result_df

    sub_major, sub_minor, sub_score = [], [], []
    work_major, work_minor, work_score = [], [], []

    for _, row in result_df.iterrows():
        all_text = row.get("all_text", "") or ""
        title_text = row.get("title_clean", row.get("title", "")) or ""

        # subject
        if subj_empty:
            mj, mn, sc = subject_etc[0], subject_etc[1], 0.0
        else:
            mj, mn, sc = classify_weighted_general(
                all_text,
                subject_items,
                title_text=title_text,
                pos_max=pos_max,
                pos_min=pos_min,
                title_boost=title_boost,
                title_overlap_ratio=title_overlap_ratio,
                short_len=short_len,
                long_len=long_len,
                short_penalty=short_penalty,
                long_penalty=long_penalty,
                decay_steps=decay_steps,
                length_bonus_scale=length_bonus_scale,
                compound_replacers=compound_replacers,
                stop_words=stop_words,
            )
            if not mj:
                mj, mn = subject_etc
        sub_major.append(mj)
        sub_minor.append(mn)
        sub_score.append(sc)

        # work
        if work_empty:
            mj2, mn2, sc2 = work_etc[0], work_etc[1], 0.0
        else:
            mj2, mn2, sc2 = classify_weighted_general(
                all_text,
                work_items,
                title_text=title_text,
                pos_max=pos_max,
                pos_min=pos_min,
                title_boost=title_boost,
                title_overlap_ratio=title_overlap_ratio,
                short_len=short_len,
                long_len=long_len,
                short_penalty=short_penalty,
                long_penalty=long_penalty,
                decay_steps=decay_steps,
                length_bonus_scale=length_bonus_scale,
                compound_replacers=compound_replacers,
                stop_words=stop_words,
            )
            if not mj2:
                mj2, mn2 = work_etc
        work_major.append(mj2)
        work_minor.append(mn2)
        work_score.append(sc2)

    result_df.loc[:, "주제 대분류"] = sub_major
    result_df.loc[:, "주제 중분류"] = sub_minor
    result_df.loc[:, "작업유형 대분류"] = work_major
    result_df.loc[:, "작업유형 중분류"] = work_minor
    result_df.loc[:, "주제_score"] = sub_score
    result_df.loc[:, "작업유형_score"] = work_score

    return result_df


# =========================================================
# 4) 경로 규칙 (output 아래: tagging / html / runs)
#    - .env의 OUT_DIR=/home/ssm-user/jupyter/output 은 "루트"로만 사용
#    - 실제 CSV는 output/tagging, 실행로그/메타는 output/runs
# =========================================================
def resolve_paths(base_dir: str, start_date: str, building_id: int, run_id: str) -> dict:
    out_root = os.getenv("OUT_DIR", f"{base_dir}/output")  # .env의 OUT_DIR은 루트 취급
    tagging_dir = os.getenv("TAGGING_DIR", f"{out_root}/tagging")
    html_dir = os.getenv("HTML_DIR", f"{out_root}/html")
    runs_dir = os.getenv("RUNS_DIR", f"{out_root}/runs")

    yyyymm = start_date.replace("-", "")[:6]

    out_csv_path = os.getenv(
        "OUT_CSV_PATH",
        f"{tagging_dir}/tagged_{building_id}_{yyyymm}_{run_id}.csv",
    )
    out_run_meta_path = os.getenv(
        "OUT_RUN_META_PATH",
        f"{runs_dir}/nlp_model_{building_id}_{yyyymm}_{run_id}.txt",
    )

    return {
        "out_root": out_root,
        "tagging_dir": tagging_dir,
        "html_dir": html_dir,
        "runs_dir": runs_dir,
        "yyyymm": yyyymm,
        "out_csv_path": out_csv_path,
        "out_run_meta_path": out_run_meta_path,
    }


# =========================================================
# 5) main
# =========================================================
def main():
    # 0) .env 로딩
    env_path = os.getenv("ENV_PATH", "/home/ssm-user/jupyter/.env")
    load_dotenv(env_path)

    # 1) 런타임 파라미터
    building_id = int(os.getenv("BUILDING_ID", "95"))
    start_date = os.getenv("START_DATE", "2025-12-01")
    end_date = os.getenv("END_DATE", "2026-01-01")  # exclusive
    run_id = os.getenv("RUN_ID", datetime.now(KST).strftime("%Y%m%dT%H%M%S"))

    base_dir = os.getenv("BASE_DIR", "/home/ssm-user/jupyter")

    paths = resolve_paths(base_dir=base_dir, start_date=start_date, building_id=building_id, run_id=run_id)
    os.makedirs(paths["tagging_dir"], exist_ok=True)
    os.makedirs(paths["runs_dir"], exist_ok=True)
    os.makedirs(paths["html_dir"], exist_ok=True)  # 다음 단계에서 html도 쓰므로 미리 생성

    print(f"[INFO] BUILDING_ID={building_id} START_DATE={start_date} END_DATE={end_date} RUN_ID={run_id}")
    print(f"[INFO] OUT_ROOT={paths['out_root']}")
    print(f"[INFO] TAGGING_DIR={paths['tagging_dir']}")
    print(f"[INFO] OUT_CSV_PATH={paths['out_csv_path']}")

    # 2) 사전 로드 (공통)
    compound_words, stop_words = load_text_dict_from_db()

    # 3) 원천 VOC 조회
    voc_df = fetch_result1_df(building_id, start_date, end_date)

    # 4) taxonomy 로드
    subject_tax_df, work_tax_df = fetch_taxonomy_dfs(version=os.getenv("TAXONOMY_VERSION") or None)

    print(f"[INFO] voc_df rows={len(voc_df)} cols={len(voc_df.columns)}")
    print(f"[INFO] subject_tax rows={len(subject_tax_df)} / work_tax rows={len(work_tax_df)}")

    # 5) 태깅
    tagged_df = run_retagging_df(
        result_df=voc_df,
        subject_tax=subject_tax_df,
        work_tax=work_tax_df,
        compound_words=compound_words,
        stop_words=stop_words,
    )

    tagged_df = tagged_df.copy()

    # 빈 DataFrame 처리
    if not tagged_df.empty:
        tagged_df.loc[:, "run_id"] = run_id
        tagged_df.loc[:, "tagged_at"] = datetime.now(KST).isoformat()
    else:
        tagged_df["run_id"] = []
        tagged_df["tagged_at"] = []

    # 6) 저장
    tagged_df.to_csv(paths["out_csv_path"], index=False, encoding="utf-8-sig")
    print(f"[INFO] saved csv: {paths['out_csv_path']}")

    # 7) runs 메타 로그 저장(배치 추적용)
    try:
        with open(paths["out_run_meta_path"], "w", encoding="utf-8") as f:
            f.write(f"created_at_kst={datetime.now(KST).isoformat()}\n")
            f.write(f"building_id={building_id}\n")
            f.write(f"start_date={start_date}\n")
            f.write(f"end_date={end_date}\n")
            f.write(f"run_id={run_id}\n")
            f.write(f"csv_path={paths['out_csv_path']}\n")
            f.write(f"voc_rows={len(voc_df)}\n")
            f.write(f"subject_tax_rows={len(subject_tax_df)}\n")
            f.write(f"work_tax_rows={len(work_tax_df)}\n")
        print(f"[INFO] saved run meta: {paths['out_run_meta_path']}")
    except Exception as e:
        print(f"[WARN] failed to write run meta: {e}")

    # 샘플 출력
    if not tagged_df.empty:
        print(tagged_df.head(3).to_string(index=False))
    else:
        print("[INFO] tagged_df is empty (but csv has been written).")


if __name__ == "__main__":
    main()