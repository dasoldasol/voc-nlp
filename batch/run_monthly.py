#!/usr/bin/env python3
# /home/ssm-user/jupyter/batch/run_monthly.py
# -*- coding: utf-8 -*-
"""
월간 VOC 분석 배치 오케스트레이터

3가지 실행 모드:
  - full: 태깅 → HTML → S3 → 검수시트 (기본)
  - refresh: 검수시트 → HTML 재생성 → S3 (검수 반영)
  - tagging-only: 태깅 → 검수시트만 (HTML 안 만듦)

S3 경로 구조:
  s3://hdcl-csp-prod/stat/voc/{yyyymm}/{building_id}/
    ├── tagged_{building_id}_{yyyymm}_{run_id}.csv
    └── dashboard_{building_id}_{yyyymm}_{run_id}.html

사용법:
  # [현재] 1단계: 자동태깅 + HTML + 검수시트 업로드
  python run_monthly.py --mode full --all-buildings --auto-month

  # [현재] 2단계: 검수 완료 후 HTML 재생성
  python run_monthly.py --mode refresh --all-buildings --auto-month

  # [미래] 학습된 모델로 전체 실행
  python run_monthly.py --mode full --all-buildings --auto-month
"""
from common_db import load_dotenv, fetch_active_buildings
from s3_uploader import get_s3_uploader
from gspread_manager import get_gspread_manager
import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd

# =========================================================
# 경로 설정
# =========================================================
DEFAULT_BASE_DIR = "/home/ssm-user/jupyter"
DEFAULT_ENV_PATH = "/home/ssm-user/jupyter/.env"
DEFAULT_LOG_DIR = "/home/ssm-user/jupyter/logs"

BATCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BATCH_DIR)

# 환경별 .env 파일 경로 매핑
ENV_PATHS = {
    "prd": "/home/ssm-user/jupyter/.env",
    "dev": os.path.join(PROJECT_ROOT, ".env.local"),
}
if BATCH_DIR not in sys.path:
    sys.path.insert(0, BATCH_DIR)

KST = timezone(timedelta(hours=9))

# 실행 모드
MODE_FULL = "full"              # 태깅 → HTML → S3 → 검수시트
MODE_REFRESH = "refresh"        # 검수시트 → HTML 재생성 → S3
MODE_TAGGING_ONLY = "tagging-only"  # 태깅 → 검수시트만


# =========================================================
# 로거 설정
# =========================================================
logger = logging.getLogger("run_monthly")


def setup_logger(log_dir: str, run_id: str, mode: str, single_building_id: int = None) -> str:
    """로거 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    if single_building_id:
        log_filename = f"run_monthly_{mode}_building-{single_building_id}_{run_id}.log"
    else:
        log_filename = f"run_monthly_{mode}_{run_id}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return log_path


def log_separator(char: str = "=", length: int = 60):
    logger.info(char * length)


# =========================================================
# 환경 로딩
# =========================================================
def load_env_config(env_path: str = None) -> dict:
    if env_path is None:
        env_path = os.getenv("ENV_PATH", DEFAULT_ENV_PATH)
    
    load_dotenv(env_path)
    
    return {
        "BASE_DIR": os.getenv("BASE_DIR", DEFAULT_BASE_DIR),
        "OUT_DIR": os.getenv("OUT_DIR", f"{DEFAULT_BASE_DIR}/output"),
        "LOG_DIR": os.getenv("LOG_DIR", DEFAULT_LOG_DIR),
        "BUILDING_ID": os.getenv("BUILDING_ID", ""),
        "START_DATE": os.getenv("START_DATE", ""),
        "END_DATE": os.getenv("END_DATE", ""),
        "TAXONOMY_VERSION": os.getenv("TAXONOMY_VERSION", ""),
        "ENV_PATH": env_path,
        "S3_BUCKET": os.getenv("S3_BUCKET", "hdcl-csp-prod"),
        "S3_PREFIX": os.getenv("S3_PREFIX", "stat/voc"),
    }


# =========================================================
# 날짜 유틸리티
# =========================================================
def get_previous_month(ref_date: Optional[datetime] = None) -> tuple[int, int]:
    if ref_date is None:
        ref_date = datetime.now(KST)
    
    year = ref_date.year
    month = ref_date.month - 1
    
    if month < 1:
        month = 12
        year -= 1
    
    return year, month


def get_month_date_range(year: int, month: int) -> tuple[str, str]:
    start_date = f"{year:04d}-{month:02d}-01"
    
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1
    
    end_date = f"{next_year:04d}-{next_month:02d}-01"
    return start_date, end_date


def generate_run_id(prefix: str = "auto") -> str:
    now = datetime.now(KST)
    return f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"


# =========================================================
# 스크립트 실행
# =========================================================
def run_script(
    script_name: str,
    env_vars: dict,
    dry_run: bool = False,
    timeout: int = 600,
) -> Dict:
    """batch 디렉토리 내 스크립트 실행"""
    
    script_path = os.path.join(BATCH_DIR, script_name)
    python_bin = sys.executable
    
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_vars.items() if v})
    
    if dry_run:
        logger.info(f"    [DRY-RUN] {script_name}")
        logger.debug(f"    실행: {python_bin} {script_path}")
        return {"success": True, "script": script_name, "returncode": 0, 
                "stdout": "[DRY-RUN]", "stderr": ""}
    
    try:
        logger.debug(f"    실행: {python_bin} {script_path}")
        
        result = subprocess.run(
            [python_bin, script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.debug(f"      {line}")
        
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.warning(f"      [stderr] {line}")
        
        return {
            "success": result.returncode == 0,
            "script": script_name,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    
    except subprocess.TimeoutExpired:
        logger.error(f"    타임아웃 ({timeout}초 초과)")
        return {"success": False, "script": script_name, "returncode": -1, 
                "error": f"타임아웃 ({timeout}초 초과)"}
    except Exception as e:
        logger.error(f"    예외: {e}")
        return {"success": False, "script": script_name, "returncode": -1, 
                "error": str(e)}


# =========================================================
# 태깅 실행 (nlp_model.py)
# =========================================================
def run_tagging(
    building_id: int,
    building_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    env_config: dict,
    dry_run: bool = False,
) -> Dict:
    """태깅 실행"""
    
    result = {
        "building_id": building_id,
        "building_name": building_name,
        "success": False,
        "error": None,
    }
    
    env_vars = {
        "BUILDING_ID": str(building_id),
        "START_DATE": start_date,
        "END_DATE": end_date,
        "RUN_ID": run_id,
        "ENV_PATH": env_config["ENV_PATH"],
        "BASE_DIR": env_config["BASE_DIR"],
        "OUT_DIR": env_config["OUT_DIR"],
        "TAXONOMY_VERSION": env_config.get("TAXONOMY_VERSION", ""),
    }
    
    logger.info(f"    nlp_model.py 실행...")
    tagging_result = run_script("nlp_model.py", env_vars, dry_run=dry_run)
    
    if not tagging_result["success"]:
        result["error"] = tagging_result.get("error") or tagging_result.get("stderr", "")[:200]
        logger.error(f"    [FAIL] 태깅 실패: {result['error']}")
        return result
    
    logger.info(f"    [OK] 태깅 완료")
    result["success"] = True
    return result


# =========================================================
# HTML 생성 (keyword_analysis.py)
# =========================================================
def run_analysis(
    building_id: int,
    building_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    env_config: dict,
    csv_path: str = None,
    dry_run: bool = False,
) -> Dict:
    """HTML 생성"""
    
    result = {
        "building_id": building_id,
        "building_name": building_name,
        "success": False,
        "error": None,
    }
    
    env_vars = {
        "BUILDING_ID": str(building_id),
        "START_DATE": start_date,
        "END_DATE": end_date,
        "RUN_ID": run_id,
        "ENV_PATH": env_config["ENV_PATH"],
        "BASE_DIR": env_config["BASE_DIR"],
        "OUT_DIR": env_config["OUT_DIR"],
    }
    
    # 특정 CSV 경로 지정 (검수 데이터 사용 시)
    if csv_path:
        env_vars["TAGGED_CSV_PATH"] = csv_path
    
    logger.info(f"    keyword_analysis.py 실행...")
    analysis_result = run_script("keyword_analysis.py", env_vars, dry_run=dry_run)
    
    if not analysis_result["success"]:
        result["error"] = analysis_result.get("error") or analysis_result.get("stderr", "")[:200]
        logger.error(f"    [FAIL] HTML 생성 실패: {result['error']}")
        return result
    
    logger.info(f"    [OK] HTML 생성 완료")
    result["success"] = True
    return result


# =========================================================
# 검수 시트에서 CSV 생성
# =========================================================
def create_reviewed_csv(
    gspread_mgr,
    building_id: int,
    building_name: str,
    yyyymm: str,
    reviewed_dir: str,
    run_id: str,
    tagging_dir: str,
) -> Dict:
    """검수 시트에서 데이터 다운로드하여 기존 태깅 CSV와 병합 후 reviewed CSV 생성"""
    
    result = {
        "building_id": building_id,
        "building_name": building_name,
        "success": False,
        "csv_path": None,
        "total_rows": 0,
        "reviewed_rows": 0,
        "error": None,
    }
    
    # 1) 기존 자동 태깅 CSV 찾기
    import glob
    tagging_pattern = os.path.join(tagging_dir, f"tagged_{building_id}_{yyyymm}_*.csv")
    tagging_files = sorted(glob.glob(tagging_pattern), reverse=True)
    
    if not tagging_files:
        result["error"] = f"기존 태깅 CSV 없음: {tagging_pattern}"
        return result
    
    # 가장 최근 태깅 파일 사용
    tagging_csv_path = tagging_files[0]
    
    try:
        df_tagged = pd.read_csv(tagging_csv_path, dtype=str)
    except Exception as e:
        result["error"] = f"태깅 CSV 읽기 실패: {e}"
        return result
    
    # 2) 검수 시트에서 데이터 다운로드
    df_review, error = gspread_mgr.download_all_data(yyyymm, building_id)
    
    if error:
        result["error"] = error
        return result
    
    if df_review is None or df_review.empty:
        result["error"] = "검수 시트 데이터 없음"
        return result
    
    result["total_rows"] = len(df_review)
    result["reviewed_rows"] = len(df_review[df_review.get("검수완료", pd.Series()).fillna("").astype(str).str.upper() == "Y"])
    
    # 3) voc_id 기준으로 검수 태깅 컬럼만 병합
    # 검수 시트에서 가져올 컬럼: voc_id + 태깅 컬럼들
    review_columns = ["voc_id", "주제 대분류", "주제 중분류", "작업유형 대분류", "작업유형 중분류"]
    available_review_cols = [c for c in review_columns if c in df_review.columns]
    df_review_subset = df_review[available_review_cols].copy()
    
    # 기존 태깅 CSV에서 태깅 컬럼 제거 후 검수 데이터로 대체
    tag_cols_to_drop = ["주제 대분류", "주제 중분류", "작업유형 대분류", "작업유형 중분류"]
    df_tagged_base = df_tagged.drop(columns=[c for c in tag_cols_to_drop if c in df_tagged.columns], errors="ignore")
    
    # voc_id 기준 병합
    df_tagged_base["voc_id"] = df_tagged_base["voc_id"].astype(str)
    df_review_subset["voc_id"] = df_review_subset["voc_id"].astype(str)
    
    df_merged = df_tagged_base.merge(df_review_subset, on="voc_id", how="left")
    
    # 4) reviewed CSV 저장
    os.makedirs(reviewed_dir, exist_ok=True)
    csv_filename = f"reviewed_{building_id}_{yyyymm}_{run_id}.csv"
    csv_path = os.path.join(reviewed_dir, csv_filename)
    
    df_merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    result["success"] = True
    result["csv_path"] = csv_path
    
    return result


# =========================================================
# MODE: full - 태깅 → HTML → S3 → 검수시트
# =========================================================
def process_building_full(
    building_id: int,
    building_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    yyyymm: str,
    env_config: dict,
    s3_uploader,
    gspread_mgr,
    dry_run: bool = False,
) -> Dict:
    """full 모드: 태깅 → HTML → S3 → 검수시트"""
    
    result = {
        "building_id": building_id,
        "building_name": building_name,
        "success": False,
        "tagging": None,
        "analysis": None,
        "s3_upload": None,
        "gsheet_upload": None,
        "error_stage": None,
        "error_message": None,
    }
    
    logger.info("")
    logger.info(f"  [{building_id}] {building_name}")
    
    # 1) 태깅
    logger.info(f"  [1/4] 태깅")
    tagging_result = run_tagging(
        building_id, building_name, start_date, end_date,
        run_id, env_config, dry_run
    )
    result["tagging"] = tagging_result
    
    if not tagging_result["success"]:
        result["error_stage"] = "태깅"
        result["error_message"] = tagging_result.get("error")
        return result
    
    # 2) HTML 생성
    logger.info(f"  [2/4] HTML 생성")
    analysis_result = run_analysis(
        building_id, building_name, start_date, end_date,
        run_id, env_config, dry_run=dry_run
    )
    result["analysis"] = analysis_result
    
    if not analysis_result["success"]:
        result["error_stage"] = "HTML 생성"
        result["error_message"] = analysis_result.get("error")
        return result
    
    # 3) S3 업로드
    if s3_uploader and s3_uploader.is_available() and not dry_run:
        logger.info(f"  [3/4] S3 업로드")
        tagging_dir = os.path.join(env_config["OUT_DIR"], "tagging")
        html_dir = os.path.join(env_config["OUT_DIR"], "html")
        
        s3_result = s3_uploader.upload_building_outputs(
            building_id=building_id,
            yyyymm=yyyymm,
            run_id=run_id,
            tagging_dir=tagging_dir,
            html_dir=html_dir,
        )
        result["s3_upload"] = s3_result
        
        if s3_result["success"]:
            logger.info(f"    [OK] S3 업로드 완료")
        else:
            logger.warning(f"    [WARN] S3 업로드 일부 실패")
    else:
        logger.info(f"  [3/4] S3 업로드 (건너뜀)")
    
    # 4) 검수 시트 업로드
    if gspread_mgr and gspread_mgr.is_available() and not dry_run:
        logger.info(f"  [4/4] 검수 시트 업로드")
        tagging_dir = os.path.join(env_config["OUT_DIR"], "tagging")
        csv_path = os.path.join(tagging_dir, f"tagged_{building_id}_{yyyymm}_{run_id}.csv")
        
        gsheet_result = gspread_mgr.upload_tagged_csv(
            csv_path=csv_path,
            building_id=building_id,
            building_name=building_name,
            yyyymm=yyyymm,
        )
        result["gsheet_upload"] = gsheet_result
        
        if gsheet_result["success"]:
            logger.info(f"    [OK] 신규={gsheet_result['rows_uploaded']}건, 유지={gsheet_result.get('rows_preserved', 0)}건")
            
            # 드롭다운 유효성 검사 설정
            dropdown_result = gspread_mgr.set_dropdown_validation(yyyymm)
            if dropdown_result["success"]:
                logger.info(f"    [OK] 드롭다운 설정 완료")
            else:
                logger.warning(f"    [WARN] 드롭다운 설정 실패: {dropdown_result['error']}")
        else:
            logger.warning(f"    [WARN] 업로드 실패: {gsheet_result['error']}")
    else:
        logger.info(f"  [4/4] 검수 시트 업로드 (건너뜀)")
    
    result["success"] = True
    return result


# =========================================================
# MODE: refresh - 검수시트 → HTML 재생성 → S3
# =========================================================
def process_building_refresh(
    building_id: int,
    building_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    yyyymm: str,
    env_config: dict,
    s3_uploader,
    gspread_mgr,
    dry_run: bool = False,
) -> Dict:
    """refresh 모드: 검수시트 → HTML 재생성 → S3"""
    
    result = {
        "building_id": building_id,
        "building_name": building_name,
        "success": False,
        "reviewed_csv": None,
        "analysis": None,
        "s3_upload": None,
        "error_stage": None,
        "error_message": None,
    }
    
    logger.info("")
    logger.info(f"  [{building_id}] {building_name}")
    
    # 1) 검수 시트에서 데이터 다운로드 → reviewed CSV 생성
    logger.info(f"  [1/3] 검수 데이터 다운로드")
    
    if not gspread_mgr or not gspread_mgr.is_available():
        result["error_stage"] = "검수 데이터"
        result["error_message"] = "Google Sheets 연결 불가"
        logger.error(f"    [FAIL] {result['error_message']}")
        return result
    
    if dry_run:
        logger.info(f"    [DRY-RUN] 검수 데이터 다운로드")
        reviewed_csv_path = None
    else:
        # S3에서 최신 tagged CSV 다운로드
        if not s3_uploader or not s3_uploader.is_available():
            result["error_stage"] = "S3 다운로드"
            result["error_message"] = "S3 연결 불가"
            logger.error(f"    [FAIL] {result['error_message']}")
            return result
        
        s3_key = s3_uploader.find_latest_tagged_csv(building_id, yyyymm)
        if not s3_key:
            result["error_stage"] = "S3 다운로드"
            result["error_message"] = f"S3에 tagged CSV 없음: {yyyymm}/{building_id}"
            logger.error(f"    [FAIL] {result['error_message']}")
            return result
        
        temp_tagging_dir = os.path.join(env_config["OUT_DIR"], "temp_tagging")
        os.makedirs(temp_tagging_dir, exist_ok=True)
        local_csv_path = os.path.join(temp_tagging_dir, os.path.basename(s3_key))
        
        download_result = s3_uploader.download_file(s3_key, local_csv_path)
        if not download_result["success"]:
            result["error_stage"] = "S3 다운로드"
            result["error_message"] = download_result["error"]
            logger.error(f"    [FAIL] {result['error_message']}")
            return result
        
        logger.info(f"    [OK] S3에서 tagged CSV 다운로드 완료")
        
        reviewed_dir = os.path.join(env_config["OUT_DIR"], "reviewed")
        reviewed_result = create_reviewed_csv(
            gspread_mgr=gspread_mgr,
            building_id=building_id,
            building_name=building_name,
            yyyymm=yyyymm,
            reviewed_dir=reviewed_dir,
            run_id=run_id,
            tagging_dir=temp_tagging_dir,
        )
        result["reviewed_csv"] = reviewed_result
        
        if not reviewed_result["success"]:
            result["error_stage"] = "검수 데이터"
            result["error_message"] = reviewed_result.get("error")
            logger.error(f"    [FAIL] {result['error_message']}")
            return result
        
        reviewed_csv_path = reviewed_result["csv_path"]
        logger.info(f"    [OK] {reviewed_result['total_rows']}건 (검수완료: {reviewed_result['reviewed_rows']}건)")
    
    # 2) HTML 생성 (검수 데이터 사용)
    logger.info(f"  [2/3] HTML 생성 (검수 데이터 기반)")
    analysis_result = run_analysis(
        building_id, building_name, start_date, end_date,
        run_id, env_config,
        csv_path=reviewed_csv_path,
        dry_run=dry_run
    )
    result["analysis"] = analysis_result
    
    if not analysis_result["success"]:
        result["error_stage"] = "HTML 생성"
        result["error_message"] = analysis_result.get("error")
        return result
    
    # 3) S3 업로드 (HTML만 덮어쓰기)
    if s3_uploader and s3_uploader.is_available() and not dry_run:
        logger.info(f"  [3/3] S3 업로드 (HTML 갱신)")
        html_dir = os.path.join(env_config["OUT_DIR"], "html")
        html_filename = f"dashboard_{building_id}_{yyyymm}_{run_id}.html"
        html_local = os.path.join(html_dir, html_filename)
        html_s3_key = s3_uploader.build_s3_key(yyyymm, building_id, html_filename)
        
        s3_result = s3_uploader.upload_file(html_local, html_s3_key)
        result["s3_upload"] = s3_result
        
        if s3_result["success"]:
            logger.info(f"    [OK] {s3_result['s3_uri']}")
        else:
            logger.warning(f"    [WARN] {s3_result['error']}")
    else:
        logger.info(f"  [3/3] S3 업로드 (건너뜀)")
    
    result["success"] = True
    return result


# =========================================================
# MODE: tagging-only - 태깅 → 검수시트만
# =========================================================
def process_building_tagging_only(
    building_id: int,
    building_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    yyyymm: str,
    env_config: dict,
    gspread_mgr,
    dry_run: bool = False,
) -> Dict:
    """tagging-only 모드: 태깅 → 검수시트만"""
    
    result = {
        "building_id": building_id,
        "building_name": building_name,
        "success": False,
        "tagging": None,
        "gsheet_upload": None,
        "error_stage": None,
        "error_message": None,
    }
    
    logger.info("")
    logger.info(f"  [{building_id}] {building_name}")
    
    # 1) 태깅
    logger.info(f"  [1/2] 태깅")
    tagging_result = run_tagging(
        building_id, building_name, start_date, end_date,
        run_id, env_config, dry_run
    )
    result["tagging"] = tagging_result
    
    if not tagging_result["success"]:
        result["error_stage"] = "태깅"
        result["error_message"] = tagging_result.get("error")
        return result
    
    # 2) 검수 시트 업로드
    if gspread_mgr and gspread_mgr.is_available() and not dry_run:
        logger.info(f"  [2/2] 검수 시트 업로드")
        tagging_dir = os.path.join(env_config["OUT_DIR"], "tagging")
        csv_path = os.path.join(tagging_dir, f"tagged_{building_id}_{yyyymm}_{run_id}.csv")
        
        gsheet_result = gspread_mgr.upload_tagged_csv(
            csv_path=csv_path,
            building_id=building_id,
            building_name=building_name,
            yyyymm=yyyymm,
        )
        result["gsheet_upload"] = gsheet_result
        
        if gsheet_result["success"]:
            logger.info(f"    [OK] 신규={gsheet_result['rows_uploaded']}건, 유지={gsheet_result.get('rows_preserved', 0)}건")
            
            # 드롭다운 유효성 검사 설정
            dropdown_result = gspread_mgr.set_dropdown_validation(yyyymm)
            if dropdown_result["success"]:
                logger.info(f"    [OK] 드롭다운 설정 완료")
            else:
                logger.warning(f"    [WARN] 드롭다운 설정 실패: {dropdown_result['error']}")
        else:
            logger.warning(f"    [WARN] 업로드 실패: {gsheet_result['error']}")
    else:
        logger.info(f"  [2/2] 검수 시트 업로드 (건너뜀)")
    
    result["success"] = True
    return result


# =========================================================
# 메인 배치 함수
# =========================================================
def run_monthly_batch(
    mode: str = MODE_FULL,
    env_path: str = None,
    all_buildings: bool = False,
    auto_month: bool = False,
    year: int = None,
    month: int = None,
    building_id: int = None,
    run_id_prefix: str = "auto",
    dry_run: bool = False,
    no_s3_upload: bool = False,
    no_gsheet: bool = False,
) -> Dict:
    """월간 배치 메인 함수"""
    
    start_time = datetime.now(KST)
    
    # 1) .env 로딩
    env_config = load_env_config(env_path)
    
    # 2) run_id 생성
    run_id = generate_run_id(run_id_prefix)
    
    # 3) 대상 월 결정
    if year and month:
        start_date, end_date = get_month_date_range(year, month)
    elif auto_month:
        year, month = get_previous_month()
        start_date, end_date = get_month_date_range(year, month)
    else:
        env_start = env_config.get("START_DATE", "")
        env_end = env_config.get("END_DATE", "")
        if env_start and env_end:
            start_date = env_start
            end_date = env_end
            year = int(start_date[:4])
            month = int(start_date[5:7])
        else:
            year, month = get_previous_month()
            start_date, end_date = get_month_date_range(year, month)
    
    target_month = f"{year:04d}-{month:02d}"
    yyyymm = f"{year:04d}{month:02d}"
    
    # 4) 대상 Building 결정
    single_building_id = None
    
    if building_id:
        single_building_id = building_id
        buildings = [{"id": building_id, "name": ""}]
        all_bldgs = {b["id"]: b["name"] for b in fetch_active_buildings()}
        if building_id in all_bldgs:
            buildings[0]["name"] = all_bldgs[building_id]
        else:
            buildings[0]["name"] = f"Building_{building_id}"
    elif all_buildings:
        buildings = fetch_active_buildings()
        if not buildings:
            print("[ERROR] 활성 Building이 없습니다.")
            return {"success": 0, "failed": 0, "total": 0}
    else:
        env_bid = env_config.get("BUILDING_ID", "")
        if not env_bid:
            print("[ERROR] BUILDING_ID가 지정되지 않았습니다.")
            return {"success": 0, "failed": 0, "total": 0}
        
        single_building_id = int(env_bid)
        buildings = [{"id": single_building_id, "name": ""}]
        all_bldgs = {b["id"]: b["name"] for b in fetch_active_buildings()}
        if single_building_id in all_bldgs:
            buildings[0]["name"] = all_bldgs[single_building_id]
        else:
            buildings[0]["name"] = f"Building_{single_building_id}"
    
    # 5) 로거 설정
    log_dir = env_config.get("LOG_DIR", DEFAULT_LOG_DIR)
    log_path = setup_logger(log_dir, run_id, mode, single_building_id)
    
    # 6) S3 업로더 초기화
    s3_uploader = None
    s3_enabled = not no_s3_upload and mode in [MODE_FULL, MODE_REFRESH]
    if s3_enabled:
        s3_uploader = get_s3_uploader(
            bucket_name=env_config.get("S3_BUCKET"),
            prefix=env_config.get("S3_PREFIX"),
        )
    
    # 7) GSpread 매니저 초기화
    gspread_mgr = None
    gsheet_enabled = not no_gsheet
    if gsheet_enabled:
        gspread_mgr = get_gspread_manager()
    
    # 8) 배치 시작 로그
    logger.info("월간 VOC 분석 배치 시작")
    
    log_separator()
    logger.info("월간 VOC 분석 배치")
    log_separator()
    logger.info(f"  실행 모드     : {mode}")
    logger.info(f"  대상 월       : {target_month}")
    logger.info(f"  기간          : {start_date} ~ {end_date}")
    logger.info(f"  RUN_ID        : {run_id}")
    logger.info(f"  대상 Building : {len(buildings)}개")
    for b in buildings:
        logger.info(f"                  - [{b['id']}] {b['name']}")
    logger.info(f"  DRY-RUN       : {dry_run}")
    
    # 모드별 설명
    if mode == MODE_FULL:
        logger.info(f"  처리 순서     : 태깅 → HTML → S3 → 검수시트")
    elif mode == MODE_REFRESH:
        logger.info(f"  처리 순서     : 검수시트 → HTML → S3")
    elif mode == MODE_TAGGING_ONLY:
        logger.info(f"  처리 순서     : 태깅 → 검수시트")
    
    # S3 상태
    if s3_enabled:
        if s3_uploader and s3_uploader.is_available():
            logger.info(f"  S3 업로드     : 활성화")
        else:
            logger.info(f"  S3 업로드     : 초기화 실패")
    else:
        logger.info(f"  S3 업로드     : 비활성화")
    
    # GSheet 상태
    if gsheet_enabled:
        if gspread_mgr and gspread_mgr.is_available():
            logger.info(f"  검수 시트     : 활성화")
        else:
            logger.info(f"  검수 시트     : 초기화 실패")
    else:
        logger.info(f"  검수 시트     : 비활성화")
    
    log_separator()
    
    # 8-1) taxonomy 시트 업데이트 (full 또는 tagging-only 모드일 때)
    if mode in [MODE_FULL, MODE_TAGGING_ONLY] and gspread_mgr and gspread_mgr.is_available() and not dry_run:
        logger.info("")
        logger.info("Taxonomy 시트 업데이트")
        
        try:
            from nlp_model import fetch_taxonomy_dfs
            subject_tax_df, work_tax_df = fetch_taxonomy_dfs()
            
            # DataFrame 합치기
            subject_tax_df = subject_tax_df.copy()
            subject_tax_df["taxonomy_type"] = "SUBJECT"
            work_tax_df = work_tax_df.copy()
            work_tax_df["taxonomy_type"] = "WORK"
            
            taxonomy_df = pd.concat([subject_tax_df, work_tax_df], ignore_index=True)
            
            tax_result = gspread_mgr.update_taxonomy_sheet(taxonomy_df)
            if tax_result["success"]:
                logger.info(f"  [OK] taxonomy 시트 업데이트 완료")
            else:
                logger.warning(f"  [WARN] taxonomy 시트 업데이트 실패: {tax_result['error']}")
        except Exception as e:
            logger.warning(f"  [WARN] taxonomy 조회 실패: {e}")
    
    # 9) Building 처리
    results = []
    success_count = 0
    failed_count = 0
    
    logger.info("")
    logger.info("Building 처리 시작")
    
    for idx, bldg in enumerate(buildings, 1):
        logger.info("")
        logger.info(f"[{idx}/{len(buildings)}]")
        
        if mode == MODE_FULL:
            result = process_building_full(
                building_id=bldg["id"],
                building_name=bldg["name"],
                start_date=start_date,
                end_date=end_date,
                run_id=run_id,
                yyyymm=yyyymm,
                env_config=env_config,
                s3_uploader=s3_uploader,
                gspread_mgr=gspread_mgr,
                dry_run=dry_run,
            )
        elif mode == MODE_REFRESH:
            result = process_building_refresh(
                building_id=bldg["id"],
                building_name=bldg["name"],
                start_date=start_date,
                end_date=end_date,
                run_id=run_id,
                yyyymm=yyyymm,
                env_config=env_config,
                s3_uploader=s3_uploader,
                gspread_mgr=gspread_mgr,
                dry_run=dry_run,
            )
        elif mode == MODE_TAGGING_ONLY:
            result = process_building_tagging_only(
                building_id=bldg["id"],
                building_name=bldg["name"],
                start_date=start_date,
                end_date=end_date,
                run_id=run_id,
                yyyymm=yyyymm,
                env_config=env_config,
                gspread_mgr=gspread_mgr,
                dry_run=dry_run,
            )
        else:
            logger.error(f"알 수 없는 모드: {mode}")
            continue
        
        results.append(result)
        
        if result["success"]:
            success_count += 1
        else:
            failed_count += 1
    
    end_time = datetime.now(KST)
    elapsed = end_time - start_time
    
    # 10) 최종 결과
    logger.info("")
    log_separator()
    logger.info("배치 완료")
    log_separator()
    logger.info(f"  실행 모드  : {mode}")
    logger.info(f"  처리 성공  : {success_count}개")
    logger.info(f"  처리 실패  : {failed_count}개")
    logger.info(f"  소요시간   : {elapsed}")
    
    # 실패 상세
    if failed_count > 0:
        logger.info("")
        logger.info("  [실패 상세]")
        for r in results:
            if not r["success"]:
                logger.info(f"    - [{r['building_id']}] {r['building_name']}")
                logger.info(f"      단계: {r.get('error_stage')}")
                logger.info(f"      에러: {r.get('error_message')}")
    
    # 산출물 경로
    logger.info("")
    logger.info("  [산출물 경로]")
    if mode in [MODE_FULL, MODE_TAGGING_ONLY]:
        logger.info(f"    CSV       : {env_config['OUT_DIR']}/tagging/")
    if mode in [MODE_FULL, MODE_REFRESH]:
        logger.info(f"    HTML      : {env_config['OUT_DIR']}/html/")
    if mode == MODE_REFRESH:
        logger.info(f"    검수CSV   : {env_config['OUT_DIR']}/reviewed/")
    if s3_enabled and s3_uploader and s3_uploader.is_available():
        logger.info(f"    S3        : s3://{env_config.get('S3_BUCKET')}/{env_config.get('S3_PREFIX')}/{yyyymm}/")
    if gsheet_enabled and gspread_mgr and gspread_mgr.is_available():
        logger.info(f"    검수시트  : https://docs.google.com/spreadsheets/d/{gspread_mgr.spreadsheet_id}/")
    logger.info(f"    LOG       : {log_path}")
    log_separator()

    # 11) 배치 실행 메타데이터 저장
    batch_result = {
        "mode": mode,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "target_month": target_month,
        "yyyymm": yyyymm,
        "run_id": run_id,
        "log_path": log_path,
        "total": len(buildings),
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }

    if not dry_run:
        import json
        runs_dir = os.path.join(env_config["OUT_DIR"], "runs")
        os.makedirs(runs_dir, exist_ok=True)
        meta_filename = f"run_monthly_{mode}_{run_id}.json"
        meta_path = os.path.join(runs_dir, meta_filename)

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(batch_result, f, ensure_ascii=False, indent=2)
            logger.info(f"[INFO] 배치 메타데이터 저장: {meta_path}")
        except Exception as e:
            logger.warning(f"[WARN] 메타데이터 저장 실패: {e}")

    return batch_result


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="월간 VOC 분석 배치 오케스트레이터",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 모드:
  full         : 태깅 → HTML → S3 → 검수시트 (기본)
  refresh      : 검수시트 → HTML 재생성 → S3 (검수 반영)
  tagging-only : 태깅 → 검수시트만 (HTML 안 만듦)

사용 예시:
  # 로컬 개발 환경에서 특정 빌딩 테스트
  python run_monthly.py --env dev --mode full --building-id 95 --year 2025 --month 12

  # EC2 운영 환경에서 전체 빌딩 처리
  python run_monthly.py --env prd --mode full --all-buildings --auto-month

  # 검수 완료 후 HTML 재생성 (운영)
  python run_monthly.py --env prd --mode refresh --all-buildings --auto-month

  # 실행 계획만 확인 (dry-run)
  python run_monthly.py --env dev --mode full --building-id 95 --year 2025 --month 12 --dry-run
        """,
    )
    
    parser.add_argument("--mode", type=str, default=MODE_FULL,
                        choices=[MODE_FULL, MODE_REFRESH, MODE_TAGGING_ONLY],
                        help=f"실행 모드 (기본: {MODE_FULL})")

    parser.add_argument("--env", "-e", type=str, default="prd",
                        choices=["prd", "dev"],
                        help="실행 환경 (prd: EC2 운영, dev: 로컬 개발, 기본: prd)")
    
    parser.add_argument("--all-buildings", "-a", action="store_true",
                        help="모든 활성 Building 처리")
    
    parser.add_argument("--auto-month", action="store_true",
                        help="전월 자동 계산")
    
    parser.add_argument("--year", "-y", type=int, default=None,
                        help="대상 연도")
    
    parser.add_argument("--month", "-m", type=int, default=None,
                        help="대상 월")
    
    parser.add_argument("--building-id", "-b", type=int, default=None,
                        help="특정 Building ID")
    
    parser.add_argument("--run-id-prefix", type=str, default="auto",
                        help="run_id 접두어 (기본: auto)")
    
    parser.add_argument("--no-s3-upload", action="store_true",
                        help="S3 업로드 비활성화")
    
    parser.add_argument("--no-gsheet", action="store_true",
                        help="Google Sheets 연동 비활성화")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 실행 없이 계획만 출력")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if (args.year is None) != (args.month is None):
        print("[ERROR] --year와 --month는 함께 지정해야 합니다.")
        sys.exit(1)
    
    if args.year and args.auto_month:
        print("[WARN] --year/--month와 --auto-month 함께 지정됨. --year/--month 우선.")

    # 환경별 .env 경로 매핑
    env_path = ENV_PATHS.get(args.env, DEFAULT_ENV_PATH)

    result = run_monthly_batch(
        mode=args.mode,
        env_path=env_path,
        all_buildings=args.all_buildings,
        auto_month=args.auto_month,
        year=args.year,
        month=args.month,
        building_id=args.building_id,
        run_id_prefix=args.run_id_prefix,
        dry_run=args.dry_run,
        no_s3_upload=args.no_s3_upload,
        no_gsheet=args.no_gsheet,
    )
    
    if result.get("failed", 0) > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()