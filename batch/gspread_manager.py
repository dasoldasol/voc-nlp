# /home/ssm-user/jupyter/batch/gspread_manager.py
# -*- coding: utf-8 -*-
"""
Google Sheets 연동 모듈

검수용 스프레드시트에 태깅 결과를 업로드하고,
검수 완료된 데이터를 다운로드합니다.

스프레드시트 구조:
  - 월별 탭 (예: 202512, 202601)
  - 각 탭에 전체 빌딩 데이터 포함
  - building_id, building_name 컬럼으로 필터링
  - _taxonomy 탭: 드롭다운용 taxonomy 목록 (대분류-중분류 쌍)

필요 패키지:
  pip install gspread google-auth
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

KST = timezone(timedelta(hours=9))

logger = logging.getLogger("run_monthly")

# 기본 설정
DEFAULT_CREDENTIALS_PATH = "/home/ssm-user/jupyter/.credentials/google_sheets.json"
DEFAULT_SPREADSHEET_ID = "1hge3C5n5kUVRcO40wmBIIqjuPNyHSMPX0SHtaawzbK4"

# Google Sheets API 스코프
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# 검수 시트 컬럼 정의
REVIEW_COLUMNS = [
    # 기본 정보 (수정 불가)
    "voc_id",
    "building_id",
    "building_name",
    "voc_date",
    "title_clean",
    "request_contents",
    "reply",
    # 자동 태깅 결과 (수정 불가)
    "주제 대분류",
    "주제 중분류",
    "작업유형 대분류",
    "작업유형 중분류",
    # 검수 입력 컬럼
    "검수_주제 대분류",
    "검수_주제 중분류",
    "검수_작업유형 대분류",
    "검수_작업유형 중분류",
    "검수완료",
    "검수자",
    "검수일시",
]


class GSpreadManager:
    """Google Sheets 연동 클래스"""
    
    def __init__(
        self,
        credentials_path: str = None,
        spreadsheet_id: str = None,
    ):
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_SHEETS_CREDENTIALS", DEFAULT_CREDENTIALS_PATH
        )
        self.spreadsheet_id = spreadsheet_id or os.getenv(
            "GOOGLE_SHEETS_ID", DEFAULT_SPREADSHEET_ID
        )
        
        self.client = None
        self.spreadsheet = None
        self._initialized = False
        self._init_error = None
        
        self._init_client()
    
    def _init_client(self):
        """gspread 클라이언트 초기화"""
        if not GSPREAD_AVAILABLE:
            self._init_error = "gspread 또는 google-auth 패키지가 설치되지 않았습니다. pip install gspread google-auth"
            logger.warning(f"GSpread 초기화 실패: {self._init_error}")
            return
        
        if not os.path.exists(self.credentials_path):
            self._init_error = f"인증 파일 없음: {self.credentials_path}"
            logger.warning(f"GSpread 초기화 실패: {self._init_error}")
            return
        
        try:
            credentials = Credentials.from_service_account_file(
                self.credentials_path,
                scopes=SCOPES,
            )
            self.client = gspread.authorize(credentials)
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            self._initialized = True
            logger.debug(f"GSpread 초기화 완료: {self.spreadsheet.title}")
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"GSpread 초기화 실패: {self._init_error}")
    
    def is_available(self) -> bool:
        """사용 가능 여부"""
        return self._initialized and self.spreadsheet is not None
    
    def get_init_error(self) -> Optional[str]:
        """초기화 실패 원인"""
        return self._init_error
    
    def _get_or_create_worksheet(self, tab_name: str, rows: int = 1000, cols: int = 20) -> Optional[gspread.Worksheet]:
        """
        탭(워크시트) 가져오기, 없으면 생성
        """
        if not self.is_available():
            return None
        
        try:
            # 기존 탭 찾기
            worksheet = self.spreadsheet.worksheet(tab_name)
            logger.debug(f"기존 탭 사용: {tab_name}")
            return worksheet
        except gspread.WorksheetNotFound:
            # 새 탭 생성
            try:
                worksheet = self.spreadsheet.add_worksheet(
                    title=tab_name,
                    rows=rows,
                    cols=cols,
                )
                logger.info(f"새 탭 생성: {tab_name}")
                return worksheet
            except Exception as e:
                logger.error(f"탭 생성 실패 ({tab_name}): {e}")
                return None
    
    def _prepare_upload_df(self, df: pd.DataFrame, building_name: str) -> pd.DataFrame:
        """
        업로드용 DataFrame 준비
        - 필요한 컬럼만 선택
        - 검수 컬럼 추가 (빈 값)
        - building_name 추가
        """
        result = pd.DataFrame()
        
        # 기본 컬럼 매핑
        col_mapping = {
            "voc_id": "voc_id",
            "building_id": "building_id",
            "voc_date": "voc_date",
            "title_clean": "title_clean",
            "request_contents": "request_contents",
            "reply": "reply",
            "주제 대분류": "주제 대분류",
            "주제 중분류": "주제 중분류",
            "작업유형 대분류": "작업유형 대분류",
            "작업유형 중분류": "작업유형 중분류",
        }
        
        for target_col, source_col in col_mapping.items():
            if source_col in df.columns:
                result[target_col] = df[source_col].fillna("").astype(str)
            else:
                result[target_col] = ""
        
        # building_name 추가
        result["building_name"] = building_name
        
        # 검수 컬럼 추가 (빈 값)
        result["검수_주제 대분류"] = ""
        result["검수_주제 중분류"] = ""
        result["검수_작업유형 대분류"] = ""
        result["검수_작업유형 중분류"] = ""
        result["검수완료"] = ""
        result["검수자"] = ""
        result["검수일시"] = ""
        
        # 컬럼 순서 정렬
        result = result[REVIEW_COLUMNS]
        
        return result
    
    def _col_num_to_letter(self, col_num: int) -> str:
        """컬럼 번호를 알파벳으로 변환 (1=A, 2=B, ...)"""
        result = ""
        while col_num > 0:
            col_num, remainder = divmod(col_num - 1, 26)
            result = chr(65 + remainder) + result
        return result
    
    def upload_tagged_csv(
        self,
        csv_path: str,
        building_id: int,
        building_name: str,
        yyyymm: str,
    ) -> Dict:
        """
        태깅된 CSV를 검수 시트에 업로드
        - 검수완료=Y인 기존 데이터는 유지
        - 새 데이터만 추가/업데이트
        
        Args:
            csv_path: 태깅된 CSV 파일 경로
            building_id: 빌딩 ID
            building_name: 빌딩 이름
            yyyymm: 대상 월 (탭 이름으로 사용)
        
        Returns:
            dict: {"success": bool, "rows_uploaded": int, "rows_preserved": int, "error": str}
        """
        result = {
            "success": False,
            "rows_uploaded": 0,
            "rows_preserved": 0,
            "error": "",
        }
        
        if not self.is_available():
            result["error"] = self._init_error or "GSpread 미초기화"
            return result
        
        if not os.path.exists(csv_path):
            result["error"] = f"CSV 파일 없음: {csv_path}"
            return result
        
        try:
            # CSV 읽기
            df_new = pd.read_csv(csv_path, dtype=str)
            if df_new.empty:
                result["error"] = "CSV 파일이 비어있습니다."
                return result
            
            # 업로드용 DataFrame 준비
            upload_df = self._prepare_upload_df(df_new, building_name)
            
            # 탭 가져오기/생성
            worksheet = self._get_or_create_worksheet(yyyymm)
            if worksheet is None:
                result["error"] = f"탭 생성/접근 실패: {yyyymm}"
                return result
            
            # 기존 데이터 확인
            existing_data = worksheet.get_all_values()
            
            if len(existing_data) == 0:
                # 빈 시트: 헤더 + 데이터 추가
                header = REVIEW_COLUMNS
                rows = upload_df.values.tolist()
                worksheet.append_row(header)
                if rows:
                    worksheet.append_rows(rows)
                result["rows_uploaded"] = len(rows)
            else:
                # 기존 데이터 있음
                header = existing_data[0]
                data_rows = existing_data[1:]
                
                df_existing = pd.DataFrame(data_rows, columns=header)
                
                # building_id 컬럼 확인
                if "building_id" not in df_existing.columns:
                    # building_id 컬럼이 없으면 기존 방식으로 처리
                    df_other_buildings = df_existing
                    df_this_building = pd.DataFrame(columns=header)
                else:
                    # 다른 빌딩 데이터
                    df_other_buildings = df_existing[df_existing["building_id"].astype(str) != str(building_id)].copy()
                    # 이 빌딩의 기존 데이터
                    df_this_building = df_existing[df_existing["building_id"].astype(str) == str(building_id)].copy()
                
                # 검수완료=Y인 행 추출 (유지할 데이터)
                if "검수완료" in df_this_building.columns and not df_this_building.empty:
                    df_preserved = df_this_building[df_this_building["검수완료"].str.upper() == "Y"].copy()
                else:
                    df_preserved = pd.DataFrame(columns=header)
                
                result["rows_preserved"] = len(df_preserved)
                
                # 새 데이터에서 검수완료된 voc_id 제외
                preserved_voc_ids = set()
                if "voc_id" in df_preserved.columns and not df_preserved.empty:
                    preserved_voc_ids = set(df_preserved["voc_id"].astype(str).tolist())
                
                if "voc_id" in upload_df.columns:
                    df_new_only = upload_df[~upload_df["voc_id"].astype(str).isin(preserved_voc_ids)].copy()
                else:
                    df_new_only = upload_df.copy()
                
                result["rows_uploaded"] = len(df_new_only)
                
                # 최종 데이터 조합: 다른 빌딩 + 검수완료된 행 + 새 데이터
                all_rows = []
                
                if not df_other_buildings.empty:
                    # 컬럼 순서 맞추기
                    for col in REVIEW_COLUMNS:
                        if col not in df_other_buildings.columns:
                            df_other_buildings[col] = ""
                    all_rows.extend(df_other_buildings[REVIEW_COLUMNS].values.tolist())
                
                if not df_preserved.empty:
                    # 컬럼 순서 맞추기
                    for col in REVIEW_COLUMNS:
                        if col not in df_preserved.columns:
                            df_preserved[col] = ""
                    all_rows.extend(df_preserved[REVIEW_COLUMNS].values.tolist())
                
                if not df_new_only.empty:
                    all_rows.extend(df_new_only.values.tolist())
                
                # 시트 클리어 후 재작성
                worksheet.clear()
                worksheet.append_row(REVIEW_COLUMNS)
                if all_rows:
                    worksheet.append_rows(all_rows)
            
            result["success"] = True
            logger.info(f"검수 시트 업로드 완료: {yyyymm} 탭, building_id={building_id}, 신규={result['rows_uploaded']}건, 유지={result['rows_preserved']}건")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"검수 시트 업로드 실패: {e}")
        
        return result
    
    def download_reviewed_data(
        self,
        yyyymm: str,
        building_id: int = None,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """
        검수 완료된 데이터 다운로드
        
        Args:
            yyyymm: 대상 월 (탭 이름)
            building_id: 특정 빌딩만 (None이면 전체)
        
        Returns:
            (DataFrame, error_message)
            - 검수완료=Y인 데이터만 반환
            - 검수 컬럼 값을 원본 컬럼에 병합
        """
        if not self.is_available():
            return None, self._init_error or "GSpread 미초기화"
        
        try:
            worksheet = self.spreadsheet.worksheet(yyyymm)
        except gspread.WorksheetNotFound:
            return None, f"탭 없음: {yyyymm}"
        except Exception as e:
            return None, str(e)
        
        try:
            # 전체 데이터 가져오기
            data = worksheet.get_all_values()
            if len(data) < 2:
                return None, "데이터 없음"
            
            header = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=header)
            
            # building_id 필터링
            if building_id is not None:
                df = df[df["building_id"].astype(str) == str(building_id)]
            
            if df.empty:
                return None, f"building_id={building_id} 데이터 없음"
            
            # 검수완료=Y 필터링
            reviewed_df = df[df["검수완료"].str.upper() == "Y"].copy()
            
            if reviewed_df.empty:
                return None, "검수 완료된 데이터 없음"
            
            # 검수 컬럼 → 원본 컬럼 병합
            reviewed_df = self._merge_review_columns(reviewed_df)
            
            logger.info(f"검수 데이터 다운로드: {yyyymm}, {len(reviewed_df)}건")
            return reviewed_df, ""
            
        except Exception as e:
            return None, str(e)
    
    def _merge_review_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        검수 컬럼 값을 원본 컬럼에 병합
        검수값이 있으면 덮어씀, 없으면 원본 유지
        """
        result = df.copy()
        
        review_mappings = [
            ("검수_주제 대분류", "주제 대분류"),
            ("검수_주제 중분류", "주제 중분류"),
            ("검수_작업유형 대분류", "작업유형 대분류"),
            ("검수_작업유형 중분류", "작업유형 중분류"),
        ]
        
        for review_col, target_col in review_mappings:
            if review_col in result.columns and target_col in result.columns:
                # 검수값이 있는 경우만 덮어씀
                mask = result[review_col].notna() & (result[review_col].str.strip() != "")
                result.loc[mask, target_col] = result.loc[mask, review_col]
        
        return result
    
    def get_review_status(self, yyyymm: str) -> Dict:
        """
        검수 현황 조회
        
        Returns:
            dict: {
                "total": int,
                "reviewed": int,
                "pending": int,
                "by_building": {building_id: {"total": n, "reviewed": n}, ...}
            }
        """
        status = {
            "total": 0,
            "reviewed": 0,
            "pending": 0,
            "by_building": {},
        }
        
        if not self.is_available():
            return status
        
        try:
            worksheet = self.spreadsheet.worksheet(yyyymm)
            data = worksheet.get_all_values()
            
            if len(data) < 2:
                return status
            
            header = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=header)
            
            status["total"] = len(df)
            status["reviewed"] = len(df[df["검수완료"].str.upper() == "Y"])
            status["pending"] = status["total"] - status["reviewed"]
            
            # 빌딩별 현황
            for bid in df["building_id"].unique():
                bid_df = df[df["building_id"] == bid]
                status["by_building"][bid] = {
                    "total": len(bid_df),
                    "reviewed": len(bid_df[bid_df["검수완료"].str.upper() == "Y"]),
                }
            
        except Exception as e:
            logger.warning(f"검수 현황 조회 실패: {e}")
        
        return status
    
    def download_all_data(
        self,
        yyyymm: str,
        building_id: int = None,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """
        전체 데이터 다운로드 (검수 여부 무관)
        검수 컬럼 값을 원본 컬럼에 병합하여 반환
        
        Args:
            yyyymm: 대상 월 (탭 이름)
            building_id: 특정 빌딩만 (None이면 전체)
        
        Returns:
            (DataFrame, error_message)
        """
        if not self.is_available():
            return None, self._init_error or "GSpread 미초기화"
        
        try:
            worksheet = self.spreadsheet.worksheet(yyyymm)
        except gspread.WorksheetNotFound:
            return None, f"탭 없음: {yyyymm}"
        except Exception as e:
            return None, str(e)
        
        try:
            data = worksheet.get_all_values()
            if len(data) < 2:
                return None, "데이터 없음"
            
            header = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=header)
            
            # building_id 필터링
            if building_id is not None:
                df = df[df["building_id"].astype(str) == str(building_id)]
            
            if df.empty:
                return None, f"building_id={building_id} 데이터 없음"
            
            # 검수 컬럼 → 원본 컬럼 병합
            df = self._merge_review_columns(df)
            
            # 날짜 컬럼 변환
            date_columns = ["voc_date", "write_date", "reply_write_date"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            
            return df, ""
            
        except Exception as e:
            return None, str(e)
    
    def update_taxonomy_sheet(self, taxonomy_df: pd.DataFrame) -> Dict:
        """
        taxonomy 시트 업데이트 (대분류-중분류 쌍으로 저장)
        빌딩 관리자가 참고할 수 있도록 DB 형식 그대로 표시

        Args:
            taxonomy_df: DataFrame with columns [taxonomy_type, major, minor]

        Returns:
            dict: {"success": bool, "error": str}
        """
        result = {
            "success": False,
            "error": "",
        }

        if not self.is_available():
            result["error"] = self._init_error or "GSpread 미초기화"
            return result

        try:
            # taxonomy 시트 가져오기/생성
            tab_name = "_taxonomy"
            worksheet = self._get_or_create_worksheet(tab_name, rows=500, cols=10)
            if worksheet is None:
                result["error"] = f"탭 생성/접근 실패: {tab_name}"
                return result

            # 시트 클리어
            worksheet.clear()

            # SUBJECT와 WORK 분리
            subject_df = taxonomy_df[taxonomy_df["taxonomy_type"] == "SUBJECT"].copy()
            work_df = taxonomy_df[taxonomy_df["taxonomy_type"] == "WORK"].copy()

            # 주제: 대분류-중분류 쌍 (중복 제거, 정렬)
            subject_pairs = subject_df[["major", "minor"]].drop_duplicates().sort_values(["major", "minor"]).values.tolist()

            # 작업유형: 대분류-중분류 쌍 (중복 제거, 정렬)
            work_pairs = work_df[["major", "minor"]].drop_duplicates().sort_values(["major", "minor"]).values.tolist()

            # 최대 행 수 계산
            max_len = max(len(subject_pairs), len(work_pairs))

            # 헤더
            headers = ["주제_대분류", "주제_중분류", "작업유형_대분류", "작업유형_중분류"]

            # 데이터 준비
            rows = [headers]
            for i in range(max_len):
                row = []
                # 주제 대분류, 중분류
                if i < len(subject_pairs):
                    row.extend(subject_pairs[i])
                else:
                    row.extend(["", ""])
                # 작업유형 대분류, 중분류
                if i < len(work_pairs):
                    row.extend(work_pairs[i])
                else:
                    row.extend(["", ""])
                rows.append(row)

            # 데이터 쓰기
            if rows:
                worksheet.append_rows(rows)

            result["success"] = True
            logger.info(f"taxonomy 시트 업데이트 완료: 주제 {len(subject_pairs)}쌍, 작업유형 {len(work_pairs)}쌍")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"taxonomy 시트 업데이트 실패: {e}")

        return result

    def restore_taxonomy_sheet(self) -> Dict:
        """
        _taxonomy 시트를 DB 형식(대분류-중분류 쌍)으로 복구
        현재 시트의 데이터를 읽어서 기존 형식으로 변환

        Returns:
            dict: {"success": bool, "error": str}
        """
        result = {
            "success": False,
            "error": "",
        }

        if not self.is_available():
            result["error"] = self._init_error or "GSpread 미초기화"
            return result

        try:
            tax_worksheet = self.spreadsheet.worksheet("_taxonomy")
            tax_data = tax_worksheet.get_all_values()

            if len(tax_data) < 2:
                result["error"] = "_taxonomy 시트에 데이터 없음"
                return result

            tax_header = tax_data[0]
            tax_rows = tax_data[1:]

            # 이미 기존 형식인지 확인
            if "주제_대분류" in tax_header and "주제_중분류" in tax_header:
                logger.info("_taxonomy 시트가 이미 기존 형식입니다.")
                result["success"] = True
                return result

            # 새 형식(대분류별 열)에서 대분류-중분류 쌍 추출
            subject_pairs = []
            work_pairs = []

            for col_idx, header in enumerate(tax_header):
                if header.startswith("주제_"):
                    major = header.replace("주제_", "")
                    for row in tax_rows:
                        if col_idx < len(row) and row[col_idx].strip():
                            subject_pairs.append([major, row[col_idx].strip()])
                elif header.startswith("작업유형_"):
                    major = header.replace("작업유형_", "")
                    for row in tax_rows:
                        if col_idx < len(row) and row[col_idx].strip():
                            work_pairs.append([major, row[col_idx].strip()])

            # 정렬
            subject_pairs = sorted(subject_pairs, key=lambda x: (x[0], x[1]))
            work_pairs = sorted(work_pairs, key=lambda x: (x[0], x[1]))

            # 최대 행 수 계산
            max_len = max(len(subject_pairs), len(work_pairs))

            # 헤더
            headers = ["주제_대분류", "주제_중분류", "작업유형_대분류", "작업유형_중분류"]

            # 데이터 준비
            rows = [headers]
            for i in range(max_len):
                row = []
                if i < len(subject_pairs):
                    row.extend(subject_pairs[i])
                else:
                    row.extend(["", ""])
                if i < len(work_pairs):
                    row.extend(work_pairs[i])
                else:
                    row.extend(["", ""])
                rows.append(row)

            # 시트 클리어 후 재작성
            tax_worksheet.clear()
            tax_worksheet.append_rows(rows)

            result["success"] = True
            logger.info(f"taxonomy 시트 복구 완료: 주제 {len(subject_pairs)}쌍, 작업유형 {len(work_pairs)}쌍")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"taxonomy 시트 복구 실패: {e}")

        return result

    def set_dropdown_validation(self, yyyymm: str) -> Dict:
        """
        검수 컬럼에 드롭다운 유효성 검사 설정
        - 대분류: _taxonomy 시트의 unique 값 목록
        - 중분류: _taxonomy 시트의 unique 값 목록 (전체)

        Args:
            yyyymm: 대상 월 (탭 이름)

        Returns:
            dict: {"success": bool, "error": str}
        """
        result = {
            "success": False,
            "error": "",
        }

        if not self.is_available():
            result["error"] = self._init_error or "GSpread 미초기화"
            return result

        try:
            worksheet = self.spreadsheet.worksheet(yyyymm)

            # 현재 데이터 행 수 확인
            data = worksheet.get_all_values()
            if len(data) < 2:
                result["error"] = "데이터 없음"
                return result

            num_rows = len(data)
            header = data[0]

            # _taxonomy 시트에서 대분류/중분류 값 가져오기
            tax_worksheet = self.spreadsheet.worksheet("_taxonomy")
            tax_data = tax_worksheet.get_all_values()

            if len(tax_data) < 2:
                result["error"] = "_taxonomy 시트에 데이터 없음"
                return result

            tax_header = tax_data[0]
            tax_rows = tax_data[1:]
            tax_df = pd.DataFrame(tax_rows, columns=tax_header)

            # 대분류/중분류 unique 값 추출
            subject_majors = sorted([x for x in tax_df["주제_대분류"].dropna().unique().tolist() if x.strip()])
            subject_minors = sorted([x for x in tax_df["주제_중분류"].dropna().unique().tolist() if x.strip()])
            work_majors = sorted([x for x in tax_df["작업유형_대분류"].dropna().unique().tolist() if x.strip()])
            work_minors = sorted([x for x in tax_df["작업유형_중분류"].dropna().unique().tolist() if x.strip()])

            # 드롭다운 목록 로깅
            logger.info(f"=== 드롭다운 목록 ===")
            logger.info(f"[검수_주제 대분류] {subject_majors}")
            logger.info(f"[검수_주제 중분류] {subject_minors}")
            logger.info(f"[검수_작업유형 대분류] {work_majors}")
            logger.info(f"[검수_작업유형 중분류] {work_minors}")
            logger.info(f"[검수완료] ['Y']")

            # 드롭다운 설정
            validations = {
                "검수_주제 대분류": subject_majors,
                "검수_주제 중분류": subject_minors,
                "검수_작업유형 대분류": work_majors,
                "검수_작업유형 중분류": work_minors,
                "검수완료": ["Y"],
            }

            for col_name, values in validations.items():
                if col_name not in header or not values:
                    continue

                col_idx = header.index(col_name) + 1

                body = {
                    "requests": [
                        {
                            "setDataValidation": {
                                "range": {
                                    "sheetId": worksheet.id,
                                    "startRowIndex": 1,
                                    "endRowIndex": num_rows,
                                    "startColumnIndex": col_idx - 1,
                                    "endColumnIndex": col_idx,
                                },
                                "rule": {
                                    "condition": {
                                        "type": "ONE_OF_LIST",
                                        "values": [{"userEnteredValue": v} for v in values]
                                    },
                                    "showCustomUi": True,
                                    "strict": False,
                                }
                            }
                        }
                    ]
                }
                self.spreadsheet.batch_update(body)
                logger.debug(f"드롭다운 설정: {col_name} -> {len(values)}개")

            result["success"] = True
            logger.info(f"드롭다운 유효성 검사 설정 완료: {yyyymm}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"드롭다운 설정 실패: {e}")

        return result


def get_gspread_manager(
    credentials_path: str = None,
    spreadsheet_id: str = None,
) -> GSpreadManager:
    """
    GSpreadManager 인스턴스 생성
    
    환경변수:
      - GOOGLE_SHEETS_CREDENTIALS: 인증 JSON 파일 경로
      - GOOGLE_SHEETS_ID: 스프레드시트 ID
    """
    return GSpreadManager(
        credentials_path=credentials_path,
        spreadsheet_id=spreadsheet_id,
    )