# /home/ssm-user/jupyter/batch/s3_uploader.py
# -*- coding: utf-8 -*-
"""
S3 업로드 모듈

output 파일을 S3에 업로드합니다.

S3 경로 구조:
  s3://hdcl-csp-prod/stat/voc/{yyyymm}/{building_id}/
    ├── tagged_{building_id}_{yyyymm}_{run_id}.csv
    └── dashboard_{building_id}_{yyyymm}_{run_id}.html
"""
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

KST = timezone(timedelta(hours=9))

logger = logging.getLogger("run_monthly")


class S3Uploader:
    """S3 업로드 클래스"""
    
    def __init__(
        self,
        bucket_name: str = "hdcl-csp-prod",
        prefix: str = "stat/voc",
        region: str = "ap-northeast-2",
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self.region = region
        self.s3_client = None
        self._initialized = False
        self._init_error = None
        self._init_client()
    
    def _init_client(self):
        """S3 클라이언트 초기화 (IAM Role 또는 환경변수 사용)"""
        if not BOTO3_AVAILABLE:
            self._init_error = "boto3 패키지가 설치되지 않았습니다."
            logger.warning(f"S3 업로더 초기화 실패: {self._init_error}")
            return
        
        try:
            self.s3_client = boto3.client("s3", region_name=self.region)
            # 연결 테스트 (버킷 존재 확인)
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self._initialized = True
            logger.debug(f"S3 클라이언트 초기화 완료: s3://{self.bucket_name}/{self.prefix}/")
        except NoCredentialsError:
            self._init_error = "AWS 자격 증명을 찾을 수 없습니다."
            logger.warning(f"S3 업로더 초기화 실패: {self._init_error}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "403":
                self._init_error = f"S3 버킷 접근 권한 없음: {self.bucket_name}"
            elif error_code == "404":
                self._init_error = f"S3 버킷이 존재하지 않음: {self.bucket_name}"
            else:
                self._init_error = str(e)
            logger.warning(f"S3 업로더 초기화 실패: {self._init_error}")
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"S3 업로더 초기화 실패: {self._init_error}")
    
    def is_available(self) -> bool:
        """S3 클라이언트 사용 가능 여부"""
        return self._initialized and self.s3_client is not None
    
    def get_init_error(self) -> Optional[str]:
        """초기화 실패 원인 반환"""
        return self._init_error
    
    def _get_content_type(self, filepath: str) -> str:
        """파일 확장자에 따른 Content-Type 반환"""
        ext = os.path.splitext(filepath)[1].lower()
        content_type_map = {
            ".csv": "text/csv; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        return content_type_map.get(ext, "application/octet-stream")
    
    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> Dict:
        """
        단일 파일 업로드
        
        Args:
            local_path: 로컬 파일 경로
            s3_key: S3 키 (버킷 이름 제외)
            content_type: Content-Type (None이면 자동 추론)
        
        Returns:
            dict: {"success": bool, "s3_uri": str, "error": str}
        """
        if not self.is_available():
            return {
                "success": False,
                "s3_uri": "",
                "error": self._init_error or "S3 클라이언트 미초기화",
            }
        
        if not os.path.exists(local_path):
            return {
                "success": False,
                "s3_uri": "",
                "error": f"파일 없음: {local_path}",
            }
        
        if content_type is None:
            content_type = self._get_content_type(local_path)
        
        try:
            extra_args = {"ContentType": content_type}
            
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs=extra_args,
            )
            
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            return {"success": True, "s3_uri": s3_uri, "error": ""}
        
        except ClientError as e:
            return {"success": False, "s3_uri": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "s3_uri": "", "error": str(e)}

    def download_file(
        self,
        s3_key: str,
        local_path: str,
    ) -> Dict:
        """
        S3에서 파일 다운로드
        
        Args:
            s3_key: S3 키 (버킷 이름 제외)
            local_path: 저장할 로컬 경로
        
        Returns:
            dict: {"success": bool, "local_path": str, "error": str}
        """
        if not self.is_available():
            return {
                "success": False,
                "local_path": "",
                "error": self._init_error or "S3 클라이언트 미초기화",
            }
        
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=local_path,
            )
            return {"success": True, "local_path": local_path, "error": ""}
        
        except ClientError as e:
            return {"success": False, "local_path": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "local_path": "", "error": str(e)}

    def find_latest_tagged_csv(
        self,
        building_id: int,
        yyyymm: str,
    ) -> Optional[str]:
        """
        S3에서 가장 최근 tagged CSV의 S3 key 찾기
        
        Returns:
            S3 key 또는 None
        """
        if not self.is_available():
            return None
        
        prefix = f"{self.prefix}/{yyyymm}/{building_id}/tagged_{building_id}_{yyyymm}_"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )
            
            if "Contents" not in response:
                return None
            
            # 가장 최근 파일 (LastModified 기준)
            files = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)
            if files:
                return files[0]["Key"]
            return None
        
        except Exception as e:
            logger.warning(f"S3 파일 목록 조회 실패: {e}")
            return None
    
    def build_s3_key(
        self,
        yyyymm: str,
        building_id: int,
        filename: str,
    ) -> str:
        """
        S3 키 생성
        
        구조: {prefix}/{yyyymm}/{building_id}/{filename}
        예시: stat/voc/202512/95/tagged_95_202512_auto_20260101_020000.csv
        """
        return f"{self.prefix}/{yyyymm}/{building_id}/{filename}"
    
    def upload_building_outputs(
        self,
        building_id: int,
        yyyymm: str,
        run_id: str,
        tagging_dir: str,
        html_dir: str,
    ) -> Dict:
        """
        Building별 output 파일 업로드
        
        로컬:
          - {tagging_dir}/tagged_{building_id}_{yyyymm}_{run_id}.csv
          - {html_dir}/dashboard_{building_id}_{yyyymm}_{run_id}.html
        
        S3:
          - s3://{bucket}/{prefix}/{yyyymm}/{building_id}/tagged_...csv
          - s3://{bucket}/{prefix}/{yyyymm}/{building_id}/dashboard_...html
        
        Returns:
            dict: {
                "success": bool,
                "csv": {"uploaded": bool, "s3_uri": str, "error": str},
                "html": {"uploaded": bool, "s3_uri": str, "error": str},
            }
        """
        result = {
            "success": False,
            "csv": {"uploaded": False, "s3_uri": "", "error": ""},
            "html": {"uploaded": False, "s3_uri": "", "error": ""},
        }
        
        if not self.is_available():
            error_msg = self._init_error or "S3 클라이언트 미초기화"
            result["csv"]["error"] = error_msg
            result["html"]["error"] = error_msg
            return result
        
        # CSV 업로드
        csv_filename = f"tagged_{building_id}_{yyyymm}_{run_id}.csv"
        csv_local = os.path.join(tagging_dir, csv_filename)
        csv_s3_key = self.build_s3_key(yyyymm, building_id, csv_filename)
        
        if os.path.exists(csv_local):
            csv_result = self.upload_file(csv_local, csv_s3_key)
            result["csv"] = {
                "uploaded": csv_result["success"],
                "s3_uri": csv_result["s3_uri"],
                "error": csv_result["error"],
            }
        else:
            result["csv"]["error"] = f"파일 없음: {csv_local}"
        
        # HTML 업로드
        html_filename = f"dashboard_{building_id}_{yyyymm}_{run_id}.html"
        html_local = os.path.join(html_dir, html_filename)
        html_s3_key = self.build_s3_key(yyyymm, building_id, html_filename)
        
        if os.path.exists(html_local):
            html_result = self.upload_file(html_local, html_s3_key)
            result["html"] = {
                "uploaded": html_result["success"],
                "s3_uri": html_result["s3_uri"],
                "error": html_result["error"],
            }
        else:
            result["html"]["error"] = f"파일 없음: {html_local}"
        
        # 전체 성공 여부 (CSV와 HTML 모두 성공)
        result["success"] = result["csv"]["uploaded"] and result["html"]["uploaded"]
        
        return result
    
    def upload_batch_outputs(
        self,
        results: List[Dict],
        yyyymm: str,
        run_id: str,
        tagging_dir: str,
        html_dir: str,
    ) -> Dict:
        """
        배치 전체 결과물 업로드
        
        Args:
            results: process_building() 결과 리스트
            yyyymm: 대상 월 (예: "202512")
            run_id: 실행 ID
            tagging_dir: CSV 디렉토리
            html_dir: HTML 디렉토리
        
        Returns:
            dict: {
                "total": int,
                "success": int,
                "failed": int,
                "skipped": int,
                "details": list,
            }
        """
        upload_results = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
        }
        
        if not self.is_available():
            logger.warning(f"S3 업로드 건너뜀: {self._init_error}")
            upload_results["skipped"] = len([r for r in results if r.get("success")])
            return upload_results
        
        # 성공한 Building만 업로드
        successful_buildings = [r for r in results if r.get("success", False)]
        upload_results["total"] = len(successful_buildings)
        
        for bldg_result in successful_buildings:
            building_id = bldg_result["building_id"]
            building_name = bldg_result.get("building_name", "")
            
            logger.info(f"    [{building_id}] {building_name} S3 업로드 중...")
            
            upload_result = self.upload_building_outputs(
                building_id=building_id,
                yyyymm=yyyymm,
                run_id=run_id,
                tagging_dir=tagging_dir,
                html_dir=html_dir,
            )
            
            detail = {
                "building_id": building_id,
                "building_name": building_name,
                "success": upload_result["success"],
                "csv": upload_result["csv"],
                "html": upload_result["html"],
            }
            upload_results["details"].append(detail)
            
            if upload_result["success"]:
                upload_results["success"] += 1
                logger.info(f"      [OK] CSV : {upload_result['csv']['s3_uri']}")
                logger.info(f"      [OK] HTML: {upload_result['html']['s3_uri']}")
            else:
                upload_results["failed"] += 1
                if not upload_result["csv"]["uploaded"]:
                    logger.warning(f"      [FAIL] CSV : {upload_result['csv']['error']}")
                if not upload_result["html"]["uploaded"]:
                    logger.warning(f"      [FAIL] HTML: {upload_result['html']['error']}")
        
        return upload_results

def get_s3_uploader(
    bucket_name: str = None,
    prefix: str = None,
) -> Optional[S3Uploader]:
    """
    환경변수 또는 인자로 S3Uploader 인스턴스 생성
    
    환경변수:
      - S3_BUCKET: 버킷 이름 (기본: hdcl-csp-prod)
      - S3_PREFIX: 경로 prefix (기본: stat/voc)
      - AWS_DEFAULT_REGION: 리전 (기본: ap-northeast-2)
    
    Returns:
        S3Uploader 또는 None (초기화 실패 시에도 인스턴스 반환, is_available()로 확인)
    """
    bucket = bucket_name or os.getenv("S3_BUCKET", "hdcl-csp-prod")
    prefix = prefix or os.getenv("S3_PREFIX", "stat/voc")
    region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
    
    return S3Uploader(bucket_name=bucket, prefix=prefix, region=region)