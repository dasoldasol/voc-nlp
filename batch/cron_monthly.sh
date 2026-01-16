#!/bin/bash
# /home/ssm-user/jupyter/batch/cron_monthly.sh
# 
# 월간 VOC 분석 배치 cron 실행 스크립트
# 매월 1일 02:00 실행 (전월 데이터, 모든 활성 Building)
#
# crontab 등록:
#   crontab -e
#   0 2 1 * * /home/ssm-user/jupyter/batch/cron_monthly.sh >> /home/ssm-user/jupyter/logs/cron_monthly.log 2>&1
#

# 경로 설정
BASE_DIR="/home/ssm-user/jupyter"
BATCH_DIR="${BASE_DIR}/batch"
LOG_DIR="${BASE_DIR}/logs"
PYTHON_BIN="/home/ssm-user/pyenv/bin/python3.11"

# 로그 디렉토리 생성
mkdir -p "${LOG_DIR}"

# 시작 로그
echo ""
echo "========================================"
echo "cron_monthly.sh 시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# run_monthly.py 실행 (전월 자동 + 모든 활성 Building)
cd "${BATCH_DIR}"
${PYTHON_BIN} run_monthly.py --all-buildings --auto-month --run-id-prefix cron

EXIT_CODE=$?

# 종료 로그
echo ""
echo "========================================"
echo "cron_monthly.sh 종료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "EXIT_CODE: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}