# AI-text-summarize  
**AI 모델 서빙 2. 텍스트 요약**

텍스트 요약 모델을 마이크로서비스 형태로 서빙하는 프로젝트
FastAPI 기반으로 구성되어 있으며, 입력된 긴 문장을 압축하여 핵심 내용을 추출하는 기능을 제공
Kubernetes 환경에서 배포 가능하도록 설계되었으며, 추론 API는 RESTful 방식으로 제공

---

## 프로젝트 개요

- 자연어 입력을 받아 핵심 요약 결과를 반환하는 API 제공
- FastAPI 기반 REST API 서버로, 추후 TorchServe 전환 및 KServe 배포를 위한 구조 반영
- 마이크로서비스 아키텍처에 적합한 경량화 및 모듈화 설계

---

## 로컬 실행 (개발용)

### 1. Python 3.10 가상환경 설정 및 실행

```bash
# 가상환경 생성
python3.10 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# FastAPI 서버 실행
python3.10 -m uvicorn app_summarize:app --reload --port 9001
