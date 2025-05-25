FROM python:3.10-slim

WORKDIR /app

# 스크립트 및 설정 복사
COPY app_summarize.py .
COPY requirements.txt .

# 의존성 설치 및 KoBART 모델 사전 다운로드
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "\
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration; \
PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization', cache_dir='./kobart_model'); \
BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization', cache_dir='./kobart_model')" && \
    rm -rf /root/.cache/pip

# 모델 캐시 폴더 복사 (필요시 이미지 빌드 후 COPY 가능)
COPY kobart_model ./kobart_model

# 실행
CMD ["python", "app_summarize.py"]