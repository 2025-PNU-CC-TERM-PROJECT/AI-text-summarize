from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import re
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


app = FastAPI()
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization', cache_dir='./kobart_model')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization', cache_dir='./kobart_model')

def clean_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text) 
    return text.strip()

@app.post("/summarize")
async def summarize(text: str = Body(..., embed=True)):
    try:
        if not text or len(text.strip()) < 20:
            return JSONResponse(status_code=400, content={"error": "텍스트가 너무 짧습니다. 최소 20자 이상 입력해주세요."})

        
        input_len = len(text.split())
        max_len = min(128, max(40, int(input_len * 0.5)))  
        min_len = max(20, int(max_len * 0.5))

        inputs = tokenizer(text, return_tensors="pt")

        summary_ids = model.generate(
                inputs['input_ids'],
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                length_penalty=1.2,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


        return JSONResponse(content={"summary": summary_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
