from kserve import Model, ModelServer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

class KoBARTSummaryModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            'digit82/kobart-summarization', cache_dir='./kobart_model'
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            'digit82/kobart-summarization', cache_dir='./kobart_model'
        )
        self.ready = True
        return self

    def predict(self, request: dict, context: dict = None) -> dict:
        try:
            text = request.get("instances", [{}])[0].get("text", "")
            if not text or len(text.strip()) < 20:
                return {"error": "텍스트가 너무 짧습니다. 최소 20자 이상 입력해주세요."}

            input_len = len(text.split())
            max_len = min(128, max(40, int(input_len * 0.5)))
            min_len = max(20, int(max_len * 0.5))

            inputs = self.tokenizer(text, return_tensors="pt")

            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                length_penalty=1.2,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return {"summary": summary}
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    model = KoBARTSummaryModel("kobart-summary")
    model.load()
    ModelServer().start([model])