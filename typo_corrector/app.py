from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
from utils import find_closest_candidate, select_best_prediction

app = FastAPI()

# 모델과 토크나이저 로드
model_path = "./models/checkpoint-7700"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 후보군 데이터 로드
candidate_file = "./data/datasets/dataset_candidate.json"
with open(candidate_file, 'r') as f:
    json_dataset = json.load(f)
candidates = [data['annotation']['cor_sentence'] for data in json_dataset['data']]


# 입력 데이터 모델 정의
class TextInput(BaseModel):
    text: str


# 엔드포인트
@app.post("/correct")
async def correct_text(input: TextInput):
    try:
        # 입력 문장 토큰화
        tokenized = tokenizer(input.text, return_tensors="pt", max_length=128, truncation=True)
        input_ids = tokenized["input_ids"].to(device)

        # 모델로 여러 개의 문장 생성
        with torch.no_grad():
            res = model.generate(
                inputs=input_ids,
                num_beams=10,
                num_return_sequences=5,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=2.5,
                length_penalty=0.5,
                no_repeat_ngram_size=3,
                max_length=input_ids.size()[1] + 2,
                early_stopping=True,
                min_length=max(1, input_ids.size()[1] - 5)
            ).cpu().tolist()

        # 생성된 문장 디코딩
        predictions = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in res]

        # 후보 문장의 평균 길이 계산
        avg_candidate_length = sum(len(c) for c in candidates) / len(candidates)

        # 최적의 원시 예측 선택
        raw_prd_sentence = select_best_prediction(predictions, candidates, n_gram=2,
                                                  avg_candidate_length=avg_candidate_length)

        # 최종 예측 문장 선택
        final_prd_sentence, top_candidates = find_closest_candidate(input.text, predictions, candidates)

        # 결과 반환
        return {
            "raw_prd_sentence": raw_prd_sentence,
            "corrected_text": final_prd_sentence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
