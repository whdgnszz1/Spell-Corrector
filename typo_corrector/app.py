from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import os
from utils.embedding_manager import FastEmbeddingManager
from utils.correction_utils import find_best_correction

app = FastAPI()

# 모델과 토크나이저 로드
model_path = "./models"
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

# 임베딩 관리자 초기화
embedding_model = "BAAI/bge-m3"
precomputed_dir = "./embeddings"
embedding_manager = None

try:
    embedding_manager = FastEmbeddingManager(model_name=embedding_model, precomputed_dir=precomputed_dir)
    print(f"Embedding manager initialized with model: {embedding_model}")

    # 후보 문장 설정
    if not hasattr(embedding_manager, 'candidates') or embedding_manager.candidates is None:
        print("Setting candidates for embedding manager...")
        embedding_manager.candidates = candidates

    # 임베딩 미리 계산 (없는 경우)
    if precomputed_dir and not os.path.exists(os.path.join(precomputed_dir, 'embeddings.npy')):
        print("Precomputing embeddings...")
        os.makedirs(precomputed_dir, exist_ok=True)
        embedding_manager.precompute_embeddings(candidates, output_dir=precomputed_dir)

except Exception as e:
    print(f"Error initializing embedding manager: {e}")
    print("Falling back to basic methods.")


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

        raw_prd_sentence = predictions[0]

        # 임베딩 기반 최종 예측 문장 선택
        if embedding_manager:
            # 최종 교정 텍스트는 사용하지 않고, 후보 목록만 가져옴
            _, top_candidates = find_best_correction(
                input.text,
                predictions,
                embedding_manager,
                correct_label=None,  # API에서는 정답 레이블 없음
                top_k=10,
                length_tolerance=5
            )

            # 상위 후보 정보 구성
            top_candidates_info = []
            for cand_info in top_candidates[:3]:
                top_candidates_info.append({
                    "text": cand_info[0],
                    "length_diff": cand_info[1],
                    "edit_distance": cand_info[2],
                    "score": cand_info[3],
                    "char_similarity": cand_info[4],
                    "semantic_similarity": cand_info[5],
                    "is_model_prediction": cand_info[6]
                })

            # 최종 교정 텍스트를 top_candidates에서 가져옴
            final_prd_sentence = top_candidates[0][0] if top_candidates else raw_prd_sentence
        else:
            # 임베딩 관리자가 없는 경우 기본 방식 사용
            final_prd_sentence = raw_prd_sentence
            top_candidates_info = []

        # 결과 반환
        response = {
            "input_text": input.text,
            "model_prediction": raw_prd_sentence,
            "corrected_text": final_prd_sentence,
            "all_predictions": predictions[:3]  # 상위 3개 원시 예측만 반환
        }

        # 임베딩 기반 후보가 있으면 추가
        if embedding_manager and top_candidates_info:
            response["top_candidates"] = top_candidates_info

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
