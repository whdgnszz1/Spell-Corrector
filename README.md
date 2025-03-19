# Seq2Seq 기반 한글 오타 교정 프로젝트

이 프로젝트는 Seq2Seq 모델을 활용하여 오류가 있는 한글 문장을 올바른 문장으로 교정하는 작업을 수행합니다. 
데이터 증강 기법을 통해 훈련 데이터를 확장하고, F0.5 점수를 기준으로 모델 성능을 평가합니다.

## 주요 기능
- **데이터 증강**: 한글 문장에 대해 대체, 삽입, 삭제, 교환 등의 증강 기법을 적용하여 오류 문장과 정답 문장 쌍을 생성합니다.
- **Seq2Seq 모델**: 사전 학습된 Seq2Seq 모델(예: BART, T5 등)을 사용하여 문장 교정을 수행합니다.
- **F0.5 점수 평가**: 모델의 예측 결과를 F0.5 점수로 평가하여 교정 성능을 측정합니다.

---

## 설치 및 환경 설정

### 필요한 패키지
프로젝트를 실행하기 위해 필요한 패키지들은 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어로 설치할 수 있습니다:
```bash
pip install -r requirements.txt
```

### 환경 요구 사항
- Python: 3.8 이상
- CUDA: GPU를 사용하려면 CUDA가 설치된 환경 필요 (선택 사항, fp16=True 설정 시 사용)
- 운영 체제: Windows, macOS, Linux 지원

---

## 1. 학습

### 1-1. train.py
모델 학습을 위한 메인 스크립트입니다.

- 데이터 로드 및 증강 (make_dataset)
- 데이터 전처리 및 토큰화 (preprocess_function)
- Seq2Seq 모델 학습 및 평가 (train, CustomSeq2SeqTrainer)

### 1-2. utils/train_util.py
한글 처리 및 데이터 증강을 위한 유틸리티 함수들이 포함되어 있습니다.
- decompose_hangul: 한글 문자를 초성, 중성, 종성으로 분리
- compose_hangul: 초성, 중성, 종성 인덱스를 조합하여 한글 문자 생성

증강 함수:
- augment_substitute: 인접 자판 키로 대체
- augment_insert: 랜덤 한글 문자 삽입
- augment_delete: 문자 삭제
- augment_transpose: 인접 문자 교환
- augment_sentence: 위 방법 중 하나를 무작위로 적용

---

## 학습 실행 방법

### 1. 데이터 준비
훈련 및 검증 데이터는 JSON 형식으로 준비해야 합니다. </br>
각 데이터는 annotation 필드 아래 err_sentence(오류 문장)와 cor_sentence(정답 문장)를 포함해야 합니다.

데이터 형식 예시:
```json
{
  "data": [
    {
      "annotation": {
        "err_sentence": "이 문장은 오타가 있습니댜",
        "cor_sentence": "이 문장은 오타가 있습니다"
      }
    },
    {
      "annotation": {
        "err_sentence": "맞춤법이 틀린 문장",
        "cor_sentence": "맞춤법이 맞는 문장"
      }
    }
  ]
}
```
- 훈련 데이터: train_data_path_list에 지정할 파일 경로
- 검증 데이터: validation_data_path_list에 지정할 파일 경로

### 2. 설정 파일 (config.yaml) 작성
모델 학습에 필요한 설정은 config.yaml 파일에 정의됩니다.

### 3. 학습 실행 명령어

```bash
python train.py --config-file config/base-config.yaml
```

---


## 2. 평가

### 2-1. evaluation.py
학습된 모델을 평가하기 위한 메인 스크립트입니다. 테스트 데이터셋을 로드하고, 모델을 사용해 오류 문장을 교정한 후, F0.5 점수를 계산하여 성능을 평가합니다.

- 데이터 로드: load_datasets 함수를 통해 테스트 데이터(test_file)와 후보 데이터(candidate_file)를 로드합니다.
- 모델 예측: Seq2Seq 모델을 사용해 오류 문장에 대한 예측을 생성하며, num_beams=10, num_return_sequences=5 등의 설정으로 여러 예측을 생성합니다.
- 후처리: post_process_prediction으로 예측 문장을 정제하고, select_best_prediction과 find_closest_candidate를 통해 최적의 예측을 선택합니다.
- 성능 평가: calc_f_05 함수로 precision, recall, F0.5 점수를 계산합니다.
- 결과 저장: 평가 결과를 CSV 파일로 저장하며, precision이 1이 아닌 경우 별도의 파일에 세부 정보를 기록합니다.

### 2-2. utils/eval_util.py
평가에 필요한 유틸리티 함수들이 포함되어 있습니다.

- get_ngram: 입력 텍스트에서 n-gram을 생성합니다. (예: "안녕하세요" → ['안녕', '녕하', '하세', '세요'] for n=2)
- calc_f_05: 정답 문장과 예측 문장의 F0.5 점수를 계산합니다. Precision과 recall을 기반으로 β=0.5로 가중치를 둡니다.
- is_hangul: 텍스트가 한글인지 확인합니다. (유니코드 범위 \uAC00 ~ \uD7A3 사용)
- select_best_prediction: 모델의 여러 예측 중 최적의 예측을 선택합니다. Precision, recall proxy, 길이 차이를 고려한 점수로 평가합니다.
- find_closest_candidate: 오류 문장과 예측을 바탕으로 후보 문장 중 가장 적합한 문장을 선택합니다. 길이 차이, 편집 거리(Levenshtein Distance), n-gram 유사도, 자모 유사도 등을 종합적으로 계산합니다.

---

## 평가 실행 방법

### 1. 평가 데이터 준비
테스트 데이터는 JSON 형식으로 준비해야 하며, err_sentence와 cor_sentence를 포함해야 합니다. </br>
후보 데이터는 선택적으로 ./data/datasets/dataset_candidate.json에 준비할 수 있습니다.

### 2. 평가 실행 명령어

```bash
python evaluation.py --gpu_no 0 --model_path ./models/checkpoint-7700 --test_file ./data/test.json --eval_length 100 --save_path ./data/results
```

- --gpu_no: 사용할 GPU 번호 (예: 0). CPU 사용 시 cpu로 설정됩니다.
- --model_path: 평가할 모델의 경로 (예: ./models/checkpoint-7700).
- --test_file: 테스트 데이터 파일 경로 (예: ./data/test.json).
- --eval_length: 평가할 데이터 개수 (선택 사항, 생략 시 전체 데이터 사용).
- --save_path: 평가 결과를 저장할 경로 (예: ./data/results).
- -pb: 진행 바를 비활성화하려면 추가 (기본값은 활성화).

---

## 3. 애플리케이션

### 3-1. app.py
학습된 모델을 사용하여 실시간으로 문장을 교정하는 FastAPI 애플리케이션입니다. </br>
사용자는 POST 요청을 통해 오류 문장을 전송하고, 교정된 문장을 응답으로 받을 수 있습니다.

- 모델 로드: 학습된 Seq2Seq 모델과 토크나이저를 ./models/checkpoint-7700 경로에서 로드합니다. GPU가 사용 가능하면 CUDA를 활용하고, 그렇지 않으면 CPU를 사용합니다.
- 후보 데이터 로드: ./data/datasets/dataset_candidate.json 파일에서 후보 문장 데이터를 로드하여 예측 시 참고합니다.
- 문장 교정 엔드포인트: /correct 엔드포인트로 POST 요청을 받아 오류 문장을 교정합니다. </br>
모델은 여러 예측을 생성하고, select_best_prediction과 find_closest_candidate를 통해 최적의 예측을 선택합니다.
- 예측 생성: model.generate를 통해 최대 5개의 예측 문장을 생성하며, num_beams=10, do_sample=True, temperature=0.7 등의 파라미터로 다양성과 품질을 조정합니다.
- 결과 반환: 원시 예측 문장(raw_prd_sentence)과 최종 교정된 문장(corrected_text)을 JSON 형식으로 반환합니다.

---

## 애플리케이션 실행 방법

### 1. 어플리케이션 실행 명령어

```bash
python app.py
```

### 2. API 사용 방법
엔드포인트: /correct
메서드: POST
요청 본문:
```json
{
  "text": "오타가 있는 문장"
}
```

응답:
```
{
  "raw_prd_sentence": "모델의 초기 예측 문장",
  "corrected_text": "최종 교정된 문장"
}
```
