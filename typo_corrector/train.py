"""
시퀀스-투-시퀀스(Seq2Seq) 모델을 사용한 텍스트 교정 모델 학습 스크립트

이 스크립트는 오류가 있는 텍스트를 정확한 텍스트로 변환하는 Seq2Seq 모델을 훈련합니다.
주요 기능:
- 다양한 데이터 증강 기법을 통한 학습 데이터 확장
- 사전 훈련된 Transformer 모델 활용
- 다양한 평가 메트릭(F0.5, BLEU, GLEU, 정확한 일치율) 지원
- 조기 종료(early stopping) 및 최적 모델 저장
"""

# 필요한 라이브러리 임포트
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertModel  # 트랜스포머 모델 관련 클래스 (토크나이저, Seq2Seq 모델, BERT 모델)
import datasets  # Hugging Face 데이터셋 라이브러리 (데이터 로딩 및 처리)
from transformers import DataCollatorForSeq2Seq  # 배치 데이터 처리 (패딩, 동적 배치 등)
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, get_scheduler  # Seq2Seq 모델 훈련 관련 클래스 및 스케줄러
import os  # 파일 시스템 조작 (디렉토리 생성, 환경 변수 설정 등)
import sys  # 시스템 관련 기능 (종료, 인자 처리 등)
import json  # JSON 파일 처리
from datetime import datetime  # 날짜/시간 처리 (로깅)
import argparse  # 명령줄 인자 파싱
from omegaconf import OmegaConf  # YAML 설정 파일 처리 라이브러리
import random  # 난수 생성 (데이터 증강, 샘플링)
import numpy as np  # 수치 계산
import torch  # PyTorch 딥러닝 프레임워크
from torch.utils.tensorboard import SummaryWriter  # 텐서보드 로깅 (학습 과정 시각화)
from utils.generators import augment_sentence  # 데이터 증강 유틸리티
from utils.train_utils import advanced_augment_data, back_translation_augment  # 데이터 증강 유틸리티
from utils.eval_utils import calc_precision_recall_f05, calc_bleu, calc_gleu  # 평가 메트릭 계산 유틸리티


def seed_everything(seed):
    """
    재현성을 위한 시드 설정 함수

    모든 난수 생성 관련 라이브러리(random, numpy, torch 등)의 시드를 고정하여
    실험의 재현성을 확보합니다.

    Args:
        seed (int): 시드 값
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 파이썬 해시 시드 설정
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경을 위한 설정
    torch.backends.cudnn.deterministic = True  # CuDNN 백엔드의 비결정적 알고리즘 비활성화
    torch.backends.cudnn.benchmark = False  # CuDNN 벤치마크 기능 비활성화


# 데이터 증강을 별도의 함수로 분리
def augment_data(original_sentences, augment_prob=0.1, advanced=False):
    """
    원본 문장을 증강하여 오류 문장과 정답 문장 쌍을 생성

    데이터 증강을 통해 학습 데이터를 확장하고 모델의 일반화 성능을 향상시킵니다.
    원본 문장을 인위적으로 오류가 있는 문장으로 변환하여 입력-타겟 쌍을 생성합니다.

    Args:
        original_sentences (list): 원본 정답 문장 리스트
        augment_prob (float): 증강 확률 (기본값: 0.1) - 각 토큰이 변형될 확률
        advanced (bool): 고급 증강 기법 사용 여부 (기본값: False)

    Returns:
        dict: 증강된 오류 문장과 정답 문장 리스트를 담은 딕셔너리
    """
    if advanced:
        # 고급 증강 기법 사용 (utils.train_utils에 구현됨)
        return advanced_augment_data(original_sentences, augment_prob)

    # 기본 증강 로직
    augmented_data = {'err_sentence': [], 'cor_sentence': []}
    for cor in original_sentences:
        # 각 문장에 대해 augment_sentence 함수를 사용하여 오류 버전 생성
        err = augment_sentence(cor, prob=augment_prob)
        augmented_data['err_sentence'].append(err)
        augmented_data['cor_sentence'].append(cor)
    return augmented_data


# 데이터셋 생성 함수
def make_dataset(train_data_path_list, validation_data_path_list, config):
    """
    훈련 및 검증 데이터셋을 로드하고, 훈련 데이터에 대해 증강을 수행하여 데이터셋을 생성

    여러 JSON 파일에서 데이터를 로드하고, 다양한 증강 기법을 적용하여 학습 데이터를 확장합니다.
    또한 토크나이저를 사용해 최대 시퀀스 길이를 계산합니다.

    Args:
        train_data_path_list (list): 훈련 데이터 파일 경로 리스트
        validation_data_path_list (list): 검증 데이터 파일 경로 리스트
        config (OmegaConf): 설정 객체

    Returns:
        datasets.DatasetDict: 훈련 및 검증 데이터셋
        int: 계산된 최대 길이 (토큰화 시 사용)
    """
    # 데이터 저장용 딕셔너리 초기화
    loaded_data_dict = {
        'train': {'err_sentence': [], 'cor_sentence': []},
        'validation': {'err_sentence': [], 'cor_sentence': []}
    }

    # 훈련 데이터 로드
    for i, train_data_path in enumerate(train_data_path_list):
        try:
            with open(train_data_path, 'r', encoding='utf-8') as f:
                _temp_json = json.load(f)
            # 오류 문장과 정답 문장을 리스트에 추가 (JSON 데이터에서 추출)
            loaded_data_dict['train']['err_sentence'].extend(
                list(map(lambda x: str(x['annotation']['err_sentence']), _temp_json['data'])))
            loaded_data_dict['train']['cor_sentence'].extend(
                list(map(lambda x: str(x['annotation']['cor_sentence']), _temp_json['data'])))
            print(f'train data {i} : {len(_temp_json["data"])} samples loaded')
        except FileNotFoundError:
            print(f"Error: Training file {train_data_path} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {train_data_path}.")
            sys.exit(1)

    # 원본 정답 문장 저장 (데이터 증강을 위해)
    original_cor_sentences = loaded_data_dict['train']['cor_sentence'].copy()

    # 훈련 데이터 증강 - 인위적으로 오류가 있는 문장 생성
    augmented_train = augment_data(
        original_cor_sentences,
        augment_prob=config.augment_prob,
        advanced=config.use_advanced_augmentation
    )
    loaded_data_dict['train']['err_sentence'].extend(augmented_train['err_sentence'])
    loaded_data_dict['train']['cor_sentence'].extend(augmented_train['cor_sentence'])

    # Back-translation 증강 (활성화된 경우)
    # 다른 언어로 번역 후 다시 원래 언어로 번역하여 자연스러운 변형 생성
    if config.use_back_translation:
        print(f"Performing back-translation augmentation...")
        bt_augmented = back_translation_augment(
            original_cor_sentences,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            sample_ratio=config.bt_sample_ratio
        )
        loaded_data_dict['train']['err_sentence'].extend(bt_augmented['err_sentence'])
        loaded_data_dict['train']['cor_sentence'].extend(bt_augmented['cor_sentence'])
        print(f"Added {len(bt_augmented['err_sentence'])} back-translated samples")

    # 정확한 문장 추가 (원본 데이터의 일정 비율)
    # 모델이 이미 올바른 문장에 대해서는 변경하지 않도록 학습
    num_original = len(original_cor_sentences)
    num_to_add = int(num_original * config.identity_sample_ratio)
    indices = random.sample(range(num_original), num_to_add)
    for idx in indices:
        cor = original_cor_sentences[idx]
        loaded_data_dict['train']['err_sentence'].append(cor)  # 입력 문장과 타겟 문장이 동일
        loaded_data_dict['train']['cor_sentence'].append(cor)

    # 검증 데이터 로드
    for i, validation_data_path in enumerate(validation_data_path_list):
        try:
            with open(validation_data_path, 'r', encoding='utf-8') as f:
                _temp_json = json.load(f)
            loaded_data_dict['validation']['err_sentence'].extend(
                list(map(lambda x: str(x['annotation']['err_sentence']), _temp_json['data'])))
            loaded_data_dict['validation']['cor_sentence'].extend(
                list(map(lambda x: str(x['annotation']['cor_sentence']), _temp_json['data'])))
            print(f'validation data {i} : {len(_temp_json["data"])} samples loaded')
        except FileNotFoundError:
            print(f"Error: Validation file {validation_data_path} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {validation_data_path}.")
            sys.exit(1)

    # 데이터셋 객체 생성 (Hugging Face datasets 형식)
    dataset_dict = {
        split: datasets.Dataset.from_dict(data, split=split)
        for split, data in loaded_data_dict.items()
    }
    dataset = datasets.DatasetDict(dataset_dict)

    # 데이터 통계 출력
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Validation dataset size: {len(dataset['validation'])}")

    # 토크나이저 로드 및 최대 길이 계산
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    target_lengths = [len(tokenizer.encode(sent)) for sent in loaded_data_dict['train']['cor_sentence']]
    max_length = int(np.percentile(target_lengths, 95))  # 상위 95% 백분위수로 최대 길이 설정
    print(f"Calculated max_length: {max_length}")
    return dataset, max_length


# 데이터 전처리 함수
def preprocess_function(df, tokenizer, src_col, tgt_col, max_length):
    """
    데이터셋을 토큰화하고 모델 입력 형식으로 전처리

    텍스트 문장을 토큰화하고 패딩/잘라내기를 적용하여 일정한 길이의 시퀀스로 변환합니다.
    소스(오류 문장)와 타겟(정답 문장) 모두 토큰화합니다.

    Args:
        df (datasets.Dataset): 입력 데이터셋
        tokenizer (AutoTokenizer): 토크나이저 객체
        src_col (str): 소스 컬럼 이름 (오류 문장)
        tgt_col (str): 타겟 컬럼 이름 (정답 문장)
        max_length (int): 토큰화 시 최대 길이

    Returns:
        dict: 토큰화된 입력과 레이블
    """
    inputs = df[src_col]  # 소스 문장
    targets = df[tgt_col]  # 타겟 문장

    # 입력 문장 토큰화
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

    # 타겟 문장 토큰화
    labels = tokenizer(
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    # 레이블 설정 (모델이 이를 타겟으로 사용)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics_extended(eval_pred, tokenizer, n_gram=2):
    """
    평가 메트릭(F0.5 스코어 및 추가 메트릭)을 계산하는 확장 함수

    모델의 예측 성능을 평가하기 위한 여러 메트릭을 계산합니다:
    - F0.5 스코어: 정밀도에 더 가중치를 두는 F 스코어
    - BLEU 스코어: 기계 번역 평가에 널리 사용되는 메트릭
    - GLEU 스코어: 문법 오류 수정 작업에 적합한 메트릭
    - 정확한 일치율: 예측과 정답이 완벽히 일치하는 비율

    Args:
        eval_pred (tuple): 예측값과 레이블
        tokenizer: 토크나이저 객체
        n_gram (int): n-gram 크기 (기본값: 2)

    Returns:
        dict: 평가 메트릭 결과
    """
    predictions, labels = eval_pred
    # 모델 출력(토큰 ID)을 텍스트로 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # -100 토큰을 패드 토큰으로 변환 (transformers 라이브러리는 -100을 손실 계산에서 무시)
    labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 평가 메트릭 계산
    f_05_scores = []
    bleu_scores = []
    gleu_scores = []
    perfect_match_count = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        # F0.5 스코어 계산 (정밀도에 더 큰 가중치)
        precision, recall, f_05 = calc_precision_recall_f05(label, pred, n_gram)
        f_05_scores.append(f_05)

        # BLEU 스코어 계산 (n-gram 일치 기반 점수)
        bleu = calc_bleu(label, pred, n_gram)
        bleu_scores.append(bleu)

        # GLEU 스코어 계산 (문법 오류 수정을 위한 변형 BLEU)
        gleu = calc_gleu(label, pred, n_gram)
        gleu_scores.append(gleu)

        # 완벽한 일치 확인
        if pred.strip() == label.strip():
            perfect_match_count += 1

    # 결과 취합 - 각 메트릭의 평균 계산
    results = {
        "f_05": sum(f_05_scores) / len(f_05_scores) if f_05_scores else 0.0,
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "gleu": sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0,
        "exact_match": perfect_match_count / len(decoded_preds) if decoded_preds else 0.0
    }

    return results


# 커스텀 트레이너 클래스
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer를 상속하여 예측 단계에서 생성 파라미터를 사용자 정의하고
    조기 종료(early stopping) 기능을 추가한 클래스입니다.
    """

    def __init__(self, *args, calculated_max_length=None, early_stopping_patience=3, early_stopping_threshold=0.001,
                 **kwargs):
        """
        초기화 함수

        Args:
            calculated_max_length (int): 계산된 최대 시퀀스 길이
            early_stopping_patience (int): 개선 없이 기다릴 평가 횟수
            early_stopping_threshold (float): 개선으로 간주할 최소 임계값
            *args, **kwargs: 기본 Seq2SeqTrainer에 전달할 인수
        """
        super().__init__(*args, **kwargs)
        self.calculated_max_length = calculated_max_length
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # 최상의 메트릭 초기화 (greater_is_better에 따라 초기값 설정)
        self.best_metric = -float('inf') if self.args.greater_is_better else float('inf')
        self.patience_counter = 0  # 개선 없이 지난 평가 횟수
        self.best_model_checkpoint = None  # 최상의 모델 체크포인트 경로

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        예측 단계(eval/predict)에서 호출되는 함수를 오버라이드하여
        생성 매개변수(beam search, top-k, top-p 등)를 커스텀합니다.

        Args:
            model: 모델
            inputs: 입력 데이터
            prediction_loss_only: 손실만 계산할지 여부
            ignore_keys: 무시할 키 목록

        Returns:
            tuple: (손실, 생성된 토큰, 레이블)
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            # 생성 없이 손실만 계산하는 경우 기본 동작 수행
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # 입력 데이터 준비 (기기 이동, 텐서 변환 등)
        inputs = self._prepare_inputs(inputs)

        # 생성 매개변수 설정
        gen_kwargs = {
            "max_length": self.calculated_max_length,  # 최대 생성 길이
            "num_beams": self.args.generation_num_beams if hasattr(self.args, 'generation_num_beams') else 4,  # 빔 탐색 수
            "no_repeat_ngram_size": 2,  # 반복 n-gram 방지 크기
            "length_penalty": 2.0,  # 길이 페널티
            "early_stopping": True,  # 생성 조기 종료
            "do_sample": self.args.do_sample if hasattr(self.args, 'do_sample') else False,  # 샘플링 여부
            "top_k": self.args.top_k if hasattr(self.args, 'top_k') else 50,  # top-k 샘플링
            "top_p": self.args.top_p if hasattr(self.args, 'top_p') else 1.0,  # top-p 샘플링
        }

        # 텍스트 생성
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        labels = inputs["labels"]
        return (None, generated_tokens, labels)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """
        로깅, 저장, 평가가 필요한 시점에 호출되는 함수를 오버라이드하여
        조기 종료 로직을 추가합니다.

        Returns:
            결과: 기본 함수의 결과 반환
        """
        # 기본 동작 수행 (로깅, 저장, 평가)
        result = super()._maybe_log_save_evaluate(*args, **kwargs)

        # 최상의 모델 로드 옵션이 활성화되고 체크포인트가 있는 경우
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # 최상의 모델 선택에 사용할 메트릭 확인
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            # 현재 메트릭 가져오기
            if hasattr(self.state, 'metrics') and self.state.metrics is not None:
                current_metric = self.state.metrics.get(metric_to_check)
                if current_metric is not None:
                    # 최대화/최소화 여부에 따른 비교 연산자 선택
                    operator = np.greater if self.args.greater_is_better else np.less

                    # 현재 메트릭이 최상의 메트릭보다 더 좋은지 확인
                    if operator(current_metric, self.best_metric):
                        # 메트릭이 개선된 경우
                        self.best_metric = current_metric
                        self.patience_counter = 0  # 인내심 카운터 리셋
                        self.best_model_checkpoint = self.state.best_model_checkpoint
                    else:
                        # 메트릭이 개선되지 않은 경우
                        self.patience_counter += 1
                        print(
                            f"No improvement in {metric_to_check}. Patience: {self.patience_counter}/{self.early_stopping_patience}")

                        # 지정된 횟수를 초과한 경우 학습 중단
                        if self.patience_counter >= self.early_stopping_patience:
                            self.control.should_training_stop = True
                            print(
                                f"Early stopping triggered after {self.patience_counter} evaluations without improvement.")
        return result


# 모델 로드 함수
def load_model(config):
    """
    설정에 따라 적절한 모델을 로드하는 함수

    지정된 사전 훈련 모델을 로드하고 필요한 설정을 적용합니다.

    Args:
        config (OmegaConf): 설정 객체

    Returns:
        model: 로드된 모델
        tokenizer: 토크나이저
    """
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Start ======')

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # 모델 로드 - Transformer 또는 일반 Seq2Seq
    if config.use_transformer:
        print(f"Loading Transformer model: {config.pretrained_model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name)
    else:
        print(f"Loading standard Seq2Seq model: {config.pretrained_model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name)

    # 모델 매개변수 수 계산 (모델 크기 확인)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Finished ======')

    return model, tokenizer


# 학습 함수
def train(config):
    """
    모델을 로드하고 데이터셋을 준비하여 학습을 수행하는 메인 함수

    전체 학습 프로세스를 조정하고 실행합니다:
    1. 시드 설정
    2. 데이터셋 준비
    3. 모델 및 토크나이저 로드
    4. 데이터 전처리
    5. 학습 설정 및 실행
    6. 모델 저장 및 평가

    Args:
        config (OmegaConf): 설정 객체

    Returns:
        trainer: 학습된 트레이너 객체
    """
    # 재현성을 위한 시드 설정
    seed_everything(config.seed if hasattr(config, 'seed') else 42)

    # 텐서보드 설정 (학습 로그 시각화)
    writer = SummaryWriter(log_dir=config.logging_dir)

    # 데이터셋 로드
    print('====== Data Load Start ======')
    dataset, calculated_max_length = make_dataset(
        config.train_data_path_list,
        config.validation_data_path_list,
        config
    )
    print('====== Data Load Finished ======')

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model(config)

    # 데이터 콜레이터 초기화 (배치 처리를 위한 객체)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100  # 손실 계산 시 무시할 패딩 토큰 ID
    )

    print('====== Data Preprocessing Start ======')
    # 데이터셋 토큰화
    dataset_tokenized = dataset.map(
        lambda d: preprocess_function(d, tokenizer, config.src_col, config.tgt_col, calculated_max_length),
        batched=True,
        batch_size=config.per_device_train_batch_size,
        num_proc=max(1, config.preprocessing_num_workers if hasattr(config, 'preprocessing_num_workers') else 1),
        desc="Running tokenizer on dataset",
    )
    print('====== Data Preprocessing Finished ======')

    # 훈련 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,  # 출력 디렉토리
        learning_rate=config.learning_rate,  # 학습률
        per_device_train_batch_size=config.per_device_train_batch_size,  # 트레이닝 배치 크기
        per_device_eval_batch_size=config.per_device_eval_batch_size,  # 평가 배치 크기
        gradient_accumulation_steps=config.gradient_accumulation_steps if hasattr(config,
                                                                                  'gradient_accumulation_steps') else 1,
        # 그래디언트 누적 단계
        num_train_epochs=config.num_train_epochs,  # 에포크 수
        fp16=config.fp16,  # 16비트 부동소수점 사용 여부
        weight_decay=config.weight_decay,  # 가중치 감쇠
        do_eval=config.do_eval,  # 평가 수행 여부
        evaluation_strategy=config.evaluation_strategy,  # 평가 전략
        warmup_ratio=config.warmup_ratio,  # 워밍업 비율
        log_level=config.log_level,  # 로깅 레벨
        logging_dir=config.logging_dir,  # 로깅 디렉토리
        logging_strategy=config.logging_strategy,  # 로깅 전략
        logging_steps=config.logging_steps,  # 로깅 단계
        eval_steps=config.eval_steps,  # 평가 단계
        save_strategy=config.save_strategy,  # 저장 전략
        save_steps=config.save_steps,  # 저장 단계
        save_total_limit=config.save_total_limit,  # 최대 저장 체크포인트 수
        load_best_model_at_end=config.load_best_model_at_end,  # 학습 종료 시 최상의 모델 로드
        metric_for_best_model=config.metric_for_best_model,  # 최상의 모델 선택 기준
        greater_is_better=config.greater_is_better,  # 메트릭 최대화 여부
        dataloader_num_workers=config.dataloader_num_workers,  # 데이터로더 워커 수
        group_by_length=config.group_by_length,  # 길이별 그룹화 여부
        report_to=config.report_to,  # 보고 대상(tensorboard 등)
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,  # 분산 학습 미사용 파라미터 찾기
        predict_with_generate=config.predict_with_generate,  # 생성 기반 예측 사용
        label_smoothing_factor=config.label_smoothing_factor,  # 레이블 스무딩 계수
        resume_from_checkpoint=(
            True if hasattr(config, 'resume_checkpoint_path') and config.resume_checkpoint_path else None
        ),  # 체크포인트에서 재개 여부
        generation_num_beams=config.generation_num_beams if hasattr(config, 'generation_num_beams') else 4,  # 생성 시 빔 수
    )

    # 커스텀 트레이너 초기화
    # Hugging Face의 Seq2SeqTrainer를 확장한 커스텀 트레이너 객체 생성
    trainer = CustomSeq2SeqTrainer(
        model=model,  # 학습할 모델
        args=training_args,  # 학습 관련 설정값들
        train_dataset=dataset_tokenized['train'],  # 토큰화된 훈련 데이터셋
        eval_dataset=dataset_tokenized['validation'],  # 토큰화된 검증 데이터셋
        tokenizer=tokenizer,  # 텍스트 토큰화를 위한 토크나이저
        data_collator=data_collator,  # 배치 데이터 생성을 위한 콜레이터
        compute_metrics=lambda eval_pred: compute_metrics_extended(
            eval_pred, tokenizer, n_gram=config.n_gram if hasattr(config, 'n_gram') else 2
        ),  # 평가 메트릭 계산 함수 - F0.5, BLEU, GLEU 등 계산
        calculated_max_length=calculated_max_length,  # 앞서 계산된 최대 시퀀스 길이
        early_stopping_patience=config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else 3,
        # 몇 번의 평가 후에도 성능이 개선되지 않으면 중단
        early_stopping_threshold=config.early_stopping_threshold if hasattr(config,
                                                                            'early_stopping_threshold') else 0.001,
        # 성능 개선으로 간주할 최소 임계값
    )

    # 학습 실행 - 체크포인트에서 이어서 할지 새로 시작할지 결정
    if hasattr(config, 'resume_checkpoint_path') and config.resume_checkpoint_path:
        # 기존 체크포인트에서 학습 재개
        print(f"Resuming training from checkpoint: {config.resume_checkpoint_path}")
        train_result = trainer.train(resume_from_checkpoint=config.resume_checkpoint_path)
    else:
        # 처음부터 새로 학습 시작
        print("Starting training from scratch")
        train_result = trainer.train()

    # 학습이 완료된 모델 저장 (모델 가중치와 설정 저장)
    trainer.save_model()
    print(f"Model saved to {config.output_dir}")

    # 학습 과정에서의 지표 저장 (손실, 학습률 등)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)  # 로그에 기록
    trainer.save_metrics("train", metrics)  # 파일로 저장
    trainer.save_state()  # 옵티마이저 상태 등 추가 정보 저장

    # 최종 평가 수행 - 전체 검증 데이터셋에 대한 최종 성능 측정
    print("Performing final evaluation...")
    metrics = trainer.evaluate(
        max_length=calculated_max_length,  # 생성할 최대 시퀀스 길이
        num_beams=config.generation_num_beams if hasattr(config, 'generation_num_beams') else 4,  # 빔 서치에 사용할 빔 수
        metric_key_prefix="final_eval"  # 메트릭 이름 접두사
    )
    trainer.log_metrics("final_eval", metrics)  # 평가 결과 로깅
    trainer.save_metrics("final_eval", metrics)  # 평가 결과 저장
    print(f"Final evaluation metrics: {metrics}")

    # 학습된 트레이너 객체 반환
    return trainer


if __name__ == '__main__':
    """
    스크립트의 메인 실행 부분

    명령줄 인자를 파싱하고 설정 파일을 로드하여 학습 프로세스를 시작합니다.
    """
    # 커맨드라인 인자 파싱 - 설정 파일 경로를 인자로 받음
    parser = argparse.ArgumentParser(description="향상된 Seq2Seq 모델 학습 스크립트")
    parser.add_argument('--config-file', type=str, required=True, help="설정 파일 경로")
    args = parser.parse_args(sys.argv[1:])

    # YAML 형식의 설정 파일 로드
    config = OmegaConf.load(args.config_file)

    # 결과 저장을 위한 디렉토리 생성
    save_path = './data/results'  # 기본 결과 저장 경로
    os.makedirs(save_path, exist_ok=True)  # 경로가 없으면 생성
    os.makedirs(config.output_dir, exist_ok=True)  # 모델 출력 디렉토리 생성
    os.makedirs(config.logging_dir, exist_ok=True)  # 로그 디렉토리 생성

    # 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES  # 사용할 GPU 지정
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 토크나이저 병렬 처리 활성화

    # 학습 시작 정보 로깅
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Start ==========')
    print(f'DEVICE : {config.CUDA_VISIBLE_DEVICES}')  # 사용 중인 GPU 장치
    print(f'MODEL NAME : {config.pretrained_model_name}')  # 사용할 사전 훈련 모델
    print(
        f'USE TRANSFORMER : {config.use_transformer if hasattr(config, "use_transformer") else False}')  # 트랜스포머 모델 사용 여부
    print(
        f'USE ADVANCED AUGMENTATION : {config.use_advanced_augmentation if hasattr(config, "use_advanced_augmentation") else False}')  # 고급 데이터 증강 사용 여부
    print(
        f'USE BACK TRANSLATION : {config.use_back_translation if hasattr(config, "use_back_translation") else False}')  # 역번역 증강 사용 여부

    # 훈련 및 검증 파일 정보 출력
    print('TRAIN FILE PATH :')
    for _path in config.train_data_path_list:
        print(f' - {_path}')
    print('VALIDATION FILE PATH :')
    for _path in config.validation_data_path_list:
        print(f' - {_path}')
    print(f'SAVE PATH : {config.output_dir}')  # 모델 저장 경로

    # 학습 함수 호출하여 훈련 시작
    trainer = train(config)

    # 학습 종료 로깅
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Finished ==========')
