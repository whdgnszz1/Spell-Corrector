from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import sys
import json
from datetime import datetime
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
from utils import calc_f_05, augment_sentence


# 데이터 증강을 별도의 함수로 분리
def augment_data(original_sentences, augment_prob=0.1):
    """
    원본 문장을 증강하여 오류 문장과 정답 문장 쌍을 생성

    Args:
        original_sentences (list): 원본 정답 문장 리스트
        augment_prob (float): 증강 확률 (기본값: 0.1)

    Returns:
        dict: 증강된 오류 문장과 정답 문장 리스트
    """
    augmented_data = {'err_sentence': [], 'cor_sentence': []}
    for cor in original_sentences:
        err = augment_sentence(cor, prob=augment_prob)  # 외부 유틸리티 함수로 문장 증강
        augmented_data['err_sentence'].append(err)
        augmented_data['cor_sentence'].append(cor)
    return augmented_data


# 데이터셋 생성 함수
def make_dataset(train_data_path_list, validation_data_path_list, augment_prob=0.1):
    """
    훈련 및 검증 데이터셋을 로드하고, 훈련 데이터에 대해 증강을 수행하여 데이터셋을 생성

    Args:
        train_data_path_list (list): 훈련 데이터 파일 경로 리스트
        validation_data_path_list (list): 검증 데이터 파일 경로 리스트
        augment_prob (float): 증강 확률 (기본값: 0.1)

    Returns:
        datasets.DatasetDict: 훈련 및 검증 데이터셋
        int: 계산된 최대 길이
    """
    # 데이터 저장용 딕셔너리 초기화
    loaded_data_dict = {
        'train': {'err_sentence': [], 'cor_sentence': []},
        'validation': {'err_sentence': [], 'cor_sentence': []}
    }

    # 훈련 데이터 로드
    for i, train_data_path in enumerate(train_data_path_list):
        try:
            with open(train_data_path, 'r', encoding='utf-8') as f:  # 인코딩 명시
                _temp_json = json.load(f)
            # 오류 문장과 정답 문장을 리스트에 추가
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

    # 원본 정답 문장 저장
    original_cor_sentences = loaded_data_dict['train']['cor_sentence'].copy()

    # 훈련 데이터 증강
    augmented_train = augment_data(original_cor_sentences, augment_prob)
    loaded_data_dict['train']['err_sentence'].extend(augmented_train['err_sentence'])
    loaded_data_dict['train']['cor_sentence'].extend(augmented_train['cor_sentence'])

    # 정확한 문장 추가 (원본 데이터의 10%)
    num_original = len(original_cor_sentences)
    num_to_add = int(num_original * 0.1)  # 10%
    indices = random.sample(range(num_original), num_to_add)  # 무작위 인덱스 선택
    for idx in indices:
        cor = original_cor_sentences[idx]
        loaded_data_dict['train']['err_sentence'].append(cor)  # 오류 문장과 정답 문장 동일하게 추가
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

    # 데이터셋 객체 생성
    dataset_dict = {
        split: datasets.Dataset.from_dict(data, split=split)
        for split, data in loaded_data_dict.items()
    }
    dataset = datasets.DatasetDict(dataset_dict)

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
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)  # 입력 토큰화
    with tokenizer.as_target_tokenizer():  # 타겟 토큰화 컨텍스트
        labels = tokenizer(targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels['input_ids']  # 레이블 추가
    return model_inputs


# 커스텀 트레이너 클래스
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer를 상속하여 예측 단계에서 생성 파라미터를 사용자 정의."""

    def __init__(self, *args, calculated_max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculated_max_length = calculated_max_length  # 최대 길이 저장

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        예측 단계에서 생성 파라미터를 조정하여 문장을 생성

        Args:
            model: 학습 모델
            inputs: 입력 데이터
            prediction_loss_only (bool): 손실만 계산할지 여부
            ignore_keys: 무시할 키 목록

        Returns:
            tuple: (손실, 생성된 토큰, 레이블)
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {  # 생성 파라미터 설정
            "max_length": self.calculated_max_length,
            "num_beams": 4,  # 빔 서치 크기
            "no_repeat_ngram_size": 2,  # 반복 방지
            "length_penalty": 2.0,  # 길이 패널티
            "early_stopping": True,  # 조기 종료
        }
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        labels = inputs["labels"]
        return (None, generated_tokens, labels)


# 학습 함수
def train(config):
    """
    모델을 로드하고 데이터셋을 준비하여 학습을 수행

    Args:
        config (OmegaConf): 설정 객체
    """
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Start ======')
    model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name)  # 사전 학습된 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)  # 토크나이저 로드
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Finished ======')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)  # 데이터 콜레이터 초기화

    print(f'[{_now_time}] ====== Data Load Start ======')
    dataset, calculated_max_length = make_dataset(
        config.train_data_path_list, config.validation_data_path_list, augment_prob=0.1
    )
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Load Finished ======')

    print(f'[{_now_time}] ====== Data Preprocessing Start ======')
    # 데이터셋 토큰화
    dataset_tokenized = dataset.map(
        lambda d: preprocess_function(d, tokenizer, config.src_col, config.tgt_col, calculated_max_length),
        batched=True, batch_size=config.per_device_train_batch_size
    )
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Preprocessing Finished ======')

    def compute_metrics(eval_pred):
        """
        평가 메트릭(F0.5 스코어)을 계산

        Args:
            eval_pred (tuple): 예측값과 레이블

        Returns:
            dict: F0.5 평균 점수
        """
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # 예측 디코딩
        # -100 토큰을 패드 토큰으로 변환
        labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  # 레이블 디코딩
        f_05_list = [calc_f_05(label, pred, n_gram=2)[2] for pred, label in zip(decoded_preds, decoded_labels)]
        return {"f_05": sum(f_05_list) / len(f_05_list)}

    # 훈련 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        weight_decay=config.weight_decay,
        do_eval=config.do_eval,
        evaluation_strategy=config.evaluation_strategy,
        warmup_ratio=config.warmup_ratio,
        log_level=config.log_level,
        logging_dir=config.logging_dir,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        dataloader_num_workers=config.dataloader_num_workers,
        group_by_length=config.group_by_length,
        report_to=config.report_to,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        predict_with_generate=config.predict_with_generate,
        label_smoothing_factor=config.label_smoothing_factor,
        resume_from_checkpoint=(
            True if hasattr(config, 'resume_checkpoint_path') and config.resume_checkpoint_path else None
        ),
    )

    # 커스텀 트레이너 초기화
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized['train'],
        eval_dataset=dataset_tokenized['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        calculated_max_length=calculated_max_length,
    )

    # resume_from_checkpoint가 있을 경우 checkpoint부터 학습 재개
    if hasattr(config, 'resume_checkpoint_path') and config.resume_checkpoint_path:
        trainer.train(resume_from_checkpoint=config.resume_checkpoint_path)
    else:
        trainer.train()


if __name__ == '__main__':
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description="Seq2Seq 모델 학습 스크립트")
    parser.add_argument('--config-file', type=str, required=True, help="설정 파일 경로")
    args = parser.parse_args(sys.argv[1:])

    # 설정 파일 로드
    config = OmegaConf.load(args.config_file)

    # 결과 저장 디렉토리 생성
    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    # 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # 학습 시작 로그 출력
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Start ==========')
    print(f'DEVICE : {config.CUDA_VISIBLE_DEVICES}')
    print(f'MODEL NAME : {config.pretrained_model_name}')
    print('TRAIN FILE PATH :')
    for _path in config.train_data_path_list:
        print(f' - {_path}')
    print('VALIDATION FILE PATH :')
    for _path in config.validation_data_path_list:
        print(f' - {_path}')
    print(f'SAVE PATH : {config.output_dir}')

    # 학습 실행
    train(config)

    # 학습 종료 로그 출력
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Finished ==========')
