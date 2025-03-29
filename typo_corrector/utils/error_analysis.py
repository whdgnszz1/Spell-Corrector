import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm


def categorize_error(original, predicted):
    """
    오류 유형을 분류하는 함수

    Args:
        original (str): 원본 문장
        predicted (str): 예측 문장

    Returns:
        dict: 오류 유형별 발생 횟수 딕셔너리
    """
    error_types = {
        'perfect_match': 0,
        'spacing_error': 0,
        'typo': 0,
        'word_order': 0,
        'missing_word': 0,
        'extra_word': 0,
        'wrong_word': 0,
        'grammar_error': 0,
        'other': 0
    }

    # 완벽 일치 확인
    if original.strip() == predicted.strip():
        error_types['perfect_match'] = 1
        return error_types

    # 공백만 다른 경우 (띄어쓰기 오류)
    if ''.join(original.split()) == ''.join(predicted.split()):
        error_types['spacing_error'] = 1
        return error_types

    # 단어 단위 분석
    orig_words = original.split()
    pred_words = predicted.split()

    # 단어 순서 오류 (같은 단어들이지만 순서가 다른 경우)
    if sorted(orig_words) == sorted(pred_words) and orig_words != pred_words:
        error_types['word_order'] = 1
        return error_types

    # 단어별 상세 분석
    orig_set = set(orig_words)
    pred_set = set(pred_words)

    # 빠진 단어 있는지 확인
    if len(orig_set - pred_set) > 0:
        error_types['missing_word'] = 1

    # 추가된 단어 있는지 확인
    if len(pred_set - orig_set) > 0:
        error_types['extra_word'] = 1

    # 타이포 확인 (레벤슈타인 거리 사용)
    for o_word in orig_words:
        for p_word in pred_words:
            if o_word != p_word and levenshtein_distance(o_word, p_word) <= 2:
                error_types['typo'] = 1
                break

    # 문법 오류 패턴 체크 (한국어 예시)
    grammar_patterns = [
        r'은/는', r'이/가', r'을/를', r'와/과', r'에/에서',  # 조사 오류
        r'았/었', r'겠/었겠', r'됬/됐',  # 시제 오류
        r'시/으시'  # 존칭 오류
    ]

    for pattern in grammar_patterns:
        if re.search(pattern, original) or re.search(pattern, predicted):
            error_types['grammar_error'] = 1
            break

    # 다른 단어 사용 확인
    if not error_types['typo'] and (len(orig_set - pred_set) > 0 or len(pred_set - orig_set) > 0):
        error_types['wrong_word'] = 1

    # 어떤 카테고리에도 해당하지 않으면 기타로 분류
    if sum(error_types.values()) == 0:
        error_types['other'] = 1

    return error_types


def levenshtein_distance(s1, s2):
    """
    두 문자열 간의 레벤슈타인 거리 계산

    Args:
        s1 (str): 첫 번째 문자열
        s2 (str): 두 번째 문자열

    Returns:
        int: 레벤슈타인 거리
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if not s2:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def analyze_error_types(trainer, eval_dataset, tokenizer, max_length, output_dir='error_analysis'):
    """
    에러 유형 분석을 수행하고 결과를 저장하는 함수

    Args:
        trainer: 훈련된 모델 트레이너
        eval_dataset: 평가 데이터셋
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        output_dir (str): 결과 저장 경로

    Returns:
        None: 파일로 결과를 저장합니다
    """
    os.makedirs(output_dir, exist_ok=True)

    # 모델을 사용하여 예측
    predictions = trainer.predict(
        eval_dataset,
        max_length=max_length,
        num_beams=4
    )

    # 원본과 예측 문장 디코딩
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in
              predictions.label_ids]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 에러 유형 분석
    error_analysis = []
    error_counts = defaultdict(int)

    for i, (original, predicted) in enumerate(tqdm(zip(decoded_labels, decoded_preds), desc="Analyzing errors")):
        error_types = categorize_error(original, predicted)

        # 에러 유형 카운트 업데이트
        for error_type, count in error_types.items():
            error_counts[error_type] += count

        # 개별 샘플 정보 저장
        sample_info = {
            'id': i,
            'original': original,
            'predicted': predicted,
            **error_types
        }
        error_analysis.append(sample_info)

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(error_analysis)

    # 결과 저장
    df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False, encoding='utf-8-sig')

    # 에러 유형 별 통계 계산
    total_samples = len(decoded_labels)
    error_stats = {error: count / total_samples * 100 for error, count in error_counts.items()}

    # 에러 통계 시각화
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(error_stats.keys()), y=list(error_stats.values()))
    plt.title('Error Type Distribution (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))

    # 에러 통계 저장
    with open(os.path.join(output_dir, 'error_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total samples: {total_samples}\n\n")
        f.write("Error Type Distribution:\n")
        for error, percentage in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{error}: {percentage:.2f}%\n")

    print(f"Error analysis completed. Results saved to {output_dir}")
    return error_counts