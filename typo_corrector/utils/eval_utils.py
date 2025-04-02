"""
텍스트 교정 모델의 평가를 위한 유틸리티 모듈

이 모듈은 모델이 생성한 예측 문장을 평가하기 위한 다양한 메트릭과 유틸리티 함수를 제공합니다.
주요 기능으로는 BLEU, GLEU, F0.5 점수 계산, 최적의 예측 선택, n-gram 생성 등이 있습니다.
텍스트 교정 시스템의 성능을 다양한 관점에서 측정하고 분석할 수 있게 해줍니다.
"""

from nltk.translate.bleu_score import sentence_bleu  # BLEU 점수 계산용 NLTK 함수
from nltk.translate.bleu_score import SmoothingFunction  # BLEU 스무딩 함수 (짧은 문장 보정)
from nltk.translate.gleu_score import sentence_gleu  # GLEU 점수 계산용 NLTK 함수 (Google BLEU)
from Levenshtein import distance as levenshtein_distance  # 편집 거리 계산 라이브러리
import hgtk  # 한글 자모 분해 라이브러리
from collections import Counter


def is_hangul(text):
    """
    주어진 텍스트가 한글로만 구성되어 있는지 확인

    각 문자가 한글 유니코드 범위(AC00-D7A3)에 포함되는지 검사합니다.

    Args:
        text (str): 검사할 텍스트

    Returns:
        bool: 텍스트의 모든 문자가 한글 유니코드 범위(\uAC00 ~ \uD7A3)에 속하면 True, 아니면 False

    Example:
        >>> is_hangul("안녕")
        True
        >>> is_hangul("Hello")
        False
    """
    return all('\uAC00' <= char <= '\uD7A3' for char in text)  # 한글 유니코드 범위 검사


# 최적의 예측을 선택하는 함수
def select_best_prediction(predictions, candidates, n_gram=2, avg_candidate_length=None):
    """
    여러 예측 중에서 최적의 예측을 선택

    여러 모델 예측 중에서 후보 문장들과 가장 유사한 예측을 선택합니다.
    유사도는 n-gram 기반 정밀도, 재현율 대용치, 길이 유사성을 고려합니다.

    Args:
        predictions (list): 모델의 예측 문장 리스트
        candidates (list): 후보 문장 리스트 (정답 또는 참조 문장들)
        n_gram (int): n-gram 크기 (기본값: 2)
        avg_candidate_length (float): 후보 문장의 평균 길이 (기본값: None, None이면 계산)

    Returns:
        str: 최적의 예측 문장
    """
    best_score = -float('inf')  # 초기 최고 점수는 음의 무한대
    best_pred = None  # 최적의 예측 (아직 없음)

    # 후보 문장 평균 길이 계산 (제공되지 않은 경우)
    if avg_candidate_length is None:
        avg_candidate_length = sum(len(c) for c in candidates) / len(candidates)

    # 각 예측에 대해 점수 계산
    for pred in predictions:
        # 예측 문장의 n-gram 집합
        pred_ngrams = set(get_ngram(pred, n_gram))

        # 모든 후보 문장의 n-gram을 하나의 집합으로 통합
        candidate_ngrams = set()
        for cand in candidates:
            candidate_ngrams.update(get_ngram(cand, n_gram))

        # 정밀도 계산: (예측과 후보의 공통 n-gram) / (예측 문장의 총 n-gram)
        precision = len(pred_ngrams.intersection(candidate_ngrams)) / len(pred_ngrams) if pred_ngrams else 0

        # 재현율 대용 지표: (예측과 후보의 공통 n-gram) / (모든 후보의 총 n-gram)
        recall_proxy = len(pred_ngrams.intersection(candidate_ngrams)) / len(
            candidate_ngrams) if candidate_ngrams else 0

        # 길이 차이에 따른 페널티 점수
        length_diff = abs(len(pred) - avg_candidate_length)
        length_score = 1 / (1 + length_diff)  # 길이 차이가 적을수록 높은 점수

        # 종합 점수 계산 (가중치 적용)
        score = 0.6 * precision + 0.2 * recall_proxy + 0.2 * length_score

        # 현재까지의 최고 점수보다 높으면 갱신
        if score > best_score:
            best_score = score
            best_pred = pred

    return best_pred  # 최적의 예측 반환


def find_closest_candidate(err_sentence, raw_preds, candidates, top_n=3):
    """
    오류 문장과 모델의 예측을 바탕으로 후보 문장 중 가장 적합한 후보를 선택

    입력된 오류 문장과 모델 예측, 그리고 여러 후보 문장들을 비교하여
    가장 적합한 후보를 선택합니다. 다양한 유사도 메트릭을 종합적으로 활용합니다.

    Args:
        err_sentence (str): 오류가 포함된 입력 문장
        raw_preds (list): 모델이 생성한 예측 문장 리스트
        candidates (list): 비교할 후보 문장 리스트
        top_n (int): 반환할 상위 후보의 개수 (기본값: 3)

    Returns:
        tuple: (final_candidate, top_candidates)
            - final_candidate (str): 최종 선택된 후보 문장
            - top_candidates (list): 상위 top_n개의 후보와 점수 정보 (각 요소는 튜플)

    Notes:
        - 점수 계산은 길이 차이, 편집 거리, 모델 예측과의 유사도 등을 종합적으로 고려
        - 한글은 자모 분해, 영어는 소문자 변환을 통해 유사성을 계산
    """
    err_length = len(err_sentence)  # 오류 문장의 길이
    candidates_with_score = []  # 후보와 점수 정보를 저장할 리스트

    # 각 후보 문장에 대해 점수 계산
    for candidate in candidates:
        cand_length = len(candidate)  # 후보 문장의 길이
        length_diff = abs(err_length - cand_length)  # 길이 차이
        edit_distance = levenshtein_distance(err_sentence, candidate)  # 오류 문장과 후보 간의 편집 거리

        # 모델 예측과 후보 간의 편집 거리 계산
        raw_pred_edit_distances = [levenshtein_distance(raw_pred, candidate) for raw_pred in raw_preds]
        avg_raw_pred_edit_distance = sum(raw_pred_edit_distances) / len(raw_preds)  # 평균 편집 거리

        # n-gram 유사도 계산 함수 (내부 정의)
        def ngram_similarity(str1, str2, n=2):
            """
            두 문자열 간의 n-gram 유사도를 계산합니다.

            자카드 유사도 방식을 사용해 두 문자열의 n-gram 집합 간 유사도를 계산합니다.

            Args:
                str1 (str): 첫 번째 문자열
                str2 (str): 두 번째 문자열
                n (int): n-gram 크기 (기본값: 2)

            Returns:
                float: n-gram 기반 유사도 (0~1 사이 값)
            """
            ngrams1 = set(get_ngram(str1, n))  # 첫 번째 문자열의 n-gram 집합
            ngrams2 = set(get_ngram(str2, n))  # 두 번째 문자열의 n-gram 집합

            # 빈 집합 처리
            if not ngrams1 or not ngrams2:
                return 0

            # 자카드 유사도: 교집합 크기 / 최대 집합 크기
            return len(ngrams1.intersection(ngrams2)) / max(len(ngrams1), len(ngrams2))

        # 모델 예측과 후보 간의 유사도 계산
        similarity_scores = [ngram_similarity(raw_pred, candidate, n=2) for raw_pred in raw_preds]
        avg_similarity_score = sum(similarity_scores) / len(similarity_scores)  # 평균 유사도
        max_similarity_score = max(similarity_scores)  # 최대 유사도

        # 의미 있는 유사도가 있는 경우에만 점수 계산
        if avg_similarity_score > 0:
            # 수정된 점수 계산식: 낮은 점수가 더 좋은 후보를 의미
            score = (length_diff * 0.2) + (edit_distance * 0.2) + (avg_raw_pred_edit_distance * 0.3) - \
                    (avg_similarity_score * 2.0) + max(0, cand_length - err_length) * 0.2

            # 한글과 영어에 따라 유사성 계산 방식 달리 적용
            if is_hangul(err_sentence):
                # 한글인 경우 자모 분해 후 공통 자모 비율 계산
                err_jamo = ''.join([hgtk.letter.decompose(char) for char in err_sentence if hgtk.checker.is_hangul(char)])
                cand_jamo = ''.join([hgtk.letter.decompose(char) for char in candidate if hgtk.checker.is_hangul(char)])
                # 자모 유사도 = 공통 자모 수 / 최대 자모 집합 크기
                jamo_similarity = len(set(err_jamo) & set(cand_jamo)) / max(len(set(err_jamo)), len(set(cand_jamo)), 1)
            else:
                # 영어인 경우 소문자 변환 후 공통 문자 비율 계산
                err_lower = err_sentence.lower()  # 오류 문장을 소문자로 변환
                cand_lower = candidate.lower()  # 후보 문장을 소문자로 변환
                # 문자 유사도 = 공통 문자 수 / 최대 문자 집합 크기
                jamo_similarity = len(set(err_lower) & set(cand_lower)) / max(len(set(err_lower)), len(set(cand_lower)),
                                                                              1)

            # 후보와 모든 점수 정보를 튜플로 저장
            candidates_with_score.append(
                (
                    candidate,  # 후보 문장
                    length_diff,  # 길이 차이
                    edit_distance,  # 편집 거리
                    avg_similarity_score,  # 평균 유사도
                    max_similarity_score,  # 최대 유사도
                    score,  # 종합 점수
                    avg_raw_pred_edit_distance,  # 평균 편집 거리
                    jamo_similarity  # 자모/문자 유사도
                )
            )

    # total_score(index 5)를 기준으로 오름차순 정렬 (낮은 점수가 더 좋은 후보)
    candidates_with_score.sort(key=lambda x: x[5])
    top_candidates = candidates_with_score[:top_n]  # 상위 n개 후보 선택

    # 최종 후보 선택 로직
    if top_candidates:
        min_score = top_candidates[0][5]  # 최소(최고) 점수
        # 동일 점수의 후보들 필터링
        tied_candidates = [cand for cand in top_candidates if cand[5] == min_score]

        if len(tied_candidates) > 1:
            # 동점인 경우: 자모 유사도 높고, 편집 거리 낮은 후보 선택
            # x[7]: jamo_similarity(높을수록 좋음), x[2]: edit_distance(낮을수록 좋음)
            final_candidate = max(tied_candidates, key=lambda x: (x[7], -x[2]))[0]
        else:
            final_candidate = tied_candidates[0][0]  # 유일한 최고 점수 후보
    else:
        # 후보가 없는 경우 기본값 처리
        final_candidate = candidates[0] if candidates else ""

    return final_candidate, top_candidates


def get_ngram(text, n):
    """
    주어진 텍스트에서 n-gram을 생성

    텍스트의 연속된 n개 문자로 구성된 모든 부분 문자열 목록을 반환합니다.
    텍스트 유사도 계산 등에 활용됩니다.

    Args:
        text (str): n-gram을 생성할 입력 텍스트
        n (int): n-gram의 크기 (예: 2라면 2-gram, 3이라면 3-gram 등)

    Returns:
        list: 텍스트에서 추출된 n-gram의 리스트. 각 n-gram은 텍스트의 연속된 n개의 문자로 구성됨

    Example:
        >>> get_ngram("안녕하세요", 2)
        ['안녕', '녕하', '하세', '세요']
    """
    # 슬라이딩 윈도우 방식으로 모든 가능한 n-gram 생성
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def calc_bleu(reference, hypothesis, n_gram=4):
    """
    BLEU 점수를 계산하는 함수

    BLEU(Bilingual Evaluation Understudy)는 기계 번역 평가에서 널리 사용되는 지표이지만,
    텍스트 교정 작업에도 활용됩니다. n-gram 일치도를 기반으로 생성 품질을 평가합니다.

    Args:
        reference (str): 참조 문장 (정답)
        hypothesis (str): 가설 문장 (예측)
        n_gram (int): 최대 n-gram 크기 (기본값: 4)

    Returns:
        float: BLEU 점수 (0~1)
    """

    # 문장을 단어 단위로 토큰화
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # 빈 문장 처리
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0

    # BLEU 점수 계산 설정
    # 각 n-gram 단위에 대해 동일한 가중치 적용 (1/n_gram)
    weights = [1.0 / n_gram] * min(n_gram, 4)  # 최대 4-gram까지만 사용

    # 짧은 문장에 대한 페널티 완화를 위한 스무딩 함수 적용
    smoothing = SmoothingFunction().method1

    try:
        # NLTK의 sentence_bleu 함수 호출
        return sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)
    except Exception:
        # 예외 발생 시 0 반환
        return 0.0


def calc_gleu(reference, hypothesis, n_gram=4):
    """
    GLEU(Google-BLEU) 점수를 계산하는 함수

    GLEU는 Google에서 개발한 BLEU의 변형으로, 문법 오류 교정 작업에 더 적합합니다.
    정밀도와 재현율 요소를 모두 고려하여 BLEU보다 특히 짧은 교정에 더 민감합니다.

    Args:
        reference (str): 참조 문장 (정답)
        hypothesis (str): 가설 문장 (예측)
        n_gram (int): 최대 n-gram 크기 (기본값: 4)

    Returns:
        float: GLEU 점수 (0~1)
    """

    # 문장을 단어 단위로 토큰화
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # 빈 문장 처리
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0

    try:
        # NLTK의 sentence_gleu 함수 호출 (max_len으로 최대 n-gram 크기 제한)
        return sentence_gleu([ref_tokens], hyp_tokens, max_len=n_gram)
    except Exception:
        # 예외 발생 시 0 반환
        return 0.0


def calc_precision_recall_f05(reference, candidate, ngram=2):
    """
    n-gram 기반의 precision, recall, F0.5 점수를 계산하는 함수

    Args:
        reference (str): 참조 문장 (올바른 정답)
        candidate (str): 후보 문장 (모델 예측)
        ngram (int): n-gram 크기 (기본값: 2)

    Returns:
        tuple: (precision, recall, f0.5 점수)
    """
    # 문자열 정확히 일치하면 모든 점수 1
    if reference == candidate:
        return 1.0, 1.0, 1.0

    # n-gram 생성 함수
    def get_ngrams(text, n):
        # 문자열 가장자리에 경계 표시 추가
        padded = '##' + text + '##'
        ngrams = []
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i:i + n])
        return ngrams

    # n-gram 집합 생성
    ref_ngrams = Counter(get_ngrams(reference, ngram))
    cand_ngrams = Counter(get_ngrams(candidate, ngram))

    # 공통 n-gram 개수 계산
    common_ngrams = sum((ref_ngrams & cand_ngrams).values())

    # precision, recall 계산
    precision = common_ngrams / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0
    recall = common_ngrams / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0

    # F0.5 점수 계산 (precision에 더 높은 가중치)
    beta = 0.5
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)

    return precision, recall, f_score