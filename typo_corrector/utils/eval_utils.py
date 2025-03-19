from Levenshtein import distance as levenshtein_distance
import hangul_jamo


def get_ngram(text, n):
    """
    주어진 텍스트에서 n-gram을 생성

    Args:
        text (str): n-gram을 생성할 입력 텍스트
        n (int): n-gram의 크기 (예: 2라면 2-gram, 3이라면 3-gram 등)

    Returns:
        list: 텍스트에서 추출된 n-gram의 리스트. 각 n-gram은 텍스트의 연속된 n개의 문자로 구성됨

    Example:
        >>> get_ngram("안녕하세요", 2)
        ['안녕', '녕하', '하세', '세요']
    """
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def calc_f_05(cor_sentence, final_prd_sentence, ngram):
    """
    정답 문장과 예측 문장 간의 F0.5 점수를 계산합니다. F0.5는 정밀도를 재현율보다 더 중요시하는 지표

    Args:
        cor_sentence (str): 정답 문장 (참고 문장)
        final_prd_sentence (str): 모델이 예측한 문장
        ngram (int): 비교에 사용할 n-gram의 크기

    Returns:
        tuple: (precision, recall, f_05)
            - precision (float): 정밀도 (예측이 얼마나 정확한지)
            - recall (float): 재현율 (정답을 얼마나 잘 맞췄는지)
            - f_05 (float): F0.5 점수

    Notes:
        - TP (True Positive): 정답과 예측이 일치하는 n-gram 수
        - FP (False Positive): 예측에만 있는 n-gram 수
        - FN (False Negative): 정답에만 있는 n-gram 수
    """
    cor_ngrams = set(get_ngram(cor_sentence, ngram))  # 정답 문장의 n-gram을 집합으로 변환
    prd_ngrams = set(get_ngram(final_prd_sentence, ngram))  # 예측 문장의 n-gram을 집합으로 변환

    TP = len(cor_ngrams.intersection(prd_ngrams))  # 공통 n-gram 수 (True Positive)
    FP = len(prd_ngrams - cor_ngrams)  # 예측에만 있는 n-gram 수 (False Positive)
    FN = len(cor_ngrams - prd_ngrams)  # 정답에만 있는 n-gram 수 (False Negative)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 정밀도 계산, 분모가 0이면 0 반환
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # 재현율 계산, 분모가 0이면 0 반환
    if precision + recall == 0:
        f_05 = 0  # 정밀도와 재현율이 모두 0이면 F0.5도 0
    else:
        # F0.5 점수 계산: β=0.5로 정밀도를 더 중요시
        f_05 = (1 + 0.5 ** 2) * (precision * recall) / (0.5 ** 2 * precision + recall)

    return precision, recall, f_05


def is_hangul(text):
    """
    주어진 텍스트가 한글로만 구성되어 있는지 확인

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

    Args:
        predictions (list): 모델의 예측 문장 리스트
        candidates (list): 후보 문장 리스트
        n_gram (int): n-gram 크기 (기본값: 2)
        avg_candidate_length (float): 후보 문장의 평균 길이 (기본값: None)

    Returns:
        str: 최적의 예측 문장
    """
    best_score = -float('inf')
    best_pred = None

    if avg_candidate_length is None:
        avg_candidate_length = sum(len(c) for c in candidates) / len(candidates)

    for pred in predictions:
        pred_ngrams = set(get_ngram(pred, n_gram))
        candidate_ngrams = set()
        for cand in candidates:
            candidate_ngrams.update(get_ngram(cand, n_gram))

        precision = len(pred_ngrams.intersection(candidate_ngrams)) / len(pred_ngrams) if pred_ngrams else 0
        recall_proxy = len(pred_ngrams.intersection(candidate_ngrams)) / len(
            candidate_ngrams) if candidate_ngrams else 0

        length_diff = abs(len(pred) - avg_candidate_length)
        length_score = 1 / (1 + length_diff)

        score = 0.6 * precision + 0.2 * recall_proxy + 0.2 * length_score

        if score > best_score:
            best_score = score
            best_pred = pred

    return best_pred


def find_closest_candidate(err_sentence, raw_preds, candidates, top_n=3):
    """
    오류 문장과 모델의 예측을 바탕으로 후보 문장 중 가장 적합한 후보를 선택

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

            Args:
                str1 (str): 첫 번째 문자열
                str2 (str): 두 번째 문자열
                n (int): n-gram 크기 (기본값: 2)

            Returns:
                float: n-gram 기반 유사도 (0~1 사이 값)
            """
            ngrams1 = set(get_ngram(str1, n))  # 첫 번째 문자열의 n-gram 집합
            ngrams2 = set(get_ngram(str2, n))  # 두 번째 문자열의 n-gram 집합
            if not ngrams1 or not ngrams2:
                return 0
            return len(ngrams1.intersection(ngrams2)) / max(len(ngrams1), len(ngrams2))  # 교집합 크기 / 최대 n-gram 수

        # 모델 예측과 후보 간의 유사도 계산
        similarity_scores = [ngram_similarity(raw_pred, candidate, n=2) for raw_pred in raw_preds]
        avg_similarity_score = sum(similarity_scores) / len(similarity_scores)  # 평균 유사도
        max_similarity_score = max(similarity_scores)  # 최대 유사도

        # 수정된 점수 계산식: 낮은 점수가 더 좋은 후보를 의미
        if avg_similarity_score > 0:
            score = (length_diff * 0.2) + (edit_distance * 0.2) + (avg_raw_pred_edit_distance * 0.3) - \
                    (avg_similarity_score * 2.0) + max(0, cand_length - err_length) * 0.2

            # 한글과 영어에 따라 유사성 계산
            if is_hangul(err_sentence):
                err_jamo = hangul_jamo.decompose(err_sentence)  # 오류 문장을 한글 자모로 분해
                cand_jamo = hangul_jamo.decompose(candidate)  # 후보 문장을 한글 자모로 분해
                jamo_similarity = len(set(err_jamo) & set(cand_jamo)) / max(len(set(err_jamo)), len(set(cand_jamo)), 1)
            else:
                err_lower = err_sentence.lower()  # 오류 문장을 소문자로 변환 (영어 처리)
                cand_lower = candidate.lower()  # 후보 문장을 소문자로 변환
                jamo_similarity = len(set(err_lower) & set(cand_lower)) / max(len(set(err_lower)), len(set(cand_lower)),
                                                                              1)

            # 후보와 점수 정보를 튜플로 저장
            candidates_with_score.append(
                (candidate, length_diff, edit_distance, avg_similarity_score, max_similarity_score, score,
                 avg_raw_pred_edit_distance, jamo_similarity)
            )

    # total_score를 기준으로 오름차순 정렬 (낮은 점수가 더 좋은 후보)
    candidates_with_score.sort(key=lambda x: x[5])  # x[5]는 total_score
    top_candidates = candidates_with_score[:top_n]  # 상위 n개 후보 선택

    # 최종 후보 선택
    if top_candidates:
        min_score = top_candidates[0][5]  # 최소 점수
        tied_candidates = [cand for cand in top_candidates if cand[5] == min_score]  # 동일 점수의 후보 필터링
        if len(tied_candidates) > 1:
            # 점수가 동일한 경우, jamo_similarity가 높고 edit_distance가 낮은 후보 선택
            final_candidate = max(tied_candidates, key=lambda x: (x[7], -x[2]))[
                0]  # x[7]: jamo_similarity, x[2]: edit_distance
        else:
            final_candidate = tied_candidates[0][0]  # 유일한 최소 점수 후보
    else:
        final_candidate = candidates[0] if candidates else ""  # 후보가 없으면 첫 번째 후보 또는 빈 문자열 반환

    return final_candidate, top_candidates
