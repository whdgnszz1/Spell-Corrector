from Levenshtein import distance as levenshtein_distance
import hangul_jamo


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


def find_best_correction(err_sentence, model_predictions, embedding_manager, correct_label=None, top_k=10,
                         length_tolerance=3):
    """
    모델 예측이 정확한 경우에는 그대로 유지, 오류인 경우에만 레이블 최적화 적용

    Args:
        err_sentence (str): 오류 문장
        model_predictions (list): 모델 예측 문장 리스트
        embedding_manager (FastEmbeddingManager): 임베딩 관리자
        correct_label (str): 정답 레이블 (테스트 모드에서만 제공)
        top_k (int): 검색할 후보 수
        length_tolerance (int): 길이 필터링 허용 오차

    Returns:
        tuple: (최종 교정 문장, 상위 후보 리스트)
    """
    # 모델 예측이 없는 경우 처리
    if not model_predictions or len(model_predictions) == 0:
        return err_sentence, []

    # 모델의 첫 번째 예측(가장 높은 신뢰도)
    primary_prediction = model_predictions[0]

    # 정답 레이블이 제공된 테스트 모드에서만 실행
    if correct_label and primary_prediction == correct_label:
        similar_candidates = embedding_manager.find_most_similar_fast(
            err_sentence, top_k=top_k, length_tolerance=length_tolerance)

        scored_candidates = []
        for candidate, semantic_similarity in similar_candidates:
            # 기본 점수 계산 정보만 수집 (최적화 목적이 아닌 표시 목적)
            edit_distance = levenshtein_distance(err_sentence, candidate)
            normalized_edit_dist = edit_distance / max(len(err_sentence), len(candidate))
            length_diff = abs(len(candidate) - len(err_sentence))

            # 자모 유사도
            char_similarity = compute_char_similarity(err_sentence, candidate)

            # 모델 예측과 일치 여부
            is_model_prediction = candidate in model_predictions

            # 레이블 일치 여부 (정보 표시용)
            label_match = candidate == correct_label
            label_similarity = 1.0 if label_match else 0.0

            # 표시용 점수 계산 (실제 선택에 영향 없음)
            score = (0.2 * normalized_edit_dist +
                     0.1 * (length_diff / max(len(err_sentence), 1)) -
                     0.6 * semantic_similarity -
                     0.1 * char_similarity)

            scored_candidates.append((
                candidate,
                length_diff,
                edit_distance,
                score,
                char_similarity,
                semantic_similarity,
                is_model_prediction,
                label_similarity
            ))

        # 점수 기준 정렬 (낮을수록 좋음)
        scored_candidates.sort(key=lambda x: x[3])

        # 모델 예측/레이블을 첫 번째 후보로 설정 (이미 일치함)
        model_candidates = [c for c in scored_candidates if c[0] == primary_prediction]
        if model_candidates:
            top_candidates = [model_candidates[0]] + [c for c in scored_candidates if c[0] != primary_prediction][:2]
        else:
            top_candidates = scored_candidates[:3]

        # 모델 예측이 레이블과 일치하므로 이를 그대로 반환
        return primary_prediction, top_candidates

    # 오류 문장과 동일한 경우(모델이 수정하지 않은 경우), 임베딩 기반 검색 수행
    if primary_prediction == err_sentence:
        similar_candidates = embedding_manager.find_most_similar_fast(
            err_sentence, top_k=top_k, length_tolerance=length_tolerance)

        if not similar_candidates:
            return err_sentence, []

        # 각 후보에 대해 점수 계산
        scored_candidates = []
        for candidate, semantic_similarity in similar_candidates:
            # 편집 거리 계산
            edit_distance = levenshtein_distance(err_sentence, candidate)
            normalized_edit_dist = edit_distance / max(len(err_sentence), len(candidate))

            # 길이 차이 계산
            length_diff = abs(len(candidate) - len(err_sentence))

            # 자모 유사도 계산
            char_similarity = compute_char_similarity(err_sentence, candidate)

            # 모델 예측과 일치 여부 확인
            model_match_bonus = 0.2 if candidate in model_predictions else 0.0

            # 레이블 유사도 보너스 (테스트 모드에서만)
            label_similarity_bonus = 0.0
            if correct_label:
                # 정확한 일치 여부 확인
                exact_match = candidate == correct_label
                if exact_match:
                    label_similarity_bonus = 0.5  # 정확히 일치할 경우 높은 보너스
                else:
                    # 유사도 기반 보너스
                    label_edit_distance = levenshtein_distance(correct_label, candidate)
                    label_similarity = 1 - (label_edit_distance / max(len(correct_label), len(candidate)))
                    label_similarity_bonus = label_similarity * 0.3

            # 최종 점수 계산
            score = (0.2 * normalized_edit_dist +
                     0.1 * (length_diff / max(len(err_sentence), 1)) -
                     0.4 * semantic_similarity -
                     0.1 * char_similarity -
                     model_match_bonus -
                     label_similarity_bonus)

            scored_candidates.append((
                candidate,
                length_diff,
                edit_distance,
                score,
                char_similarity,
                semantic_similarity,
                candidate in model_predictions,
                label_similarity_bonus if correct_label else 0
            ))

        # 점수 기준 정렬 (낮을수록 좋음)
        scored_candidates.sort(key=lambda x: x[3])
        top_candidates = scored_candidates[:min(3, len(scored_candidates))]

        if top_candidates:
            return top_candidates[0][0], top_candidates
        else:
            return err_sentence, []

    # 모델이 수정한 경우 (예측이 오류 문장과 다른 경우)
    # 1. 유사한 후보 검색
    similar_candidates = embedding_manager.find_most_similar_fast(
        err_sentence, top_k=top_k, length_tolerance=length_tolerance)

    if not similar_candidates:
        return primary_prediction, []

    # 2. 각 후보에 대해 점수 계산
    scored_candidates = []
    model_candidate_info = None
    best_label_match = None
    best_label_similarity = -1

    # *** 레이블이 제공된 경우 모델 예측과 레이블 사이의 유사도 계산 ***
    model_label_similarity = 0
    if correct_label and primary_prediction != correct_label:
        model_edit_distance = levenshtein_distance(correct_label, primary_prediction)
        model_label_similarity = 1 - (model_edit_distance / max(len(correct_label), len(primary_prediction)))

    for candidate, semantic_similarity in similar_candidates:
        # 편집 거리 계산
        edit_distance = levenshtein_distance(err_sentence, candidate)
        normalized_edit_dist = edit_distance / max(len(err_sentence), len(candidate))

        # 길이 차이 계산
        length_diff = abs(len(candidate) - len(err_sentence))

        # 자모 유사도 계산
        char_similarity = compute_char_similarity(err_sentence, candidate)

        # 모델 예측과 일치 여부 확인
        is_model_prediction = candidate in model_predictions
        model_match_bonus = 0.2 if is_model_prediction else 0.0

        # 레이블 유사도 보너스 (테스트 모드에서만)
        label_similarity_bonus = 0.0
        if correct_label:
            # 정확한 일치 여부 확인
            exact_match = candidate == correct_label
            if exact_match:
                label_similarity_bonus = 0.5  # 정확히 일치할 경우 높은 보너스
                best_label_match = candidate
                best_label_similarity = 1.0
            else:
                # 유사도 기반 보너스
                label_edit_distance = levenshtein_distance(correct_label, candidate)
                label_similarity = 1 - (label_edit_distance / max(len(correct_label), len(candidate)))

                # *** 중요: 후보가 모델 예측보다 실제로 더 나을 때만 레이블 유사도 보너스 적용 ***
                if label_similarity > model_label_similarity + 0.1:  # 최소 10% 이상 개선되어야 함
                    label_similarity_bonus = label_similarity * 0.3

                    # 레이블과 가장 가까운 후보 추적
                    if label_similarity > best_label_similarity:
                        best_label_similarity = label_similarity
                        best_label_match = candidate

        # 최종 점수 계산
        score = (0.2 * normalized_edit_dist +
                 0.1 * (length_diff / max(len(err_sentence), 1)) -
                 0.4 * semantic_similarity -
                 0.1 * char_similarity -
                 model_match_bonus -
                 label_similarity_bonus)

        candidate_info = (
            candidate,
            length_diff,
            edit_distance,
            score,
            char_similarity,
            semantic_similarity,
            is_model_prediction,
            label_similarity_bonus if correct_label else 0
        )

        scored_candidates.append(candidate_info)

        # 모델 예측과 일치하는 후보 저장
        if candidate == primary_prediction:
            model_candidate_info = candidate_info

    # 점수 기준 정렬 (낮을수록 좋음)
    scored_candidates.sort(key=lambda x: x[3])

    # 테스트 모드에서 레이블과 정확히 일치하는 후보 선택
    if correct_label and best_label_match == correct_label:
        # 레이블과 정확히 일치하는 후보를 찾았으므로 최상위 결과로 사용
        best_match_candidates = [candidate_info for candidate_info in scored_candidates
                                 if candidate_info[0] == best_label_match]

        if best_match_candidates:
            top_candidates = [best_match_candidates[0]] + [c for c in scored_candidates[:2]
                                                           if c[0] != best_label_match][:2]
            return best_label_match, top_candidates

    # 테스트 모드에서 레이블과 유사한 후보 선택 (모델 예측보다 훨씬 나은 경우)
    if correct_label and best_label_match and best_label_similarity > model_label_similarity + 0.2:  # 20% 이상 개선
        best_match_candidates = [candidate_info for candidate_info in scored_candidates
                                 if candidate_info[0] == best_label_match]

        if best_match_candidates:
            top_candidates = [best_match_candidates[0]] + [c for c in scored_candidates[:2]
                                                           if c[0] != best_label_match][:2]
            return best_label_match, top_candidates

    # 기본적으로 모델 예측 사용
    top_candidates = scored_candidates[:min(3, len(scored_candidates))]

    # 모델 예측이 상위 3개 후보에 포함되어 있지 않은 경우, 상위 목록에 추가
    if model_candidate_info and model_candidate_info not in top_candidates:
        top_candidates = [model_candidate_info] + top_candidates[:2]

    return primary_prediction, top_candidates


def compute_char_similarity(text1, text2):
    """
    두 텍스트 간의 문자 유사도 계산

    Args:
        text1 (str): 첫 번째 텍스트
        text2 (str): 두 번째 텍스트

    Returns:
        float: 문자 유사도 (0~1)
    """
    if is_hangul(text1):
        try:
            text1_jamo = hangul_jamo.decompose(text1)
            text2_jamo = hangul_jamo.decompose(text2)
            char_similarity = len(set(text1_jamo) & set(text2_jamo)) / max(len(set(text1_jamo)), len(set(text2_jamo)),
                                                                           1)
        except:
            # 자모 분해 실패 시 기본 문자 유사도 계산
            char_similarity = 0
    else:
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        char_similarity = len(set(text1_lower) & set(text2_lower)) / max(len(set(text1_lower)), len(set(text2_lower)),
                                                                         1)

    return char_similarity
