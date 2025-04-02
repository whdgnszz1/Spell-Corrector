import random
import hgtk
from utils.hangul import choseong_adjacent, jungseong_adjacent

CHO = hgtk.letter.CHO
JOONG = hgtk.letter.JOONG


def generate_typo(char):
    """한글 문자에 오타를 생성하는 함수"""
    # 한글 범위 체크
    if 0xAC00 <= ord(char) <= 0xD7A3:
        # 한글 분리
        cho, jung, jong = hgtk.letter.decompose(char)

        # 랜덤으로 오타 유형 선택
        typo_type = random.choice(['cho', 'jung', 'jong'])

        if typo_type == 'cho':
            # 초성 변경: CHO 리스트에서 다음 초성으로 이동
            cho_idx = (hgtk.letter.CHO.index(cho) + 1) % len(hgtk.letter.CHO)
            cho = hgtk.letter.CHO[cho_idx]
        elif typo_type == 'jung':
            # 중성 변경: JOONG 리스트에서 다음 중성으로 이동
            jung_idx = (hgtk.letter.JOONG.index(jung) + 1) % len(hgtk.letter.JOONG)
            jung = hgtk.letter.JOONG[jung_idx]
        elif typo_type == 'jong':
            if jong:  # 종성이 있는 경우
                jong_idx = (hgtk.letter.JONG.index(jong) + 1) % len(hgtk.letter.JONG)
                jong = hgtk.letter.JONG[jong_idx]
            else:
                # 종성이 없는 경우, 랜덤 종성 추가 (빈 종성 제외)
                jong = random.choice(hgtk.letter.JONG[1:])

        # 변형된 한글 조합
        return hgtk.letter.compose(cho, jung, jong)
    return char


def generate_word_typo(word, typo_count=1):
    """단어에서 최대 typo_count개의 문자만 오타로 변경"""
    if not word:
        return word

    # 한글 문자의 위치 찾기
    hangul_indices = [i for i, char in enumerate(word) if 0xAC00 <= ord(char) <= 0xD7A3]

    # 한글이 없으면 원본 반환
    if not hangul_indices:
        return word

    # 오타로 변경할 위치 선택 (최대 typo_count개)
    typo_positions = random.sample(hangul_indices, min(typo_count, len(hangul_indices)))

    # 결과 문자열 생성
    result = list(word)
    for pos in typo_positions:
        result[pos] = generate_typo(word[pos])

    return ''.join(result)


def substitute(char):
    """키보드에서 인접한 키를 눌러 발생하는 오타 생성"""
    if not (0xAC00 <= ord(char) <= 0xD7A3):  # 한글인지 확인
        return char

    # 한글 분리
    cho, jung, jong = hgtk.letter.decompose(char)

    # 기본적으로 기존 자모 유지
    new_cho = cho
    new_jung = jung
    new_jong = jong

    # 초성, 중성, 종성 중 무작위로 하나 선택
    part_to_change = random.choice(['cho', 'jung', 'jong'])

    if part_to_change == 'cho':
        # 초성을 인접한 키로 변경
        cho_idx = CHO.index(cho)
        if cho_idx in choseong_adjacent:
            new_cho_idx = random.choice(choseong_adjacent[cho_idx])
            new_cho = CHO[new_cho_idx]

    elif part_to_change == 'jung':
        # 중성을 인접한 키로 변경
        jung_idx = JOONG.index(jung)
        if jung_idx in jungseong_adjacent:
            new_jung_idx = random.choice(jungseong_adjacent[jung_idx])
            new_jung = JOONG[new_jung_idx]

    elif part_to_change == 'jong':
        # 종성 추가/제거/변경
        if jong == '':  # 종성이 없는 경우, 랜덤 종성 추가
            new_jong = random.choice(hgtk.letter.JONG[1:])  # 빈 종성 제외
        else:
            # 종성이 있는 경우, 제거하거나 다른 종성으로 변경
            new_jong = random.choice([''] + [j for j in hgtk.letter.JONG if j != jong and j != ''])

    # 변형된 한글 조합
    try:
        return hgtk.letter.compose(new_cho, new_jung, new_jong)
    except hgtk.exception.CompositionError:
        return char  # 조합 불가능한 경우 원래 문자 반환


def insert_jamo(char):
    """문자 뒤에 자음 또는 모음만 추가하는 함수"""
    # 자음(초성)과 모음(중성) 중 하나 선택
    jamo_type = random.choice(['cho', 'jung'])

    if jamo_type == 'cho':
        # 자음 중 하나 선택
        cho_idx = random.choice(list(choseong_adjacent.keys()))
        if cho_idx < len(CHO):
            return char + CHO[cho_idx]
    else:
        # 모음 중 하나 선택
        jung_idx = random.choice(list(jungseong_adjacent.keys()))
        if jung_idx < len(JOONG):
            return char + JOONG[jung_idx]

    return char  # 변경 불가능한 경우


def delete_jamo(jamos):
    """
    자모 리스트에서 랜덤하게 하나의 자모를 삭제

    Args:
        jamos (list): (자모 타입, 자모 문자) 튜플 리스트

    Returns:
        list: 삭제 후 남은 자모 리스트
    """
    if len(jamos) <= 1:
        return []  # 자모가 하나 이하면 삭제 후 빈 리스트 반환
    delete_position = random.randint(0, len(jamos) - 1)
    return [j for i, j in enumerate(jamos) if i != delete_position]


def transpose_jamo(word):
    """
    단어의 자모를 분해하고, 인접한 자모를 교환한 뒤 다시 합치는 함수

    Args:
        word (str): 원본 단어

    Returns:
        str: 자모 교환 후 재구성된 단어
    """
    jamo_sequence = decompose_sentence(word)
    if len(jamo_sequence) < 2:
        return word

    # 자모 교환
    i = random.randint(0, len(jamo_sequence) - 2)
    jamo_sequence[i], jamo_sequence[i + 1] = jamo_sequence[i + 1], jamo_sequence[i]

    # 자모 재구성
    return recompose_jamos(jamo_sequence)


def decompose_sentence(sentence):
    """
    문장을 자모 단위로 분해하는 함수

    Args:
        sentence (str): 입력 문장

    Returns:
        list: (자모 타입, 자모 문자) 튜플의 리스트
    """
    jamos = []
    for char in sentence:
        if hgtk.checker.is_hangul(char):
            cho, jung, jong = hgtk.letter.decompose(char)
            jamos.append(('cho', cho))
            jamos.append(('jung', jung))
            if jong != '':  # 종성이 있는 경우
                jamos.append(('jong', jong))
        else:
            jamos.append(('char', char))  # 한글 외 문자는 그대로 추가
    return jamos


def recompose_jamos(jamos):
    """
    자모 리스트를 다시 한글 문자로 조합

    Args:
        jamos (list): (자모 타입, 자모 문자) 튜플 리스트

    Returns:
        str: 조합된 문자열
    """
    result = []
    i = 0

    while i < len(jamos):
        # 초성+중성+종성 또는 초성+중성 조합 시도
        if i + 1 < len(jamos) and jamos[i][0] == 'cho' and jamos[i + 1][0] == 'jung':
            cho = jamos[i][1]
            jung = jamos[i + 1][1]
            jong = ''

            # 종성이 있는지 확인
            if i + 2 < len(jamos) and jamos[i + 2][0] == 'jong':
                jong = jamos[i + 2][1]
                i += 3
            else:
                i += 2

            # 한글 조합 시도
            try:
                combined = hgtk.letter.compose(cho, jung, jong)
                result.append(combined)
            except Exception:
                # 조합 실패 시 개별 자모 추가 대신 채움 문자 사용
                if cho in hgtk.const.CHOSUNG:
                    # 초성 채움 문자 사용 (ㅇ + 해당 초성)
                    try:
                        result.append(hgtk.letter.compose(cho, 'ㅏ', ''))
                    except:
                        result.append(cho)  # 실패하면 그대로 추가
                else:
                    result.append(cho)

                if jung in hgtk.const.JUNGSUNG:
                    # 중성은 'ㅇ'과 결합
                    try:
                        result.append(hgtk.letter.compose('ㅇ', jung, ''))
                    except:
                        result.append(jung)  # 실패하면 그대로 추가
                else:
                    result.append(jung)

                if jong:
                    if jong in hgtk.const.JONGSUNG:
                        try:
                            # 종성은 'ㅇ'+'ㅏ'+종성으로 결합
                            result.append(hgtk.letter.compose('ㅇ', 'ㅏ', jong))
                        except:
                            result.append(jong)  # 실패하면 그대로 추가
                    else:
                        result.append(jong)
        else:
            # 일반 문자나 단일 자모는 그대로 추가
            result.append(jamos[i][1])
            i += 1

    return ''.join(result)


def augment_sentence(sentence, prob=0.1):
    """
    다양한 오타 생성 방법을 무작위로 선택하여 문장을 증강하는 함수

    Args:
        sentence (str): 원본 문장
        prob (float): 증강 확률

    Returns:
        str: 증강된 문장
    """
    # 문장을 단어 단위로 분리
    words = sentence.split()
    # 사용 가능한 증강 방법들
    methods = ['substitute', 'insert_jamo', 'delete_jamo', 'transpose_jamo']
    augmented_words = []

    for word in words:
        # 랜덤으로 증강 방법 하나 선택
        method = random.choice(methods)

        if method == 'substitute':
            # 각 문자에 대해 prob 확률로 substitute 적용
            augmented_word = ''.join(
                substitute(char) if random.random() < prob else char
                for char in word
            )
        elif method == 'insert_jamo':
            # 각 문자 뒤에 prob 확률로 자음/모음 추가
            augmented_word = ''
            for char in word:
                augmented_word += char
                if random.random() < prob:
                    augmented_word += insert_jamo(char)[-1]  # 마지막 문자(추가된 자모)만 사용
            augmented_word = augmented_word.rstrip()
        elif method == 'delete_jamo':
            # 단어를 자모 단위로 분해 후 삭제
            jamos = decompose_sentence(word)
            if random.random() < prob and jamos:
                jamos = delete_jamo(jamos)
            augmented_word = recompose_jamos(jamos)
        elif method == 'transpose_jamo':
            # prob 확률로 자모 교환
            augmented_word = transpose_jamo(word) if random.random() < prob else word

        augmented_words.append(augmented_word)

    # 증강된 단어들을 다시 문장으로 조합
    return ' '.join(augmented_words)
