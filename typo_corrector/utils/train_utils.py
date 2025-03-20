import random


# 한글 분리 및 조합 함수
def decompose_hangul(char):
    """
    한글 문자를 초성, 중성, 종성으로 분리
    - 한글 유니코드 범위(0xAC00 ~ 0xD7A3)에 속하지 않으면 원본 문자를 반환
    """
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return char
    code = ord(char) - 0xAC00  # 한글 유니코드 오프셋 계산
    jongseong = code % 28  # 종성 인덱스 계산
    jungseong = ((code - jongseong) // 28) % 21  # 중성 인덱스 계산
    choseong = ((code - jongseong) // 28) // 21  # 초성 인덱스 계산
    return choseong, jungseong, jongseong


def compose_hangul(choseong, jungseong, jongseong):
    """
    초성, 중성, 종성 인덱스를 조합하여 한글 문자를 생성
    - 유니코드 계산 공식: 0xAC00 + (초성 * 21 + 중성) * 28 + 종성
    """
    return chr(0xAC00 + (choseong * 21 + jungseong) * 28 + jongseong)


# 인접 키 정의 (두벌식 자판 기준)
choseong_adjacent = {
    0: [1, 2, 6], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3, 5],
    5: [4, 6], 6: [0, 5, 7], 7: [6, 8], 8: [7, 9], 9: [8]
}  # 초성에 대한 인접 키 딕셔너리 (두벌식 자판 기준)
jungseong_adjacent = {
    0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2], 4: [0, 5], 5: [4]
}  # 중성에 대한 인접 키 딕셔너리 (두벌식 자판 기준)


# 증강 함수
def substitute(char, choseong_adjacent, jungseong_adjacent):
    """
    한글 문자의 초성 또는 중성을 인접한 자판 키로 대체
    - 한글이 아니면 원본 문자를 반환
    - 초성과 중성 중 하나를 랜덤으로 선택해 대체
    """
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return char
    choseong, jungseong, jongseong = decompose_hangul(char)
    choice = random.choice(['choseong', 'jungseong'])  # 초성 또는 중성 선택
    if choice == 'choseong' and choseong in choseong_adjacent:
        new_choseong = random.choice(choseong_adjacent[choseong])  # 인접 초성으로 대체
        return compose_hangul(new_choseong, jungseong, jongseong)
    elif choice == 'jungseong' and jungseong in jungseong_adjacent:
        new_jungseong = random.choice(jungseong_adjacent[jungseong])  # 인접 중성으로 대체
        return compose_hangul(choseong, new_jungseong, jongseong)
    return char  # 대체 불가능하면 원본 반환


def augment_substitute(sentence, prob=0.1):
    """
    문장의 각 문자에 대해 일정 확률(prob)로 대체 증강을 수행
    - substitute 함수를 사용해 초성 또는 중성을 인접 키로 대체
    """
    augmented = [substitute(char, choseong_adjacent, jungseong_adjacent)
                 if random.random() < prob else char
                 for char in sentence]
    return ''.join(augmented)


def augment_insert(sentence, prob=0.1):
    """
    문장의 각 문자 뒤에 일정 확률(prob)로 새로운 한글 문자를 삽입
    - 삽입된 문자는 랜덤 초성과 중성으로 구성되며, 종성은 없음
    """
    augmented = []
    for char in sentence:
        augmented.append(char)
        if random.random() < prob:
            new_choseong = random.choice(list(choseong_adjacent.keys()))  # 랜덤 초성 선택
            new_jungseong = random.choice(list(jungseong_adjacent.keys()))  # 랜덤 중성 선택
            augmented.append(compose_hangul(new_choseong, new_jungseong, 0))
    return ''.join(augmented)


def augment_delete(sentence, prob=0.1):
    """
    문장의 각 문자를 일정 확률(prob)로 삭제
    - 문장이 너무 짧거나 모두 삭제되면 최소 한 문자를 유지
    """
    if len(sentence) <= 1:
        return sentence
    augmented = [char for char in sentence if random.random() >= prob]
    if not augmented:  # 모두 삭제된 경우
        augmented = [random.choice(sentence)]  # 원본에서 랜덤 문자 선택
    return ''.join(augmented)


def augment_transpose(sentence, prob=0.1):
    """
    문장에서 인접한 두 문자를 일정 확률(prob)로 교환
    - 문장 길이가 2 미만이면 교환하지 않음.
    """
    if len(sentence) < 2:
        return sentence
    augmented = list(sentence)
    for i in range(len(augmented) - 1):
        if random.random() < prob:
            augmented[i], augmented[i + 1] = augmented[i + 1], augmented[i]  # 인접 문자 교환
    return ''.join(augmented)


def augment_sentence(sentence, prob=0.1):
    """
    문장에 대해 무작위로 하나의 증강 방법을 선택해 적용
    - 증강 방법: 대체, 삽입, 삭제, 교환 중 하나
    """
    methods = [augment_substitute, augment_insert, augment_delete, augment_transpose]
    method = random.choice(methods)  # 랜덤으로 증강 방법 선택
    return method(sentence, prob)
