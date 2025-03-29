"""
텍스트 데이터 증강을 위한 유틸리티 모듈

이 모듈은 자연어 처리(NLP)에서 사용되는 다양한 텍스트 증강 기법을 제공합니다.
특히 한글 문장 증강에 중점을 두고, 오탈자 생성, 단어 교체, 백트랜슬레이션 등의 방법을 구현합니다.
모델 훈련 데이터를 확장하고 텍스트 교정 모델의 일반화 성능을 향상시키는 데 사용됩니다.
"""

import random  # 랜덤 연산을 위한 모듈
import nltk  # 자연어 처리 도구 모듈
from gensim.models import KeyedVectors  # 워드 임베딩 모델 로드를 위한 클래스
import torch  # PyTorch 딥러닝 프레임워크
from transformers import MarianMTModel, MarianTokenizer  # 번역 모델 및 토크나이저
import logging  # 로깅을 위한 모듈
from tqdm import tqdm  # 진행 상태 표시 바
import os  # 파일 시스템 조작 
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리
from functools import lru_cache  # 메모이제이션 데코레이터

# NLTK 데이터 확인 및 다운로드
try:
    nltk.data.find('corpora/stopwords')  # stopwords 데이터가 있는지 확인
except LookupError:
    nltk.download('stopwords', quiet=True)  # 없으면 다운로드 (조용한 모드)

# 로거 설정
logger = logging.getLogger(__name__)  # 현재 모듈의 로거 가져오기

# 사전 로드 속도를 개선하기 위한 임베딩 딕셔너리 (전역 캐시)
word_embedding_cache = {}


# 한글 처리 관련 함수들
def decompose_hangul(char):
    """
    한글 문자를 초성, 중성, 종성으로 분리

    현대 한글(완성형)은 초성 19개, 중성 21개, 종성 28개(종성 없음 포함)로 구성됩니다.
    이를 유니코드 값을 기준으로 분리합니다.

    Args:
        char (str): 단일 한글 문자

    Returns:
        tuple or str: 초성, 중성, 종성 인덱스 튜플 또는 원본 문자(한글이 아닌 경우)
    """
    if not (0xAC00 <= ord(char) <= 0xD7A3):  # 한글 유니코드 범위(U+AC00 ~ U+D7A3) 체크
        return char
    code = ord(char) - 0xAC00  # 한글 유니코드 시작점(AC00)을 기준으로 오프셋 계산
    jongseong = code % 28  # 종성 인덱스 (0: 종성 없음)
    jungseong = ((code - jongseong) // 28) % 21  # 중성 인덱스
    choseong = ((code - jongseong) // 28) // 21  # 초성 인덱스
    return choseong, jungseong, jongseong


def compose_hangul(choseong, jungseong, jongseong):
    """
    초성, 중성, 종성 인덱스를 조합하여 한글 문자를 생성

    decompose_hangul의 역연산을 수행합니다.

    Args:
        choseong (int): 초성 인덱스 (0-18)
        jungseong (int): 중성 인덱스 (0-20)
        jongseong (int): 종성 인덱스 (0-27, 0은 종성 없음)

    Returns:
        str: 조합된 한글 문자
    """
    # 유니코드 계산 공식: 0xAC00 + (초성 * 21 + 중성) * 28 + 종성
    return chr(0xAC00 + (choseong * 21 + jungseong) * 28 + jongseong)


# 한글 자판 위치에 따른 인접 키 정의
# 두벌식 자판 기준으로 초성과 중성의 인접 키를 정의합니다
choseong_adjacent = {
    0: [1, 2, 6], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3, 5],
    5: [4, 6], 6: [0, 5, 7], 7: [6, 8], 8: [7, 9], 9: [8]
}  # 초성에 대한 인접 키 딕셔너리 (두벌식 자판 기준)

jungseong_adjacent = {
    0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2], 4: [0, 5], 5: [4]
}  # 중성에 대한 인접 키 딕셔너리 (두벌식 자판 기준)


# 기본 증강 함수들
def substitute(char, choseong_adjacent, jungseong_adjacent):
    """
    한글 문자의 초성 또는 중성을 인접한 자판 키로 대체하는 함수

    오타 생성을 시뮬레이션하기 위해 키보드 상에서 인접한 키를 선택합니다.

    Args:
        char (str): 변형할 한글 문자
        choseong_adjacent (dict): 초성 인접 키 맵핑
        jungseong_adjacent (dict): 중성 인접 키 맵핑

    Returns:
        str: 변형된 문자 또는 원본 문자(변경 불가능한 경우)
    """
    if not (0xAC00 <= ord(char) <= 0xD7A3):  # 한글이 아니면 원본 반환
        return char

    choseong, jungseong, jongseong = decompose_hangul(char)  # 한글 분리
    choice = random.choice(['choseong', 'jungseong'])  # 초성 또는 중성 중 하나를 랜덤 선택

    if choice == 'choseong' and choseong in choseong_adjacent:
        # 인접한 초성 중 하나를 선택하여 대체
        new_choseong = random.choice(choseong_adjacent[choseong])
        return compose_hangul(new_choseong, jungseong, jongseong)
    elif choice == 'jungseong' and jungseong in jungseong_adjacent:
        # 인접한 중성 중 하나를 선택하여 대체
        new_jungseong = random.choice(jungseong_adjacent[jungseong])
        return compose_hangul(choseong, new_jungseong, jongseong)

    return char  # 대체 불가능한 경우 원본 반환


def augment_substitute(sentence, prob=0.1):
    """
    문장의 각 문자에 대해 일정 확률로 대체 증강을 수행하는 함수

    Args:
        sentence (str): 원본 문장
        prob (float): 각 문자를 대체할 확률 (0.0 ~ 1.0)

    Returns:
        str: 증강된 문장
    """
    # 각 문자에 대해 확률에 따라 substitute 함수 적용
    augmented = [substitute(char, choseong_adjacent, jungseong_adjacent)
                 if random.random() < prob else char
                 for char in sentence]
    return ''.join(augmented)  # 문자 리스트를 문자열로 결합


def augment_insert(sentence, prob=0.1):
    """
    문장의 각 문자 뒤에 일정 확률로 새로운 한글 문자를 삽입하는 함수

    삽입 증강은 문자를 추가하여 오타를 시뮬레이션합니다.

    Args:
        sentence (str): 원본 문장
        prob (float): 각 문자 뒤에 새 문자를 삽입할 확률

    Returns:
        str: 증강된 문장
    """
    augmented = []
    for char in sentence:
        augmented.append(char)  # 원본 문자 추가
        if random.random() < prob:  # 확률에 따라 새 문자 삽입
            # 랜덤 초성과 중성으로 새 문자 생성 (종성 없음)
            new_choseong = random.choice(list(choseong_adjacent.keys()))
            new_jungseong = random.choice(list(jungseong_adjacent.keys()))
            augmented.append(compose_hangul(new_choseong, new_jungseong, 0))
    return ''.join(augmented)  # 문자 리스트를 문자열로 결합


def augment_delete(sentence, prob=0.1):
    """
    문장의 각 문자를 일정 확률로 삭제하는 함수

    삭제 증강은 문자가 누락된 오타를 시뮬레이션합니다.

    Args:
        sentence (str): 원본 문장
        prob (float): 각 문자를 삭제할 확률

    Returns:
        str: 증강된 문장
    """
    if len(sentence) <= 1:  # 문장이 너무 짧으면 그대로 반환
        return sentence

    # 확률에 따라 삭제하지 않을 문자만 선택
    augmented = [char for char in sentence if random.random() >= prob]

    if not augmented:  # 모든 문자가 삭제된 경우
        augmented = [random.choice(sentence)]  # 최소 하나의 문자 유지

    return ''.join(augmented)  # 문자 리스트를 문자열로 결합


def augment_transpose(sentence, prob=0.1):
    """
    문장에서 인접한 두 문자를 일정 확률로 교환하는 함수

    전치 증강은 타이핑 시 발생하는 문자 순서 오류를 시뮬레이션합니다.

    Args:
        sentence (str): 원본 문장
        prob (float): 인접 문자를 교환할 확률

    Returns:
        str: 증강된 문장
    """
    if len(sentence) < 2:  # 문장 길이가 2 미만이면 교환 불가
        return sentence

    augmented = list(sentence)  # 문자열을 리스트로 변환
    for i in range(len(augmented) - 1):
        if random.random() < prob:
            # 인접한 두 문자의 위치 교환
            augmented[i], augmented[i + 1] = augmented[i + 1], augmented[i]

    return ''.join(augmented)  # 문자 리스트를 문자열로 결합


def augment_sentence(sentence, prob=0.1):
    """
    문장에 대해 무작위로 하나의 증강 방법을 선택해 적용하는 함수

    여러 증강 방법 중 하나를 랜덤하게 선택하여 적용합니다.

    Args:
        sentence (str): 원본 문장
        prob (float): 증강 적용 확률

    Returns:
        str: 증강된 문장
    """
    # 사용 가능한 증강 방법들
    methods = [augment_substitute, augment_insert, augment_delete, augment_transpose]

    # 랜덤으로 증강 방법 하나 선택
    method = random.choice(methods)

    # 선택된 방법으로 증강 적용
    return method(sentence, prob)


# 단어 임베딩 기반 증강 함수들
def get_word_embeddings(embedding_path):
    """
    워드 임베딩 모델을 로드하는 함수

    중복 로드를 방지하기 위해 캐싱 메커니즘을 사용합니다.

    Args:
        embedding_path (str): 워드 임베딩 모델 파일 경로

    Returns:
        KeyedVectors: 워드 임베딩 모델 또는 None (로드 실패 시)
    """
    global word_embedding_cache  # 전역 캐시 변수 사용

    # 이미 로드된 모델이 있으면 캐시에서 반환
    if embedding_path in word_embedding_cache:
        return word_embedding_cache[embedding_path]

    try:
        # 워드 임베딩 모델 로드 (Word2Vec 바이너리 형식)
        model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

        # 캐시에 저장
        word_embedding_cache[embedding_path] = model
        return model
    except Exception as e:
        logger.warning(f"워드 임베딩 로드 실패: {e}")
        return None


@lru_cache(maxsize=1024)  # 최근 1024개 결과 캐싱
def get_similar_word(word, embedding_model, top_n=5):
    """
    워드 임베딩을 사용하여 유사한 단어를 찾는 함수

    입력 단어와 의미적으로 유사한 단어를 임베딩 공간에서 찾습니다.
    자주 사용되는 단어는 LRU 캐시를 통해 속도를 개선합니다.

    Args:
        word (str): 대상 단어
        embedding_model (KeyedVectors): 워드 임베딩 모델
        top_n (int): 검색할 유사 단어 수

    Returns:
        str: 유사한 단어 또는 원래 단어 (유사어 찾기 실패 시)
    """
    # 모델이 없거나 단어가 임베딩에 없는 경우
    if embedding_model is None or word not in embedding_model.key_to_index:
        return word

    try:
        # 입력 단어와 가장 유사한 top_n개 단어 찾기
        similar_words = embedding_model.most_similar(word, topn=top_n)

        # 유사 단어 중 하나를 랜덤하게 선택
        return random.choice([w for w, _ in similar_words])
    except:
        return word  # 오류 발생 시 원래 단어 반환


def word_replacement(sentence, embedding_model=None, replace_prob=0.1):
    """
    문장 내 단어를 의미적으로 유사한 단어로 대체하는 함수

    의미 변화는 최소화하면서 다양성을 증가시키는 증강 방법입니다.

    Args:
        sentence (str): 원본 문장
        embedding_model (KeyedVectors): 워드 임베딩 모델
        replace_prob (float): 단어 대체 확률

    Returns:
        str: 단어가 대체된 문장
    """
    if embedding_model is None:  # 임베딩 모델이 없으면 원본 반환
        return sentence

    words = sentence.split()  # 문장을 단어로 분리
    result = []

    for word in words:
        # 확률에 따라 단어 대체 (단어 길이가 2 이상인 경우만)
        if random.random() < replace_prob and len(word) > 1:
            result.append(get_similar_word(word, embedding_model))
        else:
            result.append(word)  # 원래 단어 유지

    return ' '.join(result)  # 단어 리스트를 문장으로 결합


# 글자 위치 변경 기반 증강
def swap_word_position(sentence, swap_prob=0.1):
    """
    문장 내 인접한 단어의 위치를 바꾸는 함수

    문장 구조 오류를 시뮬레이션하는 증강 방법입니다.

    Args:
        sentence (str): 원본 문장
        swap_prob (float): 위치 변경 확률

    Returns:
        str: 단어 위치가 변경된 문장
    """
    words = sentence.split()  # 문장을 단어로 분리

    if len(words) < 2:  # 단어가 2개 미만이면 교환 불가
        return sentence

    for i in range(len(words) - 1):
        if random.random() < swap_prob:
            # 인접한 두 단어의 위치 교환
            words[i], words[i + 1] = words[i + 1], words[i]

    return ' '.join(words)  # 단어 리스트를 문장으로 결합


# 백트랜슬레이션 증강 (언어 A -> 언어 B -> 언어 A)
def back_translation_augment(sentences, src_lang='ko', tgt_lang='en', sample_ratio=0.3, token=None):
    """
    백트랜슬레이션을 통한 문장 증강 함수

    원본 언어에서 타겟 언어로 번역 후 다시 원본 언어로 번역하면서
    다양한 표현으로 변형되는 특성을 활용한 증강 방법입니다.

    Args:
        sentences (list): 원본 문장 리스트
        src_lang (str): 소스 언어 코드 (기본값: 'ko')
        tgt_lang (str): 타겟 언어 코드 (기본값: 'en')
        sample_ratio (float): 전체 데이터 중 증강할 비율 (0~1)
        token (str, optional): Hugging Face 인증 토큰

    Returns:
        dict: 증강된 오류 문장('err_sentence')과 정답 문장('cor_sentence') 쌍
    """
    # 데이터 샘플링 (전체의 sample_ratio만큼만 증강)
    sample_size = max(1, int(len(sentences) * sample_ratio))
    sampled_sentences = random.sample(sentences, sample_size)

    try:
        # 번역 모델 및 토크나이저 로드
        src_to_tgt_model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'  # 소스->타겟 번역 모델
        tgt_to_src_model_name = 'Helsinki-NLP/opus-mt-tc-big-en-ko'  # 타겟->소스 번역 모델 (특정 모델 사용)

        # 소스->타겟 번역을 위한 토크나이저와 모델
        src_to_tgt_tokenizer = MarianTokenizer.from_pretrained(src_to_tgt_model_name, token=token)
        src_to_tgt_model = MarianMTModel.from_pretrained(src_to_tgt_model_name, token=token)

        # 타겟->소스 번역을 위한 토크나이저와 모델
        tgt_to_src_tokenizer = MarianTokenizer.from_pretrained(tgt_to_src_model_name, token=token)
        tgt_to_src_model = MarianMTModel.from_pretrained(tgt_to_src_model_name, token=token)

        # 번역 함수 정의 (배치 처리)
        def translate(texts, model, tokenizer, batch_size=8):
            """
            주어진 텍스트를 배치 단위로 번역하는 내부 함수

            Args:
                texts (list): 번역할 텍스트 리스트
                model: 번역 모델
                tokenizer: 토크나이저
                batch_size (int): 배치 크기

            Returns:
                list: 번역된 텍스트 리스트
            """
            # 입력 데이터를 배치로 분할
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            translated = []

            for batch in tqdm(batches, desc="Translating"):  # 진행률 표시
                # 토큰화 및 모델 입력 준비
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

                # 번역 수행 (그래디언트 계산 없이)
                with torch.no_grad():
                    outputs = model.generate(**inputs)

                # 출력 토큰을 텍스트로 디코딩
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translated.extend(decoded)

            return translated

        # 순방향 번역 (소스 -> 타겟)
        intermediate_texts = translate(sampled_sentences, src_to_tgt_model, src_to_tgt_tokenizer)

        # 역방향 번역 (타겟 -> 소스)
        back_translated_texts = translate(intermediate_texts, tgt_to_src_model, tgt_to_src_tokenizer)

        # 결과 구성 (원본과 다른 경우만 포함)
        augmented_data = {'err_sentence': [], 'cor_sentence': []}
        for orig, bt in zip(sampled_sentences, back_translated_texts):
            if orig != bt:  # 원본과 다른 경우만 추가 (의미 있는 변형인 경우)
                augmented_data['err_sentence'].append(bt)  # 번역된 결과를 오류 문장으로
                augmented_data['cor_sentence'].append(orig)  # 원본을 정답 문장으로

        return augmented_data

    except Exception as e:
        logger.error(f"백트랜슬레이션 실패: {e}")  # 오류 로깅
        return {'err_sentence': [], 'cor_sentence': []}  # 빈 결과 반환


# 고급 증강 기법을 적용한 데이터 증강
def advanced_augment_data(original_sentences, augment_prob=0.1, embedding_model=None):
    """
    다양한 고급 증강 기법을 사용하여 원본 문장을 증강하는 함수

    여러 증강 방법을 조합하여 다양한 변형을 생성합니다.
    병렬 처리를 통해 대량의 데이터를 효율적으로 처리합니다.

    Args:
        original_sentences (list): 원본 정답 문장 리스트
        augment_prob (float): 증강 확률
        embedding_model: 워드 임베딩 모델 (선택적)

    Returns:
        dict: 증강된 오류 문장('err_sentence')과 정답 문장('cor_sentence') 쌍
    """
    augmented_data = {'err_sentence': [], 'cor_sentence': []}

    # 병렬 처리를 위한 함수 정의
    def process_sentence(cor):
        """
        단일 문장에 대한 증강을 처리하는 내부 함수

        여러 증강 방법 중 하나를 랜덤하게 선택하여 적용합니다.

        Args:
            cor (str): 원본 문장

        Returns:
            str: 증강된 문장
        """
        # 사용할 증강 방법을 랜덤하게 선택
        aug_method = random.choice([
            lambda s: augment_sentence(s, augment_prob),  # 기본 문자 단위 증강
            lambda s: word_replacement(s, embedding_model, augment_prob),  # 유사 단어 교체
            lambda s: swap_word_position(s, augment_prob),  # 단어 위치 교환
        ])
        return aug_method(cor)  # 선택된 방법으로 증강 적용

    # 병렬 처리로 증강 수행 (CPU 코어 수에 비례한 워커 수 사용)
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as executor:
        # 모든 문장에 대해 병렬로 증강 작업 제출
        futures = [executor.submit(process_sentence, cor) for cor in original_sentences]

        # 결과 수집
        for cor, future in zip(original_sentences, futures):
            try:
                err = future.result()  # 증강 결과 가져오기
                if err != cor:  # 원본과 다른 경우만 추가 (의미 있는 변형인 경우)
                    augmented_data['err_sentence'].append(err)  # 증강 결과를 오류 문장으로
                    augmented_data['cor_sentence'].append(cor)  # 원본을 정답 문장으로
            except Exception as e:
                logger.error(f"문장 증강 실패: {e}")  # 오류 로깅

    return augmented_data


# 워드 임베딩 관련 유틸리티 함수들
def load_word_embeddings_dummy():
    """
    임베딩이 없을 경우 사용하는 더미 함수

    임베딩 파일이 없을 때 호출되며, 경고 메시지와 함께 None을 반환합니다.

    Returns:
        None: 임베딩 모델 대신 None 반환
    """
    logger.warning("워드 임베딩 파일이 지정되지 않았습니다. 워드 임베딩 기반 증강은 사용할 수 없습니다.")
    return None


def load_word_embeddings(embedding_path):
    """
    워드 임베딩을 로드하는 함수

    임베딩 파일 경로를 확인하고 로드를 시도합니다.

    Args:
        embedding_path (str): 워드 임베딩 파일 경로

    Returns:
        object or None: 로드된 임베딩 모델 또는 None (실패 시)
    """
    # 임베딩 경로가 없거나 파일이 존재하지 않으면 더미 함수 호출
    if embedding_path is None or not os.path.exists(embedding_path):
        return load_word_embeddings_dummy()

    try:
        # 워드 임베딩 로드 시도
        return get_word_embeddings(embedding_path)
    except Exception as e:
        logger.error(f"워드 임베딩 로드 실패: {e}")  # 오류 로깅
        return None  # 로드 실패 시 None 반환


# 추가 증강 방법들
def random_casing(sentence, prob=0.1):
    """
    문장 내 단어의 대소문자를 랜덤하게 변경하는 함수

    영어 문장에서 대소문자 오류를 시뮬레이션하는 증강 방법입니다.

    Args:
        sentence (str): 원본 문장
        prob (float): 변경 확률

    Returns:
        str: 대소문자가 변경된 문장
    """
    words = sentence.split()  # 문장을 단어로 분리
    result = []

    for word in words:
        if random.random() < prob:
            # 다양한 대소문자 변환 방법 중 하나를 랜덤하게 선택
            case_method = random.choice([
                str.upper,  # 모두 대문자로
                str.lower,  # 모두 소문자로
                str.capitalize,  # 첫 글자만 대문자로
                lambda w: w.capitalize() if len(w) > 1 else w  # 첫 글자만 대문자로 (1글자는 변경 안 함)
            ])
            result.append(case_method(word))  # 선택된 방법으로 변환
        else:
            result.append(word)  # 원래 단어 유지

    return ' '.join(result)  # 단어 리스트를 문장으로 결합


def punctuation_noise(sentence, prob=0.1):
    """
    문장 내 구두점에 오류를 추가하는 함수

    구두점 관련 오류를 시뮬레이션하는 증강 방법입니다.
    두 가지 방식으로 작동합니다:
    1. 기존 구두점을 일정 확률로 제거
    2. 무작위 위치에 새로운 구두점 추가

    Args:
        sentence (str): 원본 문장
        prob (float): 구두점 변경 확률 (0.0 ~ 1.0)

    Returns:
        str: 구두점이 변경된 문장
    """
    # 일반적인 구두점 리스트
    punctuations = [',', '.', '!', '?', ';', ':']
    chars = list(sentence)  # 문자열을 리스트로 변환하여 수정 가능하게 함

    # 1단계: 기존 구두점 제거 (일정 확률로)
    for i in range(len(chars)):
        if chars[i] in punctuations and random.random() < prob:
            chars[i] = ' '  # 구두점을 공백으로 대체

    # 문자열로 다시 결합
    result = ''.join(chars)
    words = result.split()  # 단어 단위로 분리

    # 문장이 너무 짧으면 구두점 추가 생략
    if len(words) <= 1:
        return result

    # 2단계: 무작위로 구두점 추가
    for i in range(len(words) - 1):
        # 과도한 구두점 추가 방지를 위해 확률 절반으로 감소
        if random.random() < prob / 2:
            # 단어 뒤에 무작위 구두점 추가
            words[i] = words[i] + random.choice(punctuations)

    return ' '.join(words)  # 단어 리스트를 문장으로 결합
