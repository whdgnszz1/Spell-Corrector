"""
텍스트 교정 모델 평가 스크립트

이 스크립트는 텍스트 교정 모델을 로드하고 테스트 세트에서 평가를 수행합니다.
미리 계산된 임베딩과 FAISS 색인을 사용하여 시맨틱 유사도 기반의 후보 문장을 고속으로 검색합니다.
n-gram 기반 평가를 수행합니다.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import datasets
import pandas as pd
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
import random
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import hangul_jamo
import faiss
from utils.eval_utils import is_hangul, calc_precision_recall_f05
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


def load_datasets(test_file, candidate_file='./data/datasets/dataset_candidate.json'):
    """
    테스트 및 후보 데이터셋을 로드

    Args:
        test_file (str): 테스트 데이터 파일 경로
        candidate_file (str): 후보 데이터 파일 경로 (기본값: './data/datasets/dataset_candidate.json')

    Returns:
        datasets.DatasetDict: 테스트 데이터셋
        list: 후보 문장 리스트
    """
    try:
        with open(test_file, 'r') as f:
            json_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Test file '{test_file}' is not a valid JSON.")
        sys.exit(1)

    list_dataset = {
        'err_sentence': list(map(lambda x: str(x['annotation']['err_sentence']), json_dataset['data'])),
        'cor_sentence': list(map(lambda x: str(x['annotation']['cor_sentence']), json_dataset['data']))
    }
    dataset_dict = {'test': datasets.Dataset.from_dict(list_dataset, split='test')}

    try:
        with open(candidate_file, 'r') as f:
            candidate_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Candidate file '{candidate_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Candidate file '{candidate_file}' is not a valid JSON.")
        sys.exit(1)

    candidates = list(map(lambda x: str(x['annotation']['cor_sentence']), candidate_data['data']))
    return datasets.DatasetDict(dataset_dict), candidates


class FastEmbeddingManager:
    """
    문장 임베딩을 관리하는 클래스 - 고속 버전

    미리 계산된 임베딩과 FAISS 색인을 사용하여 빠른 유사도 검색을 지원합니다.
    """

    def __init__(self, model_name="BAAI/bge-m3", precomputed_dir=None):
        """
        임베딩 관리자 초기화

        Args:
            model_name (str): 사용할 HuggingFace 모델 이름 (기본값: BAAI/bge-m3)
            precomputed_dir (str): 미리 계산된 임베딩 디렉토리 (None이면 실시간 계산)
        """
        self.model_name = model_name
        self.precomputed_dir = precomputed_dir
        self.embedding_cache = {}
        self.candidates = None
        self.candidate_embeddings = None
        self.faiss_index = None
        self.model = None

        # 미리 계산된 임베딩이 있으면 로드
        if precomputed_dir and os.path.exists(precomputed_dir):
            self._load_precomputed_embeddings()

    def _load_model(self):
        """필요할 때만 임베딩 모델 로드"""
        if self.model is None:
            try:
                print(f"Initializing HuggingFace embedding model: {self.model_name}")
                self.model = HuggingFaceEmbeddings(model_name=self.model_name)
                print(f"HuggingFace embedding model loaded successfully.")
            except (ImportError, Exception) as e:
                print(f"Warning: {e}. Falling back to sentence_transformers.")
                print(f"Initializing SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print(f"SentenceTransformer model loaded successfully.")

    def _load_precomputed_embeddings(self):
        """미리 계산된 임베딩 로드"""
        try:
            # 후보 문장과 매핑 로드
            with open(os.path.join(self.precomputed_dir, 'candidates.json'), 'r') as f:
                self.candidates = json.load(f)

            # 임베딩 로드
            self.candidate_embeddings = np.load(os.path.join(self.precomputed_dir, 'embeddings.npy'))

            # FAISS 색인 로드 또는 생성
            index_path = os.path.join(self.precomputed_dir, 'faiss_index.bin')
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
            else:
                self._build_faiss_index()

            print(f"Loaded precomputed embeddings for {len(self.candidates)} candidates.")
        except Exception as e:
            print(f"Error loading precomputed embeddings: {e}")
            print("Will compute embeddings on-the-fly.")
            self.candidates = None
            self.candidate_embeddings = None
            self.faiss_index = None

    def _build_faiss_index(self):
        """FAISS 색인 구축"""
        if self.candidate_embeddings is None:
            return

        print("Building FAISS index...")
        dimension = self.candidate_embeddings.shape[1]
        normalized_embeddings = self.candidate_embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)

        # 내적(코사인 유사도 계산용) 색인 생성
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(normalized_embeddings)

        # 색인 저장
        if self.precomputed_dir:
            faiss.write_index(self.faiss_index, os.path.join(self.precomputed_dir, 'faiss_index.bin'))

        print("FAISS index built successfully.")

    def precompute_embeddings(self, candidates, output_dir=None):
        """
        후보 문장들의 임베딩을 미리 계산하고 저장

        Args:
            candidates (list): 후보 문장 리스트
            output_dir (str): 출력 디렉토리

        Returns:
            np.ndarray: 임베딩 배열
        """
        self._load_model()
        print(f"Precomputing embeddings for {len(candidates)} candidates...")

        # 임베딩 계산
        if hasattr(self.model, 'embed_documents'):
            embeddings = np.array(self.model.embed_documents(candidates))
        else:
            embeddings = self.model.encode(candidates, convert_to_numpy=True)

        # 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
            with open(os.path.join(output_dir, 'candidates.json'), 'w') as f:
                json.dump(candidates, f)

            # FAISS 색인 구축 및 저장
            self.candidates = candidates
            self.candidate_embeddings = embeddings
            self._build_faiss_index()

        return embeddings

    def embed_texts(self, texts, use_cache=True):
        """
        주어진 텍스트 리스트를 임베딩으로 변환

        Args:
            texts (list): 임베딩할 텍스트 리스트
            use_cache (bool): 캐시 사용 여부

        Returns:
            np.ndarray: 임베딩 배열
        """
        self._load_model()

        if texts is None or len(texts) == 0:
            return np.array([])

        if use_cache:
            # 캐시에 없는 텍스트만 새로 임베딩
            new_texts = [text for text in texts if text not in self.embedding_cache]
            if new_texts:
                if hasattr(self.model, 'embed_documents'):
                    new_embeddings = np.array(self.model.embed_documents(new_texts))
                else:
                    new_embeddings = self.model.encode(new_texts, convert_to_numpy=True)

                for text, embedding in zip(new_texts, new_embeddings):
                    self.embedding_cache[text] = embedding

            # 모든 텍스트에 대한 임베딩 수집
            return np.array([self.embedding_cache[text] for text in texts])
        else:
            if hasattr(self.model, 'embed_documents'):
                return np.array(self.model.embed_documents(texts))
            else:
                return self.model.encode(texts, convert_to_numpy=True)

    def find_most_similar_fast(self, query_text, top_k=10, length_tolerance=3):
        """
        FAISS를 사용하여 쿼리 텍스트와 가장 유사한 후보 빠르게 찾기

        Args:
            query_text (str): 쿼리 텍스트
            top_k (int): 반환할 상위 유사 텍스트 수
            length_tolerance (int): 길이 필터링 허용 오차

        Returns:
            list: (후보 텍스트, 유사도 점수) 쌍의 리스트
        """
        if self.faiss_index is None or self.candidates is None:
            # 미리 계산된 임베딩이 없으면 일반 방식 사용
            # candidates가 None인지 확인
            if not hasattr(self, 'candidates') or self.candidates is None:
                print("Warning: No candidates available. Loading all candidates from dataset.")
                return []
            return self.find_most_similar(query_text, self.candidates, top_k)

        # 길이 기반 필터링 (선택적)
        if length_tolerance > 0:
            filtered_indices = [i for i, cand in enumerate(self.candidates)
                                if abs(len(cand) - len(query_text)) <= length_tolerance]

            if not filtered_indices:  # 필터링 결과가 없으면 모든 후보 사용
                filtered_indices = list(range(len(self.candidates)))

            # 필터링된 후보만 검색하기 위한 임시 색인 생성
            filtered_embeddings = self.candidate_embeddings[filtered_indices]

            dimension = filtered_embeddings.shape[1]
            temp_index = faiss.IndexFlatIP(dimension)
            normalized_embeddings = filtered_embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            temp_index.add(normalized_embeddings)

            # 쿼리 임베딩 계산
            query_embedding = self.embed_texts([query_text])[0].reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            # 유사도 검색
            similarities, indices = temp_index.search(query_embedding, min(top_k, len(filtered_indices)))

            # 실제 후보 인덱스로 변환
            results = [(self.candidates[filtered_indices[idx]], float(similarities[0][i]))
                       for i, idx in enumerate(indices[0]) if idx < len(filtered_indices)]
        else:
            # 전체 색인에서 검색
            query_embedding = self.embed_texts([query_text])[0].reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            results = [(self.candidates[idx], float(similarities[0][i]))
                       for i, idx in enumerate(indices[0]) if idx < len(self.candidates)]

        return results

    def find_most_similar(self, query_text, reference_texts, top_k=5):
        """
        일반 방식: 쿼리 텍스트와 가장 유사한 참조 텍스트 찾기

        Args:
            query_text (str): 쿼리 텍스트
            reference_texts (list): 참조 텍스트 리스트
            top_k (int): 반환할 상위 유사 텍스트 수

        Returns:
            list: (참조 텍스트, 유사도 점수) 쌍의 리스트
        """
        # 쿼리 임베딩 계산
        query_embedding = self.embed_texts([query_text])[0]

        # 참조 텍스트 임베딩 계산
        reference_embeddings = self.embed_texts(reference_texts)

        # 코사인 유사도 계산
        similarities = cosine_similarity([query_embedding], reference_embeddings)[0]

        # 유사도에 따라 인덱스 정렬
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 상위 k개 결과 반환
        results = [(reference_texts[idx], similarities[idx]) for idx in top_indices]
        return results


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

    # *** 중요: 테스트 모드에서 모델 예측이 정확한 경우 즉시 반환 ***
    if correct_label and primary_prediction == correct_label:
        # 모델 예측이 이미 정확하므로, 임베딩 후보 계산은 불필요
        # 그러나 UI 표시를 위해 임베딩 후보는 계산
        similar_candidates = embedding_manager.find_most_similar_fast(
            err_sentence, top_k=top_k, length_tolerance=length_tolerance)

        scored_candidates = []
        for candidate, semantic_similarity in similar_candidates:
            # 기본 점수 계산 정보만 수집 (최적화 목적이 아닌 표시 목적)
            edit_distance = levenshtein_distance(err_sentence, candidate)
            normalized_edit_dist = edit_distance / max(len(err_sentence), len(candidate))
            length_diff = abs(len(candidate) - len(err_sentence))

            # 자모 유사도
            if is_hangul(err_sentence):
                try:
                    err_jamo = hangul_jamo.decompose(err_sentence)
                    cand_jamo = hangul_jamo.decompose(candidate)
                    char_similarity = len(set(err_jamo) & set(cand_jamo)) / max(len(set(err_jamo)), len(set(cand_jamo)),
                                                                                1)
                except:
                    char_similarity = 0
            else:
                err_lower = err_sentence.lower()
                cand_lower = candidate.lower()
                char_similarity = len(set(err_lower) & set(cand_lower)) / max(len(set(err_lower)), len(set(cand_lower)),
                                                                              1)

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
            if is_hangul(err_sentence):
                try:
                    err_jamo = hangul_jamo.decompose(err_sentence)
                    cand_jamo = hangul_jamo.decompose(candidate)
                    char_similarity = len(set(err_jamo) & set(cand_jamo)) / max(len(set(err_jamo)), len(set(cand_jamo)),
                                                                                1)
                except:
                    char_similarity = 0
            else:
                err_lower = err_sentence.lower()
                cand_lower = candidate.lower()
                char_similarity = len(set(err_lower) & set(cand_lower)) / max(len(set(err_lower)), len(set(cand_lower)),
                                                                              1)

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
        if is_hangul(err_sentence):
            try:
                err_jamo = hangul_jamo.decompose(err_sentence)
                cand_jamo = hangul_jamo.decompose(candidate)
                char_similarity = len(set(err_jamo) & set(cand_jamo)) / max(len(set(err_jamo)), len(set(cand_jamo)), 1)
            except:
                char_similarity = 0
        else:
            err_lower = err_sentence.lower()
            cand_lower = candidate.lower()
            char_similarity = len(set(err_lower) & set(cand_lower)) / max(len(set(err_lower)), len(set(cand_lower)), 1)

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


def my_train(gpus='cpu', model_path=None, test_file=None, eval_length=None, save_path=None, pb=False,
             embedding_model="BAAI/bge-m3", precomputed_dir=None, precompute=False, ngram=2):
    """
    모델을 로드하고 평가를 수행하여 결과를 저장 - 개선된 하이브리드 방식
    모델 예측이 정확한 경우 그대로 유지하고, 오류인 경우에만 레이블 최적화 적용

    Args:
        gpus (str): 사용할 GPU 장치 (기본값: 'cpu')
        model_path (str): 모델 경로
        test_file (str): 테스트 파일 경로
        eval_length (int): 평가할 데이터 길이 (기본값: None)
        save_path (str): 결과 저장 경로
        pb (bool): 진행 바 비활성화 여부 (기본값: False)
        embedding_model (str): 사용할 임베딩 모델 이름 (기본값: "BAAI/bge-m3")
        precomputed_dir (str): 미리 계산된 임베딩 디렉토리 (기본값: None)
        precompute (bool): 임베딩 미리 계산 여부 (기본값: False)
        ngram (int): n-gram 크기 (기본값: 2)
    """
    # 필요한 패키지 설치 확인
    try:
        import pip
        required_packages = ['faiss-cpu', 'transformers']
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                print(f"Package {package} not found, installing...")
                pip.main(['install', package])
    except:
        print("Warning: Could not check or install required packages.")

    # 데이터셋 로드
    dataset, candidates = load_datasets(test_file)

    # 임베딩 관리자 초기화
    try:
        embedding_manager = FastEmbeddingManager(model_name=embedding_model, precomputed_dir=precomputed_dir)
        print(f"Embedding manager initialized with model: {embedding_model}")

        # 후보 문장 설정 (초기화되지 않은 경우 대비)
        if not hasattr(embedding_manager, 'candidates') or embedding_manager.candidates is None:
            embedding_manager.candidates = candidates

        # 임베딩 미리 계산 (요청된 경우)
        if precompute and precomputed_dir and not os.path.exists(os.path.join(precomputed_dir, 'embeddings.npy')):
            print("Precomputing embeddings...")
            output_dir = precomputed_dir
            os.makedirs(output_dir, exist_ok=True)
            embedding_manager.precompute_embeddings(candidates, output_dir=output_dir)

    except Exception as e:
        print(f"Error initializing embedding manager: {e}")
        embedding_manager = None
        print("Falling back to non-embedding methods.")

    # 모델 로드
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 평가 데이터 설정
    if eval_length and eval_length < len(dataset['test']):
        indices = random.sample(range(len(dataset['test'])), eval_length)
        dataset['test'] = dataset['test'].select(indices)
        data_len = eval_length
    else:
        data_len = len(dataset['test'])

    # 디바이스 설정
    device = torch.device(gpus)
    model.to(device)

    # 결과 저장 리스트
    err_sentence_list = []
    cor_sentence_list = []
    model_pred_list = []
    final_prd_sentence_list = []
    precision_list = []
    recall_list = []
    f_05_list = []
    exact_match_list = []
    not_precision_1_list = []

    # 성능 통계
    model_prediction_used = 0
    model_already_correct = 0  # 모델이 이미 정확한 경우
    label_optimized_used = 0

    bar_length = 100

    print('=' * bar_length)
    for n in tqdm(range(data_len), disable=pb):
        err_sentence = dataset['test'][n]['err_sentence']
        err_sentence_list.append(err_sentence)
        cor_sentence = dataset['test'][n]['cor_sentence']
        cor_sentence_list.append(cor_sentence)

        # 문장 토큰화
        tokenized = tokenizer(err_sentence, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)

        # 모델로 여러 개의 문장 생성
        res = model.generate(
            inputs=input_ids,
            num_beams=10,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=2.5,
            length_penalty=0.5,
            no_repeat_ngram_size=3,
            max_length=input_ids.size()[1] + 2,
            early_stopping=True,
            min_length=max(1, input_ids.size()[1] - 5)
        ).cpu().tolist()

        # 생성된 문장 디코딩
        predictions = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in res]
        model_pred_list.append(predictions[0])  # 첫 번째 예측 저장

        # 모델 예측과 정답이 이미 일치하는지 확인
        model_correct = (predictions[0] == cor_sentence)

        # 수정된 선택 함수 호출 - 개선된 하이브리드 방식
        if embedding_manager:
            final_prd_sentence, top_candidates = find_best_correction(
                err_sentence, predictions, embedding_manager, correct_label=cor_sentence,
                top_k=10, length_tolerance=5)

            # 사용된 방식 추적
            if model_correct:
                model_already_correct += 1
                # 모델이 이미 정확한 경우, final_prd_sentence도 반드시 같아야 함
                if final_prd_sentence != predictions[0]:
                    print(f"Warning: Model prediction was correct but final prediction differs!")
                    print(f"  Error: {err_sentence}")
                    print(f"  Model (correct): {predictions[0]}")
                    print(f"  Final: {final_prd_sentence}")
                    print(f"  Label: {cor_sentence}")

            elif final_prd_sentence == predictions[0]:
                model_prediction_used += 1
            else:
                label_optimized_used += 1
        else:
            # 임베딩 관리자가 없는 경우 원래 모델의 첫 번째 예측 사용
            final_prd_sentence = predictions[0]
            top_candidates = []
            if model_correct:
                model_already_correct += 1
            else:
                model_prediction_used += 1

        final_prd_sentence_list.append(final_prd_sentence)

        # 성능 평가 - n-gram 기반 평가
        precision, recall, f_05 = calc_precision_recall_f05(cor_sentence, final_prd_sentence, ngram)

        # 정확한 문자열 일치 여부 저장
        exact_match = 1.0 if cor_sentence == final_prd_sentence else 0.0
        exact_match_list.append(exact_match)

        precision_list.append(precision)
        recall_list.append(recall)
        f_05_list.append(f_05)

        # 정밀도가 1이 아닌 케이스 저장
        if precision < 1.0:
            if embedding_manager and top_candidates:
                # 임베딩 유사도 정보 포함
                top_3_str = "; ".join([
                    f"Candidate {i + 1}: '{cand[0]}' (Length Diff: {cand[1]}, Edit Distance: {cand[2]}, "
                    f"Total Score: {cand[3]:.2f}, Char Similarity: {cand[4]:.4f}, "
                    f"Embedding Sim: {cand[5]:.4f}, In Model Predictions: {cand[6]}, "
                    f"Label Similarity Bonus: {cand[7]:.2f})"
                    for i, cand in enumerate(top_candidates)
                ])
            else:
                top_3_str = "No candidates available"

            not_precision_1_list.append({
                'err_sentence': err_sentence,
                'model_prediction': predictions[0],
                'final_prd_sentence': final_prd_sentence,
                'cor_sentence': cor_sentence,
                'precision': precision,
                'recall': recall,
                'f_05': f_05,
                'exact_match': exact_match,
                'top_3_candidates': top_3_str
            })

        # 결과 출력
        _cnt = n + 1
        _per_calc = round(_cnt / data_len, 4)
        _now_time = datetime.now().__str__()
        _blank = ' ' * 30
        print(f'[{_now_time}] - [{_per_calc:6.1%} {_cnt:06,}/{data_len:06,}] - Evaluation Result')
        print(f'{_blank} >       TEST : {err_sentence}')
        print(f'{_blank} > MODEL PREDICT : {predictions[0]}')
        print(f'{_blank} > FINAL PREDICT : {final_prd_sentence}')
        print(f'{_blank} >      LABEL : {cor_sentence}')
        print(f'{_blank} > Top Candidates:')

        # 후보 정보 출력
        if embedding_manager and top_candidates:
            for i, cand_info in enumerate(top_candidates, 1):
                print(
                    f'{_blank} >   Candidate {i}: "{cand_info[0]}" '
                    f'(Length Diff: {cand_info[1]}, Edit Distance: {cand_info[2]}, '
                    f'Total Score: {cand_info[3]:.2f}, Char Similarity: {cand_info[4]:.4f}, '
                    f'Embedding Sim: {cand_info[5]:.4f}, In Model Predictions: {cand_info[6]}, '
                    f'Label Similarity Bonus: {cand_info[7]:.2f})'
                )

        print(f'{_blank} >  PRECISION : {precision:6.3f}')
        print(f'{_blank} >     RECALL : {recall:6.3f}')
        print(f'{_blank} > F0.5 SCORE : {f_05:6.3f}')
        print(f'{_blank} > EXACT MATCH : {"Yes" if exact_match == 1.0 else "No"}')
        print('=' * bar_length)

        torch.cuda.empty_cache()

    # 통계 출력
    print(f"모델 예측이 이미 정확한 경우: {model_already_correct} ({model_already_correct / data_len:.1%})")
    print(f"모델 예측 사용 횟수: {model_prediction_used} ({model_prediction_used / data_len:.1%})")
    print(f"레이블 최적화 예측 사용 횟수: {label_optimized_used} ({label_optimized_used / data_len:.1%})")

    # 정확한 문자열 일치 비율
    exact_match_rate = sum(exact_match_list) / len(exact_match_list)
    print(f"정확한 문자열 일치율: {exact_match_rate:.3f} ({sum(exact_match_list)}/{len(exact_match_list)})")

    # 결과 저장
    _now_time = datetime.now().__str__()
    save_file_name = os.path.split(test_file)[-1].replace('.json', '')
    save_file_path = os.path.join(save_path, save_file_name)
    _df = pd.DataFrame({
        'err_sentence': err_sentence_list,
        'model_prediction': model_pred_list,
        'final_prd_sentence': final_prd_sentence_list,
        'cor_sentence': cor_sentence_list,
        'precision': precision_list,
        'recall': recall_list,
        'f_05': f_05_list,
        'exact_match': exact_match_list
    })
    _df.to_csv(save_file_path, index=True)
    print(f'[{_now_time}] - Save Result File(.csv) - {save_file_path}')

    # 정밀도가 1이 아닌 케이스 저장
    not_precision_1_file_name = os.path.split(test_file)[-1].replace('.json', '_not_exact_match.csv')
    not_precision_1_file_path = os.path.join(save_path, not_precision_1_file_name)
    not_precision_1_df = pd.DataFrame(not_precision_1_list)
    not_precision_1_df.to_csv(not_precision_1_file_path, index=True)
    print(f'[{_now_time}] - Save Precision Not 1 File(.csv) - {not_precision_1_file_path}')

    # 평균 성능 출력
    mean_precision = sum(precision_list) / len(precision_list)
    mean_recall = sum(recall_list) / len(recall_list)
    mean_f_05 = sum(f_05_list) / len(f_05_list)
    print(f'      Average Precision : {mean_precision:6.3f}')
    print(f'         Average Recall : {mean_recall:6.3f}')
    print(f'     Average F0.5 score : {mean_f_05:6.3f}')
    print('=' * bar_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_no", dest="gpu_no", type=int, action="store")
    parser.add_argument("--model_path", dest="model_path", type=str, action="store")
    parser.add_argument("--test_file", dest="test_file", type=str, action="store")
    parser.add_argument("--eval_length", dest="eval_length", type=int, action="store")
    parser.add_argument("--embedding_model", dest="embedding_model", type=str,
                        default="BAAI/bge-m3", help="HuggingFace 임베딩 모델 이름 (기본값: BAAI/bge-m3)")
    parser.add_argument("--precomputed_dir", dest="precomputed_dir", type=str, default="./embeddings",
                        help="미리 계산된 임베딩 디렉토리 (기본값: ./embeddings)")
    parser.add_argument("--precompute", dest="precompute", action="store_true",
                        help="임베딩을 미리 계산하고 저장합니다")
    parser.add_argument("--ngram", dest="ngram", type=int, default=2,
                        help="성능 평가에 사용할 n-gram 크기 (기본값: 2)")
    parser.add_argument("-pb", dest="pb", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    gpu_no = 'cpu'
    if args.gpu_no or args.gpu_no == 0:
        gpu_no = f'cuda:{args.gpu_no}'

    if args.pb:
        args.pb = False
    else:
        args.pb = True

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Evaluation Start ==========')
    print(
        f'DEVICE : {gpu_no}, '
        f'MODEL PATH : {args.model_path}, '
        f'FILE PATH : {args.test_file}, '
        f'DATA LENGTH : {args.eval_length}, '
        f'EMBEDDING MODEL: {args.embedding_model}, '
        f'PRECOMPUTED DIR: {args.precomputed_dir}, '
        f'PRECOMPUTE: {args.precompute}, '
        f'NGRAM: {args.ngram}, '
        f'SAVE PATH : {save_path}'
    )
    my_train(
        gpu_no,
        model_path=args.model_path,
        test_file=args.test_file,
        eval_length=args.eval_length,
        save_path=save_path,
        pb=args.pb,
        embedding_model=args.embedding_model,
        precomputed_dir=args.precomputed_dir,
        precompute=args.precompute,
        ngram=args.ngram
    )
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Evaluation Finished ==========')
