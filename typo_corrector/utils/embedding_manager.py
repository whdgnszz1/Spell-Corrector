import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


class FastEmbeddingManager:
    """
    문장 임베딩을 관리하는 클래스

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
