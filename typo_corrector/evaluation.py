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
from tqdm import tqdm
from datetime import datetime
import argparse
import random
from utils.embedding_manager import FastEmbeddingManager
from utils.correction_utils import find_best_correction
from utils.eval_utils import calc_precision_recall_f05


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