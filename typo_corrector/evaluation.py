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
from utils import calc_f_05, find_closest_candidate, select_best_prediction


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


CONSONANTS = set('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')  # 한글 초성 집합
VOWELS = set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')  # 한글 중성 집합


def post_process_prediction(pred, candidates):
    """
    예측 문장을 후처리하여 후보에 있는 단어 또는 길이가 2 이상인 단어만 남김

    Args:
        pred (str): 예측 문장
        candidates (list): 후보 문장 리스트

    Returns:
        str: 후처리된 예측 문장
    """
    candidate_words = set()
    for cand in candidates:
        candidate_words.update(cand.split())
    words = pred.split()
    cleaned_words = [word for word in words if word in candidate_words or len(word) > 2]
    return " ".join(cleaned_words) if cleaned_words else pred


def my_train(gpus='cpu', model_path=None, test_file=None, eval_length=None, save_path=None, pb=False):
    """
    모델을 로드하고 평가를 수행하여 결과를 저장

    Args:
        gpus (str): 사용할 GPU 장치 (기본값: 'cpu')
        model_path (str): 모델 경로
        test_file (str): 테스트 파일 경로
        eval_length (int): 평가할 데이터 길이 (기본값: None)
        save_path (str): 결과 저장 경로
        pb (bool): 진행 바 비활성화 여부 (기본값: False)
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset, candidates = load_datasets(test_file)

    if eval_length and eval_length < len(dataset['test']):
        indices = random.sample(range(len(dataset['test'])), eval_length)
        dataset['test'] = dataset['test'].select(indices)
        data_len = eval_length
    else:
        data_len = len(dataset['test'])

    device = torch.device(gpus)
    model.to(device)

    avg_candidate_length = sum(len(c) for c in candidates) / len(candidates)

    err_sentence_list = []
    cor_sentence_list = []
    raw_prd_sentence_list = []
    final_prd_sentence_list = []
    precision_list = []
    recall_list = []
    f_05_list = []
    not_precision_1_list = []

    ngram = 2
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

        raw_prd_sentence = select_best_prediction(predictions, candidates, ngram, avg_candidate_length)
        raw_prd_sentence_list.append(raw_prd_sentence)

        final_prd_sentence, top_candidates = find_closest_candidate(err_sentence, predictions, candidates)
        final_prd_sentence_list.append(final_prd_sentence)

        precision, recall, f_05 = calc_f_05(cor_sentence, final_prd_sentence, ngram)
        precision_list.append(precision)
        recall_list.append(recall)
        f_05_list.append(f_05)

        if precision != 1:
            top_3_str = "; ".join([
                f"Candidate {i + 1}: '{cand[0]}' (Length Diff: {cand[1]}, Edit Distance: {cand[2]}, Avg Similarity Score: {cand[3]:.4f}, Total Score: {cand[5]:.2f}, Avg RAW PREDICT Edit Distance: {cand[6]:.2f}, Common Chars: {cand[7]})"
                for i, cand in enumerate(top_candidates)
            ])
            not_precision_1_list.append({
                'err_sentence': err_sentence,
                'raw_prd_sentence': raw_prd_sentence,
                'final_prd_sentence': final_prd_sentence,
                'cor_sentence': cor_sentence,
                'precision': precision,
                'recall': recall,
                'f_05': f_05,
                'top_3_candidates': top_3_str
            })

        _cnt = n + 1
        _per_calc = round(_cnt / data_len, 4)
        _now_time = datetime.now().__str__()
        _blank = ' ' * 30
        print(f'[{_now_time}] - [{_per_calc:6.1%} {_cnt:06,}/{data_len:06,}] - Evaluation Result')
        print(f'{_blank} >       TEST : {err_sentence}')
        print(f'{_blank} > RAW PREDICT : {raw_prd_sentence}')
        print(f'{_blank} > FINAL PREDICT : {final_prd_sentence}')
        print(f'{_blank} >      LABEL : {cor_sentence}')
        print(f'{_blank} > Top 3 Candidates:')
        for i, (cand, length_diff, edit_dist, avg_sim_score, max_sim_score, total_score, raw_pred_edit_dist,
                common_chars) in enumerate(top_candidates, 1):
            print(
                f'{_blank} >   Candidate {i}: "{cand}" (Length Diff: {length_diff}, Edit Distance: {edit_dist}, Avg Similarity Score: {avg_sim_score:.4f}, Max Similarity Score: {max_sim_score:.4f}, Total Score: {total_score:.2f}, Avg RAW PREDICT Edit Distance: {raw_pred_edit_dist:.2f}, Common Chars: {common_chars})'
            )
        print(f'{_blank} >  PRECISION : {precision:6.3f}')
        print(f'{_blank} >     RECALL : {recall:6.3f}')
        print(f'{_blank} > F0.5 SCORE : {f_05:6.3f}')
        print('=' * bar_length)

        torch.cuda.empty_cache()

    _now_time = datetime.now().__str__()
    save_file_name = os.path.split(test_file)[-1].replace('.json', '') + '.csv'
    save_file_path = os.path.join(save_path, save_file_name)
    _df = pd.DataFrame({
        'err_sentence': err_sentence_list,
        'raw_prd_sentence': raw_prd_sentence_list,
        'final_prd_sentence': final_prd_sentence_list,
        'cor_sentence': cor_sentence_list,
        'precision': precision_list,
        'recall': recall_list,
        'f_05': f_05_list
    })
    _df.to_csv(save_file_path, index=True)
    print(f'[{_now_time}] - Save Result File(.csv) - {save_file_path}')

    not_precision_1_file_name = os.path.split(test_file)[-1].replace('.json', '_precision_not_1.csv')
    not_precision_1_file_path = os.path.join(save_path, not_precision_1_file_name)
    not_precision_1_df = pd.DataFrame(not_precision_1_list)
    not_precision_1_df.to_csv(not_precision_1_file_path, index=True)
    print(f'[{_now_time}] - Save Precision Not 1 File(.csv) - {not_precision_1_file_path}')

    mean_precision = sum(precision_list) / len(precision_list)
    mean_recall = sum(recall_list) / len(recall_list)
    mean_f_05 = sum(f_05_list) / len(f_05_list)
    print(f'       Evaluation Ngram : {ngram}')
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
        f'DEVICE : {gpu_no}, MODEL PATH : {args.model_path}, FILE PATH : {args.test_file}, DATA LENGTH : {args.eval_length}, SAVE PATH : {save_path}')
    my_train(gpu_no, model_path=args.model_path, test_file=args.test_file, eval_length=args.eval_length,
             save_path=save_path, pb=args.pb)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Evaluation Finished ==========')
