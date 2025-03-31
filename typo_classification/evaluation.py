import argparse
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


class SpellCorrectionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # 문자열 변환 확인
        if sentence is None:
            sentence = ""
        else:
            sentence = str(sentence)  # 문자열로 변환 보장

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def load_model_and_tokenizer(model_path):
    """모델과 토크나이저를 로드합니다."""
    print(f"모델 로드 중: {model_path}")

    # 모델 체크포인트인 경우 (.pt 파일)
    if model_path.endswith('.pt'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"체크포인트 로드 완료 (에폭: {checkpoint['epoch']}, 검증 정확도: {checkpoint['val_accuracy']:.4f})")

    # save_pretrained()로 저장된 모델 디렉토리인 경우
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        print(f"모델 디렉토리에서 로드 완료")

    return model, tokenizer


def predict_case(sentence, model, tokenizer, device):
    """문장에 대한 오류 유형을 예측합니다."""
    model.eval()

    # 문자열 변환 확인
    if sentence is None:
        sentence = ""
    else:
        sentence = str(sentence)

    encoding = tokenizer(
        sentence,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    # 예측된 값을 다시 1, 2, 3으로 매핑 (0->1, 1->2, 2->3)
    return preds.item() + 1


def evaluate_model(model, tokenizer, test_loader, device):
    """모델을 평가하고 성능 지표를 반환합니다."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="평가 중")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 클래스 이름 설정 (원래 레이블로 매핑: 0->1, 1->2, 2->3)
    target_names = ['클래스1', '클래스2', '클래스3']

    # 분류 보고서 생성
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)

    # 혼동 행렬 생성
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 클래스별 정확도 계산
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    class_accuracy = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]

    return {
        'report': report,
        'confusion_matrix': conf_matrix,
        'class_accuracy': class_accuracy,
        'class_total': class_total,
        'overall_accuracy': sum(class_correct) / sum(class_total)
    }


def main():
    parser = argparse.ArgumentParser(description='한국어 맞춤법 오류 분류 모델 평가')
    parser.add_argument('--model_path', type=str, default='./results/best_model.pt',
                        help='모델 체크포인트 또는 모델 디렉토리 경로')
    parser.add_argument('--data_path', type=str, default='./data/datasets/dataset_candidate_all.json',
                        help='평가할 데이터셋 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드로 실행')

    args = parser.parse_args()

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to(device)

    # 대화형 모드
    if args.interactive:
        print("\n===== 대화형 평가 모드 =====")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")

        case_descriptions = {
            1: "맞춤법/띄어쓰기 오류",
            2: "잘못된 어휘 사용",
            3: "문법 오류"
        }

        while True:
            input_text = input("\n문장을 입력하세요: ")

            if input_text.lower() in ['quit', 'exit', '종료']:
                break

            if not input_text.strip():
                continue

            predicted_case = predict_case(input_text, model, tokenizer, device)
            print(f"예측 결과: 케이스 {predicted_case} ({case_descriptions[predicted_case]})")

    # 테스트 데이터셋 평가
    else:
        print("\n===== 테스트 데이터셋 평가 =====")

        # 데이터 로드
        print(f"데이터 로드 중: {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 'err_sentence'와 'case'를 추출하고, case를 0, 1, 2로 매핑 (1->0, 2->1, 3->2)
        err_sentences = [item['annotation']['err_sentence'] for item in data['data']]
        cases = [item['annotation']['case'] - 1 for item in data['data']]

        print(f"전체 데이터 크기: {len(cases)}개 샘플")
        print(f"클래스 분포: {np.bincount(cases)}")

        # 훈련/검증/테스트 데이터 분할 (60% 훈련, 20% 검증, 20% 테스트)
        train_val_sentences, test_sentences, train_val_labels, test_labels = train_test_split(
            err_sentences, cases, test_size=0.2, random_state=42
        )

        # 테스트 데이터셋 생성
        test_dataset = SpellCorrectionDataset(test_sentences, test_labels, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        print(f"테스트 데이터 크기: {len(test_sentences)}개 샘플")

        # 모델 평가
        print("모델 평가 중...")
        results = evaluate_model(model, tokenizer, test_loader, device)

        # 결과 출력
        print("\n===== 평가 결과 =====")
        print(f"전체 정확도: {results['overall_accuracy']:.4f}")
        print("\n클래스별 정확도:")
        for i in range(3):
            print(f"클래스 {i + 1}: {results['class_accuracy'][i]:.4f} ({results['class_total'][i]}개 샘플)")

        print("\n혼동 행렬:")
        print(results['confusion_matrix'])

        print("\n분류 보고서:")
        print(results['report'])

        # 결과 저장
        results_dir = os.path.dirname(args.model_path)
        results_file = os.path.join(results_dir, 'evaluation_results.txt')

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("===== 평가 결과 =====\n")
            f.write(f"전체 정확도: {results['overall_accuracy']:.4f}\n\n")
            f.write("클래스별 정확도:\n")
            for i in range(3):
                f.write(f"클래스 {i + 1}: {results['class_accuracy'][i]:.4f} ({results['class_total'][i]}개 샘플)\n")

            f.write("\n혼동 행렬:\n")
            f.write(str(results['confusion_matrix']) + "\n\n")

            f.write("분류 보고서:\n")
            f.write(results['report'] + "\n")

        print(f"\n결과가 {results_file}에 저장되었습니다.")


if __name__ == '__main__':
    main()