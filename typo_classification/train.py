import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random

# 결과 저장 경로 설정
save_path = './results'
os.makedirs(save_path, exist_ok=True)

# 텐서보드 로깅 설정
log_dir = os.path.join(save_path, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


# 재현성을 위한 시드 설정
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[{datetime.now()}] 시드가 {seed}로 설정되었습니다.")


seed_everything(42)

print(f"[{datetime.now()}] ====== 데이터 로드 시작 ======")
# 1. 데이터 로드 및 전처리
with open('./data/datasets/dataset_candidate_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 'err_sentence'와 'case'를 추출하고, case를 0, 1, 2로 매핑 (1->0, 2->1, 3->2)
err_sentences = [item['annotation']['err_sentence'] for item in data['data']]
cases = [item['annotation']['case'] - 1 for item in data['data']]

print(f"전체 데이터 크기: {len(cases)}개 샘플")
print(f"클래스 분포: {np.bincount(cases)}")

# 2. 훈련/검증 데이터 분할 (80% 훈련, 20% 검증)
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    err_sentences, cases, test_size=0.2, random_state=42
)

print(f"훈련 데이터 크기: {len(train_sentences)}개 샘플")
print(f"검증 데이터 크기: {len(val_sentences)}개 샘플")
print(f"[{datetime.now()}] ====== 데이터 로드 완료 ======")

print(f"[{datetime.now()}] ====== 모델 로드 시작 ======")
# 3. BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
print(f"토크나이저 로드 완료: bert-base-multilingual-cased")


# 4. 커스텀 데이터셋 클래스 정의
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

        # 추가: None이나 다른 타입이 아닌지 확인하고 문자열로 변환
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


# 5. 데이터셋 및 데이터로더 생성
print(f"[{datetime.now()}] ====== 데이터셋 및 데이터로더 생성 시작 ======")
train_dataset = SpellCorrectionDataset(train_sentences, train_labels, tokenizer)
val_dataset = SpellCorrectionDataset(val_sentences, val_labels, tokenizer)


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
print(f"배치 크기: {batch_size}")
print(f"훈련 배치 수: {len(train_loader)}")
print(f"검증 배치 수: {len(val_loader)}")
print(f"[{datetime.now()}] ====== 데이터셋 및 데이터로더 생성 완료 ======")

# 6. BERT 모델 로드 (분류 클래스 수: 3)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
print(f"모델 로드 완료: bert-base-multilingual-cased (분류 클래스 수: 3)")

# 모델 매개변수 수 계산 (모델 크기 확인)
model_size = sum(p.numel() for p in model.parameters())
print(f"모델 크기: {model_size / 1e6:.2f}M 파라미터")
print(f"[{datetime.now()}] ====== 모델 로드 완료 ======")

# 7. 손실 함수와 옵티마이저 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"학습 장치: {device}")
model.to(device)

criterion = CrossEntropyLoss()
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
print(f"옵티마이저: AdamW (learning_rate: {learning_rate})")


# 훈련 및 평가 함수 정의
def train_epoch(model, data_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # tqdm을 사용하여 진행 상황 표시
    progress_bar = tqdm(data_loader, desc=f"에폭 {epoch + 1} 훈련", leave=False)

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 모델 출력
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 손실 누적
        total_loss += loss.item()

        # 정확도 계산
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 진행 상황 업데이트
        progress_bar.set_postfix({
            'loss': f"{total_loss / (step + 1):.4f}",
            'acc': f"{correct / total:.4f}"
        })

        # 100 배치마다 중간 로깅
        if (step + 1) % 100 == 0:
            print(f"  배치 {step + 1}/{len(data_loader)}, 손실: {total_loss / (step + 1):.4f}, 정확도: {correct / total:.4f}")

    # 평균 손실 및 정확도 계산
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="평가", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 예측값과 실제값 저장
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 진행 상황 업데이트
            progress_bar.set_postfix({
                'loss': f"{total_loss / len(data_loader):.4f}",
                'acc': f"{correct / total:.4f}"
            })

    # 클래스별 정확도 계산
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    class_accuracy = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy, class_accuracy, class_total


# 8. 훈련 루프
print(f"[{datetime.now()}] ====== 훈련 시작 ======")
num_epochs = 3
best_val_accuracy = 0
best_model_path = os.path.join(save_path, 'best_model.pt')

for epoch in range(num_epochs):
    print(f"[{datetime.now()}] === 에폭 {epoch + 1}/{num_epochs} 시작 ===")

    # 훈련
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

    # 훈련 메트릭 로깅
    print(f"에폭 {epoch + 1}/{num_epochs}, 훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.4f}")
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)

    # 검증
    val_loss, val_acc, class_acc, class_total = evaluate(model, val_loader, criterion, device)

    # 검증 메트릭 로깅
    print(f"에폭 {epoch + 1}/{num_epochs}, 검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}")
    print(
        f"클래스별 정확도: 클래스1: {class_acc[0]:.4f} ({class_total[0]}개), 클래스2: {class_acc[1]:.4f} ({class_total[1]}개), 클래스3: {class_acc[2]:.4f} ({class_total[2]}개)")
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/validation', val_acc, epoch)

    # 클래스별 정확도 로깅
    for i in range(3):
        writer.add_scalar(f'Accuracy/class_{i + 1}', class_acc[i], epoch)

    # 최고 성능 모델 저장
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, best_model_path)
        print(f"새로운 최고 성능 모델 저장됨: 정확도 {val_acc:.4f}")

    print(f"[{datetime.now()}] === 에폭 {epoch + 1}/{num_epochs} 완료 ===")

print(f"[{datetime.now()}] ====== 훈련 완료 ======")
print(f"최고 검증 정확도: {best_val_accuracy:.4f}")

# 9. 모델 저장
final_model_path = os.path.join(save_path, 'spell_correction_model')
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"최종 모델 저장 경로: {final_model_path}")


# 10. 예측 함수 정의
def predict(sentence, model, tokenizer, device):
    model.eval()
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

    # 예측된 값을 다시 1, 2, 3으로 매핑
    return preds.item() + 1


# 예시 예측
example_sentences = [
    "노란 잔수함",
    "대한밍국의 수도는 서울입니다",
    "저는 학교에서 공부를 합니닷"
]

print("\n===== 예시 문장 예측 결과 =====")
for sentence in example_sentences:
    predicted_case = predict(sentence, model, tokenizer, device)
    print(f'"{sentence}" 예측 결과: 케이스 {predicted_case}')

# 텐서보드 닫기
writer.close()
print(f"[{datetime.now()}] ====== 스크립트 실행 완료 ======")
