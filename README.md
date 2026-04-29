# Qwen3-8B LogicKor SFT Reproduction Guide

이 저장소는 Qwen3-8B 기반 LogicKor SFT 모델을 재현/평가하기 위한 최소 코드 패키지입니다.
대용량 모델 가중치와 학습 데이터는 GitHub에 포함하지 않고 별도 Google Drive 링크로 공유하는 것을 전제로 합니다.
- 가중치 및 학습데이터 링크: https://drive.google.com/drive/folders/15bH5YiUSgpqKj9z8SV4SePUcTA8xylIw?usp=sharing

## 1. 포함/제외 범위

### 포함

```text
configs/          학습 config
train/            LoRA SFT 학습 코드
logickor_eval/    LogicKor 생성, Judge 평가, 점수 집계 코드
requirements/     학습/평가 가상환경 핵심 패키지 목록
scripts/          학습부터 scoring까지 실행 예시 스크립트
```

### 제외

```text
data/             학습 데이터 배치 위치, Git 제외
models/           공유받은 모델/adapter/merged 모델 배치 위치, Git 제외
runs/             학습 산출물, Git 제외
generated/        LogicKor 생성 결과, Git 제외
evaluated/        OpenAI Judge 평가 결과, Git 제외
results/          최종 score/report, Git 제외
```

## 2. Google Drive에서 받을 파일

공유자는 아래 파일/디렉토리를 Google Drive로 전달합니다.

```text
학습 데이터:
  data/logickor_sft_high_converted.jsonl

학습 완료 모델 예시:
  models/qwen3_8b_sft_high/adapter/
  models/qwen3_8b_sft_high/merged/
```

평가만 수행하려면 `models/qwen3_8b_sft_high/merged/`가 있으면 됩니다.
재학습까지 수행하려면 `data/logickor_sft_high_converted.jsonl`이 필요합니다.

## 3. 가상환경

이 프로젝트는 학습과 추론/평가 환경을 분리해서 사용했습니다.

| 환경 | 용도 | requirements |
|---|---|---|
| `etri` | LoRA SFT 학습 | `requirements/etri-training.txt` |
| `etri-infer` | vLLM 생성, OpenAI Judge, scoring | `requirements/etri-infer.txt` |

### 학습 환경 설치 예시

```bash
conda create -n etri python=3.12 -y
conda activate etri
pip install -r requirements/etri-training.txt

# CUDA 12.8 환경에서 사용한 PyTorch 예시
pip install --upgrade --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
```

### 평가 환경 설치 예시

```bash
conda create -n etri-infer python=3.12 -y
conda activate etri-infer
pip install -r requirements/etri-infer.txt
```

> 실제 CUDA/드라이버/vLLM 조합에 따라 PyTorch와 vLLM 설치 버전은 조정이 필요할 수 있습니다.

## 4. 디렉토리 배치

Google Drive에서 받은 파일을 아래처럼 배치합니다.

```text
qwen3_8b_logickor_sft/
├── data/
│   └── logickor_sft_high_converted.jsonl
└── models/
    └── qwen3_8b_sft_high/
        ├── adapter/
        └── merged/
```

## 5. 학습 방법

학습은 `etri` 환경에서 실행합니다.

```bash
conda activate etri

bash scripts/train.sh \
  configs/train_qwen3_8b_sft.yaml \
  runs/qwen3_8b_sft_high
```

동일 명령을 직접 쓰면 다음과 같습니다.

```bash
python train/train_lora.py \
  --config configs/train_qwen3_8b_sft.yaml \
  --output-dir runs/qwen3_8b_sft_high \
  --seed 42
```

학습이 완료되면 기본적으로 다음 산출물이 생성됩니다.

```text
runs/qwen3_8b_sft_high/adapter/       LoRA adapter
runs/qwen3_8b_sft_high/merged/        base + adapter 병합 모델
runs/qwen3_8b_sft_high/run_meta.json  학습 메타데이터
```

## 6. LogicKor 생성

평가용 생성은 `etri-infer` 환경에서 실행합니다.

```bash
conda activate etri-infer

bash scripts/generate.sh models/qwen3_8b_sft_high/merged
```

직접 실행 예시는 다음과 같습니다.

```bash
python logickor_eval/generator.py \
  --model models/qwen3_8b_sft_high/merged \
  --gpu_devices 0 \
  --model_len 4096
```

`generator.py`는 `logickor_eval/questions.jsonl`을 읽고, 현재 작업 디렉토리 기준 `generated/<model-path>/` 아래에 결과를 저장합니다.

## 7. OpenAI Judge 평가

OpenAI API key는 코드나 shell script에 직접 쓰지 말고 환경변수로만 주입합니다.

```bash
conda activate etri-infer
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

bash scripts/evaluate.sh generated/models/qwen3_8b_sft_high/merged
```

직접 실행 예시는 다음과 같습니다.

```bash
python logickor_eval/evaluator.py \
  -o generated/models/qwen3_8b_sft_high/merged \
  -k "$OPENAI_API_KEY" \
  -j gpt-4.1 \
  -t 30
```

평가 결과는 현재 작업 디렉토리 기준 `evaluated/<model-output-relative-path>/`에 저장됩니다.

## 8. Scoring

```bash
conda activate etri-infer

bash scripts/score.sh 'evaluated/models/qwen3_8b_sft_high/merged/*.jsonl'
```

직접 실행 예시는 다음과 같습니다.

```bash
python logickor_eval/score.py \
  -p 'evaluated/models/qwen3_8b_sft_high/merged/*.jsonl'
```

출력은 category별 single-turn/multi-turn 점수와 전체 평균 점수입니다.

## 9. 재현 시 주의사항

1. API key, HF token 등 민감정보는 절대 GitHub에 commit하지 않습니다.
2. 모델 가중치, 학습 데이터, 생성/평가 결과는 `.gitignore`에 의해 제외됩니다.
3. `configs/train_qwen3_8b_sft.yaml`의 `cuda_visible_devices`는 실행 장비에 맞게 수정하세요.
4. merged model은 디스크를 크게 사용합니다. LoRA adapter만 공유하거나 보관할 경우 `train/merge_adapter.py`로 필요 시 병합할 수 있습니다.
5. 평가 결과는 OpenAI Judge 모델/버전, thread 수, API 상태에 따라 약간 달라질 수 있습니다.

## 10. 참고

- LogicKor 원본 저장소: https://github.com/instructkr/LogicKor
- 본 패키지의 `logickor_eval/`은 재현 편의를 위한 최소 평가 코드입니다.
