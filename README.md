###### 개인 프로젝트 / *추후 RAG, DPO, Continual Learning에 관한 내용도 추가 예정입니다.*

# CodeMind-Extended
- 본 프로젝트는 goorm CodeMind팀의 [파이널 프로젝트](https://github.com/LimYeri/CodeMind_project)를 확장한 개인 연구입니다. 이 프로젝트에서는 Llama3-8B-Instruct 모델을 Fine-Tuning하기 위해 unsloth 라이브러리를 사용합니다.
- CodeMind Project : 코딩 테스트 문제 해결과 학습을 보조하기 위해 개발된 언어 모델입니다. 이 모델은 LeetCode 유저들이 작성한 포스팅 글을 활용해 Fine-Tuning 되었으며, 코딩 테스트에 특화된 답안을 제시하는 것을 목표로 합니다.

---

## 모델 세부 정보
  - **모델 이름**:
    - [**LimYeri/CodeMind-Llama3.1-8B-unsloth**](https://huggingface.co/LimYeri/CodeMind-Llama3.1-8B-unsloth)
    - [**LimYeri/CodeMind-Llama3-8B-unsloth_v4-one**](https://huggingface.co/LimYeri/CodeMind-Llama3-8B-unsloth_v4-one)
  - **기본 모델**:
    -  meta-llama/Meta-Llama-3.1-8B-Instruct
    -  meta-llama/Meta-Llama-3-8B-Instruct
  - **unsloth 모델**:
    - unsloth/Meta-Llama-3.1-8B-Instruct
    - unsloth/llama-3-8b-Instruct-bnb-4bit
  - **언어**: 영어
  - **라이선스**: MIT


## 주요 기능
  - 문제 유형 및 접근법 설명
  - Python 정답 코드 생성


## 훈련 데이터
  - [LimYeri/LeetCode_Python_Solutions_v2](https://huggingface.co/datasets/LimYeri/LeetCode_Python_Solutions_v2): Leetcode 문제의 Python 솔루션
  - [LimYeri/LeetCode_Python_Solutions_Data](https://huggingface.co/datasets/LimYeri/LeetCode_Python_Solutions_Data): Leetcode 문제의 Python 솔루션
    ###### 데이터 내용은 같습니다.


## 사용된 라이브러리
  - [unsloth](https://github.com/unslothai/unsloth): 자연어 처리 모델의 훈련 및 튜닝을 간편하게 만들기 위한 라이브러리
  - [transformers](https://github.com/huggingface/transformers): 자연어 처리 모델을 위한 라이브러리
  - [datasets](https://github.com/huggingface/datasets): 데이터셋 처리 및 관리 라이브러리
  - [bitsandbytes](https://github.com/TimDettmers/bitsandbytes): 최적화된 연산을 위한 라이브러리
  - [peft](https://github.com/huggingface/peft): 파인 튜닝을 위한 라이브러리
  - [trl](https://github.com/huggingface/trl): 언어 모델 튜닝을 위한 라이브러리
  - [pandas](https://github.com/pandas-dev/pandas): 데이터 조작을 위한 라이브러리


## 파일 구조
  - **dataset/**: 데이터 세트 파일을 포함합니다.
  - **fine-tuning/**: fine tuning 관련 노트북 및 스크립트를 포함합니다.
  - **demo.ipynb**: 데모 노트북으로 모델 사용 예제가 포함되어 있습니다.

---

## 사용 방법
이 모델은 HuggingFace의 모델 허브를 통해 액세스할 수 있으며, API를 사용하여 응용 프로그램에 통합할 수 있습니다. 코딩 문제 또는 프로그래밍 관련 질문을 제공하면 모델이 관련 설명, 코드 스니펫 또는 가이드를 생성합니다.
###### LimYeri/CodeMind-Llama3.1-8B-unsloth

```python
# 자세한 사항은 demo-Llama3.1.ipynb 확인
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from IPython.display import display, Markdown

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "LimYeri/CodeMind-Llama3.1-8B-unsloth", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "system", "content": "You are a kind coding test teacher."},
    {"role": "user", "content": "코딩 문제나 질문을 여기에 입력하세요."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 3000, use_cache = True,
                         temperature = 0.5, min_p = 0.3)
text = (tokenizer.batch_decode(outputs))[0].split('assistant<|end_header_id|>\n\n')[1].strip()
display(Markdown(text))
```

## 훈련 과정

### 모델 및 토크나이저 로드
```python
# 자세한 사항은 fine-tuning/[CodeMind] Llama3.1 8B Conversational + unsloth.ipynb 확인
max_seq_length = 3000 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### LoRA 구성 및 모델 준비
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)
```

### 훈련
```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_steps = 200,
        num_train_epochs = 5,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to="wandb",
        output_dir = "path",
        save_strategy="epoch",
    ),
)
```
###### LimYeri/CodeMind-Llama3-8B-unsloth_v4-one 관한 자세한 사항은 fine-tuning/Llama3 QLoRA unsloth - python solutions-v4 - one.ipynb를 확인해 주세요.

## 제한 사항 및 윤리적 고려사항
- 모델의 출력은 학습 데이터에 기반하므로 항상 정확하지 않을 수 있습니다.
- 중요한 결정이나 실세계 문제 해결에 모델 출력을 사용하기 전에 반드시 검증이 필요합니다.
  
