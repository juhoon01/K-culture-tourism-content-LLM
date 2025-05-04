# Cell 1: 필수 라이브러리 임포트
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Cell 2: 데이터셋 로드 및 전처리 함수 정의
def load_and_preprocess(path: str, tokenizer, max_length: int = 2048):
    raw_ds = load_dataset("json", data_files=path, split="train")
    
    def preprocess(example):
        prompt = (
            f"### 지시문:\n{example['instruction']}\n\n"
            f"### 입력:\n{example['input']}\n\n"
            "### 응답:\n"
        )
        full = prompt + example["output"] + tokenizer.eos_token
        tokens = tokenizer(full, truncation=True, max_length=max_length)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = raw_ds.map(
        preprocess,
        remove_columns=raw_ds.column_names,
        batched=False
    )
    return tokenized

# Cell 3: 모델 및 토크나이저 로드 + Quantization 설정
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
model = prepare_model_for_kbit_training(model)

# Cell 4: LoRA 구성 및 적용
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# Cell 5: 데이터셋 준비
dataset_path = "QA_dataset.jsonl"
tokenized_ds = load_and_preprocess(dataset_path, tokenizer)

# Cell 6: Trainer 설정
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Cell 7: 파인튜닝 시작
trainer.train()

# Cell 8: 모델 및 토크나이저 저장
model.save_pretrained("./output/llama3.2-3B-instruct-lora")
tokenizer.save_pretrained("./output/llama3.2-3B-instruct-lora")
