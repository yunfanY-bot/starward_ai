import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

def safe_save_model(model, output_dir):
    # Merge LoRA weights with base model
    model = model.merge_and_unload()
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Remove duplicate tensors
    if 'lm_head.weight' in state_dict and 'model.embed_tokens.weight' in state_dict:
        if torch.equal(state_dict['lm_head.weight'], state_dict['model.embed_tokens.weight']):
            del state_dict['lm_head.weight']
    
    # Save the model
    save_model(model, f"{output_dir}/model.safetensors", state_dict)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

# Load the dataset
import pandas as pd

df = pd.read_parquet("hf://datasets/fschlatt/trump-tweets/data/train-00000-of-00001-4349c5d45dbcd707.parquet")
texts = df['text'].tolist()

# Prepare the dataset
dataset = Dataset.from_dict({"text": texts})

# Load the tokenizer and model
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Prepare for LoRA fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
model = get_peft_model(model, lora_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=1e-4,
    remove_unused_columns=False,  # Add this line
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
safe_save_model(model, "./fine_tuned_gemma")
tokenizer.save_pretrained("./fine_tuned_gemma")