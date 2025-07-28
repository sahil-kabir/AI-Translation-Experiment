import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_linear_schedule_with_warmup
from itertools import product
from sklearn.model_selection import KFold
from peft import LoraConfig, TaskType, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    # Placeholder for data loading logic
    # train = pd.read_csv(...)  
    # Return train data
    pass

def initialize_model_and_tokenizer(model_name, hparams):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config, device_map="auto")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=hparams['r'],
        lora_alpha=hparams['lora_alpha'],
        lora_dropout=hparams['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    return tokenizer, model

def fine_tune_model(model, tokenizer, train_fold, epochs, batch_size, warmup_steps, total_steps, lr, weight_decay):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    start = 0
    train_length = len(train_fold)
    for epoch in range(epochs):
        model.train()
        while start < train_length:
            batch = train_fold.iloc[range(start, min(start + batch_size, train_length))]
            fr_input = tokenizer(list(batch["fr_title"].astype(str)), padding=True, truncation=True, return_tensors="pt")
            en_input = tokenizer(list(batch["en_title"].astype(str)), padding=True, truncation=True, return_tensors="pt")
            outputs = model(input_ids=en_input['input_ids'].to(device), 
                            attention_mask=en_input['attention_mask'].to(device), 
                            labels=fr_input['input_ids'].to(device))
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            start += batch_size

def evaluate_model(model, tokenizer, test_fold, batch_size):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(test_fold), batch_size):
            batch = test_fold.iloc[i:min(i + batch_size, len(test_fold))]
            fr_input = tokenizer(list(batch["fr_title"].astype(str)), padding=True, truncation=True, return_tensors="pt")
            en_input = tokenizer(list(batch["en_title"].astype(str)), padding=True, truncation=True, return_tensors="pt")
            outputs = model(input_ids=en_input['input_ids'].to(device), 
                            attention_mask=en_input['attention_mask'].to(device), 
                            labels=fr_input['input_ids'].to(device))
            losses.append(outputs.loss.item())
    return sum(losses) / len(losses)

def hyperparameter_search(hyperparameters, batch_size, epochs, train_length, k):
    warmup_steps = int(0.1 * (train_length // batch_size))
    total_steps = (train_length // batch_size) * epochs

    train = load_data()  # Load your training data here
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_params = None
    best_score = float('inf')

    for combination in product(*hyperparameters.values()):
        hparams = dict(zip(hyperparameters.keys(), combination))
        losses = []
        for fold_num, (train_index_all, test_index_all) in enumerate(kf.split(train)):
            train_index = train_index_all[:train_length]
            test_index = test_index_all[:200]
            train_fold = train.iloc[train_index]
            test_fold = train.iloc[test_index]

            tokenizer, model = initialize_model_and_tokenizer("Helsinki-NLP/opus-mt-tc-big-en-fr", hparams)
            fine_tune_model(model, tokenizer, train_fold, epochs, batch_size, warmup_steps, total_steps, hparams['learning_rate'], hparams['weight_decay'])
            fold_loss = evaluate_model(model, tokenizer, test_fold, batch_size)
            losses.append(fold_loss)

        hparam_loss = sum(losses) / k
        if hparam_loss < best_score:
            best_score = hparam_loss
            best_params = hparams

    return best_score, best_params

def main():
    hyperparameters = {
        'learning_rate': [1e-4, 3e-5, 5e-6],
        'weight_decay': [0.001],
        'r': [2, 4, 8],
        'lora_alpha': [16, 32],
        'lora_dropout': [0.1]
    }
    
    batch_size = 32
    epochs = 2
    train_length = 10000
    k = 3

    best_score, best_params = hyperparameter_search(hyperparameters, batch_size, epochs, train_length, k)
    print(f"Best average loss: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

if __name__ == "__main__":
    main()
