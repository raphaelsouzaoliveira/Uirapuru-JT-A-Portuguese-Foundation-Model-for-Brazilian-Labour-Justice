#tipo = 'Llama3.1Original'
#tipo = 'Llama3.2Original'
tipo = 'Tucano2'
#tipo = 'Tucano2_ptBRJT'
#tipo = 'Llama3_2_ptBRJT'

frac = 1
top_themes = 12
field = 'conteudo'
rate_data = 1
epochs = 12
lora = False
percentual_congelado = 0.4

target_modules = []
learning_rate=2e-5
model_base_path_tokenizer = ''
if tipo == 'Llama3.1Original' or tipo == 'Llama3.2Original':
    batch_size = 6
    grad_accumulation = 120
    subpasta = 'LLM/ptBR'
    model_folder = 'model'
elif tipo == 'Tucano2':
    batch_size = 12
    grad_accumulation = 60
    subpasta = 'LLM/ptBR'
    model_folder = 'model'
elif tipo == 'Tucano2_ptBRJT':
    batch_size = 12
    grad_accumulation = 60
    subpasta = 'LLMJT'
    model_folder = 'checkpoint-2450'
    model_base_path_tokenizer = '/dados/LLMJT/saida/embeddings/modelo_base/Tucano2/tokenizer'
    #target_modules = ["q_proj", "v_proj"]
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif tipo == 'Llama3_2_ptBRJT':
    batch_size = 6
    grad_accumulation = 120
    subpasta = 'LLMJT'
    model_folder = 'checkpoint-5667'
    model_base_path_tokenizer = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/tokenizer'
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]    

model_path = f'/dados/{subpasta}/saida/embeddings/modelo/{tipo}/'
if (model_base_path_tokenizer == ''):
    model_base_path_tokenizer = model_path + 'tokenizer'

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_EVALUATE_OFFLINE'] = '1'

import pandas as pd
from datasets import load_dataset, DatasetDict, load_from_disk
import torch.distributed as dist
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback)
import evaluate
import numpy as np
from deepspeed.accelerator import get_accelerator
import glob
from peft import get_peft_model, LoraConfig, TaskType
import json

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Treináveis: {trainable_params} | Total: {all_params} | %: {100 * trainable_params / all_params:.4f}%")

path = '/dados/classificacao_assunto/'
train_file = os.path.join(path, f"train_data_{top_themes}_top_themes.parquet")
test_file = os.path.join(path, f"test_data_{top_themes}_top_themes.parquet")

df = pd.read_pickle(path + 'dados_finais_full.pkl', compression='zip')
df_temas = pd.read_csv(path + 'temas.csv')

top_themes_ids = df['cd_tema'].explode().value_counts().head(top_themes)
top_themes_df = top_themes_ids.reset_index()
top_themes_df.columns = ['cd_tema', 'count']
top_themes_list = top_themes_ids.index.tolist()
joined_themes = top_themes_df.merge(df_temas, on='cd_tema', how='inner')

id2class = joined_themes.set_index('cd_tema')['ds_tema'].to_dict()
class2id  = joined_themes.set_index('ds_tema')['cd_tema'].to_dict()

df_temas = None
top_themes_df = None
joined_themes = None
top_themes_ids = None
del df_temas, top_themes_df, joined_themes, top_themes_ids
    
if not (os.path.exists(train_file) and os.path.exists(test_file)):    
    mask = df['cd_tema'].apply(lambda x: any(item in top_themes_list for item in x))
    df_filtered = df[mask]
    mask = None
    df = None
    del mask, df

    df_filtered.loc[:, 'cd_tema_top'] = df_filtered['cd_tema'].apply(lambda x: [item for item in x if item in top_themes_list])
    df_filtered.loc[:, 'stratify_key'] = df_filtered['cd_tema_top'].apply(lambda x: x[0] if len(x) > 0 else None)
    df_filtered.rename(columns={'cd_tema': 'cd_tema_full', 'cd_tema_top': 'cd_tema'}, inplace=True)

    df_train, df_test = train_test_split(df_filtered, test_size=0.1, random_state=42, stratify=df_filtered['stratify_key'])

    df_train = df_train.drop(columns=['stratify_key'])
    df_test = df_test.drop(columns=['stratify_key'])

    df_train.reset_index(drop=True).to_parquet(train_file)
    df_test.reset_index(drop=True).to_parquet(test_file)

    #train_limit = int(len(df_train) * rate_data)
    #test_limit = int(len(df_test) * rate_data)

    #dataset_train = Dataset.from_pandas(df_train.reset_index(drop=True)).to_iterable_dataset()
    #dataset_test = Dataset.from_pandas(df_test.reset_index(drop=True)).to_iterable_dataset()

    #columns = list(dataset_train.features.keys())

    df_filtered = None
    df_train = None
    df_test = None
    del df_filtered, df_train, df_test

tokenizer = AutoTokenizer.from_pretrained(model_base_path_tokenizer)

if (tipo == 'Llama3.1Original' or tipo == 'Llama3.2Original' or tipo == 'Llama3_2_ptBRJT'):
    tokenizer.unk_token = "<|finetune_right_pad_id|>"
    tokenizer.bos_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|end_of_text|>"
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
else:
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
    
def preprocess_function(example):    
    text = example[field]

    classes = list(id2class.keys())
    num_classes = len(classes)

    labels = [0.0] * num_classes

    temas = example['cd_tema']
    if isinstance(temas, list):
        for theme_id in temas:
            try:
                label_index = classes.index(theme_id)
                labels[label_index] = 1.0
            except ValueError:
                print(f"Warning: Theme ID {theme_id} not found in id2class.")
                pass

    tokenized_example = tokenizer(text, 
                                  truncation=True,
                                  max_length=8192
                                  )

    tokenized_example['labels'] = labels

    return tokenized_example
    
processed_data_path = os.path.join(model_path, f"tokenized_dataset_{field}")
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if os.path.exists(processed_data_path):
    print(f"Loading tokenized dataset from {processed_data_path}...")
    tokenized_dataset = load_from_disk(processed_data_path)
else:    
    if local_rank == 0:
        print(local_rank, 'begin tokenized_dataset')
        dataset = load_dataset("parquet", data_files={"train": train_file, "test": test_file})
        cols_to_remove = dataset["train"].column_names
        
        tokenized_dataset = dataset.map(
            preprocess_function,
            remove_columns=cols_to_remove,
            desc="Tokenizing dataset",
            load_from_cache_file=True,
            num_proc=os.cpu_count()
        )
        tokenized_dataset.save_to_disk(processed_data_path)
        print(local_rank, 'end tokenized_dataset')
    
    print(local_rank, 'wait')
    
    if not dist.is_initialized():
         torch.cuda.set_device(local_rank)
         dist.init_process_group(backend="nccl")
    dist.barrier() 
    print(local_rank, 'begin load tokenized_dataset')
    tokenized_dataset = load_from_disk(processed_data_path)
    print(local_rank, 'end load tokenized_dataset')

tokenized_dataset["train"] = tokenized_dataset["train"].shuffle(seed=42)
tokenized_dataset["test"] = tokenized_dataset["test"].shuffle(seed=42)

if rate_data < 1:
    train_limit = int(len(dataset["train"]) * rate_data)
    test_limit = int(len(dataset["test"]) * rate_data)

    tokenized_dataset["train"] = tokenized_dataset["train"].select(range(train_limit))
    tokenized_dataset["test"] = tokenized_dataset["test"].select(range(test_limit))


dataset_train = None
dataset_test = None
del dataset_train, dataset_test



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
           f'{model_path}{model_folder}', 
           num_labels=len(list(id2class.keys())),
           id2label=id2class, 
           label2id=class2id, 
           problem_type = "multi_label_classification",
           attn_implementation="flash_attention_2",
           torch_dtype=torch.bfloat16,
           device_map={"": local_rank})
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# 1. Identificação dinâmica do modelo base (Hugging Face nativo)
if hasattr(model, 'model'):          # Padrão para Llama, Mistral, Tucano
    base_transformer = model.model
elif hasattr(model, 'transformer'):  # Padrão para GPT/BERT-like
    base_transformer = model.transformer
else:
    base_transformer = model

# 2. Identificação das camadas e embeddings
if hasattr(base_transformer, 'layers'):
    layers_ref = base_transformer.layers
elif hasattr(base_transformer, 'h'):
    layers_ref = base_transformer.h
else:
    raise AttributeError("Não foi possível determinar a estrutura de camadas deste modelo.")

embeddings_ref = getattr(base_transformer, 'embed_tokens', getattr(base_transformer, 'word_embeddings', None))
total_layers = len(layers_ref)

if lora == True and len(target_modules) > 0:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,        
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        modules_to_save=["score"]
    )

    model = get_peft_model(model, peft_config)
    
    num_layers_to_freeze = int(total_layers * percentual_congelado)
    print(f"--> [Híbrido] Congelando adaptadores das primeiras {num_layers_to_freeze}/{total_layers} camadas.")
    
    # Varre os parâmetros de forma segura usando os nomes gerados pelo PEFT
    for name, param in model.named_parameters():
        # Foca apenas nos pesos do LoRA aplicados nas camadas internas
        if "lora_" in name:
            for i in range(num_layers_to_freeze):
                # Verifica se o nome do parâmetro pertence à camada 'i' (ex: .layers.0.)
                if f".layers.{i}." in name or f".h.{i}." in name:
                    param.requires_grad = False
                    
    model.print_trainable_parameters()
    
    learning_rate=2e-4
else:    
    # Partial Tuning Puro (Sem LoRA - Congela a base de baixo para cima)
    print("--> [Partial Tuning Mode] Congelando parâmetros estruturais do modelo base...")
    
    # Congelar Embeddings (essencial para não corromper o vocabulário base)
    if embeddings_ref is not None:
        for param in embeddings_ref.parameters():
            param.requires_grad = False
            
    # Congelar as primeiras N camadas
    num_layers_to_freeze = int(total_layers * percentual_congelado)
    for i in range(num_layers_to_freeze):
        for param in layers_ref[i].parameters():
            param.requires_grad = False
            
    print(f"-> [Sucesso] Embeddings e {num_layers_to_freeze}/{total_layers} camadas iniciais congelados.")
    
    # Garante que a cabeça de classificação continue aberta para treino
    for param in model.score.parameters():
        param.requires_grad = True
    
    model.enable_input_require_grads()
# Garante o sumário real no log do cluster
count_trainable_parameters(model)

hamming_metric = evaluate.load("/hamming_metric.py")
aucprc_metric = evaluate.load("/aucprc_metric.py")
aucroc_metric = evaluate.load("/aucroc_metric.py")
accuracy_metric =  evaluate.load("accuracy", "multilabel")
f1_metric =  evaluate.load("f1", "multilabel")
precision_metric =  evaluate.load("precision", "multilabel")
recall_metric =  evaluate.load("recall", "multilabel")
#clf_metrics = evaluate.combine([accuracy, f1, precision, recall, hamming_metric])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # 1. Aplica a Sigmoid para obter probabilidades contínuas (necessário para AUC-PRC)
    probs = sigmoid(logits)
    
    # 2. Binariza as predições usando threshold de 0.5 (necessário para F1, Acc, etc.)
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    
    # 3. Calcula as métricas originais usando 'preds' (binarizado)
    accuracy_results = accuracy_metric.compute(predictions=preds, references=labels)
    
    f1_results_micro = f1_metric.compute(predictions=preds, references=labels, average="micro")
    f1_results_micro['f1_micro'] = f1_results_micro.pop('f1')
    
    precision_results_micro = precision_metric.compute(predictions=preds, references=labels, average="micro")
    precision_results_micro['precision_micro'] = precision_results_micro.pop('precision')
    
    recall_results_micro = recall_metric.compute(predictions=preds, references=labels, average="micro")
    recall_results_micro['recall_micro'] = recall_results_micro.pop('recall')
    
    f1_results_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")
    f1_results_macro['f1_macro'] = f1_results_macro.pop('f1')
    
    precision_results_macro = precision_metric.compute(predictions=preds, references=labels, average="macro")
    precision_results_macro['precision_macro'] = precision_results_macro.pop('precision')
    
    recall_results_macro = recall_metric.compute(predictions=preds, references=labels, average="macro")
    recall_results_macro['recall_macro'] = recall_results_macro.pop('recall')
    
    hamming_results = hamming_metric.compute(predictions=preds, references=labels)
    
    
    auc_prc_micro = aucprc_metric.compute(predictions=probs, references=labels, average="micro")
    auc_prc_micro['auc_prc_micro'] = auc_prc_micro.pop('auc_prc')
    auc_prc_macro = aucprc_metric.compute(predictions=probs, references=labels, average="macro")
    auc_prc_macro['auc_prc_macro'] = auc_prc_macro.pop('auc_prc')
    
    auc_roc_micro = aucroc_metric.compute(predictions=probs, references=labels, average="micro")
    auc_roc_micro['auc_roc_micro'] = auc_roc_micro.pop('auc_roc')
    auc_roc_macro = aucroc_metric.compute(predictions=probs, references=labels, average="macro")
    auc_roc_macro['auc_roc_macro'] = auc_roc_macro.pop('auc_roc')
    
    return {
        **accuracy_results,
        **f1_results_micro, 
        **precision_results_micro, 
        **recall_results_micro, 
        **f1_results_macro,
        **precision_results_macro, 
        **recall_results_macro, 
        **hamming_results,
        **auc_prc_micro,
        **auc_prc_macro,
        **auc_roc_micro,
        **auc_roc_macro
    }

#labels = np.load('labels.npy')
#predictions = np.load('predictions.npy')

#compute_metrics((predictions, labels))

#import logging

# Set the logging level for the elastic agent to WARNING
#logging.getLogger("torch.distributed.elastic").setLevel(logging.WARNING)
#logging.getLogger("transformers").setLevel(logging.INFO)

#num_gpus = int(os.environ.get("WORLD_SIZE", 1))
#steps_per_epoch = int(train_limit / (batch_size * grad_accumulation * num_gpus))
#total_steps = steps_per_epoch * epochs
#print(num_gpus, steps_per_epoch, total_steps)

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
else:    
    num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    num_gpus = max(1, num_gpus) # Garante pelo menos 1 para evitar divisão por zero

steps = int(len(tokenized_dataset["train"]))
step_size = (num_gpus * grad_accumulation * batch_size)
steps = (steps // step_size)
print(f"Total GPUs: {num_gpus} | Step size global: {step_size} | Eval/Save steps: {steps}")

str_lora = ''
if lora and len(target_modules)>0:
    str_lora = '_lora'
    
path_model_multilabel = f"{model_path}multilabel_model_{field}_{model_folder}{str_lora}"

    
training_args = TrainingArguments(
   output_dir=path_model_multilabel,
   learning_rate=learning_rate,
   per_device_train_batch_size=batch_size,
   per_device_eval_batch_size=batch_size,
   gradient_accumulation_steps=grad_accumulation,     
   gradient_checkpointing=True,
   bf16=True,                      
   deepspeed="/embeddings/deepspeed_config/jt_multlabel.json",
   ddp_find_unused_parameters=False,  
   num_train_epochs=epochs,
   weight_decay=0.01,
   eval_strategy="steps",       
   eval_steps=steps,               
   save_strategy="steps",       
   save_steps=steps,
   logging_strategy="steps",
   logging_steps= 1, 
   load_best_model_at_end=False, 
   #metric_for_best_model="f1_macro", 
   metric_for_best_model="auc_prc_macro", 
   greater_is_better=True,
   save_total_limit=2,
   dataloader_pin_memory=False,
   #dataloader_num_workers=0
   dataloader_num_workers=num_gpus, 
   #dataloader_prefetch_factor=2,
)

TRAINING_LOG_LOSS = f"{path_model_multilabel}/log_loss.npy"
TRAINING_STOP_FILE_NAME = f"{path_model_multilabel}/stop-training"


#Continue train if checkpoint exist
output_dir_checkpoint = os.path.join(f"{path_model_multilabel}", "checkpoint-")
checkpoints = glob.glob(output_dir_checkpoint + '*')
print(checkpoints)
if len(checkpoints) > 0:
    #kwargs['overwrite_output_dir'] = True
    #resume_from_checkpoint = True
    last_checkpoint = str(np.max(np.char.replace(checkpoints, output_dir_checkpoint, '').astype(int)))
    resume_from_checkpoint = output_dir_checkpoint + last_checkpoint
    TRAINING_LOG_LOSS = TRAINING_LOG_LOSS.replace('.npy', last_checkpoint + '.npy')
    #FILE_FINAL_LOSS_TRAIN = FILE_FINAL_LOSS_TRAIN.replace('.npz', last_checkpoint + '.npz')
    #FILE_FINAL_LOSS_EVAL = FILE_FINAL_LOSS_EVAL.replace('.npz', last_checkpoint + '.npz')
else:
    resume_from_checkpoint = None
    
class StopByFileCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def on_step_end(self, args, state, control, **kwargs):
        np.save(TRAINING_LOG_LOSS, state.log_history)
        if os.path.exists(TRAINING_STOP_FILE_NAME):
            control.should_save = True
            control.should_log = True
            #control.should_evaluate = True
            control.should_training_stop = True
            print(f'"{TRAINING_STOP_FILE_NAME}" encotrado, o treinamento sera finalizado')
    def on_epoch_end(self, args, state, control, **kwargs):
        """Executado ao final de cada época para limpar o cache da GPU"""
        print(f"\nLimpando cache da GPU ao fim da época {state.epoch}...")
        get_accelerator().empty_cache()
        
    def on_train_end(self, args, state, control, **kwargs):
        if os.path.exists(TRAINING_STOP_FILE_NAME):
            try:
                os.remove(TRAINING_STOP_FILE_NAME)
            except FileNotFoundError:
                print('Arquivo ja foi removido')

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   processing_class=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
   callbacks=[EarlyStoppingCallback(early_stopping_patience=5), StopByFileCallback]
)
    
result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
eval_results = trainer.evaluate()


# 1. Salvar o histórico completo de logs (Treino e Validação) em JSON
log_history = trainer.state.log_history
caminho_history_json = f'{path_model_multilabel}/trainer_log_history.json'
with open(caminho_history_json, 'w') as f:
    json.dump(log_history, f, indent=4)

# 2. Isolar os logs de validação em um CSV para análise rápida do Early Stop
try:
    df_logs = pd.DataFrame(log_history)
    # O Hugging Face intercala logs de treino e validação. Filtramos apenas os de avaliação:
    if 'eval_loss' in df_logs.columns:
        df_evals = df_logs.dropna(subset=['eval_loss']).copy()
        caminho_history_csv = f'{path_model_multilabel}/historico_validacao.csv'
        df_evals.to_csv(caminho_history_csv, index=False)
        print(f"Histórico de validação salvo em {caminho_history_csv}")
except Exception as e:
    print(f"Erro ao gerar CSV de validação: {e}")

# 3. Salvar os resultados globais do treinamento (TrainOutput) e avaliação final
caminho_train_metrics = f'{path_model_multilabel}/metricas_finais_treino.json'
with open(caminho_train_metrics, 'w') as f:
    json.dump(result.metrics, f, indent=4)

caminho_eval_metrics = f'{path_model_multilabel}/metricas_finais_validacao.json'
with open(caminho_eval_metrics, 'w') as f:
    json.dump(eval_results, f, indent=4)
    
#f=open(f'{path_model_multilabel}/loss_treino.npz','wb')
#np.savez_compressed(f, loss_treino=result)
#f.close()

#f=open(f'{path_model_multilabel}/loss_validacao.npz','wb')
#np.savez_compressed(f, loss_avaliacao=eval_results)
#f.close()