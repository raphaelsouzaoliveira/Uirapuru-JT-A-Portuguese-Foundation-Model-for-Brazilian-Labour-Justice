## **Imports**
import pandas as pd
import xml.etree.ElementTree as ET
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, LlamaForCausalLM, TrainerCallback, PreTrainedTokenizerFast
import os
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from pathlib import Path
import numpy as np
import shutil
from peft import LoraConfig, get_peft_model, TaskType


def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Treináveis: {trainable_params} | Total: {all_params} | %: {100 * trainable_params / all_params:.4f}%")


#################################
######### **Variables** #########
entailment_path = '/dados/LLM/jt/entailment/'
train_batch_size = 48
num_epochs = 4
lora = True
percentual_congelado = 0.7

#name_model = 'uirapuru2'
#name_model = 'uirapuru3_2'
#name_model = 'llama2_original'
#name_model = 'sabia2'
#name_model = 'sabia'
#name_model = 'llama3_1_original'
#name_model = 'llama3_2_original'
#name_model = 'tucano2'
name_model = 'uirapuru2_jt'
#name_model = 'uirapuru3_2_jt'

base_name = 'model'
target_modules = []
if name_model == 'uirapuru2':
    ##### UIRAPURU LLAMA2 #####    
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Llama2_ptBR/'
    #base_name = "checkpoint-4690" #BEST CHECKPOINT
    base_name = "checkpoint-10184"
    model_base_path_tokenizer = model_base_path + 'tokenizer'
elif name_model == 'uirapuru3_2':
    ##### UIRAPURU LLAMA3.2 #####    
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Llama3_2_ptBR/'
    #base_name = "best-checkpoint/checkpoint-4356" #BEST CHECKPOINT
    base_name = "checkpoint-16605"
    model_base_path_tokenizer = model_base_path + 'tokenizer'
elif name_model == 'llama2_original':
    ##### LLAMA2 ORIGINAL #####
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Llama2Original/'    
    model_base_path_tokenizer = model_base_path + 'tokenizer'
elif name_model == 'llama3_1_original':
    ##### LLAMA3.1 ORIGINAL #####
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Llama3.1Original/'
    model_base_path_tokenizer = model_base_path + 'tokenizer'
elif name_model == 'llama3_2_original':
    ##### LLAMA3.2 ORIGINAL #####
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Llama3.2Original/'
    model_base_path_tokenizer = model_base_path + 'tokenizer'
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif name_model == 'sabia2':
    ##### SABIA2 #####
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Sabia2/'
    model_base_path_tokenizer = model_base_path + 'tokenizer'
elif name_model == 'sabia':
    ##### SABIA #####
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Sabia/'
    model_base_path_tokenizer = model_base_path + 'tokenizer'
elif name_model == 'tucano2':
    ##### TUCANO 2 #####
    model_base_path = '/dados/LLM/ptBR/saida/embeddings/modelo/Tucano2/'
    model_base_path_tokenizer = model_base_path + 'tokenizer'
    target_modules = ["q_proj", "v_proj"]
elif name_model == 'uirapuru2_jt':
    ##### UIRAPURU TUCANO 2 #####    
    model_base_path = '/dados/LLMJT/saida/embeddings/modelo/Tucano2_ptBRJT/'    
    base_name = "checkpoint-2450"
    model_base_path_tokenizer = '/dados/LLMJT/saida/embeddings/modelo_base/Tucano2/tokenizer'
    #target_modules = ["q_proj", "v_proj"]
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif name_model == 'uirapuru3_2_jt':
    ##### UIRAPURU LLAMA3.2 #####    
    model_base_path = '/dados/LLMJT/saida/embeddings/modelo/Llama3_2_ptBRJT/'
    base_name = "checkpoint-5667"
    model_base_path_tokenizer = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/tokenizer'
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
else:
    raise ValueError('Name model not supported.')

model_validation = os.path.join(model_base_path, "rte_jt_" +"model_validation_" + name_model + '_' + base_name.replace('/','-'))
str_lora = ''
if lora and len(target_modules)>0:
    str_lora = '_lora'
output_dir=os.path.join(model_base_path, "rte_jt_" + name_model + '_' + base_name.replace('/','-').replace('checkpoint-','') + '_epoch_'+ str(num_epochs) + str_lora)
#################################

if not Path(output_dir).exists():
    Path(output_dir).mkdir()

## **Methods**

# **Dataset**

## **Load**
df_train = pd.read_csv(entailment_path + 'train.csv')

df_dev = pd.read_csv(entailment_path + 'val.csv')

df_test = pd.read_csv(entailment_path + 'test.csv')

## **Pre-processing**
# Lowercase each element
df_train['sentence1'] = df_train['sentence1'].apply(lambda x: x.lower())
df_dev['sentence1'] = df_dev['sentence1'].apply(lambda x: x.lower())
df_test['sentence1'] = df_test['sentence1'].apply(lambda x: x.lower())

df_train['sentence2'] = df_train['sentence2'].apply(lambda x: x.lower())
df_dev['sentence2'] = df_dev['sentence2'].apply(lambda x: x.lower())
df_test['sentence2'] = df_test['sentence2'].apply(lambda x: x.lower())

df_train['label'] = df_train['entailment']
df_dev['label'] = df_dev['entailment']
df_test['label'] = df_test['entailment']

print("Show train_data shape: {}".format(df_train.shape))
print("Show dev_data shape: {}".format(df_dev.shape))
print("Show test_data shape: {}".format(df_test.shape))

ds = DatasetDict()

ds['train'] = Dataset.from_pandas(df_train[['sentence1','sentence2', 'label']])
ds['validation'] = Dataset.from_pandas(df_dev[['sentence1','sentence2', 'label']])
ds['test'] = Dataset.from_pandas(df_test[['sentence1','sentence2', 'label']])

# **Load Model**
if not Path(model_validation).exists():
    if name_model == 'uirapuru2' or name_model == 'uirapuru2_jt':
        tokenizer = AutoTokenizer.from_pretrained(model_base_path_tokenizer)
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"
    elif name_model == 'uirapuru3_2' or name_model == 'uirapuru3_2_jt':    
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_base_path_tokenizer)
        tokenizer.unk_token = "<|finetune_right_pad_id|>"
        tokenizer.bos_token = "<|begin_of_text|>"
        tokenizer.eos_token = "<|end_of_text|>"
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_base_path_tokenizer)
        if tokenizer.unk_token is None:
            tokenizer.unk_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
        
        #raise ValueError('Name model not supported.')

    model_path = os.path.join(model_base_path, base_name)
    model = LlamaForCausalLM.from_pretrained(model_path)
    count_trainable_parameters(model)
    
    tokenizer.save_pretrained(model_validation)
    model.save_pretrained(model_validation)

model = SentenceTransformer(model_validation)
shutil.rmtree(model_validation)

if model._first_module().tokenizer.pad_token_id is None or model._first_module().tokenizer.pad_token_id < 0:
    print('Config pad_token_id')
    model._first_module().tokenizer.pad_token_id = model._first_module().tokenizer.eos_token_id

# Identificação dinâmica do grafo de camadas e embeddings
base_transformer = model._first_module().auto_model
if hasattr(base_transformer, 'layers'):
    layers_ref = base_transformer.layers
    embeddings_ref = getattr(base_transformer, 'embed_tokens', None)
elif hasattr(base_transformer, 'model') and hasattr(base_transformer.model, 'layers'):
    layers_ref = base_transformer.model.layers
    embeddings_ref = getattr(base_transformer.model, 'embed_tokens', None)
else:
    raise AttributeError("Não foi possível determinar a estrutura de camadas/embeddings deste modelo.")

total_layers = len(layers_ref)

if lora == True and len(target_modules) > 0:    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,        
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules
    )
    
    model._first_module().auto_model = get_peft_model(model._first_module().auto_model, peft_config)
    
    num_layers_to_freeze = int(total_layers * percentual_congelado)
    camadas_congeladas = list(range(num_layers_to_freeze))
    camadas_finais = list(range(num_layers_to_freeze, total_layers))
    
    print(f"--> [Híbrido] Congelando adaptadores das primeiras {num_layers_to_freeze}/{total_layers} camadas: {camadas_congeladas}")
    print(f"--> [Híbrido] Adaptadores ativos apenas nas camadas finais: {camadas_finais}")
    
    for i in camadas_congeladas:
        for param in layers_ref[i].parameters():
                param.requires_grad = False
    
    model._first_module().auto_model.print_trainable_parameters()

else:    
    # Partial Tuning Puro (Freeze Bottom Layers)
    print("--> [Partial Tuning Mode] Congelando parâmetros ...")
    
    # 1. Congelar Embeddings estruturais se existirem
    if embeddings_ref is not None:
        for param in embeddings_ref.parameters():
            param.requires_grad = False
            
    # 2. Congelar as primeiras N camadas
    num_layers_to_freeze = int(total_layers * percentual_congelado)
    for i in range(num_layers_to_freeze):
        for param in layers_ref[i].parameters():
            param.requires_grad = False
            
    print(f"-> [Sucesso] Embeddings e {num_layers_to_freeze}/{total_layers} camadas iniciais congelados.")

# Garante o sumário real no log do cluster
count_trainable_parameters(model)
dev_evaluator = BinaryClassificationEvaluator(
    sentences1=ds['validation']["sentence1"],
    sentences2=ds['validation']["sentence2"],
    labels=ds['validation']["label"],
    name="rte_jt-dev",
)
eval_results = model.evaluate(dev_evaluator)

f=open(os.path.join(output_dir, "result_eval.npz"),'wb')
np.savez_compressed(f, result=eval_results)
f.close()

test_evaluator = BinaryClassificationEvaluator(
    sentences1=ds['test']["sentence1"],
    sentences2=ds['test']["sentence2"],
    labels=ds['test']["label"],
    name="rte_jt-test",
)
test_results = model.evaluate(test_evaluator)

f=open(os.path.join(output_dir, "result_test.npz"),'wb')
np.savez_compressed(f, result=test_results)
f.close()

# **FineTuning**

# 3. Define our training loss
#https://sbert.net/docs/package_reference/sentence_transformer/losses.html#softmaxloss
train_loss = losses.SoftmaxLoss(model=model, 
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=2)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    
    learning_rate=1e-5 if not lora else 5e-5, 
    weight_decay=0.01,                        
    lr_scheduler_type="cosine",               
    
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=1,
    run_name="rte_jt",  # Will be used in W&B if `wandb` is installed
)


TRAINING_LOG_LOSS = os.path.join(output_dir, "loss.npy")
TRAINING_STOP_FILE_NAME = os.path.join(output_dir, "stop-training")

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
        
        def on_train_end(self, args, state, control, **kwargs):
            if os.path.exists(TRAINING_STOP_FILE_NAME):
                try:
                    os.remove(TRAINING_STOP_FILE_NAME)
                except FileNotFoundError:
                    print('Arquivo ja foi removido')

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    loss=train_loss,
    evaluator=dev_evaluator,
    callbacks=[StopByFileCallback]
)
result = trainer.train()

f=open(os.path.join(output_dir, "result_finetunig.npz"),'wb')
np.savez_compressed(f, result=result)
f.close()

eval_results = trainer.evaluate()

f=open(os.path.join(output_dir, "result_finetunig_eval.npz"),'wb')
np.savez_compressed(f, result=eval_results)
f.close()

test_results = test_evaluator(model)

f=open(os.path.join(output_dir, "result_finetunig_test.npz"),'wb')
np.savez_compressed(f, result=test_results)
f.close()