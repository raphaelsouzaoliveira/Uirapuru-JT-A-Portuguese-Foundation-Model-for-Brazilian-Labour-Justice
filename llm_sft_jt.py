import torch
import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    #BitsAndBytesConfig,
    #TrainingArguments,
    PreTrainedTokenizerFast
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import pandas as pd
import os

# --- CONFIGURAÇÃO ---
num_epochs = 2
train_batch_size = 64
lora = True
percentual_congelado = 0.7

#tipo = 'UirapuruJT3.2'
tipo = 'UirapuruJT2'
#tipo = 'Tucano2'
#tipo = 'Llama3.2'

target_modules = []
if tipo == 'Llama3.2':    
    model_name = 'Llama 3.2'    
    model_base_path = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/'
    base_name = "model"
    tokenizer_path = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/tokenizer'
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif tipo == 'Tucano2':
    model_name = 'Tucano 2'    
    model_base_path = '/dados/LLMJT/saida/embeddings/modelo_base/Tucano2'    
    base_name = "model"
    tokenizer_path = '/dados/LLMJT/saida/embeddings/modelo_base/Tucano2/tokenizer'    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif tipo == 'UirapuruJT3.2':    
    #Llama 3.2 finetuning JT
    model_name = 'Uirapuru JT (Llama 3.2)'    
    model_base_path = '/dados/LLMJT/saida/embeddings/modelo/Llama3_2_ptBRJT/'
    base_name = "checkpoint-5667"
    tokenizer_path = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/tokenizer'
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif tipo == 'UirapuruJT2':    
    #Tucano2 finetuning JT
    model_name = 'Uirapuru JT (Tucano 2)'    
    model_base_path = '/dados/LLMJT/saida/embeddings/modelo/Tucano2_ptBRJT/'    
    base_name = "checkpoint-2450"
    tokenizer_path = '/dados/LLMJT/saida/embeddings/modelo_base/Tucano2/tokenizer'    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]    

if tipo == 'UirapuruJT2' or tipo == 'Tucano2':
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
elif tipo == 'UirapuruJT3.2' or tipo == 'Llama3.2':    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.unk_token = "<|finetune_right_pad_id|>"
    tokenizer.bos_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|end_of_text|>"
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

str_lora = ''
if lora and len(target_modules)>0:
    str_lora = '_lora'
output_dir=os.path.join(model_base_path, "implication-adapter_" + base_name.replace('/','-').replace('checkpoint-','') + '_epoch_'+ str(num_epochs) + str_lora)


def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Treináveis: {trainable_params} | Total: {all_params} | %: {100 * trainable_params / all_params:.4f}%")
    
entailment_path = '/dados/LLM/jt/entailment/'

df_implication_train = pd.read_csv(entailment_path + 'train.csv')
df_implication_val = pd.read_csv(entailment_path + 'val.csv')

df_implication = pd.concat([df_implication_train, df_implication_val])
df_implication = df_implication[df_implication['entailment']==1][['sentence1', 'sentence2']]
df_implication

# 2. Formatação do Prompt
# O Llama 3.2 não é um modelo "Instruct" nativamente (a versão base), 
# então criamos um template claro para ele aprender o padrão.
def formatting_prompts_func(examples):
    output_texts = []
    for premissa, implicacao, eos in zip(examples['premissa'], examples['implicacao'], examples['eos']):
        # Estrutura: Instrução + Entrada -> Saída
        text = f"""### Instrução: Dada a frase a seguir, escreva uma implicação jurídica trabalhista direta (consequência). 
        ### Premissa:{premissa}
### Implicação: {implicacao}{eos}"""
        output_texts.append(text)
    return output_texts


data = []

for _, row in df_implication.iterrows():
    data.append({"premissa": row['sentence1'], 
                 "implicacao": row['sentence2'],
                 "eos": tokenizer.eos_token})

dataset = Dataset.from_list(data)

def apply_formatting(batch):
    return {"text": formatting_prompts_func(batch)}

dataset = dataset.map(apply_formatting, batched=True,
                      remove_columns=dataset.column_names)
                      
# 3. Carregamento do Modelo com Quantização (QLoRA)
# Isso permite treinar em GPUs com menos memória (ex: T4, L4, RTX 3060)
#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.float16,
#)

model_path = os.path.join(model_base_path, base_name)
model = LlamaForCausalLM.from_pretrained(model_path,
    #quantization_config=bnb_config,
    device_map="auto")
    
# 1. Identificar o grafo de camadas e embeddings
if hasattr(model, 'model'):
    base_transformer = model.model
elif hasattr(model, 'transformer'):
    base_transformer = model.transformer
else:
    base_transformer = model

if hasattr(base_transformer, 'layers'):
    layers_ref = base_transformer.layers
elif hasattr(base_transformer, 'h'):
    layers_ref = base_transformer.h
else:
    raise AttributeError("Não foi possível determinar as camadas ocultas deste modelo.")

embeddings_ref = getattr(base_transformer, 'embed_tokens', getattr(base_transformer, 'word_embeddings', None))
total_layers = len(layers_ref)
num_layers_to_freeze = int(total_layers * percentual_congelado)

peft_config = None
if lora == True and len(target_modules) > 0:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,        
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules
    )
    
    # Encapsula o modelo com os adaptadores LoRA
    model = get_peft_model(model, peft_config)
    
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
    
else:    
    # Partial Tuning Puro (Sem LoRA - Congela a base de baixo para cima)
    print(f"--> [Partial Tuning] Congelando os pesos base das {num_layers_to_freeze}/{total_layers} primeiras camadas.")
    
    if embeddings_ref is not None:
        for param in embeddings_ref.parameters():
            param.requires_grad = False
            
    for i in range(num_layers_to_freeze):
        for param in layers_ref[i].parameters():
            param.requires_grad = False
    
    model.enable_input_require_grads()
    
count_trainable_parameters(model)

# 5. Configuração do Treinamento
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-4 if lora else 2e-5,
    logging_steps=10,    
    fp16=False,
    bf16=False,
    save_strategy="no",
    #optim="paged_adamw_8bit" # Otimizador eficiente em memória
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    dataset_text_field="text",
    completion_only_loss=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "response_template": "### Implicação:" 
    }
)

# 6. Inicializar o Trainer (SFT)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #peft_config=peft_config,
    #formatting_func=formatting_prompts_func,
    args=training_args
)

print("Iniciando treinamento...", tipo, "lora", lora)
print(f"{datetime.datetime.now()}")
trainer.train()

print("Treinamento concluído. Salvando adaptador...")
print(f"{datetime.datetime.now()}")
trainer.model.save_pretrained(output_dir)