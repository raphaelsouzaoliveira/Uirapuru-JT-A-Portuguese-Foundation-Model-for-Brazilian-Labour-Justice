import glob, os
from transformers import (PreTrainedTokenizerFast, LlamaForCausalLM,
                         DataCollatorForLanguageModeling,
                         Trainer, TrainingArguments)
import datasets
from pathlib import Path
import pandas as pd

import datetime
import time

from pynvml import *

import torch
import gc

import torch.distributed as dist
import torch.multiprocessing as mp

#import logging

#logging.basicConfig(level=logging.INFO)
#transformers_logger = logging.getLogger("transformers")
#transformers_logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def print_gpu_utilization():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.", flush=True)
    except:
        print("NVML Shared Library Not Found", flush=True)

print_gpu_utilization()

root_dir = '/dados/LLMJT/saida/embeddings/modelo/Llama3_2_ptBRJT/'
root_dir_tokenizer = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/tokenizer'
tokenizer = PreTrainedTokenizerFast.from_pretrained(root_dir_tokenizer)
tokenizer.unk_token = "<|finetune_right_pad_id|>"
tokenizer.bos_token = "<|begin_of_text|>"
tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = "<|finetune_right_pad_id|>"

training_args = TrainingArguments(root_dir + 'Teste_eval',
                                  #fp16=True,
                                  fp16=False,
                                  bf16=True,
                                  do_eval=True,
                                  #per_device_eval_batch_size=6,
                                  per_device_eval_batch_size=3,
                                  dataloader_num_workers=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')),
                                  deepspeed='/embeddings/deepspeed_config/llama3_jt_eval.json'
                                  )
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

ds = datasets.load_dataset(path='/dados/LLMJT/saida/embeddings/dados_consolidados/modelo/Llama3_2_ptBRJT/tokenizado/',
                           streaming=True,
                           cache_dir='/dados/LLMJT/saida/embeddings/dados_consolidados/modelo/Llama3_2_ptBRJT/cache')

#ds = ds['test'].shuffle(seed=42, buffer_size=10_000)
ds = ds['test'].shuffle(buffer_size=12_500)

controle_execucao = root_dir + 'controle_execucao_eval.csv'
if not Path(controle_execucao).exists():
    df_controle_execucao = pd.DataFrame(data=[], columns=['chave', 'execucao', 'eval_loss', 'eval_runtime',
                                                          'eval_samples_per_second', 'eval_steps_per_second'])
    df_controle_execucao.to_csv(controle_execucao, index=False)
df_controle_execucao = pd.read_csv(controle_execucao)

for dir in sorted(glob.glob(root_dir=root_dir, pathname='checkpoint-*'), key=lambda x: os.path.getctime(os.path.join(root_dir, x))): 
#for dir in sorted(glob.glob(root_dir=root_dir, pathname='checkpoint-*'), key=os.path.getctime):
    if Path(root_dir + dir).exists() and not (df_controle_execucao['chave'].eq(dir)).any():
        print(datetime.datetime.now(), dir)
        
        model_transformer = LlamaForCausalLM.from_pretrained(root_dir + dir)
        model_transformer.eval()
        print_gpu_utilization()
        
        trainer = Trainer(
                model=model_transformer,
                eval_dataset=ds.take(12500),
                #eval_dataset=ds.take(50000),
                data_collator=data_collator,
                args=training_args
            )
        
        eval_results = trainer.evaluate()
        
        eval_results['chave'] = dir
        eval_results['execucao'] = datetime.datetime.now()

        df_controle_execucao.loc[len(df_controle_execucao)] = eval_results
        df_controle_execucao.to_csv(controle_execucao, index=False)
        print_gpu_utilization()
        
        trainer.accelerator.free_memory()     
        
        model_transformer = None
        trainer = None
        del model_transformer, trainer
        
        gc.collect()
        print_gpu_utilization()
        torch.cuda.empty_cache()
        print_gpu_utilization()
        break
#    time.sleep(3600)
#    print(f"{datetime.datetime.now()}")