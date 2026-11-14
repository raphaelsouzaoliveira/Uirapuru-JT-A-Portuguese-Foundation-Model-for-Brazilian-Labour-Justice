# -*- coding: utf-8 -*-
import requests
import zipfile
import io
import re
import shutil
from pathlib import Path
from transformers import (BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, GPT2Tokenizer, RobertaTokenizer, 
                          RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,                           
                          LlamaTokenizer, LlamaForCausalLM, set_seed, LlamaConfig, 
                          TrainerCallback, PreTrainedTokenizerFast,
                          MistralConfig, AutoTokenizer, GptOssConfig
                          )
from peft import PeftModel
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tokenizers  import BertWordPieceTokenizer, ByteLevelBPETokenizer, SentencePieceBPETokenizer
from . import TikTokenTokenizer
from datasets import Dataset
import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import multiprocessing as mpp

from ..utils import ler_dados, salvar_dados, paralelizar, obtem_faixa_paralelismo
from util import compacta_pasta, descompacta_pasta
import numpy as np
import nltk
import math
import torch
from pynvml import *

import torch.distributed as dist
import torch.multiprocessing as mp
import torch 

from numbers import Number
import warnings
from collections import Counter
from functools import reduce
from operator import add
import pickle
from collections import deque
import time
import gc

#import tensorflow as tf

import os
import glob

#from memory_profiler import profile

#os.environ['NCCL_P2P_DISABLE'] = '0'
#os.environ['OMP_NUM_THREADS'] = '16'
#datasets.config.IN_MEMORY_MAX_SIZE = 39_845_888

device = 'cpu'
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(0, torch.cuda.device_count())])
    device = 'cuda'

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def print_gpu_utilization():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.", flush=True)
    except Exception:
        print("NVML Shared Library Not Found", flush=True)


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

print_gpu_utilization()

tokenizer = None
tokenizerOriginal= None
tokenizerNovo = None
special_tokens = None
unk_id = None
cls_id = None
sep_id = None
pad_id = None
model_transformer = None
data_collator = None
windows_size = None

def __inicializa_tokenizador(cfg):
    
    global tokenizer
    global tokenizerOriginal
    global tokenizerNovo
    global special_tokens
    global unk_id
    global cls_id
    global sep_id
    global pad_id
    global windows_size

    cam_modelo = cfg["caminho_embeddings"]["cam_modelo"]
    tipo_embeddings = cfg["tipo_embeddings"]
    cam_tokenizador = Path(cfg["saida"]["cam_modelo"] + '/tokenizer')
    if not cam_tokenizador.exists():
        cam_tokenizador = Path(cam_modelo + '/tokenizer')

    if tipo_embeddings == 'BERTimbau':
        if not windows_size:
            windows_size = 512
        if not tokenizer:
            tokenizer = BertTokenizer.from_pretrained(cam_tokenizador, do_lower_case=False, strip_accents=False)        
        if not tokenizerOriginal:
            tokenizerOriginal = BertTokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'), do_lower_case=False, strip_accents=False)
        if not tokenizerNovo:
            tokenizerNovo = BertWordPieceTokenizer(vocab=str(Path(cam_modelo + '/tokenizer/vocab.txt')), lowercase=True, clean_text=True, handle_chinese_chars=False, strip_accents=False)            
        if not special_tokens:            
            special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.cls_token, tokenizerOriginal.sep_token, tokenizerOriginal.pad_token, tokenizerOriginal.mask_token]
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token) 
        if not cls_id:
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.cls_token)
        if not sep_id:
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.sep_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    elif tipo_embeddings == 'GPT2_ptBR':
        if not windows_size:
            windows_size = 1024
        if not tokenizer:
            tokenizer = GPT2Tokenizer.from_pretrained(cam_tokenizador, do_lower_case=False, strip_accents=False)
        if not tokenizerOriginal:
            tokenizerOriginal = GPT2Tokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'), do_lower_case=False, strip_accents=False)
        if not tokenizerNovo:
            tokenizerNovo = ByteLevelBPETokenizer(vocab=str(Path(cam_modelo + '/tokenizer/vocab.json')), merges=str(Path(cam_modelo + '/tokenizer/merges.txt')), lowercase=True)
        if not special_tokens:
            special_tokens=[tokenizerOriginal.pad_token]
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token) 
        if not cls_id:
            #Nao existe cls token
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)
        if not sep_id:
            #Nao existe sep token
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)
        if not pad_id:
            #Nao existe pad token (TODO)
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)
    elif tipo_embeddings == 'RoBERTa_ptBR':
        if not windows_size:
            windows_size = 512
        if not tokenizer:
            tokenizer = RobertaTokenizer.from_pretrained(cam_tokenizador, do_lower_case=False, strip_accents=False)
        if not tokenizerOriginal:
            tokenizerOriginal = RobertaTokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'), do_lower_case=False, strip_accents=False)
        if not tokenizerNovo:
            tokenizerNovo = ByteLevelBPETokenizer(vocab=str(Path(cam_modelo + '/tokenizer/vocab.json')), merges=str(Path(cam_modelo + '/tokenizer/merges.txt')), lowercase=True)
        if not special_tokens:
            special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.cls_token, tokenizerOriginal.sep_token, tokenizerOriginal.pad_token, tokenizerOriginal.mask_token]
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token) 
        if not cls_id:
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.cls_token)
        if not sep_id:
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.sep_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    elif tipo_embeddings == 'Cabrita' or tipo_embeddings == 'Sabia':
        if not windows_size:
            windows_size = 2048
        if not tokenizer:
            tokenizer = LlamaTokenizer.from_pretrained(cam_tokenizador, do_lower_case=False, strip_accents=False)            
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.unk_token = "<unk>"
            tokenizer.bos_token = "<s>"
            tokenizer.eos_token = "</s>"
        if not tokenizerOriginal:
            tokenizerOriginal = LlamaTokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'), do_lower_case=False, strip_accents=False)
            tokenizerOriginal.add_special_tokens({'pad_token': '[PAD]'})
            tokenizerOriginal.unk_token = "<unk>"
            tokenizerOriginal.bos_token = "<s>"
            tokenizerOriginal.eos_token = "</s>"
        if not tokenizerNovo:            
            vocab_output_path, merges_output_path = __gera_vocab_merges_files(cam_tokenizador, tokenizer)

            tokenizerNovo = SentencePieceBPETokenizer(vocab=vocab_output_path, merges=merges_output_path)
        if not special_tokens:            
            special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.pad_token]            
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)             
        if not cls_id:            
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.bos_token)            
        if not sep_id:            
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.eos_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    elif tipo_embeddings == 'Llama2' or tipo_embeddings == 'Mistral':
        if not windows_size:
            windows_size = 4096
        if not tokenizer:
            #O mistral usa o llamatokenizer
            #tokenizer = LlamaTokenizer.from_pretrained(cam_tokenizador, do_lower_case=False, strip_accents=False)
            tokenizer = LlamaTokenizer.from_pretrained(cam_tokenizador)

            tokenizer.unk_token = "<unk>"
            tokenizer.bos_token = "<s>"
            tokenizer.eos_token = "</s>"
            tokenizer.pad_token = "<pad>"
        if not tokenizerOriginal:            
            #O mistral usa o llamatokenizer
            #tokenizerOriginal = LlamaTokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'), do_lower_case=False, strip_accents=False)
            tokenizerOriginal = LlamaTokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'))

            tokenizerOriginal.unk_token = "<unk>"
            tokenizerOriginal.bos_token = "<s>"
            tokenizerOriginal.eos_token = "</s>"
            tokenizerOriginal.pad_token = "<pad>"
        if not tokenizerNovo:            
            #vocab_output_path, merges_output_path = __gera_vocab_merges_files(cam_tokenizador, tokenizer)

            #tokenizerNovo = SentencePieceBPETokenizer(vocab=vocab_output_path, merges=merges_output_path)
            tokenizerNovo = None
        if not special_tokens:            
            special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.pad_token, tokenizerOriginal.bos_token, tokenizerOriginal.eos_token]            
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)             
        if not cls_id:            
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.bos_token)            
        if not sep_id:            
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.eos_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    elif tipo_embeddings == 'Tucano2':
        if not windows_size:
            windows_size = 4096
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(cam_tokenizador)

            tokenizer.unk_token = "<unk>"
            tokenizer.bos_token = "<s>"
            tokenizer.eos_token = "</s>"
            tokenizer.pad_token = "<pad>"
        if not tokenizerOriginal:            
            tokenizerOriginal = AutoTokenizer.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'))

            tokenizerOriginal.unk_token = "<unk>"
            tokenizerOriginal.bos_token = "<s>"
            tokenizerOriginal.eos_token = "</s>"
            tokenizerOriginal.pad_token = "<pad>"
        if not tokenizerNovo:            
            tokenizerNovo = None
        if not special_tokens:            
            special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.pad_token, tokenizerOriginal.bos_token, tokenizerOriginal.eos_token]            
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)             
        if not cls_id:            
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.bos_token)            
        if not sep_id:            
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.eos_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    elif tipo_embeddings == 'Llama3' or tipo_embeddings == 'Llama3_1' or tipo_embeddings == 'Llama3_2':
        #https://github.com/rasbt/LLMs-from-scratch/blob/27a6a7e64a97a07da2030ed8c291cbb3e1a4bd0a/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb        
        if not windows_size:
            windows_size = 8192
        #if tipo_embeddings == 'Llama3':
        #    if not windows_size:
        #        windows_size = 8192
        #else:
        #    if not windows_size:
        #        windows_size = 131072
        if not tokenizer:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(cam_tokenizador)

            tokenizer.unk_token = "<|finetune_right_pad_id|>"
            tokenizer.bos_token = "<|begin_of_text|>"
            tokenizer.eos_token = "<|end_of_text|>"
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        if not tokenizerOriginal:
            tokenizerOriginal = PreTrainedTokenizerFast.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'))

            tokenizerOriginal.unk_token = "<|finetune_right_pad_id|>"
            tokenizerOriginal.bos_token = "<|begin_of_text|>"
            tokenizerOriginal.eos_token = "<|end_of_text|>"
            tokenizerOriginal.pad_token = "<|finetune_right_pad_id|>"
        if not tokenizerNovo:
            tokenizerNovo = None
        if not special_tokens:            
            special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.pad_token, tokenizerOriginal.bos_token, tokenizerOriginal.eos_token]            
        if not unk_id:
            unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)             
        if not cls_id:            
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.bos_token)            
        if not sep_id:            
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.eos_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    elif tipo_embeddings == 'GptOss':
        if not windows_size:
            windows_size = 8192
        if not tokenizer:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(cam_tokenizador)

            #tokenizer.unk_token = "<|finetune_right_pad_id|>"
            tokenizer.bos_token = "<|startoftext|>"
            tokenizer.eos_token = "<|endoftext|>"
            tokenizer.pad_token = "<|return|>"
        if not tokenizerOriginal:
            tokenizerOriginal = PreTrainedTokenizerFast.from_pretrained(Path(cam_modelo + '/tokenizerOriginal'))

            #tokenizerOriginal.unk_token = "<|finetune_right_pad_id|>"
            tokenizerOriginal.bos_token = "<|startoftext|>"
            tokenizerOriginal.eos_token = "<|endoftext|>"
            tokenizerOriginal.pad_token = "<|return|>"
        if not tokenizerNovo:
            tokenizerNovo = None
        if not special_tokens:            
            #special_tokens=[tokenizerOriginal.unk_token, tokenizerOriginal.pad_token, tokenizerOriginal.bos_token, tokenizerOriginal.eos_token]            
            special_tokens=[tokenizerOriginal.pad_token, tokenizerOriginal.bos_token, tokenizerOriginal.eos_token]
        #if not unk_id:
        #    unk_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.unk_token)             
        if not cls_id:            
            cls_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.bos_token)            
        if not sep_id:            
            sep_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.eos_token)
        if not pad_id:
            pad_id = tokenizerOriginal.convert_tokens_to_ids(tokenizerOriginal.pad_token)
    else:
        raise ValueError("Permitido apenas o tipos de embeddings GPT2_ptBR, BERTimbau, RoBERTa_ptBR, Cabrita, Sabia, Llama2, Llama3, Llama3_1, Llama3_2, Mistral, GptOss, Tucano2")


def __gera_vocab_merges_files(cam_tokenizador, tokenizer):    
    vocab_output_path = Path(os.path.join(cam_tokenizador, 'vocab.json'))
    merges_output_path = Path(os.path.join(cam_tokenizador, 'merges.txt'))
    if not (vocab_output_path.exists() and vocab_output_path.exists()):        
        from tqdm import trange, tqdm
        from json import dump
        from os import linesep
        sp = tokenizer.get_spm_processor()
        #vocab = {sp.id_to_piece(index): index for index in trange(sp.GetPieceSize())}
        #caso exista um unusual line terminator precisa substituir senao nao e possivel carregar o SentencePieceBPETokenizer         
        vocab = {sp.id_to_piece(index).replace('\u2028',''): index for index in trange(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]
        
        with open(vocab_output_path, "w", encoding="utf-8") as vocab_f:
            with open(merges_output_path, "w", encoding="utf-8") as merges_f:                    
                # Save content
                dump(vocab, vocab_f, ensure_ascii=False)
                merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{linesep}", merges))
    
    return str(vocab_output_path), str(merges_output_path)
                
def __treina_tokenizador_from_scratch(cfg):
    tipo_embeddings = cfg["tipo_embeddings"]
    cam_tokenizador = Path(cfg["saida"]["cam_modelo"] + '/tokenizer')
    tokenizador_existe = True
    tokenizador_existe = cam_tokenizador.exists()
    
    if tokenizador_existe:
        print('Tokenizador já treinado')        
    else:
        cam_tokenizador.mkdir(parents=True, exist_ok=True)

        cam_paralelismo = Path(cfg["cam_saida_paralelismo"])
        if not cam_paralelismo.exists():
            cam_paralelismo.mkdir(parents=True, exist_ok=True)
        input_train_tokenizer = cfg["tokenizer"]["tipo_input_train"]
        if input_train_tokenizer == 'text':
            descompacta_pasta(Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados.zip')), cam_paralelismo)
            arquivos = list(cam_paralelismo.glob("*.txt"))            
        else:
            df_dados_freq = ler_dados(Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados_frequencia.parquet')))
            arquivos = [str(Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados_frequencia.tsv')))]
            
            df_dados_freq = df_dados_freq[df_dados_freq['piece'].str.len() > 1]            
            df_dados_freq.to_csv(arquivos[0], sep="\t", header=False, index=False)
            df_dados_freq = None
            del df_dados_freq
        tipo_train_tokenizer = cfg["tokenizer"]["tipo_train"]
        
        tipo_train_fator = len(arquivos)
        if tipo_train_tokenizer != 'total':
            tipo_train_fator = cfg["tokenizer"]["tipo_train_fator"]
        
        if tipo_embeddings == 'Llama2' or tipo_embeddings == 'Mistral':
            model_prefix = Path(cfg["saida"]["cam_modelo"] + '/temporario/')
            if not model_prefix.exists():
                model_prefix.mkdir(parents=True, exist_ok=True)
            
            params = {"model_type": "bpe",
                      "vocab_size": cfg["params_tipo_embeddings"]["geral"]["vocab_size"],          
                      "self_test_sample_size": 0,
                      "input_format": input_train_tokenizer,
                      "character_coverage": 1.0,
                      "num_threads": os.cpu_count(),
                      "split_digits": True,
                      "allow_whitespace_only_pieces": True,
                      "byte_fallback": True,
                      "unk_surface": r" \342\201\207 ",
                      "normalization_rule_name": "identity",
                      "unk_id": 0,
                      "bos_id": 1,
                      "eos_id": 2,
                      "pad_id": 3,                                
                      "unk_piece":"<unk>",
                      "bos_piece": "<s>",
                      "eos_piece": "</s>",
                      "pad_piece": "<pad>",
                      "train_extremely_large_corpus": True,
                      "max_sentence_length": 500000000,
                      "seed_sentencepiece_size": 5000000}
            
            for i in range(0, len(arquivos), tipo_train_fator):
                print(i, len(arquivos[i:i+tipo_train_fator]), flush=True)
                model_prefix_temp = Path(str(model_prefix) + '/' + str(i))
                if not model_prefix_temp.exists():
                    model_prefix_temp.mkdir(parents=True, exist_ok=True)
                SentencePieceTrainer.train(input=arquivos[i:i+tipo_train_fator],
                                           model_prefix=str(model_prefix_temp) + '/tokenizer',
                                           **params
                                           )
                
            previous_model_prefix = ""
            models = list(model_prefix.rglob("*.model"))
            for cam_model in models:
                if str(previous_model_prefix) != "":
                    sp_previous = SentencePieceProcessor()
                    sp_previous.load("{}.model".format(str(previous_model_prefix) + '/tokenizer'))
                    sp_previous_proto = sp_pb2_model.ModelProto()
                    sp_previous_proto.ParseFromString(sp_previous.serialized_model_proto())
                    
                    sp = SentencePieceProcessor()                    
                    sp.load("{}.model".format(str(cam_model).replace('.model', '')))
                    sp_proto = sp_pb2_model.ModelProto()
                    sp_proto.ParseFromString(sp.serialized_model_proto())
                    
                    previous_vocab_tokens = {p.piece for p in sp_previous_proto.pieces}
                    min_score_in_precious_vocab = sp_previous_proto.pieces[-1].score
                    
                    if tipo_train_tokenizer == 'incremental':
                        previous_model_prefix = Path(str(cam_model).replace('tokenizer.model', '') + '/combined' )
                        
                        #new_tokens = []
                        for p in sp_proto.pieces:
                            if p.piece not in previous_vocab_tokens:
                                #new_tokens.append(p.piece)
                                new_piece = sp_pb2_model.ModelProto().SentencePiece()
                                new_piece.piece = p.piece
                                new_piece.score = p.score + min_score_in_precious_vocab  # Lower priority for new tokens
                                sp_previous_proto.pieces.append(new_piece)
                        #print('####################', flush=True)
                        #print(len(new_tokens), flush=True)
                        #print('####################', flush=True)
                                            
                        if not previous_model_prefix.exists():
                            previous_model_prefix.mkdir(parents=True, exist_ok=True)
                        # **Save combined model:**                    
                        with open(str(previous_model_prefix) + '/tokenizer.model', 'wb') as f:
                            f.write(sp_previous_proto.SerializeToString())
                    elif tipo_train_tokenizer == 'media':
                        #TODO
                        previous_model_prefix =  str(cam_model).replace('tokenizer.model', '')
                    else:
                        raise ValueError('Apenas valido para incremental e media')
                else:     
                    previous_model_prefix =  str(cam_model).replace('tokenizer.model', '')

            #O mistral usa o llamatokenizer
            #tokenizer = LlamaTokenizer.from_pretrained(model_prefix, do_lower_case=False, strip_accents=False)
            tokenizer = LlamaTokenizer.from_pretrained(str(previous_model_prefix))
            
            tokenizer.save_pretrained(str(cam_tokenizador))
            tokenizer.save_pretrained(cfg["caminho_embeddings"]["cam_modelo"] + '/tokenizerOriginal')
            tokenizer = None
            
            shutil.rmtree(model_prefix)
            shutil.rmtree(cam_paralelismo)
        elif tipo_embeddings == 'Llama3_2' or tipo_embeddings == 'GptOss':
            if input_train_tokenizer != 'tsv':
                raise ValueError("Para o Llama3.2 ou GptOss é permitido apenas o arquivo de frequência no formato tsv, com as colunas piece e freq")

            df_dados_freq = ler_dados(Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados_frequencia.parquet')))
            df_dados_freq = df_dados_freq[(df_dados_freq.freq>25) & (df_dados_freq.piece.str.len() > 1)]
            #df_dados_freq = df_dados_freq[df_dados_freq.piece.str.len() > 1]
            df_dados_freq = df_dados_freq.sort_values(by=['freq'], ascending=False)            
            data = df_dados_freq[['piece', 'freq']].values
            df_dados_freq = None
            del df_dados_freq
            enc = TikTokenTokenizer.SimpleBytePairEncoding.train(data, vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"], pat_str=TikTokenTokenizer.get_llama3_pattern(), mergeable_ranks=None)
            
            fileSimpleBytePairEncodingBackup = Path(cfg["saida"]["cam_modelo"] + '/SimpleBytePairEncoding/')
            if not fileSimpleBytePairEncodingBackup.exists():
                fileSimpleBytePairEncodingBackup.mkdir(parents=True, exist_ok=True)
            
            with open(str(fileSimpleBytePairEncodingBackup) + '/SimpleBytePairEncoding.pkl', 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(enc, outp, pickle.HIGHEST_PROTOCOL)            
            
            model_prefix = Path(cfg["saida"]["cam_modelo"] + '/temporario/')
            if not model_prefix.exists():
                model_prefix.mkdir(parents=True, exist_ok=True)

            TikTokenTokenizer.convert_tiktoken(enc, tipo_embeddings.lower(), model_prefix)

            tokenizer = AutoTokenizer.from_pretrained(model_prefix) 
            
            tokenizer.save_pretrained(str(cam_tokenizador))
            tokenizer.save_pretrained(cfg["caminho_embeddings"]["cam_modelo"] + '/tokenizerOriginal')
            tokenizer = None
            
            shutil.rmtree(model_prefix)
            
        #elif tipo_embeddings == 'Falcon':
        #    if not tokenizer:
        #        if not cam_tokenizador.exists():            
        #            model_prefix = Path(cfg["cam_saida_paralelismo"] + '/tokenizer/')
        #            if not model_prefix.exists():
        #                model_prefix.mkdir(parents=True, exist_ok=True)      
        #            
        #            SentencePieceTrainer.train(input=str(Path(__file__).parent) + '/dados_apoio/data.txt',                                         
        #                                model_prefix=str(model_prefix) + "/temp",  
        #                                #shuffle_input_sentence=False,
        #                                #train_extremely_large_corpus=True,
        #                                # hyperparameters of tokenizer
        #                                #max_sentence_length=16384,                                                                                
        #                                unk_id=11,
        #                                bos_id=11,
        #                                eos_id=11,
        #                                pad_id=11,                                
        #                                unk_piece="<|endoftext|>",
        #                                bos_piece="<|endoftext|>",
        #                                eos_piece="<|endoftext|>",
        #                                pad_piece='<|endoftext|>',
        #                                model_type="BPE",
        #                                vocab_size=10,
        #                                special_tokens = [">>TITLE<<", ">>ABSTRACT<<", ">>INTRODUCTION<<", ">>SUMMARY<<",
        #                                                  ">>COMMENT<<", ">>ANSWER<<", ">>QUESTION<<", ">>DOMAIN<<",
        #                                                  ">>PREFIX<<", ">>SUFFIX<<", ">>MIDDLE<<", "<|endoftext|>"],
        #                                #split_digits=True,
        #                                #split_by_unicode_script=True,
        #                                #byte_fallback=True,
        #                                #allow_whitespace_only_pieces=True,
        #                                #remove_extra_whitespaces=False,
        #                                #normalization_rule_name="nfkc",
        #                                )
        #            tokenizer = FalconTokenizer(str(model_prefix) + "/temp.model")
        #
        #            shutil.rmtree(model_prefix)
        #        else:
        #            tokenizer = FalconTokenizer.from_pretrained(cam_tokenizador, do_lower_case=False, strip_accents=False)            
        #            tokenizer.add_special_tokens([">>TITLE<<", ">>ABSTRACT<<", ">>INTRODUCTION<<", ">>SUMMARY<<",
        #                                                  ">>COMMENT<<", ">>ANSWER<<", ">>QUESTION<<", ">>DOMAIN<<",
        #                                                  ">>PREFIX<<", ">>SUFFIX<<", ">>MIDDLE<<", "<|endoftext|>"])
        #            tokenizer.unk_token = "<|endoftext|>"
        #            tokenizer.bos_token = "<|endoftext|>"
        #            tokenizer.eos_token = "<|endoftext|>"
        #    if not tokenizerOriginal:
        #        tokenizerOriginal = None
        #    if not tokenizerNovo:
        #        tokenizerNovo = SentencePieceBPETokenizer()
        #        tokenizerNovo.unk_token = "<|endoftext|>"
        #        tokenizerNovo.bos_token = "<|endoftext|>"
        #        tokenizerNovo.eos_token = "<|endoftext|>"
        #        tokenizerNovo.pad_token = "<|endoftext|>"
        #        tokenizerNovo.unk_token_id = 11
        #        tokenizerNovo.bos_token_id = 11
        #        tokenizerNovo.eos_token_id = 11
        #        tokenizerNovo.pad_token_id = 11
        #    if not special_tokens:            
        #        special_tokens=[">>TITLE<<", ">>ABSTRACT<<", ">>INTRODUCTION<<", ">>SUMMARY<<",
        #                        ">>COMMENT<<", ">>ANSWER<<", ">>QUESTION<<", ">>DOMAIN<<",
        #                        ">>PREFIX<<", ">>SUFFIX<<", ">>MIDDLE<<", "<|endoftext|>"]         
        #    if not unk_id:
        #        unk_id = 11
        #    if not cls_id:            
        #        cls_id = 11
        #    if not sep_id:            
        #        sep_id = 11
        #    if not pad_id:
        #        pad_id = 11
        else:
            raise ValueError("Permitido apenas o tipos de embeddings Llama2,  Falcon")
    __inicializa_tokenizador(cfg)

def __inicializa_modelo(cfg):
    
    global model_transformer
    global data_collator

    cam_modelo = cfg["caminho_embeddings"]["cam_modelo"]
    tipo_embeddings = cfg["tipo_embeddings"]
    if tipo_embeddings == 'BERTimbau':
        if not model_transformer:
            model_transformer = BertForMaskedLM.from_pretrained(Path(cam_modelo + '/model'))
            model_transformer.train()
        if not data_collator:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)        
    elif tipo_embeddings == 'GPT2_ptBR':
        if not model_transformer:
            model_transformer = GPT2LMHeadModel.from_pretrained(Path(cam_modelo + '/model'))
            model_transformer.train()
        if not data_collator:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'RoBERTa_ptBR':
        if not model_transformer:
            model_transformer = RobertaForMaskedLM.from_pretrained(Path(cam_modelo + '/model'))
            model_transformer.train()
        if not data_collator:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    elif tipo_embeddings == 'Cabrita' or tipo_embeddings == 'Sabia':
        if not model_transformer:
            model_transformer = LlamaForCausalLM.from_pretrained(Path(cam_modelo + '/model'))
            if tipo_embeddings == 'Cabrita':
                model_transformer = PeftModel.from_pretrained(model_transformer, Path(cam_modelo + '/modelPeft'), is_trainable=True)
            model_transformer.train()
        if not data_collator:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'Llama2' or  tipo_embeddings == 'Tucano2':
        if not model_transformer:
            if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
                model_transformer = AutoModelForCausalLM.from_pretrained(Path(cam_modelo + '/model'), 
                                                                         torch_dtype=torch.bfloat16,
                                                                         attn_implementation="flash_attention_2")
            else:
                # Set seed before initializing model.
                set_seed(0)
                #config = CONFIG_MAPPING["llama"]()
                config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=4096, vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"])


                model_transformer = AutoModelForCausalLM.from_config(config)
                model_transformer.enable_input_require_grads()
                n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
                print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            model_transformer.train()
                        
            #target_modules = "q_proj,v_proj".split(',')
            #modules_to_save = None
            #if modules_to_save is not None:
            #    modules_to_save = modules_to_save.split(',')
            #lora_rank = 8
            #lora_dropout = 0.1
            #lora_alpha = 32.            
            #peft_config = LoraConfig(
            #    task_type=TaskType.CAUSAL_LM,
            #    target_modules=target_modules,
            #    inference_mode=False,
            #    r=lora_rank, lora_alpha=lora_alpha,
            #    lora_dropout=lora_dropout,
            #    modules_to_save=modules_to_save)
            #model_transformer = get_peft_model(model_transformer, peft_config)
            #
            #model_transformer.print_trainable_parameters()
            #old_state_dict = model_transformer.state_dict
            #model_transformer.state_dict = (
            #    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
            #).__get__(model_transformer, type(model_transformer))
        if not data_collator:            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'Llama3':
        if not model_transformer:
            if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
                model_transformer = AutoModelForCausalLM.from_pretrained(Path(cam_modelo + '/model'))
            else:
                # Set seed before initializing model.
                set_seed(0)
                #config = CONFIG_MAPPING["llama"]()
                #config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=8192, vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"])
                config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=8192, 
                                    vocab_size=128256, intermediate_size=14336, rope_theta=500000, num_key_value_heads=8)


                model_transformer = AutoModelForCausalLM.from_config(config)
                model_transformer.enable_input_require_grads()
                n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
                print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            model_transformer.train()                     
        if not data_collator:            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'Llama3_1':
        if not model_transformer:
            if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
                model_transformer = AutoModelForCausalLM.from_pretrained(Path(cam_modelo + '/model'))
            else:
                # Set seed before initializing model.
                set_seed(0)
                #config = CONFIG_MAPPING["llama"]()
                #config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=8192, vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"])
                config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=131072, 
                                    vocab_size=128256, intermediate_size=14336, rope_theta=500000, num_key_value_heads=8,
                                    rope_scaling= {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 
                                                    'original_max_position_embeddings': 8192, 
                                                    'rope_type': 'llama3'
                                                    })


                model_transformer = AutoModelForCausalLM.from_config(config)
                model_transformer.enable_input_require_grads()
                n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
                print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            model_transformer.train()                     
        if not data_collator:            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'Llama3_2':
        if not model_transformer:
            if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
                model_transformer = AutoModelForCausalLM.from_pretrained(Path(cam_modelo + '/model'), 
                                                                         torch_dtype=torch.bfloat16,
                                                                         attn_implementation="flash_attention_2")
            else:
                # Set seed before initializing model.
                set_seed(0)
                #config = CONFIG_MAPPING["llama"]()
                #config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=8192, vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"])
                #1B 
                #  hidden_size = 2048
                #  num_hidden_layers = 16
                #  num_attention_heads = 32
                #3B 
                #  hidden_size = 3072
                #  num_hidden_layers = 28
                #  num_attention_heads = 24
                config = LlamaConfig(rms_norm_eps=1e-05, max_position_embeddings=131072, 
                                    hidden_size=3072, num_hidden_layers=28, num_attention_heads=24, tie_word_embeddings=True,
                                    vocab_size=128256, intermediate_size=8192, rope_theta=500000, num_key_value_heads=8,
                                    rope_scaling= {'factor': 32.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 
                                                    'original_max_position_embeddings': 8192,
                                                    "rope_type": "llama3"},
                                    attn_implementation="flash_attention_2",
                                    torch_dtype=torch.bfloat16)


                model_transformer = AutoModelForCausalLM.from_config(config)
                model_transformer.enable_input_require_grads()
                n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
                print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            model_transformer.train()                     
        if not data_collator:            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'Mistral':
        if not model_transformer:
            if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
                model_transformer = AutoModelForCausalLM.from_pretrained(Path(cam_modelo + '/model'))
            else:
                # Set seed before initializing model.
                set_seed(0)            
                config = MistralConfig(vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"])

                model_transformer = AutoModelForCausalLM.from_config(config)
                model_transformer.enable_input_require_grads()
                n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
                print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            model_transformer.train()                        
        if not data_collator:            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif tipo_embeddings == 'GptOss':
        if not model_transformer:
            if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
                model_transformer = AutoModelForCausalLM.from_pretrained(Path(cam_modelo + '/model'))
            else:
                # Set seed before initializing model.
                set_seed(0)            
                config = GptOssConfig(vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"],
                                    num_hidden_layers=24, num_local_experts=32, bos_token_id=128000,
                                    eos_token_id=128001, pad_token_id=128004)

                model_transformer = AutoModelForCausalLM.from_config(config)
                model_transformer.enable_input_require_grads()
                n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
                print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            model_transformer.train()                        
        if not data_collator:            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    #elif tipo_embeddings == 'Falcon':
    #    if not model_transformer:
    #        # Set seed before initializing model.
    #        set_seed(0)
    #        config = CONFIG_MAPPING["falcon"]()
    #
    #        model_transformer = AutoModelForCausalLM.from_config(config)
    #        n_params = sum({p.data_ptr(): p.numel() for p in model_transformer.parameters()}.values())
    #        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    #
    #        model_transformer.resize_token_embeddings(len(tokenizer))
    #                    
    #        target_modules = "q_proj,v_proj".split(',')
    #        modules_to_save = None
    #        if modules_to_save is not None:
    #            modules_to_save = modules_to_save.split(',')
    #        lora_rank = 8
    #        lora_dropout = 0.1
    #        lora_alpha = 32.            
    #        peft_config = LoraConfig(
    #            task_type=TaskType.CAUSAL_LM,
    #            target_modules=target_modules,
    #            inference_mode=False,
    #            r=lora_rank, lora_alpha=lora_alpha,
    #            lora_dropout=lora_dropout,
    #            modules_to_save=modules_to_save)
    #        model_transformer = get_peft_model(model_transformer, peft_config)
    #
    #        model_transformer.print_trainable_parameters()
    #        old_state_dict = model_transformer.state_dict
    #        model_transformer.state_dict = (
    #            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    #        ).__get__(model_transformer, type(model_transformer))
    #    if not data_collator:            
    #        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        raise ValueError("Permitido apenas o tipos de embeddings GPT2_ptBR, BERTimbau, RoBERTa_ptBR, Cabrita, Sabia, Llama, Llama3, Llama3_1, Llama3_2, GptOss, Falcon, Tucano2")

'''
def __limpa_modelo():
    global tokenizer
    tokenizer = None
    
    global tokenizerOriginal
    tokenizerOriginal= None

    global tokenizerNovo
    tokenizerNovo = None

    global special_tokens
    special_tokens = None
    
    global unk_id
    unk_id = None

    global cls_id
    cls_id = None

    global sep_id
    sep_id = None

    global pad_id
    pad_id = None
    
    global model_transformer
    model_transformer = None

    global data_collator
    data_collator = None
        
    del tokenizer, tokenizerOriginal, tokenizerNovo, special_tokens, unk_id, cls_id, sep_id, pad_id, model_transformer, data_collator
'''

def treina_tokenizador(cfg):
    __carregar_embeddings(cfg)
    
    if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]:
        __inicializa_tokenizador(cfg)
        if cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["fine_tokenizer"]:
            __treina_tokenizador(cfg)
    else:
        __treina_tokenizador_from_scratch(cfg)

def gerar_embeddings(cfg):
    """
    Gera o modelo NLP

    Args:
        cfg (Dict): dicionario com as configuracoes de embeddings.
    """
    cam_modelo = cfg["saida"]["cam_modelo"]

    treina_tokenizador(cfg)
    
    __inicializa_modelo(cfg)

    if not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"] and cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["fine_tokenizer"]:
        print('Initialised random emb for new tokens', len(tokenizer))
        model_transformer.resize_token_embeddings(len(tokenizer))
    print_gpu_utilization()
    
    cam_data = Path(cfg["preparacao_dados"]["cam_dados_modelo"])
    
    #cam_data_tokenizado_final = Path(os.path.join(os.path.join(cam_data, cfg["tipo_embeddings"] + '_' + cfg["tipo_dado"]), "tokenizado_final"))
    #
    #if not cam_data_tokenizado_final.exists():
    #    preparar_dados_treino_modelo_tokenizado(cfg)
    #
    #    cam_data_tokenizado_base = Path(os.path.join(cam_data, cfg["tipo_embeddings"] + '_' + cfg["tipo_dado"] + "/tokenizado"))
    #
    #    lm_tokenized_dataset_train = None
    #    lm_tokenized_dataset_test = None
    #    for pasta in [str(x) for x in cam_data_tokenizado_base.glob("*")]:
    #        lm_tokenized_ds = datasets.load_from_disk(os.path.join(pasta, "train.hf"))        
    #        if lm_tokenized_dataset_train == None:
    #            lm_tokenized_dataset_train = lm_tokenized_ds
    #        else:
    #            lm_tokenized_dataset_train = datasets.concatenate_datasets([lm_tokenized_dataset_train, lm_tokenized_ds])
    #        
    #        lm_tokenized_ds = datasets.load_from_disk(os.path.join(pasta, "test.hf"))        
    #        if lm_tokenized_dataset_test == None:
    #            lm_tokenized_dataset_test = lm_tokenized_ds
    #        else:
    #            lm_tokenized_dataset_test = datasets.concatenate_datasets([lm_tokenized_dataset_test, lm_tokenized_ds])
    #    
    #    lm_tokenized_dataset_train.save_to_disk(os.path.join(cam_data_tokenizado_final, 'train.hf'), max_shard_size="10GB")
    #    lm_tokenized_dataset_test.save_to_disk(os.path.join(cam_data_tokenizado_final, 'test.hf'), max_shard_size="10GB")
    #
    #    shutil.rmtree(cam_data_tokenizado_base, ignore_errors=True)
    #
    #lm_tokenized_dataset_train = datasets.load_from_disk(os.path.join(cam_data_tokenizado_final, 'train.hf'))
    #lm_tokenized_dataset_test = datasets.load_from_disk(os.path.join(cam_data_tokenizado_final, 'test.hf'))
            
    ###definidos no deepspeed_config####
    batch_size_train = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["batch_size_train"]    
    batch_size_eval = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["batch_size_eval"]    
    gradient_accumulation_steps=cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["gradient_accumulation_steps"]
        
    steps = int(pd.read_csv(str(Path(os.path.join(os.path.join(cam_data, cfg["tipo_embeddings"] + '_' + cfg["tipo_dado"] + "/tokenizado"), "quantitativo.csv")))).sum()['train'])
    step_size = (len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) * gradient_accumulation_steps * batch_size_train)
    steps = (steps // step_size)
        
    kwargs = dict()
    kwargs['per_device_train_batch_size'] = batch_size_train
    kwargs['per_device_eval_batch_size'] = batch_size_eval
    kwargs['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    kwargs['eval_strategy'] = cfg["params_tipo_embeddings"]["geral"]["eval"]["strategy"]
    if kwargs['eval_strategy'] == "steps":
        kwargs['eval_steps'] = max(int(steps / cfg["params_tipo_embeddings"]["geral"]["eval"]["per_epochs"]), 1)
    kwargs['num_train_epochs'] = cfg["params_tipo_embeddings"]["geral"]["num_train_epochs"]
    kwargs['save_strategy'] = "steps"
    kwargs['save_steps'] = max(int(steps / cfg["params_tipo_embeddings"]["geral"]["save_per_epochs"]), 1)
    kwargs['save_total_limit'] = cfg["params_tipo_embeddings"]["geral"]["save_total_limit"]
    kwargs['logging_strategy'] = "steps"
    kwargs['logging_steps'] = 1
    kwargs['logging_dir'] = cfg["params_tipo_embeddings"]["geral"]["caminho_log"]
    kwargs['dataloader_num_workers'] = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    #kwargs['load_best_model_at_end'] = True
    kwargs['metric_for_best_model'] = "loss"
    kwargs['greater_is_better'] = False
    kwargs['fp16'] = cfg["params_tipo_embeddings"]["geral"]["fp16"]
    kwargs['bf16'] = cfg["params_tipo_embeddings"]["geral"]["bf16"]
    #kwargs['fp16_opt_level'] = "O3"
    #kwargs['skip_memory_metrics'] = False
    if 'deepspeed_config' in cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]:
        kwargs['deepspeed'] = str(Path(__file__).parent) + '/' + cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["deepspeed_config"]
    kwargs['gradient_checkpointing'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["gradient_checkpointing"]
    kwargs['learning_rate'] = float(cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["learning_rate"])
    kwargs['weight_decay'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["weight_decay"]
    kwargs['adam_beta1'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["adam_beta1"]
    kwargs['adam_beta2'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["adam_beta2"]
    kwargs['adam_epsilon'] = float(cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["adam_epsilon"])
    if 'max_grad_norm' in cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]:
        kwargs['max_grad_norm'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["max_grad_norm"]
    if 'lr_scheduler_type' in cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]:
        kwargs['lr_scheduler_type'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["lr_scheduler_type"]
    if 'lr_scheduler_kwargs' in cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]:
        kwargs['lr_scheduler_kwargs'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["lr_scheduler_kwargs"]    
    if 'warmup_steps' in cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]:
        kwargs['warmup_steps'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["warmup_steps"]
    if 'warmup_ratio' in cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]:
        kwargs['warmup_ratio'] = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["warmup_ratio"]
    kwargs['max_steps'] = 1 + (steps * cfg["params_tipo_embeddings"]["geral"]["num_train_epochs"])
    kwargs['torch_compile'] = True
    
    TRAINING_LOG_LOSS = os.path.join(kwargs['logging_dir'], 'log_loss.npy')
    FILE_FINAL_LOSS_TRAIN = cfg["saida"]["loss"]["cam_treino"]
    FILE_FINAL_LOSS_EVAL = cfg["saida"]["loss"]["cam_avaliacao"]
    
    #Continue train if checkpoint exist
    output_dir_checkpoint = os.path.join(cam_modelo, "checkpoint-")
    checkpoints = glob.glob(output_dir_checkpoint + '*')
    if len(checkpoints) > 0:
        #kwargs['overwrite_output_dir'] = True
        #resume_from_checkpoint = True
        last_checkpoint = str(np.max(np.char.replace(checkpoints, output_dir_checkpoint, '').astype(int)))
        resume_from_checkpoint = output_dir_checkpoint + last_checkpoint
        TRAINING_LOG_LOSS = TRAINING_LOG_LOSS.replace('.npy', last_checkpoint + '.npy')
        FILE_FINAL_LOSS_TRAIN = FILE_FINAL_LOSS_TRAIN.replace('.npz', last_checkpoint + '.npz')
        FILE_FINAL_LOSS_EVAL = FILE_FINAL_LOSS_EVAL.replace('.npz', last_checkpoint + '.npz')
    else:
        resume_from_checkpoint = None

    training_args = TrainingArguments(
        cam_modelo,
        **kwargs
    )

    print('#', device, '-', os.environ['CUDA_VISIBLE_DEVICES'], '-', torch.cuda.device_count(), '#', flush=True)
    #datasets.config.IN_MEMORY_MAX_SIZE = 39_845_888
    torch.cuda.empty_cache()
    model_transformer.config.use_cache = False

    TRAINING_STOP_FILE_NAME = cfg["params_tipo_embeddings"]["geral"]["early_stopping"]["cam_arquivo"]    
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
    
    #class LogCallback(TrainerCallback):
    #    def on_step_end(self, args, state, control, **kwargs):
    #        np.save(os.path.join(args.logging_dir, 'log_loss.npy'), state.log_history)

    ds = datasets.load_dataset(path=str(Path(os.path.join(cam_data, cfg["tipo_embeddings"] + '_' + cfg["tipo_dado"] + "/tokenizado"))),
                           streaming=True,
                           cache_dir=str(Path(os.path.join(cam_data, cfg["tipo_embeddings"] + '_' + cfg["tipo_dado"] + "/cache"))))
    
    kwargs_data = dict()
    kwargs_data['train_dataset'] = ds['train'].shuffle(seed=42, buffer_size=(step_size*cfg["params_tipo_embeddings"]["geral"]["buffer_size_step"]))
    if kwargs['eval_strategy'] != "no" or cfg["params_tipo_embeddings"]["geral"]["eval"]["end"]:
        kwargs_data['eval_dataset'] = ds['test'].shuffle(seed=42, buffer_size=10_000)
    
    trainer = Trainer(
        model=model_transformer,
        args=training_args,
        data_collator=data_collator,
        callbacks=[StopByFileCallback],
        **kwargs_data                
    )

    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print_summary(result)
    
    if cfg["params_tipo_embeddings"]["geral"]["eval"]["end"]:
        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        f=open(FILE_FINAL_LOSS_EVAL,'wb')
        np.savez_compressed(f, loss_avaliacao=eval_results)
        f.close()

    trainer.save_model(cam_modelo)
    tokenizer.save_pretrained(cam_modelo + '/tokenizer')

    f=open(FILE_FINAL_LOSS_TRAIN,'wb')
    np.savez_compressed(f, loss_treino=result)
    f.close()

def __ajuste_texto(texto, alfanumerico=False):
    texto = str(texto).lower()    
    texto = texto.replace('\0','')
    texto = texto.replace('\u200b',' ')
    texto = texto.replace("¯",' ')
    texto = texto.replace("•",' ')
    texto = texto.replace("·",'.')
    texto = texto.replace("…",'.')
    texto = texto.replace("★",' ')
    texto = texto.replace("▁",'_')
    texto = texto.replace("—",'-')
    texto = texto.replace("→",'-')
    texto = texto.replace("《", " ")
    texto = texto.replace("", " ")
       
    texto = texto.replace(" .",'.')
    texto = texto.replace(" ,",',')
    texto = texto.replace(" !",'!')
    texto = texto.replace(" ?",'!')
    texto = texto.replace(" :",':')
    
    texto = re.sub(r"--+", "-", texto)
    texto = re.sub(r"\.\.+", ".", texto)
    texto = re.sub(r"__+", "_", texto)
    texto = re.sub(r"::+", ":", texto)
    texto = re.sub(r"!!+", "!", texto)
    texto = re.sub(r"\?\?+", "?", texto)
    texto = re.sub(r",,+", ",", texto)
    texto = re.sub(r"##+", "#", texto)
    texto = re.sub(r"\$\$+", "$", texto)
    texto = re.sub(r"@@+", "@", texto)
    texto = re.sub(r"\*\*+", "\*", texto)
    texto = re.sub(r"\+\++", "\+", texto)
    texto = re.sub(r"''+", "'", texto)
    texto = re.sub(r'""+', '"', texto)
    texto = re.sub(r"//+", "/", texto)
    texto = re.sub(r"==+", "==", texto)
        
    start_code, stop_code = '2580', '25FF'
    start_idx, stop_idx = [int(code, 16) for code in (start_code, stop_code)]  # from hexadecimal to unicode code point    
    for unicode_idx in range(start_idx, stop_idx+1):
        texto = texto.replace(chr(unicode_idx),' ')

    texto = re.sub(r"(http|https|www|ftp)\S+", " url ", texto)
    texto = re.sub(r"\S+@\S+\.\S+", " email ", texto)

    #if alfanumerico:
    #    texto = re.sub(r"\s\S*\d+\S*", " alfanumérico ", texto)        
    #    texto = re.sub(r"( +alfanumérico)( *alfanumérico *)+", " alfanumérico ", texto)
    
    #retira multiplos espaços em branco
    texto = re.sub(r"  +", " ", texto)

    return texto


def is_valid_math_expression(expression):
    expression = re.sub(r"\s+", "", expression)
    # Expressão regular para validar a estrutura básica da expressão matemática
    pattern_base = '[\{\[(]*[-]?((\d+)([.,]\d+)*)?(a|b|c|x|y|z)?(²)?[\]\})]*([-+/\*\^][\{\[(]*((\d+)([.,]\d+)*)?(a|b|c|x|y|z)?(²)?[\]\})]*)*'
    pattern = r'^' + pattern_base + '(=' + pattern_base + ')?' + '$'
    
    if not re.match(pattern, expression):        
        return False
    
    
    def __isbalanced(expression, char_open, char_close):
        # Verifica se estão balanceados usando uma pilha
        stack = deque()    
        for char in expression:
            if char == char_open:
                stack.append(char)
            elif char == char_close:
                if not stack:                    
                    return False  # Mais fechamentos do que aberturas
                stack.pop()

        if len(stack) != 0:  # balanceados se a pilha estiver vazia no final
            return False
        
        return True
    
    if not __isbalanced(expression, '(', ')'):
        return False
    if not __isbalanced(expression, '[', ']'):
        return False
    if not __isbalanced(expression, '{', '}'):
        return False
    
    return True

def __isNumber(p):
    patternHora = "^(\d|[0-1]\d|[2][0-3])(\:[0-5]\d){1,2}(\.\d+)*(h)?$"
    patternHora2 = "^(\d+h)?(\d+(m|min))?(\d+(s|seg))?$"
    patternHora3 = "^\d+h-\d+(h)?$"
    patternGrau = "^\d+([.,]\d+)*º$"    
    
    p = p.lstrip('Rr$€').strip()
    if (is_valid_math_expression(p) or isinstance(p, Number) or
       re.match(patternHora, p) or re.match(patternHora2, p) or 
       re.match(patternHora3, p) or re.match(patternGrau, p)):        
        return [*p]
    return []
    
def __trata_valor(p):
    p = p.replace('\u200b',' ')
    if len(p) > 1: 
        p = p.strip('‘´“”,.!?-*/#$:"\'\\%|-_=+’;<>«»').strip()        
        p = p.lstrip(')]}').strip()
        p = p.rstrip('([{').strip()        
    
    with warnings.catch_warnings(record=True) as w:
        pieces = __isNumber(p)        
        if len(pieces) == 0:
            pieces = []
            for piece in re.split(r'/|,|\||#|_|;|\(|\)|\[|\]|\{|\}|\?|:', p):
                if len(piece) > 0:
                    ps = __isNumber(piece)
                    if len(ps) > 0:
                        pieces.extend(ps)
                    else:
                        piece = piece.strip('()[]{}"').strip()                                                
                        if re.match(r'^[A-Za-zÀ-ÖØ-öø-ÿ-]+$', piece) and len(piece) <= 30:
                            pieces.append(piece)                        
                
        #if __isNumber(p):
        #    p = p.lstrip('Rr$').strip()
        #    pieces = [*p]
        #else:
        #    pieces = []
        #    for piece in re.split(r'/|,|\||#|_|;|\(|\)|\[|\]|\{|\}', p):
        
        if len(w):
            #print(p, flush=True)
            pieces = []
    return pieces

def preparar_dados_treino_tokenizador(cfg, arquivos):    
    cam_saida_paralelismo = Path(cfg["cam_saida_paralelismo"])
    if not cam_saida_paralelismo.exists():
        cam_saida_paralelismo.mkdir(parents=True, exist_ok=True)
    elif cam_saida_paralelismo.exists():
        shutil.rmtree(cam_saida_paralelismo)
        cam_saida_paralelismo.mkdir(parents=True, exist_ok=True)
    
    def text_split(job, faixa, arquivos, cam_saida_paralelismo, cfg):
        text_data = []
        file_count = 0        
        print(job, len(arquivos[faixa]), flush=True)

        def __salva_arquivo(job, cam_saida_paralelismo, file_count, text_data):
            file_name = os.path.join(cam_saida_paralelismo, str(job) + '_' + str(file_count) + '.txt')

            with open(file_name, mode='w', encoding="utf-8") as f:
                f.write('\n'.join(text_data))

        freq = {}
        for arquivo in arquivos[faixa]:
            dados = ler_dados(arquivo)[[cfg["colunas"]["conteudo"]]]
            print(job, '-', file_count, len(text_data), dados.shape, arquivo, flush=True)
            dados = dados.drop_duplicates(subset=[cfg["colunas"]["conteudo"]], keep="last").reset_index(drop=True)
            print(job, '-', file_count, len(text_data), dados.shape, arquivo, flush=True)
            for _, row in dados.iterrows():
                texto = __ajuste_texto(row[cfg["colunas"]["conteudo"]], False)
                for piece in texto.strip().split():
                    for p in __trata_valor(piece):
                        freq.setdefault(p, 0)
                        freq[p] += 1

                text_data.append(texto)
                if (len(text_data) == 10000):
                    __salva_arquivo(job, cam_saida_paralelismo, file_count, text_data)
                    text_data = []
                    file_count += 1

        if len(text_data) > 0:
            __salva_arquivo(job, cam_saida_paralelismo, file_count, text_data)

        with open(os.path.join(cam_saida_paralelismo, str(job) + '.pkl'), 'wb') as f:
            pickle.dump(freq, f)
        return [job]
    
    resultado = paralelizar(
                    len(arquivos),
                    text_split,
                    params={"arquivos": arquivos,
                            "cam_saida_paralelismo": cam_saida_paralelismo,
                            "cfg": cfg},
                    pasta_temporaria=cam_saida_paralelismo,
                    num_procs=cfg["n_jobs"],
                )
    caminho_frequencia = Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados_frequencia.parquet'))
    if not caminho_frequencia.exists():
        caminho_frequencia.parent.mkdir(parents=True, exist_ok=True)

    freqs = []
    for job in resultado:
        arquivo_frequencia = os.path.join(cam_saida_paralelismo, str(job) + '.pkl')
        print(arquivo_frequencia, flush=True)
        with open(arquivo_frequencia, 'rb') as f:        
            freq = pickle.load(f)
        if len(freqs) == 0:
            freqs = [freq]
        else:
            freqs.append(freq)
            freqs = [reduce(add, (map(Counter, freqs)))]
        os.remove(arquivo_frequencia)

    freq = freqs[0]
    df_freq = pd.DataFrame(data={'piece': freq.keys(), 'freq': freq.values()})
    df_freq['piece'].replace('', np.nan, inplace=True)
    df_freq.dropna(inplace=True)
    salvar_dados(df_freq, caminho_frequencia)

    caminhos_entrada = [str(x) for x in cam_saida_paralelismo.glob("*.txt")]
    print(len(caminhos_entrada))    
    caminho_destino = Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados.zip'))
    if not caminho_destino.exists():
        caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    compacta_pasta(caminhos_entrada, caminho_destino)
    shutil.rmtree(cam_saida_paralelismo)

def preparar_dados_treino_modelo(cfg, arquivos):
    cam_data = Path(cfg["preparacao_dados"]["cam_dados_modelo"])
    
    if not cam_data.exists():
        cam_data.mkdir(parents=True, exist_ok=True)
    #elif cam_data.exists():
    #    shutil.rmtree(cam_data)
    #    cam_data.mkdir(parents=True, exist_ok=True)
    
    def data_split(job, faixa, arquivos, cam_data, cfg):        
        count = 0
        count_file = 0
        file_count = 0
        print(job, len(arquivos[faixa]), flush=True)

        for arquivo in arquivos[faixa]:            
            dados = ler_dados(arquivo)[[cfg["colunas"]["conteudo"]]]
            #print(job, '-', dados.shape, arquivo, flush=True)            
            dados[cfg["colunas"]["conteudo"]] = dados[cfg["colunas"]["conteudo"]].apply(lambda x: __ajuste_texto(x))
            dados = dados.drop_duplicates(subset=[cfg["colunas"]["conteudo"]], keep="last").reset_index(drop=True)
            #print(job, '-', dados.shape, arquivo, flush=True)
            
            if dados.shape[0] > 1:
                dados_treino_temp, dados_teste_temp = train_test_split(dados, test_size=0.1, random_state=42)
                dados = None
                del dados
            else:
                dados_treino_temp = pd.DataFrame(columns=dados.columns)
                dados_teste_temp = dados                
                
            if count_file == 0:
                dados_treino = dados_treino_temp
                dados_teste = dados_teste_temp
            else:
                dados_treino = pd.concat([dados_treino, dados_treino_temp])
                dados_teste = pd.concat([dados_teste, dados_teste_temp])
            dados_treino_temp = None
            dados_teste_temp = None
            del dados_treino_temp, dados_teste_temp
            count = count + 1
            count_file = count_file + 1

            print(job, '-', count, '-', file_count, '-', count_file, dados_treino.shape, dados_teste.shape, arquivo, flush=True)

            if (dados_treino.memory_usage().sum() + dados_teste.memory_usage().sum()) > 1000000:
                salvar_dados(dados_treino, str(cam_data) + '/' + str(job) + '_' + str(file_count) + '_train.parquet')
                salvar_dados(dados_teste, str(cam_data) + '/' + str(job) + '_' + str(file_count) + '_test.parquet')
                dados_treino = None
                dados_teste = None
                del dados_treino, dados_teste

                count_file = 0
                file_count = file_count + 1
        
        if count_file != 0:
            salvar_dados(dados_treino, str(cam_data) + '/' + str(job) + '_' + str(file_count) + '_train.parquet')
            salvar_dados(dados_teste, str(cam_data) + '/' + str(job) + '_' + str(file_count) + '_test.parquet')

        dados_treino = None
        dados_treino_temp = None
        dados_teste = None
        dados_teste_temp = None
        del dados_treino, dados_treino_temp, dados_teste, dados_teste_temp

        return [job]
    
    paralelizar(
        len(arquivos),
        data_split,
        params={"arquivos": arquivos,
                "cam_data": cam_data,
                "cfg": cfg},
        pasta_temporaria=cfg["cam_saida_paralelismo"],
        num_procs=cfg["n_jobs"],
    )
#@profile
def preparar_dados_treino_modelo_tokenizado(cfg):
    cam_data = Path(cfg["preparacao_dados"]["cam_dados_modelo"])
    cam_data_tokenizado_base = os.path.join(cam_data, cfg["tipo_embeddings"] + '_' + cfg["tipo_dado"] + "/tokenizado")
    Path(os.path.join(cam_data_tokenizado_base, 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(cam_data_tokenizado_base, 'test')).mkdir(parents=True, exist_ok=True)

    caminhos_dados_treino = sorted([str(x) for x in Path(cam_data).glob("*_train.parquet")], key=os.path.getctime)
    #caminhos_dados_treino = [str(x) for x in Path(cam_data).glob("1_11_train.parquet")]
    
    #print(caminhos_dados_treino)

    __inicializa_tokenizador(cfg)

    n_procs = cfg["n_jobs"]
    if n_procs == -1:
        n_procs = mpp.cpu_count() - 2
    
    cam_quantitativo = str(Path(os.path.join(cam_data_tokenizado_base, "quantitativo.csv")))
    semaphore_quant_file_name = cam_quantitativo.replace('.csv','_semaphore.txt')
    while Path(semaphore_quant_file_name).exists():
        time.sleep(10)    
    open(semaphore_quant_file_name,'w').close()
    if not Path(cam_quantitativo).exists():
        pd.DataFrame(data=[], columns=['chave', 'train','test']).to_csv(cam_quantitativo, index=False)        
    os.remove(semaphore_quant_file_name)
    
    for caminho in caminhos_dados_treino:
        index = os.path.basename(caminho).replace('_train.parquet', '')

        #if (index != '1_11'):
        #    continue

        semaphore_file_name = Path(os.path.join(cam_data_tokenizado_base, index + '_semaphore.txt'))
        #print(index, pd.read_csv(cam_quantitativo)['chave'].isin([index]).any())
        if not semaphore_file_name.exists() and not pd.read_csv(cam_quantitativo)['chave'].isin([index]).any():
            open(semaphore_file_name,'w').close()
            print(index)
            row = [index]
            cam_data_tokenizado = Path(os.path.join(os.path.join(cam_data_tokenizado_base, 'train'), index + '.parquet'))
            #cam_data_tokenizado.parent.mkdir(parents=True, exist_ok=True)

            for tipo in ['train', 'test']:
                df = ler_dados(caminho.replace('train', tipo))
                print(index, tipo, len(df))
                quant_records = 0
                limite_memoria = 10000000
                limite_itens = 200000
                
                #if df.memory_usage().sum() > limite_memoria:
                faixas = obtem_faixa_paralelismo(1, len(df), int(len(df) / (int(df.memory_usage().sum() / (limite_memoria+1))+1)))
                #faixas = obtem_faixa_paralelismo(1, len(df), limite_itens)
                print(faixas)
                for faixa in faixas:                        
                    df_temp = df[faixa[1]]

                    def processar_salvar_dataset_tokenized(df_temp, limite_itens, n_procs, conteudo_coluna, caminho_tokenizado):
                        dataset= __create_dataset_tokenized(df_temp, limite_itens, n_procs, conteudo_coluna)                                                
                        quant_file = caminho_tokenizado.replace('.parquet', '_quant.txt')
                        with open(quant_file, 'w') as f:
                            f.write(str(dataset.shape[0]))

                        dataset.to_parquet(caminho_tokenizado)
                        dataset = None
                        del dataset

                    if faixa[0] == 0:
                        caminho_tokenizado = str(cam_data_tokenizado).replace('train', tipo)
                    else:
                       caminho_tokenizado = str(cam_data_tokenizado).replace('train', tipo).replace('.parquet','_' + str(faixa[0]) + '.parquet')

                    p = mpp.Process(target=processar_salvar_dataset_tokenized, args=(df_temp, limite_itens, n_procs, cfg["colunas"]["conteudo"], caminho_tokenizado))
                    p.start()
                    p.join()
                    
                    with open(caminho_tokenizado.replace('.parquet', '_quant.txt'), 'r') as f:
                        quant_records_faixa = int(f.read())

                    quant_records = quant_records + quant_records_faixa

                    os.remove(caminho_tokenizado.replace('.parquet', '_quant.txt'))
                    
                    #dataset=__create_dataset_tokenized(df_temp, limite_itens, n_procs, cfg["colunas"]["conteudo"])
                    
                    #quant_records = quant_records + dataset.shape[0]
                    
                    #if faixa[0] == 0:
                    #    dataset.to_parquet(str(cam_data_tokenizado).replace('train', tipo))
                    #else:
                    #    dataset.to_parquet(str(cam_data_tokenizado).replace('train', tipo).replace('.parquet','_' + str(faixa[0]) + '.parquet'))
                    
                    df_temp = None
                    #dataset = None
                    #del df_temp, dataset
                    del df_temp
                row.append(quant_records)
                df = None
                del df
                print(f"Collected {gc.collect()} objects.") 
                #dataset.save_to_disk(os.path.join(cam_data_tokenizado, tipo + '.hf'))
            
            while Path(semaphore_quant_file_name).exists():
                time.sleep(10)    
            open(semaphore_quant_file_name,'w').close()
            df_quantitativo = pd.read_csv(cam_quantitativo)
            df_quantitativo.loc[len(df_quantitativo)] = row
            df_quantitativo.to_csv(cam_quantitativo, index=False)
            os.remove(semaphore_quant_file_name)
            
            for tipo in ['train', 'test']:
                os.remove(caminho.replace('train', tipo))
            os.remove(semaphore_file_name)
            break
            

def __treina_tokenizador(cfg):        
    """
    Adiciona o novo vocabulario ao tokenizador

    Args:
        cfg (Dict): dicionario com as configuracoes de embeddings.
    """
    cam_saida_tokenizador = Path(cfg["saida"]["cam_modelo"] + '/tokenizer')
    if cam_saida_tokenizador.exists():
        print('Tokenizador já treinado')
        return
    
    print_gpu_utilization()
    cam_data = Path(cfg["cam_saida_paralelismo"])
    if not cam_data.exists():
        cam_data.mkdir(parents=True, exist_ok=True)
    descompacta_pasta(Path(os.path.join(cfg["preparacao_dados"]["cam_dados_tokenizador"], 'dados.zip')), cam_data)

    print('Treinar Tokenizador', flush=True)
    vocab = list(tokenizer.get_vocab().keys())
    print(len(vocab), flush=True)
    
    # Customize training
    tokenizerNovo.train(files=[str(x) for x in cam_data.glob("*.txt")],
                        vocab_size=cfg["params_tipo_embeddings"]["geral"]["vocab_size"], min_frequency=2, 
                        show_progress=True,                         
                        special_tokens=special_tokens
                        )

    new_vocab = list(tokenizerNovo.get_vocab().keys())    
    new_vocab = list(set(new_vocab)-set(vocab))
    print(len(new_vocab), new_vocab)

    print('Adding new tokens to vocab')
    n_orig_tokens = len(tokenizer)        
    add = tokenizer.add_tokens(new_vocab)
    print('Original no. of tokens: %s'%n_orig_tokens)
    print(add)
    print('Final no. of tokens: %s'%len(tokenizer))
    print_gpu_utilization()

    tokenizer.save_pretrained(str(cam_saida_tokenizador))
    
    shutil.rmtree(cam_data)

    __inicializa_tokenizador(cfg)

def __carregar_embeddings(cfg):
    cam_modelo = Path(cfg["caminho_embeddings"]["cam_modelo"])
    
    if (not cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["from_scratch"]) and (not cam_modelo.exists()):        
        cam_saida_paralelismo = Path(cfg["cam_saida_paralelismo"])
        url_embeddings = cfg["params_tipo_embeddings"][cfg["tipo_embeddings"]]["url_embeddings"]

        if cam_saida_paralelismo.exists():
            shutil.rmtree(str(cam_saida_paralelismo))
        if not cam_saida_paralelismo.exists():
            cam_saida_paralelismo.mkdir(parents=True, exist_ok=True)
        if not cam_modelo.parent.exists():
            cam_modelo.parent.mkdir(parents=True, exist_ok=True)

        session = requests.Session()

        response = session.get(url_embeddings, params = {'confirm': 1}, stream = True)    

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip:        
            zip.extractall(cam_saida_paralelismo)

            shutil.move(Path(str(cam_saida_paralelismo) + '/' + cfg["tipo_embeddings"]), 
                        cam_modelo)

        shutil.rmtree(str(cam_saida_paralelismo))

def __convert_tokens_to_ids(text, tokenizer, tokenizerOriginal, unk_id, cls_id, sep_id):
    tag_inicio = '__INICIO__ '
    doc_ids = []    
    for sentence in nltk.sent_tokenize(text):
        sentence = sentence.replace(tag_inicio,'')
        #sentence = tag_inicio + sentence        
        sent_ids = [cls_id]
        #for token in nltk.word_tokenize(sentence):
        #    if token != tag_inicio.strip():                
        #        #id = tokenizer.convert_tokens_to_ids(token)
        #        #if (id == unk_id):        
        #        #    ids = [tokenizerOriginal.convert_tokens_to_ids(t) for t in tokenizerOriginal.tokenize(token)]                    
        #        #else:
        #        #    ids = [id]
        #        ids = tokenizer.encode_plus(text=token, add_special_tokens=False, 
        #                            return_attention_mask=False)['input_ids']
        #        if (len(ids) > 0):
        #            sent_ids.extend(ids)
        ids = tokenizer.encode_plus(text=sentence, add_special_tokens=False, 
                                    return_attention_mask=False)['input_ids']
        if (len(ids) > 0):
            sent_ids.extend(ids)
            
        sent_ids.append(sep_id)  
        if(len(sent_ids) > 5):
            doc_ids.append(sent_ids)        
    return doc_ids

def __doc_tokenizer(docs, tokenizer, tokenizerOriginal, unk_id, cls_id, sep_id, pad_id, block_size):    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    special_tokens_mask = []
    #labels = []
    for text in docs['text']:
        text_input_ids = []
        text_token_type_ids = []
        text_attention_mask = []
        text_special_tokens_mask = []
        for text_id in __convert_tokens_to_ids('__INICIO__ ' + text, tokenizer, tokenizerOriginal, unk_id, cls_id, sep_id):
            text_input_ids.append(text_id)
            text_token_type_ids.append(list(np.zeros(len(text_id), dtype=int)))
            text_attention_mask.append(list(np.ones(len(text_id), dtype=int)))
            stm = []
            stm.append(1)
            stm.extend(np.zeros(len(text_id) - 2, dtype=int))
            stm.append(1)
            text_special_tokens_mask.append(stm)
        
        text_result = {'input_ids': text_input_ids, 'token_type_ids': text_token_type_ids, 'attention_mask': text_attention_mask, 'special_tokens_mask': text_special_tokens_mask}
        
        text_result = __group_texts(text_result, pad_id, block_size)
        
        input_ids.extend(text_result["input_ids"])
        token_type_ids.extend(text_result["token_type_ids"])
        attention_mask.extend(text_result["attention_mask"])
        special_tokens_mask.extend(text_result["special_tokens_mask"])
        #labels.extend(text_result["labels"])
    
    #return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'special_tokens_mask': special_tokens_mask, 'labels': labels}
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'special_tokens_mask': special_tokens_mask}
    

def __group_texts(examples, pad_id, block_size):
    back_size = int(block_size*0.1)    
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    size = len(concatenated_examples[list(examples.keys())[0]])    
    slices = []        
    start = 0
    end = min(block_size, size)
    slices.append(slice(start, end))
    while (end < size):
        start = start + block_size - back_size
        end = start + block_size
        if (end > size):
            end = size
            start = end - block_size        

        slices.append(slice(start, end))
    
    def __pad(k, t, pad_id, block_size):
      size_pad = max(block_size - len(t), 0)
      if size_pad > 0:
        if k == 'input_ids': 
          t.extend(np.full(size_pad, pad_id))
        elif k == 'token_type_ids':
          t.extend(np.full(size_pad, 0))
        elif k == 'attention_mask': 
          t.extend(np.full(size_pad, 0))
        elif k == 'special_tokens_mask':
          t.extend(np.full(size_pad, 1))
        else:
          raise ValueError("Experado apenas 'input_ids', 'token_type_ids', 'attention_mask' e 'special_tokens_mask' para k")
      
      return t

    # Split by chunks of max_len.
    result = {
        k: [__pad(k, t[sl], pad_id, block_size) for sl in slices]
        for k, t in concatenated_examples.items()
    }

    #result["labels"] = result["input_ids"].copy()
    return result

def __data_generator(df, campo):
    """
    A generator function that yields data from the DataFrame one by one.
    This is the core of the IterableDataset approach, as it avoids loading
    the entire dataset into memory.
    """
    for _, row in df.iterrows():
        yield {"text": row[campo]}
        
#@profile
def __create_dataset_tokenized(df, limite_itens, jobs, campo):
    
    lm_tokenized_dataset = None
    faixas = obtem_faixa_paralelismo(1, len(df), limite_itens)    

    for _, faixa in faixas:        
        #df_temp = df[faixa]
        #ds = Dataset.from_pandas(pd.DataFrame(df_temp[campo].values, columns=['text']))        
        #df_temp = None
        #del df_temp
        ds = Dataset.from_dict({"text": df.iloc[faixa][campo].values})

        lm_tokenized_ds = ds.map(__doc_tokenizer, fn_kwargs={"tokenizer": tokenizer, 
                                                        "tokenizerOriginal": None, 
                                                        "unk_id": unk_id, 
                                                        "cls_id": cls_id, 
                                                        "sep_id": sep_id,
                                                        "pad_id": pad_id,
                                                        "block_size": windows_size}, 
                            batched=True, num_proc=jobs, 
                            remove_columns=["text"], 
                            keep_in_memory=True,
                           # cache_file_name='/dados/LLM/ptBR_tokOrig/saida/cache_dataset/tokenized_data.arrow'
                            )
        
        if lm_tokenized_dataset == None:
            lm_tokenized_dataset = lm_tokenized_ds
        else:
            lm_tokenized_dataset = datasets.concatenate_datasets([lm_tokenized_dataset, lm_tokenized_ds])
    
        ds = None
        lm_tokenized_ds = None
        del ds, lm_tokenized_ds
        print(f"Collected {gc.collect()} objects.")
    
    return lm_tokenized_dataset