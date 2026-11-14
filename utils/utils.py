# -*- coding: utf-8 -*-
import itertools
import multiprocessing as mp
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict
from os.path import join
import os

import joblib
import numpy as np
import yaml
import pandas as pd
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

import sys
from types import ModuleType, FunctionType
from gc import get_referents

import psutil

from functools import wraps
import errno
import signal
import copy

def carregar_cfg(path: str) -> Dict:
    """
    Carrega o arquivo .yaml com os parametros de configuracao

    Args:
        path (str): caminho ao arquivo

    Returns:
        Dict: Configuracao carregada do .yaml
    """
    with open(path, mode="r", encoding="utf-8") as entrada:
        return yaml.safe_load(entrada)

def carregar_super_cfg(pCFG):
    """
    Carrega o arquivo .yaml com os parametros de configuracao de forma hierarquica. Para isso a chave super de
    conter os yaml(s) pai e na chave content contem o dicionario com as configuracoes

    Args:
        pCFG (Dict): Configuracao carregada do .yaml

    Returns:
        Dict: Configuracao carregada do .yaml
    """
    if "super" in pCFG:        
        cfgs = pCFG["super"]        
        CFGPai = {}
        for cfg in cfgs:
            CFGComplementar = carregar_super_cfg(carregar_cfg("integracao/" + cfg + ".yaml"))
            CFGPai = copy.deepcopy(concatenar_dicionarios(CFGPai, CFGComplementar))
        return concatenar_dicionarios(CFGPai, copy.deepcopy(pCFG["content"]))
    else:
        return copy.deepcopy(pCFG["content"])
        
def concatenar_dicionarios(dict_base, dict_concat):
    """
    Concatena dois dicionarios preservando todas as suas entradas.

    Args:
        dict_base (Dict): dicionario que sera usado como base para a concatenacao.
        dict_concat (Dict): dicionario que sera concatenado ao dicionario base.

    Returns:
        Dict: dicionario concatenado.
    """
    for key, val in dict_base.items():
        if type(val) == dict:
            if key in dict_concat and type(dict_concat[key] == dict):
                concatenar_dicionarios(dict_base[key], dict_concat[key])
        else:
            if key in dict_concat:
                dict_base[key] = dict_concat[key]

    for key, val in dict_concat.items():
        if not key in dict_base:
            dict_base[key] = val

    return dict_base


def verificar_caminhos(caminhos):
    """
    Verifica se todos os caminhos de entradas obrigatorios existem.

    Args:
        caminhos (lis[str]): lista com todos os caminhos de entradas obrigatorios.

    Raises:
        FileNotFoundError: caminho de entrada nao existe.
    """
    for k in caminhos:
        path_k = Path(caminhos[k])
        if not path_k.exists():
            raise FileNotFoundError(f"Caminho de entrada nao existe: {path_k}")


def criar_caminhos(caminhos: Dict):
    """
    Cria as pastas ate os caminhos especificados, e remove cada caminho se ja 
    existir.

    Args:
        caminhos (Dict): Dicionario onde cada item aponta para um caminho no
        sistema de arquivos
    """
    for k in caminhos:
        path_k = Path(caminhos[k])
        if not path_k.parent.exists():
            path_k.parent.mkdir(parents=True, exist_ok=True)
        elif path_k.exists():
            if path_k.is_dir():
                shutil.rmtree(path_k)
            else:
                path_k.unlink()

def obtem_faixa_paralelismo(num_procs, total_items, limite_itens=500, valor_base = 0):
    items_faltantes = 0
    if (total_items <= num_procs):
        items = 1
        num_procs = total_items
        num_slices = num_procs
    else:    
        items = int(total_items / num_procs)
        num_slices = num_procs
        if items > limite_itens:
            num_slices = int(total_items / limite_itens)
        
        items = int(total_items / num_slices)
        items_faltantes = total_items - (items * num_slices)
    #define os itens que serao processados por cada cpu
    #por exemplo, com 4 cpus e 1000 itens para processar
    #teremos as faixas (0, 249); (250, 499); (500, 749); e (750, 999)
    faixas = [(i, slice(valor_base + ((i * items) if ((items_faltantes == 0) or (i == 0)) else (i * items) + i if (i < items_faltantes) else (i * items) + i - (i - items_faltantes + 1) + 1), 
                        valor_base + (min((i + 1) * items if (items_faltantes == 0) else ((i + 1) * items) + i + 1 if (i < items_faltantes) else ((i + 1) * items) + (i + 1) - (i - items_faltantes + 1),
                                      total_items)))) 
              for i in range(num_slices)]

    return faixas

def paralelizar(total_items, funcao, params, pasta_temporaria, num_procs=-1, memoria_objetos=None):
    """
    Paraleliza a execucao de uma determinada funcao na quantidade de nucleos desejada.

    Args:
        total_items (int): total de itens a ser paralelizado.
        funcao: funcao que sera executada pelo paralelismo passando os dados e os parametros desejados.
        params (Dict): dicionario contendo os parametros que serao usados na chamada da funcao desejada. 
        pasta_temporaria (str): caminho da pasta temporaria utilizada pelo paralelismo.
        num_procs (int): quantidade de CPUs usadas no palelismo, caso informe -1 ira utilizar todas os nucleos disponiveis.
        memoria_objetos (dict): dicionario (fixo e distribuido) com o tamanho em Mb dos objetos

    Returns:
        list: contetndo o resultado final do paralelismo.
    """

    if num_procs == -1:
        num_procs = mp.cpu_count() - 2
    
    if memoria_objetos != None:
        memoria_livre = int(psutil.virtual_memory().free/1024/1024)
        tamanho_memoria_fixo = memoria_objetos['fixo']
        tamanho_memoria_distribuido = memoria_objetos['distribuido']
        for i in range(1, num_procs+2):
            tamanho_memoria_proc = tamanho_memoria_fixo + (int(tamanho_memoria_distribuido / i))
            tamanho_memoria_proc = tamanho_memoria_proc * 1.05

            if ((tamanho_memoria_proc * i) > memoria_livre):
                break
        
        if (i > 1):
            num_procs = i - 1
        else:
            raise ValueError("Memória insuficiente. É necessário no mínimo " + str((tamanho_memoria_fixo + tamanho_memoria_proc)) + " Mb livre para executar este procedimento")

    faixas = obtem_faixa_paralelismo(num_procs, total_items, total_items)

    resultados = joblib.Parallel(n_jobs=num_procs, temp_folder=pasta_temporaria)(
        joblib.delayed(funcao)(
            i,
            sl,
            **params,
        )
        for i, sl in faixas
    )
    resultados = list(itertools.chain.from_iterable(resultados))
    
    return resultados


def concatenar_arquivos(cam_entrada, cam_saida, ext="csv"):
    """
    Concatena o conteudo de todos os arquivos de uma pasta e gera um unico arquivo de saida.

    Args:
        cam_entrada (str): caminho para pasta onde estao os arquivos que serao concatenados.
        cam_saida (str): caminho do arquivo de saida que sera gerado.
        ext (str): extensao que sera usada na criacao do arquivo de saida. (padrao eh "csv").
    """
    arquivos = sorted(list(Path(cam_entrada).glob(f"*.{ext}")))
    with open(cam_saida, mode="w", encoding="utf-8") as saida:
        for i_arq, arq in enumerate(arquivos):
            with open(arq, encoding="utf-8") as entrada:
                for i_lin, linha in enumerate(entrada):
                    if i_lin == 0 and i_arq > 0:
                        continue
                    saida.write(linha)


def filtrar_dados(df_dados, colunas):
    """
    Filtra um conjunto de dados e ordena-o.

    Args:
        df_dados (Dataframe): conjunto de dados que sera filtrado e ordenado.
        colunas (Dict): dicionario contendo duas listas, uma com as colunas pra ordenar o conjunto de dados, e outra lista para filtrar apenas as colunas desejadas. 

    Returns:
        Dataframe: conjunto de dados com o resultando da ordenacao e da aplicacao do filtro de colunas.
    """
    df_saida = df_dados.sort_values(by=colunas["ordem"])
    ordem_filtro = {e: i for i, e in enumerate(colunas["filtro"])}
    filtro = set(df_saida.columns).intersection(set(colunas["filtro"]))
    filtro = sorted(filtro, key=lambda x: ordem_filtro[x])
    return df_saida[filtro]


def salvar_dados(df_dados, cam_saida):
    """
    Salva o Dataframe no caminho informado de acordo com a extensao do nome do arquivo.
    Extensoes permitidas .pkl e .zip    

    Args:
        df_dados (Dataframe): conjunto de dados que sera salvo.
        cam_saida (str): caminho do arquivo que sera salvo.

    """
    if Path(cam_saida).suffix == '.pkl':
        df_dados.reset_index(drop=True).to_pickle(cam_saida, compression='zip')
    elif Path(cam_saida).suffix == '.parquet':
        df_dados.to_parquet(cam_saida, engine='pyarrow', index=False)
    elif Path(cam_saida).suffix == '.zip':
        df_dados.to_csv(cam_saida, compression='zip', index=False)
    else:
        raise ValueError(f"Extensao de arquivo nao suportada. Permitido apenas .pkl, .parquet  e .zip.")
    
    #limpa variaveis
    df_dados = None
    del df_dados


def ler_dados(cam_entrada, separador=","):
    """
    Ler o Dataframe no caminho informado de acordo com a extensao do nome do arquivo.

    Args:       
        cam_entrada (str): caminho do arquivo que sera lido.
        separador (str): delimitador usado para separar os no arquivo (padrao eh ",").
    
    Returns:
        Dataframe: conjunto de dados lidos do caminho informado.

    """
    if Path(cam_entrada).suffix == '.pkl':
        return pd.read_pickle(cam_entrada, compression='zip')
    elif Path(cam_entrada).suffix == '.parquet':
        return pd.read_parquet(cam_entrada)
    else:
        if Path(cam_entrada).suffix == '.zip':
            return pd.read_csv(cam_entrada, sep=separador, compression='zip')
        else:
            return pd.read_csv(cam_entrada, sep=separador)


def ajustar_caminhos(cfg, id_modelo):
    """
    Ajusta os caminhos contidos no cfg para incluir o id do modelo.

    Args:       
        cfg (Dict): dicionario que sera iterado para alterar os caminhos.
        id_modelo (int): identificador do modelo.
    
    Returns:
        Dict: dicionario ajustado.

    """
    for chave, valor in cfg.items():
        p = Path(valor)    
        cfg[chave] = join(join(p.parent, str(id_modelo)), p.name)

    return cfg

def ajustar_caminhos_base(cfg, cam_base, p_valor='saida/'):
    """
    Ajusta os caminhos contidos no cfg para incluir o caminho base no inicio do valor quando
    no valor tem p_valor passado por parametro.

    Args:       
        cfg (Dict): dicionario que sera iterado para alterar os caminhos.
        cam_base (str): caminho base a ser colocado no inicio do valor dicionario.
        p_valor (str): texto que deve ser encontrado no valor do dicionario para incluir o caminho base    
    Returns:
        Dict: dicionario ajustado.

    """            
    for chave, valor in cfg.items():
        if type(valor) == dict: 
            cfg[chave] = ajustar_caminhos_base(valor, cam_base, p_valor)
        else:
            if type(valor) == str and valor.startswith(p_valor):
                cfg[chave] = cam_base + valor

    return cfg

def recuperar_termos_tf_idf_mais_relevantes(df_filtrado, col_termo, top_n=2):
    """
    Recupera os top_n termos mais relevantes de um conjunto de dados.

    Args:
        df_filtrado (Dataframe): conjunto de dados, filtrado por grupo, a ser analisado.
        col_termo (str): nome da coluna no conjunto de dados que representa o termo.
        top_n (int): indca a quantidade de termos mais relevantes que serao extraidos do conjunto de dados.

    Returns:
        list: com os termos mais relevantes.
    """
    results = ['Indefinido']
    try:
        tfidf = TfidfVectorizer(stop_words=stopwords.words("portuguese"))

        df = df_filtrado[col_termo]
        # Caso o dataframe nao possua registro retorna o termo 'Inexistente'
        if np.shape(df)[0] == 0:        
            return 'Inexistente'

        responses = tfidf.fit_transform(df)  # df.apply(lambda x: np.str_(x)))
        feature_names = np.array(tfidf.get_feature_names_out())

        #df1 = pd.DataFrame(responses.toarray(), columns=feature_names)
        #print('Média TFIDF:', df1.mean().sort_values(ascending=False)[:top_n].index.values)
        
        tfidf_sorting = np.argsort(responses.toarray()).flatten()[::-1]

        results = np.unique(feature_names[tfidf_sorting][:top_n])
        results = np.char.replace(results.astype(str), '_', ' ')
    except Exception as erro:
        print(erro)

    return results

def calcula_memoria(objeto):
    """
    Calcula o tamanho de uso de memoria do objeto passado em megabytes

    Args:
        objeto: objeto para calculo do consumo de memoria
    """
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(objeto, BLACKLIST):
        raise TypeError('calcula_memoria() não suporta argumento do tipo: '+ str(type(objeto)))
    ids_visto = set()
    tamanho = 0
    objetos = [objeto]
    while objetos:
        need_referents = []
        for obj in objetos:
            if not isinstance(obj, BLACKLIST) and id(obj) not in ids_visto:
                ids_visto.add(id(obj))
                tamanho += sys.getsizeof(obj)
                need_referents.append(obj)
        objetos = get_referents(*need_referents)
    return tamanho / 1000000

class TimeoutError(Exception):
    pass

def timeout(segundos=10, mensagem=os.strerror(errno.ETIME)):
    """
    Permite anotar as funcoes definido o tempo de processamento

    Args:
        segundos (int): tempo de espera para o processamento da funcao (valor padrao 10 segundos)
        mensagem (str): mensagem caso estoure o tempo (valor padrao Timer expired)
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(mensagem)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,segundos) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator    