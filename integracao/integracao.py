# coding: utf-8

import copy
import datetime

from ..utils import ajustar_caminhos, concatenar_dicionarios

def treino_tokenizador(p_cfg):
    """
    Realiza a fase de treino do tokenizador conforme configuracoes.

    Args:
        p_cfg (Dict): dicionario com as configuracoes para execucao do pipeline de treinamento de embeddings.
    """
    inicio = datetime.datetime.now()
    
    cfg = copy.deepcopy(p_cfg)
                                                    
    try:
        # ajustar caminhos pelo modelo
        #embeddings
        cfg["comum"]["configuracao"]["embeddings"]["caminho_embeddings"]["cam_modelo"] = cfg["comum"]["configuracao"]["embeddings"]["caminho_embeddings"]["cam_modelo"] + '/' + cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"]
        cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["cam_modelo"] = cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["cam_modelo"] + '/' + cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"] + '_' + cfg["comum"]["configuracao"]["geral"]["tipo_dado"]
        cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["loss"] = ajustar_caminhos(cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["loss"], 
                                                                                                   cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"] + '_' + cfg["comum"]["configuracao"]["geral"]["tipo_dado"])
        
        if cfg["treino_embeddings"]["embeddings"]:
            print(f"{datetime.datetime.now()} - Treino Tokenizador")
            
            from ..embeddings import treina_tokenizador
            
            treina_tokenizador(
                concatenar_dicionarios(
                    concatenar_dicionarios(
                        cfg["treino_embeddings"]["configuracao"]["embeddings"],
                        cfg["comum"]["configuracao"]["embeddings"]),
                    cfg["comum"]["configuracao"]["geral"]))
            
    except Exception as erro:
        print(
            f"{datetime.datetime.now()} - Treino Tokenizador - Erro: {erro}")            
        raise

    print(f"{inicio} - Treino Tokenizador - Inicio")
    print(f"{datetime.datetime.now()} - Treino Tokenizador - Fim")
    return True

def treino(p_cfg):
    """
    Realiza a fase de treino conforme configuracoes.

    Args:
        p_cfg (Dict): dicionario com as configuracoes para execucao do pipeline de treinamento de embeddings.
    """
    inicio = datetime.datetime.now()

    if not treino_tokenizador(p_cfg):
        return
    
    cfg = copy.deepcopy(p_cfg)
                                                    
    try:
        # ajustar caminhos pelo modelo
        #embeddings
        cfg["comum"]["configuracao"]["embeddings"]["caminho_embeddings"]["cam_modelo"] = cfg["comum"]["configuracao"]["embeddings"]["caminho_embeddings"]["cam_modelo"] + '/' + cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"]
        cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["cam_modelo"] = cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["cam_modelo"] + '/' + cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"] + '_' + cfg["comum"]["configuracao"]["geral"]["tipo_dado"]
        cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["loss"] = ajustar_caminhos(cfg["treino_embeddings"]["configuracao"]["embeddings"]["saida"]["loss"], 
                                                                                                   cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"] + '_' + cfg["comum"]["configuracao"]["geral"]["tipo_dado"])
        
        cfg["comum"]["configuracao"]["embeddings"]["params_tipo_embeddings"]["geral"]["early_stopping"] = ajustar_caminhos(cfg["comum"]["configuracao"]["embeddings"]["params_tipo_embeddings"]["geral"]["early_stopping"], 
                                                                                                                              cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"] + '_' + cfg["comum"]["configuracao"]["geral"]["tipo_dado"])
        cfg["comum"]["configuracao"]["embeddings"]["params_tipo_embeddings"]["geral"]["caminho_log"] = cfg["comum"]["configuracao"]["embeddings"]["params_tipo_embeddings"]["geral"]["caminho_log"] + '/' + cfg["comum"]["configuracao"]["embeddings"]["tipo_embeddings"] + '_' + cfg["comum"]["configuracao"]["geral"]["tipo_dado"]

        if cfg["treino_embeddings"]["embeddings"]:
            print(f"{datetime.datetime.now()} - Treino")
            
            from ..embeddings import gerar_embeddings
            
            gerar_embeddings(
                concatenar_dicionarios(
                    concatenar_dicionarios(
                        cfg["treino_embeddings"]["configuracao"]["embeddings"],
                        cfg["comum"]["configuracao"]["embeddings"]),
                    cfg["comum"]["configuracao"]["geral"]))
            
    except Exception as erro:
        print(
            f"{datetime.datetime.now()} - Treino - Erro: {erro}")            
        raise

    print(f"{inicio} - Treino - Inicio")
    print(f"{datetime.datetime.now()} - Treino - Fim")                        