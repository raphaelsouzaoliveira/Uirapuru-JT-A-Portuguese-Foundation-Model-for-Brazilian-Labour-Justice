# coding: utf-8
"""
Executa a tecnica de embeddings nos dados de treino apos a limpeza isoladamente via linha de comando.
"""
import datetime

import yaml

from ..utils import (carregar_cfg, concatenar_dicionarios)

from . import gerar_embeddings

if __name__ == "__main__":
    CFG = carregar_cfg("integracao/cfg.yaml")

    print(f"{datetime.datetime.now()} - Embeddings - Inicio")

    gerar_embeddings(
        concatenar_dicionarios(
            concatenar_dicionarios(
                CFG["treino_embeddings"]["configuracao"]["embeddings"],
                CFG["comum"]["configuracao"]["embeddings"]),
            CFG["comum"]["configuracao"]["geral"]))

    print(f"{datetime.datetime.now()} - Embeddings - Fim")
