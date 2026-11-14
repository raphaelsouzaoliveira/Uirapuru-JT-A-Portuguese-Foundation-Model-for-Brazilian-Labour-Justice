# coding: utf-8
from utils import carregar_cfg, carregar_super_cfg, ajustar_caminhos_base
from integracao import treino
import copy
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

if __name__ == "__main__":
    cam_cfg = "integracao/cfg_treino_tucano_ptBR_jud.yaml"
    CFG = carregar_super_cfg(carregar_cfg(cam_cfg))
    
    if CFG["comum"]["configuracao"]["geral"]["cam_base"] != "":                
        CFG = ajustar_caminhos_base(copy.deepcopy(CFG), CFG["comum"]["configuracao"]["geral"]["cam_base"])
    treino(CFG)