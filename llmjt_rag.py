import torch
import os
import glob
import pandas as pd
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.pipelines.pt_utils import KeyDataset

tipo = 'UirapuruJT3.2'
#tipo = 'UirapuruJT2'
DEBUG = False

max_length = None
if tipo == 'UirapuruJT3.2':    
    model_path = '/dados/LLMJT/saida/embeddings/modelo/Llama3_2_ptBRJT/checkpoint-5667'
    tokenizer_path = '/dados/LLMJT/saida/embeddings/modelo_base/Llama3_2/tokenizer'
    model_multilabel_path = '/dados/LLMJT/saida/embeddings/modelo/Llama3_2_ptBRJT/multilabel_model_conteudo_checkpoint-5667_70_congelamento/checkpoint-187'
    max_length = 8192
elif tipo == 'UirapuruJT2':    
    #Tucano2 finetuning JT
    model_name = 'Uirapuru JT (Tucano 2)'
    model_path = '/dados/LLMJT/saida/embeddings/modelo/Tucano2_ptBRJT/checkpoint-2450'
    tokenizer_path = '/dados/LLMJT/saida/embeddings/modelo_base/Tucano2/tokenizer'    
    model_multilabel_path = '/dados/LLMJT/saida/embeddings/modelo/Tucano2_ptBRJT/multilabel_model_conteudo_checkpoint-2450_70_congelamento/checkpoint-204'
    max_length = 4096

PARQUET_PATH = "/dados/LLMJT/saida/extrator/final/*_decisoes-arrevista.parquet"
CHROMA_DB_DIR = model_path + f"_chroma_db_jt_{tipo}_gpu"
BATCH_SIZE = 24  # Tamanho do batch para o pipeline da GPU
INSERT_BATCH_SIZE = 5000 # Tamanho do batch para salvar no Chroma

# Configuração do Splitter para RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

class PipelineIngestao:
    def __init__(self):
        print("Carregando modelos na memória...")
        device = 0 if torch.cuda.is_available() else -1
        
        # Otimizando a pipeline com fp16 para economizar VRAM se aplicável
        self.classifier = pipeline(
            "text-classification", 
            model=model_multilabel_path, 
            tokenizer=tokenizer_path,
            #device=device,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        if self.classifier.tokenizer.pad_token is None:
            self.classifier.tokenizer.pad_token = self.classifier.tokenizer.eos_token
            self.classifier.tokenizer.pad_token_id = self.classifier.tokenizer.eos_token_id
            self.classifier.model.config.pad_token_id = self.classifier.tokenizer.eos_token_id
        
        # Mapeamento dinâmico do modelo
        old_id2label = self.classifier.model.config.id2label
        sorted_keys = sorted(map(int, old_id2label.keys()))
        new_id2label = {i: old_id2label[k] for i, k in enumerate(sorted_keys)}
        self.classifier.model.config.id2label = new_id2label
        self.classifier.model.config.label2id = {v: k for k, v in new_id2label.items()}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cuda' if device == 0 else 'cpu'}
        )
        self.vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
        self.threshold = 0.509

def carregar_tudo(padrao_glob: str):
    """Lê todos os parquets e concatena para uso na biblioteca datasets."""
    arquivos = glob.glob(padrao_glob)
    dfs = []
    for arquivo in arquivos:
        dfs.append(pd.read_parquet(arquivo))
    return pd.concat(dfs, ignore_index=True)

if __name__ == '__main__':
    pipeline_jt = PipelineIngestao()
    
    # 1. Carrega os dados
    print("Carregando arquivos parquet...")
    df_completo = carregar_tudo(PARQUET_PATH)
    if DEBUG:
        df_completo = df_completo.head(100)
        
    print(f"Quantidade de decisões com 'conteudo' NaN/Nulo: {df_completo['conteudo'].isna().sum()}")
    
    df_completo['conteudo'] = df_completo['conteudo'].fillna("").astype(str)
        
    # 2. Converte para HuggingFace Dataset (Resolve o aviso de eficiência)
    hf_dataset = Dataset.from_pandas(df_completo)
    
    docs_buffer = []
    
    # 3. Processamento Sequencial Otimizado na GPU
    print("Iniciando predição multirrótulo e extração de chunks...")
    
    # Passando os dados no pipeline usando KeyDataset generator pattern
    # O pipeline iterará em background de forma eficiente
    gerador_pipeline = pipeline_jt.classifier(
        KeyDataset(hf_dataset, "conteudo"), 
        batch_size=BATCH_SIZE, 
        truncation=True, 
        max_length=max_length,
        top_k=None
    )
    
    for idx, out in enumerate(tqdm(gerador_pipeline, total=len(hf_dataset))):
        # Filtra os temas pelo threshold
        temas_ativos = [item['label'] for item in out if item['score'] > pipeline_jt.threshold]
        
        # Se não encontrou tema, passa string vazia para não quebrar metadado
        if not temas_ativos:
            temas_ativos = ['']
            
        conteudo_original = hf_dataset[idx]['conteudo']
        
        # RAG: Quebra o texto em chunks menores adequados para a janela de embeddings
        chunks = text_splitter.split_text(conteudo_original)
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "temas": ", ".join(temas_ativos), 
                    "regional": str(hf_dataset[idx]['regional']),
                    "ano": str(hf_dataset[idx]['ano'])
                }
            )
            docs_buffer.append(doc)
            
        # Gravação em lotes no Chroma
        if len(docs_buffer) >= INSERT_BATCH_SIZE:
            pipeline_jt.vectordb.add_documents(docs_buffer)
            docs_buffer = []
            
    # Salva os documentos restantes
    if docs_buffer:
        pipeline_jt.vectordb.add_documents(docs_buffer)

    print("Processamento e ingestão concluídos com sucesso.")