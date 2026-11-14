Para treinar o modelo LLM é necessário utilizar o deepspeed para reduzir o consumo de memória das GPUs e viabilizar o treinamento.

Para isso os passos descritos a seguir ajudaram na definição do arquivo de configuração do deepspeed.

Obs.: 
- Para rodar o Llama2 em 4 GPUs A100 40 GiB foi necessário seguir até o passo 4 e foi criado o arquivo de configuração embeddings/deepspeed_config/llama2.json
- Ao colocar o argumento de treino gradient_checkpointing=True foi necessário chamar o método model_transformer.enable_input_require_grads() na inicialização do modelo e antes de iniciar o trainamento foi acrescentado:
    - torch.cuda.empty_cache()
    - model_transformer.config.use_cache = False

https://huggingface.co/docs/transformers/main_classes/deepspeed

1 - First of all set batch size to 1 (you can always use gradient accumulation for any desired effective batch size).

2 - Enable --gradient_checkpointing 1 (HF Trainer) or directly model.gradient_checkpointing_enable() - if OOM then

3 - Try ZeRO stage 2 first. if OOM then

4 - Try ZeRO stage 2 + offload_optimizer - if OOM then

5 - Switch to ZeRO stage 3 - if OOM then

6 - Enable offload_param to cpu - if OOM then

7 - Enable offload_optimizer to cpu - if OOM then

8 - If you still can’t fit a batch size of 1 first check various default values and lower them if you can. For example, if you use generate and you don’t use a wide search beam make it narrower as it’d take a lot of memory.

9 - Definitely use mixed half-precision over fp32 - so bf16 on Ampere and higher GPUs and fp16 on older gpu architectures.

10 - If you still OOM you could add more hardware or enable ZeRO-Infinity - that is switch offloads offload_param and offload_optimizer to nvme. You need to make sure it’s a very fast nvme. As an anecdote I was able to infer BLOOM-176B on a tiny GPU using ZeRO-Infinity except it was extremely slow. But it worked
