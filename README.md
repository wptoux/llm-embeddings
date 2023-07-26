# llm-embeddings

基于大语言模型（LLM）的Embedding向量模型。相对于基于Bert等encoder模型构建Embedding，基于LLM构建Embedding具有一定的优势，包括以下方面
+ 更大的参数量，更多的预训练数据，更强的语义理解能力：Bert-base等模型有110M参数，而即使较小的LLM，如ChatGLM-6B、Bloom-7B、LLAMA-7B等，也有数十亿参数，这些LLM的预训练数据也远多于Bert等模型，因此往往具有更好的语义理解能力。
+ 更长的上下文长度。Bert预训练长度一般为512，而LLM则至少支持2048以上的上下文。

因此，在信息的检索等领域，基于LLM的向量模型具有一定的优势。

但LLM参数量大，微调起来相对困难。本仓库实现了基于LoRA的LLM向量模型微调方法，并实现了分布式CoSent损失函数来支持分布式训练。训练得到的向量模型相比基于Bert等的向量模型，在效果上具有一定的提升。

## 训练方法

以Bloom7b为例，使用text2vec-base-chinese-paraphrase-dataset作为训练数据。训练数据来自text2vec项目（[https://github.com/shibing624/text2vec](https://github.com/shibing624/text2vec)）。

```bash
torchrun --nnodes 1 --nproc_per_node=$NGPU train.py \
    --model_name_or_path bigscience/bloomz-7b1 \
    --data_path data/text2vec-base-chinese-paraphrase-dataset.jsonl \
    --eval_data_path data/STS-B/STS-B.valid.data \
    --embedding_method last_token_2l \
    --bf16 true \
    --output_dir $OUTPUT \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --optim adamw_torch \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --model_max_length 256
```


