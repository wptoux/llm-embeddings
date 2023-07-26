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

## 分布式CoSent Loss
CoSent Loss是计算一个batch内的正负样本，因此，batch size的提升也可以提升一定的训练效果。但单卡可以容纳的样本量有限，通过all_gather操作，可以把多个device的样本embedding收集起来，统一计算loss。考虑到all_gather操作会阻断loss的反向传播，可以只对当前device对应的样本做反向传播。考虑到torch ddp会对所有device的梯度求平均后进行梯度更新，这种方法可以利用到batch内的所有样本来进行计算。即：

```python
def gather_tensor(t):
    ret = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(ret, t.contiguous())

    ret[dist.get_rank()] = t

    return torch.cat(ret, dim=0)

emb1 = get_embeddings(input_ids, attention_mask)
emb2 = get_embeddings(input_ids_b, attention_mask_b)
loss = cosent_loss(gather_tensor(emb1), gather_tensor(emb2), gather_tensor(labels))
```


## LoRA训练
考虑到LoRA会改变模型的输出分布，额外针对layer中输出相关的层设置lora可以提高模型效果。

以Bloom为例，除默认的query_key_value外，在dense，dense_h_to_4h，dense_4h_to_h等层中也加入lora，可以提升模型效果。

```
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=16, lora_dropout=0.1,
                         target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'])
```