import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback

import json

from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model

from gpt_cosent_model import GPTCosentModel


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    embedding_method: Optional[str] = field(default="last_token")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SupervisedDataset():
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        self.s1 = []
        self.s2 = []
        self.label = []

        for line in open(data_path, 'r'):
            d = json.loads(line)

            if 'text1' in d and 'text2' in d and 'label' in d:
                self.s1.append(d['text1'])
                self.s2.append(d['text2'])
                self.label.append(d['label'])
            elif 'sentence1' in d and 'sentence2' in d and 'label' in d:
                self.s1.append(d['sentence1'])
                self.s2.append(d['sentence2'])
                self.label.append(d['label'])
            else:
                raise Exception('Unsupported data format')

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        a = self.tokenizer(self.s1[i], max_length=self.max_length,
                           padding='max_length', truncation=True, return_tensors='pt')

        b = self.tokenizer(self.s2[i], max_length=self.max_length,
                           padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': a.input_ids[0],
            'attention_mask': a.attention_mask[0],
            'input_ids_b': b.input_ids[0],
            'attention_mask_b': b.attention_mask[0],
            'labels': self.label[i]
        }
    

from scipy.stats import spearmanr
def spearman_score(eval_prediction):
    emb1, emb2 = eval_prediction.predictions
    
    sims = F.cosine_similarity(torch.tensor(emb1).float(), torch.tensor(emb2).float()).numpy()
    
    return {
        'spearman': spearmanr(sims, eval_prediction.label_ids).correlation
    }

from torch.distributed.elastic.multiprocessing.errors import record

@record
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.ddp_find_unused_parameters = False  # 开启checkpointing需要关闭ddp_find_unused_parameters

    logging.basicConfig(filename=os.path.join(training_args.output_dir, 'trainer_log.log'), level=logging.INFO)

    rank = dist.get_rank()

    if rank == 0:
        print('model_args:', model_args)
        print('data_args:', data_args)
        print('training_args:', training_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map={'': rank}
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=16, lora_dropout=0.1,
                             target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'])

    model = get_peft_model(model, peft_config)

    if rank == 0:
        model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = GPTCosentModel(model, tokenizer, embedding_method=model_args.embedding_method)
    
    if rank == 0:
        print(model)

    dataset = SupervisedDataset(data_args.data_path, tokenizer, max_length=training_args.model_max_length)
    eval_dataset = SupervisedDataset(data_args.eval_data_path, tokenizer, max_length=training_args.model_max_length)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset, 
                      eval_dataset=eval_dataset, compute_metrics=spearman_score)
    trainer.train()

    trainer.save_model(os.path.join(training_args.output_dir, 'final_model'))


if __name__ == "__main__":
    train()
