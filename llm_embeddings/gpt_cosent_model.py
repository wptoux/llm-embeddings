import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class GPTCosentModel(nn.Module):
    def __init__(self, model, tokenizer, device='cuda', embedding_method='last_token_2l', disable_kv_cache=True):
        super().__init__()

        if disable_kv_cache:
            model.config.use_cache = False

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_method = embedding_method

    def get_embeddings(self, input_ids, attention_mask=None):
        method = self.embedding_method

        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        if method == 'last_token':
            emb = model_output.hidden_states[-1][:, -1, :]

            return emb

        elif method == 'last_token_2l':
            all_hidden = model_output.hidden_states

            emb = (all_hidden[1][:, -1] + all_hidden[-1][:, -1]) / 2

            return emb

        elif method == 'last_token_mean':
            all_hidden = model_output.hidden_states

            emb = torch.mean(torch.stack([h[:, -1] for h in all_hidden]), dim=0)

            return emb

        elif method == 'weighted_mean':
            hidden = model_output.hidden_states[-1]

            device = input_ids.device
            dtype = hidden.dtype

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).to(dtype=dtype)

            weights = torch.arange(1, hidden.size(1) + 1, device=device).to(dtype=dtype) / hidden.size(1)
            weights = weights.unsqueeze(0).unsqueeze(-1).expand(hidden.size())

            mask_expanded *= weights

            emb = torch.sum(hidden * mask_expanded, dim=1) / (mask_expanded.sum(dim=1) + 1e-9)

            return emb

        elif method == 'weighted_mean_exp':
            hidden = model_output.hidden_states[-1]

            device = input_ids.device
            dtype = hidden.dtype

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).to(dtype=dtype)

            weights = torch.arange(hidden.size(1) - 1, -1, step=-1, device=device).to(dtype=dtype)
            weights = 0.99 ** weights
            weights = weights.unsqueeze(0).unsqueeze(-1).expand(hidden.size())

            mask_expanded *= weights

            emb = torch.sum(hidden * mask_expanded, dim=1) / (mask_expanded.sum(dim=1) + 1e-9)

            return emb

        else:
            raise Exception('unsupported method')

    def forward(self, input_ids, attention_mask=None, input_ids_b=None, attention_mask_b=None, labels=None, ddp=True):
        emb1 = self.get_embeddings(input_ids, attention_mask)

        if input_ids_b is None or labels is None:
            return emb1

        emb2 = self.get_embeddings(input_ids_b, attention_mask_b)

        if ddp:
            loss = self.get_loss(self.gather_tensor(emb1), self.gather_tensor(emb2), self.gather_tensor(labels))
        else:
            loss = self.get_loss(emb1, emb2, labels)

        return loss, emb1, emb2

    def get_loss(self, emb1, emb2, labels):
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8) * 20
        cosine_sim = cosine_sim[:, None] - cosine_sim[None, :]  # 相当于计算batch内样本两两之间的差

        labels = (labels[:, None] < labels[None, :]).long()

        cosine_sim = cosine_sim - (1 - labels) * 1e12  # 对于负样本，其cosine_sim变成了一个极小的负数，其exp约为0，求和时不会计入
        cosine_sim = torch.cat((torch.zeros(0, dtype=cosine_sim.dtype, device=cosine_sim.device),
                                cosine_sim.view(-1)), dim=0)

        loss = torch.logsumexp(cosine_sim, dim=0)

        return loss

    def gather_tensor(self, t):
        ret = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(ret, t.contiguous())

        ret[dist.get_rank()] = t

        return torch.cat(ret, dim=0)

    def encode(self, txts, batch_size=32, max_length=512, normalize=False):
        if isinstance(txts, str):
            txts = [txts]

        with torch.no_grad():
            ret = []

            for i in tqdm(range(0, len(txts), batch_size)):
                ipt = self.tokenizer(txts[i: i + batch_size], return_tensors='pt', truncation=True,
                                     max_length=max_length, padding='longest')
                ipt = dict((k, v.to(device=self.device)) for k, v in ipt.items())

                emb = self.get_embeddings(**ipt)

                if normalize:
                    emb = F.normalize(emb)

                ret.append(emb.cpu().numpy())

            return np.concatenate(ret, axis=0)