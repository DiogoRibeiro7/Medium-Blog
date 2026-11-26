"""A Hugging Face causal language model behind the loops_core.Model interface.

Prompts in a batch are padded on the left; the attention mask hides the padding and position ids
count only real tokens, so a padded batch decodes exactly like each prompt on its own (tested).
"""

import numpy as np
import torch

import loops_core as L


class HFModel(L.Model):
    def __init__(self, model, tokenizer=None):
        self.model = model.eval()
        self.tok = tokenizer

    @classmethod
    def load(cls, name, revision, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
        return cls(model, tok)

    def _batch(self, seqs):
        n = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), n), dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s) :] = torch.tensor(s, dtype=torch.long)
            mask[i, n - len(s) :] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids, mask, pos

    @torch.inference_mode()
    def start(self, prompts):
        ids, self.mask, pos = self._batch(prompts)
        out = self.model(input_ids=ids, attention_mask=self.mask, position_ids=pos, use_cache=True)
        # transformers 5 returns the key-value cache as `cache` (formerly `past_key_values`)
        self.cache, self.pos = out.cache, pos[:, -1]
        return out.logits[:, -1, :].float().numpy()

    @torch.inference_mode()
    def step(self, ids):
        ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)[:, None]
        self.mask = torch.cat([self.mask, torch.ones_like(ids)], dim=1)
        self.pos = self.pos + 1
        out = self.model(
            input_ids=ids,
            attention_mask=self.mask,
            position_ids=self.pos[:, None],
            cache=self.cache,
            use_cache=True,
        )
        self.cache = out.cache
        return out.logits[:, -1, :].float().numpy()

    @torch.inference_mode()
    def score(self, contexts, targets):
        """Log-probability and top-1/top-2 gap at every target token, teacher-forced after its context."""
        seqs = [list(c) + list(t) for c, t in zip(contexts, targets, strict=True)]
        ids, mask, pos = self._batch(seqs)
        ls = torch.log_softmax(self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float(), -1)
        n = ids.shape[1]
        lp = np.zeros((len(seqs), max(len(t) for t in targets)))
        mg = np.zeros_like(lp)
        for i, t in enumerate(targets):
            k = len(t)
            rows = ls[i, n - k - 1 : n - 1]  # position j predicts token j + 1
            lp[i, :k] = rows[torch.arange(k), torch.as_tensor(t)].numpy()
            top2 = rows.topk(2, dim=-1).values
            mg[i, :k] = (top2[:, 0] - top2[:, 1]).numpy()
        return lp, mg
