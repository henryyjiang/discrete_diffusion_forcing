import logging
import gc
import json
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union, Dict
import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
from peft import PeftConfig, PeftModel
import numpy as np
import os
import jinja2

# Import LLaDA model related modules
from model_cache.llada.modeling_llada import LLaDAModelLM
from model_cache.llada.configuration_llada import LLaDAConfig

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="TemplateLM")

import random
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_bisection_schedule(block_length):
    """
    Compute the bisection schedule for a block.
    Returns list of positions to decode at each iteration.
    
    For a block of length L, iterations proceed as:
    - j=k-1: positions at multiples of 2^(k-1)
    - j=k-2: positions at multiples of 2^(k-2) (excluding already decoded)
    - ...
    - j=0: all remaining positions (multiples of 1)
    
    where k = ceil(log2(L))
    """
    if block_length == 0:
        return []
    
    k = 0
    while (1 << k) < block_length:
        k += 1
    
    schedule = []
    decoded_positions = set()
    
    # Iterate from j = k-1 down to 0
    for j in range(k-1, -1, -1):
        step = 1 << j  # 2^j
        iteration_positions = []
        
        for pos in range(0, block_length, step):
            if pos not in decoded_positions:
                iteration_positions.append(pos)
                decoded_positions.add(pos)
        
        if iteration_positions:
            schedule.append({
                'iteration': k - 1 - j,
                'j': j,
                'step': step,
                'positions': iteration_positions
            })
    
    return schedule


def apply_bisection_mask(input_ids, prompt_length, mask_id, block_size, iteration_j):
    """
    Apply bisection masking for a specific iteration level j.
    Masks all positions that are NOT multiples of 2^j within each block.
    
    Args:
        input_ids: [B, L] input token IDs
        prompt_length: [B] prompt lengths for each sequence
        mask_id: Token ID to use for masking
        block_size: Size of each block
        iteration_j: Current bisection iteration level (masks positions not divisible by 2^j)
    
    Returns:
        masked_input: [B, L] input with appropriate positions masked
        mask_indices: [B, L] boolean mask indicating which positions are masked
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    masked_input = input_ids.clone()
    mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    
    step = 1 << iteration_j  # 2^j
    
    for batch_idx in range(B):
        prompt_len = prompt_length[batch_idx].item()
        
        # Process each block
        block_start = prompt_len
        while block_start < L:
            block_end = min(block_start + block_size, L)
            
            for pos in range(block_start, block_end):
                relative_pos = pos - block_start
                # Mask positions that are NOT multiples of 2^j
                if relative_pos % step != 0:
                    masked_input[batch_idx, pos] = mask_id
                    mask_indices[batch_idx, pos] = True
            
            block_start = block_end
    
    return masked_input, mask_indices


def create_bidirectional_attention_mask(seq_length, prompt_length, block_size, device=None, dtype=None):
    """
    Create bidirectional attention mask for bisection sampling.
    All positions can attend to all other positions (full bidirectional attention).
    """
    if dtype is None:
        dtype = torch.float16
    
    # For bidirectional attention, return zeros (allow all attention)
    attention_mask = torch.zeros((1, 1, seq_length, seq_length), dtype=dtype, device=device)
    return attention_mask


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


@register_model("llada-bisection")
class LLaDABisectionEvalWrapper(TemplateLM):
    """
    LLaDA model wrapper for evaluation with bisection sampling-aware decoding.
    """
    
    def __init__(
        self,
        base_model_name_or_path: str = None,
        peft_model_name_or_path: str = None,
        max_length: Optional[int] = 2048,
        batch_size: Optional[int] = 1,
        device: Optional[str] = "cuda",
        dtype: str = "auto",
        mask_token_id: int = 126336,  # LLaDA mask token
        block_size: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int = 1234,
        mc_num: int = 8,
        log_type: str = 'ftb',
        nll_type: str = 'mc',
        add_bos_token: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.mc_num = mc_num
        self.log_type = log_type
        self.nll_type = nll_type
        self.add_bos_token = add_bos_token
        
        set_seed(seed)
        
        # Load tokenizer
        eval_logger.info(f"Loading tokenizer from {base_model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load base model
        eval_logger.info(f"Loading base model from {base_model_name_or_path}")
        dtype_torch = get_dtype(dtype)
        
        config = LLaDAConfig.from_pretrained(base_model_name_or_path)
        self.model = LLaDAModelLM.from_pretrained(
            base_model_name_or_path,
            config=config,
            torch_dtype=dtype_torch,
            device_map=device,
        )
        
        # Load PEFT adapter if provided
        if peft_model_name_or_path:
            eval_logger.info(f"Loading PEFT adapter from {peft_model_name_or_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_model_name_or_path,
                is_trainable=False,
            )
        
        self.model.eval()
        eval_logger.info("Model loaded successfully")

    def _forward_process(self, input_ids, prompt_length=None):
        """
        Forward process with bisection masking for evaluation.
        Randomly samples a bisection level j and masks accordingly.
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        masked_input = input_ids.clone()
        p_mask = torch.ones((B, L), dtype=torch.float32, device=device)
        
        if prompt_length is None:
            prompt_length = torch.zeros(B, dtype=torch.long, device=device)
        
        for batch_idx in range(B):
            prompt_len = prompt_length[batch_idx].item()
            
            block_start = prompt_len
            while block_start < L:
                block_end = min(block_start + self.block_size, L)
                block_len = block_end - block_start
                
                if block_len == 0:
                    break
                
                # Find k such that 2^k >= block_len
                k = 0
                while (1 << k) < block_len:
                    k += 1
                
                # Sample random j from [0, k-1]
                if k > 0:
                    j = torch.randint(0, k, (1,), device=device).item()
                else:
                    j = 0
                
                step = 1 << j  # 2^j
                
                # Compute masking probability
                if k > 0:
                    mask_prob = sum(1 for j_test in range(k) if step != (1 << j_test)) / k
                else:
                    mask_prob = 1.0
                
                # Mask positions not divisible by 2^j
                for pos in range(block_start, block_end):
                    relative_pos = pos - block_start
                    if relative_pos % step != 0:
                        masked_input[batch_idx, pos] = self.mask_token_id
                        p_mask[batch_idx, pos] = mask_prob if mask_prob > 0 else 1.0
                
                block_start = block_end
        
        p_mask = torch.clamp(p_mask, min=1e-8)
        return masked_input, p_mask

    def get_logits(self, input_ids, prompt_index=None):
        """
        Get logits from the model with bidirectional attention.
        """
        B, L = input_ids.shape
        
        # Create bidirectional attention mask (all zeros for full attention)
        attention_mask = torch.zeros(
            (B, 1, L, L),
            dtype=torch.float16,
            device=self.device
        )
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_bias=attention_mask)
            logits = outputs.logits
        
        return logits

    def _bisection_generate_block(self, context_ids, block_length, prompt_length):
        """
        Generate a single block using bisection sampling schedule.
        
        Args:
            context_ids: [B, context_length] - prompt + previously generated blocks
            block_length: Length of the block to generate
            prompt_length: Length of the prompt
            
        Returns:
            generated_block: [B, block_length] - generated tokens for this block
        """
        B = context_ids.shape[0]
        device = context_ids.device
        
        # Initialize the block with mask tokens
        current_block = torch.full(
            (B, block_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Compute bisection schedule
        schedule = compute_bisection_schedule(block_length)
        
        # Decode according to bisection schedule
        for iteration_info in schedule:
            j = iteration_info['j']
            positions = iteration_info['positions']
            
            if not positions:
                continue
            
            # Create full sequence: context + current block
            full_seq = torch.cat([context_ids, current_block], dim=1)
            
            # Create bidirectional attention mask
            seq_len = full_seq.shape[1]
            attention_mask = create_bidirectional_attention_mask(
                seq_len,
                prompt_length,
                self.block_size,
                device=device,
                dtype=torch.float16
            )
            
            # Get logits for the full sequence
            with torch.no_grad():
                outputs = self.model(full_seq, attention_bias=attention_mask)
                logits = outputs.logits
            
            # Extract logits for positions to decode in this iteration
            for pos in positions:
                absolute_pos = context_ids.shape[1] + pos
                token_logits = logits[:, absolute_pos, :] / self.temperature
                
                # Apply top-k and top-p filtering
                if self.top_k is not None:
                    token_logits = top_k_logits(token_logits, self.top_k)
                if self.top_p is not None:
                    token_logits = top_p_logits(token_logits, self.top_p)
                
                # Sample token
                probs = F.softmax(token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Update the block
                current_block[:, pos] = next_token
        
        return current_block

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text using bisection sampling for evaluation.
        """
        results = []
        
        for request in tqdm(requests, desc="Generating..."):
            # Extract context and generation parameters
            context = request.args[0]
            
            # Tokenize context
            if self.add_bos_token:
                context = self.tokenizer.bos_token + context
            
            context_ids = torch.tensor(
                self.tokenizer.encode(context),
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            
            prompt_length = context_ids.shape[1]
            
            # Determine generation length
            max_gen_length = self.max_length - prompt_length
            num_blocks = (max_gen_length + self.block_size - 1) // self.block_size
            
            generated_blocks = []
            
            # Generate blocks sequentially
            for block_idx in range(num_blocks):
                # Determine block length
                remaining_length = max_gen_length - len(generated_blocks) * self.block_size
                current_block_length = min(self.block_size, remaining_length)
                
                if current_block_length <= 0:
                    break
                
                # Concatenate context with previously generated blocks
                if generated_blocks:
                    current_context = torch.cat(
                        [context_ids] + generated_blocks,
                        dim=1
                    )
                else:
                    current_context = context_ids
                
                # Generate current block using bisection
                block = self._bisection_generate_block(
                    current_context,
                    current_block_length,
                    prompt_length
                )
                
                generated_blocks.append(block)
                
                # Check for EOS token
                if self.tokenizer.eos_token_id in block[0]:
                    break
            
            # Concatenate all generated blocks
            if generated_blocks:
                generated_ids = torch.cat(generated_blocks, dim=1)
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
            else:
                generated_text = ""
            
            results.append(generated_text)
        
        return results

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        """
        Evaluate target negative log-likelihood using Monte Carlo sampling
        with bisection masking.
        """
        if self.log_type == 'btf':
            seq = torch.concatenate([target, prefix])[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        
        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
            prompt_length = torch.tensor([len(prefix)] * self.batch_size, device=self.device)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)
            prompt_length = torch.tensor([len(target)] * self.batch_size, device=self.device)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq, prompt_length)
            
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(
                logits[mask_indices],
                seq[mask_indices],
                reduction='none'
            ) / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        """
        Evaluate target negative log-likelihood using autoregressive approach.
        """
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0)
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(
                prefix.shape[1] + target.shape[1],
                device=self.device
            ) < prefix.shape[1]
        else:
            prompt_index = torch.arange(
                prefix.shape[1] + target.shape[1],
                device=self.device
            ) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous()
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous()

        mask_index = torch.ones(
            (perturbed_.shape[1], perturbed_.shape[1]),
            dtype=torch.bool
        )
        
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        
        perturbed_[mask_index] = self.mask_token_id
        
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat(
                [prefix.repeat(perturbed_.shape[0], 1), perturbed_],
                dim=-1
            )
        else:
            perturbed_seq = torch.cat(
                [perturbed_, target.repeat(perturbed_.shape[0], 1)],
                dim=-1
            )

        logits_ = []
        num = (len(perturbed_seq) + self.batch_size - 1) // self.batch_size
        
        for i in range(num):
            end = min((i + 1) * self.batch_size, len(perturbed_seq))
            perturbed_seq_ = perturbed_seq[i * self.batch_size:end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones(
            (perturbed_.shape[1], perturbed_.shape[1]),
            dtype=torch.bool
        )
        
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        
        mask_index[temp_index] = False
        
        if self.log_type == 'ftb':
            logits_index = torch.cat([
                torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool),
                mask_index
            ], dim=-1)
        else:
            logits_index = torch.cat([
                mask_index,
                torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)
            ], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(
                logits[logits_index],
                target[0],
                reduction='sum'
            ).cpu().item()
        else:
            loss = F.cross_entropy(
                logits[logits_index],
                prefix[0],
                reduction='sum'
            ).cpu().item()
        
        return loss

    def _encode_pair(self, context, continuation):
        """Encode context and continuation pair."""
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
            
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        # Truncate on the left if needed
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(
                f"Text length {len(whole_enc)} is larger than {self.max_length}, "
                f"cutoff on the left side"
            )
            context_remain = context_enc_len - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning("All context (prompt) is truncated.")
                context_enc = []
                continuation_enc = whole_enc[-self.max_length:]
        
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for multiple-choice or other evaluation tasks.
        """
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type in ['ar_ftb', 'ar_btf']:
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: Implement greedy decoding check
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()