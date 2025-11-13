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

def create_bisection_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    """
    Creates a complete bidirectional attention mask for bisection sampling.
    All tokens can see all other tokens (full bidirectional attention).
    
    Args:
        prompt_length: Length of the prompt (first irregular block)
        max_length: Maximum total sequence length
        block_size: Size of each regular block
        device: Device to create tensor on
        dtype: Data type for the attention mask
        
    Returns:
        attention_mask: Tensor of shape [1, 1, max_length, max_length] with all zeros (full attention)
    """
    if dtype is None:
        dtype = torch.bfloat16
    
    # For bidirectional attention, return all zeros (no masking)
    attention_mask = torch.zeros((1, 1, max_length, max_length), device=device, dtype=dtype)
    
    return attention_mask

def extract_attention_mask(full_mask, start_pos, input_length, cache_length):
    """
    Extract the relevant portion of attention mask for current forward pass.
    For bisection sampling, this just returns zeros (full attention).
    
    Args:
        full_mask: Complete attention mask [1, 1, max_length, max_length]
        start_pos: Starting position in the full sequence
        input_length: Length of current input sequence
        cache_length: Length of cached sequence
        
    Returns:
        attention_mask: Extracted mask [1, 1, input_length, cache_length + input_length]
    """
    total_length = cache_length + input_length
    
    # For bidirectional attention, return all zeros
    extracted_mask = torch.zeros((1, 1, input_length, total_length), 
                                 device=full_mask.device, dtype=full_mask.dtype)
    
    return extracted_mask

def bisection_iteration_mask(block_length, j, device):
    """
    Create a mask for bisection iteration j.
    Returns True for positions that should be MASKED (not multiples of 2^j).
    
    Args:
        block_length: Length of the block
        j: Bisection iteration level (0 to k-1)
        device: Device to create tensor on
        
    Returns:
        mask: Boolean tensor of shape [block_length] where True means MASK
    """
    step = 1 << j  # 2^j
    positions = torch.arange(block_length, device=device)
    # Mask positions that are NOT multiples of 2^j
    mask = (positions % step) != 0
    return mask

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

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    
    confidence = initial_confidence.clone()
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0, initial_confidence

@register_model("bisection")
class DreamLoRABisection(TemplateLM):
    def __init__(
        self,
        base_model_name_or_path: Union[str, transformers.PreTrainedModel],
        peft_model_name_or_path: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 1024,
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 32,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 32,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.2,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        block_size: Optional[int] = 4,
        mask_token_id: Optional[int] = 126336,
        skip_threshold: Optional[float] = 0.9,
        sampling_strategy: Optional[str] = "default",
        save_dir: Optional[str] = None,
        show_speed: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(base_model_name_or_path, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        if not (parallelize or accelerator.num_processes > 1):
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        self.lora_path = peft_model_name_or_path
        self.block_size = block_size
        self.skip_threshold = skip_threshold
        self.sampling_strategy = sampling_strategy
        
        self.target_dtype = get_dtype(dtype)
        
        self._create_model_and_tokenizer(base_model_name_or_path, dtype, trust_remote_code)

        if isinstance(base_model_name_or_path, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
        else:
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.block_size = block_size
        self.mask_token_id = mask_token_id

        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps
        
        self.backend = "causal"
        self.truncation = False

        self.save_dir = save_dir
        self.show_speed = show_speed

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def eot_token_id(self): 
        return self.tokenizer.eos_token_id

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        target_dtype = get_dtype(dtype)
        
        config = LLaDAConfig.from_pretrained(pretrained)
        self.model = LLaDAModelLM.from_pretrained(
            pretrained, 
            config=config,
            torch_dtype=target_dtype,
            trust_remote_code=False,
        ).eval()
        
        peft_config = PeftConfig.from_pretrained(self.lora_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        
        if target_dtype is not None and target_dtype != "auto":
            self.model = self.model.to(target_dtype)
        
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}

        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _count_tokens_after_truncation(self, response_text: str, until_terms: List[str] = None) -> int:
        """
        Unified token counting function: calculates the number of non-126081 tokens after truncating the response.
        """
        truncated_text = response_text
        if until_terms and not self.escape_until:
            for term in until_terms:
                if len(term) > 0:
                    truncated_text = truncated_text.split(term)[0]
        
        generated_answer_ids = torch.tensor(self.tokenizer(truncated_text)["input_ids"])
        return int((generated_answer_ids != 126081).sum())

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_bisection_single(self, prompt):
        """
        Generates a response using bisection sampling strategy.
        Each block undergoes bisection iterations from coarse to fine.
        
        Returns: generated_sequence (List[int]) - List of generated token IDs
        """
        self.model.eval()
        
        mask_id = self.mask_token_id
        block_size = self.block_size
        skip_threshold = self.skip_threshold
        
        # Pre-generate the full attention mask (bidirectional for bisection)
        prompt_length = prompt.shape[1]
        full_attention_mask = create_bisection_block_attention_mask(
            prompt_length=prompt_length,
            max_length=self.max_length,
            block_size=block_size,
            device=self.device,
            dtype=self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        )
        
        with torch.inference_mode():
            x_t = prompt.to(self.device)
            
            # Track block states with bisection iteration levels
            # Each block has a 'current_j' indicating current bisection level
            block_states = {
                0: {  # Prompt block
                    'start_pos': 0,
                    'end_pos': prompt.shape[1],
                    'state': 'completed',
                    'current_j': None,  # Prompt doesn't use bisection
                },
            }
            
            past_key_values = None
            cache_length = 0
            step = 0
            eos_detected = False
            
            # Add initial blocks
            num_initial_blocks = min((self.max_new_tokens // block_size), 4)  # Start with a few blocks
            for b in range(num_initial_blocks):
                new_block_id = len(block_states)
                new_start_pos = x_t.shape[1]
                x_t = torch.cat([x_t, torch.tensor([[mask_id] * block_size]).to(self.device)], dim=1)
                
                block_length = block_size
                # Calculate k (max bisection level)
                k = 0
                while (1 << k) < block_length:
                    k += 1
                
                block_states[new_block_id] = {
                    'start_pos': new_start_pos,
                    'end_pos': new_start_pos + block_size,
                    'state': 'active',
                    'current_j': k-1,
                    'k': k,  # Maximum bisection level
                }
            
            while True:
                step += 1
                
                # Check termination
                mask_index = (x_t == mask_id)
                if mask_index.sum() == 0:
                    break
                
                # Check if we should add more blocks
                if len(block_states) - 1 < (self.max_new_tokens // block_size) and not eos_detected:
                    # Check if the last block has progressed enough
                    last_block_id = max(block_states.keys())
                    if block_states[last_block_id]['state'] in ['to_cache', 'cached']:
                        # Add a new block
                        new_block_id = len(block_states)
                        new_start_pos = x_t.shape[1]
                        x_t = torch.cat([x_t, torch.tensor([[mask_id] * block_size]).to(self.device)], dim=1)
                        
                        block_length = block_size
                        k = 0
                        while (1 << k) < block_length:
                            k += 1
                        
                        block_states[new_block_id] = {
                            'start_pos': new_start_pos,
                            'end_pos': new_start_pos + block_size,
                            'state': 'active',
                            'current_j':k-1,
                            'k': k,
                        }
               
                
                # Determine blocks to cache (completed blocks)
                blocks_to_cache = [bid for bid, state in block_states.items() 
                                 if state['state'] == 'to_cache']
                
                update_kvcache = 0
                if blocks_to_cache:
                    earliest_block_id = min(blocks_to_cache)
                    earliest_pos = block_states[earliest_block_id]['start_pos']
                    latest_block_id = max(blocks_to_cache)
                    latest_pos = block_states[latest_block_id]['end_pos']
                    update_kvcache = latest_pos - earliest_pos
                
                # Create input sequence
                process_start_pos = cache_length
                
                if update_kvcache > 0:
                    earliest_block_to_cache = min(blocks_to_cache)
                    input_seq = x_t[:, block_states[earliest_block_to_cache]['start_pos']:]
                    process_start_pos = block_states[earliest_block_to_cache]['start_pos']
                else:
                    active_blocks = [bid for bid, state in block_states.items() if state['state'] == 'active']
                    if active_blocks:
                        earliest_active_after_cache = float('inf')
                        for bid in active_blocks:
                            if block_states[bid]['start_pos'] >= cache_length:
                                earliest_active_after_cache = min(earliest_active_after_cache, block_states[bid]['start_pos'])
                        
                        if earliest_active_after_cache < float('inf'):
                            input_seq = x_t[:, earliest_active_after_cache:]
                            process_start_pos = earliest_active_after_cache
                        else:
                            input_seq = x_t[:, cache_length:]
                            if cache_length >= x_t.shape[1]:
                                break
                    else:
                        break
                
                if input_seq.shape[1] == 0:
                    break
                
                # Extract attention mask
                input_length = input_seq.shape[1]
                attention_mask = extract_attention_mask(
                    full_mask=full_attention_mask,
                    start_pos=process_start_pos,
                    input_length=input_length,
                    cache_length=cache_length
                )

                outputs = self.model(
                    input_seq,
                    attention_bias=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    update_kvcache=update_kvcache+cache_length,
                )
                
                logits = outputs.logits
                
                # Update cache if needed
                if update_kvcache > 0:
                    past_key_values = outputs.past_key_values
                    for block_id in blocks_to_cache:
                        block_states[block_id]['state'] = 'cached'
                
                # Process active blocks with bisection strategy
                for block_id in sorted(block_states.keys()):
                    if block_states[block_id]['state'] != 'active':
                        continue
                    
                    block_start = block_states[block_id]['start_pos']
                    block_end = block_states[block_id]['end_pos']
                    current_j = block_states[block_id]['current_j']
                    k = block_states[block_id]['k']
                    block_length = block_end - block_start
                    
                    # Get bisection mask for current iteration j
                    bisection_mask = bisection_iteration_mask(block_length, current_j, self.device)
                    
                    # Get mask positions for this block and iteration
                    block_mask_index = mask_index.clone()
                    block_mask_index[:, :block_start] = False
                    block_mask_index[:, block_end:] = False
                    
                    # Only process positions dictated by bisection mask
                    for i in range(block_length):
                        abs_pos = block_start + i
                        if not bisection_mask[i]:  # This position should be unmasked in iteration j
                            block_mask_index[:, abs_pos] = False
                    
                    if block_mask_index.sum() == 0:
                        # Current iteration complete, advance to next j
                        block_states[block_id]['current_j'] -= 1
                        if block_states[block_id]['current_j'] < 0:
                            # All bisection iterations complete
                            block_states[block_id]['state'] = 'to_cache'
                        continue
                    
                    # Calculate relative position of logits
                    logit_offset = block_start - process_start_pos
                    block_rel_positions = torch.where(block_mask_index[0, block_start:block_end])[0]
                    
                    if block_rel_positions.size(0) > 0:
                        block_mask_logits = logits[:, logit_offset + block_rel_positions, :]
                        
                        confidence, x0, initial_confidence = sample_tokens(
                            block_mask_logits.squeeze(0), 
                            self.temperature, 
                            top_p=self.top_p, 
                            top_k=self.top_k, 
                            neg_entropy=(self.sampling_strategy == "neg_entropy"),
                            margin_confidence=(self.sampling_strategy == "margin_confidence")
                        )
                        
                        # Apply skip threshold
                        high_conf_indices = torch.where(initial_confidence > skip_threshold)[0]
                        
                        # Update tokens
                        if len(high_conf_indices) > 0:
                            for idx in high_conf_indices:
                                abs_pos = block_start + block_rel_positions[idx]
                                x_t[0, abs_pos] = x0[idx]
                            
                            # Check for EOS
                            eos_token_id = 126081
                            if eos_token_id is not None:
                                for idx in high_conf_indices:
                                    if x0[idx].item() == eos_token_id:
                                        eos_detected = True
                                        break
                        
                        # Check if current iteration j is complete
                        mask_index = (x_t == mask_id)
                        block_mask_index = mask_index.clone()
                        block_mask_index[:, :block_start] = False
                        block_mask_index[:, block_end:] = False
                        
                        # Filter by bisection mask
                        for i in range(block_length):
                            abs_pos = block_start + i
                            if not bisection_mask[i]:
                                block_mask_index[:, abs_pos] = False
                        
                        if block_mask_index.sum() == 0:
                            # Current iteration complete
                            block_states[block_id]['current_j'] -= 1
                            if block_states[block_id]['current_j'] < 0:
                                block_states[block_id]['state'] = 'to_cache'

                if update_kvcache > 0:
                    cache_length += update_kvcache
                
                if step > 250:
                    print(f"WARNING: Hit safety check at step {step}. Exiting generation loop.")
                    break
        
        generated_sequence = x_t[0, prompt.shape[1]:].tolist()
        return generated_sequence

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        start_time = time.time()
        
        num_tokens = 0
        num_nfe = 0
        
        bar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)), desc="Running generate_until requests")
        
        for i, req in enumerate(requests):
            question = req.args[0]
            gen_kwargs = req.args[1]
            
            contexts = [question]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
            )

            input_ids = context_enc[0].unsqueeze(0)
            
            if input_ids.shape[1] > self.max_length - self.max_new_tokens:
                eval_logger.warning(f"Prompt length {input_ids.shape[1]} is larger than {self.max_length-self.max_new_tokens}, cutoff on the left side")
                input_ids = input_ids[:, -(self.max_length-self.max_new_tokens):]
            
            # Use bisection sampling generation
            generated_answer = self._generate_bisection_single(input_ids)
            
            cont_toks_list = self.tokenizer.batch_decode([generated_answer], skip_special_tokens=True)
            s = cont_toks_list[0]
            
            if self.show_speed:
                num_tokens += self._count_tokens_after_truncation(s, gen_kwargs.get("until", []))
                num_nfe += 1
            
            if not self.escape_until:
                for term in gen_kwargs.get("until", []):
                    if len(term) > 0:
                        s = s.split(term)[0]
            
            res.append(s)
            bar.update(1)
        
        bar.close()
        
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            final_time = time.time()
            total_time = final_time - start_time
            
            final_stats = {
                "processed_samples": len(res),
                "total_samples": len(requests), 
                "total_tokens": int(num_tokens),
                "total_nfe": int(num_nfe),
                "total_time": total_time,
                "tokens_per_second": float(num_tokens) / total_time if total_time > 0 else 0.0,
                "nfe_per_token": float(num_nfe) / float(num_tokens) if num_tokens > 0 else 0.0,
                "timestamp": final_time
            }
            final_stats_path = os.path.join(self.save_dir, f'rank_{self.rank}_final_stats.json')
            with open(final_stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        if self.show_speed:
            final_time = time.time()
            total_time = final_time - start_time
            print(f"\n=== Final Statistics ===")
            print(f"Processed samples: {len(res)}")
            print(f"Total tokens: {num_tokens}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Throughput: {num_tokens / total_time:.2f} tokens/s")
            print(f"Total NFE: {num_nfe}")
        
        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])

        input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        
        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
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
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0)
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous()
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous()

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
            
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len-cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning(f"All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        print(ds[0])
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
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

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