import torch
from utils.util import forward_process_length, shift_logits,forward_process
import torch.nn.functional as F

def bisection_sampling_aware_mask(input_ids, prompt_lengths, mask_id, block_size, eos_id):
    """
    Apply bisection sampling-aware masking strategy.
    
    For each block, randomly sample a bisection iteration level j,
    then mask all positions that are NOT multiples of 2^j within that block.
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    # Create a copy for masking
    masked_input = input_ids.clone()
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    p_mask = torch.ones_like(input_ids, dtype=torch.float32)
    
    for batch_idx in range(B):
        prompt_len = prompt_lengths[batch_idx].item()
        
        # Process each block
        l = prompt_len
        while l < L:
            r = min(l + block_size, L)
            block_len = r - l
            
            if block_len == 0:
                break
            
            # Find k such that 2^k >= block_len
            k = 0
            while (1 << k) < block_len:
                k += 1
            
            # Sample a random bisection iteration j from [0, k-1]
            if k > 0:
                j = torch.randint(0, k, (1,), device=device).item()
            else:
                j = 0
            
            # Create mask: exclude positions that are NOT multiples of 2^j
            step = 1 << j  # 2^j
            
            # Compute masking probability for this block
            if k > 0:
                mask_prob = sum(1 for j_test in range(k) 
                              if step != (1 << j_test)) / k
            else:
                mask_prob = 1.0
            
            for pos in range(l, r):
                relative_pos = pos - l
                # If position is NOT a multiple of 2^j, mask it
                if relative_pos % step != 0:
                    # Ensure we don't go out of bounds
                    if pos < L:
                        masked_input[batch_idx, pos] = mask_id
                        masked_indices[batch_idx, pos] = True
                        p_mask[batch_idx, pos] = mask_prob if mask_prob > 0 else 1.0
            
            l = r
    
    # Ensure p_mask doesn't have zeros (would cause division by zero)
    p_mask = torch.clamp(p_mask, min=1e-8)
    
    return masked_input, masked_indices, p_mask


def compute_bisection_sampling_aware_loss(
    input_ids,
    denoiser,
    question_length,
    mask_id,
    block_size,
    enable_shift,
    share_steps,
    self_align,
    feature_align,
    self_step,
    eos_id,
):
    """
    Compute loss using bisection sampling-aware training strategy.
    """
    # Use LLaDA's mask_id
    mask_id = 126336
    
    B, L = input_ids.shape
    
    # Apply bisection sampling-aware masking
    noisy_batch, masked_indices, p_mask = bisection_sampling_aware_mask(
        input_ids, 
        prompt_lengths=question_length, 
        mask_id=mask_id, 
        block_size=block_size,
        eos_id=eos_id
    )
    
    # Preserve prompt (don't mask it)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    
    noisy_batch = noisy_batch.to(denoiser.device)
    
    # For bidirectional attention, just pass None (full attention)
    # Or use zeros for attention_bias (no masking)
    attention_mask = torch.zeros(
        [B, 1, L, L], 
        dtype=torch.float16, 
        device=denoiser.device
    )
    
    # Forward pass - use attention_bias for LLaDA model
    logits = denoiser(noisy_batch, attention_bias=attention_mask).logits
    
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                ref_logits = denoiser(
                    noisy_batch,
                    attention_bias=torch.zeros(
                        [B, 1, L, L],
                        dtype=torch.float16,
                        device=denoiser.device
                    )
                ).logits
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        
        token_loss = F.cross_entropy(
            logits[masked_indices], 
            ref_logits[masked_indices], 
            reduction='none'
        ) / p_mask[masked_indices]
    else:
        token_loss = F.cross_entropy(
            logits[masked_indices], 
            input_ids[masked_indices], 
            reduction='none'
        ) / p_mask[masked_indices]
    
    losses = {
        'loss': token_loss.mean(),
    }
    
    return losses

def compute_loss_by_config(
    input_ids,
    denoiser,
    question_length,
    mask_id,
    block_size,
    enable_shift,
    share_steps,
    self_align,
    feature_align,
    self_step,
    eos_id,
    config
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'bisection_sampling_aware':
        return compute_bisection_sampling_aware_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'llada':
        return compute_llada_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'dream':
        return compute_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def compute_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    logits=denoiser(noisy_batch,attention_mask=attention_mask).logits
    logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                # denoiser.eval()
                ref_logits=denoiser(noisy_batch,attention_mask=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
                # denoiser.train()
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
def compute_normal_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    logits=denoiser(noisy_batch).logits
    logits=shift_logits(logits)
    token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
import torch
def compute_llada_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    mask_id=126336
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    # print(noisy_batch)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    # print(type(denoiser),noisy_batch.shape,attention_mask.shape)
    logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
    # logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                ref_logits=denoiser(noisy_batch,attention_bias=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                # ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 


# def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None):
#     """Bidirectional attention mask with block structure."""
#     B, seq_len = input_ids.shape
#     attn_mask = torch.full((B, 1, seq_len, seq_len), float('-inf'), dtype=torch.float32, device=device)
    
#     for i in range(B):
#         # Prompt: bidirectional attention
#         attn_mask[i, :, :, :prompt_length[i]] = 0.0
        
#         # Block division
#         num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size
        
#         for b in range(num_blocks):
#             block_start = prompt_length[i] + b * block_size
#             block_end = min(block_start + block_size, seq_len)
            
#             # Within-block bidirectional attention
#             attn_mask[i, :, block_start:block_end, block_start:block_end] = 0.0
            
#             # Cross-block bidirectional attention
#             for other_b in range(num_blocks):
#                 if other_b != b:
#                     other_start = prompt_length[i] + other_b * block_size
#                     other_end = min(other_start + block_size, seq_len)
#                     attn_mask[i, :, block_start:block_end, other_start:other_end] = 0.0
    
#     return attn_mask

def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None):
    """Bidirectional attention mask with block structure for LLaDA."""
    B, seq_len = input_ids.shape
    
    # Initialize to zeros (allow attention everywhere for bidirectional)
    attn_mask = torch.zeros((B, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    
    for i in range(B):
        # For bidirectional within blocks, we actually want all zeros (full attention)
\        pass
    
    return attn_mask


def shift_logits(logits):
    """Shift logits for next-token prediction."""
    return logits[:, :-1, :]
    
if __name__ == "__main__":
    seq_len = 10
    input_ids = torch.randint(0, 100, (2, seq_len))  # 示例输入
    block_size = 4
    prompt_length = torch.tensor([2, 4])  # 示例prompt长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_mask = build_custom_float_attention_mask(input_ids, prompt_length, block_size, device)
    print(attn_mask)