# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        if steps <= i:  # corner case
            continue

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache_and_q_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True, use_q_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        need_compute_q = (current_block_start, current_block_end, transfer_index[:, current_block_start:])

        while True:
            # print(">>>", i, need_compute_q)
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position,
                           use_q_cache=True, need_compute_q=need_compute_q).logits

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                            x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            need_compute_q = (current_block_start, current_block_end, transfer_index)
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    return x, nfe


@torch.no_grad()
def generate_coarse_to_fine(
    model,
    prompt,
    *,
    steps          = 128,
    gen_length     = 128,
    block_length   = 128,
    temperature    = 0.,
    remasking      = "low_confidence",
    mask_id        = 126336,
    threshold      = None,
    tokenizer      = None,
    debug = False
):
    """
    Coarse-to-fine masked-diffusion decoding that:

    1.  Runs one *global* forward pass every outer iteration.
    2.  Uses `get_transfer_index` with `num_transfer_tokens = block_length`
        to pick the logical block (the k highest-confidence masked tokens)
        **and** to commit their first samples.
    3.  Iteratively refines only those positions, re-using the KV cache
        for the rest of the sequence.
    """
    device = model.device
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # how many logical blocks (= outer iterations) do we need?
    assert gen_length % block_length == 0
    num_iters        = gen_length // block_length
    steps_per_iter   = steps // num_iters

    nfe = 0  # number of forward evaluations
    for outer in range(num_iters):
        if debug:
            print(f"outer: {outer}")

        # ------------------------------------------------------------------
        # 0.  GLOBAL pass – obtain fresh logits & prefix KV cache
        # ------------------------------------------------------------------
        out               = model(x, use_cache=True)
        logits            = out.logits                    # (1, L, V)
        past_key_values   = out.past_key_values           # cache-1
        nfe              += 1

        # ------------------------------------------------------------------
        # 1.  Use *existing* helper to select a logical block
        #     and do the very first token transfer inside that block.
        # ------------------------------------------------------------------
        mask_index      = (x == mask_id)                 # (1, L) bool
        if mask_index.sum() == 0:                        # nothing left?
            break

        # each batch element asks for the same quota (= block_length)
        quota = mask_index.sum(dim=1).clamp(max=block_length)

        # get_transfer_index returns both the sampled tokens (x0)
        # *and* the boolean mask of positions chosen for transfer
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        x0, block_sel = get_transfer_index(
                            logits,
                            temperature,
                            remasking,
                            mask_index,
                            x,
                            quota,
                            threshold,
                            endoftext_id=endoftext_id
        )
        # `block_sel` is our logical block (shape 1×L, bool)

        block_positions = block_sel[0].nonzero(as_tuple=False).squeeze(-1)
        transfer_schedule = get_num_transfer_tokens(block_sel, steps_per_iter)  # (1, steps_per_iter)
        inner_step = 0
        if debug:
            print(f"\tblock_sel: {block_sel}")

        # ------------------------------------------------------------------
        # 2.  Refinement loop – only run the model on the scattered block
        # ------------------------------------------------------------------
        while True:
            still_masked = (x[:, block_positions] == mask_id)
            if debug:
                print(f"\tstill: {still_masked}")
            if still_masked.sum() == 0:
                break

            if debug:
                print(f"\tblock_positions: {block_positions}")
            # x_block = x[:, block_positions]                          # shape 1×K'
            logits_block = model(
                x[:, block_positions],
                past_key_values=past_key_values,         # reuse prefix cache
                use_cache=True,
                replace_position=block_sel,
                q_positions=block_positions,
                k_positions=torch.arange(len(x[0]),  # ← full or cached
                                         device=x.device)
            ).logits
            nfe += 1

            # how many tokens to transfer *this* inner step?
            quota_step = transfer_schedule[:, inner_step] \
                         if threshold is None else None

            x0, transfer_idx = get_transfer_index(
                                   logits_block,
                                   temperature,
                                   remasking,
                                   still_masked,
                                   x[:, block_positions],
                                   quota_step,
                                   threshold)

            if debug:
                print(f"\ttransfer_idx: {transfer_idx}")
                print(f"\tx0 {x0}")
                # exit(0)

            # # The following triggers“advanced indexing” that produces a copy, not a view
            # x[:, block_positions][transfer_idx] = x0[transfer_idx]

            # 1) build the mask that lives in *x*’s coordinate system
            abs_transfer_cols = block_positions[transfer_idx.squeeze(0)]  # (K̃,)
            # 2) in-place write
            x[0, abs_transfer_cols] = x0[transfer_idx]  # batch-size is 1

            if debug:
                print(f"\tx[:, block_positions] {x[:, block_positions]}")
                print(f"\tx {x}")
            tokens = [(idx, tokenizer.decode(e)) for idx, e in enumerate(x[0])]

            if debug:
                print(f"\ttokens {tokens}")
                print("---")
            inner_step += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index,
                       x, num_transfer_tokens, threshold=None, endoftext_id=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    # # confidence = torch.where(x0 != skip_endoftext, confidence, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    # transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    # if threshold is not None:
    #     num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    # for j in range(confidence.shape[0]):
    #     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
    #     transfer_index[j, select_index] = True
    #     if threshold is not None:
    #         for k in range(1, num_transfer_tokens[j]):
    #             if confidence[j, select_index[k]] < threshold:
    #                 transfer_index[j, select_index[k]] = False

    # skip end of text id
    non_eot = mask_index & (x0 != endoftext_id)
    # print(non_eot)
    # Count per-sample how many non-EoT slots:
    counts = non_eot.sum(dim=1)
    # print(counts)
    enough = counts >= num_transfer_tokens
    # print(enough)
    # Choose either non_eot or full mask_index per sample
    mask_use = mask_index.clone()
    # for rows where enough is True, replace mask_use with non_eot
    mask_use[enough] = non_eot[enough]
    # print('---')
    # print(mask_index)
    # print(mask_use)
    # Apply mask_use to confidence
    confidence = confidence.masked_fill(~mask_use, -np.inf)
    # print(valid_conf)

    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index


@torch.no_grad()
def generate_with_finegrained_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, tokenizer=None
):
    '''
    Fine-grained KV cache decoding.
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK].
    '''
    device = model.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 初始化block_kv_cache
        if num_block == 0:
            # 全量前向，拿到初始KV cache和logits
            output = model(x, use_cache=True)
            past_key_values = output.past_key_values

            # 对当前block做第一次去噪
            mask_index = (x == mask_id)
            mask_index[:, current_block_end:] = 0
            x0, transfer_index = get_transfer_index(
                output.logits, temperature, remasking, mask_index,
                x, num_transfer_tokens[:, 0] if threshold is None else None, threshold
            )
            x[transfer_index] = x0[transfer_index]

            # 沿用 generate_with_dual_cache 的 prefix 和 suffix cache 思想
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, current_block_start:current_block_end] = 1

            # 第一轮后，下一轮需要重算KV的位置就是本轮transfer的位置
            need_compute_kv = (current_block_start, current_block_end, transfer_index)
            start_step = 1
            nfe += 1
        else:
            start_step = 0

        for step in range(start_step, steps_per_block):
            nfe += 1
            # 只调用一次model，replace_position指定本轮需要重算KV的位置
            output = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
                need_compute_kv=need_compute_kv
            )

            # 选出本轮transfer的位置
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            x0, transfer_index = get_transfer_index(
                output.logits, temperature, remasking, mask_index, x[:, current_block_start:current_block_end],
                num_transfer_tokens[:, step] if threshold is None else None, threshold
            )
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            # 下一轮replace_position = transfer_index
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, current_block_start:current_block_end] = 1
            need_compute_kv = (current_block_start, current_block_end, transfer_index)
            # 终止条件
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

    return x, nfe


def main():
    device = 'cuda'

    from model.modeling_llada import LLaDAModelLM  # avoid circular import
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
