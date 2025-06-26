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
from model.modeling_llada import LLaDAModelLM

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
             remasking='low_confidence', mask_id=126336, threshold=None):
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
             remasking='low_confidence', mask_id=126336, threshold=None):
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
            remasking='low_confidence', mask_id=126336, threshold=None):
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


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
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

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
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
    remasking='low_confidence', mask_id=126336, threshold=None
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

    # 用于追踪每个block内每个位置的KV状态
    # 0: 未去噪, 1: 刚去噪, 2: 已稳定
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 追踪每个位置的状态
        position_status = torch.zeros((x.shape[0], block_length), dtype=torch.int, device=device)
        # 记录每个位置最晚被transfer的轮数
        last_transfer_step = torch.full((x.shape[0], block_length), -1, dtype=torch.int, device=device)

        # 1. 初始化block_kv_cache
        if num_block == 0:
            # 全量前向，拿到初始KV cache和logits
            output = model(x, use_cache=True)
            past_key_values = output.past_key_values
            logits = output.logits

            # 对当前block做第一次去噪
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            x0, transfer_index = get_transfer_index(
                logits[:, current_block_start:current_block_end, :], temperature, remasking, mask_index,
                x[:, current_block_start:current_block_end], num_transfer_tokens[:, 0] if threshold is None else None,
                threshold
            )
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

            # 初始化block_kv_cache
            block_kv_cache = []
            for layer_kv in past_key_values:
                k, v = layer_kv
                block_kv_cache.append((
                    k[:, :, current_block_start:current_block_end, :],
                    v[:, :, current_block_start:current_block_end, :]
                ))

            # 更新状态
            position_status[transfer_index] = 1
            last_transfer_step[transfer_index] = 0  # step=0

            # 其余逻辑进入多轮去噪循环（step从1开始）
            start_step = 1
            nfe += 1
        else:
            # 复用上一块的cache
            block_kv_cache = []
            for layer_kv in past_key_values:
                k, v = layer_kv
                block_kv_cache.append((
                    k[:, :, current_block_start:current_block_end, :],
                    v[:, :, current_block_start:current_block_end, :]
                ))
            start_step = 0

        # 2. block内多轮去噪
        for step in range(start_step, steps_per_block):
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # 只对当前block做mask
            logits = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=None  # 这里我们自己管理block_kv_cache
            ).logits

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index, x[:, current_block_start:current_block_end],
                num_transfer_tokens[:, step] if threshold is None else None, threshold
            )
            # 更新x
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

            # 3. 更新状态
            # transfer_index==True的位置: 1（刚去噪），并记录step
            # transfer_index==False且上轮为1: 2（已稳定）
            # transfer_index==False且上轮为0: 0（未去噪）
            position_status[transfer_index] = 1
            position_status[(position_status == 1) & (~transfer_index)] = 2
            last_transfer_step[transfer_index] = step

            # 4. 只对transfer_index==True的位置重算KV，其余复用
            # 先准备输入
            if transfer_index.any():
                # 只对transfer位置做前向
                indices = transfer_index.nonzero(as_tuple=True)
                # 取出这些位置的token
                x_transfer = x[:, current_block_start:current_block_end].clone()
                # 只保留transfer位置，其余设为mask_id
                x_transfer[~transfer_index] = mask_id
                # 重新计算这些位置的KV
                output = model(
                    x_transfer,
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=None
                )
                new_kv = output.past_key_values
                # 更新block_kv_cache
                for l, (k, v) in enumerate(new_kv):
                    # 只更新transfer位置
                    block_kv_cache[l][0][:, :, indices[1], :] = k[:, :, indices[1], :]
                    block_kv_cache[l][1][:, :, indices[1], :] = v[:, :, indices[1], :]

            # 5. 组装完整past_key_values
            # prompt部分
            new_past_key_values = []
            for l, (k, v) in enumerate(past_key_values):
                # prompt部分
                k_prompt = k[:, :, :current_block_start, :]
                v_prompt = v[:, :, :current_block_start, :]
                # 当前block部分
                k_block = block_kv_cache[l][0]
                v_block = block_kv_cache[l][1]
                # 后缀部分
                k_suffix = k[:, :, current_block_end:, :]
                v_suffix = v[:, :, current_block_end:, :]
                # 拼接
                k_full = torch.cat([k_prompt, k_block, k_suffix], dim=2)
                v_full = torch.cat([v_prompt, v_block, v_suffix], dim=2)
                new_past_key_values.append((k_full, v_full))
            past_key_values = new_past_key_values

            # 6. 终止条件
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

    return x, nfe

def main():
    device = 'cuda'

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
