import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
from transformers import LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer, LlamaConfig, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import (
    BaseModelOutputWithPast
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import List, Optional, Tuple, Union
from transformers.processing_utils import Unpack
# from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward
import math
import inspect

class LinkedListCache(Cache):
    def __init__(self, layers = 32):
        super().__init__()
        self.k_list_list = [[] for _ in range(layers)]
        self.v_list_list = [[] for _ in range(layers)]
    def get_seq_length(self, layer_idx: Optional[int] = 0):
        if len(self.k_list_list[layer_idx]) == 0:
            return 0
        slen = self.k_list_list[layer_idx][0].shape[-2]
        return sum([saved_k.shape[-2] for saved_k in self.k_list_list[layer_idx]])# len(self.k_list_list[layer_idx]) * slen

    
class KVEfficientAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        *inputs
    ) -> torch.Tensor:
        list_length = inputs[0]

        query = inputs[1]

        # 3) Parse keys (a slice of length `list_length`)
        #    keys will be a tuple of Tensors
        key_list = inputs[2 : 2 + list_length]

        # 4) Parse values (the next `list_length` items)
        value_list = inputs[2 + list_length : 2 + 2*list_length]

        # 5) Next arguments: attention_mask, scaling, dropout, n_rep, training
        #    We'll assume you pass them in this order.
        attention_mask = inputs[2 + 2*list_length]
        scaling        = inputs[3 + 2*list_length]   
        dropout        = inputs[4 + 2*list_length]
        n_rep          = inputs[5 + 2*list_length]
        training       = inputs[6 + 2*list_length]

        key = torch.cat(key_list, dim=-2)
        value = torch.cat(value_list, dim=-2)
        batch, num_key_value_heads, slen, head_dim = key.shape
        num_attention_heads = num_key_value_heads * n_rep

        # Expand key and value to match num_attention_heads if repetition is needed
        if n_rep > 1:
            # Add a dimension for repetition, expand, and reshape
            key_ext = key[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim).reshape(
                batch, num_attention_heads, slen, head_dim
            )
            value_ext = value[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim).reshape(
                batch, num_attention_heads, slen, head_dim
            )
        # If n_rep == 1, no expansion is needed
        else:
            key_ext = key
            value_ext = value

        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(query, key_ext.transpose(2, 3)) * scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_ext.shape[-2]]
            attn_scores = attn_scores + causal_mask

        # Compute attention weights via softmax
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        dropout_seed = None
        # Apply dropout to attention weights if applicable
        if training and dropout > 0.0:
            prob = 1.0 - dropout
            dropout_seed = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()
            torch.manual_seed(dropout_seed)
            dropout_mask = (torch.rand_like(attn_weights) < prob)
            attn_weights_dropped = attn_weights * dropout_mask / prob
        else:
            attn_weights_dropped = attn_weights

        # Compute attention output: A @ V
        attn_output = torch.matmul(attn_weights_dropped, value_ext)

        # Transpose to match expected output shape
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(query, attn_weights, *key_list, *value_list)
        ctx.attention_mask = attention_mask
        ctx.scaling = scaling
        ctx.dropout = dropout
        ctx.n_rep = n_rep
        ctx.training = training
        ctx.batch = batch
        ctx.num_attention_heads = num_attention_heads
        ctx.slen = slen
        ctx.head_dim = head_dim
        ctx.dropout_seed = dropout_seed
        ctx.list_length = list_length
        return attn_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for memory-efficient attention with integrated key/value repetition.

        Args:
            grad_output: Gradient of the output, shape (batch, seqlen, num_attention_heads, head_dim)

        Returns:
            Tuple of gradients for each input (d_query, d_key, d_value, None, None, None, None, None)
        """
        # Retrieve saved tensors and parameters
        # query, key, value, attn_weights = ctx.saved_tensors
        saved_tensors = ctx.saved_tensors
        # query, key_list, value_list = saved_tensors[0], saved_tensors[1:1+ctx.list_length], saved_tensors[1+ctx.list_length:1+2*ctx.list_length]
        query, attn_weights, key_list, value_list = saved_tensors[0], saved_tensors[1], saved_tensors[2:2+ctx.list_length], saved_tensors[2+ctx.list_length:2+2*ctx.list_length]
        attention_mask = ctx.attention_mask
        scaling = ctx.scaling
        dropout = ctx.dropout
        n_rep = ctx.n_rep
        training = ctx.training
        batch = ctx.batch
        num_attention_heads = ctx.num_attention_heads
        slen = ctx.slen
        head_dim = ctx.head_dim
        num_key_value_heads = num_attention_heads // n_rep
        key = torch.cat(key_list, dim=-2)
        value = torch.cat(value_list, dim=-2)
        if n_rep > 1:
            # Add a dimension for repetition, expand, and reshape
            key = key[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim).reshape(
                batch, num_attention_heads, slen, head_dim
            )
            value = value[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim).reshape(
                batch, num_attention_heads, slen, head_dim
            )
        # If n_rep == 1, no expansion is needed
        else:
            key = key
            value = value

        # attn_scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        # attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        # Adjust grad_output to match pre-transpose shape
        dO = grad_output.transpose(1, 2).contiguous()  # (batch, num_attention_heads, seqlen, head_dim)
        
        attn_weights = attn_weights.to(dO.dtype)
        key = key.to(dO.dtype)
        value = value.to(dO.dtype)
        query = query.to(dO.dtype)
        # Okay this is incorrect
        # Recompute dropped attention weights if dropout was applied
        if training and dropout > 0.0:
            prob = 1.0 - dropout
            dropout_seed = ctx.dropout_seed
            torch.manual_seed(dropout_seed)
            dropout_mask = (torch.rand_like(attn_weights) < prob)
            attn_weights_dropped = attn_weights * dropout_mask / prob
            
        else:
            attn_weights_dropped = attn_weights
        # Gradient w.r.t. value: dV = A^T @ dO
        d_value = torch.matmul(attn_weights_dropped.transpose(2, 3), dO)

        # Gradient w.r.t. attn_weights_dropped: dA_dropped = dO @ V^T
        d_attn_weights_dropped = torch.matmul(dO, value.transpose(2, 3))

        # Gradient w.r.t. attn_weights (before dropout)
        # d_attn_weights = d_attn_weights_dropped  # Simplified; assumes dropout gradient passes through
        # element-wise multiplication
        if training and dropout > 0.0:
            d_attn_weights = d_attn_weights_dropped * dropout_mask / prob
        else:
            d_attn_weights = d_attn_weights_dropped

        # Gradient through softmax: dS = A * (dA - sum(A * dA, dim=-1))
        sum_d_A = (attn_weights * d_attn_weights).sum(dim=-1, keepdim=True)
        dS = attn_weights * (d_attn_weights - sum_d_A)

        # Gradients w.r.t. query and key
        d_query = torch.matmul(dS, key) * scaling  # dQ = dS @ K * scaling
        d_key = torch.matmul(dS.transpose(2, 3), query) * scaling  # dK = dS^T @ Q * scaling

        # Accumulate gradients for original key and value if repeated
        if n_rep > 1:
            num_key_value_heads = num_attention_heads // n_rep
            d_key = d_key.view(batch, num_key_value_heads, n_rep, slen, head_dim).sum(dim=2)
            d_value = d_value.view(batch, num_key_value_heads, n_rep, slen, head_dim).sum(dim=2)
        d_key_tuple = torch.split(d_key, [k.shape[-2] for k in key_list], dim=-2)
        d_value_tuple = torch.split(d_value, [v.shape[-2] for v in value_list], dim=-2)
        # Return gradients, with None for inputs that don’t require gradients

        return (None,) + (d_query,) + d_key_tuple + d_value_tuple + (None, None, None, None, None)


def get_model(improved, bridges = [], dataType=torch.float32, r = 16, model_name = "meta-llama/Llama-3.1-8B", multiplier = 100):
    alpha = 0.01
    index_to_order_in_source = []
    count_appearance_source = {}
    for i in range(32):
        count_appearance_source[i] = 0
    for i in range(len(bridges)):    
        source, destination = bridges[i]
        index_to_order_in_source.append(count_appearance_source[source])
        count_appearance_source[source] += 1
    # print(bridges)
    # print(index_to_order_in_source)
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict_in_generate=True,
        output_hidden_states=False,
        attn_implementation="eager",
        torch_dtype=dataType
    )
    dataType = torch.float32
    # source_code = inspect.getsource(model.model.forward)
    # ffn_source_code = inspect.getsource(model.model.layers[0].mlp.forward)
    # att_source_code = inspect.getsource(model.model.layers[0].self_attn.forward)
    # with open('ffn_forward_function.txt', 'w') as file:
    #     file.write(ffn_source_code)
    # with open('forward_function.txt', 'w') as file:
    #     file.write(source_code)
    # with open('att_forward_function.txt', 'w') as file:
    #     file.write(att_source_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    #wte = model.transformer.tokens_embedg
    if improved:
        class LlamaAttentionLora(LlamaAttention):

            def __init__(self, config: LlamaConfig, layer_idx: int):
                super().__init__(config, layer_idx)
                self.q_projA_lora = nn.Linear(
                    config.hidden_size, r, dtype = dataType, bias=config.attention_bias
                )
                self.q_projB_lora = nn.Linear(
                    r, config.num_attention_heads * self.head_dim, dtype = dataType, bias=config.attention_bias
                )
                self.k_projA_lora = nn.Linear(
                    config.hidden_size, r, dtype = dataType, bias=config.attention_bias
                )
                self.k_projB_lora = nn.Linear(
                    r, config.num_key_value_heads * self.head_dim, dtype = dataType, bias=config.attention_bias
                )
                self.v_projA_lora = nn.Linear(
                    config.hidden_size, r, dtype = dataType, bias=config.attention_bias
                )
                self.v_projB_lora = nn.Linear(
                    r, config.num_key_value_heads * self.head_dim, dtype = dataType, bias=config.attention_bias
                )
                self.o_projA_lora = nn.Linear(
                    config.num_attention_heads * self.head_dim, r, dtype = dataType, bias=config.attention_bias
                )
                self.o_projB_lora = nn.Linear(
                    r, config.hidden_size, dtype = dataType, bias=config.attention_bias
                )
            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
                **kwargs,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                # with torch.no_grad():
                for _ in range(1):
                    if output_attentions:
                        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
                        logger.warning_once(
                            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                        )
                        return super().forward(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )

                    bsz, q_len, _ = hidden_states.size()

                    query_states = self.q_proj(hidden_states) + alpha * self.q_projB_lora(self.q_projA_lora(hidden_states))
                    key_states = self.k_proj(hidden_states) + alpha * self.k_projB_lora(self.k_projA_lora(hidden_states))
                    value_states = self.v_proj(hidden_states) + alpha * self.v_projB_lora(self.v_projA_lora(hidden_states))

                    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                    if position_embeddings is None:
                        logger.warning_once(
                            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                            "removed and `position_embeddings` will be mandatory."
                        )
                        cos, sin = self.rotary_emb(value_states, position_ids)
                    else:
                        cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    
                    if past_key_value is not None:
                        # sin and cos are specific to RoPE models; cache_position needed for the static cache
                        past_key_value.k_list_list[self.layer_idx].append(key_states)
                        past_key_value.v_list_list[self.layer_idx].append(value_states)
                        key_list, value_list = past_key_value.k_list_list[self.layer_idx], past_key_value.v_list_list[self.layer_idx]

                    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
                    # Reference: https://github.com/pytorch/pytorch/issues/112577.

                    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
                    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
                    d_k = self.head_dim
                    l = [len(key_list), query_states] + key_list + value_list + [attention_mask, 1.0/math.sqrt(d_k), self.attention_dropout, self.num_key_value_groups, self.training]
                    attn_output = KVEfficientAttention.apply(*l)
                    attn_output = attn_output.view(bsz, q_len, -1)

                    attn_output = self.o_proj(attn_output) + alpha * self.o_projB_lora(self.o_projA_lora(attn_output))

                    return attn_output, None, past_key_value
                
        class LlamaMLPLora(LlamaMLP):
            def __init__(self, config, r):
                super().__init__(config)
                self.r = r
                self.gate_projA_lora = nn.Linear(self.hidden_size, r, dtype = dataType, bias=config.mlp_bias)
                self.gate_projB_lora = nn.Linear(r, self.intermediate_size, dtype = dataType, bias=config.mlp_bias)
                self.up_projA_lora = nn.Linear(self.hidden_size, r, dtype = dataType, bias=config.mlp_bias)
                self.up_projB_lora = nn.Linear(r, self.intermediate_size, dtype = dataType, bias=config.mlp_bias)
                self.down_projA_lora = nn.Linear(self.intermediate_size, r, dtype = dataType, bias=config.mlp_bias)
                self.down_projB_lora = nn.Linear(r, self.hidden_size, dtype = dataType, bias=config.mlp_bias)
            
            def forward(self, x):
                #with torch.no_grad():
                for _ in range(1):
                    if self.config.pretraining_tp > 1:
                        # error out we don't support pretraining_tp > 1
                        raise NotImplementedError
                    else:
                        up_proj = self.up_proj(x) + alpha * self.up_projB_lora(self.up_projA_lora(x))
                        gate_proj = self.gate_proj(x) + alpha * self.gate_projB_lora(self.gate_projA_lora(x))
                        y = self.act_fn(gate_proj) * up_proj
                        down_proj = self.down_proj(y) + alpha * self.down_projB_lora(self.down_projA_lora(y))
                    return down_proj


        class LlamaDecoderLayerLora(LlamaDecoderLayer):
            def __init__(self, config: LlamaConfig, layer_idx: int):
                super().__init__(config, layer_idx)
                self.mlp = LlamaMLPLora(config, r)
                self.bridge_down_proj_layers = nn.ModuleList(
                    nn.Linear(config.hidden_size, r, dtype = dataType) for _ in range(count_appearance_source[layer_idx])
                )
                self.self_attn = LlamaAttentionLora(config=config, layer_idx=layer_idx)
        
        multiplier = multiplier * alpha
        class LlamaModelModified(LlamaModel):
            def __init__(self, config):
                super().__init__(config)
                self.layers = nn.ModuleList(
                            [LlamaDecoderLayerLora(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
                        )
                self.bridge_up_proj_layers = nn.ModuleList(
                    nn.Linear(r, config.hidden_size, dtype = dataType) for _ in range(len(bridges))
                )
                for i in range(len(bridges)):
                    nn.init.zeros_(self.bridge_up_proj_layers[i].bias)
                    nn.init.zeros_(self.bridge_up_proj_layers[i].weight)
                # Initialize biases to zero
                self.connections = [0 for _ in range(len(bridges))]
                for i in range(len(bridges)):
                    self.connections[i] = None


            def forward(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
            ) -> Union[Tuple, BaseModelOutputWithPast]:
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                use_cache = use_cache if use_cache is not None else self.config.use_cache
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                if (input_ids is None) ^ (inputs_embeds is not None):
                    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

                if self.gradient_checkpointing and self.training and use_cache:
                    use_cache = False

                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)

                # kept for BC (non `Cache` `past_key_values` inputs)
                return_legacy_cache = False
                if use_cache and not isinstance(past_key_values, Cache):
                    return_legacy_cache = True
                    if past_key_values is None:
                        past_key_values = DynamicCache()
                    else:
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

                if cache_position is None:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)

                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
                hidden_states = inputs_embeds

                # create position embeddings to be shared across the decoder layers
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # decoder layers
                all_hidden_states = () if output_hidden_states else None
                all_self_attns = () if output_attentions else None
                next_decoder_cache = None

                for i, decoder_layer in enumerate(self.layers):
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            causal_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                            cache_position,
                            position_embeddings,
                        )
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )

                    hidden_states = layer_outputs[0]
                    for index, pair in zip(range(len(bridges)), bridges):
                        source, destination = pair
                        if i == destination:
                            if not (self.connections[index] == None):
                                linear_device = next(self.bridge_up_proj_layers[index].parameters()).device
                                self.connections[index].to(linear_device)
                                hidden_states = hidden_states + self.bridge_up_proj_layers[index](self.connections[index]* multiplier).to(hidden_states.device)
                    for index, pair in zip(range(len(bridges)), bridges):
                        if i == source:
                            self.connections[index] = self.layers[source].bridge_down_proj_layers[index_to_order_in_source[index]](hidden_states)

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

                hidden_states = self.norm(hidden_states)

                # add hidden states from the last decoder layer
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                next_cache = next_decoder_cache if use_cache else None
                if return_legacy_cache:
                    next_cache = next_cache.to_legacy_cache()

                if not return_dict:
                    return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=next_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )

        class LlamaForCausalLMModified(LlamaForCausalLM):
            _tied_weights_keys = ["lm_head.weight"]
        
            def __init__(self, config):
                super().__init__(config)
                self.model = LlamaModelModified(config)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

                # Initialize weights and apply final processing
                self.post_init()
                # also zero out lora weights
                for name, param in self.named_parameters():
                    if "A_lora" in name or "bridge_up_proj_layers" in name:
                        nn.init.zeros_(param)

        llamamodel = model.model
        model = LlamaForCausalLMModified(model.config)
        model.enable_input_require_grads()
    return model, tokenizer