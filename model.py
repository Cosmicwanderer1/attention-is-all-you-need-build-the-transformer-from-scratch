"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    
    token_to_id = {}
    for tok in specials:
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    
    for sent in sentences:
        for tok in sent.split():
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    return {idx: tok for tok, idx in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    unk_id = token_to_id[unk_token]
    return [token_to_id.get(tok, unk_id) for tok in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list

    return [id_to_token[i] for i in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id]*(max_len - len(ids))

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.as_tensor(padded_sequences,dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch
import math

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    even_idx = torch.arange(0, d_model, 2, dtype=torch.float32)
    return torch.exp(even_idx * -(math.log(10000.0) / d_model))

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe = pe.clone()

    angles = position * div_term
    pe[:, 0::2] = torch.sin(angles).to(pe.dtype)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe = pe.clone()
    angles = position * div_term

    pe[:, 1::2] = torch.cos(angles).to(pe.dtype)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)

    div_term = compute_positional_div_term(d_model)

    position = build_position_index_column(max_len)

    pe = fill_even_indices_with_sin(pe, position, div_term)
    
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    L = embedded_batch.size(1)
    return embedded_batch + positional_encoding[:L]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    return (token_ids!=pad_id).unsqueeze(1).unsqueeze(1)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None, None, :, :]

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return torch.matmul(query, key.transpose(-2, -1))

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    scores = scores/math.sqrt(d_k)
    return scores

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    w = torch.softmax(masked_scores, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)   # 把 NaN 替换成 0
    return w

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights,value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    scores = compute_raw_attention_scores(query, key)

    d_k = query.shape[-1]

    scores = scale_attention_scores(scores, d_k)

    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)
    
    att_weights = softmax_attention_weights(scores)

    context = apply_attention_weights_to_values(att_weights, value)

    return context, att_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.shape
    d_k = d_model // num_heads

    return tensor.view(B, L, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.permute(0, 2, 1, 3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    multi_head_tensor = multi_head_tensor.permute(0, 2, 1, 3)

    B, L, H, d_k = multi_head_tensor.shape

    d_model = H * d_k

    return multi_head_tensor.reshape(B, L, H * d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    out=torch.matmul(x, weight.transpose(-2, -1))

    if bias is not None:
        out = out + bias
    
    return out

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    
    Q = apply_linear_projection(x, w_q, b_q)
    K = apply_linear_projection(x, w_k, b_k)
    V = apply_linear_projection(x, w_v, b_v)

    return Q, K, V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    
    q = split_last_dim_into_heads(q, num_heads)
    q_h = transpose_heads_before_sequence(q)
        
    k = split_last_dim_into_heads(k, num_heads)
    k_h = transpose_heads_before_sequence(k)

    v = split_last_dim_into_heads(v, num_heads)
    v_h = transpose_heads_before_sequence(v)

    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    

    context, weights = scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)
    return context, weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    
    context = merge_heads_back_to_model_dim(context)

    out = apply_linear_projection(context, w_o, b_o)

    return out

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.


    Q = apply_linear_projection(query, w_q, None)
    K = apply_linear_projection(key, w_k, None)
    V = apply_linear_projection(value, w_v, None)


    Q_h, K_h, V_h = split_qkv_into_heads(Q, K, V, num_heads)


    context, att_weights = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask)

    output = merge_heads_and_project_output(context, w_o, None)

    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    return torch.relu(torch.matmul(x,w1) + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    out = torch.matmul(hidden, w2) + b2
    return out

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)

    ffn_x = apply_ffn_second_linear(hidden, w2, b2)
    return ffn_x

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    
    mean, var = compute_layer_norm_mean_and_variance(x)

    x_hat = (x - mean) / torch.sqrt(var + eps)

    y = gamma * x_hat + beta

    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    
    x = residual_input + sublayer_output

    return normalize_and_scale_with_gamma_beta(x, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    
    mask = keep_mask.to(dtype=x.dtype)
    return x * mask * (1.0 / keep_prob)

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    
    sublayer_out = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, mask=src_mask)

    output = apply_residual_add_and_norm(x, sublayer_out, gamma, beta, eps=1e-05)


    return output

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    
    x_hat = position_wise_feed_forward_network(x, w1, b1, w2, b2)

    output = apply_residual_add_and_norm(x, x_hat, gamma, beta, eps=1e-5)

    return output

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    

    att_out = encoder_layer_self_attention_sublayer(
        x,
        layer_params["w_q"],
        layer_params["w_k"],
        layer_params["w_v"],
        layer_params["w_o"],
        layer_params["attn_gamma"],
        layer_params["attn_beta"],
        num_heads,
        src_mask=src_mask
    )

    ffn_out = encoder_layer_feed_forward_sublayer(
        att_out,
        w1 = layer_params["w1"],
        b1 = layer_params["b1"],
        w2 = layer_params["w2"],
        b2 = layer_params["b2"],
        gamma = layer_params["ffn_gamma"],
        beta = layer_params["ffn_beta"]
    )

    return ffn_out

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    
    hidden = x
    for lay_params in encoder_layer_params_list:
        hidden = assemble_encoder_layer(hidden, lay_params, num_heads, src_mask)
    
    return hidden

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    att_out = assemble_multi_head_attention_forward(
        y, y, y, w_q, w_k, w_v, w_o, num_heads, mask=tgt_mask)


    output = apply_residual_add_and_norm(y, att_out, gamma, beta, eps=1e-05)

    return output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]

    attn_out = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, mask=src_mask)

    output = apply_residual_add_and_norm(y, attn_out, gamma, beta, eps=1e-05)

    return output

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    
    y_hat = position_wise_feed_forward_network(y, w1, b1, w2, b2)

    output = apply_residual_add_and_norm(y, y_hat, gamma, beta)

    return output

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN.

    layer_params keys (all torch tensors):
      masked self-attention : w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta
      cross-attention       : w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta
      feed-forward          : w1, b1, w2, b2, ffn_gamma, ffn_beta
    """
    # TODO: chain the three decoder sublayers using params from layer_params.
     # 1. 掩码自注意力子层：Q/K/V 都来自 y，用 tgt_mask（因果 + padding）
    y = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params["w_q_self"],
        layer_params["w_k_self"],
        layer_params["w_v_self"],
        layer_params["w_o_self"],
        layer_params["self_gamma"],
        layer_params["self_beta"],
        num_heads,
        tgt_mask=tgt_mask,
    )

    # 2. 交叉注意力子层：Q 来自 y，K/V 来自 encoder_output，用 src_mask
    y = decoder_layer_cross_attention_sublayer(
        y,
        encoder_output,
        layer_params["w_q_cross"],
        layer_params["w_k_cross"],
        layer_params["w_v_cross"],
        layer_params["w_o_cross"],
        layer_params["cross_gamma"],
        layer_params["cross_beta"],
        num_heads,
        src_mask=src_mask,
    )

    # 3. FFN 子层：无 mask
    y = decoder_layer_feed_forward_sublayer(
        y,
        layer_params["w1"],
        layer_params["b1"],
        layer_params["w2"],
        layer_params["b2"],
        layer_params["ffn_gamma"],
        layer_params["ffn_beta"],
    )

    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    
    hidden = y

    for layer_params in decoder_layer_params_list:
        hidden = assemble_decoder_layer(hidden, encoder_output, layer_params, num_heads, src_mask=src_mask, tgt_mask=tgt_mask)


    return hidden

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    
    output = apply_linear_projection(decoder_output, output_projection_weight, bias=output_projection_bias)

    return output

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    
    return token_embedding_weight.transpose(-2, -1)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # model_params keys: src_embedding, tgt_embedding, encoder_layers, decoder_layers, output_projection
    # TODO: embed src with src_embedding and tgt with tgt_embedding, add PE, build masks,
    #       run encoder/decoder, project to log probs.
    
    # 1. 取参数
    src_embedding = model_params["src_embedding"]       # (V, d_model)
    tgt_embedding = model_params["tgt_embedding"]       # (V, d_model)
    encoder_layers = model_params["encoder_layers"]     # list of dicts
    decoder_layers = model_params["decoder_layers"]     # list of dicts
    output_projection = model_params["output_projection"]  # (V, d_model)

    # 2. d_model 从 embedding 宽度读取
    d_model = src_embedding.shape[-1]

    B, S = src_ids.shape
    _, T = tgt_ids.shape


    # 3. 源端：embedding lookup → 缩放 → 加位置编码
    src_emb = src_embedding[src_ids]
    src_emb = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    src_pe = build_sinusoidal_positional_encoding(S, d_model)
    src_emb = add_positional_encoding_to_embeddings(src_emb, src_pe)

    # 4. 目标端：embedding lookup → 缩放 → 加位置编码
    tgt_emb = tgt_embedding[tgt_ids]
    tgt_emb = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)
    tgt_pe = build_sinusoidal_positional_encoding(T, d_model)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, tgt_pe)

    
    # 5. 构建 masks
    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(T)
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, causal_mask)

    # 6. encoder stack
    encoder_output = stack_encoder_layers(src_emb, encoder_layers, num_heads, src_mask)

    # 7. decoder stack
    decoder_output = stack_decoder_layers(tgt_emb, encoder_output, decoder_layers, num_heads, src_mask, tgt_mask)

    # 8. 输出投影 + log softmax
    logits = apply_final_output_projection(decoder_output, output_projection,None)

    log_prob = apply_log_softmax_over_vocab(logits)

    return log_prob

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    
    def rand(shape):
        t = torch.randn(shape, dtype=torch.float32) * 0.02
        t.requires_grad_(True)
        return t
      

    def zeros(shape):
        return torch.zeros(shape, dtype=torch.float32, requires_grad=True)

    def ones(shape):
        return torch.ones(shape, dtype=torch.float32, requires_grad=True)

    return {
        # 自注意力投影：随机权重，无 bias
        "w_q": rand((d_model, d_model)),
        "w_k": rand((d_model, d_model)),
        "w_v": rand((d_model, d_model)),
        "w_o": rand((d_model, d_model)),

        # FFN：权重随机，偏置为 0
        "w1": rand((d_model, d_ff)),
        "b1": zeros((d_ff,)),
        "w2": rand((d_ff, d_model)),
        "b2": zeros((d_model,)),

        # 两个 LayerNorm：gamma=1, beta=0，不随机
        "attn_gamma": ones((d_model,)),
        "attn_beta": zeros((d_model,)),
        "ffn_gamma": ones((d_model,)),
        "ffn_beta": zeros((d_model,)),
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    
    def rand(shape):
        t = torch.randn(shape, dtype=torch.float32) * 0.02
        t.requires_grad_(True)
        return t
      

    def zeros(shape):
        return torch.zeros(shape, dtype=torch.float32, requires_grad=True)

    def ones(shape):
        return torch.ones(shape, dtype=torch.float32, requires_grad=True)

    return {
        # 自注意力投影：随机权重，无 bias
        "w_q_self": rand((d_model, d_model)),
        "w_k_self": rand((d_model, d_model)),
        "w_v_self": rand((d_model, d_model)),
        "w_o_self": rand((d_model, d_model)),

        # 交叉注意力投影：随机权重，无 bias
        "w_q_cross": rand((d_model, d_model)),
        "w_k_cross": rand((d_model, d_model)),
        "w_v_cross": rand((d_model, d_model)),
        "w_o_cross": rand((d_model, d_model)),

        # FFN：权重随机，偏置为 0
        "w1": rand((d_model, d_ff)),
        "b1": zeros((d_ff,)),
        "w2": rand((d_ff, d_model)),
        "b2": zeros((d_model,)),

        # 两个 LayerNorm 一个crossNorm：gamma=1, beta=0，不随机
        "self_gamma": ones((d_model,)),
        "self_beta": zeros((d_model,)),
        "cross_gamma": ones((d_model,)),
        "cross_beta": zeros((d_model,)),
        "ffn_gamma": ones((d_model,)),
        "ffn_beta": zeros((d_model,)),
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    
    def embedding():
        t = torch.randn(vocab_size, d_model, dtype=torch.float32) * 0.02
        t.requires_grad_(True)
        return t

    src_embedding = embedding()
    tgt_embedding = embedding()

    if tie_weights:
        output_projection = tgt_embedding          # 同一个对象
    else:
        output_projection = embedding()            # 独立对象

    return {
        "src_embedding": src_embedding,
        "tgt_embedding": tgt_embedding,
        "output_projection": output_projection,
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    
    seen = set()          # 已加入的张量 id 集合
    result = []           # 有序的输出列表

    def add_from_dict(params_dict):
        for tensor in params_dict.values():
            if not isinstance(tensor, torch.Tensor):
                continue                       # 跳过非张量（防御性）
            if not tensor.requires_grad:
                continue                       # 只收集可训练张量
            tid = id(tensor)
            if tid in seen:
                continue                       # 已见过，跳过（处理 weight tying）
            seen.add(tid)
            result.append(tensor)

    # 1. 所有 encoder 层（按顺序）
    for layer_params in encoder_layer_params:
        add_from_dict(layer_params)

    # 2. 所有 decoder 层（按顺序）
    for layer_params in decoder_layer_params:
        add_from_dict(layer_params)

    # 3. embedding / projection
    add_from_dict(embedding_params)

    return result

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    
    batch, tgt_seq = target_ids.shape

    start_col = torch.full(
        (batch, 1),
        start_token_id,
        dtype=target_ids.dtype,
        device=target_ids.device,
    )

    shifted = torch.cat([start_col, target_ids[:, :-1]], dim=1)
    return shifted

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    """
    lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
    Linear warmup until `warmup_steps`, then inverse-sqrt decay.
    """
    


    scale = d_model ** -0.5
    warmup_factor = step * (warmup_steps ** -1.5)

    decay_factor = step ** -0.5

    return float(scale * min(warmup_factor, decay_factor))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    

    if vocab_size is None:
        vocab_size = shape[-1]
    
    if vocab_size <= 2:
        raise ValueError(f"vocab_size must be > 2 for (V-2) smoothing, got {vocab_size}")

    value = epsilon / (vocab_size - 2)
    return torch.full(shape, value, dtype=torch.float32)

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 68 - compute_adam_parameter_update (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

