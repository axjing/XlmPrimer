""" 
用于高效推理模型的引擎。

所有操作均围绕token序列展开：
- 用户可向引擎发送token序列
- 引擎返回下一个token

说明：
- 引擎不涉及任何分词处理，仅处理纯tokenID序列。

整体设计尽可能追求高效。
"""


import torch
import torch.nn.functional as F
import warnings
from contextlib import contextmanager
import ast
import re
from collections import deque
from src.trainer.distributed import compute_init, autodetect_device_type
from src.engine.utils_checkpoints import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def _null_context():
    # kept for API compatibility if needed elsewhere
    yield


def _safe_eval_math(expr: str):
    """Safely evaluate simple math expressions using AST.

    Supported operators: +, -, *, /, % and unary +/-. Power (**) and any calls
    or names are disallowed.
    """
    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        return None

    def _check_and_eval(n):
        if isinstance(n, ast.Expression):
            return _check_and_eval(n.body)
        if isinstance(n, ast.BinOp):
            left = _check_and_eval(n.left)
            right = _check_and_eval(n.right)
            if left is None or right is None:
                return None
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Mod):
                return left % right
            # disallow Pow, FloorDiv, Bitwise ops
            return None
        if isinstance(n, ast.UnaryOp):
            operand = _check_and_eval(n.operand)
            if operand is None:
                return None
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            return None
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            return None
        # For Python <3.8 compatibility, Num
        if isinstance(n, ast.Num):
            return n.n
        # All other node types disallowed
        return None

    result = _check_and_eval(node)
    return result


_STR_COUNT_RE = re.compile(r"^\s*(['\"])(.*)\1\.count\(\s*(['\"])(.*)\3\s*\)\s*$")


def use_calculator(expr: str):
    """Evaluate restricted expressions: simple math or '<str>'.count('<sub>')"""
    if not isinstance(expr, str):
        return None
    # Remove commas from numbers like '1,000'
    expr = expr.replace(",", "")

    # Quick reject dangerous tokens
    low = expr.lower()
    for bad in ['__', 'import', 'exec', 'eval', 'compile', 'open', 'input', 'globals', 'locals', 'getattr', 'setattr']:
        if bad in low:
            return None

    # Math expression path: allow digits, operators and parentheses
    if re.fullmatch(r"[0-9+\-*/ %.()]+", expr):
        if "**" in expr:
            return None
        return _safe_eval_math(expr)

    # String.count() path
    m = _STR_COUNT_RE.match(expr)
    if m:
        hay = m.group(2)
        needle = m.group(4)
        try:
            return hay.count(needle)
        except Exception:
            return None

    return None

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Lazy allocation for cache tensors to avoid huge upfront allocations.
        # `seq_len` is treated as a maximum hint; actual buffers are allocated
        # only when needed via `_ensure_capacity` or during prefill.
        self._device = device
        self._dtype = dtype
        self._alloc_len = 0  # current allocated time dimension
        self.k_cache = None
        self.v_cache = None
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

        # Heuristic: allocate immediately in the common prefill case when
        # batch_size==1 and seq_len is small, otherwise postpone allocation.
        try:
            hint = int(seq_len)
        except Exception:
            hint = 0
        if batch_size == 1 and hint > 0 and hint <= 4096:
            self._ensure_capacity(hint)

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def _ensure_capacity(self, required_len: int):
        """Ensure k_cache/v_cache have capacity for `required_len` sequence length.

        If current allocation is smaller, allocate new tensors with the same
        device/dtype and copy existing contents (if any).
        """
        if required_len <= self._alloc_len:
            return
        # clamp to max_seq_len if provided (>0)
        target = required_len
        if hasattr(self, 'max_seq_len') and self.max_seq_len is not None and self.max_seq_len > 0:
            target = min(target, self.max_seq_len)

        new_shape = (self.n_layers, self.batch_size, target, self.n_heads, self.head_dim)
        # allocate new buffers
        new_k = torch.empty(new_shape, device=self._device, dtype=self._dtype)
        new_v = torch.empty(new_shape, device=self._device, dtype=self._dtype)
        # initialize to zero for safety
        new_k.zero_()
        new_v.zero_()
        # copy old data if present
        if self.k_cache is not None:
            old_len = self._alloc_len
            new_k[:, :, :old_len, :, :].copy_(self.k_cache)
            new_v[:, :, :old_len, :, :].copy_(self.v_cache)

        self.k_cache = new_k
        self.v_cache = new_v
        self._alloc_len = target

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        if self.k_cache is None or self.v_cache is None:
            return None, None
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        # ensure we have room for the advance
        max_pos = int(self.cache_seqlens.max().item() + num_tokens)
        if max_pos > self._alloc_len:
            self._ensure_capacity(max_pos)
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        other_pos = other.get_pos()
        # ensure capacity for other_pos
        if other_pos > 0:
            self._ensure_capacity(other_pos)
            # copy the underlying tensors
            if other.k_cache is not None:
                self.k_cache[:, :, :other_pos, :, :].copy_(other.k_cache[:, :, :other_pos, :, :])
            if other.v_cache is not None:
                self.v_cache[:, :, :other_pos, :, :].copy_(other.v_cache[:, :, :other_pos, :, :])
            self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
