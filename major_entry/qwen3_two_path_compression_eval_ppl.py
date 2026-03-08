# eval_ppl.py
import os, time, json, math
from pathlib import Path
from typing import Optional, Any, Union, Dict, Tuple, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download

from major_entry.compressor import Compressor, NoneCompressor

import argparse
import json


class ResidualAdd(torch.nn.Module):
    """
    A helper module to replace the "+" operator in residual connections.
    This allows attaching hooks specifically to the residual merge step.
    """

    def forward(self, residual, hidden_states):
        # args[0] = residual (skip connection)
        # args[1] = hidden_states (processed output from Attn/MLP)
        return residual + hidden_states


def patch_qwen_with_residual_modules(model):
    """
    Monkey-patches the loaded Qwen model to use ResidualAdd modules
    instead of python '+' operators for residual connections.
    """
    # 1. Identify the DecoderLayer class dynamically to avoid import errors
    #    (Assumes model.model.layers is a ModuleList of the decoder layers)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        print("Warning: Could not find model.model.layers to patch. Skipping patch.")
        return model

    layer_sample = model.model.layers[0]
    decoder_layer_cls = layer_sample.__class__
    print(f"Patching class: {decoder_layer_cls.__name__}")

    # 2. Inject the helper modules into every existing layer instance
    for name, module in model.named_modules():
        if isinstance(module, decoder_layer_cls):
            # We add them as submodules so they register properly
            module.add_module("attn_residual_add", ResidualAdd())

    # 3. Define the custom forward function that uses these new modules
    #    We use generic type hints to match the imports available in this script
    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # --- PATCH : Use module instead of "hidden_states = residual + hidden_states" ---
        hidden_states = self.attn_residual_add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

    # 4. Apply the monkey patch to the CLASS
    #    This affects all instances of this class
    decoder_layer_cls.forward = custom_forward

    return model


# --- END PATCHING LOGIC ---


def format_bytes(n: float, binary: bool = False) -> str:
    """
    Format bytes using K/M/G/T suffix.
    binary=False -> 1K = 1000
    binary=True  -> 1Ki = 1024
    """
    if n < 0:
        return "-" + format_bytes(-n, binary)

    base = 1024 if binary else 1000
    suffixes = (
        ["B", "K", "M", "G", "T", "P"]
        if not binary
        else ["B", "Ki", "Mi", "Gi", "Ti", "Pi"]
    )

    for suf in suffixes:
        if n < base:
            return f"{n:.1f}{suf}" if suf != "B" else f"{int(n)}{suf}"
        n /= base
    return f"{n:.1f}{suffixes[-1]}"


def format_duration(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s {ms:03d}ms"


def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def load_model(
    model_id_or_path: str, dtype_str: str, load_in_8bit: bool, load_in_4bit: bool
):
    dtype = _dtype_from_str(dtype_str)
    kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}

    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=bool(load_in_8bit),
            load_in_4bit=bool(load_in_4bit),
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype

    return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)


def make_compressor(name: str) -> Compressor:
    name = name.lower()
    if name in ("none",):
        return NoneCompressor()
    raise ValueError(f"Unknown compressor: {name}")


def _resolve_compressor(spec: Union[str, "Compressor"], param_name: str) -> Compressor:
    if isinstance(spec, str):
        return make_compressor(spec)
    if isinstance(spec, Compressor):
        return spec
    raise TypeError(f"{param_name} must be str or Compressor, got: {type(spec)}")


class TrafficMeter:
    def __init__(self):
        self.stats: Dict[str, Dict[str, int]] = {}
        self.total_bytes = 0
        self.total_tokens = 0
        self.total_tx = 0

    def record(self, key: str, nbytes: int, ntokens: int):
        nb = int(nbytes)
        nt = int(ntokens)
        st = self.stats.get(key)
        if st is None:
            st = {"bytes": 0, "tokens": 0, "tx": 0}
            self.stats[key] = st
        st["bytes"] += nb
        st["tokens"] += nt
        st["tx"] += 1
        self.total_bytes += nb
        self.total_tokens += nt
        self.total_tx += 1

    def totals(self) -> Dict[str, float]:
        bpt = (
            (self.total_bytes / self.total_tokens)
            if self.total_tokens
            else float("nan")
        )
        return {
            "total_bytes": float(self.total_bytes),
            "total_tokens": float(self.total_tokens),
            "total_tx": float(self.total_tx),
            "bytes_per_token": float(bpt),
        }

    def reset(self):
        self.stats.clear()
        self.total_bytes = 0
        self.total_tokens = 0
        self.total_tx = 0


class Pipeline:
    def __init__(self, compressor: Compressor, meter: TrafficMeter):
        self.compressor = compressor
        self.meter = meter

    def run(self, x: torch.Tensor, *, key: str, ntokens: int) -> torch.Tensor:
        p = self.compressor.compress(x)
        self.meter.record(key, p.nbytes, ntokens)
        return self.compressor.decompress(p, x.device, x.dtype)


def make_norm_pre_hook(pipeline: Pipeline, key: str):
    """
    Pre-hook for LayerNorm.
    Input args: (hidden_states,)
    We compress hidden_states before it enters the norm.
    """

    def _hook(module, args):
        hs = args[0]
        if not torch.is_tensor(hs):
            return args
        if hs.dim() >= 2:
            ntokens = int(hs.shape[0] * hs.shape[1])
        else:
            return args
        processed_hs = pipeline.run(hs, key=key, ntokens=ntokens)
        return (processed_hs,) + args[1:]

    return _hook


def make_residual_pre_hook(pipeline: Pipeline, key: str):
    """
    Pre-hook for ResidualAdd.
    Input args: (residual, hidden_states)
    We ONLY compress 'residual' (the skip connection).
    'hidden_states' (the main branch) is left alone.
    """

    def _hook(module, args):
        residual = args[0]
        current_hidden_states = args[1]
        if not torch.is_tensor(residual):
            return args
        if residual.dim() >= 2:
            ntokens = int(residual.shape[0] * residual.shape[1])
        else:
            return args
        processed_residual = pipeline.run(
            residual, key=key, ntokens=0
        )  # keep on pipeline, ntokens = 0 to avoid duplicate calculating.
        return (processed_residual, current_hidden_states)

    return _hook


def install_boundary_hooks(
    model,
    boundaries: List[Tuple[str, str, str]],
    norm_pipeline: Pipeline,
    residual_pipeline: Pipeline,
):
    handles = []
    embed = model.model.embed_tokens
    layers = model.model.layers
    for where, src, dst in boundaries:
        norm_key = f"{where}:{src}->{dst}:{norm_pipeline.compressor.name}"
        residual_key = f"{where}:{src}->{dst}:{residual_pipeline.compressor.name}"
        if where == "embed":
            # handles.append(embed.register_forward_hook(make_hook(pipeline, key)))
            raise NotImplementedError("Didn't support embed output hook!")
        elif where.startswith("layer:"):
            i = (
                int(where.split(":")[1]) + 1
            )  # plus one because we inject hook on next layer input rather than current layer output
            # if i > 10:  # TEST, current loss maybe because i impact early layer?
            handles.append(
                layers[i].input_layernorm.register_forward_pre_hook(
                    make_norm_pre_hook(norm_pipeline, norm_key)
                )
            )
            handles.append(
                layers[i].attn_residual_add.register_forward_pre_hook(
                    make_residual_pre_hook(residual_pipeline, residual_key)
                )
            )
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def _input_device(model) -> torch.device:
    try:
        return model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device


def make_default_plan(num_layers: int):
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")
    embed_node = output_node = "node0"
    base = num_layers // 4
    rem = num_layers % 4
    c0 = base
    c1 = base + (1 if rem >= 1 else 0)
    c2 = base + (1 if rem >= 2 else 0)
    c3 = base + (1 if rem >= 3 else 0)
    e = c0 // 2
    l = c0 - e
    layer_to_node = [""] * num_layers
    for i in range(e):
        layer_to_node[i] = "node0"
    for i in range(num_layers - l, num_layers):
        layer_to_node[i] = "node0"
    mid = [i for i, v in enumerate(layer_to_node) if not v]
    k = 0
    for _ in range(c1):
        layer_to_node[mid[k]] = "node1"
        k += 1
    for _ in range(c2):
        layer_to_node[mid[k]] = "node2"
        k += 1
    for _ in range(c3):
        layer_to_node[mid[k]] = "node3"
        k += 1
    return embed_node, layer_to_node, output_node


def find_boundaries(embed_node, layer_to_node, output_node):
    b = []
    if not layer_to_node:
        return b
    if embed_node != layer_to_node[0]:
        b.append(("embed", embed_node, layer_to_node[0]))
    for i in range(len(layer_to_node) - 1):
        if layer_to_node[i] != layer_to_node[i + 1]:
            b.append((f"layer:{i}", layer_to_node[i], layer_to_node[i + 1]))
    last_where = f"layer:{len(layer_to_node) - 1}"
    if layer_to_node[-1] != output_node:
        b.append((last_where, layer_to_node[-1], output_node))
    if output_node != embed_node:
        b.append(("output", output_node, embed_node))
    return b


@torch.no_grad()
def eval_wikitext2_ppl(
    model,
    tokenizer,
    meter: Optional["TrafficMeter"] = None,
    wandb_run: Optional[Any] = None,
    max_length: int = 2048,
    stride: int = 512,
    log_every: int = 1,  # 每 N 个 batch log 一次
    first_k_tokens: int = 0,
    batch_windows: int = 1,  # 每个 batch 包含多少个 window
):
    """
    Batch-able WikiText2 perplexity evaluation.
    - 每个 batch forward 一次
    - tqdm / wandb 每个 batch 更新一次
    - wandb 的 step 使用 batch_idx: 0,1,2,...

    Returns:
      ppl, total_nll, total_loss_tokens
    """
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(_input_device(model))

    if first_k_tokens and first_k_tokens > 0:
        ids = ids[:, : min(first_k_tokens, ids.size(1))]

    seqlen = ids.size(1)

    # 生成窗口列表 (begin, end, trg_len)，保持与 batch=1 完全一致的 trg_len 逻辑
    begins = list(range(0, seqlen, stride))
    windows = []
    prev_end = 0
    for begin in begins:
        end = min(begin + max_length, seqlen)
        trg_len = end - prev_end
        windows.append((begin, end, trg_len))
        prev_end = end
        if end == seqlen:
            break

    # pad token：没有 pad_token_id 的 tokenizer 用 eos 兜底
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    total_nll = 0.0
    total_loss_tokens = 0

    batch_starts = list(range(0, len(windows), batch_windows))
    pbar = tqdm(batch_starts, desc="wikitext2 ppl", unit="batch")

    for batch_idx, batch_start in enumerate(pbar):
        batch = windows[batch_start : batch_start + batch_windows]
        B = len(batch)
        if B == 0:
            break

        # pad 到本 batch 最大长度
        lens = [end - begin for (begin, end, _) in batch]
        L = max(lens)

        x = torch.full((B, L), pad_id, dtype=ids.dtype, device=ids.device)
        attn = torch.zeros((B, L), dtype=torch.long, device=ids.device)
        y = torch.full((B, L), -100, dtype=ids.dtype, device=ids.device)

        # 组装 input / labels
        for b, (begin, end, trg_len) in enumerate(batch):
            l = end - begin
            xb = ids[:, begin:end].squeeze(0)  # [l]

            x[b, :l] = xb
            attn[b, :l] = 1

            yb = xb.clone()
            # 只让最后 trg_len tokens 参与 loss
            yb[:-trg_len] = -100
            y[b, :l] = yb

        # forward：不传 labels，手动算 per-example NLL（等价 HF shift + ignore_index=-100）
        out = model(input_ids=x, attention_mask=attn, use_cache=False)
        logits = out.logits  # [B, L, V]

        shift_logits = logits[:, :-1, :]  # [B, L-1, V]
        shift_labels = y[:, 1:]  # [B, L-1]
        valid_mask = shift_labels != -100

        shift_logits = logits[:, :-1, :]  # [B, L-1, V]
        shift_labels = y[:, 1:]  # [B, L-1]

        V = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.reshape(-1, V).float(),  # fp32 compute
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        batch_nll = float(loss_flat.sum().item())
        batch_tok = int((shift_labels != -100).sum().item())

        total_nll += batch_nll
        total_loss_tokens += batch_tok

        # 每个 batch 更新一次 tqdm / wandb
        if batch_tok > 0:
            batch_avg_nll = batch_nll / batch_tok
            batch_ppl = math.exp(batch_avg_nll)
        else:
            batch_avg_nll = float("nan")
            batch_ppl = float("nan")

        avg_ppl = (
            math.exp(total_nll / total_loss_tokens)
            if total_loss_tokens > 0
            else float("nan")
        )

        traffic = meter.totals() if meter is not None else None

        postfix = {
            "batch": f"{batch_idx}",
            "cur_ppl": f"{batch_ppl:.2f}",
            "avg_ppl": f"{avg_ppl:.2f}",
        }
        if traffic is not None and not math.isnan(traffic["bytes_per_token"]):
            postfix["B/tok"] = f"{traffic['bytes_per_token']:.1f}"
            postfix["bytes"] = format_bytes(traffic["total_bytes"])
            postfix["tx"] = f"{int(traffic['total_tx']):,}"
        pbar.set_postfix(postfix)

        # wandb：每 N 个 batch log 一次
        if (
            wandb_run is not None
            and log_every > 0
            and ((batch_idx + 1) % log_every == 0)
        ):
            log_dict = {
                "eval_batch/batch_idx": batch_idx,
                "eval_batch/batch_tokens": batch_tok,
                "eval_batch/batch_avg_nll": batch_avg_nll,
                "eval_batch/batch_ppl": batch_ppl,
                "eval_total/total_tokens": total_loss_tokens,
                "eval_total/total_nll": total_nll,
                "eval_total/total_avg_nll": (total_nll / total_loss_tokens)
                if total_loss_tokens > 0
                else float("nan"),
                "eval_total/total_avg_ppl": avg_ppl,
            }
            if traffic is not None:
                log_dict.update(
                    {
                        "traffic/total_bytes": traffic["total_bytes"],
                        "traffic/total_tokens": traffic["total_tokens"],
                        "traffic/total_tx": traffic["total_tx"],
                        "traffic/bytes_per_token": traffic["bytes_per_token"],
                    }
                )

            # 关键：wandb step 用 batch_idx（0,1,2,3...）
            wandb_run.log(log_dict, step=batch_idx)

    ppl = (
        math.exp(total_nll / total_loss_tokens)
        if total_loss_tokens > 0
        else float("nan")
    )
    return ppl, total_nll, total_loss_tokens


def run_ppl_eval(
    *,
    model_name: str = "Qwen/Qwen3-32B",
    model_dir: str = "/mnt/ssd/liaw/Qwen/Qwen3-32B",
    dtype: str = "fp16",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    norm_compressor: Union[str, "Compressor"] = "none",
    residual_compressor: Union[str, "Compressor"] = "none",
    max_length: int = 2048,
    stride: int = 512,
    first_k_tokens: int = 0,
    batch_windows: int = 2,
    wandb: bool = True,
    wandb_project: str = "decentralized-infer",
    wandb_run_name: Optional[str] = None,
    wandb_log_every: int = 10,
    result_dir: Optional[str] = None,
):
    load4 = bool(load_in_4bit)
    load8 = bool(load_in_8bit) and (not load4)

    norm_comp = _resolve_compressor(norm_compressor, "norm_compressor")
    residual_comp = _resolve_compressor(residual_compressor, "residual_compressor")

    wandb_run = None
    if wandb:
        import wandb as _wandb

        wandb_run = _wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            resume="never",
            config={
                "model_name": model_name,
                "model_dir": model_dir,
                "dtype": dtype,
                "load_in_8bit": load8,
                "load_in_4bit": load4,
                "norm_compressor": norm_comp.name,
                "residual_compressor": residual_comp.name,
                "max_length": max_length,
                "stride": stride,
                "first_k_tokens": first_k_tokens,
                "batch_windows": batch_windows,
            },
        )

    if not os.path.exists(model_dir):
        # Only download if absolutely necessary. Repeated calls to snapshot_download
        # are safe but add unnecessary startup time/checks if you know the model is there.
        print(f"Local model not found at {model_dir}. Downloading...")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_name, local_dir=model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = load_model(model_dir, dtype, load_in_8bit=load8, load_in_4bit=load4)

    model = patch_qwen_with_residual_modules(model)

    num_layers = len(model.model.layers)
    embed_node, layer_to_node, output_node = make_default_plan(num_layers)
    boundaries = find_boundaries(embed_node, layer_to_node, output_node)

    print(boundaries)

    meter = TrafficMeter()
    norm_pipeline = Pipeline(norm_comp, meter)
    residual_pipeline = Pipeline(residual_comp, meter)

    handles = install_boundary_hooks(
        model, boundaries, norm_pipeline, residual_pipeline
    )

    t0 = time.time()
    ppl, total_nll, total_loss_tokens = eval_wikitext2_ppl(
        model,
        tokenizer,
        meter=meter,
        wandb_run=wandb_run,
        max_length=max_length,
        stride=stride,
        log_every=wandb_log_every,
        first_k_tokens=first_k_tokens,
        batch_windows=batch_windows,
    )
    t1 = time.time()
    remove_hooks(handles)

    totals = meter.totals()

    out_dir = (
        Path(result_dir)
        if (result_dir is not None and str(result_dir).strip() != "")
        else (Path.cwd() / "results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{run_id}.json"

    record = {
        "script_name": "eval_ppl.py",
        "run_id": run_id,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_run_s": t1 - t0,
        "time_run_hms": format_duration(t1 - t0),
        "args": {
            "model_name": model_name,
            "model_dir": model_dir,
            "dtype": dtype,
            "load_in_8bit": load8,
            "load_in_4bit": load4,
            "norm_compressor": norm_comp.name,
            "residual_compressor": residual_comp.name,
            "max_length": max_length,
            "stride": stride,
            "first_k_tokens": first_k_tokens,
            "batch_windows": batch_windows,
        },
        "results": {
            "avg_ppl": float(ppl),
            "total_nll": total_nll,
            "total_loss_tokens": total_loss_tokens,
        },
        "traffic_totals": totals,
        "traffic_per_link": meter.stats,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/ppl_final": ppl,
                "traffic/bytes_per_token": totals["bytes_per_token"],
                "eval/seconds": (t1 - t0),
            }
        )
        wandb_run.finish()

    return ppl, totals, str(out_path)


def parse_args():
    p = argparse.ArgumentParser(description="WikiText2 PPL eval")

    # model / load
    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Just a label for logging/record; model is loaded from --model_dir",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/ssd/liaw/Qwen/Qwen3-8B",
        help="Local model directory to load tokenizer/model from",
    )

    p.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--load_in_4bit", action="store_true", default=False)

    # eval
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument(
        "--first_k_tokens",
        type=int,
        default=0,
        help="Only evaluate first K tokens (0 means full length)",
    )
    p.add_argument("--batch_windows", type=int, default=2)

    # compressor (built-in only)
    p.add_argument(
        "--compressor",
        type=str,
        default="none",
        help="Legacy shorthand: sets both norm and residual compressors",
    )
    p.add_argument(
        "--norm_compressor",
        type=str,
        default=None,
        help="Built-in compressor name for norm path (defaults to --compressor)",
    )
    p.add_argument(
        "--residual_compressor",
        type=str,
        default=None,
        help="Built-in compressor name for residual path (defaults to --compressor)",
    )

    # wandb
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--wandb_project", type=str, default="decentralized-infer")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_log_every", type=int, default=10)

    # results
    p.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help='Where to save results. Default: "./results" in current working dir.',
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    norm_comp = (
        args.norm_compressor if args.norm_compressor is not None else args.compressor
    )
    residual_comp = (
        args.residual_compressor
        if args.residual_compressor is not None
        else args.compressor
    )

    ppl, totals, out_path = run_ppl_eval(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        norm_compressor=norm_comp,
        residual_compressor=residual_comp,
        max_length=args.max_length,
        stride=args.stride,
        first_k_tokens=args.first_k_tokens,
        batch_windows=args.batch_windows,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_log_every=args.wandb_log_every,
        result_dir=args.result_dir,
    )

    print("\n=== DONE ===")
    print(f"PPL = {ppl}")
    print("Traffic totals:", json.dumps(totals, indent=2))
    print("Saved record:", out_path)
