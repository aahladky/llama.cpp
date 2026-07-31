#!/usr/bin/env python3
"""Build a tiny deterministic mixtral-style MoE GGUF for cache correctness testing.

Arch: "llama" with n_expert>0 (the plain mixtral-style MoE path, no shared
expert). Tokenizer is copied verbatim from the real llama-3 BPE vocab test
fixture (models/ggml-vocab-llama-bpe.gguf) so tokenization is fully valid;
everything else (hidden dim, layers, experts) is deliberately tiny so the
model loads and runs in well under a second on either CPU or GPU.

All weights are generated from a fixed numpy RandomState seed, so re-running
this script produces byte-identical tensors every time.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "gguf-py"))

import numpy as np
import gguf

SEED = 20260730
N_EMBD = 64
N_LAYER = 3
N_HEAD = 4
N_HEAD_KV = 4
N_FF = 128          # per-expert FFN inner dim
N_EXPERT = 8
N_EXPERT_USED = 2
RMS_EPS = 1e-5

def build(out_path: str):
    rng = np.random.RandomState(SEED)

    vr = gguf.GGUFReader(str(REPO / "models/ggml-vocab-llama-bpe.gguf"))

    def get_str_arr(name):
        f = vr.fields[name]
        out = []
        for idx in f.data:
            part = f.parts[idx]
            out.append(bytes(part).decode('utf-8', errors='replace') if hasattr(part, 'tobytes') else str(part))
        return out

    def get_int_arr(name):
        f = vr.fields[name]
        return [int(f.parts[idx][0]) for idx in f.data]

    def get_scalar_str(name):
        f = vr.fields[name]
        return bytes(f.parts[f.data[0]]).decode('utf-8')

    def get_scalar_int(name):
        f = vr.fields[name]
        return int(f.parts[f.data[0]][0])

    tokens = get_str_arr('tokenizer.ggml.tokens')
    token_types = get_int_arr('tokenizer.ggml.token_type')
    merges = get_str_arr('tokenizer.ggml.merges')
    tok_model = get_scalar_str('tokenizer.ggml.model')
    tok_pre = get_scalar_str('tokenizer.ggml.pre')
    bos_id = get_scalar_int('tokenizer.ggml.bos_token_id')
    eos_id = get_scalar_int('tokenizer.ggml.eos_token_id')

    n_vocab = len(tokens)
    print(f"vocab size: {n_vocab}, bos={bos_id} eos={eos_id}", file=sys.stderr)

    w = gguf.GGUFWriter(out_path, "llama")
    w.add_name("tiny-moe-correctness-fixture")
    w.add_context_length(512)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_dimension_count(N_EMBD // N_HEAD)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_file_type(0)  # F32

    w.add_tokenizer_model(tok_model)
    w.add_tokenizer_pre(tok_pre)
    w.add_token_list(tokens)
    w.add_token_types(token_types)
    w.add_token_merges(merges)
    w.add_bos_token_id(bos_id)
    w.add_eos_token_id(eos_id)

    def randn(*shape):
        return (rng.randn(*shape) * 0.02).astype(np.float32)

    n_embd_head = N_EMBD // N_HEAD

    w.add_tensor("token_embd.weight", randn(n_vocab, N_EMBD))
    w.add_tensor("output_norm.weight", np.ones((N_EMBD,), dtype=np.float32))
    w.add_tensor("output.weight", randn(n_vocab, N_EMBD))

    for i in range(N_LAYER):
        p = f"blk.{i}"
        w.add_tensor(f"{p}.attn_norm.weight", np.ones((N_EMBD,), dtype=np.float32))
        w.add_tensor(f"{p}.attn_q.weight", randn(N_HEAD * n_embd_head, N_EMBD))
        w.add_tensor(f"{p}.attn_k.weight", randn(N_HEAD_KV * n_embd_head, N_EMBD))
        w.add_tensor(f"{p}.attn_v.weight", randn(N_HEAD_KV * n_embd_head, N_EMBD))
        w.add_tensor(f"{p}.attn_output.weight", randn(N_EMBD, N_HEAD * n_embd_head))
        w.add_tensor(f"{p}.ffn_norm.weight", np.ones((N_EMBD,), dtype=np.float32))
        w.add_tensor(f"{p}.ffn_gate_inp.weight", randn(N_EXPERT, N_EMBD))
        w.add_tensor(f"{p}.ffn_gate_exps.weight", randn(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(f"{p}.ffn_down_exps.weight", randn(N_EXPERT, N_EMBD, N_FF))
        w.add_tensor(f"{p}.ffn_up_exps.weight", randn(N_EXPERT, N_FF, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?",
                    default=str(Path(__file__).resolve().parent / "tiny-moe.gguf"))
    ap.add_argument("--n-embd", type=int, default=N_EMBD,
                    help="hidden dim; K-quant fixtures need a multiple of 256 "
                         "(QK_K row divisibility, e.g. 256)")
    ap.add_argument("--n-ff", type=int, default=N_FF,
                    help="per-expert FFN inner dim; same 256-divisibility "
                         "note as --n-embd for K-quant fixtures")
    args = ap.parse_args()
    N_EMBD = args.n_embd
    N_FF = args.n_ff
    build(args.out)
