#!/usr/bin/env python3
"""Tests for tools/quantize/apex/apex.py.

Run: python tools/quantize/apex/test_apex.py
The emulated matcher mirrors src/llama-quant.cpp: each tensor name is matched
against the config lines in order, unanchored regex_search, first match wins.
The collision tests replay that against real tensor-name fixtures vendored
from upstream apex-quant (tests/fixtures/, MIT).
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import apex  # noqa: E402

MOE_LINES_PER_LAYER = 17
DENSE_HEADER_LINES = 2
DENSE_LINES_PER_LAYER = 12

ALL_PROFILES = (sorted(apex.MOE_PROFILES) + sorted(apex.DENSE_PROFILES)
                + sorted(apex.OPT_PROFILES))


def ggml_type_names() -> set[str]:
    src = apex.REPO_ROOT / "ggml" / "src" / "ggml.c"
    text = src.read_text(encoding="utf-8", errors="replace")
    names = {m.group(1).lower() for m in
             re.finditer(r'\.type_name\s*=\s*"([^"]+)"', text)}
    if not names:
        raise AssertionError(
            "no .type_name entries parsed from %s; the ggml.c layout moved "
            "and this check must be updated before it can mean anything"
            % src)
    return names


def resolve(config: list[str], tensor_name: str) -> str | None:
    """The type llama-quantize would apply to this tensor, or None."""
    for line in config:
        name, sep, qtype = line.partition("=")
        if not sep:
            raise AssertionError("malformed config line: %r" % line)
        if re.search(name, tensor_name):
            return qtype
    return None


def capture_config(argv: list[str]) -> list[str]:
    proc = subprocess.run([sys.executable, str(HERE / "apex.py")] + argv,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError("apex.py %s failed: %s"
                             % (argv, proc.stderr.strip()))
    return proc.stdout.splitlines()


def config_for(profile: str, layers: int = 40, extra=None) -> list[str]:
    argv = ["--generate-config", "--profile", profile, "--layers",
            str(layers)]
    if extra:
        argv += extra
    return capture_config(argv)


def emit_in_process(profile: str, layers: int, extra=None) -> list[str]:
    """Generate a config without spawning a process; used by the sweep tests."""
    argv = ["--generate-config", "--profile", profile, "--layers",
            str(layers)]
    if extra:
        argv += extra
    args = apex.build_parser().parse_args(argv)
    return apex.emit_config(profile, args, layers)


# ---------------------------------------------------------------- gguf helpers

def _write_value(f, vtype: int, value) -> None:
    if vtype == 8:
        raw = value.encode()
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
    elif vtype == 4:
        f.write(struct.pack("<I", value))
    elif vtype == 5:
        f.write(struct.pack("<i", value))
    elif vtype == 10:
        f.write(struct.pack("<Q", value))
    elif vtype == 11:
        f.write(struct.pack("<q", value))
    elif vtype == 9:
        etype, items = value
        f.write(struct.pack("<I", etype))
        f.write(struct.pack("<Q", len(items)))
        for item in items:
            _write_value(f, etype, item)
    else:
        raise ValueError(vtype)


def write_gguf(path: Path, kvs, tensors=(), version: int = 3) -> None:
    """tensors: (name, dims, ggml_type). Offset is written as 0; nothing reads
    the data section, only the tensor-info records."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", version))
        if version == 1:
            f.write(struct.pack("<II", len(tensors), len(kvs)))
        else:
            f.write(struct.pack("<QQ", len(tensors), len(kvs)))
        for key, vtype, value in kvs:
            raw = key.encode()
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            f.write(struct.pack("<I", vtype))
            _write_value(f, vtype, value)
        for name, dims, ttype in tensors:
            raw = name.encode()
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ttype))
            f.write(struct.pack("<Q", 0))


def fixture_layers(tensors: list[str]) -> int:
    deepest = -1
    for name in tensors:
        m = re.match(r"blk\.(\d+)\.", name)
        if m:
            deepest = max(deepest, int(m.group(1)))
    return deepest + 1


def load_fixtures() -> dict[str, list[str]]:
    out = {}
    for path in sorted((HERE / "tests" / "fixtures").glob("tl_*.json")):
        out[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


class EmitTests(unittest.TestCase):
    def test_moe_line_count(self):
        for profile in apex.MOE_PROFILES:
            cfg = config_for(profile, 40)
            self.assertEqual(len(cfg), MOE_LINES_PER_LAYER * 40, profile)

    def test_dense_line_count(self):
        for profile in apex.DENSE_PROFILES:
            cfg = config_for(profile, 40)
            self.assertEqual(len(cfg),
                             DENSE_HEADER_LINES + DENSE_LINES_PER_LAYER * 40,
                             profile)

    def test_opt_line_count(self):
        for profile in apex.OPT_PROFILES:
            cfg = config_for(profile, 40)
            self.assertEqual(len(cfg),
                             DENSE_HEADER_LINES + DENSE_LINES_PER_LAYER * 40,
                             profile)

    def test_every_type_is_a_ggml_type(self):
        valid = ggml_type_names()
        for profile in ALL_PROFILES:
            for line in config_for(profile, 12):
                qtype = line.partition("=")[2]
                self.assertIn(qtype.lower(), valid,
                              "%s emits unknown type %s" % (profile, qtype))

    def test_moe_bands_are_positional(self):
        cfg = config_for("quality", 40)
        # quality: edge Q6_K, near Q5_K, mid IQ4_XS
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_exps.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.4.ffn_up_exps.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.5.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.9.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.10.ffn_gate_exps.weight"), "IQ4_XS")
        self.assertEqual(resolve(cfg, "blk.29.ffn_gate_exps.weight"), "IQ4_XS")
        self.assertEqual(resolve(cfg, "blk.30.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.34.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.35.ffn_gate_exps.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.39.ffn_gate_exps.weight"), "Q6_K")

    def test_moe_attention_band_is_narrower(self):
        cfg = config_for("mini", 40)
        # mini: attention edge Q4_K, mid Q3_K; edge is the first/last THREE
        self.assertEqual(resolve(cfg, "blk.2.attn_q.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.3.attn_q.weight"), "Q3_K")
        self.assertEqual(resolve(cfg, "blk.37.attn_q.weight"), "Q4_K")

    def test_moe_shared_expert_band(self):
        cfg = config_for("mini", 40)
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_shexp.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate_shexp.weight"), "Q4_K")

    def test_gate_inp_falls_through_to_the_base_mixture(self):
        cfg = config_for("balanced", 40)
        self.assertIsNone(resolve(cfg, "blk.0.ffn_gate_inp.weight"))
        self.assertIsNone(resolve(cfg, "blk.20.ffn_gate_inp.weight"))

    def test_moe_emits_shexp_not_exps_for_shared(self):
        cfg = config_for("balanced", 40)
        self.assertEqual(resolve(cfg, "blk.0.ffn_up_shexp.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.39.ffn_down_shexp.weight"), "Q8_0")

    def test_moe_emits_ssm_and_shortconv(self):
        cfg = config_for("balanced", 40)
        self.assertEqual(resolve(cfg, "blk.0.ssm_alpha.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.0.ssm_out.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.0.shortconv.in_proj.weight"),
                         "Q6_K")

    def test_dense_flat_has_no_gradient(self):
        cfg = config_for("dense-flat", 40)
        for layer in (0, 5, 20, 34, 39):
            self.assertEqual(resolve(cfg, "blk.%d.ffn_gate.weight" % layer),
                             "IQ4_XS")
            self.assertEqual(resolve(cfg, "blk.%d.ffn_up.weight" % layer),
                             "Q5_K")

    def test_dense_grad_bands(self):
        cfg = config_for("dense-grad", 40)
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.5.ffn_gate.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate.weight"), "IQ4_XS")
        self.assertEqual(resolve(cfg, "blk.0.ffn_up.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.5.ffn_up.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_up.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_down.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_down.weight"), "Q5_K")

    def test_dense_hybrid_pins_full_attention(self):
        cfg = config_for("dense-hybrid", 40)
        self.assertEqual(resolve(cfg, "blk.3.attn_q.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.3.attn_v.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.3.attn_qkv.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.3.ssm_alpha.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "Q5_K")

    def test_dense_hybrid_quality_also_pins_full_attention(self):
        # The upstream hybrid override applies to BOTH hybrid profiles; the
        # quality rebuild differs in the FFN band, not in attention.
        cfg = config_for("dense-hybrid-quality", 40)
        self.assertEqual(resolve(cfg, "blk.3.attn_q.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.3.attn_v.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.3.attn_qkv.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.3.ssm_alpha.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_down.weight"), "Q8_0")

    def test_dense_anchoring_is_load_bearing(self):
        cfg = config_for("dense-flat", 40)
        # Unanchored, "output.weight" would capture every attn_output.weight
        self.assertEqual(resolve(cfg, "output.weight"), "Q6_K")
        self.assertIsNone(resolve(cfg, "attn_output.weight"))
        self.assertEqual(resolve(cfg, "blk.3.attn_output.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "token_embd.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "IQ4_XS")
        # The same trap one level down: ffn_gate must not match ffn_gate_inp
        self.assertIsNone(resolve(cfg, "blk.0.ffn_gate_inp.weight"))
        self.assertIsNone(resolve(cfg, "blk.0.ffn_gate_shexp.weight"))

    def test_opt_balanced_allocation(self):
        cfg = config_for("opt-balanced", 40)
        self.assertEqual(resolve(cfg, "token_embd.weight"), "Q3_K")
        self.assertEqual(resolve(cfg, "output.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate.weight"), "IQ4_XS")
        self.assertEqual(resolve(cfg, "blk.20.ffn_up.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_down.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.3.attn_q.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.3.attn_qkv.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.3.ssm_out.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.3.ssm_alpha.weight"), "Q4_K")

    def test_opt_mini_allocation(self):
        cfg = config_for("opt-mini", 40)
        self.assertEqual(resolve(cfg, "token_embd.weight"), "IQ2_S")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate.weight"), "IQ3_XXS")
        self.assertEqual(resolve(cfg, "blk.3.attn_q.weight"), "Q5_K")

    def test_opt_profiles_use_q5_k_base(self):
        args = apex.build_parser().parse_args(["m.gguf", "--profile",
                                               "opt-nano"])
        self.assertEqual(apex.base_type_for("opt-nano", args), "Q5_K")

    def test_default_profile_is_balanced(self):
        cfg = config_for("balanced", 40)
        self.assertEqual(resolve(cfg, "blk.10.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_exps.weight"), "Q6_K")

    def test_custom_profile(self):
        cfg = config_for("custom", 40, ["--edge-exp", "Q5_K"])
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_shexp.weight"), "Q8_0")
        cfg = config_for("custom", 40, ["--edge-exp", "Q5_K",
                                        "--near-exp", "Q4_K",
                                        "--mid-exp", "Q3_K",
                                        "--edge-shared", "Q6_K",
                                        "--mid-shared", "Q5_K"])
        self.assertEqual(resolve(cfg, "blk.5.ffn_gate_exps.weight"), "Q4_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate_exps.weight"), "Q3_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_shexp.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_gate_shexp.weight"), "Q5_K")

    def test_dense_layers_inside_a_moe(self):
        cfg = config_for("balanced", 40, ["--dense-layers", "4"])
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.3.ffn_up.weight"), "Q8_0")
        self.assertEqual(resolve(cfg, "blk.4.ffn_gate_exps.weight"), "Q6_K")
        self.assertIsNone(resolve(cfg, "blk.0.ffn_gate_inp.weight"))

    def test_small_layer_counts_do_not_crash(self):
        for layers in (1, 2, 5, 12):
            cfg = config_for("quality", layers)
            self.assertEqual(len(cfg), MOE_LINES_PER_LAYER * layers)


class CollisionTests(unittest.TestCase):
    """Replay llama-quantize's matching against real tensor-name fixtures.

    A pattern that captures two tensors with different types ships a silent
    wrong quant: nothing in the config or the quantize log reveals it.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        if not cls.fixtures:
            raise unittest.SkipTest("no fixtures vendored")

    def test_no_conflicting_multi_match(self):
        for fixture, tensors in self.fixtures.items():
            layers = fixture_layers(tensors)
            for profile in ALL_PROFILES:
                cfg = emit_in_process(profile, layers)
                compiled = [(re.compile(ln.partition("=")[0]),
                             ln.partition("=")[0], ln.partition("=")[2])
                            for ln in cfg]
                conflicts = {}
                for name in tensors:
                    hits = [(pat, q) for rx, pat, q in compiled
                            if rx.search(name)]
                    if len(hits) > 1 and len({q for _, q in hits}) > 1:
                        conflicts[name] = hits
                self.assertEqual(
                    conflicts, {},
                    "%s x %s: %d tensor(s) matched by rules that disagree, "
                    "first one wins" % (fixture, profile, len(conflicts)))

    def test_known_benign_attn_q_qkv_ambiguity(self):
        # The MoE emitter writes unanchored "blk.N.attn_q", which also matches
        # "blk.N.attn_qkv.weight". Upstream records this as known-benign: every
        # MoE profile gives both the same type. This test pins that fact so a
        # future profile that splits them fails here instead of shipping.
        cfg = config_for("balanced", 40)
        hits = [(ln.partition("=")[0], ln.partition("=")[2]) for ln in cfg
                if re.search(ln.partition("=")[0], "blk.7.attn_qkv.weight")]
        self.assertGreater(len(hits), 1)
        self.assertEqual(len({q for _, q in hits}), 1)

    def test_fixture_expert_tensors_get_the_band_type(self):
        tensors = self.fixtures.get("tl_Qwen3.5-35B-A3B.json")
        if tensors is None:
            self.skipTest("Qwen3.5 fixture missing")
        cfg = config_for("balanced", fixture_layers(tensors))
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_exps.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.20.ffn_up_exps.weight"), "Q5_K")
        self.assertEqual(resolve(cfg, "blk.39.ffn_down_exps.weight"), "Q6_K")
        self.assertEqual(resolve(cfg, "blk.0.ffn_gate_shexp.weight"), "Q8_0")
        # norms, biases and non-weight tensors must stay unclaimed
        for name in ("blk.0.attn_norm.weight", "blk.0.ssm_a",
                     "blk.0.ssm_dt.bias", "blk.0.ffn_gate_inp.weight"):
            self.assertIsNone(resolve(cfg, name), name)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="apex-test-"))

    def tearDown(self):
        for p in sorted(self.tmp.rglob("*"), reverse=True):
            try:
                p.unlink()
            except OSError:
                pass
        self.tmp.rmdir()

    def test_reads_moe_metadata(self):
        path = self.tmp / "moe.gguf"
        write_gguf(path, [
            ("general.architecture", 8, "qwen3moe"),
            ("qwen3moe.block_count", 4, 40),
            ("qwen3moe.expert_count", 4, 128),
            ("qwen3moe.expert_used_count", 4, 8),
        ])
        meta = apex.read_gguf_meta(path)
        self.assertEqual(meta["block_count"], 40)
        self.assertEqual(meta["expert_used_count"], 8)
        self.assertEqual(meta["general.architecture"], "qwen3moe")

    def test_reads_unprefixed_keys_too(self):
        path = self.tmp / "plain.gguf"
        write_gguf(path, [("block_count", 4, 22)])
        self.assertEqual(apex.read_gguf_meta(path)["block_count"], 22)

    def test_dense_model_has_no_expert_keys(self):
        path = self.tmp / "dense.gguf"
        write_gguf(path, [
            ("general.architecture", 8, "llama"),
            ("llama.block_count", 4, 64),
        ])
        meta = apex.read_gguf_meta(path)
        self.assertEqual(meta["block_count"], 64)
        self.assertNotIn("expert_used_count", meta)

    def test_skips_arrays_of_both_shapes(self):
        path = self.tmp / "arrays.gguf"
        write_gguf(path, [
            ("general.architecture", 8, "llama"),
            ("tokenizer.ggml.tokens", 9, (8, ["<s>", "</s>", "a"])),
            ("tokenizer.ggml.token_type", 9, (4, [3, 3, 1])),
            ("llama.block_count", 4, 32),
        ])
        self.assertEqual(apex.read_gguf_meta(path)["block_count"], 32)

    def test_version_one_header(self):
        path = self.tmp / "v1.gguf"
        write_gguf(path, [("llama.block_count", 4, 22)], version=1)
        self.assertEqual(apex.read_gguf_meta(path)["block_count"], 22)

    def test_garbage_returns_none(self):
        path = self.tmp / "junk.bin"
        path.write_bytes(b"not a gguf file at all")
        self.assertIsNone(apex.read_gguf_meta(path))

    def test_detect_layers_prefers_the_files_own_count(self):
        path = self.tmp / "moe.gguf"
        write_gguf(path, [
            ("general.architecture", 8, "qwen3moe"),
            ("qwen3moe.block_count", 4, 48),
            ("qwen3moe.expert_used_count", 4, 8),
        ])
        args = apex.build_parser().parse_args(["x.gguf"])
        self.assertEqual(apex.detect_layers(path, args), 48)
        args = apex.build_parser().parse_args(["x.gguf", "--layers", "7"])
        self.assertEqual(apex.detect_layers(path, args), 7)

    def test_detect_layers_falls_back_with_a_warning(self):
        path = self.tmp / "junk.bin"
        path.write_bytes(b"nope")
        args = apex.build_parser().parse_args(["x.gguf"])
        self.assertEqual(apex.detect_layers(path, args),
                         apex.DEFAULT_LAYERS)

    def test_reads_the_tensor_inventory(self):
        path = self.tmp / "inv.gguf"
        write_gguf(path, [
            ("general.architecture", 8, "llama"),
            ("llama.block_count", 4, 1),
        ], tensors=[
            ("token_embd.weight", (32000, 5120), 0),
            ("blk.0.ffn_down.weight", (5120, 13824), 0),
            ("blk.0.attn_norm.weight", (5120,), 0),
            ("blk.0.ssm_dt.bias", (5120,), 0),
        ])
        tensors = apex.read_gguf_tensors(path)
        self.assertEqual(len(tensors), 4)
        by_name = {n: (numel, dims) for n, numel, dims in tensors}
        self.assertEqual(by_name["token_embd.weight"], (32000 * 5120, 2))
        self.assertEqual(by_name["blk.0.attn_norm.weight"], (5120, 1))
        self.assertEqual(by_name["blk.0.ssm_dt.bias"], (5120, 1))

    def test_allows_quantization_matches_llama_cpp(self):
        # src/llama-quant.cpp:287 -- 2D+, ends with "weight", not a norm
        self.assertTrue(apex.allows_quantization("blk.0.ffn_gate.weight", 2))
        self.assertTrue(apex.allows_quantization("blk.0.ffn_down_exps.weight",
                                                 3))
        self.assertFalse(apex.allows_quantization("blk.0.attn_norm.weight", 2))
        self.assertFalse(apex.allows_quantization("blk.0.ssm_a", 1))
        self.assertFalse(apex.allows_quantization("blk.0.ssm_dt.bias", 1))

    def test_version_one_tensor_inventory(self):
        path = self.tmp / "v1t.gguf"
        write_gguf(path, [("llama.block_count", 4, 1)],
                   tensors=[("token_embd.weight", (100, 100), 0)],
                   version=1)
        tensors = apex.read_gguf_tensors(path)
        self.assertEqual(tensors, [("token_embd.weight", 10000, 2)])


class EstimateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="apex-test-"))

    def tearDown(self):
        for p in sorted(self.tmp.rglob("*"), reverse=True):
            try:
                p.unlink()
            except OSError:
                pass
        self.tmp.rmdir()

    def _model(self, name="m.gguf", expert=True):
        path = self.tmp / name
        kvs = [("general.architecture", 8, "qwen3moe" if expert else "llama"),
               ("qwen3moe.block_count" if expert else "llama.block_count",
                4, 1)]
        if expert:
            kvs.append(("qwen3moe.expert_used_count", 4, 8))
        write_gguf(path, kvs, tensors=[
            ("token_embd.weight", (5000, 2000), 0),
            ("blk.0.ffn_gate_exps.weight", (2000, 1000, 8), 0),
            ("blk.0.ffn_gate_inp.weight", (8, 2000), 0),
            ("blk.0.attn_norm.weight", (2000,), 0),
        ])
        return path

    def _run(self, argv):
        return subprocess.run(
            [sys.executable, str(HERE / "apex.py")] + argv,
            capture_output=True, text=True)

    def test_estimate_matches_the_bits_per_weight_math(self):
        path = self._model()
        proc = self._run(["--estimate", "--profile", "balanced",
                          "--layers", "1", "--quiet", str(path)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # balanced: blk.0.ffn_gate_exps -> Q6_K (edge layer). token_embd has no
        # MoE rule so it falls back to base Q6_K. ffn_gate_inp is 16k elements,
        # below the F32 floor llama.cpp keeps at F32, and attn_norm is 1D.
        expected = ((10_000_000 * apex.BPW["Q6_K"])
                    + (16_000_000 * apex.BPW["Q6_K"])
                    + (16_000 * apex.BPW["F32"])
                    + (2_000 * apex.BPW["F32"])) / 8 / 1e9
        self.assertEqual(proc.stdout.strip(), "%.3f" % expected)

    def test_estimate_opt_covers_token_embd(self):
        path = self._model()
        proc = self._run(["--estimate", "--profile", "opt-balanced",
                          "--layers", "1", "--quiet", str(path)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # opt-balanced: token_embd -> Q3_K; experts uncovered -> base Q5_K;
        # ffn_gate_inp and the 1D norm stay at F32.
        expected = ((10_000_000 * apex.BPW["Q3_K"])
                    + (16_000_000 * apex.BPW["Q5_K"])
                    + (16_000 * apex.BPW["F32"])
                    + (2_000 * apex.BPW["F32"])) / 8 / 1e9
        self.assertEqual(proc.stdout.strip(), "%.3f" % expected)

    def test_estimate_verbose_prints_breakdown(self):
        path = self._model()
        proc = self._run(["--estimate", "--profile", "balanced",
                          "--layers", "1", str(path)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("estimated size:", proc.stdout)
        self.assertIn("Q6_K", proc.stdout)
        self.assertIn("F32 (kept)", proc.stdout)

    def test_estimate_rejects_a_type_outside_the_table(self):
        path = self._model()
        proc = self._run(["--estimate", "--profile", "balanced",
                          "--layers", "1", "--base-type", "Q4_K_M", str(path)])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("bits-per-weight table", proc.stderr)

    def test_estimate_accepts_a_user_config(self):
        path = self._model()
        cfg = self.tmp / "cfg.txt"
        cfg.write_text("^token_embd\\.weight$=IQ2_XXS\n", encoding="utf-8")
        proc = self._run(["--estimate", "--config", str(cfg),
                          "--base-type", "Q6_K", "--quiet", str(path)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        expected = ((10_000_000 * apex.BPW["IQ2_XXS"])
                    + (16_000_000 * apex.BPW["Q6_K"])
                    + (16_000 * apex.BPW["F32"])
                    + (2_000 * apex.BPW["F32"])) / 8 / 1e9
        self.assertEqual(proc.stdout.strip(), "%.3f" % expected)

    def test_estimate_on_unreadable_input_fails(self):
        path = self.tmp / "junk.bin"
        path.write_bytes(b"nope")
        proc = self._run(["--estimate", "--profile", "balanced", str(path)])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("tensor inventory", proc.stderr)


class CliTests(unittest.TestCase):
    def run_apex(self, argv):
        return subprocess.run(
            [sys.executable, str(HERE / "apex.py")] + argv,
            capture_output=True, text=True)

    def test_unknown_profile_exits_nonzero_and_lists_profiles(self):
        proc = self.run_apex(["--generate-config", "--profile", "nope"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("dense-hybrid-quality", proc.stderr)
        self.assertIn("i-quality", proc.stderr)
        self.assertIn("opt-balanced", proc.stderr)

    def test_custom_without_edge_exp_is_an_error(self):
        proc = self.run_apex(["--generate-config", "--profile", "custom"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("--edge-exp", proc.stderr)

    def test_arch_dense_with_moe_profile_is_an_error(self):
        proc = self.run_apex(["--generate-config", "--profile", "balanced",
                              "--arch", "dense"])
        self.assertEqual(proc.returncode, 1)

    def test_dense_layers_with_dense_profile_is_an_error(self):
        proc = self.run_apex(["--generate-config", "--profile", "dense-flat",
                              "--dense-layers", "4"])
        self.assertEqual(proc.returncode, 1)

    def test_config_and_generate_config_conflict(self):
        proc = self.run_apex(["--generate-config", "--config", "x.txt"])
        self.assertEqual(proc.returncode, 1)

    def test_generate_config_to_file(self):
        with tempfile.TemporaryDirectory(prefix="apex-test-") as tmp:
            out = Path(tmp) / "cfg.txt"
            proc = self.run_apex(["--generate-config", "--profile", "compact",
                                  "--layers", "8", "-o", str(out)])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(len(out.read_text().splitlines()),
                             MOE_LINES_PER_LAYER * 8)

    def test_missing_input_is_an_error(self):
        proc = self.run_apex(["--profile", "balanced"])
        self.assertEqual(proc.returncode, 1)

    def test_quantize_not_found_is_an_error(self):
        with tempfile.TemporaryDirectory(prefix="apex-test-") as tmp:
            gguf = Path(tmp) / "tiny.gguf"
            write_gguf(gguf, [
                ("general.architecture", 8, "qwen3moe"),
                ("qwen3moe.block_count", 4, 8),
                ("qwen3moe.expert_used_count", 4, 8),
            ])
            proc = self.run_apex(["--profile", "balanced", "--quantize",
                                  "definitely-not-here.exe", str(gguf)])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("llama-quantize not found", proc.stderr)

    def test_dense_model_with_moe_profile_warns(self):
        with tempfile.TemporaryDirectory(prefix="apex-test-") as tmp:
            gguf = Path(tmp) / "dense.gguf"
            write_gguf(gguf, [("general.architecture", 8, "llama"),
                              ("llama.block_count", 4, 8)])
            proc = self.run_apex(["--profile", "balanced", "--quantize",
                                  "definitely-not-here.exe", str(gguf)])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("no expert tensors", proc.stderr)

    def test_moe_model_with_opt_profile_warns(self):
        with tempfile.TemporaryDirectory(prefix="apex-test-") as tmp:
            gguf = Path(tmp) / "moe.gguf"
            write_gguf(gguf, [
                ("general.architecture", 8, "qwen3moe"),
                ("qwen3moe.block_count", 4, 8),
                ("qwen3moe.expert_used_count", 4, 8),
            ])
            proc = self.run_apex(["--profile", "opt-balanced", "--quantize",
                                  "definitely-not-here.exe", str(gguf)])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("expert tensors", proc.stderr)

    def test_help_exits_zero(self):
        proc = self.run_apex(["--help"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("balanced", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
