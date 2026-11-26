"""Streaming wrapper around the Underdog Woof numpy runtime (research only; the original package is imported
unmodified from models/woof, nothing is trained on its output).

    from stream_runtime import WoofStream
    tts = WoofStream(voice="juno")
    for chunk in tts.stream("Let me look through your transactions.", seed=0):   # float32 mono 24 kHz
        play(chunk)

What is different from Woof.__call__:
  * the speech model yields each 32-d latent frame (40 ms of audio) as soon as it is sampled;
  * the decoder runs incrementally: its 2-layer transformer is causal (window 250 frames) so it only needs a K/V
    cache, and every SEANet conv / transposed conv already carries its own left state, so decoding n new frames
    gives exactly the n*960 samples the whole-sentence decode would have produced for them (no lookahead).
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WOOF_DIR = os.environ.get("WOOF_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "models", "woof")
if WOOF_DIR not in sys.path:
    sys.path.insert(0, WOOF_DIR)

from underdog_woof import runtime as rt                      # noqa: E402
from underdog_woof.normalize import Normalizer               # noqa: E402
from underdog_woof.textsplit import split_text               # noqa: E402

f32 = np.float32
SR = rt.SR
FRAME = 960                                                  # samples per latent frame (25 fps at 24 kHz)
WEIGHTS = os.path.join(WOOF_DIR, "woof_int8.npz")


_ELU_ORIG = rt._elu_


def _elu_fast(x):
    """ELU with exp(x) - 1 instead of expm1: numpy's float32 exp is SIMD-vectorised here, expm1 is not. The result
    differs from the original by at most ~1.5e-7 per activation."""
    neg = np.minimum(x, f32(0))
    np.exp(neg, out=neg)
    neg -= f32(1)
    np.maximum(x, f32(0), out=x)
    x += neg
    return x


def set_fast_elu(on=True):
    """Swaps the ELU used by the decoder (whole-sentence and streaming paths alike, it is a module global)."""
    rt._elu_ = _elu_fast if on else _ELU_ORIG


if os.environ.get("WOOF_FAST_ELU") == "1":
    set_fast_elu(True)


class StreamDecoder:
    """Incremental version of rt.Decoder.__call__ sharing its (fp32) weights. push(lat[n, 32]) -> wav[n * 960]."""

    def __init__(self, dec, max_frames=1100):
        assert dec.tf is not None, "use dec_mode='fp32'"
        self.d = dec
        L, nh, hd = len(dec.tf), dec.nh, dec.hd
        self.K = np.zeros((L, nh, max_frames, hd), f32)
        self.V = np.zeros((L, nh, max_frames, hd), f32)
        fr = np.arange(max_frames, dtype=f32)[:, None] * dec.inv_freq
        emb = np.concatenate([fr, fr], -1)
        self.cos, self.sin = np.cos(emb), np.sin(emb)
        self.scale = f32(1 / math.sqrt(hd))
        self.t_tf = self.t_sea = 0.0
        self.reset()

    def reset(self):
        self.t = 0
        for kind, op in self.d.ops:
            if kind == "res":
                op[0].reset(); op[1].reset()
            elif op is not None:
                op.reset()

    def push(self, lat):
        t0 = time.perf_counter()
        d, t, n = self.d, self.t, lat.shape[0]
        nh, hd, D = d.nh, d.hd, d.D
        h = lat @ d.inp_w + d.inp_b
        cos, sin = self.cos[t:t + n], self.sin[t:t + n]
        lo = max(0, t + 1 - d.window)
        i = np.arange(t, t + n)[:, None]
        j = np.arange(lo, t + n)[None]
        bad = (j > i) | (j <= i - d.window)
        mask = bad.any()
        for li, l in enumerate(d.tf):
            y = d._ln(h, l["ln1"])
            qkv = (y @ l["qkv"]).reshape(n, 3, nh, hd).transpose(1, 2, 0, 3)
            qk = qkv[:2]
            qk = qk * cos + np.concatenate([-qk[..., hd // 2:], qk[..., :hd // 2]], -1) * sin
            K, V = self.K[li], self.V[li]
            K[:, t:t + n] = qk[1]
            V[:, t:t + n] = qkv[2]
            s = np.matmul(qk[0], K[:, lo:t + n].transpose(0, 2, 1)) * self.scale
            if mask:
                s[:, bad] = -np.inf
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = np.matmul(s, V[:, lo:t + n]).transpose(1, 0, 2).reshape(n, D)
            h = h + a @ l["o"]
            y = d._ln(h, l["ln2"])
            h = h + rt._gelu(y @ l["fc1"]) @ l["fc2"]
        self.t = t + n
        t1 = time.perf_counter()
        x = d._run(h, d.ops)
        self.t_tf += t1 - t0
        self.t_sea += time.perf_counter() - t1
        return np.ascontiguousarray(x[:, 0])


class WoofStream:
    """first_chunk / chunk: how many 40 ms latent frames are decoded per emitted audio chunk."""

    def __init__(self, weights=WEIGHTS, voice="juno", temp=0.5, eos=0.9, first_chunk=1, chunk=4, normalize=True,
                 max_spc=0.095, stall_s=0.6, cap_margin_s=1.0, schedule=None):
        self.m = rt.WoofNP(weights, voice=voice, dec_mode="fp32")
        self.sdec = StreamDecoder(self.m.dec)
        self.norm = Normalizer(use_nemo=False) if normalize else None
        self.temp, self.eos, self.first_chunk, self.chunk = temp, eos, first_chunk, chunk
        self.max_spc, self.stall_s, self.cap_margin_s = max_spc, stall_s, cap_margin_s
        self.schedule = schedule          # e.g. (1, 2, 4): frames per emitted chunk, last value repeats
        self.sr = SR
        self.stats = {}

    # ---- speech model: same maths and RNG draws as rt.Speech.generate, but a generator ----
    def frames(self, text):
        sp, m = self.m.speech, self.m
        ids = rt.encode_text(text)
        x = np.concatenate([m.voice, sp.text_emb[np.asarray(ids)], sp.bos[None]], 0).astype(f32)
        p0 = x.shape[0]
        max_frames = min(int(m.frames_max), sp.max_len - p0)
        for i in range(sp.L):
            x = sp._prefix_block(i, x, p0)
        x = np.ascontiguousarray(x[-1])
        s, e = sp._cond(x)
        eos_t = math.log(self.eos / (1 - self.eos))
        win, wb = sp.lat_in
        temp = f32(self.temp)
        for f in range(max_frames):
            if f > 3 and e > eos_t:
                break
            noise = rt.RNG.randn(sp.latent) * temp
            z = noise + sp._head(s, noise)
            yield z
            x = win @ z + wb
            for i in range(sp.L):
                x = sp._block1(i, x, p0 + f)
            s, e = sp._cond(x)

    def stream_piece(self, text, guard=True):
        """One utterance. Yields float32 chunks. With guard=True the take is cut (with a short fade) when it runs
        away (longer than max_spc seconds per character) or stalls (stall_s of silence after speech started)."""
        m, dec = self.m, self.sdec
        dec.reset()
        st = self.stats
        cap_frames = int((self.max_spc * max(1, len(text)) + self.cap_margin_s) * m.fps) if guard else 10 ** 9
        sched = list(self.schedule or (self.first_chunk, self.chunk))
        quiet_run, voiced, need, buf, nf = 0, False, sched.pop(0), [], 0
        t_sp = time.perf_counter()
        aborted = None
        for z in self.frames(text):
            buf.append(z)
            nf += 1
            if len(buf) < need and nf < cap_frames:
                continue
            t_d = time.perf_counter()
            st["t_speech"] = st.get("t_speech", 0.0) + t_d - t_sp
            wav = dec.push(np.stack(buf) * m.std + m.mean)
            st["t_dec"] = st.get("t_dec", 0.0) + time.perf_counter() - t_d
            buf, need = [], (sched.pop(0) if len(sched) > 1 else sched[0]) if sched else need
            if guard:
                fr = wav.reshape(-1, FRAME)
                for q in 20 * np.log10(np.sqrt((fr * fr).mean(1)) + 1e-9) < -50.0:
                    quiet_run = quiet_run + 1 if (q and voiced) else 0
                    voiced = voiced or not q
                if nf >= cap_frames:
                    aborted = "runaway"
                elif quiet_run * FRAME / SR >= self.stall_s:
                    aborted = "stall"
                if aborted == "runaway":
                    k = min(len(wav), 480)
                    wav[-k:] *= np.linspace(1, 0, k, dtype=f32)
                elif aborted == "stall":          # everything from here on is the silent tail: do not play it
                    break
            yield wav
            if aborted:
                break
            t_sp = time.perf_counter()
        else:
            st["t_speech"] = st.get("t_speech", 0.0) + time.perf_counter() - t_sp
            if buf:
                t_d = time.perf_counter()
                wav = dec.push(np.stack(buf) * m.std + m.mean)
                st["t_dec"] = st.get("t_dec", 0.0) + time.perf_counter() - t_d
                yield wav
        st["frames"] = st.get("frames", 0) + nf
        st["aborted"] = aborted

    def stream(self, text, seed=None, guard=True):
        if seed is not None:
            rt.RNG.seed(seed)
        self.stats = {}
        if self.norm is not None:
            text = self.norm(text)
        for piece in (split_text(text) or [text]):
            yield from self.stream_piece(piece, guard=guard)

    # ---- reference: original whole-sentence path, single take (no retries), same normalizer ----
    def full(self, text, seed=None):
        if seed is not None:
            rt.RNG.seed(seed)
        if self.norm is not None:
            text = self.norm(text)
        m = self.m
        wavs = []
        for piece in (split_text(text) or [text]):
            lat = m.latents(piece, temp=self.temp, eos=self.eos)
            wavs.append(m.dec(lat * m.std + m.mean))
        return np.concatenate(wavs).astype(f32)

    def warmup(self):
        for _ in self.stream("Hello there, this is a warm up.", seed=0):
            pass
