"""Load a trained head and score raw 16 kHz audio (Whistle embed of the last 5 s -> head)."""
import os

import numpy as np
import torch

from train_heads import DIM, HERE, T, Norm, build

SR = 16000


def load_head(name):
    ck = torch.load(os.path.join(HERE, "heads", name + ".pt"), weights_only=False)
    model = build(name, Norm(ck["mean"], ck["std"]))
    model.load_state_dict(ck["state"])
    return model.eval()


def frames_to_input(e):
    """[n, 512] frames -> right-aligned ([1, T, 512], [1, T]) arrays."""
    e = e[-T:]
    x = np.zeros((1, T, DIM), dtype=np.float32)
    m = np.zeros((1, T), dtype=np.float32)
    x[0, T - len(e):] = e
    m[0, T - len(e):] = 1
    return x, m


def embed_last5(w, audio):
    e = w.embed(np.ascontiguousarray(audio[-5 * SR:], dtype=np.float32))
    return np.asarray(e, dtype=np.float32).reshape(-1, DIM)


def score(model, w, audio):
    x, m = frames_to_input(embed_last5(w, audio))
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.from_numpy(x), torch.from_numpy(m)))[0])


def export_onnx(model, path):
    x, m = torch.zeros(1, T, DIM), torch.ones(1, T)
    torch.onnx.export(model, (x, m), path, input_names=["frames", "mask"], output_names=["logit"],
                      opset_version=17, dynamo=False)
    return path
