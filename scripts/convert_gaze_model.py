#!/usr/bin/env python3
"""Convert an appearance-based gaze model to Core ML for GazeFuse (run locally, by you).

Source: yakhyo/gaze-estimation (code MIT, https://github.com/yakhyo/gaze-estimation) — an
L2CS-Net-style (Abdelrahman et al. 2022, code MIT) binned yaw/pitch classifier+regressor.

LICENSE NOTE: the *code* is MIT (compatible with Apache-2.0), but the released *weights* were
trained on Gaze360, whose dataset licence restricts use to non-commercial research. GazeFuse
therefore does NOT ship or redistribute weights or the converted model. This script downloads
the weights from the upstream release onto your machine and converts them for personal /
research use. Check the dataset terms before any other use.

Output: GazeAppearance.mlpackage with
  input  "face"  : RGB image 448×448 (face crop, un-mirrored)
  output "gaze"  : [yaw, pitch] in radians (L2CS convention: yaw + = subject's left as seen
                   by the camera, pitch + = up), via softmax-expectation over 90 bins of 4°.

Usage:
  python3 -m venv .venv && .venv/bin/pip install torch torchvision coremltools numpy pillow  # (no OpenCV needed)
  .venv/bin/python scripts/convert_gaze_model.py --arch mobileone_s0 --out ~/Library/Application\\ Support/GazeFuse/GazeAppearance.mlpackage
"""
import argparse, os, subprocess, sys, tempfile, urllib.request

ARCHS = {
    "mobileone_s0": "mobileone_s0.pt",
    "mobilenetv2": "mobilenetv2.pt",
    "resnet18": "resnet18.pt",
}
RELEASE = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"
REPO = "https://github.com/yakhyo/gaze-estimation.git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobileone_s0", choices=sorted(ARCHS))
    ap.add_argument("--workdir", default=os.path.expanduser("~/.cache/gazefuse-model"))
    ap.add_argument("--out", default="GazeAppearance.mlpackage")
    ap.add_argument("--fp16", action="store_true", default=True)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    src = os.path.join(a.workdir, "gaze-estimation")
    if not os.path.isdir(src):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, src])
    wpath = os.path.join(a.workdir, ARCHS[a.arch])
    if not os.path.exists(wpath):
        print("downloading", RELEASE + ARCHS[a.arch])
        urllib.request.urlretrieve(RELEASE + ARCHS[a.arch], wpath)
    sys.path.insert(0, src)

    import numpy as np
    import torch
    import torch.nn.functional as F
    import coremltools as ct
    import models as gm  # upstream package; avoids utils/ (needs OpenCV)

    def get_model(arch, bins, inference_mode=True):
        if arch == "mobileone_s0":
            return gm.mobileone_s0(pretrained=False, num_classes=bins, inference_mode=inference_mode)
        if arch == "mobilenetv2":
            return gm.mobilenet_v2(pretrained=False, num_classes=bins)
        return gm.resnet18(pretrained=False, num_classes=bins)

    bins, binwidth, angle = 90, 4.0, 180.0
    net = get_model(a.arch, bins, inference_mode=True)
    state = torch.load(wpath, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    net.load_state_dict(state)
    net.eval()

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.register_buffer("idx", torch.arange(bins, dtype=torch.float32))

        def forward(self, x):  # x: 0...1 RGB
            yaw, pitch = self.m((x - self.mean) / self.std)
            y = (F.softmax(yaw, 1) * self.idx).sum(1) * binwidth - angle
            p = (F.softmax(pitch, 1) * self.idx).sum(1) * binwidth - angle
            return torch.stack([y, p], 1) * (np.pi / 180.0)

    w = Wrapped(net).eval()
    ex = torch.rand(1, 3, 448, 448)
    with torch.no_grad():
        ref = w(ex).numpy()
        traced = torch.jit.trace(w, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.ImageType(name="face", shape=(1, 3, 448, 448), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="gaze")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16 if a.fp16 else ct.precision.FLOAT32,
    )
    ml.short_description = f"GazeFuse appearance gaze ({a.arch}, yakhyo/gaze-estimation weights, Gaze360-trained; non-commercial research use)"
    ml.author = "converted locally by scripts/convert_gaze_model.py"
    ml.license = "Code MIT; weights trained on Gaze360 (non-commercial research licence). Not redistributed by GazeFuse."
    ml.save(a.out)
    # Parity check against PyTorch on the same random input.
    from PIL import Image
    img = Image.fromarray((ex[0].permute(1, 2, 0).numpy() * 255).astype("uint8"))
    with torch.no_grad():
        ref = w(torch.from_numpy(np.asarray(img).astype("float32") / 255).permute(2, 0, 1)[None]).numpy()
    try:
        got = ml.predict({"face": img})["gaze"]
        print("parity torch", ref.ravel(), "coreml", np.asarray(got).ravel())
    except Exception as e:  # prediction needs macOS
        print("skipped Core ML parity check:", e)
    print("saved", a.out)


if __name__ == "__main__":
    main()
