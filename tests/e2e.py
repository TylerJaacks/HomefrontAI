#!/usr/bin/env python3
"""End-to-end test outside the game: the C# harness writes scripted recordings with the real RecordingWriter and
Features, the trainer learns from them, and the harness checks the policy (C# forward pass == numpy's, label
conventions agree, the script was learned) and the heat files parse.

    python tests/e2e.py            (needs the .NET 10 SDK and numpy)
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))
import hfimitate  # noqa: E402
import hfrec  # noqa: E402


def harness(*args):
    cmd = ["dotnet", "run", "--project", os.path.join(HERE, "Harness"), "-c", "Release", "--", *args]
    print("$", " ".join(cmd[-len(args) - 1:]), flush=True)
    return subprocess.run(cmd).returncode


def shared_in_sync():
    """HomefrontBots carries copies of Shared/*.cs: with its checkout beside this one, they must be identical."""
    bots = os.path.join(os.path.dirname(ROOT), "HomefrontBots", "Imitation")
    if not os.path.isdir(bots):
        print("(no HomefrontBots checkout beside this one: copies not compared)")
        return
    for name in ("Features.cs", "Policy.cs", "Senses.cs"):
        a = open(os.path.join(ROOT, "Shared", name), "rb").read()
        b = open(os.path.join(bots, name), "rb").read()
        assert a == b, f"HomefrontBots/Imitation/{name} differs from Shared/{name}: copy it over"
    print("HomefrontBots/Imitation has the same shared code")


def main():
    shared_in_sync()
    work = tempfile.mkdtemp(prefix="hfimitation-e2e-")
    recs, out = os.path.join(work, "recordings"), os.path.join(work, "out")
    try:
        assert harness("synth", recs, "5") == 0, "synth failed"

        r = hfrec.read(hfrec.find([recs])[0])
        assert r.complete and r.map == "synth-arena" and len(r.samples) == 6000 * 8, (r.complete, len(r.samples))
        assert r.feature_count == len(r.feature_names) == r.samples["feat"].shape[1]
        print(f"read back {r.name}: {len(r.samples)} samples, {len(r.kills)} kills, strings {r.strings}")

        assert hfimitate.main(["stats", recs]) == 0
        assert hfimitate.main(["train", recs, "--out", out, "--epochs", "25"]) == 0

        # Heat file: header and cells parse; human time only (bots excluded).
        hp = os.path.join(out, "synth-arena.heat")
        data = open(hp, "rb").read()
        magic, ver, spacing, count, secs, sessions = struct.unpack_from("<IHfIfH", data, 0)
        assert magic == hfimitate.HEAT_MAGIC and ver == 1 and spacing == 48 and sessions == 5
        assert len(data) == 20 + 24 * count, (len(data), count)
        assert abs(secs - 5 * 6000 * 6 * 0.1) < 1.0, secs   # 6 humans x 6000 samples x 5 files
        print(f"heat: {count} cells, {secs / 60:.0f} human minutes")

        # Parity vectors: real recorded features, numpy logits.
        policy = json.load(open(os.path.join(out, "policy.json")))
        print("metrics:", {k: round(v, 3) for k, v in policy["metrics"].items() if isinstance(v, float)})
        x = r.samples["feat"][::997][:64].astype(np.float32)
        mean, std = np.array(policy["mean"], np.float32), np.array(policy["std"], np.float32)
        h = (x - mean) / std
        for l in policy["layers"]:
            W = np.array(l["w"], np.float32).reshape(l["out"], l["in"])
            h = h @ W.T + np.array(l["b"], np.float32)
            if l["act"] == "relu":
                h = np.maximum(h, 0)
        rng = np.random.default_rng(3)
        mv = rng.uniform(-1.2, 1.2, size=(500, 2)).astype(np.float32)
        # Near the boundaries too: just either side of the half-way angles, and of the still threshold. (Exactly on
        # a half-way angle, float32 and float64 may round either way; that's measure zero in real input.)
        angles = np.concatenate([np.arange(16) * np.pi / 8 + 0.004, np.arange(16) * np.pi / 8 - 0.004])
        edge = np.array([[np.cos(a), np.sin(a)] for a in angles], np.float32)
        mv = np.concatenate([mv, edge, edge * 0.29, edge * 0.31])
        cls = hfrec.move_class(mv[:, 0], mv[:, 1])
        parity = {
            "synthetic": True,
            "features": x.tolist(), "logits": h.tolist(),
            "moves": [{"fwd": float(a), "right": float(b), "cls": int(c)} for (a, b), c in zip(mv, cls)],
        }
        pp = os.path.join(work, "parity.json")
        json.dump(parity, open(pp, "w"))
        assert harness("check", os.path.join(out, "policy.json"), pp) == 0, "C# check failed"
        print("\nE2E PASSED")
        return 0
    finally:
        if "--keep" in sys.argv:
            print("kept", work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
