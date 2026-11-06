#!/usr/bin/env python3
"""HomefrontImitation trainer: recordings of real players -> a policy and heat maps for HomefrontBots.

    python hfimitate.py stats  [recordings...]               what's been recorded, humans vs bots
    python hfimitate.py train  [recordings...] [--out DIR]   train policy.json and <map>.heat into DIR

Defaults (run from HomefrontImitation\\train, the repositories beside the dedicated server as Build.ps1 expects):
recordings from  Homefront Dedicated Server\\Binaries\\HomefrontSDK\\Mods\\HomefrontImitation\\recordings
output to        Homefront Dedicated Server\\Binaries\\HomefrontSDK\\Mods\\HomefrontBots\\imitation
Then "!bots imitation reload" in game (or the next map) picks it up. Needs Python 3.10+ and numpy only.

What it learns (behaviour cloning, from human samples only unless --include-bots):
  policy   an MLP from the Features vector (Shared/Features.cs) to what the player did: which way it moved in
           the frame while an enemy was in sight (9 classes), and whether it was crouched / sprinting / jumped in
           the next sample. Bots use it to move in fights and to pick their stance.
  heat     per map, per 48-unit floor cell: seconds humans spent there, seconds spent there in a fight, kills
           made from there and deaths there. Bots prefer routes humans walk and positions humans hold and win from.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import struct
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hfrec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))   # beside HomefrontImitation: HomefrontSDK, HomefrontBots, the server
MODS = os.path.join(PROJECT, "Homefront Dedicated Server", "Binaries", "HomefrontSDK", "Mods")
DEFAULT_IN = os.path.join(MODS, "HomefrontImitation", "recordings")
DEFAULT_OUT = os.path.join(MODS, "HomefrontBots", "imitation")

SPACING = 48.0        # Nav.cs grid
HALF_HEIGHT = 42.0    # soldier collision half height, for kill positions (centres) -> floor
HEAT_MAGIC = 0x54484648  # "HFHT"
HEAT_VERSION = 1
MOVE_CLASSES = 9
HEADS = {"move": {"offset": 0, "count": MOVE_CLASSES}, "crouch": 9, "sprint": 10, "jump": 11}
OUTPUTS = 12


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------------- loading

def load(paths: list[str], maps: set[str] | None) -> list[hfrec.Recording]:
    files = hfrec.find(paths)
    recs = []
    for f in files:
        try:
            r = hfrec.read(f)
        except (hfrec.FormatError, OSError, struct.error) as e:
            log(f"  skip {f}: {e}")
            continue
        if maps and r.map.lower() not in maps:
            continue
        if len(r.samples) == 0:
            continue
        recs.append(r)
    return recs


def is_bot(r: hfrec.Recording) -> np.ndarray:
    return (r.samples["flags"] & hfrec.BOT) != 0


# ------------------------------------------------------------------------------------------------------------ stats

def stats(args) -> int:
    recs = load(args.recordings or [DEFAULT_IN], set(m.lower() for m in args.maps) if args.maps else None)
    if not recs:
        log("no recordings found")
        return 1
    by_map = defaultdict(list)
    for r in recs:
        by_map[r.map].append(r)
    schemas = sorted({r.schema for r in recs})
    log(f"{len(recs)} recordings, {len(by_map)} maps, {len(schemas)} feature schema(s)")
    groups = {"humans": False, "bots": True}
    for m, rs in sorted(by_map.items()):
        log(f"\n{m}: {len(rs)} recording(s)")
        for label, want_bot in groups.items():
            secs = kills = deaths = fight = 0.0
            strafe = moved = crouch = sprint = jumps = 0
            dist = []
            soldiers = set()
            for r in rs:
                s = r.samples
                sel = is_bot(r) == want_bot
                if not sel.any():
                    continue
                ss = s[sel]
                soldiers |= {(r.name, int(p)) for p in np.unique(ss["pid"])}
                secs += len(ss) * r.sample_seconds
                enemy = (ss["flags"] & hfrec.HAS_ENEMY) != 0
                fight += enemy.sum() * r.sample_seconds
                acc = ss["acc"][:, :2] / np.maximum(ss["accel_rate"], 1.0)[:, None]
                f, rt = hfrec.in_frame(acc, ss["frame"])
                cls = hfrec.move_class(f, rt)
                moved += int((cls[enemy] != 0).sum())
                strafe += int(np.isin(cls[enemy], [2, 3, 4, 6, 7, 8]).sum())
                crouch += int(((ss["flags"] & hfrec.CROUCHED) != 0).sum())
                sprint += int(((ss["flags"] & hfrec.SPRINTING) != 0).sum())
                jumps += int(((ss["flags"] & hfrec.JUMPED) != 0).sum())
                if enemy.any():
                    dist.append(ss["feat"][enemy, 1] * 4000.0)
                bots = {p for p, (_, b) in r.soldiers.items() if b == want_bot}
                k = r.kills
                kills += int(np.isin(k["killer"], list(bots)).sum()) if len(k) else 0
                deaths += int(np.isin(k["victim"], list(bots)).sum()) if len(k) else 0
            if secs == 0:
                continue
            n = secs / rs[0].sample_seconds
            fs = max(fight / rs[0].sample_seconds, 1)
            d = np.concatenate(dist) if dist else np.zeros(0)
            log(f"  {label:6}: {len(soldiers)} soldier-sessions, {secs / 60:.1f} min alive, "
                f"{kills:.0f} kills / {deaths:.0f} deaths (K/D {kills / max(deaths, 1):.2f}, "
                f"{kills / max(secs / 60, 1e-9):.2f} kills/min)")
            log(f"          in a fight {100 * fight / secs:.0f}% of the time"
                + (f", median range {np.median(d):.0f} u" if len(d) else "")
                + f"; in fights moving {100 * moved / fs:.0f}% (strafing {100 * strafe / fs:.0f}%); "
                  f"crouched {100 * crouch / n:.0f}%, sprinting {100 * sprint / n:.0f}%, {jumps / max(secs / 60, 1e-9):.1f} jumps/min")
    return 0


# ---------------------------------------------------------------------------------------------------------- dataset

def dataset(recs: list[hfrec.Recording], include_bots: bool):
    """X features, y_move (-1 where not a fight), y_crouch/y_sprint/y_jump of the next sample, group (recording)."""
    xs, moves, crouch, sprint, jump, groups, times = [], [], [], [], [], [], []
    for gi, r in enumerate(recs):
        s = r.samples
        keep = (s["flags"] & hfrec.LADDER) == 0
        if not include_bots:
            keep &= ~is_bot(r)
        keep &= s["accel_rate"] > 0
        if not keep.any():
            continue
        s = s[keep]
        # The next sample of the same soldier, at most 1.5 sample intervals later: stance labels come from it.
        order = np.lexsort((s["t"], s["pid"]))
        s = s[order]
        nxt = np.arange(1, len(s) + 1)
        nxt[-1] = len(s) - 1
        ok = (s["pid"][nxt] == s["pid"]) & (s["t"][nxt] - s["t"] > 0) & \
             (s["t"][nxt] - s["t"] <= 1.5 * r.sample_seconds) & (np.arange(len(s)) != len(s) - 1)
        if not ok.any():
            continue
        cur, n = s[ok], s[nxt[ok]]
        acc = cur["acc"][:, :2] / cur["accel_rate"][:, None]
        f, rt = hfrec.in_frame(acc, cur["frame"])
        move = hfrec.move_class(f, rt)
        fight = cur["feat"][:, 0] > 0.5
        moves.append(np.where(fight, move, -1))
        xs.append(cur["feat"].astype(np.float32))
        crouch.append(((n["flags"] & hfrec.CROUCHED) != 0).astype(np.float32))
        sprint.append(((n["flags"] & hfrec.SPRINTING) != 0).astype(np.float32))
        jump.append(((n["flags"] & hfrec.JUMPED) != 0).astype(np.float32))
        groups.append(np.full(len(cur), gi, dtype=np.int32))
        times.append(cur["t"].astype(np.float64))
    if not xs:
        return None
    return (np.concatenate(xs), np.concatenate(moves), np.concatenate(crouch), np.concatenate(sprint),
            np.concatenate(jump), np.concatenate(groups), np.concatenate(times))


# ------------------------------------------------------------------------------------------------------------- model

class MLP:
    """ReLU MLP, float32, Adam. Outputs: 9 move logits, then crouch, sprint, jump logits."""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.W = [(rng.standard_normal((o, i)) * math.sqrt(2.0 / i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])]
        self.b = [np.zeros(o, dtype=np.float32) for o in sizes[1:]]
        self.W[-1] *= 0.1
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.step = 0

    def forward(self, x: np.ndarray, keep: bool = False):
        acts = [x]
        h = x
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if li < len(self.W) - 1:
                h = np.maximum(h, 0.0, dtype=np.float32)
            acts.append(h)
        return (h, acts) if keep else h

    def backward(self, acts, grad_out):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = grad_out
        for li in range(len(self.W) - 1, -1, -1):
            a_in = acts[li]
            gW[li] = g.T @ a_in
            gb[li] = g.sum(axis=0)
            if li > 0:
                g = (g @ self.W[li]) * (acts[li] > 0)
        return gW, gb

    def adam(self, gW, gb, lr, wd=1e-5, b1=0.9, b2=0.999, eps=1e-8):
        self.step += 1
        params = self.W + self.b
        grads = [g + wd * p for g, p in zip(gW, self.W)] + gb
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.step)
            vh = self.v[i] / (1 - b2 ** self.step)
            p -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def losses(out, ym, yc, ys, yj, jump_weight):
    """Total loss and its gradient w.r.t. the outputs (per-batch mean)."""
    n = len(out)
    grad = np.zeros_like(out)
    lsm = log_softmax(out[:, :MOVE_CLASSES])
    fight = ym >= 0
    nf = max(int(fight.sum()), 1)
    move_loss = -lsm[fight, ym[fight]].sum() / nf
    p = np.exp(lsm)
    g = p.copy()
    g[np.arange(n), np.where(fight, ym, 0)] -= 1.0
    g[~fight] = 0
    grad[:, :MOVE_CLASSES] = g / nf
    total = move_loss
    for col, y, w in ((HEADS["crouch"], yc, 1.0), (HEADS["sprint"], ys, 1.0), (HEADS["jump"], yj, jump_weight)):
        z = out[:, col]
        q = sigmoid(z)
        wt = np.where(y > 0.5, w, 1.0)
        total += float((wt * (np.logaddexp(0, z) - y * z)).mean())
        grad[:, col] = wt * (q - y) / n
    return total, move_loss, grad


def auc(y, score):
    pos, neg = y > 0.5, y <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def train_policy(data, args, rng):
    X, ym, yc, ys, yj, grp, tt = data
    n, F = X.shape
    # Validation: whole recordings held out when there are a few, else a random slice.
    groups = np.unique(grp)
    if len(groups) >= 4:
        val_groups = rng.choice(groups, size=max(1, int(round(len(groups) * args.val))), replace=False)
        val = np.isin(grp, val_groups)
    else:
        # Few recordings: the last part of each (by time) is held out. A random per-sample split would put
        # 10 Hz neighbours on both sides and early stopping would never stop.
        val = np.zeros(n, dtype=bool)
        for g in groups:
            sel = np.flatnonzero(grp == g)
            cut = np.quantile(tt[sel], 1.0 - args.val)
            val[sel[tt[sel] >= cut]] = True
    tr = ~val
    if tr.sum() < 100 or val.sum() < 20:
        raise SystemExit(f"not enough data to train: {tr.sum()} training / {val.sum()} validation samples")
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std = np.where(std < 1e-3, 1.0, std).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)

    sizes = [F] + args.hidden + [OUTPUTS]
    model = MLP(sizes, rng)
    jump_rate = max(float(yj[tr].mean()), 1e-4)
    jump_weight = float(min(20.0, 0.1 / jump_rate)) if jump_rate < 0.1 else 1.0
    tri = np.flatnonzero(tr)
    best, best_state, bad = float("inf"), None, 0
    log(f"training {sizes} on {len(tri)} samples ({int((ym[tr] >= 0).sum())} in fights), "
        f"validating on {int(val.sum())}; jump weight {jump_weight:.1f}")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(tri)
        lr = args.lr * (0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        tot = 0.0
        for k in range(0, len(tri), args.batch):
            b = tri[k:k + args.batch]
            out, acts = model.forward(Xn[b], keep=True)
            loss, _, g = losses(out, ym[b], yc[b], ys[b], yj[b], jump_weight)
            gW, gb = model.backward(acts, g.astype(np.float32))
            model.adam(gW, gb, lr)
            tot += loss * len(b)
        vout = model.forward(Xn[val])
        vloss, vmove, _ = losses(vout, ym[val], yc[val], ys[val], yj[val], jump_weight)
        log(f"  epoch {epoch:3}: train {tot / len(tri):.4f}  val {vloss:.4f} (move {vmove:.4f})")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = ([w.copy() for w in model.W], [b.copy() for b in model.b])
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  stopping: no better for {args.patience} epochs")
                break
    if best_state is not None:
        model.W, model.b = best_state

    # Validation metrics against the obvious baselines.
    vout = model.forward(Xn[val])
    vf = ym[val] >= 0
    metrics = {"train_samples": int(tr.sum()), "val_samples": int(val.sum()), "val_fight_samples": int(vf.sum())}
    if vf.any():
        yv = ym[val][vf]
        prior = np.bincount(ym[tr][ym[tr] >= 0], minlength=MOVE_CLASSES).astype(np.float64) + 1
        prior /= prior.sum()
        lsm = log_softmax(vout[vf, :MOVE_CLASSES])
        metrics.update({
            "move_accuracy": float((lsm.argmax(axis=1) == yv).mean()),
            "move_baseline_accuracy": float((yv == prior.argmax()).mean()),
            "move_logloss": float(-lsm[np.arange(len(yv)), yv].mean()),
            "move_baseline_logloss": float(-np.log(prior[yv]).mean()),
            "move_class_rates": [float(x) for x in np.bincount(yv, minlength=MOVE_CLASSES) / len(yv)],
        })
    for name, y in (("crouch", yc), ("sprint", ys), ("jump", yj)):
        metrics[f"{name}_rate"] = float(y[val].mean())
        metrics[f"{name}_auc"] = auc(y[val], vout[:, HEADS[name]])
    # JSON has no NaN (System.Text.Json rejects it): an undefined metric is null.
    metrics = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in metrics.items()}
    return model, mean.astype(np.float32), std, metrics


def export_policy(path, model, mean, std, schema, metrics, recs, n):
    layers = []
    for li, (W, b) in enumerate(zip(model.W, model.b)):
        layers.append({"in": int(W.shape[1]), "out": int(W.shape[0]),
                       "act": "relu" if li < len(model.W) - 1 else "linear",
                       "w": [float(v) for v in W.reshape(-1)], "b": [float(v) for v in b]})
    doc = {
        "format": "homefront-imitation-policy", "version": 1, "schema": schema,
        "trained_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "samples": int(n), "recordings": len(recs), "maps": sorted({r.map for r in recs}),
        "mean": [float(v) for v in mean], "std": [float(v) for v in std],
        "layers": layers, "heads": HEADS, "metrics": metrics,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


# -------------------------------------------------------------------------------------------------------------- heat

def heat(recs: list[hfrec.Recording], include_bots: bool, out_dir: str) -> list[str]:
    """Per map: floor cells (Nav.cs grid) with human seconds, fight seconds, kills from and deaths at."""
    by_map = defaultdict(list)
    for r in recs:
        by_map[r.map.lower()].append(r)
    written = []
    for m, rs in sorted(by_map.items()):
        cells = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0])   # dwell, fight, z sum, z count, kills, deaths
        human_seconds = 0.0

        def key(x, y, z):
            return (int(math.floor(x / SPACING + 0.5)), int(math.floor(y / SPACING + 0.5)), int(round(z / 64.0)))

        for r in rs:
            s = r.samples
            sel = (s["flags"] & hfrec.LADDER) == 0
            if not include_bots:
                sel &= ~is_bot(r)
            s = s[sel]
            human_seconds += len(s) * r.sample_seconds
            fight = (s["flags"] & hfrec.HAS_ENEMY) != 0
            cx = np.floor(s["loc"][:, 0] / SPACING + 0.5).astype(np.int64)
            cy = np.floor(s["loc"][:, 1] / SPACING + 0.5).astype(np.int64)
            cz = np.round(s["floor_z"] / 64.0).astype(np.int64)
            keys, inv = np.unique(np.stack([cx, cy, cz], axis=1), axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            dwell = np.bincount(inv, minlength=len(keys)) * r.sample_seconds
            fsec = np.bincount(inv, weights=fight.astype(np.float64), minlength=len(keys)) * r.sample_seconds
            zsum = np.bincount(inv, weights=s["floor_z"].astype(np.float64), minlength=len(keys))
            zcnt = np.bincount(inv, minlength=len(keys))
            for k, d, fs, zs, zc in zip(map(tuple, keys), dwell, fsec, zsum, zcnt):
                c = cells[k]
                c[0] += d
                c[1] += fs
                c[2] += zs
                c[3] += int(zc)
            who = {p: b for p, (_, b) in r.soldiers.items()}
            for k in r.kills:
                killer, victim = int(k["killer"]), int(k["victim"])
                if killer != hfrec.NO_PID and (include_bots or not who.get(killer, True)) and k["killer_loc"].any():
                    kl = k["killer_loc"]
                    c = cells[key(kl[0], kl[1], kl[2] - HALF_HEIGHT)]
                    c[4] += 1
                    if c[3] == 0:
                        c[2], c[3] = kl[2] - HALF_HEIGHT, 1
                if include_bots or not who.get(victim, True):
                    vl = k["victim_loc"]
                    c = cells[key(vl[0], vl[1], vl[2] - HALF_HEIGHT)]
                    c[5] += 1
                    if c[3] == 0:
                        c[2], c[3] = vl[2] - HALF_HEIGHT, 1
        if not cells:
            continue
        path = os.path.join(out_dir, f"{m}.heat")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<IHfIfH", HEAT_MAGIC, HEAT_VERSION, SPACING, len(cells), human_seconds, len(rs)))
            for (cx, cy, _), (dwell, fs, zs, zc, kills, deaths) in cells.items():
                f.write(struct.pack("<iifffHH", cx, cy, zs / max(zc, 1), dwell, fs, min(kills, 65535), min(deaths, 65535)))
        os.replace(tmp, path)
        written.append(path)
        log(f"  heat {m}: {len(cells)} cells, {human_seconds / 60:.1f} soldier-minutes, {len(rs)} recording(s)")
    return written


# ------------------------------------------------------------------------------------------------------------- train

def train(args) -> int:
    rng = np.random.default_rng(args.seed)
    recs = load(args.recordings or [DEFAULT_IN], set(m.lower() for m in args.maps) if args.maps else None)
    if not recs:
        log("no recordings found")
        return 1
    # One feature layout per policy: the newest recordings' schema.
    schema = max(recs, key=lambda r: r.start).schema
    same = [r for r in recs if r.schema == schema]
    if len(same) < len(recs):
        log(f"using {len(same)} of {len(recs)} recordings (the others have an older feature schema)")
    out = args.out or DEFAULT_OUT
    os.makedirs(out, exist_ok=True)

    log(f"{len(recs)} recordings over {len({r.map for r in recs})} map(s)")
    heat(recs, args.include_bots, out)

    data = dataset(same, args.include_bots)
    if data is None:
        log("no usable samples (were there humans playing? --include-bots trains on bots too)")
        return 1
    model, mean, std, metrics = train_policy(data, args, rng)
    path = os.path.join(out, "policy.json")
    export_policy(path, model, mean, std, schema, metrics, same, len(data[0]))
    log(f"wrote {path}")
    for k, v in metrics.items():
        if isinstance(v, float):
            log(f"  {k:26} {v:.4f}")
        elif v is None:
            log(f"  {k:26} n/a")
    if "move_accuracy" in metrics:
        log(f"  fight movement: {100 * metrics['move_accuracy']:.1f}% right vs {100 * metrics['move_baseline_accuracy']:.1f}% "
            f"always guessing the commonest; log-loss {metrics['move_logloss']:.3f} vs {metrics['move_baseline_logloss']:.3f}")
    log('in game: "!bots imitation reload" (or change map)')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "train"):
        p = sub.add_parser(name)
        p.add_argument("recordings", nargs="*", help=f"files or folders (default {DEFAULT_IN})")
        p.add_argument("--maps", nargs="+", help="only these maps")
        if name == "train":
            p.add_argument("--out", help=f"output folder (default {DEFAULT_OUT})")
            p.add_argument("--include-bots", action="store_true", help="learn from bot samples too (testing)")
            p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--batch", type=int, default=512)
            p.add_argument("--lr", type=float, default=2e-3)
            p.add_argument("--val", type=float, default=0.15, help="fraction held out")
            p.add_argument("--patience", type=int, default=4)
            p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    return stats(args) if args.cmd == "stats" else train(args)


if __name__ == "__main__":
    sys.exit(main())
