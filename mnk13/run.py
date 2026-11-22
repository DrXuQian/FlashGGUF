#!/usr/bin/env python3
"""Test 13 fixed ZW810-rule dense-prefill shapes on a ZW810 or ZW810E."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

KIT = Path(__file__).resolve().parent
SRC = KIT / ".source-f16-v3"
RUNNER = SRC / "tools/run_prefill_model_asys.py"
STAGES = ("anchors", "fit", "holdout")


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def target_for(name, requested):
    target = {"PPU-ZW810": "zw810", "ZW810": "zw810",
              "PPU-ZW810E": "zw810e", "ZW810E": "zw810e"}.get(name.strip().upper())
    if target is None or requested not in ("auto", target):
        raise ValueError(f"requested {requested}, visible device is {name!r}; check --gpu")
    return target


def card_record(name, requested, d):
    target = target_for(name, requested)
    cu, warp, l2 = d["device"]["compute_units"], d["device"]["warp_size"], d["l2"]["bytes"]
    # ZW810E shares ppu_10 kernels/recipe IDs, not ZW810's CU count, L2 or latency.
    if (warp != 32 or cu <= 0 or l2 <= 0
            or (target == "zw810" and (cu != 72 or l2 != 64 << 20))):
        raise ValueError(f"unsupported {name} resources: CU={cu}, warp={warp}, L2={l2}")
    return dict(name=name, target=target, rule_source="zw810", **d)


def check_card(sdk, requested, verified_l2):
    # This process exits before the profiler starts; no retained Torch context.
    import torch
    from gguf.quants import dequantize  # noqa: F401
    from types import SimpleNamespace
    from tools.run_kpack_llama_ab import device
    name = torch.cuda.get_device_name(0)
    target = target_for(name, requested)
    # Zero tells the SDK probe to discover capacity, not to assume 64 MiB on E.
    props = torch.cuda.get_device_properties(0)
    torch_l2 = getattr(props, "L2_cache_size", 0) or getattr(props, "l2_cache_size", 0)
    l2 = verified_l2 or ((64 << 20) if target == "zw810" else torch_l2)
    d = device(SimpleNamespace(sdk=sdk, l2_bytes=l2))
    if target == "zw810e" and torch_l2 and not verified_l2 and not d["l2"]["reported"]:
        d["l2"]["source"] = "TORCH_DEVICE_PROPERTY"
    print("MNK13_DEVICE " + json.dumps(card_record(name, requested, d)), flush=True)


def logged(cmd, env, log):
    with log.open("a") as fp, subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True, errors="replace") as proc:
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                fp.write(line)
                fp.flush()
        except BaseException:
            proc.terminate()
            raise
        if proc.wait():
            raise RuntimeError(f"{Path(cmd[3]).name} rc={proc.returncode}; see {log}")


def review_table(dest, stage, points, target):
    from tools.review_prefill_cohort import ingest
    evidence = ingest(dest/stage)
    if (len(evidence["authorities"]) != 1 or next(iter(evidence["authorities"].values()))["mnk13_target"] != target):
        raise ValueError("capture was not collected for this target")
    if evidence["incomplete"] or set(evidence["points"]) != {p["id"] for p in points}:
        raise ValueError("not all frozen shapes have two confirmed samples")
    rows = []
    for p in points:
        w = p["workload"]
        measured = evidence["points"][p["id"]]["measurements"]
        if {r["case_id"] for r in measured} != set(p["catalog_ids"]):
            raise ValueError("a rejected or missing recipe cannot be counted as a measured loser")
        for r in measured:
            cs = r["confirmation_captures"]
            if r["status"] != "PASS" or [c["confirmation"] for c in cs] != [0, 1]:
                raise ValueError("incomplete direction or numerical/launch drift")
            all_us = r["complete_samples_us"]
            rows.append(dict(measured_on=target, rule_source="zw810", q=w["qtype"], M=w["m"], N=w["n"], K=w["k"], route=r["arm"],
                             catalog_recipe=p["catalog_ids"][r["case_id"]], case_id=r["case_id"],
                             roles="|".join(r["roles"]), production_table_hit=int(p["incumbent"] is not None),
                             calls=len(all_us), mean_us=statistics.fmean(all_us),
                             forward_mean_us=statistics.fmean(cs[0]["complete_samples_us"]),
                             reverse_mean_us=statistics.fmean(cs[1]["complete_samples_us"]),
                             two_capture_medians_us=r["gpu_sum_us"],
                             phases_us=json.dumps(r["phases_us"], sort_keys=True),
                             historical_control_us=p["historical_us"].get(r["case_id"], "")))
    with (dest/(stage+".csv")).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def bridge(rows, target):
    """Only a ZW810 run can bridge historical ZW810 microseconds."""
    result = []
    for n in (1024, 6144):
        controls = [r for r in rows if r["N"] == n and r["historical_control_us"] != ""]
        if len(controls) != 2 or {r["route"] for r in controls} != {"sf", "full-fp16"}:
            raise ValueError("both same-recipe historical controls are required")
        old = min(controls, key=lambda r: r["historical_control_us"])["route"]
        wins = [min(controls, key=lambda r: r[col])["route"] for col in ("forward_mean_us", "reverse_mean_us")]
        row = dict(n=n, measured_on=target, historical_device="zw810", historical=old, both_orders=wins,
                   same_card_winner=wins[0] if wins[0] == wins[1] else None)
        if target == "zw810":
            deltas = {r["route"]: 100*(r["mean_us"]/r["historical_control_us"]-1) for r in controls}
            row.update(delta_pct=deltas, comparable=wins == [old, old] and all(abs(d) <= 15 for d in deltas.values()))
        else:
            row.update(delta_pct=None, comparable=None, scope="E_CARD_PAIRED_ONLY_NO_ZW810_ABSOLUTE_GATE")
        result.append(row)
    return result


def compact(dest):
    # Deliberately exclude trace, SQLite, object code, cache, and full source manifests.
    with tarfile.open(dest/"review.tmp", "w:gz") as tar:
        for name in ("run.json", "device.json", "plan.json", "status.json", "anchor-check.json", *(s+".csv" for s in STAGES)):
            if (dest/name).is_file():
                tar.add(dest/name, name, recursive=False)
        for stage in STAGES:
            d = dest/stage
            for pattern in ("review-*.json.xz", "error-*.json", "failed-*.json", "authority.json", "modules.json",
                            "clock-lock-*.json", "clock-*.json", "summary.json"):
                files = sorted(d.glob(pattern))
                for f in (files[-1:] if pattern.startswith("review-") else files):
                    tar.add(f, stage+"/"+f.name, recursive=False)
            # Keep the actual compiler/profiler error, not just a pointer to a Box path.
            if not (dest/(stage+".csv")).exists():
                logs = sorted(d.rglob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]
                logs += [f for f in (dest/(stage+".log"), dest/"preflight.log") if f.is_file()]
                for i, f in enumerate(logs):
                    with f.open("rb") as inp:
                        inp.seek(max(0, f.stat().st_size-10000))
                        raw = (str(f)+"\n").encode()+inp.read()
                    info = tarfile.TarInfo(f"{stage}/diagnostic-{i}.txt")
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
    (dest/"review.tmp").replace(dest/"review.tgz")
    print("results=" + str(dest/"review.tgz"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0, help="one physical GPU ordinal")
    p.add_argument("--target", choices=("auto", "zw810", "zw810e"), default="auto",
                   help="check the actual SKU; E reuses candidates, not historical timings")
    p.add_argument("--l2-bytes", type=int, default=0,
                   help="only needed when the SDK reports zero: supply independently verified physical L2 bytes")
    p.add_argument("--sdk", type=Path, default=Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--phase", choices=("all", *STAGES), default="all")
    p.add_argument("--out", type=Path, help="new path or existing run for same GPU/SDK/kit")
    p.add_argument("--plan-only", action="store_true", help="exercise actual plan validation, no SDK or GPU")
    p.add_argument("--confirm-idle", action="store_true")
    p.add_argument("--allow-anchor-drift", action="store_true", help="continue as a separate timing domain")
    p.add_argument("--check-card", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    sys.path.insert(0, str(SRC))
    if a.check_card:
        check_card(a.sdk, a.target, a.l2_bytes)
        return 0
    if a.gpu < 0 or a.l2_bytes < 0 or not 1 <= a.jobs <= 64:
        p.error("nonnegative ordinal/L2 and 1..64 compiler jobs required")
    identity = read(KIT/"identity.json")
    for name, expected in identity["files"].items():
        path = SRC/name[len("source/"):] if name.startswith("source/") else KIT/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("kit payload changed: " + name)
    plan = read(KIT/"plan.json")
    print("MNK13_PLAN", plan["counts"], "target="+a.target, "stages=anchors:2 fit:7 holdout:4", flush=True)
    phases = STAGES if a.phase == "all" else (a.phase,)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=str(a.gpu))
    # Use this frozen source tree, not another Quactlize on PYTHONPATH.
    env["PYTHONPATH"] = str(SRC) + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    common = [sys.executable, "-B", "-u", str(RUNNER), "--finalist-plan", str(KIT/"plan.json"),
              "--no-native", "--configs-per-batch", "4", "--max-modules", "32", "--jobs", str(a.jobs),
              "--device", str(a.gpu)]
    def subset(stage):
        points = [p for p in plan["points"] if p["stage"] == stage]
        return points, [arg for row in points for arg in ("--point", row["id"])]
    if a.plan_only:
        with tempfile.TemporaryDirectory(prefix="mnk13-plan-") as t:
            for stage in phases:
                points, ids = subset(stage)
                subprocess.run([*common, *ids, "--plan-only", "--output", str(Path(t)/stage)], env=env, check=True)
                for point in points:
                    w = point["workload"]
                    print(stage, (w["qtype"], w["m"], w["n"], w["k"]), [c["arm"] for c in point["cases"]])
        return 0
    if not a.confirm_idle:
        p.error("confirm the selected GPU is idle, then pass --confirm-idle")
    sdk = a.sdk.resolve(strict=True)
    for entry in ("bin/hgcc", "asight/bin/asys"):
        if not os.access(sdk/entry, os.X_OK):
            p.error(f"missing executable: {sdk/entry}")
    env.update(PPU_SDK=str(sdk), CUDA_HOME=str(sdk/"CUDA_SDK"), PATH=str(sdk/"bin")+":"+env.get("PATH", ""))
    env["LD_LIBRARY_PATH"] = ":".join(map(str, (sdk/"lib", sdk/"targets/x86_64-linux/lib",
                        sdk/"CUDA_SDK/targets/x86_64-linux/lib"))) + ":" + env.get("LD_LIBRARY_PATH", "")
    out = a.out.resolve() if a.out else Path(tempfile.mkdtemp(prefix="flashgguf-mnk13-", dir=os.environ.get("OUTPUT_ROOT")
                    or ("/workspace" if os.access("/workspace", os.W_OK) else None))).resolve()
    out.mkdir(exist_ok=True)
    if any(out.iterdir()) and not (out/"run.json").is_file():
        p.error("output is not an MNK13 run; choose a new --out")
    print("MNK13_RUN", out, flush=True)
    with (out/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(gpu=a.gpu, sdk=str(sdk), kit=str(KIT), manifest=plan["sha256"],
                       kit_sha256=hashlib.sha256((KIT/"identity.json").read_bytes()).hexdigest(),
                       **{k:v for k,v in identity.items() if k!="files"})
        if (out/"run.json").is_file() and read(out/"run.json") != binding:
            p.error("resume with the same kit, GPU and SDK or use a fresh --out")
        save(out/"run.json", binding)
        save(out/"plan.json", plan)
        tic = time.monotonic()
        try:
            logged([sys.executable, "-B", "-u", __file__, "--check-card", "--sdk", str(sdk),
                    "--target", a.target, "--l2-bytes", str(a.l2_bytes)], env, out/"preflight.log")
            device = json.loads(next(line[len("MNK13_DEVICE "):] for line in reversed((out/"preflight.log").read_text().splitlines())
                                     if line.startswith("MNK13_DEVICE ")))
            if (out/"device.json").exists() and read(out/"device.json") != device:
                raise ValueError("physical card/SDK changed; start a new output rather than splicing old captures")
            save(out/"device.json", device)
            target = device["target"]
            env["FLASHGGUF_MNK13_TARGET"] = target
            for stage in phases:
                points, ids = subset(stage)
                if target == "zw810" and stage != "anchors" and (out/"anchor-check.json").exists() and not a.allow_anchor_drift:
                    if not all(r["comparable"] for r in read(out/"anchor-check.json")):
                        save(out/"status.json", dict(status="PAUSED_AFTER_VALID_ANCHORS", next_phase=stage))
                        print("MNK13_PAUSED: see anchor-check.json; inspect before continuing with --allow-anchor-drift.")
                        return 2
                folder = out/stage
                cmd = [*common, *ids, "--sdk", str(sdk), "--l2-bytes", str(device["l2"]["bytes"]),
                       "--confirm-idle", "--output", str(folder), "--cache", str(out/"jit")]
                if folder.exists():
                    cmd.append("--resume")
                setup = out/(stage+"-setup.json")
                if not setup.exists():
                    save(setup, dict(reuse=stage != "anchors" and (out/"anchors/build/receipt.json").exists()))
                if read(setup)["reuse"]:
                    cmd += ["--reuse-build", str(out/"anchors/build")]
                print(f"MNK13_START phase={stage} shapes={len(points)}", flush=True)
                logged(cmd, env, out/(stage+".log"))
                rows = review_table(out, stage, points, target)
                if stage == "anchors":
                    result = bridge(rows, target)
                    save(out/"anchor-check.json", result)
                    print("MNK13_ANCHORS", json.dumps(result), flush=True)
            save(out/"status.json", dict(status="PASS", measured_on=target, rule_source="zw810", phases=phases, seconds=time.monotonic()-tic,
                                         scope="PAIRED_ASYS_MEASUREMENT_NOT_PRODUCTION_ADMISSION"))
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            save(out/"status.json", dict(status="INCOMPLETE", error=repr(exc), requested=phases))
            print("MNK13_INCOMPLETE:", exc, file=sys.stderr)
            return 1
        finally:
            compact(out)


if __name__ == "__main__":
    raise SystemExit(main())
