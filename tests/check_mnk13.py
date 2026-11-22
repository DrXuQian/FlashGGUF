"""Host regression tests for an extracted kit. Usage: python3 -B check_kit.py DIR"""
import copy
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

kit = Path(sys.argv.pop(1)).resolve(strict=True)
spec = importlib.util.spec_from_file_location("mnk13_test_runner", kit/"run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
sys.path.insert(0, str(run.SRC))
from dev.prefill_asys.finalists import load
from dev.strategy.resources import inventory
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.tuning import digest
from tools.run_prefill_model_asys import confirmation_jobs
from tools.run_prefill_model_asys import mnk13_f16_sf_only
from tools.review_prefill_cohort import domain
from tools.build_kpack_dequant import inspect_image
from quactlize.dequant.native import verify as verify_dequant
from quactlize.runtime.compiler import sha


class KitTest(unittest.TestCase):
    def setUp(self):
        self.plan = load(json.loads((kit/"plan.json").read_bytes()))

    def test_sealed(self):
        p = copy.deepcopy(self.plan)
        p["points"].pop()
        with self.assertRaises(ValueError):
            load(p)

    def test_sources_and_orders(self):
        inv = inventory(self.plan)
        self.assertEqual(len(inv["modules"]), 5)
        for mid, s in inv["modules"].items():
            self.assertIn(s["parent"]["symbol"], Compiler.source(None, s["parent"], mid))
        for p in self.plan["points"]:
            self.assertEqual(len(p["cases"]), 3)
            a, b = confirmation_jobs(p)
            self.assertEqual(a[0][::-1], b[0])
            self.assertEqual((a[1], b[1]), (0, 1))
            if p["stage"] == "holdout":
                self.assertFalse(p["historical_us"])

    def test_altered_order_not_accepted_even_if_resealed(self):
        p = copy.deepcopy(self.plan)
        p["points"].reverse()
        p["sha256"] = digest({k:v for k,v in p.items() if k!="sha256"})
        with self.assertRaisesRegex(ValueError, "stage/shape"):
            load(p)

    def test_reviewer(self):
        p = self.plan["points"][0]
        ms = []
        for c in p["cases"]:
            cs = [dict(confirmation=i, complete_samples_us=vals) for i, vals in enumerate(([2, 7, 6], [9, 3, 3]))]
            ms.append(dict(case_id=c["id"], status="PASS", confirmation_captures=cs,
                           complete_samples_us=[2,7,6,9,3,3], arm=c["arm"], roles=c["roles"],
                           gpu_sum_us=4.5, phases_us={"main":4.5}))
        simulated = dict(authorities={"test":dict(mnk13_target="zw810e")}, incomplete=[], points={p["id"]:dict(measurements=ms)})
        with tempfile.TemporaryDirectory() as t, patch("tools.review_prefill_cohort.ingest",return_value=simulated):
            root = Path(t)
            run.review_table(root,"anchors",[p],"zw810e")
            with (root/"anchors.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertEqual(float(rows[0]["mean_us"]),statistics.mean([2,7,6,9,3,3]))
            self.assertEqual({r["measured_on"] for r in rows},{"zw810e"})
            with self.assertRaisesRegex(ValueError,"target"):
                run.review_table(root,"anchors",[p],"zw810")
            ms.pop()
            with self.assertRaisesRegex(ValueError,"missing"):
                run.review_table(root,"anchors",[p],"zw810e")

    def test_bridge_drift(self):
        rows = [dict(N=n, route=r, historical_control_us=old, forward_mean_us=old, reverse_mean_us=old, mean_us=old)
                for n, t in ((1024,(31.,46.)), (6144,(119.,108.))) for r,old in zip(("sf","full-fp16"),t)]
        self.assertTrue(all(b["comparable"] for b in run.bridge(rows,"zw810")))
        rows[0]["reverse_mean_us"] = 50
        self.assertFalse(run.bridge(rows,"zw810")[0]["comparable"])
        e = run.bridge(rows,"zw810e")[0]
        self.assertEqual(e["both_orders"],["sf","full-fp16"])
        self.assertIsNone(e["comparable"])
        self.assertIsNone(e["same_card_winner"])
        self.assertIsNone(e["delta_pct"])

    def test_device_aliases_and_real_e_resources(self):
        def props(cu,l2,warp=32):
            return dict(device=dict(compute_units=cu,warp_size=warp),l2=dict(bytes=l2))
        for cu, l2 in ((64,32<<20),(72,64<<20),(80,96<<20)):
            for name in ("PPU-ZW810E","ZW810E"):
                e=run.card_record(name,"zw810e",props(cu,l2))
                self.assertEqual((e["target"],e["device"]["compute_units"],e["l2"]["bytes"]),("zw810e",cu,l2))
        for name,cu,l2,warp in (("PPU-ZW610",64,64,32),("ZW810E-OTHER",72,64,32),
                                ("ZW810E",0,64,32),("ZW810E",72,0,32),("ZW810E",72,64,64),
                                ("ZW810",64,64,32),("ZW810",72,32,32)):
            with self.assertRaises(ValueError):
                run.card_record(name,"auto",props(cu,l2<<20,warp))
        with self.assertRaisesRegex(ValueError,"requested"):
            run.card_record("PPU-ZW810","zw810e",props(72,64<<20))
        self.assertEqual(run.card_record("PPU-ZW810","auto",props(72,64<<20))["target"],"zw810")

    def test_review_domains_cannot_mix(self):
        board = dict(device=dict(compute_units=72,warp_size=32,max_threads=1024,DriverGetVersion=13000,
                                 RuntimeGetVersion=13000),sdk="same-sdk",l2=dict(bytes=64<<20))
        def d(target, lock):
            return domain(dict(device=board,mnk13_target=target,mnk13_clock_lock=lock))
        self.assertNotEqual(d("zw810e","REQUEST_ACCEPTED"),d("zw810","REQUEST_ACCEPTED"))
        self.assertNotEqual(d("zw810e","REQUEST_ACCEPTED"),d("zw810e","REQUEST_DENIED"))

    def test_e_probe_does_not_supply_810_l2(self):
        props = SimpleNamespace(L2_cache_size=0)
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda i:"PPU-ZW810E",
                                                          get_device_properties=lambda i:props))
        imports = {"torch":torch_stub,"gguf.quants":SimpleNamespace(dequantize=lambda:None)}
        def sdk_probe(args):
            return dict(device=dict(compute_units=60,warp_size=32),
                        l2=dict(bytes=args.l2_bytes or 48<<20, reported=0 if args.l2_bytes else 48<<20))
        with patch.dict(sys.modules,imports), patch("tools.run_kpack_llama_ab.device",side_effect=sdk_probe) as probe:
            with contextlib.redirect_stdout(io.StringIO()) as log:
                run.check_card(Path("/sdk"),"zw810e",0)
            self.assertEqual(probe.call_args.args[0].l2_bytes,0)
            self.assertEqual(json.loads(log.getvalue().split(" ",1)[1])["l2"]["bytes"],48<<20)
            props.L2_cache_size = 32<<20
            with contextlib.redirect_stdout(io.StringIO()) as log:
                run.check_card(Path("/sdk"),"zw810e",0)
            self.assertEqual(probe.call_args.args[0].l2_bytes,32<<20)
            self.assertEqual(json.loads(log.getvalue().split(" ",1)[1])["l2"]["source"],"TORCH_DEVICE_PROPERTY")

    def test_sf_audit_only_for_mnk13_without_native(self):
        with tempfile.TemporaryDirectory() as t:
            args = SimpleNamespace(output=Path(t),no_native=True)
            run.save(args.output/"plan.json",self.plan)
            self.assertTrue(mnk13_f16_sf_only(args))
            self.assertEqual({p["workload"]["compute"] for p in self.plan["points"]},{"f16"})
            self.assertEqual({c["arm"] for p in self.plan["points"] for c in p["cases"]},{"sf","fq","full-fp16"})
            args.no_native=False
            self.assertFalse(mnk13_f16_sf_only(args))
            args.no_native=True
            run.save(args.output/"plan.json",{"schema":"quactlize.dense-m-span.v1"})
            self.assertFalse(mnk13_f16_sf_only(args))

    def test_unused_bf16_is_not_an_fp16_sf_gate(self):
        # Simulated hgobjdump output for the actual family counts. FP16x2 is
        # deliberately not evidence for a BF16 conversion in a strict audit.
        symbols = [f"{family}_{i}" for family,count in [
            ("prepass_unit_kernel",5),("sf_columns",10),("sf_unit16",4),("full_direct",5),
            ("full_wide",15),("full_transpose",10),("full_shared",8),("full_packed_exchange",6)]
            for i in range(count)]
        symbols += [f"sf_bfloat_units_FmtE{i}E_ELi{threads}E" for i in range(5) for threads in (128,256)]
        def check(sf_only, *, vector="vmem.ld.b32x4", omit=None):
            isa = "".join(f"Disassembly of section .text.kernel.{s}:\n\t{vector} v0\n\tv.pcnvt.f16x2 v1\n"
                          for s in symbols if s != omit)
            resources = "".join(f"Func {i} {s} RESOURCE INFO:\nvreg_number: 16\nsreg_number: 1\n"
                                "shared_memory_size: 0\nSTACK SIZE: 0\n" for i,s in enumerate(symbols))
            with patch("subprocess.run",side_effect=[SimpleNamespace(stdout=isa),SimpleNamespace(stdout=resources)]):
                return inspect_image(Path("/sdk"),Path("/image.so"),sf_f16_only=sf_only)
        with self.assertRaisesRegex(ValueError,"native BF16 output conversion missing: full_direct_0"):
            check(False)
        native,_ = check(True)
        self.assertEqual(len(native),73)
        self.assertEqual(sum(r["isa_checked"] for r in native.values()),19)
        self.assertEqual(native["full_direct_0"]["operations"]["v.pcnvt.f16x2"],1)
        self.assertFalse(native["full_direct_0"]["isa_checked"])
        with self.assertRaisesRegex(ValueError,"scalarized"):
            check(True,vector="vmem.ld.b8")
        with self.assertRaisesRegex(ValueError,"inventory differs"):
            check(True,omit="full_direct_0")

    def test_sf_receipt_never_validates_for_bf16(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            lib=root/"libquactlize_ppu_dequant.so"
            lib.write_bytes(b"test fixture")
            run.save(root/"native.json",{})
            manifest=dict(schema="quactlize.dequant-only.v1",timing="NO_GEMM",library=lib.name,
                          sha256=sha(lib),native_sha256=sha(root/"native.json"),runtime={})
            for scope in (None,"FULL","SF_FP16_ONLY","UNKNOWN"):
                run.save(root/"manifest.json",manifest if scope is None else dict(manifest,isa_scope=scope))
                for sf_only in (True,False):
                    if scope in (None,"FULL") or (sf_only and scope=="SF_FP16_ONLY"):
                        self.assertEqual(verify_dequant(root,root,sf_f16_only=sf_only)["sha256"],sha(lib))
                    else:
                        with self.assertRaisesRegex(ValueError,"ISA audit scope differs"):
                            verify_dequant(root,root,sf_f16_only=sf_only)
            run.save(root/"manifest.json",dict(manifest,isa_scope="SF_FP16_ONLY"))
            lib.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"payload differs"):
                verify_dequant(root,root,sf_f16_only=True)


if __name__ == "__main__":
    unittest.main()
