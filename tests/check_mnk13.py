"""Host regression tests for an extracted kit. Usage: python3 -B check_kit.py DIR"""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

kit = Path(sys.argv.pop(1)).resolve(strict=True)
sys.path.insert(0, str(kit/"source"))
spec = importlib.util.spec_from_file_location("mnk13_test_runner", kit/"run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
from dev.prefill_asys.finalists import load
from dev.strategy.resources import inventory
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.tuning import digest
from tools.run_prefill_model_asys import confirmation_jobs


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
        simulated = dict(incomplete=[], points={p["id"]:dict(measurements=ms)})
        with tempfile.TemporaryDirectory() as t, patch("tools.review_prefill_cohort.ingest",return_value=simulated):
            root = Path(t)
            run.review_table(root,"anchors",[p])
            with (root/"anchors.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertEqual(float(rows[0]["mean_us"]),statistics.mean([2,7,6,9,3,3]))
            ms.pop()
            with self.assertRaisesRegex(ValueError,"missing"):
                run.review_table(root,"anchors",[p])

    def test_bridge_drift(self):
        rows = [dict(N=n, route=r, historical_control_us=old, forward_mean_us=old, reverse_mean_us=old, mean_us=old)
                for n, t in ((1024,(31.,46.)), (6144,(119.,108.))) for r,old in zip(("sf","full-fp16"),t)]
        self.assertTrue(all(b["comparable"] for b in run.bridge(rows)))
        rows[0]["reverse_mean_us"] = 50
        self.assertFalse(run.bridge(rows)[0]["comparable"])


if __name__ == "__main__":
    unittest.main()
