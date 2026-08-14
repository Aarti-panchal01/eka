"""Exercise the real checkpoint guard against the exact case that failed.

The module can't be imported (it authenticates to HF and grabs a GPU at import
time), so the two functions are lifted out of the source by AST and executed
with the constants injected. That still tests the shipped code, not a copy.
"""
import ast
import glob
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path

SRC = Path(r"C:\Users\Aarti Panchal\eka\training\train_all_loras_colab.py")
tree = ast.parse(SRC.read_text(encoding="utf-8"))

wanted = {"_checkpoint_mismatch", "latest_checkpoint"}
ns = {
    "os": os, "json": json, "glob": glob, "traceback": traceback,
    "LORA_R": 8, "LORA_ALPHA": 16,
    "TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "BASE_MODEL": "Qwen/Qwen2.5-7B-Instruct",
}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
missing = wanted - set(ns)
assert not missing, f"could not lift {missing} from source"
latest_checkpoint = ns["latest_checkpoint"]


def make(root, step, *, r=8, alpha=16, max_steps=None,
         modules=None, base="Qwen/Qwen2.5-7B-Instruct"):
    d = os.path.join(root, f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    cfg = {"r": r, "lora_alpha": alpha, "base_model_name_or_path": base,
           "target_modules": modules or ns["TARGET_MODULES"]}
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w"))
    if max_steps is not None:
        json.dump({"max_steps": max_steps},
                  open(os.path.join(d, "trainer_state.json"), "w"))
    return d


def case(name, build, expect_basename, expected_steps=114):
    root = tempfile.mkdtemp()
    try:
        build(root)
        got = latest_checkpoint(root, expected_steps)
        got_name = os.path.basename(got) if got else None
        ok = got_name == expect_basename
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: got {got_name!r}, "
              f"want {expect_basename!r}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


print("checkpoint guard\n")
results = [
    # The exact founder failure: r=16 smoke checkpoint left in Drive.
    case("r=16 smoke checkpoint is rejected",
         lambda r: make(r, 12, r=16, alpha=32, max_steps=12), None),

    # And it must not come back once a fresh run is at a lower step number.
    case("rejected checkpoint cannot outrank a newer valid one",
         lambda r: (make(r, 12, r=16, alpha=32, max_steps=12),
                    make(r, 50, max_steps=114)), "checkpoint-50"),

    case("matching checkpoint resumes",
         lambda r: make(r, 50, max_steps=114), "checkpoint-50"),

    case("newest matching checkpoint wins",
         lambda r: (make(r, 25, max_steps=114), make(r, 75, max_steps=114)),
         "checkpoint-75"),

    # Right rank, wrong schedule — the silent one.
    case("12-step schedule rejected for a 114-step run",
         lambda r: make(r, 12, max_steps=12), None),

    case("different target_modules rejected",
         lambda r: make(r, 30, modules=["q_proj", "v_proj"], max_steps=114), None),

    case("different base model rejected",
         lambda r: make(r, 30, base="meta-llama/Meta-Llama-3-8B-Instruct",
                        max_steps=114), None),

    case("checkpoint with no adapter_config rejected",
         lambda r: os.makedirs(os.path.join(r, "checkpoint-9")), None),

    case("empty directory starts fresh", lambda r: None, None),
]

# Renaming has to actually happen, or the next run re-loads 15GB to find out.
root = tempfile.mkdtemp()
try:
    make(root, 12, r=16, alpha=32, max_steps=12)
    latest_checkpoint(root, 114)
    parked = os.path.isdir(os.path.join(root, "incompatible-checkpoint-12"))
    gone = not os.path.isdir(os.path.join(root, "checkpoint-12"))
    print(f"  {'OK  ' if parked and gone else 'FAIL'} bad checkpoint is renamed "
          f"aside (parked={parked}, original_gone={gone})")
    results.append(parked and gone)
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n  {sum(results)}/{len(results)} "
      f"{'ALL PASS' if all(results) else 'FAILURES ABOVE'}")
raise SystemExit(0 if all(results) else 1)
