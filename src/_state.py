"""Carries variables from one numbered script to the next (so scripts can be run one at a time)."""
import os, ast, glob, pickle, types, warnings
import matplotlib as mpl
STATE_DIR = "outputs/.state"

def _defs(path):
    """Source of top-level functions/classes/lambda assignments in an earlier script (re-created, not pickled)."""
    src = open(path, encoding="utf-8").read(); out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or (isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda)):
            out.append(ast.get_source_segment(src, n))
    return out

def _run_defs(ns, files):
    for f in files:
        for seg in _defs(f):
            try: exec(compile(seg, f, "exec"), ns)
            except NameError: pass       # default argument needs data that is not loaded yet

def restore(ns, this_file):
    here = os.path.dirname(os.path.abspath(this_file)); me = os.path.basename(this_file)
    mine = int(me[:2])
    if mine == 1: return
    prior = sorted(f for f in glob.glob(os.path.join(here, "0[1-6]_*.py")) if int(os.path.basename(f)[:2]) < mine)
    _run_defs(ns, prior)                 # functions first, so pickled references to them resolve
    pk = os.path.join(STATE_DIR, f"after_{mine-1:02d}.pkl")
    if not os.path.exists(pk): raise SystemExit(f"Missing {pk}: run the earlier scripts first (in numeric order).")
    saved = pickle.load(open(pk, "rb"))
    mpl.rcParams.update({k: v for k, v in saved.pop("__rc__").items() if k in mpl.rcParams and k != "backend"})
    ns.update(saved)
    _run_defs(ns, prior)                 # again, now that the data they use as default arguments exists

def save(ns, this_file):
    os.makedirs(STATE_DIR, exist_ok=True); keep = {}
    for k, v in ns.items():
        if k.startswith("__") or isinstance(v, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)): continue
        try: pickle.dumps(v); keep[k] = v
        except Exception: warnings.warn(f"not carried forward: {k}")
    keep["__rc__"] = {k: v for k, v in mpl.rcParams.items() if k != "backend"}
    pickle.dump(keep, open(os.path.join(STATE_DIR, f"after_{int(os.path.basename(this_file)[:2]):02d}.pkl"), "wb"))
