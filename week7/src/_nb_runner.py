"""Minimal notebook builder/executor (no nbformat/jupyter available in this sandbox).

Parses a percent-format script ("# %%" code cells, "# %% [markdown]" markdown cells),
executes every code cell in one shared namespace, captures stdout/stderr, rich DataFrame
output of the last expression and matplotlib figures, and writes a valid nbformat-4 .ipynb.

usage: python build_notebook.py script.py out.ipynb [--no-exec]
"""
import ast, base64, contextlib, io, json, os, sys, time, traceback, uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path):
    cells, cur, kind = [], [], None
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(cur).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(cur).strip("\n")))
    out = []
    for k, src in cells:
        if k == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.splitlines())
        out.append((k, src))
    return out


def df_html(obj):
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            with pd.option_context("display.max_rows", 60, "display.max_columns", 40):
                return obj.to_html(max_rows=60, max_cols=40, border=0, classes="dataframe"), repr(obj)
    except Exception:
        pass
    return None, None


def run(cells, execute=True):
    ns = {"__name__": "__main__"}
    nb_cells, count = [], 0
    for kind, src in cells:
        cid = uuid.uuid4().hex[:8]
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": src.splitlines(True)})
            continue
        outputs = []
        if execute and src.strip():
            count += 1
            t0 = time.time()
            buf, err = io.StringIO(), io.StringIO()
            last_val = None
            try:
                tree = ast.parse(src)
                body, tail = tree.body, None
                if body and isinstance(body[-1], ast.Expr):
                    tail = ast.Expression(body[-1].value)
                    body = body[:-1]
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                    exec(compile(ast.Module(body, type_ignores=[]), "<cell %d>" % count, "exec"), ns)
                    if tail is not None:
                        last_val = eval(compile(tail, "<cell %d>" % count, "eval"), ns)
            except Exception:
                tb = traceback.format_exc()
                print(f"[cell {count}] ERROR\n{tb}", file=sys.stderr)
                outputs.append({"output_type": "error", "ename": "Error", "evalue": tb.splitlines()[-1],
                                "traceback": tb.splitlines()})
                nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": count,
                                 "source": src.splitlines(True), "outputs": outputs})
                json.dump({"cells": nb_cells}, open("/tmp/partial_nb.json", "w"))
                raise
            if buf.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
            if err.getvalue().strip():
                outputs.append({"output_type": "stream", "name": "stderr", "text": err.getvalue().splitlines(True)})
            if last_val is not None:
                html, rep = df_html(last_val)
                data = {"text/plain": (rep or repr(last_val)).splitlines(True)}
                if html:
                    data["text/html"] = html.splitlines(True)
                outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
            for num in plt.get_fignums():
                fig = plt.figure(num)
                b = io.BytesIO()
                fig.savefig(b, format="png", dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                         "text/plain": ["<Figure>"]}})
            plt.close("all")
            print(f"cell {count:3d} ok  {time.time()-t0:5.1f}s", flush=True)
        nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": count if execute else None,
                         "source": src.splitlines(True), "outputs": outputs})
    return nb_cells, ns


if __name__ == "__main__":
    script, out = sys.argv[1], sys.argv[2]
    execute = "--no-exec" not in sys.argv
    os.chdir(os.path.dirname(os.path.abspath(script)) + "/..")
    cells = parse(script)
    nb_cells, ns = run(cells, execute)
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    def _j(o):
        import numpy as _np, pandas as _pd
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, (_pd.Timestamp,)): return str(o)
        return str(o)
    json.dump({"R": ns.get("R", {}), "FIGS": ns.get("FIGS", {}), "DQ": ns.get("DQ", [])}, open("outputs/results.json", "w"), indent=1, default=_j)
    print("wrote", out, len(nb_cells), "cells")
