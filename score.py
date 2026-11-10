"""Recompute Table 5 summary-SFT metrics. Python 3 standard library only."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def metrics(rows, k):
    tp = [0]*k; true = [0]*k; predicted = [0]*k
    for r in rows:
        y, p = r['true'], r['pred']
        if y not in range(k):
            raise ValueError('True label outside the fixed task label set')
        true[y] += 1
        if p in range(k):
            predicted[p] += 1
            if p == y: tp[y] += 1
    n = len(rows)
    accuracy = sum(tp)/n
    f1 = sum(2*tp[i]/(true[i]+predicted[i]) if true[i]+predicted[i] else 0 for i in range(k))/k
    chance = sum(true[i]*predicted[i] for i in range(k))/n**2
    kappa = (accuracy-chance)/(1-chance) if chance != 1 else float('nan')
    return {'accuracy':accuracy, 'macro_f1':f1, 'kappa':kappa, 'invalid':n-sum(predicted)}

def main():
    manifest = json.loads((ROOT/'datasets.json').read_text())
    out = []
    for name, cfg in manifest.items():
        rows = [json.loads(s) for s in (ROOT/cfg['file']).read_text().splitlines()]
        assert len(rows) == cfg['n'] and len({r['id'] for r in rows}) == len(rows)
        m = metrics(rows, cfg['classes'])
        assert round(m['macro_f1'],3) == cfg['expected_macro_f1'], (name,m)
        out.append({'dataset':name,'n':len(rows),**m})
        print(f"{name:15} N={len(rows):4} Acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f} kappa={m['kappa']:.3f}")
    with (ROOT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)

if __name__ == '__main__': main()
