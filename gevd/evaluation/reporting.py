"""Full-episode learning curves; no thinning, trailing windows with full windows only."""
import json
import os
from pathlib import Path
import numpy as np


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # Only tolerate a concurrently appended incomplete final line.
    return result


def rolling(values, width):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) >= width:
        cs = np.r_[0., np.cumsum(values)]
        result[width-1:] = (cs[width:] - cs[:-width]) / width
    return result


def make_report(out, exams=None):
    out = Path(out)
    os.environ.setdefault('MPLCONFIGDIR', str(out.parent/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train = [r for r in rows(out/'episodes.jsonl') if r.get('terminal', True)]
    evaluation = rows(out/'exams.jsonl') if exams is None else exams
    evaluation = [r for r in evaluation if r['episode'] > 0]
    if not train or not evaluation:
        return
    report = out/'report'
    report.mkdir(exist_ok=True)
    for width in (49, 99):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax, records, color, label in ((axes[0], train, '#2879bd', 'Training G'), (axes[1], evaluation, '#e27c27', 'Current greedy policy G')):
            x = [r['C'] for r in records]
            values = [r['return'] for r in records]
            ax.plot(x, values, color=color, alpha=.14, lw=.5)
            ax.plot(x, rolling(values, width), color=color, lw=1.3, label=f'{label}: MA{width}')
            ax.set_ylabel(label + ' (unscaled)')
        axes[1].set_xlabel('Cumulative search simulator calls (evaluation calls excluded)')
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=8, loc='best')
        display_method = 'GEVD'
        if (out/'effective_config.yaml').exists():
            import yaml
            config = yaml.safe_load((out/'effective_config.yaml').read_text(encoding='utf-8'))
            display_method = config.get('experiment', {}).get('display_method', display_method)
        fig.suptitle(f'{display_method} | {out.name} | trailing window {width}\nFaint lines: complete raw episodes; smoothing starts after {width} samples; failures included')
        fig.tight_layout()
        for extension in ('png', 'svg'):
            path = report/f'learning_curve_w{width}.{extension}'
            temp = path.with_name(path.stem+'.tmp'+path.suffix)
            fig.savefig(temp, dpi=140); os.replace(temp, path)
        plt.close(fig)


def summarize(root, budget):
    train = rows(root/'run/episodes.jsonl')
    exams = [r for r in rows(root/'run/exams.jsonl') if r['episode'] > 0]
    if not exams:
        return dict(seed=int(root.name.split('_')[1]), status='pending')
    late = [r for r in exams if r['C'] >= .8*budget]
    late_train = [r for r in train if r['terminal'] and r['C'] >= .8*budget]
    diag = rows(root/'run/optimizer_diagnostics.jsonl')
    return dict(seed=int(root.name.split('_')[1]), episodes=len(train), evaluations=len(exams),
                C=exams[-1]['C'], final=exams[-1],
                late_eval_success=sum(r['success'] for r in late), late_eval_n=len(late),
                late_train_success=sum(r['success'] for r in late_train), late_train_n=len(late_train),
                loss_last1000=float(np.mean([r['combined_loss'] for r in diag[-1000:]])) if diag else None)
