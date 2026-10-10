#!/usr/bin/env python3
"""Layer-scoped A/B op-level comparison for single-step NPU profiles.

Pipeline: config -> representative-layer window extraction -> cross-run
per-instance pairing -> agg_result.json. Before writing anything it HARD-VERIFIES
every extracted row against the ORIGINAL kernel_details.csv files (Start Time +
Duration must exist row-for-row); exits 1 on any miss. Ordering assertions and
cross-sum checks are also enforced here.

Usage: python3 agg_and_verify.py <config.py>       (see configs/ for examples)

The config is a plain python file defining: RUNS, MODULE_OF, DRAFT_MODS,
MODULE_ORDER, SCOPES, SHARD, OUTPUT, and optionally IDEA_RULES,
SEQUENCE_ASSERTS. Relative paths resolve against the workspace root.
"""
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter, OrderedDict

ROOT = os.path.expanduser('~/workspace')


def P(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


MEAN_FIELDS = ['wait_time_us', 'aicore_time_us', 'aic_total_cycles',
               'aic_mac_time_us', 'aic_mac_ratio',
               'aic_scalar_time_us', 'aic_scalar_ratio',
               'aic_mte1_time_us', 'aic_mte1_ratio',
               'aic_mte2_time_us', 'aic_mte2_ratio',
               'aic_fixpipe_time_us', 'aic_fixpipe_ratio',
               'aic_icache_miss_rate',
               'aiv_time_us', 'aiv_total_cycles',
               'aiv_vec_time_us', 'aiv_vec_ratio',
               'aiv_scalar_time_us', 'aiv_scalar_ratio',
               'aiv_mte2_time_us', 'aiv_mte2_ratio',
               'aiv_mte3_time_us', 'aiv_mte3_ratio',
               'aiv_icache_miss_rate', 'cube_utilization_pct']


def build_predicates(cluster_defs):
    out = []
    for entry in cluster_defs:
        preds = []
        for rule in entry.get('rules', []):
            if rule.get('catch_all'):
                preds.append(lambda _op: True)
                continue
            op_name = rule.get('op_name')
            op_name_re = re.compile(rule['op_name_regex']) if rule.get('op_name_regex') else None
            in_contains = rule.get('input_shapes_contains')

            def match(op, op_name=op_name, op_name_re=op_name_re, in_contains=in_contains):
                name = op.get('normalized_name', '')
                if op_name is not None and name != op_name:
                    return False
                if op_name_re is not None and not op_name_re.search(name):
                    return False
                ish = op.get('input_shapes', '') or ''
                if in_contains is not None and in_contains not in ish:
                    return False
                return True
            preds.append(match)
        out.append((entry['cluster'], preds))
    return out


def classify(op, compiled):
    for name, preds in compiled:
        if any(p(op) for p in preds):
            return name
    return None


def comm_kind(t):
    t = (t or '').lower()
    for k in ['allreduce', 'allgather', 'reducescatter', 'alltoall', 'send', 'receive', 'broadcast']:
        if k in t:
            return k
    return t


def idea_for(cfg, run, module, cluster, name):
    for (m, c, nsub), runf, text in getattr(cfg, 'IDEA_RULES', []):
        if m != module:
            continue
        if c is not None and c != cluster:
            continue
        if nsub is not None and nsub not in name:
            continue
        if runf is not None and runf != run:
            continue
        return text
    return None


def load_csv_keys(path):
    """(round(start,2), round(dur,2)) set over ALL rows of kernel_details.csv."""
    keys = set()
    with open(P(path), newline='') as f:
        rd = csv.reader(f)
        header = [h.strip() for h in next(rd)]
        ci_dur = header.index('Duration(us)')
        ci_st = header.index('Start Time(us)')
        for row in rd:
            if len(row) <= max(ci_dur, ci_st):
                continue
            try:
                keys.add((round(float(row[ci_st].strip()), 2), round(float(row[ci_dur].strip()), 2)))
            except ValueError:
                pass
    return keys


def extract_run(cfg, run_cfg, csv_keys):
    """Window extraction for one run. Returns (rows, subtotals, windows_info, meta)."""
    run = run_cfg['name']
    ro = json.load(open(P(run_cfg['raw_ops'])))
    det = json.load(open(P(run_cfg['details'])))
    sd = json.load(open(P(run_cfg['structure'])))
    spec = json.load(open(P(run_cfg['spec'])))

    dops = {d['index']: d for d in det['operators']}
    o2c = sd['op_to_component']
    ctype_of = {c['component_id']: c['type'] for c in sd['components']}
    comps_by_key = {(c['type'], c['layer_idx']): c for c in sd['components']}
    compiled = {ct: build_predicates(defs) for ct, defs in spec['component_clusters'].items()}
    ops_by_idx = {op['index']: op for op in ro['operators']}
    total = sum(op['duration_us'] for op in ro['operators'])

    def module_of_comp(c):
        return cfg.MODULE_OF[ctype_of[c]]

    # ---- time windows per scope ----
    scope_windows = []
    for module, key in cfg.SCOPES:
        if isinstance(key, tuple) and key[0] == 'gap':
            # boundary region: [end of last op of `from` modules, start of first
            # op of `to` modules (None = every other module))
            from_mods, to_mods = key[1], key[2]
            from_ends, to_starts = [], []
            for op in ro['operators']:
                c = o2c.get(str(op['index']))
                if c is None:
                    continue
                m = module_of_comp(c)
                if m in from_mods:
                    from_ends.append(op['start_time_us'] + op['duration_us'])
                elif to_mods is None or m in to_mods:
                    to_starts.append(op['start_time_us'])
            ws, we = max(from_ends), min(to_starts)
            layer_disp, incl_end = '—', False
        else:
            cops = [ops_by_idx[int(i)] for i in comps_by_key[key]['op_indices']]
            ws = min(o['start_time_us'] for o in cops)
            we = max(o['start_time_us'] + o['duration_us'] for o in cops)
            layer_disp = ('D%d' if module in cfg.DRAFT_MODS else 'L%d') % key[1]
            incl_end = True
        scope_windows.append((module, layer_disp, ws, we, incl_end))

    # ---- claim ops ----
    # pass 1: a matched op ALWAYS belongs to its own component's scope. Layer
    # windows can overlap by ~0.2us at stream seams (e.g. the next layer's first
    # op launches just before the previous layer's last op ends), so window
    # containment alone would steal matched ops across layers.
    claim = {}
    scope_of_key = {key: (module, ('D%d' if module in cfg.DRAFT_MODS else 'L%d') % key[1])
                    for module, key in cfg.SCOPES if not (isinstance(key, tuple) and key[0] == 'gap')}
    comp_key_of = {c['component_id']: (c['type'], c['layer_idx']) for c in sd['components']}
    for op in ro['operators']:
        c = o2c.get(str(op['index']))
        if c is not None:
            scope = scope_of_key.get(comp_key_of[c])
            if scope is not None:
                claim[op['index']] = scope
    # pass 2: unmatched ops (comm / glue) -> first window containing their start
    for op in ro['operators']:
        if op['index'] in claim or str(op['index']) in o2c:
            continue
        st = op['start_time_us']
        for module, layer_disp, ws, we, incl_end in scope_windows:
            if ws <= st and (st <= we if incl_end else st < we):
                claim[op['index']] = (module, layer_disp)
                break

    # ---- per-instance rows ----
    rows = []
    seen_idea = set()
    for module, layer_disp, ws, we, incl_end in scope_windows:
        scope_ops = [op for op in ro['operators'] if claim.get(op['index']) == (module, layer_disp)]
        scope_ops.sort(key=lambda o: (o['start_time_us'], o['index']))
        for op in scope_ops:
            idx = op['index']
            d = dops[idx]
            is_comm = (d['type'] or '').lower().startswith('hcom')
            if str(idx) in o2c:
                ct = ctype_of[o2c[str(idx)]]
                cluster = classify(op, compiled[ct])
                if cluster == 'dispatch_combine':
                    cluster = 'dispatch' if d['type'] == 'MoeDistributeDispatchV2' else 'combine'
                if cluster is None:
                    cluster = 'UNMATCHED_RULE'
            elif is_comm:
                cluster = comm_kind(d['type'])
            else:
                cluster = 'other'
            # pairing name: comm task IDs (hcom_xxx__503_1471_1) differ across
            # runs — pair by type only, never by the raw name
            norm = d['name'].split('__')[0] if is_comm else d['name']
            ikey = (module, cluster, d['name'])
            idea = None
            if ikey not in seen_idea:
                idea = idea_for(cfg, run, module, cluster, d['name'])
                seen_idea.add(ikey)
            row = OrderedDict()
            row['module'] = module
            row['cluster'] = cluster
            row['layer'] = layer_disp
            row['name'] = d['name']
            row['norm'] = norm
            row['type'] = d['type']
            row['shard'] = cfg.SHARD.get((run, module), '—')
            row['idea'] = idea
            row['start_us'] = op['start_time_us']
            row['duration_us'] = d['duration_us']
            row['share'] = d['duration_us'] / total
            for f in MEAN_FIELDS:
                row[f] = d.get(f)
            rows.append(row)

    subtotals = OrderedDict()
    for module, _, _, _, _ in scope_windows:
        ms = [r for r in rows if r['module'] == module]
        subtotals[module] = {'count': len(ms),
                             'total_us': sum(r['duration_us'] for r in ms),
                             'share': sum(r['share'] for r in ms)}
    windows_info = {m: {'w_start': ws, 'w_end': we} for m, _, ws, we, _ in scope_windows}
    meta = {'step_id': ro.get('step_id'), 'n_ops_full': len(ro['operators']),
            'total_us_full': total}
    return rows, subtotals, windows_info, meta, csv_keys


def build_comparison(cfg, result):
    names = [rc['name'] for rc in cfg.RUNS]
    a_name, b_name = names[0], names[1]
    comparison = []
    for module, _ in cfg.SCOPES:
        side = {}
        for run in names:
            rs = [r for r in result[run]['rows'] if r['module'] == module]
            occ = Counter()
            keyed = {}
            for r in rs:  # start-sorted
                occ[r['norm']] += 1
                keyed[(r['norm'], occ[r['norm']])] = r
            side[run] = keyed
        max_occ = Counter()
        for run in side:
            for norm, o in side[run]:
                max_occ[norm] = max(max_occ[norm], o)

        def korder(k):
            a = side[a_name].get(k)
            b = side[b_name].get(k)
            return (a['start_us'] if a else b['start_us'], k)

        for k in sorted(set(side[a_name]) | set(side[b_name]), key=korder):
            norm, occ = k
            a = side[a_name].get(k)
            b = side[b_name].get(k)
            disp = f'{norm}#{occ}' if (norm.startswith('hcom') or max_occ[norm] > 1) else norm
            comparison.append(OrderedDict(
                module=module, layer=side[a_name][k]['layer'] if k in side[a_name]
                else side[b_name][k]['layer'],
                name=disp,
                **{f'{a_name}_start': a['start_us'] if a else None,
                   f'{a_name}_dur': a['duration_us'] if a else None,
                   f'{b_name}_start': b['start_us'] if b else None,
                   f'{b_name}_dur': b['duration_us'] if b else None},
                delta=(a['duration_us'] - b['duration_us']) if (a and b) else None,
                delta_pct=((a['duration_us'] - b['duration_us']) / b['duration_us'])
                          if (a and b and b['duration_us'] > 0) else None))
    return comparison, a_name, b_name


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: agg_and_verify.py <config.py>')
    spec = importlib.util.spec_from_file_location('cfg', sys.argv[1])
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    failures = []

    result = OrderedDict()
    for run_cfg in cfg.RUNS:
        keys = load_csv_keys(run_cfg['kernel_details_csv'])
        rows, subtotals, windows_info, meta, _ = extract_run(cfg, run_cfg, keys)
        # GATE 1: every row exists row-for-row in kernel_details.csv
        miss = [r for r in rows
                if (round(r['start_us'], 2), round(r['duration_us'], 2)) not in keys]
        status = 'PASS' if not miss else 'FAIL'
        print(f"[{run_cfg['name']}] {len(rows)} rows; csv check {status}"
              + (f' — {len(miss)} MISSING, e.g. {[(m["name"], m["start_us"]) for m in miss[:3]]}'
                 if miss else ''))
        if miss:
            failures.append(f"{run_cfg['name']}: {len(miss)} rows not in kernel_details.csv")
        # GATE 2: subtotals consistent
        for m, st in subtotals.items():
            ms = [r for r in rows if r['module'] == m]
            if st['count'] != len(ms) or abs(st['total_us'] - sum(r['duration_us'] for r in ms)) > 0.01:
                failures.append(f"{run_cfg['name']}: subtotal mismatch at {m}")
        result[run_cfg['name']] = OrderedDict(meta, rows=rows, subtotals=subtotals,
                                              windows=windows_info)
        print(f"  full step: {meta['n_ops_full']} ops / {meta['total_us_full']:.1f}us"
              f" (share denominator)")
        for m, st in subtotals.items():
            print(f"  {m:<16} rows={st['count']:<4} Σ={st['total_us']:<9.1f} share={st['share']:.3f}")

    comparison, a_name, b_name = build_comparison(cfg, result)
    result['comparison'] = comparison
    result['module_order'] = cfg.MODULE_ORDER
    result['mean_fields'] = MEAN_FIELDS
    result['run_names'] = [a_name, b_name]
    result['scopes'] = [{'module': m, 'key': list(k) if isinstance(k, tuple) else k}
                        for m, k in cfg.SCOPES]

    # GATE 3: comparison sums == per-run shown totals
    for run in (a_name, b_name):
        csum = sum(c[f'{run}_dur'] for c in comparison if c[f'{run}_dur'] is not None)
        rsum = sum(s['total_us'] for s in result[run]['subtotals'].values())
        if abs(csum - rsum) > 0.01:
            failures.append(f'comparison Σ{run} {csum:.1f} != shown total {rsum:.1f}')
    # one-sided comm audit: every one-sided comm row must be explainable
    os_comm = [c for c in comparison if c['name'].startswith('hcom')
               and (c[f'{a_name}_dur'] is None or c[f'{b_name}_dur'] is None)]
    print(f"\ncomparison: {len(comparison)} rows; one-sided comm rows: {len(os_comm)}")
    for c in os_comm:
        print(f"  {c['module']:<14} {c['name']:<24} {a_name}={c[f'{a_name}_dur']} {b_name}={c[f'{b_name}_dur']}")

    # GATE 4: order-only sequence assertions (ground truth from kernel_details)
    for run, module, chain in getattr(cfg, 'SEQUENCE_ASSERTS', []):
        names_seq = [r['name'] for r in result[run]['rows'] if r['module'] == module]
        pos = -1
        ok = True
        for sub in chain:
            pos = next((i for i in range(pos + 1, len(names_seq)) if sub in names_seq[i]), -1)
            if pos < 0:
                ok = False
                break
        print(f"sequence assert [{run}/{module}]: {'PASS' if ok else 'FAIL'} ({' > '.join(chain)})")
        if not ok:
            failures.append(f'sequence assert failed: {run}/{module}')

    if failures:
        print('\nFAILURES:')
        for f_ in failures:
            print(' -', f_)
        sys.exit(1)

    json.dump(result, open(P(cfg.OUTPUT), 'w'), ensure_ascii=False, indent=1)
    print(f"\nALL GATES PASS -> {cfg.OUTPUT}")


if __name__ == '__main__':
    main()
