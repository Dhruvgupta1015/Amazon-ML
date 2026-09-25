import sys
import json
import csv
from collections import defaultdict
from evaluate_optimized_matcher import clean_name, clean_address, evaluate_pair, entity_f05, STOPWORDS, FROZEN_SPLIT_PATH, DATASET_ROOT

sys.stdout.reconfigure(encoding='utf-8')

with open(FROZEN_SPLIT_PATH, 'r', encoding='utf-8') as f:
    val_ids = json.load(f)['s1_entity_ids'][:1000]
val_set = set(val_ids)

gt = {}
with open(DATASET_ROOT / 'train' / 'train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_set:
            gt[p[0]] = set(x.strip() for x in p[1].split(',') if x.strip()) if len(p) >= 2 and p[1].strip() else set()

needed_pos = set(x for s in gt.values() for x in s)

s1_data = {}
with open(DATASET_ROOT / 'train' / 'train_source1.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if len(p) >= 4 and p[0] in val_set:
            cn, dom, indic = clean_name(p[1])
            ca, atoks, snum, post, st = clean_address(p[2], p[3])
            s1_data[p[0]] = {
                'id': p[0], 'raw_name': p[1], 'raw_addr': p[2], 'clean_name': cn, 'domain_stem': dom,
                'has_indic': indic, 'name_tokens': set(cn.split()) - STOPWORDS,
                'clean_addr': ca, 'addr_tokens': atoks, 'street_num': snum, 'postal': post, 'state': st, 'country': p[3]
            }

s23_data = {}
name_idx = defaultdict(list)
addr_idx = defaultdict(list)
snum_idx = defaultdict(list)
domain_idx = defaultdict(list)

for s_path in [DATASET_ROOT / 'train' / 'train_source2.tsv', DATASET_ROOT / 'train' / 'train_source3.tsv']:
    with open(s_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4:
                cid = p[0]
                if cid in needed_pos or len(s23_data) < 100000:
                    cn, dom, indic = clean_name(p[1])
                    ca, atoks, snum, post, st = clean_address(p[2], p[3])
                    rec = {
                        'id': cid, 'raw_name': p[1], 'raw_addr': p[2], 'clean_name': cn, 'domain_stem': dom,
                        'has_indic': indic, 'name_tokens': set(cn.split()) - STOPWORDS,
                        'clean_addr': ca, 'addr_tokens': atoks, 'street_num': snum, 'postal': post, 'state': st, 'country': p[3]
                    }
                    s23_data[cid] = rec
                    for t in rec['name_tokens']:
                        if len(t) >= 3: name_idx[(p[3], t)].append(cid)
                    for at in atoks:
                        if len(at) >= 4: addr_idx[(p[3], at)].append(cid)
                    if snum: snum_idx[(p[3], snum)].append(cid)
                    if dom: domain_idx[(p[3], dom)].append(cid)

fn_count = 0
fp_count = 0
print('=== EXAMINING FALSE NEGATIVES (MISSED TRUE MATCHES) ===')
for s1_id, s1 in list(s1_data.items())[:200]:
    true_m = gt[s1_id]
    cands = set()
    c = s1['country']
    for t in s1['name_tokens']:
        if len(t) >= 3:
            p = name_idx.get((c, t), [])
            if len(p) <= 200: cands.update(p)
    for at in s1['addr_tokens']:
        if len(at) >= 4:
            p = addr_idx.get((c, at), [])
            if len(p) <= 100: cands.update(p)
    if s1['street_num']:
        p = snum_idx.get((c, s1['street_num']), [])
        if len(p) <= 120: cands.update(p)
    if s1['domain_stem']:
        p = domain_idx.get((c, s1['domain_stem']), [])
        cands.update(p)

    pred_m = set()
    for cid in cands:
        cand = s23_data[cid]
        if evaluate_pair(s1, cand):
            pred_m.add(cid)

    missed = true_m - pred_m
    for m in missed:
        if m in s23_data:
            fn_count += 1
            cand = s23_data[m]
            retrieved = m in cands
            print(f'\n[FALSE NEGATIVE #{fn_count}] S1: {s1_id} ({s1["country"]}) | Retrieved in blocking: {retrieved}')
            print(f'   S1 Name: {s1["raw_name"]} | Clean: {s1["clean_name"]}')
            print(f'   S1 Addr: {s1["raw_addr"]} | SNum: {s1["street_num"]} | State: {s1["state"]}')
            print(f'   True Match {m}:')
            print(f'   M Name:  {cand["raw_name"]} | Clean: {cand["clean_name"]}')
            print(f'   M Addr:  {cand["raw_addr"]} | SNum: {cand["street_num"]} | State: {cand["state"]}')
            if fn_count >= 8: break
    if fn_count >= 8: break

print('\n=== EXAMINING FALSE POSITIVES (FALSE MERGES) ===')
for s1_id, s1 in list(s1_data.items())[:200]:
    true_m = gt[s1_id]
    cands = set()
    c = s1['country']
    for t in s1['name_tokens']:
        if len(t) >= 3:
            p = name_idx.get((c, t), [])
            if len(p) <= 200: cands.update(p)
    for at in s1['addr_tokens']:
        if len(at) >= 4:
            p = addr_idx.get((c, at), [])
            if len(p) <= 100: cands.update(p)
    if s1['street_num']:
        p = snum_idx.get((c, s1['street_num']), [])
        if len(p) <= 120: cands.update(p)
    if s1['domain_stem']:
        p = domain_idx.get((c, s1['domain_stem']), [])
        cands.update(p)

    pred_m = set()
    for cid in cands:
        cand = s23_data[cid]
        if evaluate_pair(s1, cand):
            pred_m.add(cid)

    fps = pred_m - true_m
    for fp in fps:
        fp_count += 1
        cand = s23_data[fp]
        print(f'\n[FALSE POSITIVE #{fp_count}] S1: {s1_id} ({s1["country"]})')
        print(f'   S1 Name: {s1["raw_name"]} | Clean: {s1["clean_name"]}')
        print(f'   S1 Addr: {s1["raw_addr"]} | SNum: {s1["street_num"]} | State: {s1["state"]}')
        print(f'   Pred Candidate {fp}:')
        print(f'   P Name:  {cand["raw_name"]} | Clean: {cand["clean_name"]}')
        print(f'   P Addr:  {cand["raw_addr"]} | SNum: {cand["street_num"]} | State: {cand["state"]}')
        if fp_count >= 8: break
    if fp_count >= 8: break
