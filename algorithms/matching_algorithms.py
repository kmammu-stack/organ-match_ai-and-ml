import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matching.compatibility_scorer import CompatibilityScorer

scorer = CompatibilityScorer()

def gale_shapley(donors: list, recipients: list) -> list:
    print("\n[Gale-Shapley] Building preference lists...")
    donor_prefs = {}
    for d in donors:
        scores = []
        for r in recipients:
            result = scorer.score_pair(d, r)
            if result:
                scores.append((r['recipient_id'], result['ahp_score']))
        donor_prefs[d['donor_id']] = [
            rid for rid, _ in sorted(scores, key=lambda x: x[1], reverse=True)
        ]

    recipient_prefs = {}
    for r in recipients:
        scores = []
        for d in donors:
            result = scorer.score_pair(d, r)
            if result:
                scores.append((d['donor_id'], result['ahp_score']))
        recipient_prefs[r['recipient_id']] = [
            did for did, _ in sorted(scores, key=lambda x: x[1], reverse=True)
        ]

    free_donors = [d['donor_id'] for d in donors]
    donor_next_proposal = {d['donor_id']: 0 for d in donors}
    recipient_current = {}
    donor_partner = {}

    while free_donors:
        donor_id = free_donors.pop(0)
        prefs = donor_prefs.get(donor_id, [])
        if donor_next_proposal[donor_id] >= len(prefs):
            continue
        recipient_id = prefs[donor_next_proposal[donor_id]]
        donor_next_proposal[donor_id] += 1

        if recipient_id not in recipient_current:
            recipient_current[recipient_id] = donor_id
            donor_partner[donor_id] = recipient_id
        else:
            current_donor = recipient_current[recipient_id]
            r_prefs = recipient_prefs.get(recipient_id, [])
            if donor_id in r_prefs and current_donor in r_prefs:
                if r_prefs.index(donor_id) < r_prefs.index(current_donor):
                    recipient_current[recipient_id] = donor_id
                    donor_partner[donor_id] = recipient_id
                    del donor_partner[current_donor]
                    free_donors.append(current_donor)
                else:
                    free_donors.append(donor_id)
            else:
                free_donors.append(donor_id)

    results = []
    for donor_id, recipient_id in donor_partner.items():
        d = next(x for x in donors if x['donor_id'] == donor_id)
        r = next(x for x in recipients if x['recipient_id'] == recipient_id)
        score = scorer.score_pair(d, r)
        if score:
            results.append({
                'algorithm': 'Gale-Shapley',
                'donor_id': donor_id,
                'recipient_id': recipient_id,
                'ahp_score': score['ahp_score'],
            })
    return results


def hungarian_matching(donors: list, recipients: list) -> list:
    print("\n[Hungarian] Building cost matrix...")
    n_d = len(donors)
    n_r = len(recipients)
    cost_matrix = np.ones((n_d, n_r))
    for i, d in enumerate(donors):
        for j, r in enumerate(recipients):
            result = scorer.score_pair(d, r)
            if result:
                cost_matrix[i][j] = 1 - result['ahp_score']

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    results = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i][j] < 1.0:
            results.append({
                'algorithm': 'Hungarian',
                'donor_id': donors[i]['donor_id'],
                'recipient_id': recipients[j]['recipient_id'],
                'ahp_score': round(1 - cost_matrix[i][j], 4),
            })
    return results


def bipartite_matching(donors: list, recipients: list) -> list:
    print("\n[Bipartite] Building graph...")
    G = nx.Graph()
    for d in donors:
        G.add_node(f"D_{d['donor_id']}", bipartite=0)
    for r in recipients:
        G.add_node(f"R_{r['recipient_id']}", bipartite=1)
    for d in donors:
        for r in recipients:
            result = scorer.score_pair(d, r)
            if result:
                G.add_edge(
                    f"D_{d['donor_id']}",
                    f"R_{r['recipient_id']}",
                    weight=result['ahp_score']
                )
    matching = nx.max_weight_matching(G, maxcardinality=True)
    results = []
    for u, v in matching:
        if u.startswith('R_'):
            u, v = v, u
        donor_id = u[2:]
        recipient_id = v[2:]
        d = next(x for x in donors if x['donor_id'] == donor_id)
        r = next(x for x in recipients if x['recipient_id'] == recipient_id)
        score = scorer.score_pair(d, r)
        if score:
            results.append({
                'algorithm': 'Bipartite',
                'donor_id': donor_id,
                'recipient_id': recipient_id,
                'ahp_score': score['ahp_score'],
            })
    return results


def kidney_exchange(donors: list, recipients: list) -> list:
    print("\n[KEP] Finding incompatible pairs for exchange...")
    incompatible_pairs = []
    for d, r in zip(donors, recipients):
        result = scorer.score_pair(d, r)
        if result is None:
            incompatible_pairs.append((d, r))

    if len(incompatible_pairs) < 2:
        print("[KEP] Not enough incompatible pairs for exchange")
        return []

    G = nx.DiGraph()
    for i in range(len(incompatible_pairs)):
        G.add_node(i)

    for i, (d_i, r_i) in enumerate(incompatible_pairs):
        for j, (d_j, r_j) in enumerate(incompatible_pairs):
            if i != j:
                result = scorer.score_pair(d_i, r_j)
                if result:
                    G.add_edge(i, j, weight=result['ahp_score'])

    exchanges = []
    used = set()

    for u, v in list(G.edges()):
        if u not in used and v not in used:
            if G.has_edge(v, u):
                d_u, r_u = incompatible_pairs[u]
                d_v, r_v = incompatible_pairs[v]
                s1 = scorer.score_pair(d_u, r_v)
                s2 = scorer.score_pair(d_v, r_u)
                if s1 and s2:
                    exchanges.append({
                        'algorithm': 'KEP',
                        'cycle': [u, v],
                        'cycle_size': 2,
                        'pairs': [
                            {'donor_id': d_u['donor_id'],
                             'recipient_id': r_v['recipient_id'],
                             'ahp_score': s1['ahp_score']},
                            {'donor_id': d_v['donor_id'],
                             'recipient_id': r_u['recipient_id'],
                             'ahp_score': s2['ahp_score']},
                        ],
                        'avg_score': round((s1['ahp_score'] + s2['ahp_score']) / 2, 4)
                    })
                    used.add(u)
                    used.add(v)
    return exchanges


def compare_algorithms(donors: list, recipients: list):
    print("\n" + "="*50)
    print("ALGORITHM COMPARISON")
    print("="*50)

    gs  = gale_shapley(donors, recipients)
    hun = hungarian_matching(donors, recipients)
    bip = bipartite_matching(donors, recipients)
    kep = kidney_exchange(donors, recipients)

    for name, results in [('Gale-Shapley', gs),
                          ('Hungarian', hun),
                          ('Bipartite', bip)]:
        if results:
            scores = [r['ahp_score'] for r in results]
            print(f"\n{name}:")
            print(f"  Matches found:    {len(results)}")
            print(f"  Avg AHP score:    {round(np.mean(scores), 4)}")
            print(f"  Max AHP score:    {round(np.max(scores), 4)}")
            print(f"  Utilization rate: {round(len(results)/len(donors)*100, 1)}%")

    print(f"\nKEP (Kidney Exchange):")
    print(f"  Exchanges found:  {len(kep)}")
    print(f"  Pairs helped:     {sum(len(e['pairs']) for e in kep)}")

    return {'gale_shapley': gs, 'hungarian': hun,
            'bipartite': bip, 'kep': kep}


if __name__ == '__main__':
    donors     = pd.read_csv('data/donors.csv')
    recipients = pd.read_csv('data/recipients.csv')
    donors.columns     = donors.columns.str.strip()
    recipients.columns = recipients.columns.str.strip()
    donors['blood_type']     = donors['blood_type'].str.strip()
    recipients['blood_type'] = recipients['blood_type'].str.strip()

    d_sample = donors.head(20).to_dict('records')
    r_sample = recipients.head(50).to_dict('records')

    print(f"Testing with {len(d_sample)} donors and {len(r_sample)} recipients")
    results = compare_algorithms(d_sample, r_sample)