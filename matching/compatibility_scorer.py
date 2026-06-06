import numpy as np
import pandas as pd

# ABO compatibility matrix
# True = compatible (donor can give to recipient)
ABO_COMPATIBILITY = {
    'O':  {'O': True,  'A': True,  'B': True,  'AB': True},
    'A':  {'O': False, 'A': True,  'B': False, 'AB': True},
    'B':  {'O': False, 'A': False, 'B': True,  'AB': True},
    'AB': {'O': False, 'A': False, 'B': False, 'AB': True},
}

# AHP weight matrix
# Criteria: blood_match, hla_score, urgency, age_delta, distance
AHP_WEIGHTS = {
    'blood_match': 0.35,
    'hla_score':   0.30,
    'urgency':     0.20,
    'age_delta':   0.10,
    'distance':    0.05,
}

class CompatibilityScorer:

    def check_abo(self, donor_blood, recipient_blood):
        """Hard constraint — returns True if ABO compatible"""
        return ABO_COMPATIBILITY[donor_blood][recipient_blood]

    def hla_score(self, donor_hla, recipient_hla):
        """
        HLA mismatch score across 6 loci.
        0 mismatches = perfect = score 1.0
        12 mismatches = worst = score 0.0
        """
        mismatches = 0
        for locus in ['A', 'B', 'C', 'DR', 'DQ', 'DP']:
            d_alleles = set(donor_hla.get(locus, []))
            r_alleles = set(recipient_hla.get(locus, []))
            mismatches += len(d_alleles.symmetric_difference(r_alleles))
        max_mismatches = 12
        return max(0.0, 1 - (mismatches / max_mismatches))

    def urgency_score(self, urgency):
        """Normalize urgency 1-4 to 0-1"""
        return (urgency - 1) / 3

    def age_delta_score(self, donor_age, recipient_age):
        """
        Smaller age difference = better.
        0 years difference = score 1.0
        50+ years difference = score 0.0
        """
        delta = abs(donor_age - recipient_age)
        return max(0, 1 - (delta / 50))

    def distance_score(self, donor_lat, donor_lon, recipient_lat, recipient_lon):
        """
        Closer distance = better score.
        Uses simple Euclidean approximation.
        0 distance = 1.0, 100+ degrees = 0.0
        """
        dist = np.sqrt((donor_lat - recipient_lat)**2 +
                       (donor_lon - recipient_lon)**2)
        return max(0, 1 - (dist / 100))

    def ahp_score(self, scores: dict):
        """
        Weighted AHP score from individual criteria scores.
        scores = {blood_match, hla_score, urgency, age_delta, distance}
        """
        total = sum(AHP_WEIGHTS[k] * scores[k] for k in AHP_WEIGHTS)
        return round(total, 4)

    def score_pair(self, donor: dict, recipient: dict):
        """
        Score a single donor-recipient pair.
        Returns None if ABO incompatible (hard constraint).
        Returns dict with all scores + final AHP score.
        """
        # Hard constraint first
        if not self.check_abo(donor['blood_type'], recipient['blood_type']):
            return None

        # Parse HLA if stored as string
        donor_hla = donor['hla'] if isinstance(donor['hla'], dict) else eval(donor['hla'])
        recipient_hla = recipient['hla'] if isinstance(recipient['hla'], dict) else eval(recipient['hla'])

        scores = {
            'blood_match': 1.0,
            'hla_score':   self.hla_score(donor_hla, recipient_hla),
            'urgency':     self.urgency_score(recipient['urgency_score']),
            'age_delta':   self.age_delta_score(donor['age'], recipient['age']),
            'distance':    self.distance_score(
                               donor['location_lat'], donor['location_lon'],
                               recipient['location_lat'], recipient['location_lon']
                           ),
        }

        return {
            'donor_id':     donor['donor_id'],
            'recipient_id': recipient['recipient_id'],
            'abo_compatible': True,
            'hla_score':    round(scores['hla_score'], 4),
            'urgency':      round(scores['urgency'], 4),
            'age_delta':    round(scores['age_delta'], 4),
            'distance':     round(scores['distance'], 4),
            'ahp_score':    self.ahp_score(scores),
        }

    def rank_recipients(self, donor: dict, recipients: list):
        """
        Given one donor, rank all compatible recipients by AHP score.
        Returns sorted list (best match first).
        """
        results = []
        for r in recipients:
            score = self.score_pair(donor, r)
            if score is not None:
                results.append(score)
        return sorted(results, key=lambda x: x['ahp_score'], reverse=True)


class TOPSISRanker:

    def rank(self, scored_pairs: list):
        """
        Apply TOPSIS on top of AHP scores for final ranking.
        Input: list of score dicts from CompatibilityScorer
        Output: same list with topsis_score added, sorted best first
        """
        if not scored_pairs:
            return []

        criteria = ['hla_score', 'urgency', 'age_delta', 'distance']
        matrix = np.array([[p[c] for c in criteria] for p in scored_pairs],
                          dtype=float)

        # Step 1 - normalize
        norms = np.sqrt((matrix**2).sum(axis=0))
        norms[norms == 0] = 1
        norm_matrix = matrix / norms

        # Step 2 - weighted normalized matrix
        weights = np.array([
            AHP_WEIGHTS['hla_score'],
            AHP_WEIGHTS['urgency'],
            AHP_WEIGHTS['age_delta'],
            AHP_WEIGHTS['distance'],
        ])
        weighted = norm_matrix * weights

        # Step 3 - ideal best and worst
        ideal_best  = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)

        # Step 4 - Euclidean distances
        dist_best  = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

        # Step 5 - closeness coefficient
        closeness = dist_worst / (dist_best + dist_worst + 1e-10)

        # Add scores back
        for i, pair in enumerate(scored_pairs):
            pair['topsis_score'] = round(float(closeness[i]), 4)

        return sorted(scored_pairs, key=lambda x: x['topsis_score'], reverse=True)


if __name__ == '__main__':
    
    donors     = pd.read_csv('data/donors.csv')
    recipients = pd.read_csv('data/recipients.csv')

# Add these two lines
    donors.columns     = donors.columns.str.strip()
    recipients.columns = recipients.columns.str.strip()
    donors['blood_type']     = donors['blood_type'].str.strip()
    recipients['blood_type'] = recipients['blood_type'].str.strip()

    scorer = CompatibilityScorer()
    
    ranker = TOPSISRanker()

    # Test with first donor
    donor = donors.iloc[0].to_dict()
    recipient_list = recipients.to_dict('records')

    print(f"Donor: {donor['donor_id']} | Blood: {donor['blood_type']}")
    print("Finding best matches...\n")

    ranked = scorer.rank_recipients(donor, recipient_list)
    final  = ranker.rank(ranked[:20])  # TOPSIS on top 20

    print(f"Total compatible recipients: {len(ranked)}")
    print(f"\nTop 5 matches (TOPSIS ranked):")
    for i, match in enumerate(final[:5], 1):
        print(f"{i}. Recipient {match['recipient_id'][:8]}... "
              f"| AHP: {match['ahp_score']} "
              f"| TOPSIS: {match['topsis_score']} "
              f"| HLA: {match['hla_score']}")