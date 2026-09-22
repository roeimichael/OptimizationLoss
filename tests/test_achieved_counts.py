import random
import unittest

from tralo.achieved_counts import achieved_caps
from tralo.global_clipper import allocate


class AchievedCountsTests(unittest.TestCase):
    def test_preserves_mask_zero_and_excess_without_using_labels(self):
        p = [[.8, .1, .1], [.7, .2, .1], [.1, .2, .7]]
        self.assertEqual(achieved_caps(p, [1, 2, None], ['a','b','c']),
                         [2, 0, None])

    def test_full_reallocation_can_change_membership_at_same_count(self):
        p = [[.40,.35,.25],[.39,.32,.29],[.49,.50,.01]]
        ids = ['a','b','c']
        caps = achieved_caps(p, [1,None,None], ids)
        self.assertEqual(caps, [2,None,None])
        self.assertEqual(allocate(p,caps,ids,'upper_bound_correction'), [0,0,1])
        self.assertEqual(allocate(p,caps,ids,'capped_first'), [0,1,0])

    def test_competing_full_destination_is_not_overfilled(self):
        p = [[.9,.05,.05],[.8,.1,.1],[.6,.3,.1],
             [.1,.8,.1],[.1,.7,.2],[.1,.1,.8]]
        self.assertEqual(allocate(p,[2,2,None],list('abcdef'),
                                 'upper_bound_correction'), [0,0,2,1,1,2])

    def test_randomized_identity_exact_counts_and_sample_permutations(self):
        rng = random.Random(812)
        for _ in range(40):
            p = []
            for i in range(17):
                row = [rng.random() for c in range(4)]
                p.append([v/sum(row) for v in row])
            ids = [str(i) for i in range(len(p))]
            caps = achieved_caps(p, [3,4,0,None], ids)
            raw = [max(range(4),key=row.__getitem__) for row in p]
            self.assertEqual(allocate(p,caps,ids,'upper_bound_correction'),raw)
            for policy in ('upper_bound_correction','capped_first'):
                out = allocate(p,caps,ids,policy)
                self.assertEqual([out.count(c) for c in range(3)],caps[:3])
                reverse = allocate(p[::-1],caps,ids[::-1],policy)
                self.assertEqual(reverse[::-1],out)

    def test_rejects_invalid_original_caps(self):
        for caps in ([True,None],[-1,None],[1]):
            with self.assertRaises(ValueError):
                achieved_caps([[.4,.6]],caps,['a'])
