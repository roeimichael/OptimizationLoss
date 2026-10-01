"""Prespecified image-plus-metadata referral caps, without outcome labels.

Every sample remains in the pooled cap. A small missing-metadata bucket is
reported but receives no separate local cap. The same caps must be used by
all training arms and by their common deployment allocator.
"""

from collections import Counter
from math import ceil


POLICIES = {
    "isic2020": {"global_rates": (0.01, 0.02),
                 "local_rates": (0.015, 0.03),
                 "local_groups": ("female", "male"),
                 "other_groups": ("missing",)},
    "celeba": {"global_rates": (0.25, 0.35),
               "local_rates": (0.35, 0.45),
               "local_groups": ("female", "male"),
               "other_groups": ()},
}
MIN_LOCAL_SUPPORT = 1000


def caps_for_unlabeled_pool(dataset, groups):
    """Return two immutable cap levels from group counts alone.

    ISIC's 1/2% pooled capacities model scarce specialist referral and its
    1.5/3% per-sex ceilings prevent a single group from consuming all slots.
    CelebA's 25/35% pooled and 35/45% group ceilings are synthetic allocation
    stress tests, not clinical or demographic fairness targets.
    """
    if dataset not in POLICIES:
        raise ValueError("unknown frozen quota dataset")
    policy = POLICIES[dataset]
    if (not isinstance(groups, (list, tuple)) or not groups or
            any(type(group) is not str for group in groups)):
        raise ValueError("one metadata group per development image required")
    counts = Counter(groups)
    if (set(counts) - set(policy["local_groups"]) -
            set(policy["other_groups"]) or
            any(counts[group] < MIN_LOCAL_SUPPORT for group in policy["local_groups"])):
        raise ValueError("unexpected or unsupported local metadata group")
    result = {}
    for level, (global_rate, local_rate) in enumerate(zip(
            policy["global_rates"], policy["local_rates"]), start=1):
        pooled = ceil(len(groups) * global_rate)
        local = {group: ceil(counts[group] * local_rate)
                 for group in policy["local_groups"]}
        if (not 0 < pooled < len(groups) or
                any(not 0 < cap < pooled for cap in local.values()) or
                sum(local.values()) <= pooled):
            raise ValueError("quota geometry is redundant or impossible")
        result[f"level{level}"] = {
            "global_cap": pooled, "local_caps": local,
            "group_counts": dict(sorted(counts.items())),
            "global_rate": global_rate, "local_rate": local_rate,
            "other_groups_global_only": list(policy["other_groups"]),
        }
    return result
