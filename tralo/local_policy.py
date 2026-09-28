"""Allocate a supplied local budget by group size, never evaluation labels."""

from collections import Counter


def size_share_caps(groups, total_local):
    """Hamilton apportionment with lexicographic group-code tie breaking."""
    if not isinstance(groups, (list, tuple)) or not groups or any(
            type(group) is not str or not group for group in groups):
        raise ValueError("groups must be nonempty strings")
    if type(total_local) is not int or not 0 <= total_local <= len(groups):
        raise ValueError("total_local must be an integer in [0, number of items]")
    sizes = Counter(groups)
    floors = {group: total_local * count // len(groups) for group, count in sizes.items()}
    remaining = total_local - sum(floors.values())
    priority = sorted(sizes, key=lambda group: (-((total_local * sizes[group]) % len(groups)), group))
    for group in priority[:remaining]:
        floors[group] += 1
    return dict(sorted(floors.items()))
