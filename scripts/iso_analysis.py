#!/usr/bin/env python3
"""Estimate isomorphism reduction for various flop textures."""

def card(rank, suit):
    return 4 * rank + suit

def suit_of(c):
    return c & 3

def rank_of(c):
    return c >> 2

def rankset_by_suit(board):
    rs = [0] * 4
    for c in board:
        rs[suit_of(c)] |= 1 << rank_of(c)
    return tuple(rs)

def iso_mapping(rs):
    """Find which higher-numbered suits map to lower-numbered suits."""
    mapping = {}
    for s1 in range(1, 4):
        for s2 in range(s1):
            if rs[s1] == rs[s2]:
                mapping[s1] = s2
                break
    return mapping

def count_canonical_cards(board_so_far):
    """Count canonical (non-eliminated) cards not in board."""
    board_set = set(board_so_far)
    rs = rankset_by_suit(board_so_far)
    mapping = iso_mapping(rs)
    
    canonical = []
    eliminated = 0
    seen_canonical = set()
    
    for c in range(52):
        if c in board_set:
            continue
        s = suit_of(c)
        if s in mapping:
            canon_c = c - s + mapping[s]
            if canon_c not in seen_canonical:
                seen_canonical.add(canon_c)
                canonical.append(canon_c)
            else:
                eliminated += 1
        else:
            seen_canonical.add(c)
            canonical.append(c)
    return canonical, eliminated

def analyze_flop(name, board):
    board_set = set(board)
    remaining = [c for c in range(52) if c not in board_set]
    raw_turn = len(remaining)  # 49
    
    # Turn isomorphism
    turn_canon, turn_elim = count_canonical_cards(board)
    
    # For each canonical turn card, count canonical river outcomes
    total_can = 0
    for tc in turn_canon:
        river_canon, _ = count_canonical_cards(board + [tc])
        total_can += len(river_canon)
    
    raw_total = raw_turn * (raw_turn - 1)  # 49 * 48 = 2352
    ratio = total_can / raw_total if raw_total > 0 else 0
    
    print(f"{name}:")
    print(f"  Turn: {raw_turn} raw → {len(turn_canon)} canonical ({turn_elim} eliminated)")
    print(f"  Total outcomes: {raw_total} raw → {total_can} canonical")
    print(f"  Reduction: {ratio:.2f}x ({(1-ratio)*100:.0f}% fewer outcomes)")

boards = [
    ("2h7dKs (rainbow)",       [card(0,2), card(5,1), card(11,3)]),
    ("2h2dKs (paired)",        [card(0,2), card(0,1), card(11,3)]),
    ("2h7hKh (monotone)",      [card(0,2), card(5,2), card(11,2)]),
    ("2h2d7d (fd+paired)",     [card(0,2), card(0,1), card(5,1)]),
    ("ThTdTs (trips)",         [card(8,2), card(8,1), card(8,3)]),
    ("2h3h4h (monotone low)",  [card(0,2), card(1,2), card(2,2)]),
    ("AhAdAs (trips A)",       [card(12,2), card(12,1), card(12,3)]),
    ("2h2d2s (trips 2)",       [card(0,2), card(0,1), card(0,3)]),
    ("AhKdQs (rainbow broadway)", [card(12,2), card(11,1), card(10,3)]),
    ("7h8d9s (rainbow mid)",   [card(5,2), card(6,1), card(7,3)]),
]

print("Isomorphism analysis (uniform ranges, all range-suit pairs isomorphic)")
print("=" * 70)
for name, board in boards:
    analyze_flop(name, board)
    print()
