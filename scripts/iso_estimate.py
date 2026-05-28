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

def iso_suits(rs, range_iso=True):
    """Find which higher-numbered suits map to lower-numbered suits."""
    mapping = {}
    for s1 in range(1, 4):
        for s2 in range(s1):
            if rs[s1] == rs[s2] and range_iso:
                mapping[s1] = s2
                break
    return mapping

def count_canonical_turn(board):
    """Count canonical turn outcomes after isomorphism."""
    board_set = set(board)
    rs = rankset_by_suit(board)
    mapping = iso_suits(rs)
    
    canonical = 0
    eliminated = 0
    seen_canonical = set()
    
    for c in range(52):
        if c in board_set:
            continue
        s = suit_of(c)
        if s in mapping:
            # Map to canonical suit
            canon_c = c - s + mapping[s]
            if canon_c not in seen_canonical:
                seen_canonical.add(canon_c)
                canonical += 1
            else:
                eliminated += 1
        else:
            canonical += 1
    return canonical, eliminated

def count_canonical_river(board, turn_card):
    """Count canonical river outcomes for a given turn card."""
    board_plus = board + [turn_card]
    board_set = set(board_plus)
    rs = rankset_by_suit(board_plus)
    mapping = iso_suits(rs)
    
    canonical = 0
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
                canonical += 1
            else:
                eliminated += 1
        else:
            canonical += 1
    return canonical, eliminated

def total_canonical_outcomes(board):
    board_set = set(board)
    remaining = [c for c in range(52) if c not in board_set]
    
    turn_can, turn_elim = count_canonical_turn(board)
    
    total_river_can = 0
    total_river_raw = 0
    # For each canonical turn card, count river reduction
    for tc in remaining:
        river_can, _ = count_canonical_river(board, tc)
        total_river_can += river_can
        total_river_raw += len([c for c in range(52) if c not in board_set and c != tc])
    
    # Weight: eliminated turn cards have 0 river outcomes
    total_raw = len(remaining) * (len(remaining) - 1)
    total_can = turn_can * 0  # placeholder
    
    # More precise: sum river outcomes for each canonical turn, 
    # but eliminated turns share the canonical's river outcomes
    # Actually the reduction is:
    # - For each canonical turn card, it has its own river outcomes
    # - For each eliminated turn card, it maps to a canonical turn card's river outcomes
    #   (with suit swap), so NO new river computation needed
    
    # Total raw outcomes = 49 turn * 48 river = 2352
    # Total canonical outcomes = sum over canonical turns of (canonical rivers for that turn)
    # BUT eliminated turns map to canonical turns, so their river outcomes are also canonical
    # The actual computation is: for each canonical turn card, compute its canonical river outcomes
    # Each eliminated turn card just needs a swap of the canonical turn's river CFVs
    
    total_can_outcomes = 0
    for tc in remaining:
        river_can, _ = count_canonical_river(board, tc)
        # Is this turn card canonical?
        s = suit_of(tc)
        rs = rankset_by_suit(board)
        mapping = iso_suits(rs)
        if s not in mapping:
            total_can_outcomes += river_can
        else:
            # This turn maps to a canonical. Don't add its river outcomes.
            pass
    
    return total_raw, total_can_outcomes, turn_can, turn_elim

# Detailed analysis for each board
for name, board in boards.items():
    board_set = set(board)
    remaining = [c for c in range(52) if c not in board_set]
    rs = rankset_by_suit(board)
    mapping = iso_suits(rs)
    
    turn_canonical = []
    turn_eliminated = 0
    for tc in remaining:
        s = suit_of(tc)
        if s in mapping:
            turn_eliminated += 1
        else:
            turn_canonical.append(tc)
    
    # For each canonical turn, count canonical river outcomes
    total_can = 0
    for tc in turn_canonical:
        rc, _ = count_canonical_river(board, tc)
        total_can += rc
    
    raw = len(remaining) * (len(remaining) - 1)
    ratio = total_can / raw if raw > 0 else 0
    print(f"{name}: {len(turn_canonical)} canonical turns, {turn_eliminated} eliminated")
    print(f"  Total: {raw} raw → {total_can} canonical ({ratio:.2f}x)")

print()
print("=""  # separator

boards = {
    "2h7dKs (rainbow)": [card(0,2), card(5,1), card(11,3)],  # rainbow
    "2h2dKs (paired)":  [card(0,2), card(0,1), card(11,3)],   # paired, h=d
    "2h7hKh (monotone)": [card(0,2), card(5,2), card(11,2)],   # monotone, d=s=c iso
    "2h2d7d (fd+paired)": [card(0,2), card(0,1), card(5,1)],   # paired, flush draw possible
    "ThTdTs (trips)":   [card(8,2), card(8,1), card(8,3)],     # trips
    "2h3h4h (monotone low)": [card(0,2), card(1,2), card(2,2)], # monotone low
}

print(f"{'Board':<25} {'Raw':>6} {'Canon':>6} {'Ratio':>6} {'TurnC':>6} {'TurnE':>6}")
print("-" * 65)
for name, board in boards.items():
    raw, can, tc, te = total_canonical_outcomes(board)
    ratio = can / raw if raw > 0 else 0
    print(f"{name:<25} {raw:>6} {can:>6} {ratio:>6.2f} {tc:>6} {te:>6}")
