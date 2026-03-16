from __future__ import annotations

import random
from collections import defaultdict

from schedule.data_loader import Lesson, PFK_DISC, PFK_TEACHER

WEEKS = 2
DAYS = 5
SLOTS_PER_DAY = 6
TOTAL_SLOTS = WEEKS * DAYS * SLOTS_PER_DAY  # 60
MAX_PAIRS_PER_DAY = 5

# Slots within a day that are allowed for ПФК (pairs 1–3, 0-indexed)
PFK_VALID_SIDS = frozenset({0, 1, 2})

# Slot weights by slot_in_day: pairs 1-4 (sid 0-3) = 1.0,
# pair 5 (sid 4) = 0.25, pair 6 (sid 5) = 0.07
_SID_WEIGHT = [1.0, 1.0, 1.0, 1.0, 0.25, 0.07]


def slot_to_wds(slot: int) -> tuple[int, int, int]:
    """Convert flat slot index to (week, day, slot_in_day), all 0-based."""
    week = slot // (DAYS * SLOTS_PER_DAY)
    remainder = slot % (DAYS * SLOTS_PER_DAY)
    day = remainder // SLOTS_PER_DAY
    slot_in_day = remainder % SLOTS_PER_DAY
    return week, day, slot_in_day


def wds_to_slot(week: int, day: int, slot_in_day: int) -> int:
    return week * DAYS * SLOTS_PER_DAY + day * SLOTS_PER_DAY + slot_in_day


def _day_of(slot: int) -> tuple[int, int]:
    """Return (week, day) for a slot."""
    week = slot // (DAYS * SLOTS_PER_DAY)
    day = (slot % (DAYS * SLOTS_PER_DAY)) // SLOTS_PER_DAY
    return week, day


def _affected_groups(lesson: Lesson) -> list[str]:
    """Groups that cannot share the slot with this lesson."""
    return lesson.stream_groups if lesson.stream_groups else [lesson.group]


def make_initial_chromosome(lessons: list[Lesson]) -> list[int]:
    """
    Greedy construction with smart ПФК placement first.

    ПФК rules (hard):
      - All sessions of the same year-stream share the same (day, slot_in_day)
        in both weeks (e.g. Monday pair-2 in week 0 AND week 1).
      - Only slots with slot_in_day ∈ {0, 1, 2} (pairs 1–3).

    All other lessons: assign slots without teacher/group conflicts,
    respecting the max-5-pairs-per-day limit.
    """
    chromosome: list[int] = [-1] * len(lessons)

    teacher_used: dict[str, set[int]] = defaultdict(set)
    group_used: dict[str, set[int]] = defaultdict(set)
    teacher_day_count: dict[tuple[str, int, int], int] = defaultdict(int)
    group_day_count: dict[tuple[str, int, int], int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Phase 1: place ПФК stream sessions (both weeks on same day/slot)    #
    # ------------------------------------------------------------------ #

    # Group ПФК lesson indices by stream id
    pfk_by_stream: dict[str, list[int]] = defaultdict(list)
    for i, lesson in enumerate(lessons):
        if lesson.discipline_short == PFK_DISC:
            pfk_by_stream[lesson.group].append(i)

    pfk_placed: set[int] = set()

    for stream_id, stream_idxs in pfk_by_stream.items():
        lesson = lessons[stream_idxs[0]]
        groups = _affected_groups(lesson)
        count = len(stream_idxs)  # typically 2

        # Try to find a (day, sid in PFK_VALID_SIDS) where:
        #  - all week-slots (wds_to_slot(w, day, sid) for w in 0..count-1)
        #    are free for every group AND the group day-count limit is met
        day_order = list(range(DAYS))
        random.shuffle(day_order)
        placed = False

        for strict in (True, False):
            if placed:
                break
            for day in day_order:
                for sid in PFK_VALID_SIDS:
                    ok = True
                    for w in range(count):
                        s = wds_to_slot(w % WEEKS, day, sid)
                        for grp in groups:
                            if s in group_used[grp]:
                                ok = False
                                break
                            if strict and group_day_count[(grp, w % WEEKS, day)] >= MAX_PAIRS_PER_DAY:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for k, idx in enumerate(stream_idxs):
                            w = k % WEEKS
                            s = wds_to_slot(w, day, sid)
                            chromosome[idx] = s
                            wd = (w, day)
                            for grp in groups:
                                group_used[grp].add(s)
                                group_day_count[(grp, *wd)] += 1
                            pfk_placed.add(idx)
                        placed = True
                        break
                if placed:
                    break

        if not placed:
            # Last resort: just use first available sid-constrained slot per week
            for k, idx in enumerate(stream_idxs):
                w = k % WEEKS
                pfk_slots = [
                    wds_to_slot(w, d, s)
                    for d in range(DAYS)
                    for s in PFK_VALID_SIDS
                ]
                free = [s for s in pfk_slots if all(s not in group_used[g] for g in groups)]
                slot = random.choice(free) if free else random.choice(pfk_slots)
                chromosome[idx] = slot
                wd = _day_of(slot)
                for grp in groups:
                    group_used[grp].add(slot)
                    group_day_count[(grp, *wd)] += 1
                pfk_placed.add(idx)

    # ------------------------------------------------------------------ #
    # Phase 2: place remaining lessons greedily                           #
    # Slots are weighted so pairs 1-4 are strongly preferred over 5-6.  #
    # ------------------------------------------------------------------ #

    def _weighted_choice(candidates: list[int]) -> int:
        weights = [_SID_WEIGHT[s % SLOTS_PER_DAY] for s in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    remaining = [i for i in range(len(lessons)) if i not in pfk_placed]
    random.shuffle(remaining)

    all_slots = list(range(TOTAL_SLOTS))

    for i in remaining:
        lesson = lessons[i]
        groups = _affected_groups(lesson)
        is_pfk = lesson.teacher == PFK_TEACHER  # should not occur here, but guard

        def is_ok(s: int) -> bool:
            if not is_pfk and s in teacher_used[lesson.teacher]:
                return False
            wd = _day_of(s)
            if not is_pfk and teacher_day_count[(lesson.teacher, *wd)] >= MAX_PAIRS_PER_DAY:
                return False
            for grp in groups:
                if s in group_used[grp]:
                    return False
                if group_day_count[(grp, *wd)] >= MAX_PAIRS_PER_DAY:
                    return False
            return True

        candidates = [s for s in all_slots if is_ok(s)]

        if candidates:
            slot = _weighted_choice(candidates)
        else:
            # Relax daily limit
            def is_ok_relaxed(s: int) -> bool:
                if not is_pfk and s in teacher_used[lesson.teacher]:
                    return False
                for grp in groups:
                    if s in group_used[grp]:
                        return False
                return True

            candidates2 = [s for s in all_slots if is_ok_relaxed(s)]
            if candidates2:
                slot = _weighted_choice(candidates2)
            else:
                if not is_pfk:
                    candidates3 = [s for s in all_slots if s not in teacher_used[lesson.teacher]]
                    slot = _weighted_choice(candidates3) if candidates3 else _weighted_choice(all_slots)
                else:
                    slot = _weighted_choice(all_slots)

        chromosome[i] = slot
        wd = _day_of(slot)

        if not is_pfk:
            teacher_used[lesson.teacher].add(slot)
            teacher_day_count[(lesson.teacher, *wd)] += 1

        for grp in groups:
            group_used[grp].add(slot)
            group_day_count[(grp, *wd)] += 1

    return chromosome
