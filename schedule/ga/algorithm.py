from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from schedule.data_loader import Lesson, PFK_DISC, PFK_TEACHER
from schedule.ga.chromosome import (
    DAYS,
    MAX_PAIRS_PER_DAY,
    PFK_VALID_SIDS,
    SLOTS_PER_DAY,
    TOTAL_SLOTS,
    WEEKS,
    _SID_WEIGHT,
    _affected_groups,
    _day_of,
    make_initial_chromosome,
    slot_to_wds,
    wds_to_slot,
)
from schedule.ga.fitness import evaluate

Chromosome = list[int]


# ------------------------------------------------------------------ #
# Index helpers                                                        #
# ------------------------------------------------------------------ #

def _build_index(
    chromosome: Chromosome,
    lessons: list[Lesson],
) -> tuple[
    dict[tuple[str, int], int],
    dict[tuple[str, int, int], int],
    dict[tuple[str, int, int], int],
    dict[tuple[str, int, int], int],
]:
    """
    Build lookup dicts for fast conflict checking.
    Returns:
        teacher_slot : (teacher, slot) -> lesson_idx
        group_slot   : (group, slot, subgroup) -> lesson_idx
        teacher_day  : (teacher, week, day) -> count
        group_day    : (group, week, day) -> count
    """
    teacher_slot: dict[tuple[str, int], int] = {}
    group_slot: dict[tuple[str, int, int], int] = {}
    teacher_day: dict[tuple[str, int, int], int] = defaultdict(int)
    group_day: dict[tuple[str, int, int], int] = defaultdict(int)

    for i, slot in enumerate(chromosome):
        s = slot % TOTAL_SLOTS
        lesson = lessons[i]
        wd = _day_of(s)
        if lesson.teacher != PFK_TEACHER:
            teacher_slot[(lesson.teacher, s)] = i
            teacher_day[(lesson.teacher, *wd)] += 1
        for grp in _affected_groups(lesson):
            group_slot[(grp, s, lesson.subgroup)] = i
            group_day[(grp, *wd)] += 1

    return teacher_slot, group_slot, teacher_day, group_day


# ------------------------------------------------------------------ #
# ПФК-specific repair                                                  #
# ------------------------------------------------------------------ #

def _repair_pfk(chromosome: Chromosome, lessons: list[Lesson]) -> Chromosome:
    """
    Ensure every ПФК year-stream has:
      - exactly 2 sessions
      - one session in week 0, one in week 1
      - both sessions on the same (day, slot_in_day)
      - slot_in_day ∈ PFK_VALID_SIDS (pairs 1–3)

    This is called before the general repair so ПФК slots are locked first.
    """
    result = chromosome[:]

    # Group ПФК sessions by stream id
    pfk_by_stream: dict[str, list[int]] = defaultdict(list)
    for i, lesson in enumerate(lessons):
        if lesson.discipline_short == PFK_DISC:
            pfk_by_stream[lesson.group].append(i)

    # Build group occupancy for non-ПФК lessons (so we can avoid them)
    group_used: dict[str, set[int]] = defaultdict(set)
    for i, slot in enumerate(result):
        lesson = lessons[i]
        if lesson.discipline_short == PFK_DISC:
            continue
        s = slot % TOTAL_SLOTS
        for grp in _affected_groups(lesson):
            group_used[grp].add(s)

    for stream_id, idxs in pfk_by_stream.items():
        lesson = lessons[idxs[0]]
        groups = _affected_groups(lesson)
        count = len(idxs)

        # Check if current placement is already valid
        current = [(result[i] % TOTAL_SLOTS) for i in idxs]
        current_wds = [slot_to_wds(s) for s in current]
        day_sids = {(d, sid) for (_, d, sid) in current_wds}
        weeks = {w for (w, _, _) in current_wds}
        sids = {sid for (_, _, sid) in current_wds}

        if (
            len(current) == count
            and len(day_sids) == 1
            and weeks == set(range(count))
            and all(s in PFK_VALID_SIDS for s in sids)
        ):
            # Already valid — mark slots as used
            for s in current:
                for grp in groups:
                    group_used[grp].add(s)
            continue

        # Remove old occupancy that came from ПФК (will be re-set)
        for s in current:
            for grp in groups:
                group_used[grp].discard(s)

        # Search for a valid (day, sid)
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
                        if not ok:
                            break
                    if ok:
                        for k, idx in enumerate(idxs):
                            w = k % WEEKS
                            s = wds_to_slot(w, day, sid)
                            result[idx] = s
                            for grp in groups:
                                group_used[grp].add(s)
                        placed = True
                        break
                if placed:
                    break

        if not placed:
            # Last resort: ignore group conflicts, just pick sid-valid slots
            day = random.randint(0, DAYS - 1)
            sid = random.choice(list(PFK_VALID_SIDS))
            for k, idx in enumerate(idxs):
                w = k % WEEKS
                s = wds_to_slot(w, day, sid)
                result[idx] = s
                for grp in groups:
                    group_used[grp].add(s)

    return result


# ------------------------------------------------------------------ #
# General repair                                                       #
# ------------------------------------------------------------------ #

def _has_hard_conflicts(chromosome: Chromosome, lessons: list[Lesson]) -> bool:
    """Return True if there are any teacher or group slot conflicts."""
    teacher_seen: dict[tuple[str, int], int] = {}
    group_seen: dict[tuple[str, int, int], int] = {}
    for i, slot in enumerate(chromosome):
        s = slot % TOTAL_SLOTS
        lesson = lessons[i]
        if lesson.teacher != PFK_TEACHER:
            key = (lesson.teacher, s)
            if key in teacher_seen:
                return True
            teacher_seen[key] = i
        for grp in _affected_groups(lesson):
            key2 = (grp, s, lesson.subgroup)
            if key2 in group_seen:
                return True
            group_seen[key2] = i
    return False


def _repair_pass(chromosome: Chromosome, lessons: list[Lesson]) -> Chromosome:
    """
    Single repair pass.

    ПФК lessons are processed FIRST so their slots are locked before non-ПФК
    lessons are evaluated — this prevents the ordering bug where a non-ПФК
    lesson registers at the same slot as a later ПФК lesson.

    Reassigned lessons use weighted slot selection (prefers pairs 1-4).
    """
    result = chromosome[:]

    teacher_slot: dict[tuple[str, int], int] = {}
    group_slot: dict[tuple[str, int, int], int] = {}
    teacher_day: dict[tuple[str, int, int], int] = defaultdict(int)
    group_day: dict[tuple[str, int, int], int] = defaultdict(int)
    to_reassign: list[int] = []

    # --- Phase A: lock ПФК slots first ---
    for i, slot in enumerate(result):
        lesson = lessons[i]
        if lesson.teacher != PFK_TEACHER:
            continue
        s = slot % TOTAL_SLOTS
        wd = _day_of(s)
        for grp in _affected_groups(lesson):
            group_slot[(grp, s, lesson.subgroup)] = i
            group_day[(grp, *wd)] += 1

    # --- Phase B: scan non-ПФК, detect conflicts ---
    for i, slot in enumerate(result):
        lesson = lessons[i]
        if lesson.teacher == PFK_TEACHER:
            continue  # already registered above
        s = slot % TOTAL_SLOTS
        wd = _day_of(s)
        groups = _affected_groups(lesson)

        conflict = (
            (lesson.teacher, s) in teacher_slot
            or teacher_day[(lesson.teacher, *wd)] >= MAX_PAIRS_PER_DAY
        )
        if not conflict:
            for grp in groups:
                if (grp, s, lesson.subgroup) in group_slot or group_day[(grp, *wd)] >= MAX_PAIRS_PER_DAY:
                    conflict = True
                    break

        if conflict:
            to_reassign.append(i)
        else:
            teacher_slot[(lesson.teacher, s)] = i
            teacher_day[(lesson.teacher, *wd)] += 1
            for grp in groups:
                group_slot[(grp, s, lesson.subgroup)] = i
                group_day[(grp, *wd)] += 1

    # --- Phase C: reassign conflicting lessons ---
    for i in to_reassign:
        lesson = lessons[i]
        groups = _affected_groups(lesson)

        def is_free(s: int) -> bool:
            wd = _day_of(s)
            if (
                (lesson.teacher, s) in teacher_slot
                or teacher_day[(lesson.teacher, *wd)] >= MAX_PAIRS_PER_DAY
            ):
                return False
            for grp in groups:
                if (grp, s, lesson.subgroup) in group_slot or group_day[(grp, *wd)] >= MAX_PAIRS_PER_DAY:
                    return False
            return True

        candidates = [s for s in range(TOTAL_SLOTS) if is_free(s)]
        if not candidates:
            # Relax daily limit — only avoid exact slot collisions
            def is_free_relaxed(s: int) -> bool:
                if (lesson.teacher, s) in teacher_slot:
                    return False
                for grp in groups:
                    if (grp, s, lesson.subgroup) in group_slot:
                        return False
                return True
            candidates = [s for s in range(TOTAL_SLOTS) if is_free_relaxed(s)]

        if candidates:
            weights = [_SID_WEIGHT[s % SLOTS_PER_DAY] for s in candidates]
            slot = random.choices(candidates, weights=weights, k=1)[0]
        else:
            # No conflict-free slot found — pick any (very rare edge case)
            slot = random.choices(
                range(TOTAL_SLOTS),
                weights=[_SID_WEIGHT[s % SLOTS_PER_DAY] for s in range(TOTAL_SLOTS)],
                k=1,
            )[0]

        result[i] = slot
        wd = _day_of(slot)
        teacher_slot[(lesson.teacher, slot)] = i
        teacher_day[(lesson.teacher, *wd)] += 1
        for grp in groups:
            group_slot[(grp, slot, lesson.subgroup)] = i
            group_day[(grp, *wd)] += 1

    return result


def _repair(chromosome: Chromosome, lessons: list[Lesson]) -> Chromosome:
    """
    Guaranteed repair: loop until zero teacher/group slot conflicts.

    Order:
      1. Fix ПФК stream alignment (_repair_pfk).
      2. Repeat _repair_pass until no teacher or group slot conflicts remain
         (max 20 iterations to avoid infinite loop; in practice 1-3 are enough).
    """
    result = _repair_pfk(chromosome, lessons)

    for _ in range(20):
        if not _has_hard_conflicts(result, lessons):
            break
        result = _repair_pass(result, lessons)

    return result


# ------------------------------------------------------------------ #
# Local search (soft constraint improvement)                           #
# ------------------------------------------------------------------ #

def _try_move(
    result: Chromosome,
    lesson_idx: int,
    target_slot: int,
    lessons: list[Lesson],
    teacher_slot: dict,
    group_slot: dict,
    teacher_day: dict,
    group_day: dict,
) -> bool:
    """
    Attempt to move lesson at lesson_idx to target_slot.
    Returns True if the move succeeded (index dicts updated in-place).
    """
    lesson = lessons[lesson_idx]
    old_slot = result[lesson_idx] % TOTAL_SLOTS

    if old_slot == target_slot:
        return False

    # Verify the index is consistent for this lesson
    if lesson.teacher != PFK_TEACHER:
        if teacher_slot.get((lesson.teacher, old_slot)) != lesson_idx:
            return False
    if group_slot.get((lesson.group, old_slot, lesson.subgroup)) != lesson_idx:
        return False

    # Check target slot is free
    if lesson.teacher != PFK_TEACHER and (lesson.teacher, target_slot) in teacher_slot:
        return False
    for grp in _affected_groups(lesson):
        if (grp, target_slot, lesson.subgroup) in group_slot:
            return False

    old_wd = _day_of(old_slot)
    new_wd = _day_of(target_slot)
    if lesson.teacher != PFK_TEACHER and teacher_day[(lesson.teacher, *new_wd)] >= MAX_PAIRS_PER_DAY:
        return False
    for grp in _affected_groups(lesson):
        if group_day[(grp, *new_wd)] >= MAX_PAIRS_PER_DAY:
            return False

    # Apply move
    if lesson.teacher != PFK_TEACHER:
        teacher_slot.pop((lesson.teacher, old_slot), None)
        teacher_day[(lesson.teacher, *old_wd)] -= 1
        teacher_slot[(lesson.teacher, target_slot)] = lesson_idx
        teacher_day[(lesson.teacher, *new_wd)] += 1

    for grp in _affected_groups(lesson):
        group_slot.pop((grp, old_slot, lesson.subgroup), None)
        group_day[(grp, *old_wd)] -= 1
        group_slot[(grp, target_slot, lesson.subgroup)] = lesson_idx
        group_day[(grp, *new_wd)] += 1

    result[lesson_idx] = target_slot
    return True


def _local_search(
    chromosome: Chromosome,
    lessons: list[Lesson],
    iterations: int = 50,
) -> Chromosome:
    """
    Targeted local search:
      1. Move lessons from late slots (pairs 5-6) to earlier free slots.
      2. Fill window gaps within a day by moving in a lesson from elsewhere.
    Does NOT move ПФК sessions.
    """
    result = chromosome[:]
    teacher_slot, group_slot, teacher_day, group_day = _build_index(result, lessons)

    # Exclude ПФК from all local moves
    non_pfk: list[int] = [
        i for i, lesson in enumerate(lessons)
        if lesson.discipline_short != PFK_DISC
    ]
    group_indices: dict[str, list[int]] = defaultdict(list)
    for i in non_pfk:
        group_indices[lessons[i].group].append(i)

    if not non_pfk:
        return result

    for it in range(iterations):
        # ---- Step A (first half of iterations): move late pairs to early slots ----
        if it < iterations // 2:
            late_indices = [
                i for i in non_pfk
                if slot_to_wds(result[i] % TOTAL_SLOTS)[2] >= 4  # sid 4 or 5
            ]
            if not late_indices:
                break  # no more late slots — done early
            lesson_idx = random.choice(late_indices)
            lesson = lessons[lesson_idx]
            old_s = result[lesson_idx] % TOTAL_SLOTS
            old_w, old_d, _ = slot_to_wds(old_s)

            # Try all early slots in the same week, prefer same day
            early_slots = [
                wds_to_slot(old_w, d, sid)
                for d in range(DAYS)
                for sid in range(4)  # pairs 1–4
            ]
            # Prioritise same day first
            early_slots.sort(key=lambda s: (0 if _day_of(s)[1] == old_d else 1, s % SLOTS_PER_DAY))
            for tgt in early_slots:
                if _try_move(result, lesson_idx, tgt, lessons, teacher_slot, group_slot, teacher_day, group_day):
                    break

        # ---- Step B (second half): fill window gaps ----
        else:
            if not group_indices:
                continue
            group = random.choice(list(group_indices.keys()))
            indices = group_indices[group]
            if len(indices) < 2:
                continue

            week = random.randint(0, WEEKS - 1)
            day_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for i in indices:
                s = result[i] % TOTAL_SLOTS
                w, d, sid = slot_to_wds(s)
                if w == week:
                    day_slots[d].append((sid, i))

            gap_day = None
            target_sid = None
            for d, entries in sorted(day_slots.items()):
                if len(entries) >= 2:
                    sorted_entries = sorted(entries)
                    for k in range(1, len(sorted_entries)):
                        if sorted_entries[k][0] - sorted_entries[k - 1][0] > 1:
                            gap_day = d
                            target_sid = sorted_entries[k - 1][0] + 1
                            break
                if gap_day is not None:
                    break

            if gap_day is None or target_sid is None or target_sid >= SLOTS_PER_DAY:
                continue

            target_slot = wds_to_slot(week, gap_day, target_sid)
            candidates_to_move = [
                i for i in indices
                if _day_of(result[i] % TOTAL_SLOTS) != (week, gap_day)
            ]
            if not candidates_to_move:
                continue

            lesson_idx = random.choice(candidates_to_move)
            _try_move(result, lesson_idx, target_slot, lessons, teacher_slot, group_slot, teacher_day, group_day)

    return result


# ------------------------------------------------------------------ #
# GA operators                                                         #
# ------------------------------------------------------------------ #

def _tournament_select(
    population: list[Chromosome],
    fitnesses: list[float],
    k: int = 3,
) -> Chromosome:
    contestants = random.sample(range(len(population)), k)
    best = max(contestants, key=lambda i: fitnesses[i])
    return population[best][:]


def _one_point_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
) -> tuple[Chromosome, Chromosome]:
    if len(parent1) < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def _swap_mutate(
    chromosome: Chromosome,
    lessons: list[Lesson],
    mutation_rate: float,
) -> Chromosome:
    """
    Swap-based mutation.  ПФК sessions from different streams are never swapped
    together (to preserve alignment).  Within the same ПФК stream swaps are
    also skipped because alignment is week-indexed.
    """
    result = chromosome[:]
    n = len(result)
    swaps = max(1, int(n * mutation_rate))

    # Precompute which indices are ПФК (to avoid disrupting their alignment)
    pfk_indices: set[int] = {
        i for i, lesson in enumerate(lessons) if lesson.discipline_short == PFK_DISC
    }

    for _ in range(swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        # Don't swap ПФК with anything — alignment is managed by repair
        if i in pfk_indices or j in pfk_indices:
            continue
        result[i], result[j] = result[j], result[i]

    return result


# ------------------------------------------------------------------ #
# Main GA                                                              #
# ------------------------------------------------------------------ #

def run_ga(
    lessons: list[Lesson],
    population_size: int = 40,
    generations: int = 200,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.02,
    elite_count: int = 3,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> tuple[Chromosome, float, list[float]]:
    """
    Run the genetic algorithm.

    Returns:
        (best_chromosome, best_fitness, fitness_history)
    """
    if not lessons:
        return [], 0.0, []

    # Initialize population: greedy + ПФК repair + local search
    population: list[Chromosome] = []
    for _ in range(population_size):
        c = make_initial_chromosome(lessons)
        c = _repair_pfk(c, lessons)            # guarantee ПФК from the start
        c = _repair_pass(c, lessons)           # fix any residual hard violations
        c = _local_search(c, lessons, iterations=200)
        population.append(c)

    fitnesses = [evaluate(c, lessons) for c in population]
    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_chromosome = population[best_idx][:]
    best_fitness = fitnesses[best_idx]
    fitness_history: list[float] = [best_fitness]

    for gen in range(generations):
        new_population: list[Chromosome] = []

        # Elitism
        elite_indices = sorted(
            range(len(fitnesses)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx][:])

        while len(new_population) < population_size:
            parent1 = _tournament_select(population, fitnesses)
            parent2 = _tournament_select(population, fitnesses)

            if random.random() < crossover_rate:
                child1, child2 = _one_point_crossover(parent1, parent2)
                # Repair after crossover (ПФК first, then general)
                child1 = _repair(child1, lessons)
                child2 = _repair(child2, lessons)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = _swap_mutate(child1, lessons, mutation_rate)
            child2 = _swap_mutate(child2, lessons, mutation_rate)

            # Re-apply ПФК repair after mutation (mutation skips ПФК, but just in case)
            child1 = _repair_pfk(child1, lessons)
            child2 = _repair_pfk(child2, lessons)

            # Apply local search to improve soft constraints
            if random.random() < 0.4:
                child1 = _local_search(child1, lessons, iterations=30)
            if random.random() < 0.4:
                child2 = _local_search(child2, lessons, iterations=30)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        fitnesses = [evaluate(c, lessons) for c in population]

        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = population[gen_best_idx][:]

        fitness_history.append(best_fitness)

        if progress_callback:
            progress_callback(gen + 1, generations, best_fitness)

    # Final guaranteed repair: ensure the winning chromosome has zero
    # teacher/group slot conflicts regardless of what the GA produced.
    best_chromosome = _repair(best_chromosome, lessons)

    return best_chromosome, best_fitness, fitness_history
