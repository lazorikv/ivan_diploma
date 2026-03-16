from __future__ import annotations

from collections import defaultdict

from schedule.data_loader import Lesson, PFK_DISC, PFK_TEACHER
from schedule.ga.chromosome import PFK_VALID_SIDS, SLOTS_PER_DAY, TOTAL_SLOTS, slot_to_wds

# Hard penalty weights — teacher/group slot conflicts are completely forbidden
HARD_TEACHER_CONFLICT = 500
HARD_GROUP_CONFLICT = 500
HARD_MAX_PAIRS_PER_DAY = 300
HARD_MAX_PAIRS_PER_WEEK = 300
HARD_PFK_PER_WEEK = 300           # >1 ПФК per week per group
HARD_PFK_STREAM_SLOT = 500        # stream not aligned to same (day, sid) in both weeks
HARD_PFK_BAD_SLOT = 400           # ПФК not in pairs 1–3

# Soft penalty weights
SOFT_WINDOW = 2
SOFT_UNEVEN = 4
# Late slots (pairs 5–6) — high penalty so GA strongly avoids them.
# Applied per lesson + per teacher = up to 40 pts per lesson vs 2 pts per gap.
LATE_SLOT_PENALTY = 20
LATE_SLOTS = {4, 5}               # slot_in_day (0-based): 4=5th pair, 5=6th pair

# Daily limits
STUDENT_MAX_PAIRS_PER_DAY = 5
TEACHER_MAX_PAIRS_PER_DAY = 5
STUDENT_MAX_PAIRS_PER_WEEK = 27   # ≈54 academic hours / 2 per lesson


def _effective_groups(lesson: Lesson) -> list[str]:
    if lesson.stream_groups:
        return lesson.stream_groups
    return [lesson.group]


def evaluate(chromosome: list[int], lessons: list[Lesson]) -> float:
    """
    Compute fitness.  fitness = 1 / (1 + penalty).
    Perfect schedule → penalty = 0 → fitness = 1.0.
    """
    penalty = 0

    teacher_slot: dict[tuple[str, int], int] = defaultdict(int)
    group_slot: dict[tuple[str, int, int], int] = defaultdict(int)
    wd_teacher: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    wd_group: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    group_total_pairs: dict[str, int] = defaultdict(int)
    pfk_week_pairs: dict[tuple[str, int], int] = defaultdict(int)
    # For ПФК stream alignment: stream_id → list of (week, day, sid)
    pfk_stream_wds: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    for idx, slot in enumerate(chromosome):
        s = slot % TOTAL_SLOTS
        week, day, sid = slot_to_wds(s)
        lesson = lessons[idx]
        groups = _effective_groups(lesson)

        if lesson.teacher != PFK_TEACHER:
            teacher_slot[(lesson.teacher, s)] += 1
            wd_teacher[(week, day, lesson.teacher)].append(sid)

        for grp in groups:
            group_slot[(grp, s, lesson.subgroup)] += 1
            wd_group[(week, day, grp)].append(sid)
            group_total_pairs[grp] += 1
            if lesson.discipline_short == PFK_DISC:
                pfk_week_pairs[(grp, week)] += 1

        if lesson.discipline_short == PFK_DISC and lesson.stream_groups:
            pfk_stream_wds[lesson.group].append((week, day, sid))

    # ------------------------------------------------------------------ #
    # Hard: teacher conflicts                                             #
    # ------------------------------------------------------------------ #
    for count in teacher_slot.values():
        if count > 1:
            penalty += HARD_TEACHER_CONFLICT * (count - 1)

    # Hard: group conflicts
    for count in group_slot.values():
        if count > 1:
            penalty += HARD_GROUP_CONFLICT * (count - 1)

    # Hard: max 5 pairs per day per group
    for slots in wd_group.values():
        if len(slots) > STUDENT_MAX_PAIRS_PER_DAY:
            penalty += HARD_MAX_PAIRS_PER_DAY * (len(slots) - STUDENT_MAX_PAIRS_PER_DAY)

    # Hard: max 5 pairs per day per teacher
    for slots in wd_teacher.values():
        if len(slots) > TEACHER_MAX_PAIRS_PER_DAY:
            penalty += HARD_MAX_PAIRS_PER_DAY * (len(slots) - TEACHER_MAX_PAIRS_PER_DAY)

    # Soft: total student load over 2 weeks
    for grp, cnt in group_total_pairs.items():
        if cnt > STUDENT_MAX_PAIRS_PER_WEEK:
            penalty += cnt - STUDENT_MAX_PAIRS_PER_WEEK

    # Hard: ПФК at most once per week per group
    for (grp, week), cnt in pfk_week_pairs.items():
        if cnt > 1:
            penalty += HARD_PFK_PER_WEEK * (cnt - 1)

    # Hard: ПФК stream — same (day, sid) in both weeks; sid in {0,1,2}
    for stream_id, locs in pfk_stream_wds.items():
        sids = {sid for (_, _, sid) in locs}
        day_sids = {(d, sid) for (_, d, sid) in locs}
        weeks = {w for (w, _, _) in locs}
        expected_count = 2  # one per week

        # Bad slot penalty: any session not in pairs 1–3
        for sid in sids:
            if sid not in PFK_VALID_SIDS:
                penalty += HARD_PFK_BAD_SLOT

        # Stream alignment: both sessions must share same (day, sid)
        if len(locs) != expected_count or len(day_sids) != 1 or weeks != {0, 1}:
            misalign = max(1, abs(len(locs) - expected_count) + (len(day_sids) - 1))
            penalty += HARD_PFK_STREAM_SLOT * misalign

    # Soft: windows (gaps) between classes per group per day
    for slots in wd_group.values():
        if len(slots) > 1:
            ss = sorted(slots)
            for i in range(1, len(ss)):
                gap = ss[i] - ss[i - 1] - 1
                if gap > 0:
                    penalty += SOFT_WINDOW * gap

    # Soft: late slots (5th–6th pair) — penalise per lesson (not per group/teacher)
    for idx, slot in enumerate(chromosome):
        s = slot % TOTAL_SLOTS
        _, _, sid = slot_to_wds(s)
        lesson = lessons[idx]
        if sid in LATE_SLOTS:
            penalty += LATE_SLOT_PENALTY
            if lesson.teacher != PFK_TEACHER:
                penalty += LATE_SLOT_PENALTY

    # ------------------------------------------------------------------ #
    # Extra steep penalty when hard violations > 5                        #
    # ------------------------------------------------------------------ #
    hard_violations = _count_hard_violations(
        teacher_slot, group_slot, wd_teacher, wd_group,
        pfk_week_pairs, pfk_stream_wds,
    )
    if hard_violations > 5:
        penalty += (hard_violations - 5) * 1000

    # Soft: uneven daily load per group per week
    groups_seen: set[str] = set()
    for (week, day, grp) in wd_group:
        groups_seen.add(grp)

    group_day_count: dict[tuple[str, int, int], int] = defaultdict(int)
    for (week, day, grp), slots in wd_group.items():
        group_day_count[(grp, week, day)] = len(slots)

    for grp in groups_seen:
        for week in range(2):
            counts = [group_day_count.get((grp, week, d), 0) for d in range(5)]
            avg = sum(counts) / 5
            variance = sum((c - avg) ** 2 for c in counts)
            penalty += SOFT_UNEVEN * variance

    return 1.0 / (1.0 + penalty)


def _count_hard_violations(
    teacher_slot: dict,
    group_slot: dict,
    wd_teacher: dict,
    wd_group: dict,
    pfk_week_pairs: dict,
    pfk_stream_wds: dict,
) -> int:
    """Count total hard constraint violations for the extra-penalty threshold."""
    v = 0

    for c in teacher_slot.values():
        if c > 1:
            v += c - 1
    for c in group_slot.values():
        if c > 1:
            v += c - 1
    for slots in wd_teacher.values():
        if len(slots) > TEACHER_MAX_PAIRS_PER_DAY:
            v += len(slots) - TEACHER_MAX_PAIRS_PER_DAY
    for slots in wd_group.values():
        if len(slots) > STUDENT_MAX_PAIRS_PER_DAY:
            v += len(slots) - STUDENT_MAX_PAIRS_PER_DAY
    for (grp, week), cnt in pfk_week_pairs.items():
        if cnt > 1:
            v += cnt - 1

    for stream_id, locs in pfk_stream_wds.items():
        day_sids = {(d, sid) for (_, d, sid) in locs}
        weeks = {w for (w, _, _) in locs}
        sids = {sid for (_, _, sid) in locs}
        if len(locs) != 2 or len(day_sids) != 1 or weeks != {0, 1}:
            v += max(1, abs(len(locs) - 2) + max(0, len(day_sids) - 1))
        for sid in sids:
            if sid not in PFK_VALID_SIDS:
                v += 1

    return v


def compute_stats(chromosome: list[int], lessons: list[Lesson]) -> dict:
    """Return human-readable schedule quality stats."""
    teacher_slot: dict[tuple[str, int], int] = defaultdict(int)
    group_slot: dict[tuple[str, int, int], int] = defaultdict(int)
    wd_teacher: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    wd_group: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    group_total_pairs: dict[str, int] = defaultdict(int)
    pfk_week_pairs: dict[tuple[str, int], int] = defaultdict(int)
    pfk_stream_wds: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    late_pairs = 0

    for idx, slot in enumerate(chromosome):
        s = slot % TOTAL_SLOTS
        week, day, sid = slot_to_wds(s)
        lesson = lessons[idx]

        if lesson.teacher != PFK_TEACHER:
            teacher_slot[(lesson.teacher, s)] += 1
            wd_teacher[(week, day, lesson.teacher)].append(sid)

        for grp in _effective_groups(lesson):
            group_slot[(grp, s, lesson.subgroup)] += 1
            wd_group[(week, day, grp)].append(sid)
            group_total_pairs[grp] += 1
            if lesson.discipline_short == PFK_DISC:
                pfk_week_pairs[(grp, week)] += 1

        if lesson.discipline_short == PFK_DISC and lesson.stream_groups:
            pfk_stream_wds[lesson.group].append((week, day, sid))

        if sid in LATE_SLOTS:
            late_pairs += 1

    teacher_conflicts = sum(c - 1 for c in teacher_slot.values() if c > 1)
    group_conflicts = sum(c - 1 for c in group_slot.values() if c > 1)
    teacher_overload = sum(
        max(0, len(s) - TEACHER_MAX_PAIRS_PER_DAY) for s in wd_teacher.values()
    )
    group_overload_daily = sum(
        max(0, len(s) - STUDENT_MAX_PAIRS_PER_DAY) for s in wd_group.values()
    )
    group_overload_weekly = sum(
        max(0, cnt - STUDENT_MAX_PAIRS_PER_WEEK) for cnt in group_total_pairs.values()
    )
    pfk_overload = sum(max(0, cnt - 1) for cnt in pfk_week_pairs.values())

    # ПФК stream alignment violations
    pfk_stream_violations = 0
    pfk_bad_slot_violations = 0
    for stream_id, locs in pfk_stream_wds.items():
        day_sids = {(d, sid) for (_, d, sid) in locs}
        weeks = {w for (w, _, _) in locs}
        sids = {sid for (_, _, sid) in locs}
        if len(locs) != 2 or len(day_sids) != 1 or weeks != {0, 1}:
            pfk_stream_violations += max(1, abs(len(locs) - 2) + max(0, len(day_sids) - 1))
        pfk_bad_slot_violations += sum(1 for sid in sids if sid not in PFK_VALID_SIDS)

    hard_violations = (
        teacher_conflicts
        + group_conflicts
        + teacher_overload
        + group_overload_daily
        + group_overload_weekly
        + pfk_overload
        + pfk_stream_violations
        + pfk_bad_slot_violations
    )

    window_gaps = 0
    for slots in wd_group.values():
        if len(slots) > 1:
            ss = sorted(slots)
            for i in range(1, len(ss)):
                window_gaps += max(0, ss[i] - ss[i - 1] - 1)

    quality = max(0.0, 100.0 - window_gaps * 0.1)

    return {
        "hard_violations": hard_violations,
        "teacher_conflicts": teacher_conflicts,
        "group_conflicts": group_conflicts,
        "teacher_overload": teacher_overload,
        "group_overload": group_overload_daily + group_overload_weekly + pfk_overload,
        "pfk_stream_violations": pfk_stream_violations,
        "pfk_bad_slot_violations": pfk_bad_slot_violations,
        "window_gaps": window_gaps,
        "late_pairs": late_pairs,
        "quality_score": round(quality, 1),
    }
