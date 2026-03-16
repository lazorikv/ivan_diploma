from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from schedule.data_loader import Lesson
from schedule.ga.chromosome import slot_to_wds

RESULT_FILE = Path(__file__).resolve().parent.parent / "schedule_result.json"

DAY_NAMES = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница"]
SLOT_TIMES = [
    "08:00–09:30",
    "09:45–11:15",
    "11:30–13:00",
    "13:45–15:15",
    "15:30–17:00",
    "17:10–18:40",
]
LESSON_TYPE_LABELS = {"lec": "Лекция", "prac": "Практика", "lab": "Лаб. работа"}
WEEK_LABELS = {1: "Верхняя (нечётная)", 2: "Нижняя (чётная)"}


def chromosome_to_entries(
    chromosome: list[int],
    lessons: list[Lesson],
) -> list[dict]:
    entries = []
    for idx, slot in enumerate(chromosome):
        week, day, slot_in_day = slot_to_wds(slot)
        lesson = lessons[idx]

        # Stream lesson: expand to one entry per group in the stream
        if lesson.stream_groups:
            groups_to_emit = lesson.stream_groups
        else:
            groups_to_emit = [lesson.group]

        for grp in groups_to_emit:
            entries.append({
                "week": week + 1,
                "day": day + 1,
                "slot": slot_in_day + 1,
                "day_name": DAY_NAMES[day],
                "slot_time": SLOT_TIMES[slot_in_day],
                "group": grp,
                "discipline_short": lesson.discipline_short,
                "discipline_full": lesson.discipline_full,
                "lesson_type": lesson.lesson_type,
                "lesson_type_label": LESSON_TYPE_LABELS.get(
                    lesson.lesson_type, lesson.lesson_type
                ),
                "teacher": lesson.teacher,
                "subgroup": lesson.subgroup,
                "is_stream": len(lesson.stream_groups) > 1,
                "stream_groups": lesson.stream_groups,
            })
    return entries


def save_result(
    chromosome: list[int],
    lessons: list[Lesson],
    fitness: float,
    stats: dict,
    generations: int,
    population_size: int,
    fitness_history: list[float],
) -> None:
    entries = chromosome_to_entries(chromosome, lessons)
    data: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "fitness": fitness,
        "quality_score": stats.get("quality_score", 0),
        "hard_violations": stats.get("hard_violations", 0),
        "window_gaps": stats.get("window_gaps", 0),
        "generations": generations,
        "population_size": population_size,
        "fitness_history": fitness_history,
        "entries": entries,
    }
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_result() -> dict | None:
    if not RESULT_FILE.exists():
        return None
    try:
        return json.loads(RESULT_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def group_by_group(entries: list[dict]) -> dict[str, dict]:
    """Organise entries into {group: {week: {day: {slot: entry}}}}."""
    result: dict[str, Any] = {}
    for e in entries:
        g = e["group"]
        w = e["week"]
        d = e["day"]
        s = e["slot"]
        result.setdefault(g, {}).setdefault(w, {}).setdefault(d, {})[s] = e
    return result


def group_by_teacher(entries: list[dict]) -> dict[str, dict]:
    """Organise entries into {teacher: {week: {day: {slot: entry}}}}."""
    result: dict[str, Any] = {}
    for e in entries:
        t = e["teacher"]
        if not t or t == "?":
            continue
        w = e["week"]
        d = e["day"]
        s = e["slot"]
        result.setdefault(t, {}).setdefault(w, {}).setdefault(d, {})[s] = e
    return result
