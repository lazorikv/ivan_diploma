from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from schedule.data_loader import (
    PFK_TEACHER,
    UPLOADS_DIR,
    _UPLOADED_GROUPS_XLSX,
    _UPLOADED_TEACHERS_XLSX,
    expand_lessons,
    get_upload_status,
    load_data,
)
from schedule.ga.algorithm import run_ga
from schedule.ga.fitness import compute_stats
from schedule.result_store import (
    DAY_NAMES,
    SLOT_TIMES,
    group_by_group,
    group_by_teacher,
    load_result,
    save_result,
    split_teachers,
)

# Римские номера пар (в модели 6 пар в день)
SLOT_ROMAN = ("I", "II", "III", "IV", "V", "VI")

def _leading_course_digit(code: str) -> int | None:
    """Первая цифра в коде группы = курс (1…9). Ноль и не-цифра — нет."""
    if not code or not code[0].isdigit():
        return None
    d = int(code[0])
    return None if d == 0 else d


def _sort_entity_codes(codes: list[str]) -> list[str]:
    """Сначала группы с курсом по цифре, затем М…, затем прочие."""

    def key(c: str) -> tuple:
        d = _leading_course_digit(c)
        if d is not None:
            return (0, d, len(c), c.casefold())
        if c.startswith("М"):
            return (1, 0, len(c), c.casefold())
        return (2, 0, len(c), c.casefold())

    return sorted(codes, key=key)


def _parse_day_param(raw: str | None) -> int | None:
    if not raw:
        return None
    s = str(raw).strip()
    if s in {"1", "2", "3", "4", "5"}:
        return int(s)
    return None


def _weekday_or_monday(raw: str | None) -> int:
    """День недели 1–5 (Пн–Пт); без параметра или при мусоре — понедельник."""
    d = _parse_day_param(raw)
    return d if d is not None else 1


def _groups_for_course(year: int, entities: list[str]) -> list[str]:
    """Все группы, код которых начинается с цифры курса (1…, 2…, 3…)."""
    pref = str(year)
    return _sort_entity_codes([c for c in entities if c.startswith(pref)])


def _mag1_groups_sorted(entities: list[str]) -> list[str]:
    return _sort_entity_codes([c for c in entities if c.startswith("М1")])


def _potok_stream_title(year: int, groups: list[str]) -> str:
    """
    Заголовок потока: «N курс — поток NБ–NИ» по двухсимвольным кодам N+буква;
    иначе диапазон полных кодов или один код.
    """
    ys = str(year)
    two_letter = [g for g in groups if len(g) == 2 and g[0] == ys and g[1].isalpha()]
    if two_letter:
        letters = sorted([g[1] for g in two_letter], key=lambda ch: ch.casefold())
        lo, hi = letters[0], letters[-1]
        return f"{year} курс — поток {ys}{lo}–{ys}{hi}"
    sg = _sort_entity_codes(groups)
    if len(sg) >= 2:
        return f"{year} курс — поток {sg[0]}–{sg[-1]}"
    if len(sg) == 1:
        return f"{year} курс — поток {sg[0]}"
    return f"{year} курс"


POTOK_MAG1_LABEL = "Магистратура 1 курс (М1…)"


def _format_generated_at(value: str | None) -> str:
    if not value:
        return ""
    s = str(value).strip()
    if "T" in s:
        date_part, time_part = s.split("T", 1)
        time_part = time_part.split(".", 1)[0]
        ymd = date_part.split("-", 2)
        if len(ymd) == 3:
            y, m, d = ymd
            return f"{d}.{m}.{y} {time_part}"
        return f"{date_part} {time_part}"
    return s


def index(request):
    result = load_result()
    status = get_upload_status()
    files_missing = (
        not Path(status["groups_path"]).exists()
        or not Path(status["teachers_path"]).exists()
    )
    context = {
        "result": result,
        "generated_at_human": _format_generated_at(result.get("generated_at") if result else None),
        "default_population": 20,
        "default_generations": 100,
        "files_missing": files_missing,
    }
    return render(request, "schedule/index.html", context)


def upload_files(request):
    if request.method == "POST":
        UPLOADS_DIR.mkdir(exist_ok=True)
        uploaded = []
        errors = []

        groups_file = request.FILES.get("groups_xlsx")
        teachers_file = request.FILES.get("teachers_xlsx")

        if not groups_file and not teachers_file:
            messages.error(request, "Выберите хотя бы один файл для загрузки.")
            return redirect("upload_files")

        for file, dest, label in [
            (groups_file, _UPLOADED_GROUPS_XLSX, "Группы"),
            (teachers_file, _UPLOADED_TEACHERS_XLSX, "Преподаватели"),
        ]:
            if file is None:
                continue
            if not file.name.endswith((".xlsx", ".xls")):
                errors.append(f"{label}: файл должен быть в формате .xlsx")
                continue
            with open(dest, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)
            uploaded.append(label)

        if errors:
            for e in errors:
                messages.error(request, e)
        if uploaded:
            messages.success(request, f"Загружено: {', '.join(uploaded)}.")

        return redirect("upload_files")

    status = get_upload_status()
    return render(request, "schedule/upload.html", {"status": status})


def delete_upload(request, file_type):
    if request.method == "POST":
        target = _UPLOADED_GROUPS_XLSX if file_type == "groups" else _UPLOADED_TEACHERS_XLSX
        if target.exists():
            target.unlink()
            messages.success(request, "Файл удалён, используется файл по умолчанию.")
        else:
            messages.info(request, "Файл не найден.")
    return redirect("upload_files")


@require_POST
def generate(request):
    try:
        population_size = int(request.POST.get("population_size", 20))
        generations = int(request.POST.get("generations", 100))
        population_size = max(5, min(population_size, 100))
        generations = max(10, min(generations, 500))
    except (ValueError, TypeError):
        population_size = 20
        generations = 100

    try:
        data = load_data()
    except FileNotFoundError as e:
        messages.error(
            request,
            f"Файл данных не найден: {e.filename}. "
            "Загрузите xlsx-файлы на странице «Загрузка данных».",
        )
        return redirect("upload_files")
    except Exception as e:
        messages.error(request, f"Ошибка при загрузке данных: {e}")
        return redirect("upload_files")

    lessons = expand_lessons(data.lessons)

    chromosome, fitness, history = run_ga(
        lessons,
        population_size=population_size,
        generations=generations,
    )

    stats = compute_stats(chromosome, lessons)
    save_result(
        chromosome=chromosome,
        lessons=lessons,
        fitness=fitness,
        stats=stats,
        generations=generations,
        population_size=population_size,
        fitness_history=history,
    )

    return redirect("index")


def _build_violation_map(entries: list[dict]) -> dict[tuple, list[str]]:
    """
    Compute hard-constraint violations for every entry.

    Returns a mapping (week, day, slot, group) → [violation_label, ...].
    Only real conflicts are flagged (stream lectures shared by many groups
    are NOT counted as teacher conflicts).
    """
    vmap: dict[tuple, list[str]] = defaultdict(list)

    # --- Teacher conflicts ---
    # Group by (teacher, week, day, slot); skip ПФК placeholder teacher.
    tslot: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        t = e.get("teacher", "")
        if t and t != PFK_TEACHER:
            tslot[(t, e["week"], e["day"], e["slot"])].append(e)

    for (teacher, week, day, slot), es in tslot.items():
        # Identify distinct "lesson" identities by (disc, frozenset(stream_groups))
        identities = {
            (e["discipline_short"], frozenset(e.get("stream_groups") or [e["group"]]))
            for e in es
        }
        if len(identities) > 1:
            for e in es:
                key = (e["week"], e["day"], e["slot"], e["group"])
                if "Конфликт преп." not in vmap[key]:
                    vmap[key].append("Конфликт преп.")

    # --- Group conflicts ---
    gslot: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        sg = e.get("subgroup", 0)
        gslot[(e["group"], e["week"], e["day"], e["slot"], sg)].append(e)

    for (group, week, day, slot, sg), es in gslot.items():
        if len(es) > 1:
            for e in es:
                key = (e["week"], e["day"], e["slot"], e["group"])
                if "Конфликт группы" not in vmap[key]:
                    vmap[key].append("Конфликт группы")

    # --- >5 pairs per day per group ---
    gday: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        gday[(e["group"], e["week"], e["day"])].append(e)

    for (group, week, day), es in gday.items():
        if len(es) > 5:
            for e in es:
                key = (e["week"], e["day"], e["slot"], e["group"])
                if ">5 пар/день" not in vmap[key]:
                    vmap[key].append(">5 пар/день")

    # --- ПФК not in pairs 1-3 ---
    for e in entries:
        if e.get("discipline_short") == "ПФК" and e["slot"] > 3:
            key = (e["week"], e["day"], e["slot"], e["group"])
            if "ПФК не 1-3 пара" not in vmap[key]:
                vmap[key].append("ПФК не 1-3 пара")

    return dict(vmap)


def _cell_equal(c1: dict | None, c2: dict | None) -> bool:
    """True if two schedule entries represent the same lesson."""
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        return False
    return (
        c1.get("discipline_short") == c2.get("discipline_short")
        and c1.get("teacher") == c2.get("teacher")
        and c1.get("subgroup") == c2.get("subgroup")
    )


def schedule_view(request):
    result = load_result()
    if result is None:
        return redirect("index")

    entries = result["entries"]
    view_type = request.GET.get("view", "group")

    # Both weeks at once — no toggle
    all_entries = entries
    if view_type == "teacher":
        by_entity_w1 = group_by_teacher([e for e in all_entries if e["week"] == 1])
        by_entity_w2 = group_by_teacher([e for e in all_entries if e["week"] == 2])
        entity_label = "Преподаватель"
    else:
        by_entity_w1 = group_by_group([e for e in all_entries if e["week"] == 1])
        by_entity_w2 = group_by_group([e for e in all_entries if e["week"] == 2])
        entity_label = "Группа"

    entities = _sort_entity_codes(
        sorted(set(by_entity_w1.keys()) | set(by_entity_w2.keys()))
    )
    selected_entity = request.GET.get("entity", entities[0] if entities else "")
    if selected_entity not in set(by_entity_w1.keys()) | set(by_entity_w2.keys()):
        selected_entity = entities[0] if entities else ""

    selected_day = _weekday_or_monday(request.GET.get("day"))

    # Pre-compute violation map for ALL entries (used to highlight bad cells)
    vmap = _build_violation_map(entries)

    def _violations_for(entry: dict | None) -> list[str]:
        if entry is None:
            return []
        key = (entry["week"], entry["day"], entry["slot"], entry["group"])
        return vmap.get(key, [])

    # --- Поток = курс: все группы, код которых начинается с цифры курса ---
    course_years = sorted(
        {y for c in entities if (y := _leading_course_digit(c)) is not None}
    )
    mag1_groups = _mag1_groups_sorted(entities)

    def _resolve_group_potok() -> tuple[str | int, list[str], str]:
        raw = (request.GET.get("potok") or "").strip().lower()
        legacy = (request.GET.get("entity") or "").strip()

        if raw in {"m1", "mag", "маг"} and mag1_groups:
            return "m1", mag1_groups, POTOK_MAG1_LABEL

        if raw.isdigit():
            y = int(raw)
            if y in course_years:
                gs = _groups_for_course(y, entities)
                if gs:
                    return y, gs, _potok_stream_title(y, gs)

        if legacy.startswith("М1") and mag1_groups:
            return "m1", mag1_groups, POTOK_MAG1_LABEL

        y_legacy = _leading_course_digit(legacy)
        if y_legacy is not None and y_legacy in course_years:
            gs = _groups_for_course(y_legacy, entities)
            if gs:
                return y_legacy, gs, _potok_stream_title(y_legacy, gs)

        if 1 in course_years:
            y = 1
            gs = _groups_for_course(y, entities)
            if gs:
                return y, gs, _potok_stream_title(y, gs)
        if course_years:
            y = course_years[0]
            gs = _groups_for_course(y, entities)
            return y, gs, _potok_stream_title(y, gs)
        if mag1_groups:
            return "m1", mag1_groups, POTOK_MAG1_LABEL
        e0 = entities[0] if entities else ""
        return 0, ([e0] if e0 else []), (f"Группа {e0}" if e0 else "Расписание")

    potok_options: list[dict[str, str]] = []
    for y in course_years:
        gs_opt = _groups_for_course(y, entities)
        potok_options.append({
            "key": str(y),
            "label": _potok_stream_title(y, gs_opt),
        })
    if mag1_groups:
        potok_options.append({"key": "m1", "label": POTOK_MAG1_LABEL})

    potok_key, potok_groups, potok_label = _resolve_group_potok()
    if potok_key == "m1":
        potok_key_str = "m1"
    elif potok_key == 0:
        potok_key_str = "0"
    else:
        potok_key_str = str(int(potok_key))

    grid_rows: list[dict] = []

    if view_type == "group":
        column_days_groups = [(selected_day, g) for g in potok_groups]

        for slot_idx in range(1, 7):
            cells = []
            for day_idx, grp in column_days_groups:
                w1g = by_entity_w1.get(grp, {}).get(1, {})
                w2g = by_entity_w2.get(grp, {}).get(2, {})
                c1 = w1g.get(day_idx, {}).get(slot_idx)
                c2 = w2g.get(day_idx, {}).get(slot_idx)
                cells.append({
                    "w1": c1,
                    "w2": c2,
                    "merged": _cell_equal(c1, c2),
                    "v1": _violations_for(c1),
                    "v2": _violations_for(c2),
                })
            grid_rows.append({
                "slot": slot_idx,
                "slot_roman": SLOT_ROMAN[slot_idx - 1],
                "slot_time": SLOT_TIMES[slot_idx - 1],
                "cells": cells,
            })
    else:
        w1_data = by_entity_w1.get(selected_entity, {}).get(1, {})
        w2_data = by_entity_w2.get(selected_entity, {}).get(2, {})
        for slot_idx in range(1, 7):
            cells = []
            for day_idx in [selected_day]:
                c1 = w1_data.get(day_idx, {}).get(slot_idx)
                c2 = w2_data.get(day_idx, {}).get(slot_idx)
                cells.append({
                    "w1": c1,
                    "w2": c2,
                    "merged": _cell_equal(c1, c2),
                    "v1": _violations_for(c1),
                    "v2": _violations_for(c2),
                })
            grid_rows.append({
                "slot": slot_idx,
                "slot_roman": SLOT_ROMAN[slot_idx - 1],
                "slot_time": SLOT_TIMES[slot_idx - 1],
                "cells": cells,
            })

    use_roman_slots = view_type == "group"
    stream_mode = view_type == "group"
    print_sheet = view_type == "group"

    context = {
        "result": result,
        "generated_at_human": _format_generated_at(result.get("generated_at")),
        "view_type": view_type,
        "selected_entity": selected_entity,
        "entities": entities,
        "entity_label": entity_label,
        "grid_rows": grid_rows,
        "day_names": DAY_NAMES,
        "selected_day": selected_day,
        "selected_day_name": DAY_NAMES[selected_day - 1],
        "potok_key_str": potok_key_str,
        "potok_label": potok_label,
        "potok_groups": potok_groups,
        "potok_options": potok_options,
        "use_roman_slots": use_roman_slots,
        "stream_mode": stream_mode,
        "print_sheet": print_sheet,
    }
    return render(request, "schedule/schedule.html", context)


def data_view(request):
    try:
        data = load_data()
    except FileNotFoundError as e:
        messages.error(
            request,
            f"Файл данных не найден: {e.filename}. "
            "Загрузите xlsx-файлы на странице «Загрузка данных».",
        )
        return redirect("upload_files")
    except Exception as e:
        messages.error(request, f"Ошибка при загрузке данных: {e}")
        return redirect("upload_files")

    lessons = data.lessons

    # Group size lookup
    group_sizes = {g.code: g.size for g in data.groups}

    # Organise by group
    groups: dict[str, list] = defaultdict(list)
    for lesson in lessons:
        groups[lesson.group].append(lesson)

    # Unique teachers
    teachers: dict[str, set] = defaultdict(set)
    for lesson in lessons:
        for t in split_teachers(lesson.teacher):
            teachers[t].add(lesson.discipline_full or lesson.discipline_short)

    # Build group summary: code, size, splits_labs
    GROUP_SPLIT_THRESHOLD = 18
    group_summary = []
    for g in sorted(data.groups, key=lambda x: x.code):
        size = g.size
        splits = size is not None and size >= GROUP_SPLIT_THRESHOLD
        lesson_count = len(groups.get(g.code, []))
        group_summary.append({
            "code": g.code,
            "size": size,
            "splits_labs": splits,
            "lesson_count": lesson_count,
        })

    context = {
        "groups": dict(sorted(groups.items())),
        "group_sizes": group_sizes,
        "group_summary": group_summary,
        "teachers": {t: sorted(d) for t, d in sorted(teachers.items())},
        "disciplines": dict(sorted(data.disciplines.items())),
        "total_groups": len(data.groups),
        "total_teachers": len(teachers),
        "total_disciplines": len(data.disciplines),
        "split_threshold": GROUP_SPLIT_THRESHOLD,
    }
    return render(request, "schedule/data.html", context)


def export_view(request):
    result = load_result()
    if result is None:
        return HttpResponse("Расписание не сгенерировано", status=404)
    response = HttpResponse(
        json.dumps(result, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
    )
    response["Content-Disposition"] = 'attachment; filename="schedule_result.json"'
    return response
