from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
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
    WEEK_LABELS,
    group_by_group,
    group_by_teacher,
    load_result,
    save_result,
    split_teachers,
    write_schedule_result_document,
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


def _parse_schedule_range(raw: str | None, view_type: str) -> str:
    """Для просмотра по группам: week (по умолчанию) или day. Для преподавателя — только day."""
    if view_type != "group":
        return "day"
    s = (raw or "").strip().lower()
    return "day" if s == "day" else "week"


def _build_stream_grid_rows(
    day_idx: int,
    potok_groups: list[str],
    by_entity_w1: dict,
    by_entity_w2: dict,
    violations_for,
) -> list[dict]:
    """Одна таблица потока: строки = пары, колонки = группы, день = day_idx."""
    column_days_groups = [(day_idx, g) for g in potok_groups]
    grid_rows: list[dict] = []
    for slot_idx in range(1, 7):
        cells = []
        for d_i, grp in column_days_groups:
            w1g = by_entity_w1.get(grp, {}).get(1, {})
            w2g = by_entity_w2.get(grp, {}).get(2, {})
            c1 = w1g.get(d_i, {}).get(slot_idx)
            c2 = w2g.get(d_i, {}).get(slot_idx)
            cells.append({
                "w1": c1,
                "w2": c2,
                "merged": _cell_equal(c1, c2),
                "v1": violations_for(c1),
                "v2": violations_for(c2),
            })
        grid_rows.append({
            "slot": slot_idx,
            "slot_roman": SLOT_ROMAN[slot_idx - 1],
            "slot_time": SLOT_TIMES[slot_idx - 1],
            "cells": cells,
        })
    return grid_rows


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
        and c1.get("lesson_type") == c2.get("lesson_type")
    )


def _teacher_slot_entries(week_data: dict, day_idx: int, slot_idx: int) -> list[dict]:
    slot_payload = week_data.get(day_idx, {}).get(slot_idx)
    if slot_payload is None:
        return []
    if isinstance(slot_payload, list):
        return slot_payload
    # Backward compatibility with old shape {slot: entry}
    if isinstance(slot_payload, dict):
        return [slot_payload]
    return []


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

    entity_keys = set(by_entity_w1.keys()) | set(by_entity_w2.keys())
    if view_type == "teacher":
        entities = sorted(entity_keys, key=lambda s: s.casefold())
    else:
        entities = _sort_entity_codes(sorted(entity_keys))
    selected_entity = request.GET.get("entity", entities[0] if entities else "")
    if selected_entity not in set(by_entity_w1.keys()) | set(by_entity_w2.keys()):
        selected_entity = entities[0] if entities else ""

    selected_day = _weekday_or_monday(request.GET.get("day"))
    schedule_range = _parse_schedule_range(request.GET.get("range"), view_type)

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
    day_grids: list[dict] = []

    if view_type == "group":
        if schedule_range == "week":
            for d in range(1, 6):
                day_grids.append({
                    "day": d,
                    "day_name": DAY_NAMES[d - 1],
                    "anchor_id": f"stream-day-{d}",
                    "grid_rows": _build_stream_grid_rows(
                        d, potok_groups, by_entity_w1, by_entity_w2, _violations_for
                    ),
                })
        else:
            grid_rows = _build_stream_grid_rows(
                selected_day,
                potok_groups,
                by_entity_w1,
                by_entity_w2,
                _violations_for,
            )
    else:
        w1_data = by_entity_w1.get(selected_entity, {}).get(1, {})
        w2_data = by_entity_w2.get(selected_entity, {}).get(2, {})
        for slot_idx in range(1, 7):
            cells = []
            for day_idx in [selected_day]:
                c1_items = _teacher_slot_entries(w1_data, day_idx, slot_idx)
                c2_items = _teacher_slot_entries(w2_data, day_idx, slot_idx)
                merged = (
                    len(c1_items) == 1
                    and len(c2_items) == 1
                    and _cell_equal(c1_items[0], c2_items[0])
                )
                cells.append({
                    "w1_items": [{"entry": e, "viol": _violations_for(e)} for e in c1_items],
                    "w2_items": [{"entry": e, "viol": _violations_for(e)} for e in c2_items],
                    "merged": merged,
                    "merged_entry": c1_items[0] if merged else None,
                    "merged_viol": _violations_for(c1_items[0]) if merged else [],
                })
            grid_rows.append({
                "slot": slot_idx,
                "slot_roman": SLOT_ROMAN[slot_idx - 1],
                "slot_time": SLOT_TIMES[slot_idx - 1],
                "cells": cells,
            })

    use_roman_slots = view_type == "group"
    stream_mode = view_type == "group"
    print_sheet = view_type == "group" and schedule_range == "day"
    week_stack = view_type == "group" and schedule_range == "week"
    context = {
        "result": result,
        "generated_at_human": _format_generated_at(result.get("generated_at")),
        "view_type": view_type,
        "selected_entity": selected_entity,
        "entities": entities,
        "entity_label": entity_label,
        "grid_rows": grid_rows,
        "day_grids": day_grids,
        "schedule_range": schedule_range,
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
        "week_upper_label": WEEK_LABELS.get(1, "Верхняя неделя"),
        "week_lower_label": WEEK_LABELS.get(2, "Нижняя неделя"),
        "week_stack": week_stack,
    }
    return render(request, "schedule/schedule.html", context)




def _normalize_dest_week_post(raw: str | None) -> int | None:
    """На целевом слоте: None — сохранять week у каждой строки; 1/2 — задать одну неделю."""
    s = (raw or "").strip().lower()
    if s in ("keep", "", "same"):
        return None
    if s in ("both", "оба"):
        return 0
    if s in ("upper", "1", "в"):
        return 1
    if s in ("lower", "2", "н"):
        return 2
    return -1


def _resolve_restrict_weeks_for_move_post(request) -> tuple[frozenset[int] | None, bool]:
    """
    Какую строку(и) взять с исходного слота: объединённая ячейка — всегда обе;
    половина В/Н — из hidden split_source_week.
    """
    cell_is_merged = request.POST.get("cell_is_merged") == "1"
    if cell_is_merged:
        return None, True

    sw = request.POST.get("split_source_week", "").strip()
    if sw == "1":
        return frozenset({1}), True
    if sw == "2":
        return frozenset({2}), True
    return frozenset(), False


def _apply_slot_coords(e: dict, *, day: int, slot: int, week_val: int) -> None:
    e["day"] = day
    e["slot"] = slot
    e["week"] = week_val
    e["day_name"] = DAY_NAMES[day - 1]
    e["slot_time"] = SLOT_TIMES[slot - 1]


def _find_blocker_same_slot(
    entries: list[dict],
    mover_set: set[int],
    *,
    group: str,
    subgroup: int,
    week: int,
    day: int,
    slot: int,
    skip_idxs: set[int] | None = None,
) -> int | None:
    skip = skip_idxs or set()
    for j, o in enumerate(entries):
        if j in mover_set or j in skip:
            continue
        if o.get("group") != group:
            continue
        if int(o.get("subgroup") or 0) != subgroup:
            continue
        if int(o.get("week") or 0) != week:
            continue
        if o.get("day") != day or o.get("slot") != slot:
            continue
        return j
    return None


def _perform_place_or_swap(
    entries: list[dict],
    mover_indices: list[int],
    *,
    new_day: int,
    new_slot: int,
    dest_week_override: int | None,
) -> int:
    """
    Ставим каждое перенесённое занятие в (new_day, new_slot [, week]).
    Если слот уже занят — меняется местами с тем занятием. Без ошибок блокировки.
    Возвращает число выполненных обменов (пар).
    """
    mover_set = set(mover_indices)
    swaps = 0
    skip_blockers: set[int] = set()

    movers_sorted = sorted(
        mover_indices,
        key=lambda ix: (int(entries[ix].get("week") or 0), ix),
    )

    for mi in movers_sorted:
        row = entries[mi]
        g = str(row.get("group") or "")
        sg = int(row.get("subgroup") or 0)
        old_d = int(row.get("day") or 1)
        old_s = int(row.get("slot") or 1)
        ow = int(row.get("week") or 1)

        tgt_w = int(dest_week_override) if dest_week_override in (1, 2) else ow

        bi = _find_blocker_same_slot(
            entries,
            mover_set,
            group=g,
            subgroup=sg,
            week=tgt_w,
            day=new_day,
            slot=new_slot,
            skip_idxs=skip_blockers,
        )

        if bi is not None:
            blocker = dict(entries[bi])
            swaps += 1
            skip_blockers.add(bi)
            entries[bi] = blocker
            _apply_slot_coords(entries[bi], day=old_d, slot=old_s, week_val=ow)

        mover_row = dict(row)
        entries[mi] = mover_row
        _apply_slot_coords(entries[mi], day=new_day, slot=new_slot, week_val=tgt_w)

    return swaps


def _entry_identity_signature(e: dict) -> tuple:
    return (
        str(e.get("discipline_short") or ""),
        str(e.get("teacher") or ""),
        int(e.get("subgroup") or 0),
        str(e.get("lesson_type") or ""),
    )


def _ensure_movers_cover_both_weeks(
    entries: list[dict],
    mover_indices: list[int],
    *,
    day: int,
    slot: int,
) -> tuple[int, int]:
    """
    Для выбранных строк на (day, slot) пытаемся гарантировать В+Н.
    Возвращает (сколько копий добавлено, сколько раз блокировало чужое занятие).
    """
    added = 0
    blocked = 0
    produced_keys: set[tuple] = set()

    for mi in mover_indices:
        if mi < 0 or mi >= len(entries):
            continue
        row = entries[mi]
        g = str(row.get("group") or "")
        sg = int(row.get("subgroup") or 0)
        w = int(row.get("week") or 0)
        if w not in (1, 2):
            continue
        other_w = 2 if w == 1 else 1
        sig = _entry_identity_signature(row)
        prod_key = (g, sg, sig, day, slot, other_w)
        if prod_key in produced_keys:
            continue

        same_slot_other_week = [
            e for e in entries
            if str(e.get("group") or "") == g
            and int(e.get("subgroup") or 0) == sg
            and int(e.get("week") or 0) == other_w
            and int(e.get("day") or 0) == day
            and int(e.get("slot") or 0) == slot
        ]
        if not same_slot_other_week:
            clone = dict(row)
            _apply_slot_coords(clone, day=day, slot=slot, week_val=other_w)
            entries.append(clone)
            produced_keys.add(prod_key)
            added += 1
            continue

        if any(_entry_identity_signature(e) == sig for e in same_slot_other_week):
            produced_keys.add(prod_key)
            continue

        blocked += 1

    return added, blocked


def _dedupe_exact_slot_lesson_rows(entries: list[dict]) -> int:
    """
    Убирает точные дубли одной и той же пары в одной и той же ячейке.
    Нужен после ручного редактирования, чтобы не ловить ложный "Конфликт группы".
    """
    seen: set[tuple] = set()
    unique_rows: list[dict] = []
    removed = 0
    for e in entries:
        key = (
            str(e.get("group") or ""),
            int(e.get("subgroup") or 0),
            int(e.get("week") or 0),
            int(e.get("day") or 0),
            int(e.get("slot") or 0),
            str(e.get("discipline_short") or ""),
            str(e.get("teacher") or ""),
            str(e.get("lesson_type") or ""),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique_rows.append(e)

    if removed:
        entries[:] = unique_rows
    return removed


def _norm_compact(s: str) -> str:
    return " ".join((s or "").split()).strip().casefold()


def _resolve_move_indices(
    entries: list[dict],
    *,
    restrict_weeks: frozenset[int] | None,
    group: str,
    old_day: int,
    old_slot: int,
    discipline_short: str,
    teacher: str,
    lesson_type: str,
    subgroup: int,
) -> list[int]:
    """
    Ручной перенос: в первую очередь слот (группа, день, пара, подгруппа, неделя).
    Дисциплина/преподаватель из формы — подсказка для уточнения, не жёсткое равенство.
    """
    candidates: list[int] = []
    for i, e in enumerate(entries):
        if e.get("group") != group:
            continue
        if e.get("day") != old_day or e.get("slot") != old_slot:
            continue
        if (e.get("subgroup") or 0) != subgroup:
            continue
        ew = e.get("week")
        if ew not in (1, 2):
            continue
        if restrict_weeks is not None and ew not in restrict_weeks:
            continue
        candidates.append(i)

    if not candidates:
        return []

    narrowed = candidates
    nd = _norm_compact(discipline_short)
    if nd:
        by_disc = [
            i
            for i in narrowed
            if _norm_compact(str(entries[i].get("discipline_short") or "")) == nd
        ]
        if by_disc:
            narrowed = by_disc

    nt = _norm_compact(teacher)
    if nt and len(narrowed) > 1:
        by_t = [
            i
            for i in narrowed
            if _norm_compact(str(entries[i].get("teacher") or "")) == nt
        ]
        if by_t:
            narrowed = by_t

    nlt = _norm_compact(lesson_type)
    if nlt and len(narrowed) > 1:
        by_lt = [
            i
            for i in narrowed
            if _norm_compact(str(entries[i].get("lesson_type") or "")) == nlt
        ]
        if by_lt:
            narrowed = by_lt

    return narrowed


def _schedule_redirect_path(raw_next: str) -> str:
    if raw_next.startswith("/") and not raw_next.startswith("//"):
        return raw_next
    return reverse("schedule")


@require_POST
def schedule_move(request):
    next_path = _schedule_redirect_path((request.POST.get("next") or "").strip())

    result = load_result()
    if result is None:
        messages.error(request, "Нет сохранённого расписания.")
        return redirect("index")

    restrict_weeks, scope_ok = _resolve_restrict_weeks_for_move_post(request)
    cell_is_merged = request.POST.get("cell_is_merged") == "1"

    dest_week_override = _normalize_dest_week_post(
        request.POST.get("target_week") or request.POST.get("dest_week")
    )
    dest_invalid = dest_week_override not in (None, 0, 1, 2)

    # Если источник был merged (В/Н), а цель — конкретная неделя,
    # переносим только эту неделю, вторая остаётся на исходном слоте.
    if cell_is_merged and dest_week_override in (1, 2):
        restrict_weeks = frozenset({int(dest_week_override)})

    group = (request.POST.get("group") or "").strip()
    discipline_short = (request.POST.get("discipline_short") or "").strip()
    teacher = (request.POST.get("teacher") or "").strip()
    lesson_type = (request.POST.get("lesson_type") or "").strip()

    try:
        subgroup = int(request.POST.get("subgroup") or 0)
    except (ValueError, TypeError):
        subgroup = 0

    try:
        old_day = int(request.POST.get("old_day") or 0)
        old_slot = int(request.POST.get("old_slot") or 0)
        new_day = int(request.POST.get("new_day") or 0)
        new_slot = int(request.POST.get("new_slot") or 0)
    except (ValueError, TypeError):
        messages.error(request, "Некорректные номера дня или пары.")
        return redirect(next_path)

    if not scope_ok:
        messages.error(
            request,
            "Не определена исходная строка (В/Н) — для половины ячейки обновите страницу.",
        )
        return redirect(next_path)

    if dest_invalid:
        messages.error(request, "Некорректный выбор целевой учебной недели.")
        return redirect(next_path)

    if not group:
        messages.error(request, "Не указана группа.")
        return redirect(next_path)

    if not (
        1 <= old_day <= 5
        and 1 <= old_slot <= 6
        and 1 <= new_day <= 5
        and 1 <= new_slot <= 6
    ):
        messages.error(request, "День или пара вне допустимого диапазона.")
        return redirect(next_path)

    entries = list(result["entries"])

    indices = _resolve_move_indices(
        entries,
        restrict_weeks=restrict_weeks,
        group=group,
        old_day=old_day,
        old_slot=old_slot,
        discipline_short=discipline_short,
        teacher=teacher,
        lesson_type=lesson_type,
        subgroup=subgroup,
    )
    if not indices and restrict_weeks is not None:
        indices = _resolve_move_indices(
            entries,
            restrict_weeks=None,
            group=group,
            old_day=old_day,
            old_slot=old_slot,
            discipline_short=discipline_short,
            teacher=teacher,
            lesson_type=lesson_type,
            subgroup=subgroup,
        )

    if not indices:
        messages.warning(request, "На исходном слоте не найдено занятие — файл не меняли.")
        return redirect(next_path)

    primary_disc = (
        discipline_short.strip()
        or str(entries[indices[0]].get("discipline_short") or "").strip()
    )

    expand_to_both = dest_week_override == 0
    dest_ov = dest_week_override if dest_week_override in (1, 2) else None

    swaps = _perform_place_or_swap(
        entries,
        indices,
        new_day=new_day,
        new_slot=new_slot,
        dest_week_override=dest_ov,
    )
    both_added = 0
    both_blocked = 0
    if expand_to_both:
        both_added, both_blocked = _ensure_movers_cover_both_weeks(
            entries,
            indices,
            day=new_day,
            slot=new_slot,
        )

    dedup_removed = _dedupe_exact_slot_lesson_rows(entries)

    result["entries"] = entries
    write_schedule_result_document(result)

    label = primary_disc[:40] + ("…" if len(primary_disc) > 40 else "")
    wl1 = WEEK_LABELS.get(1, "верхняя неделя")
    wl2 = WEEK_LABELS.get(2, "нижняя неделя")

    dest_note = ""
    if dest_week_override == 1:
        dest_note = f" Слот: {wl1}."
    elif dest_week_override == 2:
        dest_note = f" Слот: {wl2}."
    elif dest_week_override == 0:
        dest_note = f" Слот: обе недели ({wl1}/{wl2})."
    else:
        dest_note = " У строк сохранены прежние В/Н (или пара строк)."

    swap_note = f" Обменено местами пар: {swaps}." if swaps else ""
    both_note = ""
    if expand_to_both:
        both_note = f" Добавлено копий на вторую неделю: {both_added}."
        if both_blocked:
            both_note += " Для части строк занято другой парой — оставлены без дублирования."
    dedup_note = f" Удалены дубли строк: {dedup_removed}." if dedup_removed else ""

    messages.success(
        request,
        f"Поставлено: {label} ({group}) → {DAY_NAMES[new_day - 1]}, пара {new_slot}.{dest_note}{swap_note}{both_note}{dedup_note}",
    )

    return redirect(next_path)


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
