from difflib import SequenceMatcher


def validate_item(item):

    if isinstance(item, str):

        item = {
            "task": item,
            "owner": None,
            "deadline": None,
            "status": "pending",
            "evidence": item
        }

    if not isinstance(item, dict):
        return None

    task = item.get(
        "task"
    )

    owner = item.get(
        "owner"
    )

    deadline = item.get(
        "deadline"
    )

    status = item.get(
        "status",
        "pending"
    )

    evidence = item.get(
        "evidence"
    )

    warnings = []

    if not task or not str(task).strip():

        task = ""

        warnings.append(
            "Missing task"
        )

    else:

        task = str(task).strip()

    if not owner or str(owner).strip().lower() in [
        "null",
        "none",
        "unknown",
        "not specified"
    ]:

        owner = None

        warnings.append(
            "Missing owner"
        )

    else:

        owner = str(owner).strip()

    if not deadline or str(deadline).strip().lower() in [
        "null",
        "none",
        "unknown",
        "not specified"
    ]:

        deadline = None

        warnings.append(
            "Missing deadline"
        )

    else:

        deadline = str(deadline).strip()

    if not evidence:

        evidence = ""

        warnings.append(
            "Missing evidence"
        )

    else:

        evidence = str(evidence).strip()

    confidence = 1.0

    confidence -= len(warnings) * 0.15

    if evidence:
        confidence += 0.05

    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

    return {
        "task": task,
        "owner": owner,
        "deadline": deadline,
        "status": str(status),
        "evidence": evidence,
        "confidence": confidence,
        "warnings": warnings
    }


def remove_duplicates(items):

    unique = []

    for item in items:

        duplicate = False

        for existing in unique:

            similarity = SequenceMatcher(
                None,
                item["task"].lower(),
                existing["task"].lower()
            ).ratio()

            if similarity >= 0.85:

                duplicate = True

                break

        if not duplicate:

            unique.append(
                item
            )

    return unique


def validate_items(items):

    validated = []

    if not items:
        return validated

    for item in items:

        result = validate_item(
            item
        )

        if result is not None:

            validated.append(
                result
            )

    return remove_duplicates(
        validated
    )