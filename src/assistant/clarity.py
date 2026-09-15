"""Shared buyer-facing clarity families used by the assistant."""


CLARITY_FAMILIES = ("I", "SI", "VS", "VVS", "IF")


def clarity_family(value) -> str:
    """Collapse detailed Kaggle grades into the five classification families."""
    grade = str(value).strip().upper()
    if grade == "IF":
        return "IF"
    if grade.startswith("VVS"):
        return "VVS"
    if grade.startswith("VS"):
        return "VS"
    if grade.startswith("SI"):
        return "SI"
    if grade.startswith("I"):
        return "I"
    return grade
