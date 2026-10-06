import re


def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9\s.,;:%$()/\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_text(text):

    text = clean_text(text)

    return text.lower()


def is_valid_text(text):

    if not text:
        return False

    return len(str(text).strip()) >= 20