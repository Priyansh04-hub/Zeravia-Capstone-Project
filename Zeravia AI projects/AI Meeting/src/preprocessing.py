import re


def clean_transcript(text):

    text = str(text)

    text = re.sub(
        r"\[[^\]]*\]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def split_speakers(text):

    text = clean_transcript(
        text
    )

    patterns = [
        r"(?=[A-Z][A-Za-z0-9 _-]{0,40}:)",
        r"(?=\bSpeaker\s*\d*\s*:)",
        r"(?=\bPerson\s*\d*\s*:)"
    ]

    for pattern in patterns:

        parts = re.split(
            pattern,
            text
        )

        if len(parts) > 1:

            return [
                part.strip()
                for part in parts
                if part.strip()
            ]

    return [text]


def split_sentences(text):

    text = clean_transcript(
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]