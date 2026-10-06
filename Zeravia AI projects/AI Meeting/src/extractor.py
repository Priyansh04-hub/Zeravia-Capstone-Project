import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.preprocessing import clean_transcript, split_sentences


class MeetingActionExtractor:

    def __init__(self):

        self.model_name = "google/flan-t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        self.model.eval()

    def find_candidates(self, transcript):

        sentences = split_sentences(
            transcript
        )

        patterns = [
            r"\bwill\b",
            r"\bi'll\b",
            r"\bi will\b",
            r"\bwe'll\b",
            r"\bwe will\b",
            r"\bneed to\b",
            r"\bneeds to\b",
            r"\bplease\b",
            r"\bhas to\b",
            r"\bhave to\b",
            r"\bgoing to\b",
            r"\btake care of\b",
            r"\bhandle\b",
            r"\bprepare\b",
            r"\bupdate\b",
            r"\bsend\b",
            r"\bcall\b",
            r"\breview\b",
            r"\bfinish\b",
            r"\bcomplete\b",
            r"\bdeliver\b",
            r"\bby tomorrow\b",
            r"\bby next week\b",
            r"\bby monday\b",
            r"\bby tuesday\b",
            r"\bby wednesday\b",
            r"\bby thursday\b",
            r"\bby friday\b",
            r"\bby saturday\b",
            r"\bby sunday\b"
        ]

        candidates = []

        for sentence in sentences:

            text = sentence.strip()

            if len(text) < 8:
                continue

            lower = text.lower()

            if any(
                re.search(
                    pattern,
                    lower
                )
                for pattern in patterns
            ):

                candidates.append(text)

        return candidates

    def build_prompt(self, sentence):

        return f"""
Extract one genuine action item from the sentence below.

If there is no genuine action item, return:
NONE

Otherwise return exactly:

TASK: task description
OWNER: person responsible or unknown
DEADLINE: deadline or unknown

Sentence:
{sentence}
"""

    def extract_sentence(self, sentence):

        prompt = self.build_prompt(
            sentence
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

    def parse_result(self, output, evidence):

        if not output:
            return None

        if output.strip().upper() == "NONE":
            return None

        task_match = re.search(
            r"TASK:\s*(.*?)(?=\nOWNER:|\nDEADLINE:|$)",
            output,
            re.IGNORECASE | re.DOTALL
        )

        owner_match = re.search(
            r"OWNER:\s*(.*?)(?=\nDEADLINE:|$)",
            output,
            re.IGNORECASE | re.DOTALL
        )

        deadline_match = re.search(
            r"DEADLINE:\s*(.*)",
            output,
            re.IGNORECASE | re.DOTALL
        )

        if not task_match:
            return None

        task = task_match.group(1).strip()

        if not task:
            return None

        owner = (
            owner_match.group(1).strip()
            if owner_match
            else None
        )

        deadline = (
            deadline_match.group(1).strip()
            if deadline_match
            else None
        )

        if owner and owner.lower() in [
            "unknown",
            "none",
            "null"
        ]:
            owner = None

        if deadline and deadline.lower() in [
            "unknown",
            "none",
            "null"
        ]:
            deadline = None

        return {
            "task": task,
            "owner": owner,
            "deadline": deadline,
            "status": "pending",
            "evidence": evidence
        }

    def fallback_action(self, sentence):

        owner = None
        deadline = None

        deadline_patterns = [
            r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\bby tomorrow\b",
            r"\bby next week\b",
            r"\bby (today|tonight)\b"
        ]

        for pattern in deadline_patterns:

            match = re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )

            if match:

                deadline = match.group(0)

                break

        owner_patterns = [
            r"^([A-Z][a-z]+),",
            r"^([A-Z][a-z]+)\s+will\b",
            r"^([A-Z][a-z]+)\s+is going to\b"
        ]

        for pattern in owner_patterns:

            match = re.search(
                pattern,
                sentence
            )

            if match:

                owner = match.group(1)

                break

        task = sentence.strip()

        return {
            "task": task,
            "owner": owner,
            "deadline": deadline,
            "status": "pending",
            "evidence": sentence
        }

    def extract(self, transcript):

        transcript = clean_transcript(
            transcript
        )

        if not transcript:
            return []

        candidates = self.find_candidates(
            transcript
        )

        actions = []

        for sentence in candidates:

            output = self.extract_sentence(
                sentence
            )

            action = self.parse_result(
                output,
                sentence
            )

            if action is None:

                action = self.fallback_action(
                    sentence
                )

            actions.append(
                action
            )

        return self.remove_duplicates(
            actions
        )

    def remove_duplicates(self, actions):

        unique = []
        seen = set()

        for action in actions:

            key = re.sub(
                r"\s+",
                " ",
                action["task"].lower().strip()
            )

            if key not in seen:

                seen.add(key)

                unique.append(
                    action
                )

        return unique