from typing import List
import re


def valid_emails(strings: List[str]) -> List[str]:
    """Take list of potential emails and returns only valid ones"""

    pattern = re.compile("^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$")

    def is_valid_email(email: str) -> bool:
        return bool(pattern.fullmatch(valid_email_regex))

    emails = []
    for email in strings:
        if is_valid_email(email):
            emails.append(email)

    return emails