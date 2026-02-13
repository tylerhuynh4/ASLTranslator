from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass

@dataclass
class TextState:
    current_word: str = ""
    sentence: str = ""
    last_confirmed: str | None = None

class TemporalSmoother:
    """
    Smooth frame-by-frame predictions:
    - keeps a window of recent predictions
    - returns a stable token only when it dominates the window
    """
    def __init__(self, window: int, stability_ratio: float):
        self.window = window
        self.stability_ratio = stability_ratio
        self.buf: deque[str] = deque(maxlen=window)

    def update(self, pred: str) -> str | None:
        if not pred:
            return None
        self.buf.append(pred)

        if len(self.buf) < self.window:
            return None

        counts = Counter(self.buf)
        token, freq = counts.most_common(1)[0]
        ratio = freq / float(self.window)

        if ratio >= self.stability_ratio:
            return token
        return None

class TokenBuffer:
    """
    Converts confirmed tokens into a sentence.
    Supports special tokens:
      - SPACE: finalize current_word
      - DEL: delete last char (word) or last char of sentence
      - CLEAR: clear everything
    """
    def __init__(self, space_token="SPACE", delete_token="DEL", clear_token="CLEAR"):
        self.space_token = space_token
        self.delete_token = delete_token
        self.clear_token = clear_token
        self.state = TextState()

    def reset(self):
        self.state = TextState()

    def add_token(self, token: str) -> TextState:
        token = token.strip()

        if token == self.clear_token:
            self.reset()
            return self.state

        if token == self.delete_token:
            # delete from current word first; then sentence
            if self.state.current_word:
                self.state.current_word = self.state.current_word[:-1]
            elif self.state.sentence:
                self.state.sentence = self.state.sentence[:-1]
            return self.state

        if token == self.space_token:
            if self.state.current_word:
                self.state.sentence += self.state.current_word + " "
                self.state.current_word = ""
            return self.state

        # Normal alphanumeric token (letter or word)
        # If your model outputs words, this will append words continuously.
        # For alphabet classification, this builds words letter-by-letter.
        if len(token) == 1:
            self.state.current_word += token
        else:
            # If token looks like a word, treat as word boundary
            if self.state.current_word:
                self.state.sentence += self.state.current_word + " "
                self.state.current_word = ""
            self.state.sentence += token + " "

        return self.state

def clean_text(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return ""
    # Capitalize first letter; keep rest as-is
    return text[0].upper() + text[1:]
