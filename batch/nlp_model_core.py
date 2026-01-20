# /home/ssm-user/jupyter/batch/nlp_model_core.py
import re
import html
import unicodedata
import pandas as pd
import numpy as np


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


# ===== 전처리 =====
def strip_html(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = html.unescape(s)
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


ALLOWED_PATTERN = re.compile(
    r"[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s\.\,\!\?\;\:\'\"\(\)\[\]\{\}\-\_\/\&\+\%\#\@]"
)


def whitelist_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = strip_html(text)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200B-\u200D\uFE0E\uFE0F]", "", s)
    s = ALLOWED_PATTERN.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===== 문장/토큰 =====
SENT_SPLIT = re.compile(r"[\.!\?…\n\r]+|[。！？]")
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


# ===== compound / stop =====
def build_compound_replacers(compound_words: dict) -> list:
    if not compound_words:
        return []
    keys_sorted = sorted([k for k in compound_words.keys() if k], key=len, reverse=True)
    replacers = []
    for k in keys_sorted:
        v = compound_words[k]
        pattern = re.compile(re.escape(k), flags=re.IGNORECASE)
        replacers.append((pattern, v))
    return replacers


def apply_compound(text: str, replacers: list) -> str:
    if not text or not replacers:
        return text or ""
    s = text
    for rgx, rep in replacers:
        s = rgx.sub(rep, s)
    return s


def remove_stopwords(text: str, stop_words: set) -> str:
    if not text:
        return ""
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return text.strip()
    if not stop_words:
        return " ".join(tokens)
    kept = [t for t in tokens if t not in stop_words]
    return " ".join(kept)


def preprocess_for_match(raw_text: str, compound_replacers: list, stop_words: set) -> str:
    base = whitelist_text(raw_text)
    base = base.lower()
    base = apply_compound(base, compound_replacers)
    base = remove_stopwords(base, stop_words)
    base = re.sub(r"\s+", " ", base).strip()
    return base


# ===== taxonomy (DB 스키마 고정: major/minor/keywords) =====
def build_taxonomy_db(
    tax_df: pd.DataFrame,
    compound_replacers: list | None = None,
    stop_words: set | None = None,
) -> list:
    compound_replacers = compound_replacers or []
    stop_words = stop_words or set()

    if tax_df is None or tax_df.empty:
        return []

    required = ["major", "minor", "keywords"]
    for c in required:
        if c not in tax_df.columns:
            raise KeyError(f"voc_taxonomy 컬럼 누락: {c}")

    items = []
    for _, row in tax_df.iterrows():
        major = safe_str(row.get("major", "")).strip()
        minor = safe_str(row.get("minor", "")).strip()
        raw = safe_str(row.get("keywords", "")).strip()

        if not (major and minor and raw):
            continue

        parts = re.split(r"[,\;\|\n\r\t]+", raw)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        norm_patterns = []
        for p in parts:
            pn = preprocess_for_match(p, compound_replacers, stop_words)
            if pn:
                norm_patterns.append(pn)

        items.append(
            {
                "major": major,
                "minor": minor,
                "patterns": parts,
                "norm_patterns": norm_patterns,
            }
        )

    return items


def find_tax_etc(items: list) -> tuple[str, str] | None:
    for it in items:
        if it.get("major") == "기타" and it.get("minor") == "기타":
            return ("기타", "기타")
    for it in items:
        if it.get("major") == "기타":
            return (it.get("major", "기타"), it.get("minor") or "기타")
    return None


# ===== 가중치 =====
def _length_weight(token_count: int, short_len: int, long_len: int, short_penalty: float, long_penalty: float) -> float:
    if token_count <= short_len:
        return short_penalty
    if token_count >= long_len:
        return long_penalty
    return 1.0


def _overlap_with_title(sent_norm: str, title_norm: str, overlap_ratio: float) -> bool:
    if not title_norm:
        return False
    stoks = set(TOKEN_RE.findall(sent_norm))
    ttoks = set(TOKEN_RE.findall(title_norm))
    if not stoks or not ttoks:
        return False
    inter = len(stoks & ttoks)
    return inter >= max(1, int(overlap_ratio * len(stoks)))


# ===== 복합 키워드 매칭 =====
def _match_compound_keyword(pattern: str, text: str) -> bool:
    """
    복합 키워드 매칭: 패턴의 모든 단어가 텍스트에 포함되어 있으면 매칭
    - 단일 단어: 기존과 동일 (부분 문자열 검사)
    - 복합 단어 (공백 포함): 모든 단어가 텍스트에 있으면 매칭

    예시:
      - "교체" in "세면대 부품 교체 작업" → True
      - "세면대 교체" → "세면대" in text AND "교체" in text → True
    """
    if not pattern or not text:
        return False

    words = pattern.split()
    if len(words) == 1:
        # 단일 단어: 기존 로직 (부분 문자열)
        return pattern in text
    else:
        # 복합 단어: 모든 단어가 텍스트에 포함
        return all(w in text for w in words)


# ===== 분류기 =====
def classify_weighted_general(
    all_text: str,
    items: list,
    title_text: str | None = None,
    pos_max: float = 1.5,
    pos_min: float = 1.0,
    title_boost: float = 2.0,
    title_overlap_ratio: float = 0.3,
    short_len: int = 5,
    long_len: int = 60,
    short_penalty: float = 0.9,
    long_penalty: float = 0.9,
    decay_steps: int = 5,
    length_bonus_scale: float = 0.1,
    compound_replacers: list | None = None,
    stop_words: set | None = None,
) -> tuple[str, str, float]:
    compound_replacers = compound_replacers or []
    stop_words = stop_words or set()

    text = "" if all_text is None else str(all_text)
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    title_norm = preprocess_for_match(title_text or "", compound_replacers, stop_words)
    item_scores = np.zeros(len(items), dtype=float)

    for idx_s, sent in enumerate(sentences):
        tok_cnt = len(TOKEN_RE.findall(sent.lower()))
        s_norm = preprocess_for_match(sent, compound_replacers, stop_words)

        step = min(idx_s, max(1, decay_steps))
        pos_w = pos_max - step * ((pos_max - pos_min) / max(1, decay_steps))
        pos_w = max(pos_min, min(pos_max, pos_w))

        title_w = title_boost if _overlap_with_title(s_norm, title_norm, title_overlap_ratio) else 1.0
        len_w = _length_weight(tok_cnt, short_len, long_len, short_penalty, long_penalty)
        sent_w = pos_w * title_w * len_w

        for i, it in enumerate(items):
            norm_ps = it.get("norm_patterns") or []
            hits = 0
            length_bonus = 0
            for p in norm_ps:
                if p and _match_compound_keyword(p, s_norm):
                    hits += 1
                    length_bonus += len(p)
            if hits > 0:
                item_scores[i] += sent_w * (hits + length_bonus_scale * length_bonus)

    if item_scores.max() > 0:
        best_i = int(item_scores.argmax())
        return items[best_i]["major"], items[best_i]["minor"], float(item_scores[best_i])
    return "", "", 0.0