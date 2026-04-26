"""从 SaProt 风格的 satoken 串中提取纯氨基酸序列（大写残基）。"""


def extract_amino_acid_sequence_from_satoken(seq: str) -> str:
    if not isinstance(seq, str) or not seq:
        return ""
    return "".join(ch for ch in seq if ch.isupper())
