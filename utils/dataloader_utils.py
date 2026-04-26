def domain_fillin(sequence: str, domain_info: str) -> str:
    """
    Fill in the domain information into the sequence. If the domain is not continuous, link the domain with <unk> tokens.
    input:
        sequence: str, e.g. "MALWMRLLPLLALLALWGPDPAAAPSL"
        domain_info: str, e.g. "1-3_5-10", start from 1
    output:
        domain_filled_sequence: str, e.g. "MAL<unk>MRLLPL"
    """
    domain_info = domain_info.split("_")
    domain_info = [tuple(map(int, domain.split("-"))) for domain in domain_info]
    domain_info = sorted(domain_info, key=lambda x: x[0])
    domain_filled_sequence = []
    for domain in domain_info:
        start, end = domain
        domain_filled_sequence.append(sequence[start - 1 : end])
    return "<unk>".join(domain_filled_sequence)


if __name__ == "__main__":
    sequence = "MALWMRLLPLLALLALWGPDPAAAPSL"
    domain_info = "1-3_5-10"
    print(domain_fillin(sequence, domain_info))
