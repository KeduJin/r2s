def find_best_lcs_match_in_target(part, target_substring):
    """
    辅助函数：在目标子串中找到与查询部分part具有最大LCS长度的匹配。

    返回:
        一个元组 (max_lcs_length, match_end_index_in_substring)
    """
    m, n = len(part), len(target_substring)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    max_lcs = 0
    best_end_index = -1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if part[i - 1] == target_substring[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 关键：我们要找的是使用了整个 part (即最后一行) 的情况下，LCS长度最大的点
    # 这代表了 part 与 target_substring 的某个前缀的最佳匹配
    if m > 0:  # 只有当 part 不为空时才查找
        last_row = dp[m]
        for j, lcs_len in enumerate(last_row):
            if lcs_len > max_lcs:
                max_lcs = lcs_len
                best_end_index = j  # 记录在子串中的结束索引

    return max_lcs, best_end_index


def get_wildcard_lcs_score(query_with_wildcard, target_sequence):
    """
    input:
        query_with_wildcard: str, e.g. "ABC*JISD"
        target_sequence: str, e.g. "SDFIJABCSOIJPSDsaoidfj"
    output:
        final_score: float,
        total_lcs_found: int,
        total_query_len: int,
    """
    query_parts = query_with_wildcard.split("*")
    # 过滤掉因 "a**b" 等情况产生的空字符串
    query_parts = [p for p in query_parts if p]

    total_query_len = sum(len(p) for p in query_parts)
    if total_query_len == 0:
        return 1.0  # 如果查询是 '*' 或 ''，可以认为匹配度为1

    total_lcs_found = 0
    current_search_start_index = 0

    for i, part in enumerate(query_parts):
        target_substring = target_sequence[current_search_start_index:]
        if not target_substring:
            break  # 目标序列已经用完

        part_lcs, match_end_in_substring = find_best_lcs_match_in_target(
            part, target_substring
        )

        if match_end_in_substring == -1:
            # 匹配失败，可以提前退出，或者根据需求算作0分继续
            continue

        total_lcs_found += part_lcs

        # 更新下一次搜索的全局起始位置
        match_end_global = current_search_start_index + match_end_in_substring
        current_search_start_index = match_end_global

    if total_query_len == 0:
        return 0.0

    final_score = total_lcs_found / total_query_len
    return final_score, total_lcs_found, total_query_len


def get_batched_wildcard_lcs_score(
    query_with_wildcard: list[list[str]], target_sequence: list[str]
) -> list[float]:
    """
    input:
        query_with_wildcard: list[[list[str]]], e.g. [["ABC*JISD", saoid], ["ABC*JISD"]]
        target_sequence: list[str], e.g. ["SDFIJABCSOIJPSDsaoidfj", "SDFIJABCSOIJPSDsaoidfj"]
    output:
        final_score: list[float],
        total_lcs_found: list[int],
        total_query_len: list[int],
    """

    final_score = []
    if not isinstance(target_sequence, list):
        query_with_wildcard = [query_with_wildcard]
        target_sequence = [target_sequence]

    for query, target in zip(query_with_wildcard, target_sequence):
        total_lcs_found = 0
        total_query_len = 0
        for q in query:
            wildcard_q = q.replace("<unk>", "*")
            score, lcs_found, query_len = get_wildcard_lcs_score(wildcard_q, target)
            total_lcs_found += lcs_found
            total_query_len += query_len
        final_score.append(total_lcs_found / total_query_len)

    return final_score


if __name__ == "__main__":
    query_with_wildcard = [
        [
            "TLQIFLALAIGLLLGLFFGEDIKVIKWVGDVWIRLMQMAVLPYMTASLISGIGSLDTGLARNMALRGGGLLLLFWLISAAIILLMPLAFPKWVDSAFLSNESLADHPGFNPIELYIPDNPFHSLANSIVPGVVLFSITVGVALINVQKKAEFIDGLKVLTEALSKVMAFVVRLTPIGVLSIVAVAAGTLSVEVIGQLEVYFVVYIVSALLLTFVVLPLVISTLTPFSYLDVMRFSKSALLTGFITQNVLITFPLLITKSRELFQKYALETEQTDHVVDVIIPVTFNFPNCGRLLALLFIPFASWMSGADLALGDYPQLISAGIFSLFAKAQIALVFLLDLFRLPHDLFALYIPSAIINGRFDTLASVMNLFAFSVIVSVGLNGNLIWNVRKVLISLSIIILSLLLTVLATRLALQSFLKVDY",
            "YAGMTVFKTLPEAQPLSPRQGDLLSEIRQRQVLRVGYRVDRHPLAFFNNRDELVGLHVRLLNELAADLGVRIEYYPFDWPHFKDNLNTHQLDIVPGVAYDTFNIVDAALTEPYLQGHLCFLVKDFRRHDFASKDKIQGLEKLQIAISGDTLIVEKIGDRLRSKLPGVDLDVTPINEYSEFFALNDQVDALVESCEICSARALLHPEYTSMLPKELSLAYPLSFAVPYGQTEFANFLSQWIAVKKN",
        ]
    ]
    target_sequence = [
        "MPLAFKRFKIDLTLQIFLALAIGLLLGLFFGEDIKVIKWVGDVWIRLMQMAVLPYMTASLISGIGSLDTGLARNMALRGGGLLLLFWLISAAIILLMPLAFPKWVDSAFLSNESLADHPGFNPIELYIPDNPFHSLANSIVPGVVLFSITVGVALINVQKKAEFIDGLKVLTEALSKVMAFVVRLTPIGVLSIVAVAAGTLSVEVIGQLEVYFVVYIVSALLLTFVVLPLVISTLTPFSYLDVMRFSKSALLTGFITQNVLITFPLLITKSRELFQKYALETEQTDHVVDVIIPVTFNFPNCGRLLALLFIPFASWMSGADLALGDYPQLISAGIFSLFAKAQIALVFLLDLFRLPHDLFALYIPSAIINGRFDTLASVMNLFAFSVIVSVGLNGNLIWNVRKVLISLSIIILSLLLTVLATRLALQSFLKVDYVKDQMLQSMQLRSPYAGMTVFKTLPEAQPLSPRQGDLLSEIRQRQVLRVGYRVDRHPLAFFNNRDELVGLHVRLLNELAADLGVRIEYYPFDWPHFKDNLNTHQLDIVPGVAYDTFNIVDAALTEPYLQGHLCFLVKDFRRHDFASKDKIQGLEKLQIAISGDTLIVEKIGDRLRSKLPGVDLDVTPINEYSEFFALNDQVDALVESCEICSARALLHPEYTSMLPKELSLAYPLSFAVPYGQTEFANFLSQWIAVKKNTGFWQDAIDYWVYAKGARPAQKRWSIKKDVFGW"
    ]
    target_sequence_2 = [
        "MSLQSFLLLLIVTLQPSVTLTPDNSNSLTDKTLQIGTLQLLKSWKALSSTLSDSSEDSLTSRPATLQDVILSLVFSQIFLALASSSSLADDIPADTPTHSETTTTSTSPARTVFGESPLFNPSNSFAIQGSYLTLDQSLTLSLQTLSVRSTTPTTTTRPTSFSDAITDSYIPNLPDQKTPSSGNRSQNLSSPSSSPQLDHTPSTSCFSTLRPITSNKSIQPSLTPNCSSLSSTLTYLTFQNFTTPSTSTMPPSVLHNPQFIVISTAKTPNPVISISQTSARTSATSSTPIFSTIQDSNNNNNQRCLSSSSRSNNNSFSGTKSDIFTRQPSSFQLTKTLTFDNSSTNGCNPSKRSPSIFLPFHSNISRKFDSSGSFYEVNESFKNTTNNFPPVVVVTKEDDFADIDGALLCTSSSRRLKNASVSISNKDNSSSSSSSSSSNSSNSNNSRTSNSSRSKLYTSSSSESSSSSTSSSSSSSSSLSSSSTSSSSRSSSSSTSTSSSSSSSSSTSSSSTSSISSNSSSSFSSSSSSSSSTSSLSLSTSSSSSSTTTSSSSSSSSSNRSSSSSSSNSTSSSSTSPSSTSSSSSSSSSSSSSSSTSSSTNSSSSSSSSSSSSTSVSSSSSFSSSSSSTSSSSSSSPTSSSSSSTTSSSSTVSSTTPSTTTTSSTTSSSSSSSSSSSSSPSISSNSSSLSSSTNSSSTSSSSSSLSTSSSSTSSFSTNSSSSSNS"
    ]
    print(
        "Domain recovery with 100% sequence",
        get_batched_wildcard_lcs_score(query_with_wildcard, target_sequence),
    )
    print(
        "Domain recovery with another sequence",
        get_batched_wildcard_lcs_score(query_with_wildcard, target_sequence_2),
    )

    # print(get_wildcard_lcs_score(query_with_wildcard[0][0], target_sequence[0])[0])
    # print(get_wildcard_lcs_score(query_with_wildcard[0][1], target_sequence[0])[0])

    # print(get_wildcard_lcs_score(query_with_wildcard[0][0], target_sequence_2[0])[0])
    # print(get_wildcard_lcs_score(query_with_wildcard[0][1], target_sequence_2[0])[0])
