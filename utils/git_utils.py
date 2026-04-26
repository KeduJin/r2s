import subprocess


def get_commit_hash(short: bool = True) -> str:
    """
    获取当前 Git 仓库的 commit hash。

    Args:
        short (bool): 是否返回短 hash (前7位)。默认为 True。

    Returns:
        str: Git commit hash。如果获取失败（例如没有 git 环境或不是 git 仓库），返回 "Unknown"。
    """
    try:
        # 构建命令
        cmd = ["git", "rev-parse", "HEAD"]
        if short:
            cmd.insert(2, "--short")

        # 执行命令并获取输出
        commit_hash = (
            subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,  # 忽略错误输出（例如不是 git 仓库时）
            )
            .decode("ascii")
            .strip()
        )

        return commit_hash

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # 1. CalledProcessError: 命令执行返回非0状态码（例如当前目录不在 git 仓库中）
        # 2. FileNotFoundError: 系统中没有安装 git
        return "Unknown"


# 这是一个额外赠送的实用函数，通常和 commit hash 一起用
def get_branch_name() -> str:
    """获取当前 Git 分支名称"""
    try:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        branch = (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        return branch
    except Exception:
        return "Unknown"
