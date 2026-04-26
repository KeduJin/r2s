import os
import logging
import time
from typing import Optional

try:
    import loguru
except ImportError:
    loguru = None


class MyLogger:
    def __init__(self, output_dir: Optional[str] = None):
        self.my_rank = int(os.environ.get("RANK", "0"))
        if loguru is not None:
            self.logger = loguru.logger
            if output_dir and self.my_rank == 0:
                default_name = "out.log"
                if os.path.exists(os.path.join(output_dir, default_name)):
                    default_name = "out_" + str(time.time()) + ".log"
                self.logger.add(os.path.join(output_dir, default_name))
        else:
            self.logger = logging.getLogger("TED-DPLM")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
                )
                self.logger.addHandler(stream_handler)
            if output_dir and self.my_rank == 0:
                os.makedirs(output_dir, exist_ok=True)
                default_name = "out.log"
                if os.path.exists(os.path.join(output_dir, default_name)):
                    default_name = "out_" + str(time.time()) + ".log"
                file_handler = logging.FileHandler(
                    os.path.join(output_dir, default_name)
                )
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
                )
                self.logger.addHandler(file_handler)

    def info(self, message: str, main_process_only: bool = True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.info(message)
        else:
            self.logger.info(message)

    def info_dic_step(self, dict: dict, step: int, main_process_only: bool = True):
        _str = "step: " + str(step) + "\t"
        for k, v in dict.items():
            if isinstance(v, float):
                _str += f"{k}: {v:.4f}\t"
            else:
                _str += f"{k}: {v}\t"
        self.info(_str, main_process_only)

    def warning(self, message: str, main_process_only: bool = True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.warning(message)
        else:
            self.logger.warning(message)

    def error(self, message: str, main_process_only: bool = True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.error(message)
        else:
            self.logger.error(message)

    def success(self, message: str, main_process_only: bool = True):
        if main_process_only:
            if self.my_rank == 0:
                if hasattr(self.logger, "success"):
                    self.logger.success(message)
                else:
                    self.logger.info(message)
        else:
            if hasattr(self.logger, "success"):
                self.logger.success(message)
            else:
                self.logger.info(message)


if __name__ == "__main__":
    logger = MyLogger()
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.success("This is a success message.")
