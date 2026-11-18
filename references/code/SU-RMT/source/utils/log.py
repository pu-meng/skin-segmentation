import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional


class Logger(object):
    """
    Enhanced logging class, supporting:
    1. Structured JSON metrics recording
    2. Separate detailed logs and concise metrics
    """
    def __init__(self, logfile: str, metrics_file: Optional[str] = None):
        """
        Initialize logger

        Args:
            logfile (str): Main log file path (contains all console outputs)
            metrics_file (str): Structured metrics JSON file path (optional)
        """
        self.log = open(logfile, "a", encoding="utf-8")
        # 记录原始stdout，便于关闭时恢复，避免Python退出阶段flush到已关闭文件
        self._orig_stdout = sys.stdout
        self._closed = False

        # Structured metrics log (if provided path)
        self.metrics_file = metrics_file
        if metrics_file:
            # Initialize or clear metrics file
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write("")  # Clear file

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Record structured metrics to JSON file

        Args:
            metrics: Dictionary of metrics to record
        """
        if not self.metrics_file:
            return

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()

        # Write JSON line
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False)
            f.write('\n')

    def flush(self):
        """Flush buffer"""
        try:
            if self.log and not self.log.closed:
                self.log.flush()
        except Exception:
            # 在解释器退出阶段或其他异常情况下，静默忽略flush错误
            pass

    def write(self, message: str):
        """Write message to log file"""
        try:
            if self.log and not self.log.closed:
                self.log.write(message)
                return len(message)
            else:
                # 日志已关闭时，退回到原始stdout，避免 ValueError
                try:
                    self._orig_stdout.write(message)
                    return len(message)
                except Exception:
                    return 0
        except Exception:
            # 任意写入异常都不再向外抛出，防止影响主流程
            try:
                self._orig_stdout.write(message)
                return len(message)
            except Exception:
                return 0

    def isatty(self) -> bool:
        return False

    def log_config(self, config: Dict[str, Any]):
        """
        Record configuration information to JSON file (as the first line)

        Args:
            config: Configuration dictionary
        """
        if not self.metrics_file:
            return

        config_entry = {
            'type': 'config',
            'timestamp': datetime.now().isoformat(),
            **config
        }

        # Write configuration as the first entry
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            json.dump(config_entry, f, ensure_ascii=False)
            f.write('\n')

    def close(self):
        """Close log file"""
        if self._closed:
            return
        try:
            if self.log and not self.log.closed:
                self.log.close()
        finally:
            # 恢复stdout，避免后续flush/write作用于已关闭文件
            try:
                sys.stdout = self._orig_stdout
            except Exception:
                pass
            self._closed = True

    def __del__(self):
        # 确保资源被释放且stdout被恢复
        try:
            self.close()
        except Exception:
            pass
