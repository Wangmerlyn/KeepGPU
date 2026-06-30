import importlib
import logging
import sys

import keep_gpu.utilities.logger as logger_module


def test_console_handler_falls_back_to_plain_formatter_without_colorlog(monkeypatch):
    try:
        with monkeypatch.context() as context:
            context.setitem(sys.modules, "colorlog", None)
            reloaded = importlib.reload(logger_module)

            handler = reloaded._build_console_handler(logging.INFO)
            record = logging.LogRecord(
                name="keep_gpu.tests",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="hello",
                args=(),
                exc_info=None,
            )

            rendered = handler.formatter.format(record)

            assert reloaded.ColoredFormatter is None
            assert "hello" in rendered
            assert "INFO" in rendered
            assert "log_color" not in rendered
    finally:
        importlib.reload(logger_module)
