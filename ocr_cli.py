import argparse
import importlib
from pathlib import Path

from ocr_service.core.paths import set_project_root


COMMANDS = {
    "ocr": "OCR text and invoice layout (PyTorch)",
    "recognizer": "Layout or table structure recognition",
    "pipeline": "Layout, table and OCR combined into markdown",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OCR command-line tools",
        epilog="\n".join(f"  {name:<12}{text}" for name, text in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments of the command; use '<command> --help'")
    args = parser.parse_args(argv)
    set_project_root(Path(__file__).resolve().parent)
    importlib.import_module(f"ocr_service.cli.{args.command}").run(args.args)


if __name__ == "__main__":
    main()
