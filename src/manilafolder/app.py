# ───────────────────────── src/manilafolder/app.py ─────────────────────────
"""
CLI entrypoint and main application logic.

Build with: pyinstaller --onefile --windowed app.py
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import Config
from .gui import create_gui
from .logging_utils import log_error, setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="ManilaFolder - Digital Document Organization with Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Launch GUI
  %(prog)s --config custom.json  # Use custom config

Build Instructions:
  pyinstaller --onefile --windowed app.py
        """,
    )

    parser.add_argument(
        "--config", type=str, help="Path to configuration file (JSON format)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Set logging level (default: ERROR)",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configuration object

    Raises:
        RuntimeError: If config file cannot be loaded
    """
    if config_path:
        try:
            import json

            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Create config with loaded data
            config = Config(**config_data)
            return config

        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")

    return Config()


def setup_application(args: argparse.Namespace) -> tuple[Config, any]:
    """Set up application configuration and logging.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (config, logger)

    Raises:
        RuntimeError: If application setup fails
    """
    try:
        # Load configuration
        config = load_config(args.config)

        # Set up logging
        logger = setup_logger(config)

        # Log startup
        logger.info("ManilaFolder application starting")
        if args.config:
            logger.info(f"Using config file: {args.config}")

        return config, logger

    except Exception as e:
        print(f"Failed to initialize application: {e}", file=sys.stderr)
        raise


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Set up application
        config, logger = setup_application(args)

        # Create and run GUI
        gui = create_gui(config)
        gui.run()

        logger.info("ManilaFolder application exiting normally")
        return 0

    except KeyboardInterrupt:
        print("\nApplication interrupted by user", file=sys.stderr)
        return 1

    except Exception as e:
        error_msg = f"Application error: {str(e)}"
        print(error_msg, file=sys.stderr)

        # Try to log the error if possible
        try:
            log_error("Application startup failed", e)
        except Exception:
            pass  # Ignore logging errors during shutdown

        return 1


if __name__ == "__main__":
    sys.exit(main())
