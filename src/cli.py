"""Command line interface for scene perception experiments."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scene perception on an image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--output-json", default="artifacts/json/result.json")
    parser.add_argument("--output-image", default="artifacts/images/result.jpg")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise NotImplementedError(f"Pipeline entry point is not implemented yet: {args.image}")


if __name__ == "__main__":
    main()
