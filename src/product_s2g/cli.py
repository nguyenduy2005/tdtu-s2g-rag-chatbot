"""Command-line entrypoint for local S2G chatbot use."""

from __future__ import annotations

import argparse
import json

from .service import S2GChatService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Câu hỏi cần tra cứu")
    args = parser.parse_args()
    service = S2GChatService.build()
    if args.question:
        print(json.dumps(service.chat(args.question), ensure_ascii=False, indent=2))
        return 0
    print("S2G chatbot đã sẵn sàng. Nhập 'exit' để thoát.")
    while True:
        question = input("Bạn> ").strip()
        if question.casefold() in {"exit", "quit"}:
            return 0
        if question:
            print(json.dumps(service.chat(question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
