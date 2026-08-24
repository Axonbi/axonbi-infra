#!/usr/bin/env python
"""Draw a graph without Studio -- prints Mermaid, writes a PNG.

    python render_graph.py graphs.hello_flow /tmp/hello_flow.png

Useful when you just want the picture and don't want a browser or a
LangSmith login in the loop.
"""

import importlib
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    module = importlib.import_module(sys.argv[1])
    g = getattr(module, "graph").get_graph()

    print("--- ascii ---")
    try:
        g.print_ascii()
    except Exception as exc:  # grandalf missing or layout failure
        print(f"(ascii unavailable: {exc})")

    print("\n--- mermaid ---")
    print(g.draw_mermaid())

    if len(sys.argv) > 2:
        out = sys.argv[2]
        try:
            with open(out, "wb") as fh:
                fh.write(g.draw_mermaid_png())
            print(f"\nPNG written to {out}")
        except Exception as exc:  # needs network (mermaid.ink) or pyppeteer
            print(f"\nPNG failed ({exc}); use the mermaid text above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
