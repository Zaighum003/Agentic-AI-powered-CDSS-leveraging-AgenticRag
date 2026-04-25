"""
AgentCDS Graph Visualiser
==========================
Prints the compiled LangGraph topology as ASCII and optionally saves a
Mermaid diagram to disk.

Usage:
    python examples/visualise_graph.py
    python examples/visualise_graph.py --mermaid          # print Mermaid source
    python examples/visualise_graph.py --png graph.png    # render PNG (requires graphviz)
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mermaid", action="store_true",
                        help="Print Mermaid diagram source")
    parser.add_argument("--png", metavar="FILE",
                        help="Save PNG render to FILE (requires graphviz)")
    args = parser.parse_args()

    from agentcds.graph.graph import graph

    # ── ASCII topology ────────────────────────────────────────────
    print("\n=== AgentCDS LangGraph Topology ===\n")
    print("Nodes:")
    for node in graph.nodes:
        print(f"  {node}")

    # ── Mermaid source ────────────────────────────────────────────
    if args.mermaid:
        try:
            mermaid_src = graph.get_graph().draw_mermaid()
            print("\n=== Mermaid Diagram ===\n")
            print(mermaid_src)
        except AttributeError:
            print("[warn] draw_mermaid() not available in this LangGraph version")

    # ── PNG render ────────────────────────────────────────────────
    if args.png:
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            with open(args.png, "wb") as f:
                f.write(png_bytes)
            print(f"[ok] Graph PNG saved to {args.png}")
        except Exception as exc:
            print(f"[error] PNG render failed: {exc}")
            print("  Ensure graphviz is installed: pip install graphviz")


if __name__ == "__main__":
    main()
