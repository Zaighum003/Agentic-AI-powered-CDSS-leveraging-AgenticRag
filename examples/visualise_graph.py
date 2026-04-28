"""
AgentCDS Graph Visualiser
==========================
Prints the compiled LangGraph topology as Mermaid and saves a PNG copy
to disk.

Usage:
    python examples/visualise_graph.py
    python examples/visualise_graph.py --mermaid          # print Mermaid source
    python examples/visualise_graph.py --png graph.png    # render PNG to file
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mermaid", action="store_true",
                        help="Print Mermaid diagram source")
    parser.add_argument("--png", nargs="?", const="graph.png", default="graph.png",
                        metavar="FILE",
                        help="Save PNG render to FILE (default: graph.png)")
    args = parser.parse_args()

    from agentcds.graph.graph import graph
    compiled_graph = graph.get_graph()

    print("\n=== AgentCDS LangGraph Topology ===\n")
    try:
        mermaid_src = compiled_graph.draw_mermaid()
        print(mermaid_src)
    except Exception as exc:
        print(f"[warn] Mermaid rendering unavailable: {exc}")

    if args.mermaid:
        try:
            print("\n=== Mermaid Diagram ===\n")
            print(compiled_graph.draw_mermaid())
        except Exception as exc:
            print(f"[warn] draw_mermaid() failed: {exc}")

    try:
        png_bytes = compiled_graph.draw_mermaid_png()
        with open(args.png, "wb") as f:
            f.write(png_bytes)
        print(f"[ok] Graph PNG saved to {args.png}")
    except Exception as exc:
        print(f"[error] PNG render failed: {exc}")
        print("  Ensure the Mermaid PNG renderer dependencies are installed")


if __name__ == "__main__":
    main()
