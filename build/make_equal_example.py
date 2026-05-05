import json, sys, traceback
from pathlib import Path

from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.resume_builder import build_reworded_equivalent_resume


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="/home/zs7353/resume_validity/data/pairs_real.jsonl")
    ap.add_argument("--out_path", type=str, default="/home/zs7353/resume_validity/data/pair_equal_example.json")
    ap.add_argument("--row_index", type=int, default=0, help="Zero-based index into JSONL to pick base resume")
    args = ap.parse_args()

    in_p = Path(args.in_path)
    out_p = Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(in_p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if args.row_index < 0 or args.row_index >= len(lines):
            print("Row index out of range")
            sys.exit(2)
        row = json.loads(lines[args.row_index])
        base_md = row.get("base_resume", "")
        role = row.get("job_role", "")
        if not (base_md and role):
            print("Missing base_resume or job_role in selected row")
            sys.exit(2)
        gemini = GeminiClient()
        equal_md = build_reworded_equivalent_resume(gemini, role, base_md)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "job_role": role,
                "base_resume": base_md,
                "equal_resume": equal_md,
            }, f, ensure_ascii=False)
        print(f"Wrote equal example → {out_p}")
    except Exception as e:
        print(f"Failed to build equal example: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


