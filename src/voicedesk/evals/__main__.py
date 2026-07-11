import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from voicedesk.evals.report import format_console, format_markdown
from voicedesk.evals.runner import load_scenarios, run_all
from voicedesk.groq_client import GroqLLM


def select_scenarios(scenarios: list[dict], scenario_id: str | None) -> list[dict]:
    if scenario_id is None:
        return scenarios
    picked = [s for s in scenarios if s["id"] == scenario_id]
    if not picked:
        raise SystemExit(f"no scenario with id {scenario_id!r}")
    return picked


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        prog="voicedesk.evals",
        description="Run the VoiceDesk eval suite against the live Groq agent.",
    )
    p.add_argument("--scenarios", default="evals/scenarios.json")
    p.add_argument("--scenario", default=None, help="run only this scenario id")
    p.add_argument("--runs", type=int, default=3, help="runs per scenario")
    p.add_argument("--out", default=None, help="markdown report path")
    args = p.parse_args()

    scenarios = select_scenarios(load_scenarios(args.scenarios), args.scenario)
    print(f"Running {len(scenarios)} scenario(s) x {args.runs} run(s)...\n",
          file=sys.stderr)

    results = run_all(scenarios, lambda: GroqLLM(), runs=args.runs)

    print(format_console(results))

    out = args.out or f"reports/eval-{datetime.now():%Y%m%d-%H%M%S}.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(format_markdown(results))
    print(f"\nMarkdown report written to {out}")


if __name__ == "__main__":
    main()
