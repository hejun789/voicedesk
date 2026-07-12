import argparse
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from voicedesk.evals.report import format_console, format_markdown
from voicedesk.evals.runner import load_scenarios, run_scenario
from voicedesk.groq_client import DEFAULT_MODEL, GroqLLM


def select_scenarios(scenarios: list[dict], scenario_id: str | None) -> list[dict]:
    if scenario_id is None:
        return scenarios
    picked = [s for s in scenarios if s["id"] == scenario_id]
    if not picked:
        raise SystemExit(f"no scenario with id {scenario_id!r}")
    return picked


def require_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY not set — put it in .env (see .env.example)")


def resolve_model() -> str:
    """The model that will actually be used, resolved exactly as GroqLLM does."""
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def _log_retry(reason: str, wait_s: float, attempt: int) -> None:
    if reason == "throttle":
        print(f"    approaching token limit — pausing {wait_s:.1f}s",
              file=sys.stderr, flush=True)
    elif reason == "rate_limited":
        print(f"    rate limited — waiting {wait_s:.1f}s (retry {attempt})",
              file=sys.stderr, flush=True)
    else:
        print(f"    malformed tool call — resampling (retry {attempt})",
              file=sys.stderr, flush=True)


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
    p.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait between runs, to stay under API rate limits")
    args = p.parse_args()

    require_api_key()

    scenarios = select_scenarios(load_scenarios(args.scenarios), args.scenario)
    model = resolve_model()
    print(f"Model:     {model}", file=sys.stderr, flush=True)
    print(f"Scenarios: {len(scenarios)}", file=sys.stderr, flush=True)
    print(f"Runs:      {args.runs}", file=sys.stderr, flush=True)
    print(f"Running {len(scenarios)} scenario(s) x {args.runs} run(s)...\n",
          file=sys.stderr, flush=True)

    results = []
    for i, scenario in enumerate(scenarios, start=1):
        if i > 1 and args.delay:
            time.sleep(args.delay)
        scenario_results = run_scenario(
            scenario, lambda: GroqLLM(model=model, on_retry=_log_retry), runs=args.runs)
        passed = sum(1 for r in scenario_results if r.passed)
        print(f"[{i}/{len(scenarios)}] {scenario['id']} ... "
              f"{passed}/{len(scenario_results)}", file=sys.stderr, flush=True)
        results.extend(scenario_results)

    print(format_console(results, model=model))

    out = args.out or f"reports/eval-{datetime.now():%Y%m%d-%H%M%S}.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(format_markdown(results, model=model))
    print(f"\nMarkdown report written to {out}")


if __name__ == "__main__":
    main()
