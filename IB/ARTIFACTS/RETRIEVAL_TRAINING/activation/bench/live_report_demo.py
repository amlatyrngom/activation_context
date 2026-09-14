"""
Writes a small live report for a few minutes so `sky watch` can be exercised end to end:

    uv run sky exec <node> uv run python -m activation.bench.live_report_demo --seconds 180
    uv run sky watch <node> '~/activation_artifacts/live_report_demo/' IB/TMP/LIVE_REPORT_DEMO/ --until-file done.json

The page under the watched folder morphs in the editor as the points arrive; `done.json` is written last.
"""
import argparse
import json
import math
import os
import random
import time

from activation.common.reporting import HtmlReporter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--report-folder", default=os.path.expanduser("~/activation_artifacts/live_report_demo"))
    args = parser.parse_args()
    args.report_folder = os.path.expanduser(args.report_folder)
    reporter = HtmlReporter(args.report_folder, "Live report demo", "A curve that grows while you watch.",
                            eyebrow="Watch demo", refresh_seconds=10, min_render_interval_seconds=0)
    reporter.initialize_line_plot("curve", "Decaying noisy curve", "One point per tick.", "tick", "value", ["value"])
    reporter.initialize_table("ticks", "Ticks", "", ["tick", "time", "value"])
    rng = random.Random(0)
    start = time.time()
    tick = 0
    while time.time() - start < args.seconds:
        tick += 1
        value = math.exp(-tick / 20) + 0.05 * rng.random()
        reporter.add_data_point("curve", {"x": tick, "value": value})
        reporter.add_data_point("ticks", {"tick": tick, "time": time.strftime("%H:%M:%S"), "value": f"{value:.3f}"})
        reporter.set_status(tick=str(tick), elapsed=f"{time.time() - start:.0f}s")
        reporter.render(force=True)
        print(f"tick {tick} value {value:.3f}", flush=True)
        time.sleep(args.interval)
    reporter.finish()
    with open(os.path.join(args.report_folder, "done.json"), "w") as handle:
        json.dump({"ticks": tick, "seconds": time.time() - start}, handle)
    print("done", flush=True)


if __name__ == "__main__":
    main()
