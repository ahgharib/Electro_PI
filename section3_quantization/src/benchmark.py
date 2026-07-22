"""
BenchmarkRunner: runs a fixed prompt set through one engine, logging
resource usage and timing for every prompt via MetricsLogger (the same
logger class Section 4 reuses for live request monitoring).
"""
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .engines.base import InferenceEngine
from .monitoring import MetricsLogger, ResourceMonitor

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(
        self,
        engine: InferenceEngine,
        prompts: list[str],
        max_new_tokens: int = 200,
        metrics_log_path: str = "results/metrics.jsonl",
    ):
        self.engine = engine
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.monitor = ResourceMonitor()
        self.metrics = MetricsLogger(metrics_log_path)

    def run(self) -> dict:
        results = []

        with self.metrics.track("model_load", model_id=getattr(self.engine, "model_id", "unknown")):
            self.engine.load()

        try:
            for i, prompt in enumerate(self.prompts, start=1):
                logger.info("Prompt %d/%d", i, len(self.prompts))
                with self.metrics.track("generation", prompt_index=i) as ctx:
                    gen_result = self.engine.generate(prompt, max_new_tokens=self.max_new_tokens)
                    ctx["output_tokens"] = gen_result.output_tokens
                    ctx["tokens_per_second"] = round(gen_result.tokens_per_second, 3)
                results.append(asdict(gen_result))
            final_snapshot = self.monitor.snapshot()
        finally:
            self.engine.unload()

        ok_results = [r for r in results if not r["error"]]
        avg_tps = (
            sum(r["output_tokens"] / r["generation_time_s"] for r in ok_results if r["generation_time_s"] > 0)
            / len(ok_results)
            if ok_results
            else 0.0
        )

        report = {
            "precision_label": self.engine.precision_label,
            "resolved_model_id": getattr(self.engine, "resolved_model_id", None),
            "device": getattr(self.engine, "_device", None),
            "auto_fallback_triggered": getattr(self.engine, "auto_fallback_triggered", False),
            "fallback_reason": getattr(self.engine, "fallback_reason", None),
            "n_prompts": len(self.prompts),
            "n_failed": len(self.prompts) - len(ok_results),
            "avg_tokens_per_second": round(avg_tps, 2),
            "vram_peak_gb": final_snapshot.vram_peak_gb,
            "ram_used_gb": final_snapshot.ram_used_gb,
            "results": results,
        }
        return report

    @staticmethod
    def save_report(report: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        logger.info("Saved report to %s", path)

    @staticmethod
    def print_comparison_table(report_paths: list[str]) -> None:
        rows = [json.loads(Path(p).read_text()) for p in report_paths]
        print("| Precision | Device | Avg tok/s | VRAM peak (GB) | RAM (GB) | Fallback? | Failed |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            vram = r["vram_peak_gb"] if r["vram_peak_gb"] is not None else "N/A"
            fb = "yes" if r.get("auto_fallback_triggered") else "no"
            print(
                f"| {r['precision_label']} | {r.get('device', '?')} | {r['avg_tokens_per_second']} | "
                f"{vram} | {r['ram_used_gb']} | {fb} | {r['n_failed']}/{r['n_prompts']} |"
            )
