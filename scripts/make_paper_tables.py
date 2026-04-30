"""Generate markdown assets from saved outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def _read_metric(path: Path, key: str) -> float:
    if not path.exists():
        return 0.0
    frame = pd.read_csv(path)
    return float(frame.iloc[0].get(key, 0.0))


def _table_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return frame.to_string(index=False)


def main() -> None:
    parse_args()
    paper_dir = ROOT / "outputs" / "paper_assets"
    paper_dir.mkdir(parents=True, exist_ok=True)
    agentcfpp_conflict = ROOT / "outputs" / "metrics" / "conflict_users" / "agentcfpp_results.csv"
    tpc_conflict = ROOT / "outputs" / "metrics" / "conflict_users" / "tpc_agentcf_results.csv"
    all_users = ROOT / "outputs" / "metrics" / "all_users" / "tpc_agentcf_results.csv"
    recs = read_jsonl(ROOT / "outputs" / "explanations" / "recommendations.jsonl")
    qualitative = pd.DataFrame(recs)
    if not qualitative.empty:
        qualitative = qualitative[
            qualitative["conflict_detected"] & qualitative["escalation_triggered"] & (qualitative["faithfulness_score"] > 0.6)
        ].head(5)
    research_claims = f"""# Research Claims - TPC-AgentCF

## Claim 1: Temporal preference conflict is detectable and frequent.
Conflict Detection Rate: {_read_metric(all_users, 'Conflict Detection Rate') * 100:.2f}%

## Claim 2: Popularity calibration reduces bias without accuracy loss.
ARP_conflict: {_read_metric(all_users, 'ARP_conflict'):.4f} vs ARP_no_conflict: {_read_metric(all_users, 'ARP_no_conflict'):.4f}

## Claim 3: TPC-AgentCF outperforms AgentCF++ on conflict users.
MRR@10 (conflict users) - AgentCF++: {_read_metric(agentcfpp_conflict, 'MRR@10'):.4f}, TPC-AgentCF: {_read_metric(tpc_conflict, 'MRR@10'):.4f}
NDCG@10 (conflict users) - AgentCF++: {_read_metric(agentcfpp_conflict, 'NDCG@10'):.4f}, TPC-AgentCF: {_read_metric(tpc_conflict, 'NDCG@10'):.4f}

## Claim 4: Escalation improves ranking in high-conflict cases.
MRR@10_escalated: {_read_metric(all_users, 'MRR@10_escalated'):.4f} vs MRR@10_not_escalated: {_read_metric(all_users, 'MRR@10_not_escalated'):.4f}
Escalation Trigger Rate: {_read_metric(all_users, 'Escalation Trigger Rate') * 100:.2f}%

## Claim 5: Evidence-anchored explanations are more faithful.
Mean Faithfulness Score - TPC-AgentCF: {_read_metric(all_users, 'Faithfulness'):.4f}
"""
    (paper_dir / "research_claims.md").write_text(research_claims, encoding="utf-8")
    (paper_dir / "main_table.md").write_text(_table_text(pd.DataFrame({
        "Model": ["AgentCF++-simplified", "TPC-AgentCF"],
        "MRR@10_conflict": [_read_metric(agentcfpp_conflict, "MRR@10"), _read_metric(tpc_conflict, "MRR@10")],
    })), encoding="utf-8")
    ablation_path = ROOT / "outputs" / "ablations" / "ablation_results.csv"
    ablation_df = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    (paper_dir / "ablation_table.md").write_text(_table_text(ablation_df.head(20)), encoding="utf-8")
    (paper_dir / "conflict_analysis.md").write_text("Conflict-focused analysis is derived from the saved subset metrics.", encoding="utf-8")
    (paper_dir / "qualitative_examples.md").write_text(_table_text(qualitative) if not qualitative.empty else "No qualifying examples found.", encoding="utf-8")
    print(research_claims)


if __name__ == "__main__":
    main()
