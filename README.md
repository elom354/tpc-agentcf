# TPC-AgentCF

TPC-AgentCF (Temporal Preference Conflict-Aware Popularity-Calibrated AgentCF++) is a research prototype for cross-domain agentic recommendation. The core question is whether group-level popularity signals remain reliable when a user's recent behavior diverges from their longer-term preferences. This project implements a clean reimplementation-oriented prototype from scratch, with automatic data download, mock-mode execution, baselines, ablations, and paper-ready outputs.

## Key Research Question

Can temporal preference conflict be detected in cross-domain recommendation, and can that signal be used to conditionally reduce popularity influence only when the user's preference state is unstable?

## Relation to AgentCF and AgentCF++

This project is inspired by the AgentCF and AgentCF++ papers as conceptual references only. The simplified AgentCF and AgentCF++ baselines here are modular reimplementations built from scratch for reproducible experimentation.

## How This Differs From Related Work

- DyTA4Rec studies temporal memory in a single-domain setting; this prototype focuses on cross-domain memory and conflict-conditioned popularity calibration.
- MACF uses sufficiency-based stopping in multi-agent collaboration; this prototype uses conflict-triggered single-pass escalation.
- VRec focuses on general verify-style recommendation reasoning; this prototype anchors explanations to memory evidence IDs.
- Popularity debiasing work typically applies unconditional debiasing; this prototype calibrates popularity only when conflict is detected.

## Main Contributions

- TPCD: Temporal Preference Conflict Detection from short-term versus long-term memory.
- CAPC: Conflict-Aware Popularity Calibration that reduces popularity influence conditionally.
- CTE: Conflict-Triggered Escalation for second-pass reranking in unstable preference states.
- Evidence-Anchored Explanations tied to explicit memory evidence IDs.

## Installation

```bash
conda env create -f environment.yml
conda activate tpc-agentcf
```

## Quickstart

```bash
python scripts/prepare_data.py --config config/default.yaml
python scripts/run_baselines.py --config config/default.yaml
python scripts/run_tpc_agentcf.py --config config/default.yaml
python scripts/run_ablation.py --config config/default.yaml
python scripts/make_paper_tables.py --config config/default.yaml
```

## Dataset Setup

Datasets are handled automatically. Run `prepare_data.py` and the code will download, cache, unzip, preprocess, and split the dataset without any manual file movement.

## Running Scripts

- `scripts/prepare_data.py`: download, preprocess, split, and summarize the dataset.
- `scripts/run_baselines.py`: run Pop, BPR-MF, AgentCF-simplified, and AgentCF++-simplified.
- `scripts/run_tpc_agentcf.py`: run the full model, write recommendation and conflict logs, and save metrics.
- `scripts/run_ablation.py`: run ablation and sensitivity variants.
- `scripts/make_paper_tables.py`: generate markdown tables and claim summaries from saved outputs.

## Understanding Outputs

- `outputs/metrics/*`: per-model CSV results for all users and conflict-based subsets.
- `outputs/explanations/recommendations.jsonl`: recommendation outputs with explanations and evidence IDs.
- `outputs/explanations/escalation_traces.jsonl`: escalation reranking traces.
- `outputs/conflicts/conflict_log.jsonl`: per-user conflict signals.
- `outputs/ablations/`: ablation summaries.
- `outputs/paper_assets/`: markdown tables and qualitative analysis snippets.

## How to Use Results in a Paper

Focus on `conflict_users` and `high_conflict_users` metrics. TPC-AgentCF's contribution is validated on these subsets rather than requiring uniform gains on all users.

## Limitations

- The default prototype is lightweight and prioritizes reproducibility over model scale.
- MovieLens domain labels are genre-derived approximations.
- Amazon rating-only data lacks titles and item text.
- Mock mode uses deterministic templates instead of actual LLM reasoning.

## Citation

- AgentCF: placeholder citation.
- AgentCF++: placeholder citation.
- TPC-AgentCF: placeholder citation.

> This is a research prototype. The AgentCF-simplified and AgentCF++-simplified baselines are reimplementations inspired by the original papers and are NOT official implementations.
