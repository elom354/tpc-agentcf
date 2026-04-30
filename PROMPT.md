You are a senior machine learning research engineer specializing in
recommender systems and LLM-based agents. Your task is to build a
complete, clean, and reproducible research prototype from scratch.

The project name is:

    TPC-AgentCF
    (Temporal Preference Conflict-Aware Popularity-Calibrated AgentCF++)

============================================================
CRITICAL RULE — READ BEFORE WRITING ANY CODE
============================================================

DO NOT clone, fork, or copy any code from AgentCF or AgentCF++.
Their repositories are monolithic scripts designed for specific
closed LLM APIs. They are not suitable for direct reuse.

Instead:
- Read the AgentCF (WWW 2024) and AgentCF++ (SIGIR 2025) papers
  as conceptual references only.
- Build this project entirely from scratch as a clean modular
  Python package.
- The simplified AgentCF and AgentCF++ baselines in this project
  are your own reimplementations, clearly documented as such.

ALSO CRITICAL — DATASET DOWNLOAD:
All datasets must be downloaded AUTOMATICALLY by the code itself.
The user must NEVER manually download, unzip, or move any file.
The prepare_data.py script must handle everything:
download → unzip → preprocess → save.
If a file already exists locally, skip the download (cache it).

============================================================
SCIENTIFIC CONTEXT — WHAT IS AND IS NOT NOVEL
============================================================

The following ideas already exist and must NOT be claimed as
new contributions:

1. Short-term / long-term memory for LLM recommendation agents:
   DyTA4Rec (CIKM 2025) already proposes temporal-aware user
   behavior simulation with dynamic memory update in single-domain.

2. Multi-agent debate / discussion for recommendation:
   MACRec (SIGIR 2024) and MACF (arXiv 2511.18413) already propose
   multi-agent collaboration with role-specialized agents and
   stopping conditions.

3. Faithful / verifiable explanations for LLM recommenders:
   VRec (arXiv 2603.07725) proposes reason-verify-recommend.
   RobustExplain (WWW 2026) evaluates LLM explanation robustness.

4. Popularity bias in LLM-based recommenders:
   Lichtenberg et al. (2024) and Bridging Semantic Understanding
   and Popularity Bias with LLMs (WWW 2026) study this.

5. Uncertainty quantification for LLM recommendation:
   Kweon et al. (WWW 2025) quantify and decompose uncertainty
   in LLM-based recommendation.

THE ACTUAL NOVEL CONTRIBUTION OF THIS PROJECT:

   "In the cross-domain agentic recommendation setting of
   AgentCF++, when a user's short-term preferences diverge
   from their long-term preferences (temporal preference
   conflict), group popularity signals become unreliable
   and risk amplifying the wrong recommendation. We detect
   this conflict explicitly and use it to calibrate the
   weight of popularity signals conditionally — reducing
   popularity influence precisely when the user's preference
   state is most unstable."

Key distinctions:
- Unlike DyTA4Rec: we work in CROSS-DOMAIN settings (AgentCF++
  style with dual-layer domain memory), not single-domain.
  DyTA4Rec does not address cross-domain preference conflict.
- Unlike MACF: our escalation trigger is specifically the
  CONFLICT between temporal memory layers, not a generic
  sufficiency test on agent consensus.
- Unlike popularity debiasing papers: we do not always reduce
  popularity. We calibrate it CONDITIONALLY on detected conflict.
- Unlike VRec: faithfulness is tied to structured memory
  evidence IDs from the dual-layer memory architecture.

============================================================
1. PYTHON VERSION
============================================================

Use Python 3.11 exclusively.
Do NOT use Python 3.12, 3.13, or 3.14.
Reason: PyTorch, Transformers, and SentenceTransformers are most
stable on Python 3.11 as of 2025.

============================================================
2. TWO EXECUTION MODES
============================================================

The project must run in two modes:

MODE 1 — Mock mode (DEFAULT, no API key needed):
- All LLM calls use deterministic string templates.
- Runs entirely on CPU.
- SentenceTransformer embeddings are optional.
  If not available, fall back to TF-IDF cosine similarity.
- Fast: completes on a laptop in under 10 minutes.
- Used for: testing, CI, Colab quickstart, ablations.

MODE 2 — LLM mode (OPTIONAL):
- Activated by setting OPENAI_API_KEY in environment.
- Reads OPENAI_BASE_URL optionally (supports DeepSeek, etc.)
- If API key is missing, falls back to mock mode gracefully.
- Must NEVER crash if the key is absent.

Default in config: llm.backend = "mock"

============================================================
3. REPOSITORY STRUCTURE
============================================================

Create exactly this structure:

TPC-AgentCF/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── config/
│   └── default.yaml
├── data/
│   ├── README.md
│   ├── raw/                  # auto-downloaded files go here
│   └── processed/            # preprocessed CSV files go here
├── notebooks/
│   └── colab_quickstart.ipynb
├── outputs/
│   ├── README.md
│   ├── metrics/
│   │   ├── all_users/
│   │   ├── no_conflict_users/
│   │   ├── conflict_users/
│   │   └── high_conflict_users/
│   ├── explanations/
│   ├── conflicts/
│   ├── ablations/
│   └── paper_assets/
├── scripts/
│   ├── prepare_data.py
│   ├── run_baselines.py
│   ├── run_tpc_agentcf.py
│   ├── run_ablation.py
│   └── make_paper_tables.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py       # ALL download logic here
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── splits.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── mock_llm.py
│   │   └── openai_compatible.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── evidence.py
│   │   ├── short_term.py
│   │   ├── long_term.py
│   │   ├── domain_memory.py
│   │   ├── group_memory.py
│   │   └── conflict_detector.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── user_agent.py
│   │   ├── item_agent.py
│   │   └── escalation_agent.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── popularity.py
│   │   ├── bpr_mf.py
│   │   ├── agentcf_baseline.py
│   │   ├── agentcfpp_baseline.py
│   │   └── tpc_agentcf.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ranking_metrics.py
│   │   ├── diversity_metrics.py
│   │   ├── faithfulness_metrics.py
│   │   ├── conflict_metrics.py
│   │   └── report.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── seed.py
│       └── io.py
└── tests/
    ├── test_downloader.py
    ├── test_memory.py
    ├── test_conflict_detector.py
    ├── test_metrics.py
    └── test_pipeline.py

============================================================
4. AUTOMATIC DATASET DOWNLOAD — src/data/downloader.py
============================================================

THIS IS CRITICAL. The user never downloads anything manually.

Implement a DatasetDownloader class in src/data/downloader.py
with the following behavior for each dataset:

------------------------------------------------------------
4.1 MovieLens Latest Small (smoke-test dataset)
------------------------------------------------------------

URL:
  https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Behavior:
  1. Check if data/raw/ml-latest-small/ already exists.
  2. If yes: skip download (cached).
  3. If no:
     a. Download zip to data/raw/ml-latest-small.zip
        using requests with a progress bar (tqdm).
     b. Unzip into data/raw/ml-latest-small/.
     c. Delete the zip file.
  4. Return the path to data/raw/ml-latest-small/.

Files used:
  - ratings.csv (userId, movieId, rating, timestamp)
  - movies.csv  (movieId, title, genres)

------------------------------------------------------------
4.2 Amazon Reviews 5-core Cross-Domain (paper dataset)
------------------------------------------------------------

Source: Amazon Reviews 2023 (McAuley Lab)
Use the 5-core processed files from:
  https://amazon-reviews-2023.github.io

Download these two domain pairs (Books + Movies_and_TV):

Books ratings:
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz

Movies ratings:
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Movies_and_TV.csv.gz

Behavior for each file:
  1. Check if data/raw/amazon/<domain>.csv already exists.
  2. If yes: skip download.
  3. If no:
     a. Download the .csv.gz file using requests + tqdm.
     b. Decompress the .gz file into data/raw/amazon/<domain>.csv.
     c. Delete the .gz file.

The CSV columns are: user_id, item_id, rating, timestamp.

After download, the preprocess step will:
  - Assign domain label "books" or "movies" based on source file.
  - Merge both into one interactions.csv with a domain column.
  - Filter to users who appear in BOTH domains (cross-domain users).
  - Apply min_interactions_per_user filter.
  - Subsample to max_users if needed.

If download fails (network error, URL changed):
  - Log a clear warning.
  - Fall back to MovieLens automatically.
  - Do NOT crash.

------------------------------------------------------------
4.3 Dataset selection in config
------------------------------------------------------------

In config/default.yaml:

data:
  dataset: "movielens"   # "movielens" or "amazon"

When dataset = "movielens": download MovieLens only.
When dataset = "amazon": download Amazon Books + Movies only.

The rest of the pipeline (preprocess, split, evaluation) must
work identically regardless of which dataset is selected.

============================================================
5. DATA PREPROCESSING — src/data/preprocess.py
============================================================

After download, produce two CSV files in data/processed/:

interactions.csv columns:
  user_id, item_id, rating, timestamp, domain, split

items.csv columns:
  item_id, title, domain, description

For MovieLens:
  - Domain from genres using this mapping:
      contains "Drama"                 → "drama"
      contains "Comedy"                → "comedy"
      contains "Action" or "Thriller"  → "action"
      contains "Romance"               → "romance"
      anything else                    → "other"
  - A user has multiple domains if their items span multiple genre groups.
  - Positive interaction: rating >= 4.0
  - description = title + " [" + genres + "]"

For Amazon:
  - Domain = source file label ("books" or "movies")
  - Positive interaction: rating >= 4.0
  - description = item_id (title not available in 5-core rating-only)

Cross-domain user filter:
  - For Amazon: keep only users with interactions in BOTH domains.
  - For MovieLens: keep users with interactions in at least 2 domains.
  - Log how many users passed this filter.

Chronological split:
  - Sort each user's interactions by timestamp ascending.
  - Train: first 80%
  - Validation: next 10%
  - Test: last 10%
  - Minimum interactions to split: 10 per user.

Candidate set construction (for evaluation):
  - For each test interaction: 1 positive + N random negatives.
  - candidate_sample_size configurable (default 50).
  - Negatives: items the user has NOT interacted with.
  - Random seed fixed.

Output: data/processed/interactions.csv and data/processed/items.csv

============================================================
6. EVIDENCE DATACLASS — src/memory/evidence.py
============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class Evidence:
    evidence_id: str
    user_id: str
    item_id: str
    domain: str
    timestamp: float
    evidence_type: Literal[
        "short_term", "long_term", "domain", "group"
    ]
    text: str
    embedding: List[float]
    strength: float
    recency_weight: float   # exp(-lambda_decay * time_delta)
    source: Literal[
        "interaction", "reflection", "group_memory", "consolidation"
    ]
    metadata: dict = field(default_factory=dict)

@dataclass
class ConflictSignal:
    user_id: str
    domain: str
    is_conflict: bool
    conflict_score: float           # 0.0 to 1.0
    centroid_distance: float
    short_term_summary: str
    long_term_summary: str
    conflict_explanation: str
    short_term_evidence_ids: List[str]
    long_term_evidence_ids: List[str]
    timestamp: float

@dataclass
class RecommendationOutput:
    user_id: str
    item_id: str
    rank: int
    score: float
    domain: str
    explanation: str
    evidence_ids: List[str]
    evidence_types_used: List[str]
    conflict_detected: bool
    conflict_score: float
    conflict_discount_applied: float
    popularity_percentile: float
    popularity_overridden: bool
    escalation_triggered: bool
    faithfulness_score: float
    metadata: dict = field(default_factory=dict)

============================================================
7. MEMORY MODULES
============================================================

------------------------------------------------------------
7.1 ShortTermMemory — src/memory/short_term.py
------------------------------------------------------------

- Stores the last N interactions per user per domain.
  N = short_term_window (default 15, configurable).
- Each entry is an Evidence object with evidence_type="short_term".
- When a new interaction arrives:
    If size >= N: remove oldest entry.
    Add new entry.
- Recency weight: exp(-lambda_decay * (now - timestamp))
- Provides:
    get_evidence(user_id, domain) -> List[Evidence]
    get_centroid(user_id, domain) -> List[float]

------------------------------------------------------------
7.2 LongTermMemory — src/memory/long_term.py
------------------------------------------------------------

- Stores preferences that have appeared at least K times.
  K = long_term_min_support (default 3, configurable).
- Updated by consolidation from ShortTermMemory.
- Consolidation logic:
    Extract repeated preference patterns from short-term memory.
    If a preference text has cosine similarity > sim_threshold
    to an existing long-term entry: increment its support count.
    If support_count >= K: promote to long-term.
- Evidence type: "long_term"
- Recency weight lower decay (lambda * 0.3 for long-term)
- Provides:
    get_evidence(user_id, domain) -> List[Evidence]
    get_centroid(user_id, domain) -> List[float]
    consolidate(user_id, domain, short_term_evidence)

------------------------------------------------------------
7.3 DomainMemory — src/memory/domain_memory.py
------------------------------------------------------------

Implements AgentCF++-style dual-layer domain memory.

domain_separated_memory:
    Dict[user_id, Dict[domain, List[Evidence]]]
    Stores preferences independently per domain.
    Used for domain-specific decisions.

domain_fused_memory:
    Dict[user_id, Dict[target_domain, List[Evidence]]]
    Stores cross-domain preferences fused into the target domain.
    Built by a 2-step fusion:
    Step 1: Extract relevant cross-domain preferences.
      Similarity between source domain evidence and
      target domain items must exceed cross_domain_threshold.
    Step 2: Integrate into target domain with a fusion weight.
      fused_weight = base_weight * cross_domain_sim

Provides:
    get_domain_evidence(user_id, domain) -> List[Evidence]
    get_fused_evidence(user_id, target_domain) -> List[Evidence]
    update(user_id, item_id, domain, interaction_text)

------------------------------------------------------------
7.4 GroupMemory — src/memory/group_memory.py
------------------------------------------------------------

Implements AgentCF++-style group-shared memory.

Group assignment:
  1. Compute user profile embedding = mean of all user evidence
     embeddings.
  2. Cluster users with KMeans (k=num_groups, default 5).
  3. Assign each user to a group at preprocessing time.
  4. Re-cluster every reclustering_interval interactions
     (default: never re-cluster in prototype — cluster once).

Group memory storage:
  Dict[group_id, Dict[domain, List[Evidence]]]
  Stores recent interactions of all group members.
  Max size per group per domain: group_window (default 50).

Provides:
    get_group_evidence(user_id, domain) -> List[Evidence]
    get_group_centroid(user_id, domain) -> List[float]
    update(user_id, item_id, domain, interaction_text)
    get_group_id(user_id) -> int

------------------------------------------------------------
7.5 ConflictDetector — src/memory/conflict_detector.py
------------------------------------------------------------

This is the central novel component. Implement carefully.

Input:
    user_id: str
    domain: str
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory
    llm_client: LLMClient

Output: ConflictSignal

Algorithm:
    1. Get short-term evidence for user in domain.
       If fewer than min_st_size (default 3): return no conflict.
    2. Get long-term evidence for user in domain.
       If empty: return no conflict.
    3. Compute short-term centroid = mean of embeddings.
    4. Compute long-term centroid = mean of embeddings.
    5. Compute cosine distance between centroids.
    6. is_conflict = (distance > conflict_threshold)
    7. conflict_score = min(1.0, distance / max_distance)
    8. Generate short_term_summary and long_term_summary
       using llm_client.generate() with a simple prompt.
    9. Generate conflict_explanation if is_conflict.
    10. Return ConflictSignal.

Embedding backend:
    If embedding_backend = "sentence-transformers":
        Use SentenceTransformer(embedding_model)
    Else (default = "tfidf"):
        Use TfidfVectorizer fitted on all evidence texts.
        Use cosine_similarity from sklearn.

IMPORTANT: The ConflictDetector must be callable per user per
domain per recommendation batch. It is NOT called per interaction
(that would be too slow). Call it once per user at inference time.

============================================================
8. AGENTS — src/agents/
============================================================

------------------------------------------------------------
8.1 BaseAgent — base_agent.py
------------------------------------------------------------

Abstract class with:
    agent_id: str
    llm_client: LLMClient
    memory: dict  # references to relevant memory modules

    def generate_response(self, prompt: str) -> str
    def build_prompt(self, context: dict) -> str

------------------------------------------------------------
8.2 UserAgent — user_agent.py
------------------------------------------------------------

For a given user, the UserAgent:
    1. Retrieves domain evidence from DomainMemory.
    2. Retrieves short-term evidence from ShortTermMemory.
    3. Retrieves long-term evidence from LongTermMemory.
    4. Retrieves group evidence from GroupMemory.
    5. Calls ConflictDetector to get ConflictSignal.
    6. For each candidate item:
       a. Computes base_score using evidence similarity.
       b. Applies CAPC (see section 9).
       c. Returns scored candidate list.

After ranking:
    7. Checks escalation trigger conditions (section 10).
    8. If triggered: calls EscalationAgent.
    9. Returns final RecommendationOutput list.

Update after interaction (reflection):
    - If recommended item matches actual interaction: no update.
    - If mismatch: generate a reflection text via LLM.
    - Update short-term memory and domain memory.

------------------------------------------------------------
8.3 ItemAgent — item_agent.py
------------------------------------------------------------

For a given item, the ItemAgent:
    - Stores item metadata (title, domain, description).
    - Maintains a list of user evidence entries from users
      who have interacted with it.
    - Provides: get_item_profile() -> str
    - Used by UserAgent to compute base_score.

------------------------------------------------------------
8.4 EscalationAgent — escalation_agent.py
------------------------------------------------------------

A lightweight second-pass reasoning module.
NOT a multi-agent debate.

Input:
    user_id: str
    candidate_list: List[Tuple[item_id, score]]
    conflict_signal: ConflictSignal
    domain_evidence: List[Evidence]
    group_evidence: List[Evidence]

Behavior:
    1. Build a prompt that includes:
       - The conflict description (short-term vs long-term).
       - The candidate list with their scores.
       - Explicit instruction: "Re-rank these candidates
         prioritizing authentic long-term preferences over
         recent trends and popularity."
    2. Call llm_client.generate() once.
    3. Parse the re-ranked list from the response.
    4. Return EscalationOutput with:
       - reranked_list: List[item_id]
       - reasoning: str
       - evidence_ids_used: List[str]
       - long_term_preference_used: bool
       - short_term_preference_used: bool
       - popularity_overridden: bool

In mock mode:
    - Simply re-rank by giving a bonus to items whose embeddings
      are closer to the long-term memory centroid.
    - Generate a template reasoning string.

This is distinct from MACF which uses a multi-round discussion
with dynamic agent recruitment. EscalationAgent makes a single
targeted re-ranking call triggered by temporal conflict.

============================================================
9. CAPC — CONFLICT-AWARE POPULARITY CALIBRATION
============================================================

Implement in src/models/popularity.py

Step 1 — Compute item popularity at training time:
    raw_pop(item) = interactions(item) / max_item_interactions
    pop_percentile(item) = percentile rank among all items

Step 2 — Compute useful_popularity_score:
    group_centroid = GroupMemory.get_group_centroid(user, domain)
    long_term_centroid = LongTermMemory.get_centroid(user, domain)
    alignment = cosine_similarity(group_centroid, long_term_centroid)
    If alignment > alignment_threshold:
        useful_popularity_score = raw_pop(item)
    Else:
        useful_popularity_score = raw_pop(item) * alignment

Step 3 — Compute popularity_penalty:
    penalty = max(0, pop_percentile(item) - head_threshold)

Step 4 — Compute conflict_discount:
    If is_conflict:
        discount = conflict_score * max_discount
    Else:
        discount = 0.0

Step 5 — Final score:
    score =
        base_score
      + alpha_temporal * temporal_score
      + alpha_group * group_score * (1 - discount)
      + alpha_pop * useful_popularity_score * (1 - discount)
      - beta_pop_penalty * popularity_penalty * is_conflict

Where:
    base_score = cosine_similarity(user_embedding, item_embedding)
    temporal_score = weighted sum of short+long term evidence sim
    group_score = cosine_similarity(group_centroid, item_embedding)
    is_conflict = 1 if ConflictSignal.is_conflict else 0

All alpha, beta, discount_max configurable in default.yaml.

============================================================
10. ESCALATION TRIGGER CONDITIONS
============================================================

Trigger the EscalationAgent if ANY of these is true:

    Condition 1: conflict_score > escalation_threshold
                 AND is_conflict = True

    Condition 2: top-1 item pop_percentile > high_pop_threshold
                 AND is_conflict = True

    Condition 3: abs(score[rank=1] - score[rank=2]) < score_margin
                 AND is_conflict = True

If none: escalation_triggered = False, keep original ranking.

Track globally:
    - Total escalations triggered
    - Escalations triggered by each condition
    These must appear in conflict_metrics output.

============================================================
11. BASELINES — src/models/
============================================================

------------------------------------------------------------
11.1 Pop (popularity.py)
------------------------------------------------------------
Rank all candidates by raw_pop(item) descending.
No user personalization.

------------------------------------------------------------
11.2 BPR-MF (bpr_mf.py)
------------------------------------------------------------
Implement Bayesian Personalized Ranking with matrix factorization.
Use PyTorch. No GPU required.

Parameters (configurable):
    latent_dim: 32
    lr: 0.01
    weight_decay: 1e-4
    epochs: 20
    batch_size: 256

Loss: BPR loss on (user, positive_item, negative_item) triples.
Evaluation: rank candidates by predicted score.

------------------------------------------------------------
11.3 AgentCF-simplified (agentcf_baseline.py)
------------------------------------------------------------

Reimplementation of AgentCF core logic. Clearly documented
as a reimplementation, not the official code.

- User agent with FLAT memory (no short/long split, no domain split).
- Item agent with flat memory.
- Collaborative reflection: if prediction mismatches ground truth,
  generate a reflection text and update memory.
- No cross-domain logic.
- No popularity calibration.
- No conflict detection.
- Scoring: cosine similarity between user embedding and item embedding.

------------------------------------------------------------
11.4 AgentCF++-simplified (agentcfpp_baseline.py)
------------------------------------------------------------

Reimplementation of AgentCF++ core logic.

- User agent with domain-separated memory (per domain).
- User agent with domain-fused memory (2-step fusion).
- Group memory with KMeans clustering.
- Group popularity signal in scoring.
- Collaborative reflection on mismatch.
- NO temporal conflict detection.
- NO popularity calibration based on conflict.
- NO escalation.

This is the most important baseline for comparison.

------------------------------------------------------------
11.5 TPC-AgentCF (tpc_agentcf.py)
------------------------------------------------------------

The full proposed model. Integrates:
- ShortTermMemory + LongTermMemory
- DomainMemory (AgentCF++ style)
- GroupMemory (AgentCF++ style)
- ConflictDetector (TPCD)
- CAPC scoring
- EscalationAgent (CTE)
- Evidence-anchored explanations

============================================================
12. EVALUATION — src/evaluation/
============================================================

------------------------------------------------------------
12.1 Ranking metrics (ranking_metrics.py)
------------------------------------------------------------
For K in [5, 10]:
    MRR@K, NDCG@K, HitRate@K, Recall@K

------------------------------------------------------------
12.2 Diversity metrics (diversity_metrics.py)
------------------------------------------------------------
    ARP: Average Recommendation Popularity
    HIR: Head Item Ratio (fraction in top-20% popular items)
    LTC: Long Tail Coverage
    Novelty: mean(-log2(raw_pop(item)))

------------------------------------------------------------
12.3 Faithfulness metrics (faithfulness_metrics.py)
------------------------------------------------------------
    Faithfulness Score:
        For each explanation:
        1. Split into sentences.
        2. For each sentence: max cosine sim with evidence texts.
        3. Supported if max_sim > faithfulness_sim_threshold.
        4. score = supported / total
    Mean Faithfulness Score across all users.
    Unsupported Claim Ratio: 1 - faithfulness_score.

------------------------------------------------------------
12.4 Conflict metrics (conflict_metrics.py)
------------------------------------------------------------
    Conflict Detection Rate: fraction of users with conflict
    Mean Conflict Score
    ARP_conflict: ARP for conflict users
    ARP_no_conflict: ARP for no-conflict users
    Escalation Trigger Rate
    MRR@10_escalated: for escalated recommendations only
    MRR@10_not_escalated: for non-escalated only

------------------------------------------------------------
12.5 MANDATORY: SPLIT EVALUATION BY CONFLICT STATUS
------------------------------------------------------------

For EVERY model and EVERY metric, compute and save results
separately for these four user groups:

    Group 1 — ALL USERS (full test set)
    Group 2 — NO-CONFLICT USERS (conflict_score < conflict_threshold)
    Group 3 — CONFLICT USERS (conflict_score >= conflict_threshold)
    Group 4 — HIGH-CONFLICT USERS (conflict_score >= 0.5)

Note: For baselines (Pop, BPR-MF, AgentCF-simplified), conflict
groups are defined using the conflict scores computed by TPC-AgentCF
on the same users (since baselines do not run ConflictDetector).
This allows fair subset comparison.

Save to:
    outputs/metrics/all_users/<model>_results.csv
    outputs/metrics/no_conflict_users/<model>_results.csv
    outputs/metrics/conflict_users/<model>_results.csv
    outputs/metrics/high_conflict_users/<model>_results.csv

THE PRIMARY PAPER CLAIMS ARE SUPPORTED BY:
    conflict_users/ and high_conflict_users/ results.

TPC-AgentCF does NOT need to outperform AgentCF++ on all users.
It must outperform on conflict and high-conflict user subsets.
This is expected and acceptable — document it in the README.

============================================================
13. ABLATION STUDY
============================================================

Run these configurations:

A0: AgentCF++-simplified (full baseline, no TPC components)
A1: A0 + temporal memory only (short+long, no conflict detection)
A2: A0 + temporal memory + conflict detection (no CAPC)
A3: A0 + temporal memory + conflict detection + CAPC
A4: A3 + escalation agent
A5: Full TPC-AgentCF (all components, same as run_tpc_agentcf.py)

Sensitivity tests:
    S1: A3 with conflict_discount_max = 0.0 (CAPC disabled at conflict)
    S2: A3 with conflict_discount_max = 1.0 (full suppression at conflict)
    S3: A4 with escalation always ON (trigger_only = false)
    S4: A4 with escalation always OFF

For each ablation variant, save:
    outputs/ablations/<variant>_results.csv

Each file contains all 4 user groups × all metrics.

Generate:
    outputs/ablations/ablation_results.csv  (combined)

============================================================
14. CONFIGURATION FILE — config/default.yaml
============================================================

project:
  name: "TPC-AgentCF"
  seed: 42
  output_dir: "outputs"

data:
  dataset: "movielens"         # "movielens" or "amazon"
  max_users: 100
  max_items: 500
  min_interactions_per_user: 10
  candidate_sample_size: 50

memory:
  short_term_window: 15
  long_term_min_support: 3
  lambda_decay: 0.05
  embedding_backend: "tfidf"   # "tfidf" or "sentence-transformers"
  embedding_model: "all-MiniLM-L6-v2"
  sim_threshold: 0.35

conflict:
  enabled: true
  conflict_threshold: 0.35
  max_distance: 1.0
  min_short_term_size: 3
  escalation_threshold: 0.50
  score_margin: 0.05

group_memory:
  enabled: true
  num_groups: 5
  group_window: 50

popularity:
  head_percentile: 0.80
  alignment_threshold: 0.40
  conflict_discount_max: 0.60
  beta_pop_penalty: 0.20
  high_pop_threshold: 0.90

scoring:
  alpha_temporal: 0.30
  alpha_group: 0.20
  alpha_pop: 0.15

escalation:
  enabled: true
  trigger_only: true

llm:
  backend: "mock"              # "mock" or "openai"
  temperature: 0.0
  max_tokens: 256

evaluation:
  k_values: [5, 10]
  faithfulness_sim_threshold: 0.35
  run_ablations: true

============================================================
15. LLM BACKENDS — src/llm/
============================================================

llm_client.py — Abstract base:
    class LLMClient(ABC):
        def generate(self, prompt: str, **kwargs) -> str: ...

mock_llm.py — MockLLM:
    - Deterministic. Uses random.seed from project config.
    - Template-based generation for each agent type:
      * preference_statement: "User prefers {genre} content
        based on {N} recent interactions in {domain}."
      * item_profile: "{title} is a {domain} item with
        characteristics matching {keywords}."
      * conflict_explanation: "Short-term preferences lean
        toward {st_theme} while long-term preferences favor
        {lt_theme}."
      * escalation_reasoning: "Given temporal preference
        conflict, prioritizing long-term stable preferences
        over current popularity trend."
      * reflection: "The recommendation did not match.
        Updating preference toward {correct_domain}."
    - All template variables derived from actual data.
    - Mock re-ranking: items sorted by cosine similarity
      to long-term memory centroid.

openai_compatible.py — OpenAICompatibleLLM:
    - Reads OPENAI_API_KEY from os.environ.
    - Reads OPENAI_BASE_URL from os.environ (optional).
    - If key missing: log warning, fall back to MockLLM.
    - Uses openai Python client.
    - Supports any OpenAI-compatible endpoint (DeepSeek, etc.)

============================================================
16. SCRIPTS
============================================================

------------------------------------------------------------
scripts/prepare_data.py
------------------------------------------------------------

Usage:
    python scripts/prepare_data.py --config config/default.yaml
    python scripts/prepare_data.py --config config/default.yaml
                                   --dataset amazon

Steps (all automatic, no user action needed):
    1. Read config to determine dataset.
    2. Call DatasetDownloader to download and unzip.
       Show progress bar. Print "Already cached, skipping."
       if file exists.
    3. Call preprocess pipeline.
    4. Apply cross-domain user filter.
    5. Create chronological split.
    6. Build candidate sets.
    7. Save data/processed/interactions.csv and items.csv.
    8. Print summary: N users, N items, N interactions,
       N cross-domain users, conflict rate estimate.

------------------------------------------------------------
scripts/run_baselines.py
------------------------------------------------------------

Usage:
    python scripts/run_baselines.py --config config/default.yaml

Steps:
    1. Load processed data.
    2. Run Pop → save metrics.
    3. Run BPR-MF → save metrics.
    4. Run AgentCF-simplified → save metrics.
    5. Run AgentCF++-simplified → save metrics.
    6. Print comparison table.

------------------------------------------------------------
scripts/run_tpc_agentcf.py
------------------------------------------------------------

Usage:
    python scripts/run_tpc_agentcf.py --config config/default.yaml

Steps:
    1. Load processed data.
    2. Initialize all memory modules.
    3. Precompute user group assignments (KMeans).
    4. Precompute item popularity scores.
    5. For each user in test set:
       a. Run ConflictDetector.
       b. Score candidates with CAPC.
       c. Check escalation triggers.
       d. If triggered: run EscalationAgent.
       e. Save RecommendationOutput to JSONL.
       f. Save ConflictSignal to JSONL.
    6. Compute all metrics for all 4 user groups.
    7. Save metrics CSVs.
    8. Save explanations JSONL.
    9. Print summary results table.

------------------------------------------------------------
scripts/run_ablation.py
------------------------------------------------------------

Usage:
    python scripts/run_ablation.py --config config/default.yaml

Steps:
    1. Run A0 through A5 and S1 through S4.
    2. For each variant: compute all metrics, all 4 user groups.
    3. Save outputs/ablations/<variant>_results.csv.
    4. Merge into outputs/ablations/ablation_results.csv.
    5. Print ablation table.

------------------------------------------------------------
scripts/make_paper_tables.py
------------------------------------------------------------

Usage:
    python scripts/make_paper_tables.py --config config/default.yaml

Generates:
    outputs/paper_assets/main_table.md
    outputs/paper_assets/ablation_table.md
    outputs/paper_assets/conflict_analysis.md
    outputs/paper_assets/qualitative_examples.md
    outputs/paper_assets/research_claims.md

For qualitative_examples.md: find 5 users where:
    - is_conflict = True
    - escalation_triggered = True
    - faithfulness_score > 0.6
Show: user_id, domain pair, conflict summary, recommendation,
explanation, evidence used, conflict_score, pop_percentile,
whether popularity was overridden.

============================================================
17. research_claims.md — AUTO-GENERATED CONTENT
============================================================

This file is generated by make_paper_tables.py.
It reads actual numeric results and fills in this template:

---
# Research Claims — TPC-AgentCF

## Claim 1: Temporal preference conflict is detectable and frequent.
Conflict Detection Rate: {value}%
Source: conflict_metrics on all users.
Support: A0 vs A2 on MRR@10 and conflict user subset.

## Claim 2: Popularity calibration reduces bias without accuracy loss.
ARP_conflict: {value} vs ARP_no_conflict: {value}
HIR_conflict: {value} vs HIR_baseline: {value}
Source: A2 vs A3 on conflict_users metrics.
Support: Lower ARP and HIR with maintained or improved MRR@10.

## Claim 3: TPC-AgentCF outperforms AgentCF++ on conflict users.
MRR@10 (conflict users) — AgentCF++: {val}, TPC-AgentCF: {val}
NDCG@10 (conflict users) — AgentCF++: {val}, TPC-AgentCF: {val}
Source: main_results_conflict_users.csv
Note: Performance on no-conflict users may be similar.
This is expected and acceptable.

## Claim 4: Escalation improves ranking in high-conflict cases.
MRR@10_escalated: {val} vs MRR@10_not_escalated: {val}
Escalation Trigger Rate: {val}%
Source: A4 vs A3 on escalated subset.

## Claim 5: Evidence-anchored explanations are more faithful.
Mean Faithfulness Score — AgentCF++: {val}, TPC-AgentCF: {val}
Source: faithfulness_results on all users.
---

============================================================
18. CONDA ENVIRONMENT — environment.yml
============================================================

name: tpc-agentcf
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - pyyaml
  - tqdm
  - matplotlib
  - jupyter
  - ipykernel
  - pip:
      - torch
      - sentence-transformers
      - datasets
      - openai
      - rich
      - requests

============================================================
19. GOOGLE COLAB NOTEBOOK — notebooks/colab_quickstart.ipynb
============================================================

The notebook must be fully self-contained.
The user clicks "Run All" and it works. No manual steps.

Structure:

Cell 1 — Install:
    !pip install -r requirements.txt

Cell 2 — Prepare data (MovieLens, automatic):
    !python scripts/prepare_data.py --config config/default.yaml
    Show: N users, N items, N cross-domain users.

Cell 3 — Run baselines:
    !python scripts/run_baselines.py --config config/default.yaml
    Display results as pandas DataFrame.

Cell 4 — Run TPC-AgentCF:
    !python scripts/run_tpc_agentcf.py --config config/default.yaml
    Display results as pandas DataFrame.

Cell 5 — Run ablations:
    !python scripts/run_ablation.py --config config/default.yaml
    Display ablation table as DataFrame.

Cell 6 — Qualitative examples:
    Load outputs/explanations/recommendations.jsonl
    Filter: is_conflict=True AND escalation_triggered=True
    Display 5 examples as formatted pandas rows.
    Show: user_id, domain, conflict_score,
          recommended_item, explanation, faithfulness_score.

Cell 7 — Conflict analysis plots:
    Plot 1: Distribution of conflict_score across all users
            (histogram with threshold line).
    Plot 2: ARP_conflict vs ARP_no_conflict (bar chart).
    Plot 3: MRR@10 by user group (all / no-conflict /
            conflict / high-conflict) for TPC-AgentCF
            vs AgentCF++-simplified side-by-side.

Cell 8 — Generate paper tables:
    !python scripts/make_paper_tables.py
           --config config/default.yaml
    Print content of research_claims.md.

All outputs saved to /content/outputs/ in Colab.
No GPU required. Runs in mock mode by default.

============================================================
20. README.md — REQUIRED CONTENT
============================================================

Write a complete README.md in English with these sections:

1. Title + one-paragraph abstract
2. Key research question
3. Relation to AgentCF and AgentCF++
4. How this differs from related work:
   - DyTA4Rec: single-domain, no cross-domain, no CAPC
   - MACF: sufficiency-based stopping, not conflict-triggered
   - VRec: general verify, not memory-evidence-anchored
   - Popularity debiasing papers: unconditional debiasing,
     not conflict-conditional calibration
5. Main contributions (TPCD, CAPC, CTE, Evidence-Anchored)
6. Installation:
   conda env create -f environment.yml
   conda activate tpc-agentcf
7. Quickstart (5 commands, one per line, copy-paste ready)
8. Dataset setup (automatic — "just run prepare_data.py")
9. Running scripts (all 5 scripts)
10. Understanding outputs (all output files explained)
11. How to use results in a paper:
    "Focus on conflict_users and high_conflict_users metrics.
     TPC-AgentCF's contribution is validated on these subsets."
12. Limitations
13. Citation placeholders for AgentCF, AgentCF++, this project

Include this disclaimer:

"This is a research prototype. The AgentCF-simplified and
AgentCF++-simplified baselines are reimplementations inspired
by the original papers and are NOT official implementations."

============================================================
21. TESTS — tests/
============================================================

test_downloader.py:
    - MovieLens downloader returns a valid directory.
    - Re-running download uses cache (no re-download).
    - Amazon downloader gracefully fails if URL unreachable.

test_memory.py:
    - Evidence insertion and retrieval work.
    - ShortTermMemory respects window size.
    - LongTermMemory consolidation promotes after K supports.
    - Recency decay is correctly applied.

test_conflict_detector.py:
    - No conflict when short-term is empty.
    - No conflict when long-term is empty.
    - Conflict detected when embeddings diverge sufficiently.
    - No conflict when embeddings are similar.
    - ConflictSignal fields are all populated.

test_metrics.py:
    - MRR, NDCG, HitRate correct on toy data.
    - ARP and HIR correct on toy data.
    - Faithfulness score correct on simple example.

test_pipeline.py:
    - End-to-end test: 5 users, 20 items, mock mode.
    - All 4 user group CSV files are generated.
    - RecommendationOutput JSONL is generated.
    - ConflictSignal JSONL is generated.

============================================================
22. CODING STANDARDS
============================================================

- Python 3.11 type hints on every function and class.
- Dataclasses for Evidence, ConflictSignal, RecommendationOutput.
- No hardcoded paths. All paths derived from config output_dir.
- No hardcoded API keys. Keys from environment only.
- Default mode is mock. System works without any API key.
- Reproducibility: set random seeds in utils/seed.py,
  call at the start of every script.
- No print() in src/. Use logging.getLogger(__name__).
- Use tqdm for all loops over users or items in scripts.
- UTF-8 encoding for all file writes.
- JSONL format for all streaming outputs (one JSON per line).
- Docstrings on all public classes and methods.

============================================================
23. FINAL VERIFICATION CHECKLIST
============================================================

After implementation, the following must all work:

conda env create -f environment.yml
conda activate tpc-agentcf

# No manual download needed — all automatic
python scripts/prepare_data.py --config config/default.yaml

python scripts/run_baselines.py --config config/default.yaml
python scripts/run_tpc_agentcf.py --config config/default.yaml
python scripts/run_ablation.py --config config/default.yaml
python scripts/make_paper_tables.py --config config/default.yaml

Verify these files exist after running:
[ ] data/processed/interactions.csv
[ ] data/processed/items.csv
[ ] outputs/metrics/all_users/tpc_agentcf_results.csv
[ ] outputs/metrics/conflict_users/tpc_agentcf_results.csv
[ ] outputs/metrics/high_conflict_users/tpc_agentcf_results.csv
[ ] outputs/explanations/recommendations.jsonl
[ ] outputs/conflicts/conflict_log.jsonl
[ ] outputs/explanations/escalation_traces.jsonl
[ ] outputs/ablations/ablation_results.csv
[ ] outputs/paper_assets/research_claims.md
[ ] outputs/paper_assets/main_table.md
[ ] outputs/paper_assets/qualitative_examples.md
[ ] notebooks/colab_quickstart.ipynb runs end-to-end

============================================================
24. SCIENTIFIC FRAMING — DO NOT OVERCLAIM
============================================================

The README and any generated text must frame the contribution as:

"AgentCF++ models cross-domain recommendation with group-level
popularity signals through shared memory. However, it does not
distinguish between users whose preferences are temporally stable
and users who are in an active preference drift period.
We show that, precisely during drift, group popularity signals
risk amplifying recommendations that do not reflect the user's
authentic preferences.

TPC-AgentCF detects short-term/long-term preference conflict
and uses this as a signal to calibrate the influence of popularity
— reducing it conditionally when conflict is detected, rather than
applying unconditional popularity debiasing.

We do not claim to introduce temporal memory for LLM agents
(DyTA4Rec, CIKM 2025), multi-agent debate for recommendation
(MACF, 2025), or general verifiable reasoning for recommendation
(VRec, 2026). Our contribution is the specific use of temporal
preference conflict as a conditional trigger for popularity
calibration in a cross-domain agentic recommendation setting."
