"""Build the English submission report (HTML for PDF) from real code and results."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "asm2_report_en.html"


def lines(rel: str, start: str, end: str | None = None, include_end: bool = True) -> str:
    """Return the code between the first line matching ``start`` and ``end``."""
    src = (ROOT / rel).read_text().splitlines()
    i = next(n for n, line in enumerate(src) if re.search(start, line))
    if end is None:
        j = len(src) - 1
    else:
        j = next(n for n in range(i + 1, len(src)) if re.search(end, src[n]))
        if not include_end:
            j -= 1
    return "\n".join(src[i : j + 1]).rstrip()


def code(text: str, title: str) -> str:
    """Render a titled code listing."""
    return (
        f'<figure class="code"><figcaption>{html.escape(title)}</figcaption>'
        f"<pre><code>{html.escape(text)}</code></pre></figure>"
    )


def svg(name: str) -> str:
    """Inline a pre-rendered Mermaid SVG."""
    s = (ROOT / "docs" / "diagrams" / f"{name}.svg").read_text()
    return re.sub(r'style="max-width:[^"]*"', 'style="max-width:100%;height:auto"', s, count=1)


res = pd.read_csv(ROOT / "reports" / "experiment_results.csv").set_index("exp_id")
summary = json.loads((ROOT / "reports" / "selection_summary.json").read_text())
chart = base64.b64encode((ROOT / "reports" / "experiments_chart_en.png").read_bytes()).decode()
tm, hp = summary["test_metrics"], summary["tuned_hyperparams"]


def f(x: float, d: int = 3) -> str:
    """Format a metric."""
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def r(e: str, col: str, d: int = 3) -> str:
    """Shortcut for a result cell."""
    return f(res.loc[e][col], d)


MATRIX = [
    ("E01", "Rule B2", "debt + unpaid fees + no module passed in sem. 1", "Full", "–", "Non-ML baseline"),
    ("E02", "Logistic regression", "max_iter=2000", "Full", "None", "Baseline B3, interpretable"),
    ("E03", "Logistic regression", "class_weight=balanced", "Full", "Class weights", "Does reweighting help?"),
    ("E04", "Random forest", "400 trees, min_samples_leaf=3", "Full", "Class weights", "Bagging ensemble"),
    ("E05", "XGBoost", "defaults", "Full", "None", "Untuned boosting"),
    ("E06", "XGBoost", "scale_pos_weight ≈ 2.1", "Full", "Class weights", "Effect on calibration"),
    ("E07", "XGBoost", "random search, 25 candidates, 5-fold CV", "Full", "None", "Hyperparameter tuning"),
    ("E08", "XGBoost (E07 params)", "as E07", "Full", "SMOTE", "Synthetic oversampling"),
    ("E09", "XGBoost (E07 params)", "+ isotonic calibration, cv=5", "Full", "None", "Probability calibration"),
    ("E10", "XGBoost (E07 params)", "as E07", "Enrolment + financial", "None", "Earliest possible prediction"),
    ("E11", "XGBoost (E07 params)", "as E07", "+ Semester 1", "None", "Early prediction"),
    ("E12", "XGBoost (E07 params)", "as E07", "No financial features", "None", "Privacy trade-off"),
]
matrix_html = "".join(
    f'<tr><td class="id">{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td><td>{g}</td></tr>'
    for a, b, c, d, e, g in MATRIX
)


def result_row(e: str) -> str:
    """One row of the results table."""
    x = res.loc[e]
    gap = x.fairness_gap_at_25
    cv = "–" if pd.isna(x.cv_pr_auc_mean) else f"{x.cv_pr_auc_mean:.3f} ± {x.cv_pr_auc_std:.3f}"
    cls = ' class="pick"' if e == summary["selected"] else ""
    warn = " warn" if gap > 0.10 else ""
    return (
        f'<tr{cls}><td class="id">{e}</td><td class="num">{f(x.pr_auc)}</td>'
        f'<td class="num">{cv}</td><td class="num">{f(x.roc_auc)}</td>'
        f'<td class="num">{f(x.recall_at_10)}</td><td class="num">{f(x.precision_at_10)}</td>'
        f'<td class="num">{f(x.recall_at_25)}</td><td class="num">{f(x.brier)}</td>'
        f'<td class="num">{f(x.ece)}</td><td class="num{warn}">{f(gap)}</td>'
        f'<td class="num">{f(x.train_seconds, 2)}</td></tr>'
    )


results_html = "".join(result_row(e) for e in res.index)

STAGES = [
    ("1", "Data ingestion", "Nightly export from SIS, LMS, attendance, finance and care-team call log is copied into an immutable, dated snapshot; its SHA-256 is recorded.", "raw CSV snapshot + data_sha256", "file readable, required columns present", ""),
    ("2", "Data validation", "pandera schema (label values, age 15–80, grades 0–20, binary flags), missing-value ratio, minimum row count.", "validated DataFrame", "missing ≤ 5 %, rows ≥ 1,000, all checks pass", "fail → stop, alert, fallback rule B2"),
    ("3", "Preprocessing", "Binary label (Dropout = 1); stratified 60/20/20 train/validation/test split with fixed seed.", "train 2,654 · val 885 · test 885", "equal dropout rate in all splits; no overlap", ""),
    ("4", "Feature engineering", "Select feature groups, drop sensitive attributes, add approval rates, grade trend and financial-risk score; one-hot encode and scale.", "feature matrix (inside sklearn pipeline)", "no sensitive column; same code for training and scoring", ""),
    ("5", "Model training", "Twelve configurations E01–E12, each an MLflow run; randomized hyperparameter search with 5-fold CV.", "fitted pipelines + MLflow runs", "params, metrics, model and feature list logged", ""),
    ("6", "Model evaluation", "PR-AUC, CV PR-AUC ± SD, Recall/Precision@k, Brier, ECE, fairness gap; selection rule and quality gate.", "results table + selected model", "PR-AUC ≥ 0.80; fairness gap ≤ 0.10; ≤ 0.01 worse than champion", "fail → keep current champion"),
    ("7", "Deployment", "Register version with alias challenger; human approval moves alias champion; weekly DAG scores with champion and assigns tiers.", "model version + Tier A/B lists", "human approval; previous-champion kept for rollback", ""),
]
stages_html = "".join(
    f'<li class="stage"><div class="num">{n}</div><div><h4>{name}</h4><p>{op}</p><p class="chips">'
    f'<span class="chip out">Output: {out}</span><span class="chip gate">Gate: {gate}</span>'
    + (f'<span class="chip fail">{fail}</span>' if fail else "")
    + "</p></div></li>"
    for n, name, op, out, gate, fail in STAGES
)

SN = {
    "env": code(lines(".env.example", "MLFLOW_TRACKING_URI", "ALERT_EMAIL"), ".env.example – infrastructure settings as environment variables"),
    "setup": code(lines("src/dropout/experiments.py", "def main", r"splits = load_splits"), "src/dropout/experiments.py – MLflow experiment setup"),
    "matrix": code(lines("src/dropout/experiments.py", r"^MATRIX", r"^DEPENDS_ON_TUNED"), "src/dropout/experiments.py – experiment matrix as code"),
    "log": code(
        lines("src/dropout/train.py", r"with mlflow.start_run", r"start = time.perf_counter")
        + "\n        ...  # build, (tune), cross-validate and fit the pipeline\n"
        + lines("src/dropout/train.py", r"mlflow.log_metrics\(metrics\)", r"input_example"),
        "src/dropout/train.py – parameter, metric and artifact logging",
    ),
    "reg": code(lines("src/dropout/registry.py", "def register_candidate", r"Promoted"), "src/dropout/registry.py – registration, versioning and aliases"),
    "select": code(lines("src/dropout/experiments.py", r"^def select_model", r"return top if gain"), "src/dropout/experiments.py – model selection rule"),
    "feat": code(lines("src/dropout/features.py", r"^class EngineeredFeatures", r'out\["financial_risk"\]') + "\n        return out", "src/dropout/features.py – shared feature engineering"),
    "valid": code(lines("src/dropout/data.py", r"^def validate", r'logger.info\("Validation passed'), "src/dropout/data.py – validation quality gate"),
    "dargs": code(lines("dags/dropout_training_dag.py", r"^def notify_failure", r"^}"), "dags/dropout_training_dag.py – alerting callback and default arguments"),
    "tasks": code(
        lines("dags/dropout_training_dag.py", r"^def validate_data", r'return "build_features"')
        + "\n\n\n"
        + lines("dags/dropout_training_dag.py", r"^def check_drift", r'return "trigger_retraining"'),
        "dags/dropout_training_dag.py – task definitions (thin wrappers around the package)",
    ),
    "scoring_dag": code(lines("dags/dropout_training_dag.py", r"^with DAG\($", r"t_features >> t_drift"), "dags/dropout_training_dag.py – weekly scoring DAG, tasks and dependencies"),
    "training_dag": code(lines("dags/dropout_training_dag.py", r"^def run_matrix", r"t_snapshot >> t_validate_tr"), "dags/dropout_training_dag.py – training DAG"),
    "style": code(lines("src/dropout/evaluate.py", r"^def top_k_mask", r"return float\(y\[top_k_mask\(scores, fraction\)\]\.sum"), "src/dropout/evaluate.py – style, type hints and docstrings"),
    "lint": code(lines("pyproject.toml", r"^\[tool.ruff\]", r'convention = "google"'), "pyproject.toml – enforced code standards"),
    "params": code(
        lines("params.yaml", "^project:", r"min_rows")
        + "\n\nfeatures:\n  sensitive: [Gender, Nacionality, Marital status, International]\n  ...\n\n"
        + lines("params.yaml", "^evaluation:", "registered_model"),
        "params.yaml – configuration file (excerpt)",
    ),
    "load": code(lines("src/dropout/config.py", r"^def load_settings", r"^def set_global_seed", include_end=False).rstrip(), "src/dropout/config.py – environment variables and config loading"),
    "git": code(
        """# GitHub flow: feature branch -> pull request (CI: ruff + pytest) -> main -> tag
git switch -c feat/early-warning-features
git commit -m "Add approval-rate and grade-trend features"
git push -u origin feat/early-warning-features      # open PR; CI must pass
git switch main && git pull
git tag -a v0.2.0 -m "Experiment matrix E01-E12, LR champion"
git push origin v0.2.0

# Data versioning with DVC (production), remote = MinIO
dvc add data/raw/students.csv
git add data/raw/students.csv.dvc .gitignore
git commit -m "Data snapshot 2026-S3" && dvc push

# Every registered model carries the commit and data hash (registry.py / experiments.py)
#   tags: exp_id=E02, git_commit=bbc0168, data_sha256=19737f02127e, approved_by=...""",
        "Code and data versioning",
    ),
    "seed": code(
        lines("src/dropout/config.py", r"^def set_global_seed", None)
        + "\n\n# ...and the same seed is passed explicitly everywhere randomness exists:\n"
        + "#   train_test_split(..., random_state=seed)      data.py\n"
        + "#   StratifiedKFold(5, shuffle=True, random_state=seed), RandomizedSearchCV(random_state=seed)\n"
        + "#   LogisticRegression / RandomForest / XGBClassifier(random_state=seed), SMOTE(random_state=seed)",
        "src/dropout/config.py – random seed settings",
    ),
    "docker": code((ROOT / "Dockerfile").read_text().strip(), "Dockerfile"),
    "req": code((ROOT / "requirements.txt").read_text().strip(), "requirements.txt – pinned dependencies"),
}

CSS = """
@page{size:A4;margin:18mm 17mm 18mm 17mm}
*{box-sizing:border-box}
body{font-family:"Liberation Serif","Times New Roman",serif;color:#1d2430;font-size:10.8pt;line-height:1.45;margin:0}
h1,h2,h3,h4{font-family:"Liberation Sans",Arial,sans-serif;color:#1f4e8c;line-height:1.25}
h1{font-size:20pt;margin:0 0 6px}
h2{font-size:14.5pt;margin:20px 0 8px;border-bottom:1.5px solid #1f4e8c;padding-bottom:3px;break-after:avoid}
h3{font-size:11.8pt;margin:13px 0 6px;break-after:avoid}
h4{font-size:10.5pt;margin:0 0 2px;color:#1d2430}
p{margin:0 0 7px;text-align:justify}ul,ol{margin:0 0 7px;padding-left:20px}li{margin-bottom:3px}
.weight{font-family:"Liberation Sans",Arial,sans-serif;font-size:9pt;color:#5b6575;float:right;margin-top:-28px}
table{width:100%;border-collapse:collapse;margin:6px 0 10px;font-size:9.2pt}
tr{break-inside:avoid}th,td{border:1px solid #d5dae2;padding:3.5px 5px;vertical-align:top;text-align:left}
th{background:#eef3fa;font-family:"Liberation Sans",Arial,sans-serif;font-size:8.6pt}
td.num{font-family:"Liberation Mono",monospace;font-size:8.4pt;white-space:nowrap}
td.warn{color:#a1311f;font-weight:bold}tr.pick td{background:#e5f2e9}
td.id,.id{font-family:"Liberation Mono",monospace;font-weight:bold;white-space:nowrap;font-size:8.6pt}
code{font-family:"Liberation Mono",monospace;font-size:8.8pt;background:#f1f3f6;padding:0 2px}
figure{margin:6px 0 10px}
figure.code{border:1px solid #d5dae2;border-radius:4px;background:#f8f9fb}
figure.code figcaption{font-family:"Liberation Mono",monospace;font-size:8pt;color:#5b6575;padding:3px 8px;border-bottom:1px solid #d5dae2;background:#fff}
figure.code pre{margin:0;padding:6px 8px;font-size:7.6pt;line-height:1.35;white-space:pre-wrap;word-break:break-word}
figure.code code{background:none;padding:0;font-size:inherit}
.cover{border-bottom:2px solid #1f4e8c;padding-bottom:10px;margin-bottom:8px}
.eyebrow{font-family:"Liberation Sans",Arial,sans-serif;font-size:9pt;letter-spacing:.06em;text-transform:uppercase;color:#5b6575}
.meta td{border:none;padding:1px 8px 1px 0;font-size:10.2pt}.meta td:first-child{color:#5b6575;width:150px}
.note{font-size:9.2pt;color:#5b6575;border-left:3px solid #d5dae2;padding:4px 10px;margin:8px 0}
.stages{list-style:none;padding:0;margin:6px 0}
.stage{display:grid;grid-template-columns:24px 1fr;gap:8px;border:1px solid #d5dae2;border-radius:5px;padding:6px 9px;break-inside:avoid;position:relative}
.stage+.stage{margin-top:13px}.stage+.stage::before{content:"▼";position:absolute;top:-13px;left:14px;color:#8a94a3;font-size:8pt}
.stage .num{width:21px;height:21px;border-radius:50%;background:#1f4e8c;color:#fff;display:grid;place-items:center;font-weight:bold;font-size:9pt;font-family:Arial}
.stage p{margin:0;font-size:9.6pt;text-align:left}.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px!important}
.chip{font-family:"Liberation Sans",Arial;font-size:7.8pt;padding:1px 6px;border-radius:9px;border:1px solid #d5dae2}
.chip.out{background:#eef3fa}.chip.gate{background:#e5f2e9}.chip.fail{background:#f8e4df;color:#a1311f}
.fig{break-inside:avoid;margin:6px 0 10px;text-align:center}.fig svg{max-width:100%;height:auto}
.cap{font-size:8.8pt;color:#5b6575;text-align:center;margin-top:3px}
img.chart{width:100%}
.box{background:#eef7f0;border-left:4px solid #1c6b3a;padding:7px 11px;margin:8px 0;break-inside:avoid}
.box p:last-child{margin-bottom:0}
.kpi{display:flex;gap:6px;margin:6px 0}.kpi div{flex:1;border:1px solid #cfe3d5;background:#fff;padding:4px 6px;font-size:8.3pt;color:#5b6575}
.kpi b{display:block;font-family:"Liberation Mono",monospace;font-size:12pt;color:#1d2430}
.pb{break-before:page}
"""

BODY = f"""
<div class="cover">
  <div class="eyebrow">DDM501 · Individual Assignment 2 · ML Pipeline Design &amp; MLOps Analysis</div>
  <h1>ML Pipeline and MLOps Design for a Student Dropout Early Warning System</h1>
  <table class="meta">
    <tr><td>Student</td><td><b>Nguyen Thi Hong Hanh</b></td></tr>
    <tr><td>Student ID</td><td><b>25MS13316</b></td></tr>
    <tr><td>Continues</td><td>Assignment 1 – ML System Design Document (dropout early warning)</td></tr>
    <tr><td>Experimental data</td><td>UCI "Predict Students' Dropout and Academic Success" (4,424 students) as a proxy</td></tr>
    <tr><td>Stack</td><td>Python 3.11 · scikit-learn · XGBoost · pandera · MLflow 2.17 · Apache Airflow 2.10 · Docker</td></tr>
  </table>
  <div class="note">All code listings are excerpts of a working prototype, and all numbers come from actual runs. Institutional figures for BTEC FPT British College remain the assumptions stated in Assignment 1.</div>
</div>

<h2>0. Link to Assignment 1</h2>
<p><b>Problem statement (recap).</b> Every week, score the dropout risk of every enrolled student at BTEC FPT British College (assumed 12,000 students, three 15-week semesters per year) at least 8 weeks before the semester ends. <b>Tier A</b> (top 10 % risk) goes to academic advisors for personal counselling; <b>Tier B</b> (top 10–25 %) goes to the student care team for check-in calls. Business target: reduce annual dropout from 8 % to 6.5 %.</p>
<p><b>Architecture (recap).</b> Two pipelines share one feature module: a <b>training pipeline</b> runs each semester or on drift; a <b>weekly batch scoring pipeline</b> runs every Sunday. Components: nightly extraction → data validation → pseudonymised warehouse → feature computation → training with MLflow tracking → quality gate + human approval → model registry → batch scoring → advisor/care-team web app; monitoring for pipeline health, drift and fairness; fallback rule B2 when data is invalid.</p>
<p><b>Refinements after Sessions 3–5.</b></p>
<ul>
  <li><b>Proxy dataset.</b> No real college data is available, so experiments use the public UCI dataset (4,424 students, 36 attributes, 32.1 % dropout). It has no weekly features, so the feature groups "enrolment / semester 1 / semester 2" simulate predicting earlier or later.</li>
  <li><b>Split strategy.</b> The dataset has no enrolment date, so a stratified split replaces the time-based split of Assignment 1; production data will return to a semester-based split.</li>
  <li><b>Selection rule as code.</b> Trade-off 5.1 of Assignment 1 ("choose gradient boosting only if its PR-AUC beats logistic regression by ≥ 0.02") is now the function <code>select_model</code>, parameterised in <code>params.yaml</code>.</li>
  <li><b>Additions.</b> Cross-validated PR-AUC with standard deviation to separate real differences from noise; a <code>previous-champion</code> alias for one-step rollback; PSI drift detection inside the scoring DAG.</li>
</ul>

<h2>1. Pipeline Design</h2><div class="weight">25%</div>
<h3>1.1 Pipeline architecture diagram</h3>
<ol class="stages">{stages_html}</ol>
<p class="cap">Figure 1. Training pipeline: seven stages, the artefact passed between them, and the quality gate each must pass. The weekly scoring pipeline reuses stages 1, 2 and 4 with identical code, then scores with the champion model (Figure 3).</p>

<h3>1.2 Stage specifications</h3>
<table>
  <thead><tr><th style="width:13%">Stage</th><th style="width:16%">Purpose</th><th style="width:19%">Input → output</th><th>Key operations (module)</th><th style="width:21%">Quality gate</th></tr></thead>
  <tbody>
    <tr><td><b>1. Ingestion</b></td><td>Fixed, traceable copy of the data</td><td>Source exports → dated CSV snapshot + SHA-256</td><td><code>data.ingest</code>, <code>file_sha256</code>; normalise column names (the export contains a trailing tab)</td><td>File readable, required columns present</td></tr>
    <tr><td><b>2. Validation</b></td><td>Stop corrupted data before training or scoring</td><td>Raw DataFrame → validated DataFrame</td><td><code>data.validate</code>: pandera schema, value ranges, missing ratio, row count</td><td>Raises <code>DataValidationError</code> → pipeline stops; scoring DAG branches to rule B2</td></tr>
    <tr><td><b>3. Preprocessing</b></td><td>Create the label and honest evaluation sets</td><td>DataFrame → train 2,654 / val 885 / test 885</td><td><code>make_label</code>; stratified <code>split</code> with <code>random_state=42</code></td><td>Dropout rate 32.1 % in each split; disjoint indices (unit-tested)</td></tr>
    <tr><td><b>4. Feature engineering</b></td><td>Turn raw records into risk signals</td><td>Raw columns → feature matrix</td><td><code>select_columns</code> (drops Gender, Nacionality, Marital status, International); <code>EngineeredFeatures</code> (approval rates, grade trend, financial risk); <code>ColumnTransformer</code> (one-hot, scaling)</td><td>No sensitive attribute (unit-tested); divide-by-zero safe</td></tr>
    <tr><td><b>5. Training</b></td><td>Systematically compare configurations</td><td>Feature matrix → fitted pipelines</td><td><code>train.run_experiment</code>: one configuration = one MLflow run; <code>RandomizedSearchCV</code> (25 candidates)</td><td>Run must log params, metrics, model, feature list</td></tr>
    <tr><td><b>6. Evaluation</b></td><td>Select by operational metrics</td><td>Validation predictions → metrics + selected model</td><td><code>evaluate</code>, <code>select_model</code>, <code>passes_quality_gate</code></td><td>PR-AUC ≥ 0.80; gender recall gap ≤ 0.10; ≤ 0.01 below champion</td></tr>
    <tr><td><b>7. Deployment</b></td><td>Controlled release</td><td>Run → model version → Tier A/B lists</td><td><code>register_candidate</code> (challenger), <code>promote</code> (champion, keeps previous-champion), <code>score_with_champion</code>, <code>assign_and_publish</code></td><td>Human approval required; test set used once, for the selected model only</td></tr>
  </tbody>
</table>
{SN["valid"]}
{SN["feat"]}

<h3>1.3 Design rationale</h3>
<ul>
  <li><b>Structure.</b> Each stage has one responsibility and an explicit input/output artefact, so failures are localised and a single stage can be re-run. Validation comes first because bad data that slips through silently corrupts every later stage.</li>
  <li><b>Modularity and reuse.</b> Every stage is a plain, typed function in the <code>dropout</code> package that exchanges file paths or objects and has no Airflow dependency; the same functions are called by the experiment script, the DAGs and the unit tests. <code>EngineeredFeatures</code> lives <i>inside</i> the scikit-learn pipeline stored in MLflow, so scoring cannot compute features differently from training (no training-serving skew). A new algorithm is one branch in <code>build_estimator</code>; a new experiment is one line in <code>MATRIX</code>.</li>
  <li><b>Error handling and recovery.</b> (1) Invalid data → stop and score with fallback rule B2 so advisors still receive a list; (2) transient errors → Airflow retries twice with exponential back-off; (3) candidate fails the gate → keep the current champion and log the reasons; (4) a bad model in production → move the champion alias back to <code>previous-champion</code> without rebuilding; (5) dated snapshots make every task idempotent and re-runnable.</li>
  <li><b>Scalability.</b> Scoring 4,424 students takes under one second and grows linearly; for 12,000–50,000 students scoring can be partitioned by programme and run in parallel. The most expensive step, hyperparameter search (21 s for E07), runs once per semester and can be scaled with <code>n_jobs</code> or Optuna. Switching <code>MLFLOW_TRACKING_URI</code> from SQLite to PostgreSQL + MinIO supports multiple users with no code change.</li>
</ul>

<h2 class="pb">2. Experiment Tracking &amp; Metrics Analysis</h2><div class="weight">25%</div>
<h3>2.1 Experiment design</h3>
<ul>
  <li><b>Experiment variables:</b> <i>algorithm</i> (rule, logistic regression, random forest, XGBoost); <i>hyperparameters</i> (defaults vs. randomized search over n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight); <i>features</i> (enrolment, semester 1, semester 2, financial groups); <i>data preprocessing</i> (no rebalancing, class weights, SMOTE, isotonic calibration).</li>
  <li><b>Baselines:</b> E01 is rule B2 (the production fallback rule from Assignment 1); E02 is logistic regression (baseline B3). A more complex model must beat E02 by ≥ 0.02 PR-AUC to be selected.</li>
  <li><b>Fair comparison:</b> identical split and seed (42) for all runs; selection on the validation set; the test set is used <b>once</b>, for the final model.</li>
</ul>
<table>
  <thead><tr><th>ID</th><th>Algorithm</th><th>Hyperparameters</th><th>Features</th><th>Rebalancing</th><th>Question answered</th></tr></thead>
  <tbody>{matrix_html}</tbody>
</table>
{SN["matrix"]}

<h3>2.2 Metrics strategy</h3>
<table>
  <thead><tr><th style="width:14%">Type</th><th style="width:27%">Metric</th><th>Rationale</th></tr></thead>
  <tbody>
    <tr><td><b>Primary</b></td><td>PR-AUC (validation) and 5-fold CV PR-AUC ± SD</td><td>Measures how well at-risk students are ranked at the top, focusing on the minority class; the CV standard deviation shows whether a difference exceeds noise. Used for model selection.</td></tr>
    <tr><td><b>Primary (operational)</b></td><td>Recall@10 %, Recall@25 %</td><td>Correspond to Tier A (advisors) and Tiers A+B (plus care team): the share of all dropouts reached within contact capacity.</td></tr>
    <tr><td>Secondary</td><td>Precision@10 %, ROC-AUC</td><td>Whether the advisors' list is accurate; ROC-AUC for comparison with literature.</td></tr>
    <tr><td>Secondary</td><td>Brier score, ECE</td><td>The dashboard sums probabilities to estimate expected dropouts per programme, so probabilities must be calibrated.</td></tr>
    <tr><td>Constraint</td><td>Recall@25 % gap by gender</td><td>Fairness condition from Assignment 1; a gap above 0.10 makes a model ineligible.</td></tr>
    <tr><td>Secondary</td><td>Training time</td><td>Operational cost per retraining.</td></tr>
  </tbody>
</table>
<p><b>Business–model alignment.</b> Following the goal hierarchy of Assignment 1: Recall@10 % × number of dropouts = at-risk students met by advisors; × assumed 30 % retention = students kept; Recall@25 % adds the care-team tier. Note that the proxy data has a 32 % base rate, so a 10 % list can contain at most 31 % of all dropouts: Recall@10 % is capped near 0.31 even though the list is almost perfectly precise. At the assumed BTEC rate of 2.7 % per semester the constraint flips (the list has room for every case, but precision will be much lower). Metric definitions carry over unchanged; only the values differ.</p>

<h3>2.3 Code: MLflow setup, logging and model registration</h3>
{SN["env"]}
{SN["setup"]}
{SN["log"]}
{SN["reg"]}
<p>Each model version is tagged with <code>exp_id</code>, <code>git_commit</code>, <code>data_sha256</code> and <code>approved_by</code>, so any version traces back to the exact code and data. Actual registry state after the run: <code>dropout-risk</code> version 1 with aliases <code>champion</code> and <code>challenger</code>, git_commit <code>bbc0168</code>, data_sha256 <code>19737f02127e</code>; 12 runs in experiment <code>dropout-early-warning</code>.</p>

<h3>2.4 Experiment results (validation set, n = 885, 284 dropouts)</h3>
<table>
  <thead><tr><th>ID</th><th>PR-AUC</th><th>CV PR-AUC ± SD</th><th>ROC-AUC</th><th>Rec@10%</th><th>Prec@10%</th><th>Rec@25%</th><th>Brier</th><th>ECE</th><th>Gender gap</th><th>Sec.</th></tr></thead>
  <tbody>{results_html}</tbody>
</table>
<p class="cap" style="text-align:left">Green row: selected model. Red: fairness gap above 0.10 (ineligible). Maximum achievable Recall@10 % is 0.31 (88 list slots / 284 dropouts). A complete re-run from an empty tracking store reproduced every metric exactly.</p>
<div class="fig"><img class="chart" alt="Experiment results chart" src="data:image/png;base64,{chart}">
<div class="cap">Figure 2. Left: validation PR-AUC (dots) and 5-fold CV mean ± 1 SD (grey bars); E02 highlighted. Right: PR-AUC by data available at prediction time (dashed line = random, 0.32).</div></div>

<h3>2.5 Analysis of patterns and trade-offs</h3>
<ul>
  <li><b>ML clearly beats the rule.</b> E01 reaches PR-AUC {r("E01", "pr_auc")} while ML models with full features reach {r("E05", "pr_auc")}–{r("E09", "pr_auc")}; Recall@25 % rises from {r("E01", "recall_at_25")} to about 0.69. This confirms the "why ML" argument of Assignment 1.</li>
  <li><b>Among ML models, differences are within noise.</b> E02–E09 differ by at most {f(res.loc["E09"].pr_auc - res.loc["E05"].pr_auc)} validation PR-AUC, while CV standard deviations are 0.011–0.018. The best validation model, E09 ({r("E09", "pr_auc")}), beats logistic regression E02 by only {f(res.loc["E09"].pr_auc - res.loc["E02"].pr_auc, 4)}. On CV, tuned XGBoost (E07, {r("E07", "cv_pr_auc_mean")}) leads E02 ({r("E02", "cv_pr_auc_mean")}) by about 0.01, still less than one SD. With small tabular data and near-linear academic signals, complex models have little room to win.</li>
  <li><b>Ranking vs. calibration.</b> Class weighting does not improve PR-AUC but damages probabilities: ECE rises from {r("E02", "ece")} (E02) to {r("E03", "ece")} (E03); XGBoost with <code>scale_pos_weight</code> (E06) has ECE {r("E06", "ece")}. Top-k lists only need ranking, so reweighting brings cost without benefit.</li>
  <li><b>SMOTE and isotonic calibration do not pay off.</b> E08 has the highest CV PR-AUC ({r("E08", "cv_pr_auc_mean")}) but no validation gain and adds synthetic data. E09 has ECE {r("E09", "ece")}, <i>worse</i> than uncalibrated E07 ({r("E07", "ece")}): XGBoost with log-loss is already well calibrated and isotonic fitting on ~530-sample folds is noisy. Calibration should be added only when measured to help.</li>
  <li><b>Early detection vs. accuracy</b> (trade-off 5.5 of Assignment 1). Enrolment data only (E10): PR-AUC {r("E10", "pr_auc")}, close to the rule; with semester 1 (E11): {r("E11", "pr_auc")}; with semester 2 (E07): {r("E07", "pr_auc")}. Most signal arrives with the <b>first academic results</b>, supporting weekly scoring whose confidence grows once interim assessments exist.</li>
  <li><b>Privacy vs. accuracy</b> (trade-off 5.4). Removing the two financial features (E12) lowers PR-AUC by {f(res.loc["E07"].pr_auc - res.loc["E12"].pr_auc)} and raises the fairness gap to {r("E12", "fairness_gap_at_25")}, above the limit. Fee status is an important, legitimate signal: keep it, but protect it (pseudonymisation, role-based access, no amounts shown to callers).</li>
  <li><b>Fairness.</b> Random forest (E04, gap {r("E04", "fairness_gap_at_25")}) and E12 exceed 0.10 and are excluded automatically, although E04 has a competitive PR-AUC; ranking by PR-AUC alone would have hidden this.</li>
  <li><b>Cost.</b> E07 needs {r("E07", "train_seconds", 1)} s for tuning versus {r("E02", "train_seconds", 2)} s for E02; at this scale cost does not decide.</li>
</ul>
{SN["select"]}

<h3>2.6 Recommendation</h3>
<div class="box">
  <p><b>Select E02: logistic regression, no rebalancing, full feature set.</b> The best PR-AUC model (E09) is only {f(res.loc["E09"].pr_auc - res.loc["E02"].pr_auc, 4)} ahead, far below the 0.02 margin set in Assignment 1. E02 is directly interpretable for advisors (coefficients show which factors raise risk), well calibrated (ECE {r("E02", "ece")}), within the fairness limit (gap {r("E02", "fairness_gap_at_25")}) and trains instantly. <code>select_model</code> reached this decision automatically. Results on the untouched test set (885 students):</p>
  <div class="kpi">
    <div><b>{tm["pr_auc"]:.3f}</b>PR-AUC</div><div><b>{tm["precision_at_10"]:.3f}</b>Precision@10 % (Tier A)</div>
    <div><b>{tm["recall_at_25"]:.3f}</b>Recall@25 % (Tiers A+B)</div><div><b>{tm["ece"]:.3f}</b>ECE</div>
    <div><b>{tm["fairness_gap_at_25"]:.3f}</b>Gender recall gap</div>
  </div>
  <p>Test performance matches validation, so there is no sign of overfitting to the validation set. The runner-up is E07 (XGBoost; best parameters: {", ".join(f"{k}={v}" for k, v in hp.items())}). It should be re-evaluated when real weekly features exist, because non-linear interactions (e.g. rising absence <i>and</i> overdue fees) may let it clear the 0.02 margin.</p>
</div>

<h2 class="pb">3. Workflow Orchestration Design</h2><div class="weight">20%</div>
<h3>3.1 DAG architecture</h3>
<div class="fig">{svg("scoring")}<div class="cap">Figure 3. DAG <code>dropout_weekly_scoring</code> (Sunday 02:00, Asia/Ho_Chi_Minh).</div></div>
<div class="fig">{svg("training")}<div class="cap">Figure 4. DAG <code>dropout_training</code> (after each semester's labels arrive, or triggered by drift). Both DAGs load in Apache Airflow 2.10.3 with no import errors (9 and 4 tasks).</div></div>
<table>
  <thead><tr><th style="width:24%">Task</th><th>Description</th><th style="width:32%">Dependencies</th></tr></thead>
  <tbody>
    <tr><td class="id">extract_data</td><td>Copy the nightly export into a dated snapshot; return its path via XCom</td><td>None (first task)</td></tr>
    <tr><td class="id">validate_data</td><td>Branch: valid → <code>build_features</code>; invalid → <code>fallback_rule</code></td><td>Needs the snapshot from extract_data</td></tr>
    <tr><td class="id">build_features</td><td>Compute features with the shared module; write file</td><td>Runs only if data is valid</td></tr>
    <tr><td class="id">score_students</td><td>Load <code>models:/dropout-risk@champion</code>; score all students</td><td>Needs the feature file</td></tr>
    <tr><td class="id">fallback_rule</td><td>Score with rule B2 so a list is still produced</td><td>Runs only if data is invalid</td></tr>
    <tr><td class="id">assign_tiers</td><td>Rank, assign Tier A (advisor) / Tier B (care team), publish lists</td><td>After <i>either</i> scoring branch succeeds (<code>NONE_FAILED_MIN_ONE_SUCCESS</code>)</td></tr>
    <tr><td class="id">check_drift</td><td>Branch: max feature PSI &gt; 0.2 → retrain</td><td>Needs the feature file; parallel to scoring</td></tr>
    <tr><td class="id">trigger_retraining / no_drift</td><td>Trigger the training DAG, or end</td><td>Outcome of check_drift</td></tr>
    <tr><td class="id">snapshot_labelled_data</td><td>Snapshot data whose labels are complete</td><td>None (training DAG start)</td></tr>
    <tr><td class="id">validate_training_data</td><td>Fail the run if validation fails (current champion is kept)</td><td>Needs the labelled snapshot</td></tr>
    <tr><td class="id">train_and_track_experiments</td><td>Run E01–E12, log to MLflow, gate, register challenger</td><td>Needs validated data</td></tr>
    <tr><td class="id">await_human_approval</td><td>Explicit stop: a reviewer inspects MLflow and moves the champion alias</td><td>After training</td></tr>
  </tbody>
</table>

<h3>3.2 Scheduling strategy</h3>
<table>
  <thead><tr><th>DAG</th><th>Trigger type</th><th>Frequency</th><th>Reason</th></tr></thead>
  <tbody>
    <tr><td>dropout_weekly_scoring</td><td>Time-based (cron <code>0 2 * * 0</code>)</td><td>Weekly, Sunday 02:00; <code>catchup=False</code>, <code>max_active_runs=1</code></td><td>Advisors work weekly; lists must be ready before Monday 06:00 (NFR1, 4-hour SLA)</td></tr>
    <tr><td>dropout_training</td><td>Time-based + event-based (TriggerDagRunOperator on drift) + manual</td><td>Three times a year (<code>0 3 1 1,5,9 *</code>), after each semester's labels</td><td>Labels only exist after each semester; drift should not wait for the next one</td></tr>
  </tbody>
</table>

<h3>3.3 Code: DAG definitions, task definitions and dependencies</h3>
{SN["dargs"]}
{SN["tasks"]}
{SN["scoring_dag"]}
{SN["training_dag"]}

<h3>3.4 Operational considerations</h3>
<ul>
  <li><b>Failure handling.</b> Two retries with exponential back-off starting at 10 minutes; 1-hour execution timeout per task (3 hours for training). Invalid data does not fail the DAG but branches to the fallback rule. <code>max_active_runs=1</code> prevents overlapping runs; dated inputs/outputs make every task idempotent. Validation failure in the training DAG stops the run, so the current champion stays in place.</li>
  <li><b>Alerting.</b> <code>on_failure_callback</code> e-mails the data team with the task and log link. SLAs (1 h for extraction, 4 h for lists) raise SLA-miss alerts. Fallback activations and drift-triggered retraining are logged and reported.</li>
  <li><b>Monitoring.</b> Airflow UI (durations, success rate); MLflow (metrics per model version); weekly PSI and risk-score distribution (Evidently/Grafana, as in Assignment 1); leading indicators such as the share of Tier A contacted within 7 days. When semester labels arrive, real PR-AUC and Recall@k are recomputed and compared with the validation estimates.</li>
</ul>

<h2 class="pb">4. Code Quality &amp; Documentation</h2><div class="weight">20%</div>
<h3>4.1 Code standards</h3>
<ul>
  <li><b>Code style:</b> PEP 8, enforced and formatted by <code>ruff</code> (line length 100; rule sets E, F, I, B, UP, D). Result: <code>All checks passed!</code></li>
  <li><b>Docstrings:</b> Google convention on every public module, class and function, enforced by ruff's D rules.</li>
  <li><b>Type hints:</b> complete on parameters and return values; <code>dataclass</code> for experiment configs, results and settings.</li>
  <li><b>Comments:</b> only where intent is not obvious (e.g. why CV is computed, why a column name is stripped).</li>
  <li><b>Tests:</b> 6 pytest unit tests (validation rejects bad ranges, reproducible disjoint split, no sensitive features, zero-enrolment safety, top-k metrics, PSI detects shift): <code>6 passed</code>. A README documents layout and commands.</li>
</ul>
{SN["lint"]}
{SN["style"]}

<h3>4.2 Configuration</h3>
<p>Two kinds of configuration are separated. <code>params.yaml</code> holds every value that affects results (seed, split ratios, feature groups, tier sizes, gates, selection rule) and is version-controlled, so a run is fully described by <i>git commit + data hash + params.yaml</i>. Environment variables hold infrastructure locations and secrets (tracking URI, data path, alert address); they are never committed, and only <code>.env.example</code> is.</p>
{SN["params"]}
{SN["load"]}

<h2>5. Reproducibility &amp; Versioning Strategy</h2><div class="weight">10%</div>
<table>
  <thead><tr><th style="width:14%">Asset</th><th>Strategy</th></tr></thead>
  <tbody>
    <tr><td><b>Code</b></td><td>GitHub flow: <code>main</code> is always releasable; work happens on <code>feat/…</code> or <code>fix/…</code> branches merged through pull requests after CI (ruff + pytest) passes and one review. Releases are annotated semantic-version tags (<code>v0.2.0</code>); the commit hash is stored on every registered model and the tag on the Docker image.</td></tr>
    <tr><td><b>Data</b></td><td>Each extraction is an immutable, dated snapshot; its SHA-256 is logged on every MLflow run and model version. In production, DVC with a MinIO remote tracks snapshots: the small <code>.dvc</code> pointer is committed with the code, so checking out a commit restores the exact data.</td></tr>
    <tr><td><b>Model</b></td><td>MLflow Model Registry, registered model <code>dropout-risk</code>; versions increment automatically. Aliases: <code>challenger</code> (passed the gate), <code>champion</code> (serving), <code>previous-champion</code> (rollback). Mandatory tags: <code>exp_id</code>, <code>git_commit</code>, <code>data_sha256</code>, <code>approved_by</code>. Run naming: <code>&lt;exp_id&gt;-&lt;algorithm&gt;</code>, e.g. <code>E02-logreg</code>.</td></tr>
    <tr><td><b>Environment</b></td><td>Exact versions pinned in <code>requirements.txt</code>; Docker base image pinned to <code>python:3.11.10-slim</code>; <code>PYTHONHASHSEED</code> fixed.</td></tr>
  </tbody>
</table>
{SN["git"]}
{SN["seed"]}
{SN["docker"]}

<p><b>Evidence.</b> Re-running all twelve experiments from an empty MLflow store reproduced every metric exactly (only wall-clock time differs), and a unit test asserts that two calls to the split function return identical partitions.</p>

<h2>6. Limitations and Next Steps</h2>
<ul>
  <li><b>Proxy data.</b> The UCI dataset (Portugal) has a 32 % dropout rate, unlike the assumed 2.7 % per semester at BTEC, and lacks weekly features. Results demonstrate the pipeline and evaluation method, not expected production performance.</li>
  <li><b>Random split.</b> Without enrolment dates, the stratified split may be optimistic compared with a time-based split.</li>
  <li><b>Not yet run end-to-end in production infrastructure.</b> DAGs load in Airflow and all task functions were executed end-to-end (4,424 students scored; 442 assigned to Tier A, 664 to Tier B), but not under a live scheduler; the Docker image was not built in this environment.</li>
  <li><b>Next steps.</b> Connect real exports; switch to semester-based splitting; add weekly attendance and LMS features; re-run the E02 vs. E07 comparison under the 0.02 rule.</li>
</ul>

<h2>References</h2>
<ol>
  <li>Realinho, V., Machado, J., Baptista, L., &amp; Martins, M. V. (2022). Predicting Student Dropout and Academic Success. <i>Data</i>, 7(11), 146. Dataset: UCI Machine Learning Repository, id 697.</li>
  <li>Huyen, C. (2022). <i>Designing Machine Learning Systems</i>. O'Reilly Media.</li>
  <li>Kästner, C. (2025). <i>Machine Learning in Production: From Models to Products</i>. MIT Press.</li>
  <li>Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. <i>NeurIPS</i>.</li>
  <li>MLflow documentation: Tracking and Model Registry (aliases). Apache Airflow 2.10 documentation: DAGs, operators, trigger rules.</li>
  <li>Nguyen Thi Hong Hanh (2026). DDM501 Assignment 1: Early Warning System for Students at Risk of Dropping Out.</li>
</ol>
"""

page = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    "<title>DDM501 Assignment 2 – Nguyen Thi Hong Hanh</title>"
    f"<style>{CSS}</style></head><body>{BODY}</body></html>"
)
OUT.write_text(page, encoding="utf-8")
print("written", OUT)
