"""Airflow DAGs for the dropout early-warning system.

* ``dropout_weekly_scoring`` - every Sunday 02:00, lists ready before Monday 06:00.
* ``dropout_training``      - after each semester's labels arrive, or when
  the scoring DAG detects drift (triggered via TriggerDagRunOperator).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.email import send_email
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)
ALERT_EMAIL = "data-team@example.edu"


def notify_failure(context: dict[str, Any]) -> None:
    """Alerting: e-mail the data team with the failed task and log link."""
    ti = context["task_instance"]
    send_email(
        to=ALERT_EMAIL,
        subject=f"[dropout] {ti.dag_id}.{ti.task_id} failed",
        html_content=f"Run {context['run_id']} failed. Log: {ti.log_url}",
    )


DEFAULT_ARGS: dict[str, Any] = {
    "owner": "data-team",
    "retries": 2,  # transient DB/network errors
    "retry_delay": timedelta(minutes=10),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": notify_failure,
}


# ---------- task callables (thin wrappers around the dropout package) ----------
def extract_data(**ctx: Any) -> str:
    """Extract changed rows from SIS, LMS, attendance, finance and call log."""
    from dropout.pipeline_tasks import extract

    return extract(run_date=ctx["ds"])  # returns path of the snapshot


def validate_data(**ctx: Any) -> str:
    """Branch: continue with ML scoring if data is valid, else use the rule."""
    from dropout.pipeline_tasks import validate_snapshot

    snapshot = ctx["ti"].xcom_pull(task_ids="extract_data")
    return "build_features" if validate_snapshot(snapshot) else "fallback_rule"


def validate_training_data(**ctx: Any) -> None:
    """Fail the training run (no branch) when the labelled snapshot is invalid."""
    from dropout.pipeline_tasks import validate_snapshot

    if not validate_snapshot(ctx["ti"].xcom_pull(task_ids="snapshot_labelled_data")):
        raise ValueError("Training snapshot failed validation; keeping current champion")


def build_features(**ctx: Any) -> str:
    """Compute point-in-time weekly features with the shared feature module."""
    from dropout.pipeline_tasks import build_weekly_features

    return build_weekly_features(ctx["ti"].xcom_pull(task_ids="extract_data"))


def score_students(**ctx: Any) -> None:
    """Load the model with alias 'champion' and score every enrolled student."""
    from dropout.pipeline_tasks import score_with_champion

    score_with_champion(ctx["ti"].xcom_pull(task_ids="build_features"))


def fallback_rule(**ctx: Any) -> None:
    """Score with rule B2 so advisors still get a list when data is bad."""
    from dropout.pipeline_tasks import score_with_rule

    score_with_rule(ctx["ti"].xcom_pull(task_ids="extract_data"))


def assign_tiers(**ctx: Any) -> None:
    """Tier A (top 10%) to advisors, Tier B (10-25%) to the care team."""
    from dropout.pipeline_tasks import assign_and_publish

    assign_and_publish(run_date=ctx["ds"])


def check_drift(**ctx: Any) -> str:
    """Branch: trigger retraining when PSI on key features exceeds 0.2."""
    from dropout.pipeline_tasks import max_feature_psi

    psi = max_feature_psi(ctx["ti"].xcom_pull(task_ids="build_features"))
    return "trigger_retraining" if psi > 0.2 else "no_drift"


# ------------------------------- scoring DAG -------------------------------
with DAG(
    dag_id="dropout_weekly_scoring",
    description="Weekly risk scoring and tier assignment",
    schedule="0 2 * * 0",  # Sunday 02:00 (Asia/Ho_Chi_Minh)
    start_date=pendulum.datetime(2026, 1, 4, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["dropout", "scoring"],
) as scoring_dag:
    t_extract = PythonOperator(
        task_id="extract_data", python_callable=extract_data, sla=timedelta(hours=1)
    )
    t_validate = BranchPythonOperator(task_id="validate_data", python_callable=validate_data)
    t_features = PythonOperator(task_id="build_features", python_callable=build_features)
    t_score = PythonOperator(task_id="score_students", python_callable=score_students)
    t_fallback = PythonOperator(task_id="fallback_rule", python_callable=fallback_rule)
    t_assign = PythonOperator(
        task_id="assign_tiers",
        python_callable=assign_tiers,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,  # after ML or fallback path
        sla=timedelta(hours=4),  # ready before Monday 06:00
    )
    t_drift = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    t_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining", trigger_dag_id="dropout_training"
    )
    t_no_drift = EmptyOperator(task_id="no_drift")

    t_extract >> t_validate >> [t_features, t_fallback]
    t_features >> t_score >> t_assign
    t_fallback >> t_assign
    t_features >> t_drift >> [t_retrain, t_no_drift]


# ------------------------------ training DAG -------------------------------
def run_matrix(**_: Any) -> None:
    """Run the experiment matrix and log every configuration to MLflow."""
    from dropout.experiments import main

    main()


with DAG(
    dag_id="dropout_training",
    description="Retrain, evaluate, gate and register the dropout model",
    schedule="0 3 1 1,5,9 *",  # ~2 weeks after each of the 3 semesters ends
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["dropout", "training"],
) as training_dag:
    t_snapshot = PythonOperator(task_id="snapshot_labelled_data", python_callable=extract_data)
    t_validate_tr = PythonOperator(
        task_id="validate_training_data", python_callable=validate_training_data
    )
    t_train = PythonOperator(
        task_id="train_and_track_experiments",
        python_callable=run_matrix,
        execution_timeout=timedelta(hours=3),
    )
    t_wait_approval = EmptyOperator(task_id="await_human_approval")  # promotion is manual

    t_snapshot >> t_validate_tr >> t_train >> t_wait_approval
