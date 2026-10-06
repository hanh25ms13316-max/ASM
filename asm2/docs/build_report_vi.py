"""Build the Vietnamese reference report (HTML) from real code and results."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(
    "/tmp/claude-0/-home-claude/18a594be-8194-5506-8022-55d4b9bf1f05/scratchpad/asm2-tham-khao.html"
)


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
    """Render a titled code block."""
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
chart_b64 = base64.b64encode((ROOT / "reports" / "experiments_chart.png").read_bytes()).decode()


def f(x: float, d: int = 3) -> str:
    """Vietnamese decimal formatting (comma)."""
    return "–" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


MATRIX_ROWS = [
    ("E01", "Luật B2", "Nợ + chưa đóng học phí + không qua môn nào kỳ 1", "Đầy đủ", "–", "Baseline không ML"),
    ("E02", "Logistic regression", "max_iter=2000", "Đầy đủ", "Không", "Baseline B3, dễ giải thích"),
    ("E03", "Logistic regression", "class_weight=balanced", "Đầy đủ", "Trọng số lớp", "Cân bằng lớp có giúp không?"),
    ("E04", "Random forest", "400 cây, min_samples_leaf=3", "Đầy đủ", "Trọng số lớp", "Ensemble dạng bagging"),
    ("E05", "XGBoost", "Mặc định", "Đầy đủ", "Không", "Boosting chưa tinh chỉnh"),
    ("E06", "XGBoost", "scale_pos_weight ≈ 2,1", "Đầy đủ", "Trọng số lớp", "Ảnh hưởng tới hiệu chuẩn"),
    ("E07", "XGBoost", "Random search 25 tổ hợp, CV 5-fold", "Đầy đủ", "Không", "Tinh chỉnh siêu tham số"),
    ("E08", "XGBoost (tham số E07)", "Như E07", "Đầy đủ", "SMOTE", "Sinh mẫu thiểu số"),
    ("E09", "XGBoost (tham số E07)", "+ isotonic, cv=5", "Đầy đủ", "Không", "Hiệu chuẩn xác suất"),
    ("E10", "XGBoost (tham số E07)", "Như E07", "Nhập học + tài chính", "Không", "Dự đoán sớm nhất"),
    ("E11", "XGBoost (tham số E07)", "Như E07", "+ Học kỳ 1", "Không", "Dự đoán sớm"),
    ("E12", "XGBoost (tham số E07)", "Như E07", "Bỏ nhóm tài chính", "Không", "Đánh đổi quyền riêng tư"),
]

matrix_html = "".join(
    f'<tr><td class="id">{e}</td><td>{a}</td><td>{h}</td><td>{fe}</td><td>{b}</td><td>{q}</td></tr>'
    for e, a, h, fe, b, q in MATRIX_ROWS
)


def result_row(e: str) -> str:
    r = res.loc[e]
    gap = r.fairness_gap_at_25
    gap_cell = f'<td class="num{" warn" if gap > 0.10 else ""}">{f(gap)}</td>'
    cv = "–" if pd.isna(r.cv_pr_auc_mean) else f"{f(r.cv_pr_auc_mean)} ± {f(r.cv_pr_auc_std)}"
    cls = ' class="pick"' if e == summary["selected"] else ""
    return (
        f'<tr{cls}><td class="id">{e}</td><td class="num">{f(r.pr_auc)}</td><td class="num">{cv}</td>'
        f'<td class="num">{f(r.roc_auc)}</td><td class="num">{f(r.recall_at_10)}</td>'
        f'<td class="num">{f(r.precision_at_10)}</td><td class="num">{f(r.recall_at_25)}</td>'
        f'<td class="num">{f(r.brier)}</td><td class="num">{f(r.ece)}</td>{gap_cell}'
        f'<td class="num">{f(r.train_seconds, 2)}</td></tr>'
    )


results_html = "".join(result_row(e) for e in res.index)
tm = summary["test_metrics"]
hp = summary["tuned_hyperparams"]

STAGES = [
    ("1", "Ingestion", "Trích xuất hằng đêm từ 5 hệ thống, lưu thành snapshot có ngày và mã băm SHA-256",
     "File CSV thô + data_sha256", "File đọc được, đủ cột bắt buộc", ""),
    ("2", "Validation", "Kiểm tra schema bằng pandera, khoảng giá trị, tỷ lệ thiếu, số dòng tối thiểu",
     "DataFrame hợp lệ", "Thiếu ≤ 5 %, ≥ 1.000 dòng, giá trị trong khoảng", "Không đạt → dừng, cảnh báo, dùng luật B2"),
    ("3", "Preprocessing", "Tạo nhãn nhị phân (Dropout = 1), chia train/val/test 60/20/20 có phân tầng, seed cố định",
     "3 tập dữ liệu", "Tỷ lệ bỏ học 3 tập chênh < 1 điểm %; không trùng dòng", ""),
    ("4", "Feature engineering", "Chọn nhóm đặc trưng, loại thuộc tính nhạy cảm, thêm tỷ lệ qua môn, xu hướng điểm, rủi ro tài chính; one-hot và chuẩn hóa",
     "Ma trận đặc trưng", "Không có cột nhạy cảm; cùng mã nguồn cho train và chấm điểm", ""),
    ("5", "Training", "Chạy 12 cấu hình E01–E12, mỗi cấu hình là một MLflow run; tinh chỉnh bằng random search",
     "Pipeline đã huấn luyện + run trong MLflow", "Run log đủ tham số, metric, artifact", ""),
    ("6", "Evaluation", "Tính PR-AUC, Recall/Precision@k, Brier, ECE, chênh lệch công bằng, CV ± SD; áp dụng luật chọn mô hình",
     "Bảng kết quả + mô hình được chọn", "PR-AUC ≥ 0,80; chênh công bằng ≤ 0,10; không kém champion quá 0,01", "Không đạt → giữ champion cũ"),
    ("7", "Deployment", "Đăng ký vào registry với alias challenger, người duyệt chuyển sang champion; DAG tuần chấm điểm bằng champion",
     "Model version + danh sách Tier A/B", "Phải có người duyệt; giữ alias previous-champion để rollback", ""),
]
stages_html = "".join(
    f'<li class="stage"><div class="num">{n}</div><div><h4>{name}</h4><p>{op}</p>'
    f'<p class="chips"><span class="chip out">Đầu ra: {out}</span><span class="chip gate">Gate: {gate}</span>'
    + (f'<span class="chip fail">{fail}</span>' if fail else "")
    + "</p></div></li>"
    for n, name, op, out, gate, fail in STAGES
)

CSS = """
:root{--bg:#f7f7f4;--surface:#fff;--fg:#1f2421;--muted:#5f6862;--line:#dcdfd8;--accent:#2f5d8a;--accent-soft:#e6eef6;
--mark:#7a4e00;--mark-soft:#fbf1dc;--good:#256d3f;--good-soft:#e5f2e9;--bad:#a1311f;--bad-soft:#f8e4df;--codebg:#f3f4f1;
--display:"Lora",Georgia,serif;--body:"Be Vietnam Pro",system-ui,-apple-system,"Segoe UI",Arial,sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141716;--surface:#1c201e;--fg:#e6eae7;--muted:#a3aca6;--line:#2f3532;--accent:#8db8e3;--accent-soft:#1f2c39;--mark:#f0c46e;--mark-soft:#342a17;--good:#7fcf9a;--good-soft:#1c3024;--bad:#f0a08f;--bad-soft:#3a201b;--codebg:#111413;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141716;--surface:#1c201e;--fg:#e6eae7;--muted:#a3aca6;--line:#2f3532;--accent:#8db8e3;--accent-soft:#1f2c39;--mark:#f0c46e;--mark-soft:#342a17;--good:#7fcf9a;--good-soft:#1c3024;--bad:#f0a08f;--bad-soft:#3a201b;--codebg:#111413;color-scheme:dark}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--body);font-size:16px;line-height:1.7}
.wrap{max-width:860px;margin:0 auto;padding:28px 16px 72px}
.cover{display:grid;gap:12px;padding-bottom:22px;border-bottom:1px solid var(--line)}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-family:var(--display);font-size:clamp(28px,5vw,40px);line-height:1.2;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-size:26px;line-height:1.3;margin:0}
h3{font-family:var(--display);font-size:19px;line-height:1.35;margin:0;font-weight:600}
h4{font-size:15px;margin:0;font-weight:700}p{margin:0}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:6px 18px;font-size:14px;color:var(--muted)}.meta b{color:var(--fg)}
.notice{background:var(--mark-soft);border-radius:10px;padding:12px 14px;font-size:14px}.notice strong{color:var(--mark)}
nav.toc{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);margin-inline:-16px;padding:10px 16px;display:flex;gap:6px;overflow-x:auto}
nav.toc a{flex:none;font-size:13px;text-decoration:none;color:var(--fg);padding:5px 10px;border:1px solid var(--line);border-radius:999px;background:var(--surface)}
nav.toc a:hover,nav.toc a:focus-visible{border-color:var(--accent);color:var(--accent);outline:none}
section{display:grid;gap:16px;padding-top:40px;scroll-margin-top:60px}.sub{display:grid;gap:12px;padding-top:8px}
.weight{font-family:var(--mono);font-size:13px;color:var(--accent);font-weight:600}
.prose{display:grid;gap:12px}.prose p,.prose li{max-width:70ch}
ul,ol{margin:0;padding-left:22px;display:grid;gap:6px}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.5}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:none}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600;background:var(--bg);white-space:nowrap}
td.num{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}
td.warn{color:var(--bad);font-weight:600}tr.pick td{background:var(--good-soft)}
.id{font-family:var(--mono);font-size:12.5px;color:var(--accent);font-weight:600;white-space:nowrap}
code{font-family:var(--mono);font-size:.86em;background:var(--accent-soft);padding:1px 5px;border-radius:4px}
figure{margin:0}figure.code{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--codebg)}
figure.code figcaption{font-size:12.5px;font-family:var(--mono);color:var(--muted);padding:7px 12px;border-bottom:1px solid var(--line);background:var(--surface)}
figure.code pre{margin:0;padding:12px 14px;overflow-x:auto;font-size:12.5px;line-height:1.55}
figure.code code{background:none;padding:0;font-size:inherit}
.diagram{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;overflow-x:auto;color:#1f2421}
.cap{font-size:13px;color:var(--muted)}
.stages{list-style:none;padding:0;margin:0;display:grid;gap:0}
.stage{display:grid;grid-template-columns:36px 1fr;gap:12px;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:10px;position:relative}
.stage+.stage{margin-top:22px}.stage+.stage::before{content:"↓";position:absolute;top:-21px;left:24px;color:var(--muted);font-size:15px}
.stage .num{width:30px;height:30px;border-radius:50%;background:var(--accent);color:var(--surface);display:grid;place-items:center;font-weight:700;font-size:14px}
.stage p{font-size:14px}.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{font-size:12.5px;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.chip.out{background:var(--accent-soft)}.chip.gate{background:var(--good-soft)}.chip.fail{background:var(--bad-soft);color:var(--bad)}
.callout{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--good);border-radius:10px;padding:14px 16px;display:grid;gap:8px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-family:var(--mono);font-size:20px}.kpi span{font-size:12.5px;color:var(--muted)}
img.chart{width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
footer{margin-top:44px;color:var(--muted);font-size:13px}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
"""

SNIP0 = code(lines(".env.example", "MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_ROOT"), ".env.example: cấu hình MLflow qua biến môi trường")
SNIP1 = code(lines("src/dropout/experiments.py", "def main", r"splits = load_splits"), "src/dropout/experiments.py: thiết lập experiment")
SNIP2 = code(lines("src/dropout/train.py", r"with mlflow.start_run", r"start = time.perf_counter") + "\n        ...\n" + lines("src/dropout/train.py", r"mlflow.log_metrics\(metrics\)", r"input_example"), "src/dropout/train.py: log tham số, metric và artifact cho mỗi run")
SNIP3 = code(lines("src/dropout/registry.py", "def register_candidate", r"Promoted"), "src/dropout/registry.py: đăng ký phiên bản và chuyển alias")
SNIP4 = code(lines("dags/dropout_training_dag.py", "^DEFAULT_ARGS", r"^}"), "dags/dropout_training_dag.py: tham số mặc định (retry, timeout, cảnh báo)")
SNIP5 = code(lines("dags/dropout_training_dag.py", r"^def validate_data", r'return "build_features"'), "Định nghĩa task: hàm mỏng gọi vào package dropout")
SNIP6 = code(lines("dags/dropout_training_dag.py", r"^with DAG\($", r"t_features >> t_drift"), "Định nghĩa DAG chấm điểm và các phụ thuộc")
SNIP7 = code(lines("src/dropout/evaluate.py", r"^def top_k_mask", r"return float\(y\[top_k_mask\(scores, fraction\)\]\.sum"), "src/dropout/evaluate.py: ví dụ hàm có type hints và docstring")
SNIP8 = code(lines("params.yaml", "^project:", "^data:") + "\n  ...\n" + lines("params.yaml", "^evaluation:", "^quality_gate:") + "\n" + lines("params.yaml", "min_pr_auc", "max_regression"), "params.yaml (trích)")
SNIP9 = code(lines("src/dropout/config.py", r"^def load_settings", r"^def set_global_seed", include_end=False).rstrip(), "src/dropout/config.py: đọc biến môi trường và file cấu hình")
SNIP10 = code('''# Nhánh tính năng -> PR -> merge -> gắn tag phát hành
git switch -c feat/early-warning-features
git commit -m "Add approval-rate and grade-trend features"
git push -u origin feat/early-warning-features   # mở PR, CI chạy ruff + pytest
git switch main && git pull
git tag -a v0.2.0 -m "Experiment matrix E01-E12, LR champion"
git push origin v0.2.0

# Phiên bản dữ liệu với DVC (production)
dvc add data/raw/students.csv && git add data/raw/students.csv.dvc
git commit -m "Data snapshot 2026-S3" && dvc push''', "Quản lý phiên bản code và dữ liệu")
SNIP11 = code(lines("src/dropout/config.py", r"^def set_global_seed", None), "src/dropout/config.py: cố định seed (thêm random_state=seed cho split, CV, SMOTE và mọi mô hình)")
SNIP12 = code((ROOT / "Dockerfile").read_text().strip(), "Dockerfile")

BODY = f"""
<div class="wrap">
<header class="cover">
  <div class="eyebrow">DDM501 · Individual Assignment 2 · ML Pipeline Design &amp; MLOps Analysis</div>
  <h1>Pipeline và MLOps cho hệ thống cảnh báo sớm sinh viên bỏ học</h1>
  <div class="meta">
    <span>Sinh viên: <b>Nguyễn Thị Hồng Hạnh</b></span><span>Mã sinh viên: <b>25MS13316</b></span>
    <span>Tiếp nối: <b>Assignment 1</b></span><span>Dữ liệu thí nghiệm: <b>UCI, 4.424 sinh viên</b></span>
  </div>
  <div class="notice"><strong>Bản tham khảo để đọc hiểu.</strong> Code và số liệu trong bài là <strong>thật</strong>: mình đã dựng pipeline và chạy 12 thí nghiệm trên bộ dữ liệu công khai của UCI. Bộ này là dữ liệu đại diện, không phải dữ liệu của BTEC FPT. Bài nộp nên được bạn tự viết lại bằng tiếng Anh theo cách hiểu của mình. Bạn cũng nên kiểm tra quy định dùng AI của môn.</div>
</header>
<nav class="toc" aria-label="Mục lục">
  <a href="#p0">0. Nối tiếp ASM1</a><a href="#p1">1. Pipeline</a><a href="#p2">2. Thí nghiệm</a>
  <a href="#p3">3. Điều phối</a><a href="#p4">4. Chất lượng code</a><a href="#p5">5. Tái lập</a><a href="#p6">6. Hạn chế</a>
</nav>

<section id="p0">
  <h2>0. Nối tiếp Assignment 1</h2>
  <div class="prose">
    <p><strong>Nhắc lại bài toán.</strong> Mỗi tuần, hệ thống chấm điểm nguy cơ bỏ học cho mọi sinh viên đang học tại Cao đẳng Anh Quốc BTEC FPT (số liệu giả định: 12.000 sinh viên, 3 học kỳ mỗi năm). Sinh viên được chia làm hai nhóm: <strong>Tier A</strong> (10 % rủi ro cao nhất) do cố vấn học tập liên hệ, <strong>Tier B</strong> (từ 10 % đến 25 %) do bộ phận chăm sóc sinh viên gọi hỏi thăm. Kiến trúc gồm hai luồng: <strong>huấn luyện</strong> chạy mỗi học kỳ, <strong>chấm điểm theo lô</strong> chạy hằng tuần. Hai luồng dùng chung một module tính đặc trưng, và có MLflow registry, cổng chất lượng, người duyệt, luật dự phòng.</p>
    <p><strong>Những điều chỉnh so với ASM1</strong>, sau khi học Session 3–5:</p>
    <ul>
      <li><strong>Dữ liệu thí nghiệm:</strong> trường chưa có dữ liệu thật, nên dùng bộ <em>Predict Students' Dropout and Academic Success</em> của UCI làm dữ liệu đại diện. Bộ này có 4.424 sinh viên, 36 thuộc tính, tỷ lệ bỏ học 32,1 %. Điểm khác so với thiết kế ASM1: không có đặc trưng theo tuần, nên các nhóm đặc trưng "nhập học / học kỳ 1 / học kỳ 2" được dùng để mô phỏng việc dự đoán sớm hay muộn.</li>
      <li><strong>Cách chia dữ liệu:</strong> bộ UCI không có ngày nhập học, nên chia ngẫu nhiên có phân tầng thay vì chia theo thời gian. Khi có dữ liệu thật thì quay về cách chia theo học kỳ như ASM1.</li>
      <li><strong>Luật chọn mô hình được viết thành code:</strong> trade-off 5.1 của ASM1 ("chỉ chọn gradient boosting khi PR-AUC hơn logistic regression ít nhất 0,02") nay là hàm <code>select_model</code> với tham số trong <code>params.yaml</code>.</li>
      <li><strong>Bổ sung:</strong> đo PR-AUC bằng cross-validation (CV) kèm độ lệch chuẩn để phân biệt chênh lệch thật với nhiễu; thêm alias <code>previous-champion</code> để rollback; tính drift bằng PSI ngay trong DAG chấm điểm.</li>
    </ul>
  </div>
</section>

<section id="p1">
  <div class="weight">PHẦN 1 · 25%</div>
  <h2>1. Thiết kế pipeline</h2>
  <div class="sub">
    <h3>1.1 Sơ đồ pipeline</h3>
    <ol class="stages">{stages_html}</ol>
    <p class="cap">Hình 1. Bảy giai đoạn của pipeline huấn luyện. Mỗi giai đoạn nhận đầu ra của giai đoạn trước và phải qua cổng chất lượng mới được đi tiếp. Luồng chấm điểm hằng tuần dùng lại các giai đoạn 1, 2 và 4 với cùng mã nguồn, rồi chấm điểm bằng mô hình champion (Hình 3).</p>
  </div>
  <div class="sub">
    <h3>1.2 Đặc tả từng giai đoạn</h3>
    <div class="tbl"><table>
      <thead><tr><th>Giai đoạn</th><th>Mục đích</th><th>Đầu vào → đầu ra</th><th>Thao tác chính (module)</th><th>Cổng chất lượng</th></tr></thead>
      <tbody>
        <tr><td><b>1. Ingestion</b></td><td>Có bản dữ liệu cố định, truy vết được</td><td>Export từ các hệ thống nguồn → snapshot CSV + SHA-256</td><td><code>data.ingest</code>, <code>file_sha256</code>; chuẩn hóa tên cột (export gốc có ký tự tab thừa)</td><td>Đọc được file, đủ cột</td></tr>
        <tr><td><b>2. Validation</b></td><td>Chặn dữ liệu hỏng trước khi train hoặc chấm điểm</td><td>DataFrame thô → DataFrame hợp lệ</td><td><code>data.validate</code>: schema pandera (nhãn, tuổi 15–80, điểm 0–20, cờ 0/1), tỷ lệ thiếu, số dòng</td><td>Lỗi thì ném <code>DataValidationError</code> → pipeline dừng, DAG chuyển sang luật B2</td></tr>
        <tr><td><b>3. Preprocessing</b></td><td>Tạo nhãn và tập đánh giá trung thực</td><td>DataFrame → train 2.654 / val 885 / test 885</td><td><code>make_label</code>; <code>split</code> phân tầng, <code>random_state=42</code></td><td>Tỷ lệ bỏ học bằng nhau ở 3 tập (32,1 %); không trùng chỉ số (có test)</td></tr>
        <tr><td><b>4. Feature engineering</b></td><td>Biến dữ liệu thô thành tín hiệu nguy cơ</td><td>Cột thô → ma trận đặc trưng</td><td><code>select_columns</code> (bỏ Gender, Nacionality, Marital status, International); <code>EngineeredFeatures</code> (tỷ lệ qua môn, xu hướng điểm, rủi ro tài chính); <code>ColumnTransformer</code></td><td>Không có thuộc tính nhạy cảm (có test); chia 0 trả về 0</td></tr>
        <tr><td><b>5. Training</b></td><td>Thử có hệ thống nhiều cấu hình</td><td>Ma trận đặc trưng → pipeline đã fit</td><td><code>train.run_experiment</code>: một cấu hình là một MLflow run; <code>RandomizedSearchCV</code> 25 tổ hợp</td><td>Mỗi run log đủ tham số, metric, mô hình, danh sách cột</td></tr>
        <tr><td><b>6. Evaluation</b></td><td>Chọn mô hình theo metric vận hành</td><td>Dự đoán trên val → metric + mô hình được chọn</td><td><code>evaluate</code>; <code>select_model</code>; <code>passes_quality_gate</code></td><td>PR-AUC ≥ 0,80; chênh recall theo giới ≤ 0,10; không kém champion quá 0,01</td></tr>
        <tr><td><b>7. Deployment</b></td><td>Đưa mô hình vào dùng có kiểm soát</td><td>Run → model version → danh sách Tier A/B</td><td><code>register_candidate</code> (alias challenger); <code>promote</code> (champion, giữ previous-champion); <code>score_with_champion</code>, <code>assign_and_publish</code></td><td>Bắt buộc có người duyệt; test set chỉ dùng một lần cho mô hình được chọn</td></tr>
      </tbody>
    </table></div>
  </div>
  <div class="sub">
    <h3>1.3 Lý do thiết kế</h3>
    <div class="prose"><ul>
      <li><strong>Cấu trúc:</strong> mỗi giai đoạn có đầu vào và đầu ra rõ ràng, nên khi lỗi biết ngay hỏng ở bước nào và có thể chạy lại riêng bước đó. Validation đặt trước mọi bước khác, vì dữ liệu hỏng mà lọt qua thì mọi bước sau đều sai mà không ai biết.</li>
      <li><strong>Module và tái sử dụng:</strong> mỗi giai đoạn là một hàm thuần trong package <code>dropout</code>. Hàm nhận và trả đường dẫn file hoặc đối tượng, không phụ thuộc Airflow. Cùng một hàm được gọi từ script thí nghiệm, từ DAG và từ unit test. Đặc biệt, <code>EngineeredFeatures</code> nằm <em>bên trong</em> pipeline scikit-learn được lưu vào MLflow, nên lúc chấm điểm không thể vô tình tính đặc trưng khác lúc train (tránh training-serving skew). Muốn thêm thuật toán chỉ cần thêm một nhánh trong <code>build_estimator</code>; muốn thêm thí nghiệm chỉ cần thêm một dòng vào <code>MATRIX</code>.</li>
      <li><strong>Xử lý lỗi và phục hồi:</strong> (1) dữ liệu không hợp lệ thì dừng và chấm bằng luật B2, cố vấn vẫn có danh sách; (2) lỗi tạm thời như mất kết nối CSDL thì Airflow tự thử lại 2 lần, thời gian chờ tăng dần; (3) mô hình mới không qua gate thì giữ champion cũ và ghi lý do; (4) mô hình mới có vấn đề sau khi triển khai thì chuyển alias champion về <code>previous-champion</code>, không cần build lại; (5) snapshot có ngày nên chạy lại tuần nào cũng cho đúng kết quả tuần đó.</li>
      <li><strong>Khả năng mở rộng:</strong> chấm 4.424 sinh viên mất dưới 1 giây. Ở quy mô 12.000–50.000 sinh viên, thời gian tăng tuyến tính và có thể chia theo ngành để chạy song song. Phần tốn thời gian nhất là tinh chỉnh (21 giây với E07) chỉ chạy mỗi học kỳ, và có thể tăng <code>n_jobs</code> hoặc chuyển sang Optuna khi không gian tham số lớn hơn. Khi tracking có nhiều người dùng, đổi <code>MLFLOW_TRACKING_URI</code> từ SQLite sang PostgreSQL + MinIO mà không phải sửa code (giống dự án nhóm).</li>
    </ul></div>
  </div>
</section>

<section id="p2">
  <div class="weight">PHẦN 2 · 25%</div>
  <h2>2. Theo dõi thí nghiệm và phân tích metric</h2>
  <div class="sub">
    <h3>2.1 Thiết kế thí nghiệm</h3>
    <div class="prose"><ul>
      <li><strong>Các biến thí nghiệm:</strong> <em>thuật toán</em> (luật, logistic regression, random forest, XGBoost); <em>siêu tham số</em> (mặc định hoặc tinh chỉnh bằng random search); <em>đặc trưng</em> (nhập học, học kỳ 1, học kỳ 2, tài chính); <em>tiền xử lý</em> (không cân bằng, trọng số lớp, SMOTE, hiệu chuẩn isotonic).</li>
      <li><strong>Baseline:</strong> E01 là luật B2, chính là luật dự phòng trong ASM1. E02 là logistic regression (B3). Mô hình phức tạp phải hơn E02 ít nhất 0,02 PR-AUC mới được chọn.</li>
      <li><strong>Nguyên tắc so sánh công bằng:</strong> mọi cấu hình dùng chung một cách chia dữ liệu và seed 42. Chọn mô hình dựa trên tập validation. Tập test chỉ dùng <strong>một lần</strong>, cho mô hình cuối cùng.</li>
    </ul></div>
    <div class="tbl"><table>
      <thead><tr><th>ID</th><th>Thuật toán</th><th>Siêu tham số</th><th>Đặc trưng</th><th>Cân bằng lớp</th><th>Câu hỏi cần trả lời</th></tr></thead>
      <tbody>{matrix_html}</tbody>
    </table></div>
  </div>

  <div class="sub">
    <h3>2.2 Chiến lược metric</h3>
    <div class="tbl"><table>
      <thead><tr><th>Loại</th><th>Metric</th><th>Vì sao</th></tr></thead>
      <tbody>
        <tr><td><b>Chính</b></td><td>PR-AUC (validation) + CV 5-fold ± SD</td><td>Đo khả năng đưa sinh viên có nguy cơ lên đầu danh sách, tập trung vào lớp bỏ học. Độ lệch chuẩn CV cho biết chênh lệch giữa hai cấu hình có lớn hơn nhiễu không.</td></tr>
        <tr><td><b>Chính (vận hành)</b></td><td>Recall@10 %, Recall@25 %</td><td>Ứng với Tier A (cố vấn) và Tier A+B (cộng bộ phận chăm sóc): bắt được bao nhiêu phần trăm số ca bỏ học trong phạm vi năng lực liên hệ.</td></tr>
        <tr><td>Phụ</td><td>Precision@10 %, ROC-AUC</td><td>Danh sách của cố vấn có "trúng" không; ROC-AUC để so với các nghiên cứu khác.</td></tr>
        <tr><td>Phụ</td><td>Brier, ECE (sai số hiệu chuẩn)</td><td>Dashboard cộng xác suất để ước lượng số ca theo ngành, nên xác suất phải khớp thực tế.</td></tr>
        <tr><td>Phụ (ràng buộc)</td><td>Chênh Recall@25 % theo giới tính</td><td>Điều kiện công bằng của ASM1: chênh quá 0,10 thì không được chọn.</td></tr>
        <tr><td>Phụ</td><td>Thời gian huấn luyện</td><td>Chi phí vận hành mỗi học kỳ.</td></tr>
      </tbody>
    </table></div>
    <div class="prose">
      <p><strong>Nối metric mô hình với mục tiêu kinh doanh</strong> (tháp mục tiêu ở ASM1): Recall@10 % × số ca bỏ học = số sinh viên có nguy cơ được cố vấn gặp; nhân tiếp với tỷ lệ giữ chân giả định 30 % ra số sinh viên ở lại. Recall@25 % tính thêm phần do bộ phận chăm sóc phủ. Lưu ý quan trọng: trong dữ liệu UCI có tới 32 % sinh viên bỏ học, nên danh sách 10 % chỉ chứa được <strong>tối đa 31 % số ca</strong>. Vì vậy Recall@10 % chỉ khoảng 0,30, dù danh sách gần như đúng hoàn toàn (Precision@10 % ≈ 0,95). Ở BTEC với tỷ lệ nền giả định 2,7 %, ràng buộc ngược lại: danh sách đủ chỗ cho mọi ca, còn precision sẽ thấp hơn nhiều. Định nghĩa metric giữ nguyên, chỉ con số thay đổi.</p>
    </div>
  </div>

  <div class="sub">
    <h3>2.3 Code: thiết lập MLflow, log và đăng ký mô hình</h3>
    {SNIP0}
    {SNIP1}
    {SNIP2}
    {SNIP3}
    <p class="cap">Mỗi phiên bản mô hình được gắn tag <code>exp_id</code>, <code>git_commit</code>, <code>data_sha256</code> và <code>approved_by</code>, nên từ một phiên bản truy ngược được đúng code và dữ liệu đã tạo ra nó. Kết quả thật trong lần chạy này: <code>dropout-risk</code> phiên bản 1, aliases <code>champion</code> và <code>challenger</code>, git_commit <code>bbc0168</code>, data_sha256 <code>19737f02127e</code>.</p>
  </div>

  <div class="sub">
    <h3>2.4 Kết quả thí nghiệm (tập validation, n = 885, 284 ca bỏ học)</h3>
    <div class="tbl"><table>
      <thead><tr><th>ID</th><th>PR-AUC</th><th>CV PR-AUC ± SD</th><th>ROC-AUC</th><th>Recall@10%</th><th>Prec@10%</th><th>Recall@25%</th><th>Brier</th><th>ECE</th><th>Chênh giới</th><th>Giây</th></tr></thead>
      <tbody>{results_html}</tbody>
    </table></div>
    <p class="cap">Dòng tô xanh là mô hình được chọn. Số màu đỏ là chênh lệch công bằng vượt ngưỡng 0,10, nên cấu hình đó không đủ điều kiện. Recall@10 % tối đa có thể đạt là 0,31 (88 chỗ trong danh sách / 284 ca). Chạy lại toàn bộ lần hai cho ra metric giống hệt.</p>
    <img class="chart" alt="Biểu đồ PR-AUC của 12 thí nghiệm và so sánh theo thời điểm dự đoán" src="data:image/png;base64,{chart_b64}">
    <p class="cap">Hình 2. Trái: PR-AUC trên validation (chấm) và CV 5-fold ±1 độ lệch chuẩn (vạch xám); E02 tô xanh. Phải: PR-AUC theo lượng dữ liệu có tại thời điểm dự đoán (đường đứt nét là mức ngẫu nhiên 0,32).</p>
  </div>

  <div class="sub">
    <h3>2.5 Phân tích</h3>
    <div class="prose"><ul>
      <li><strong>Mọi mô hình ML đều hơn hẳn luật:</strong> E01 đạt PR-AUC {f(res.loc["E01"].pr_auc)}, còn các mô hình ML dùng đủ đặc trưng đạt {f(res.loc["E05"].pr_auc)}–{f(res.loc["E09"].pr_auc)}. Recall@25 % tăng từ {f(res.loc["E01"].recall_at_25)} lên khoảng 0,69. Điều này xác nhận câu hỏi "có cần ML không" ở ASM1.</li>
      <li><strong>Giữa các mô hình ML, chênh lệch nằm trong nhiễu:</strong> E02–E09 chỉ cách nhau tối đa {f(res.loc["E09"].pr_auc - res.loc["E05"].pr_auc)} PR-AUC trên validation, trong khi độ lệch chuẩn CV khoảng 0,011–0,018. Tốt nhất trên validation là E09 ({f(res.loc["E09"].pr_auc)}), nhưng chỉ hơn logistic regression E02 {f(res.loc["E09"].pr_auc - res.loc["E02"].pr_auc, 4)}. Trên CV, XGBoost đã tinh chỉnh (E07 {f(res.loc["E07"].cv_pr_auc_mean)}) hơn E02 ({f(res.loc["E02"].cv_pr_auc_mean)}) khoảng 0,01, vẫn chưa tới một độ lệch chuẩn. Dữ liệu dạng bảng, ít dòng, và các đặc trưng học vụ đã gần tuyến tính, nên mô hình phức tạp không có nhiều đất để thắng.</li>
      <li><strong>Cân bằng lớp làm hỏng xác suất:</strong> thêm <code>class_weight</code> vào logistic regression (E03) không tăng PR-AUC nhưng làm ECE tăng từ {f(res.loc["E02"].ece)} lên {f(res.loc["E03"].ece)}. XGBoost với <code>scale_pos_weight</code> (E06) cũng có ECE {f(res.loc["E06"].ece)}. Đây là bài học giống hệt dự án Churn của nhóm: trọng số lớp đẩy xác suất lên cao hơn thực tế, mà danh sách top-k lại không cần đến nó vì chỉ xếp hạng.</li>
      <li><strong>SMOTE và hiệu chuẩn isotonic không mang lại lợi ích rõ:</strong> E08 (SMOTE) có CV cao nhất ({f(res.loc["E08"].cv_pr_auc_mean)}) nhưng validation không hơn, và phải thêm dữ liệu tổng hợp. E09 (isotonic) có ECE {f(res.loc["E09"].ece)}, <em>kém hơn</em> E07 không hiệu chuẩn ({f(res.loc["E07"].ece)}). XGBoost với logloss vốn đã khá chuẩn, còn isotonic trên mỗi fold khoảng 530 mẫu thì bị nhiễu. Vì vậy chỉ thêm bước hiệu chuẩn khi đo được là nó có ích.</li>
      <li><strong>Dự đoán sớm và độ chính xác</strong> (trade-off 5.5 của ASM1): chỉ dùng dữ liệu nhập học (E10) cho PR-AUC {f(res.loc["E10"].pr_auc)}, gần bằng luật. Thêm kết quả học kỳ 1 (E11) lên {f(res.loc["E11"].pr_auc)}, thêm học kỳ 2 (E07) lên {f(res.loc["E07"].pr_auc)}. Phần lớn tín hiệu đến từ <strong>kết quả học tập đầu tiên</strong>. Điều này ủng hộ thiết kế chấm điểm hằng tuần, với độ tin cậy tăng dần khi có điểm giữa kỳ.</li>
      <li><strong>Quyền riêng tư và độ chính xác</strong> (trade-off 5.4): bỏ hai cột tài chính (E12) làm PR-AUC giảm {f(res.loc["E07"].pr_auc - res.loc["E12"].pr_auc)} và chênh công bằng tăng lên {f(res.loc["E12"].fairness_gap_at_25)}, vượt ngưỡng. Tình trạng học phí là tín hiệu quan trọng, nên nên giữ nhưng bảo vệ chặt: giả danh hóa, phân quyền, không hiển thị số tiền cho người liên hệ.</li>
      <li><strong>Công bằng:</strong> Random forest (E04, {f(res.loc["E04"].fairness_gap_at_25)}) và E12 vượt ngưỡng 0,10 nên bị loại tự động, dù PR-AUC của E04 cao. Nếu chỉ nhìn PR-AUC sẽ không thấy vấn đề này.</li>
      <li><strong>Chi phí:</strong> E07 mất {f(res.loc["E07"].train_seconds, 1)} giây để tinh chỉnh, trong khi E02 mất {f(res.loc["E02"].train_seconds, 2)} giây. Ở quy mô này cả hai đều rẻ, nên chi phí không phải yếu tố quyết định.</li>
    </ul></div>
  </div>

  <div class="sub">
    <h3>2.6 Khuyến nghị</h3>
    <div class="callout">
      <p><strong>Chọn E02: logistic regression, không cân bằng lớp, đủ đặc trưng.</strong> Mô hình tốt nhất theo PR-AUC (E09) chỉ hơn {f(res.loc["E09"].pr_auc - res.loc["E02"].pr_auc, 4)}, thấp hơn nhiều so với ngưỡng 0,02 đặt ra ở ASM1. E02 dễ giải thích cho cố vấn (hệ số cho biết trực tiếp yếu tố nào làm tăng nguy cơ), xác suất đã chuẩn (ECE {f(res.loc["E02"].ece)}), chênh công bằng {f(res.loc["E02"].fairness_gap_at_25)} trong ngưỡng, và huấn luyện tức thì. Code đã tự đưa ra quyết định này qua <code>select_model</code>.</p>
      <div class="kpis">
        <div class="kpi"><b>{f(tm["pr_auc"])}</b><span>PR-AUC trên test</span></div>
        <div class="kpi"><b>{f(tm["precision_at_10"])}</b><span>Precision@10 % (Tier A)</span></div>
        <div class="kpi"><b>{f(tm["recall_at_25"])}</b><span>Recall@25 % (Tier A+B)</span></div>
        <div class="kpi"><b>{f(tm["ece"])}</b><span>ECE trên test</span></div>
        <div class="kpi"><b>{f(tm["fairness_gap_at_25"])}</b><span>Chênh recall theo giới</span></div>
      </div>
      <p>Trên tập test (885 sinh viên, chỉ dùng một lần), kết quả tương đương validation nên không có dấu hiệu overfit vào validation. Phương án dự phòng là E07 (XGBoost, tham số tốt nhất: {", ".join(f"{k}={v}" for k, v in hp.items())}). Sẽ chuyển sang E07 khi dữ liệu thật có đặc trưng theo tuần, vì các tương tác phi tuyến (ví dụ vắng học tăng <em>và</em> nợ học phí) có thể giúp nó hơn E02 trên 0,02.</p>
    </div>
  </div>
</section>

<section id="p3">
  <div class="weight">PHẦN 3 · 20%</div>
  <h2>3. Thiết kế điều phối (Airflow)</h2>
  <div class="sub">
    <h3>3.1 Kiến trúc DAG</h3>
    <div class="diagram">{svg("scoring")}</div>
    <p class="cap">Hình 3. DAG <code>dropout_weekly_scoring</code>, chạy Chủ nhật 02:00 (giờ Việt Nam).</p>
    <div class="diagram">{svg("training")}</div>
    <p class="cap">Hình 4. DAG <code>dropout_training</code>, chạy sau mỗi học kỳ hoặc khi DAG chấm điểm phát hiện drift. Cả hai DAG đã được Airflow 2.10.3 nạp thành công (DagBag không có lỗi import; 9 và 4 task, múi giờ Asia/Ho_Chi_Minh).</p>
    <div class="tbl"><table>
      <thead><tr><th>Task</th><th>Việc làm</th><th>Phụ thuộc</th></tr></thead>
      <tbody>
        <tr><td class="id">extract_data</td><td>Copy export đêm thành snapshot có ngày, trả đường dẫn qua XCom</td><td>Không, là task đầu tiên</td></tr>
        <tr><td class="id">validate_data</td><td>Branch: hợp lệ thì sang <code>build_features</code>, không thì sang <code>fallback_rule</code></td><td>Cần snapshot của extract</td></tr>
        <tr><td class="id">build_features</td><td>Tính đặc trưng bằng module dùng chung, lưu file</td><td>Chỉ chạy khi dữ liệu hợp lệ</td></tr>
        <tr><td class="id">score_students</td><td>Tải mô hình <code>models:/dropout-risk@champion</code>, chấm toàn bộ sinh viên</td><td>Cần file đặc trưng</td></tr>
        <tr><td class="id">fallback_rule</td><td>Chấm bằng luật B2 để vẫn có danh sách</td><td>Chỉ chạy khi dữ liệu không hợp lệ</td></tr>
        <tr><td class="id">assign_tiers</td><td>Xếp hạng, gán Tier A cho cố vấn, Tier B cho bộ phận chăm sóc, xuất danh sách</td><td>Chạy khi <em>một trong hai</em> nhánh chấm điểm thành công (<code>NONE_FAILED_MIN_ONE_SUCCESS</code>)</td></tr>
        <tr><td class="id">check_drift</td><td>Branch: PSI lớn nhất trên các đặc trưng &gt; 0,2 thì kích hoạt huấn luyện lại</td><td>Cần file đặc trưng; chạy song song với chấm điểm</td></tr>
        <tr><td class="id">trigger_retraining</td><td>Gọi DAG <code>dropout_training</code></td><td>Chỉ khi có drift</td></tr>
        <tr><td class="id">train_and_track_experiments</td><td>Chạy ma trận thí nghiệm, gate, đăng ký challenger</td><td>Sau khi snapshot có nhãn qua validation</td></tr>
        <tr><td class="id">await_human_approval</td><td>Điểm dừng: người phụ trách xem MLflow rồi chuyển alias champion</td><td>Sau khi huấn luyện xong</td></tr>
      </tbody>
    </table></div>
  </div>
  <div class="sub">
    <h3>3.2 Chiến lược lập lịch</h3>
    <div class="tbl"><table>
      <thead><tr><th>DAG</th><th>Kiểu kích hoạt</th><th>Tần suất</th><th>Lý do</th></tr></thead>
      <tbody>
        <tr><td>dropout_weekly_scoring</td><td>Theo lịch (cron <code>0 2 * * 0</code>)</td><td>Hằng tuần, Chủ nhật 02:00; <code>catchup=False</code>, <code>max_active_runs=1</code></td><td>Cố vấn làm việc theo tuần; danh sách phải sẵn trước 06:00 thứ Hai (NFR1, có SLA 4 giờ)</td></tr>
        <tr><td>dropout_training</td><td>Theo lịch + theo sự kiện (TriggerDagRunOperator khi có drift) + thủ công</td><td>3 lần/năm (<code>0 3 1 1,5,9 *</code>), sau khi nhãn mỗi học kỳ về</td><td>Nhãn chỉ có sau mỗi học kỳ; drift thì không chờ đến kỳ sau</td></tr>
      </tbody>
    </table></div>
  </div>
  <div class="sub">
    <h3>3.3 Code DAG</h3>
    {SNIP4}
    {SNIP5}
    {SNIP6}
  </div>
  <div class="sub">
    <h3>3.4 Vận hành</h3>
    <div class="prose"><ul>
      <li><strong>Xử lý lỗi:</strong> mỗi task thử lại 2 lần, thời gian chờ tăng dần từ 10 phút; mỗi task tối đa 1 giờ, riêng task huấn luyện 3 giờ. Dữ liệu hỏng thì không fail cả DAG mà rẽ sang luật dự phòng. <code>max_active_runs=1</code> để hai lần chạy không ghi đè lên nhau. Mỗi task đọc và ghi file theo ngày nên chạy lại an toàn (idempotent).</li>
      <li><strong>Cảnh báo:</strong> <code>on_failure_callback</code> gửi email cho đội dữ liệu kèm link log. SLA 1 giờ cho trích xuất và 4 giờ cho danh sách, quá hạn thì Airflow báo. Kích hoạt luật dự phòng hay huấn luyện lại do drift cũng đều được ghi lại.</li>
      <li><strong>Giám sát:</strong> giao diện Airflow (thời gian chạy, tỷ lệ thành công); MLflow (metric theo từng phiên bản); PSI theo tuần và phân phối điểm rủi ro (Evidently/Grafana như ASM1); chỉ báo sớm như tỷ lệ Tier A được liên hệ trong 7 ngày. Khi có nhãn cuối học kỳ thì đo lại PR-AUC và Recall@k thật.</li>
    </ul></div>
  </div>
</section>

<section id="p4">
  <div class="weight">PHẦN 4 · 20%</div>
  <h2>4. Chất lượng code và tài liệu</h2>
  <div class="sub">
    <h3>4.1 Chuẩn code</h3>
    <div class="prose"><ul>
      <li><strong>Style:</strong> PEP 8, kiểm tra và format bằng <code>ruff</code> (dòng tối đa 100 ký tự; luật E, F, I, B, UP, D). Kết quả: <code>All checks passed!</code></li>
      <li><strong>Docstring:</strong> theo chuẩn Google, mọi module, lớp và hàm public đều có (luật D của ruff ép buộc).</li>
      <li><strong>Type hints:</strong> đầy đủ cho tham số và giá trị trả về; dùng <code>dataclass</code> cho cấu hình thí nghiệm và kết quả.</li>
      <li><strong>Comment:</strong> chỉ viết ở chỗ có lý do không hiển nhiên, ví dụ "export gốc có ký tự tab thừa" hay "CV để biết chênh lệch có vượt nhiễu không".</li>
      <li><strong>Test:</strong> 6 unit test (pytest) cho validation, cách chia dữ liệu, loại thuộc tính nhạy cảm, đặc trưng, metric top-k và PSI. Kết quả: <code>6 passed</code>.</li>
    </ul></div>
    {SNIP7}
  </div>
  <div class="sub">
    <h3>4.2 Cấu hình</h3>
    <div class="prose"><p>Tách hai loại cấu hình. <code>params.yaml</code> chứa mọi giá trị ảnh hưởng đến kết quả (seed, cách chia, nhóm đặc trưng, ngưỡng gate) và được commit vào Git. Biến môi trường chứa thông tin hạ tầng và bí mật (địa chỉ MLflow, đường dẫn dữ liệu), không commit; chỉ có file mẫu <code>.env.example</code>.</p></div>
    {SNIP8}
    {SNIP9}
  </div>
</section>

<section id="p5">
  <div class="weight">PHẦN 5 · 10%</div>
  <h2>5. Khả năng tái lập và quản lý phiên bản</h2>
  <div class="tbl"><table>
    <thead><tr><th>Đối tượng</th><th>Chiến lược</th></tr></thead>
    <tbody>
      <tr><td><b>Code</b></td><td>GitHub flow: <code>main</code> luôn chạy được; mỗi thay đổi làm trên nhánh <code>feat/…</code>, <code>fix/…</code>, mở PR, CI chạy ruff + pytest rồi mới merge. Phát hành bằng tag semantic version (<code>v0.2.0</code>); tag này được ghi vào image Docker và model version.</td></tr>
      <tr><td><b>Dữ liệu</b></td><td>Mỗi lần trích xuất là một snapshot bất biến có ngày. SHA-256 của file được ghi vào mọi MLflow run và model version (<code>data_sha256</code>). Khi lên production thì dùng DVC lưu lên MinIO: <code>dvc add</code> sinh file <code>.dvc</code> nhỏ để commit cùng code, nên checkout một commit là lấy lại đúng dữ liệu.</td></tr>
      <tr><td><b>Mô hình</b></td><td>MLflow Model Registry, tên mô hình <code>dropout-risk</code>. Version tăng tự động. Alias: <code>challenger</code> (vừa qua gate), <code>champion</code> (đang dùng), <code>previous-champion</code> (để rollback). Tag bắt buộc: <code>exp_id</code>, <code>git_commit</code>, <code>data_sha256</code>, <code>approved_by</code>. Tên run: <code>&lt;exp_id&gt;-&lt;algorithm&gt;</code>, ví dụ <code>E02-logreg</code>.</td></tr>
      <tr><td><b>Môi trường</b></td><td><code>requirements.txt</code> ghim đúng phiên bản; Dockerfile ghim base image <code>python:3.11.10-slim</code>.</td></tr>
    </tbody>
  </table></div>
  {SNIP10}
  {SNIP11}
  {SNIP12}
  <div class="callout"><p><strong>Bằng chứng tái lập:</strong> chạy lại toàn bộ 12 thí nghiệm từ đầu (xóa MLflow DB) cho ra bảng metric <strong>giống hệt</strong> lần trước; chỉ thời gian chạy khác. Unit test cũng kiểm tra cách chia dữ liệu giống nhau giữa hai lần gọi.</p></div>
</section>

<section id="p6">
  <h2>6. Hạn chế và bước tiếp theo</h2>
  <div class="prose"><ul>
    <li><strong>Dữ liệu đại diện:</strong> bộ UCI (Bồ Đào Nha) có tỷ lệ bỏ học 32 %, khác xa giả định 2,7 % mỗi học kỳ ở BTEC, và không có đặc trưng theo tuần. Kết quả chỉ chứng minh pipeline và phương pháp đánh giá, không dự báo hiệu năng thật ở BTEC.</li>
    <li><strong>Cách chia ngẫu nhiên:</strong> do thiếu ngày nhập học, nên kết quả có thể lạc quan hơn cách chia theo thời gian.</li>
    <li><strong>Phần chưa chạy thật:</strong> DAG đã được Airflow nạp và các hàm task đã chạy thử đầu cuối (chấm 4.424 sinh viên, ra 442 Tier A và 664 Tier B), nhưng chưa chạy trên một Airflow scheduler thật. Dockerfile chưa build trong môi trường này.</li>
    <li><strong>Bước tiếp theo:</strong> thay nguồn dữ liệu bằng export thật; chuyển sang chia theo học kỳ; thêm đặc trưng chuyên cần theo tuần; đánh giá lại E02 so với E07 theo đúng luật 0,02.</li>
  </ul></div>
</section>

<section id="tl">
  <h2>Tài liệu tham khảo</h2>
  <ol>
    <li>Realinho, V., Machado, J., Baptista, L., Martins, M. V. (2022). Predicting Student Dropout and Academic Success. <em>Data</em>, 7(11), 146. Dữ liệu: UCI Machine Learning Repository, dataset 697.</li>
    <li>Huyen, C. (2022). <em>Designing Machine Learning Systems</em>. O'Reilly.</li>
    <li>Kästner, C. (2025). <em>Machine Learning in Production</em>. MIT Press.</li>
    <li>Tài liệu MLflow (Tracking, Model Registry với aliases) và Apache Airflow 2.10.</li>
    <li>Bài giảng DDM501, Session 3–5.</li>
  </ol>
</section>
<footer>Bản tham khảo tiếng Việt cho Individual Assignment 2, DDM501. Code và số liệu sinh ra từ lần chạy thật trên bộ dữ liệu UCI.</footer>
</div>
"""

head = (
    "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Pipeline MLOps cảnh báo bỏ học</title>"
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Lora:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">'
    f"<style>{CSS}</style></head><body>"
)
OUT.write_text(head + BODY + "</body></html>", encoding="utf-8")
print("written", OUT, OUT.stat().st_size)
