# DDM501 – Individual Assignments

**Student:** Nguyen Thi Hong Hanh (25MS13316)

## Assignment 1 – ML System Design Document

**Topic:** Early Warning System for Students at Risk of Dropping Out

File: [`DDM501_Assignment1_25MS13316_NguyenThiHongHanh.pdf`](DDM501_Assignment1_25MS13316_NguyenThiHongHanh.pdf)

Contents: problem definition, requirements analysis, goals and success metrics,
high-level architecture, trade-off analysis. All institutional figures are assumed for illustration.

## Assignment 2 – ML Pipeline Design & MLOps Analysis

Report: [`DDM501_Assignment2_25MS13316_NguyenThiHongHanh.pdf`](DDM501_Assignment2_25MS13316_NguyenThiHongHanh.pdf)

Code: [`asm2/`](asm2/) – working prototype of the pipeline designed in Assignment 1:
data validation (pandera), shared feature engineering, 12 tracked experiments (MLflow),
model registry with champion/challenger aliases, Airflow DAGs, tests and Dockerfile.
See [`asm2/README.md`](asm2/README.md) for how to run it. The dataset (UCI
"Predict Students' Dropout and Academic Success") is not committed; download it to
`asm2/data/raw/students.csv`.
