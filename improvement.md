# Project Structure Improvement Plan

This document outlines a plan to improve the project's directory structure for better clarity, maintainability, and scalability.

## 1. Data Directory Restructuring

The current `data` directory has a nested and somewhat confusing structure. We will flatten and simplify it.

**Current Structure:**

```
data/
├── 01_raw/
├── 02_processed/
├── 03_intermediate/
├── 03_processed/
├── 04_models/
├── 05_output/
└── validation/
```

**Proposed Structure:**

```
data/
├── raw/
├── processed/
├── models/
├── outputs/
└── validation_reports/
```

**Rationale:**

*   Removes the numbered prefixes which can become difficult to manage.
*   Merges `02_processed` and `03_processed` into a single `processed` directory.
*   Renames `validation` to `validation_reports` for clarity.
*   Consolidates intermediate data steps into the `processed` directory, which can be handled by subdirectories if needed.

**Migration Steps:**

1.  **Create new directories:**
    *   `data/raw`
    *   `data/processed`
    *   `data/models`
    -   `data/outputs`
    *   `data/validation_reports`
2.  **Move contents:**
    *   `data/01_raw/*` -> `data/raw/`
    *   `data/02_processed/*` -> `data/processed/`
    *   `data/03_intermediate/vector_db` -> `data/processed/vector_store`
    *   `data/03_processed/chromadb` -> `data/processed/vector_store/chroma.sqlite3`
    *   `data/04_models/*` -> `data/models/`
    *   `data/05_output/*` -> `data/outputs/`
    *   `data/validation/*` -> `data/validation_reports/`
3.  **Remove old directories:**
    *   `data/01_raw`
    *   `data/02_processed`
    *   `data/03_intermediate`
    *   `data/03_processed`
    *   `data/04_models`
    *   `data/05_output`
    *   `data/validation`

## 2. Source Code Organization

The `src/pynucleus` directory can be slightly reorganized for better module grouping.

**Current Structure Issues:**

*   Some modules are large and could be broken down.
*   Utility functions are scattered.

**Proposed Changes:**

*   No major changes proposed at this time, but we should be mindful of growing modules and consider refactoring them into smaller, more focused modules in the future.

## 3. Configuration Management

The configuration files are well-placed in the `configs` directory. No changes are recommended at this time.

## 4. Scripts

The `scripts` directory is well-organized. No changes are recommended at this time.

## Summary of Benefits

*   **Improved Clarity:** A simpler, flatter directory structure is easier to understand.
*   **Easier Maintenance:** Reduces ambiguity and makes it easier to locate and manage files.
*   **Scalability:** The proposed structure can more easily accommodate new features and data pipelines.

I have created a shell script `organize_project.sh` to perform these changes automatically. Please review the script before running it.
