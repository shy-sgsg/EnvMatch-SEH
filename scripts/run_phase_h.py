#!/usr/bin/env python3
"""Initialize or resume the configured Phase H exploratory Oracle run."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml

from experiments.oracle_exploration.phase_h import prepare_phase_h_run, run_phase_h_interaction_review
from experiments.oracle_exploration.phase_h_data import run_h2_data_prep
from experiments.oracle_exploration.phase_h_search import run_phase_h_search
from experiments.oracle_exploration.phase_h_post import run_phase_h_nonparametric_postprocessing
from experiments.oracle_exploration.phase_h_groups import run_phase_h_environment_group_profiles
from experiments.oracle_exploration.phase_h_validation import run_phase_h_validation
from experiments.oracle_exploration.phase_h_figures import generate_phase_h_figures
from experiments.oracle_exploration.phase_h_report import write_phase_h_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_h.yaml")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run_dir, inputs = prepare_phase_h_run(ROOT, config)
    interaction = run_phase_h_interaction_review(ROOT, run_dir, inputs.pairs, config)
    matched = run_h2_data_prep(run_dir, inputs.cells, inputs.pairs, config)
    aligned_cells = inputs.cells
    aligned_cell_count = len(inputs.cells)
    matched_pair_count = len(inputs.pairs)
    outcome_counts = dict(inputs.outcome_counts)
    h1_tests = len(interaction["revised_tests"])
    h1_reliable = int(interaction["revised_tests"]["status"].eq("reliable").sum())
    h2_memberships = len(matched.pair_memberships)
    h2_pixels = len(matched.pixel_unique)
    h2_pairs = len(matched.pairs)
    del interaction, matched, inputs
    gc.collect()
    search = run_phase_h_search(aligned_cells, config, run_dir)
    successful_trials = int(search["trials"]["status"].eq("ok").sum()) if "status" in search["trials"] else 0
    trial_count = len(search["trials"])
    del search
    gc.collect()
    nonparametric = run_phase_h_nonparametric_postprocessing(run_dir, aligned_cells, config)
    group_profiles = run_phase_h_environment_group_profiles(run_dir, aligned_cells, config)
    validation = run_phase_h_validation(run_dir, aligned_cells, config)
    figures = generate_phase_h_figures(run_dir, aligned_cells, config)
    report_path, review_request_path = write_phase_h_report(ROOT, run_dir, config)
    print(f"Phase H run: {run_dir.relative_to(ROOT)}")
    print(f"Aligned cells: {aligned_cell_count:,}")
    print(f"Matched pairs: {matched_pair_count:,}")
    print(f"Outcome non-missing counts: {outcome_counts}")
    print(f"H1 interaction diagnostics: {h1_tests:,} tests; {h1_reliable:,} reliable")
    print(f"H2 matched populations: {h2_memberships:,} memberships; {h2_pixels:,} unique pixels; {h2_pairs:,} pairs")
    print(f"H3 matched-only search: {trial_count:,} model and subset trials; {successful_trials:,} successful fits")
    print(f"H4 nonparametric tables: {len(nonparametric['response_summary']):,} response rows; {len(nonparametric['paired_difference_summary']):,} paired summaries")
    print(f"H5 group profiles: {len(group_profiles['profiles']):,} rows across {group_profiles['stability']['group_method'].nunique():,} grouping methods")
    print(f"H6/H7 validation: {len(validation['spatial_validation_by_split']):,} spatial scores; {len(validation['period_comparison_by_split']):,} common-period scores")
    print(f"Figures: {figures['manifest']['figure_count']} figure files in PNG/SVG/PDF")
    print(f"Report: {report_path.relative_to(ROOT)}")
    print(f"ChatGPT review request: {review_request_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
