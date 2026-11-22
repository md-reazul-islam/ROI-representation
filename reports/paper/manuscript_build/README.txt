Easiest: run RUN_MODULE1_PAPER.ipynb (Module1 folder) top to bottom - full run, evaluation, figures, manuscript.

By hand (from the Module1 folder):
  1. python scripts/run_paper_pilot.py --name <RUN>            (needs runs/<RUN>/manifest.csv; pilot24_paper builds its own)
  2. python scripts/run_paper_evaluation.py --run runs/<RUN> --output-dir <REPORT> --gpu-log runs/<RUN>/gpu_monitor.csv
  3. python scripts/make_paper_figures.py --run runs/<RUN> --report-dir <REPORT>
  4. python reports/paper/manuscript_build/build_manuscript.py <REPORT> [<OUTPUT.docx>]   (needs python-docx)
Every number in Results/Discussion is read from the CSVs by results.py; statements that depend on significance or
on which configuration is best are worded from the data. Re-read the interpretive text after a new run.
