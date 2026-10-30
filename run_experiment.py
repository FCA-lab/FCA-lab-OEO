"""Experiment launcher guide (checked against the source on 2026-10-29).

1. Call chain
    run_experiment.py -> oeo_lab.experiments.cli.main(argv)
    --no-watch: cli.direct_run() -> subprocess.run(python -m selected_module)
    Default supervision: cli.start_watch() -> external experiment-watch service
        -> this launcher with --no-watch.
    The selected module's main() builds tasks and invokes the shared scheduler.
    This launcher does not train models itself.

    Selection       Module under src/oeo_lab/experiments/classification/
    completeness    completeness.py
    uci             uci_experiment.py
    babd            babd_experiment.py (balanced, then imbalanced by default)
    elliptic        elliptic_experiment.py
    training-size   training_size_experiment.py (exploratory; not required by the paper)

2. Environment and commands
    Enter the repository and activate the environment prepared using README.md.
    cd /path/to/FCA-lab-OEO
    conda activate fca-oeo
    # On the original D30 workstation, the existing environment is named fca.
    python run_experiment.py --help
    python run_experiment.py                         # Interactive menu; supervised
    python run_experiment.py --no-watch               # Interactive menu; direct execution
    python run_experiment.py uci --n-jobs 10          # Requires experiment-watch
    python run_experiment.py --no-watch uci --n-jobs 10
    python run_experiment.py --no-watch babd --experiments balanced --n-jobs 10
    python run_experiment.py --no-watch babd --experiments imbalanced --n-jobs 10
    python run_experiment.py --no-watch elliptic --n-jobs 10
    # Paper completeness setting: explicitly use max_bins=5, giving four tasks.
    python run_experiment.py --no-watch completeness --max-bins 5 --n-jobs 1
    # Groups run sequentially, not simultaneously; execution stops if a group fails.
    python run_experiment.py --no-watch --batch uci babd elliptic --n-jobs 10
    # all includes training-size and the completeness script's default binning grid.
    # It therefore differs from the paper configuration.
    python run_experiment.py --no-watch all --n-jobs 10

    --n-jobs limits concurrent task processes; it does not assign ten threads
    to a single model. Numerical thread pools within classification tasks are
    limited to one thread. Memory protection may reduce actual concurrency.
    BABD also uses separate scheduling queues for RC-OEO and the other models.
    Ten workers is an example, not a guarantee of sufficient resources.
    Concurrency affects scheduling and timing, not the number of experimental tasks.

3. Preview the parameter combinations before execution
    python run_experiment.py uci --dry-run --n-jobs 10
    python run_experiment.py babd --experiments imbalanced --dry-run --n-jobs 10
    python run_experiment.py elliptic --dry-run --n-jobs 10
    python run_experiment.py completeness --max-bins 5 --dry-run --n-jobs 1
    --help and --dry-run bypass supervision. A dry run reads data, checks
    dependencies and builds tasks, but does not train models. Its reported task
    count takes precedence over the static counts below.
    Batch execution forwards additional arguments to every selected group;
    use only arguments supported by, and appropriate for, all selected groups.

4. Default parameter combinations and training task counts
    Seeds are 11, 37 and 53, except for the completeness experiment.

    UCI: comparison, sensitivity and ablation requests are merged before training.
    - Iris/Wine/WDBC: nine maximum bin counts, 20,30,...,100; five-fold CV.
      Per seed and fold: 4 binary-input models x 9 bin settings + 4 raw-feature
      or majority-class configurations = 40 training tasks.
      Numerical datasets: 3 datasets x 3 seeds x 5 folds x 40 = 1,800 tasks.
    - Balloons/Lenses: categorical inputs are not binned; eight model
      configurations, with five/four folds respectively.
      Categorical datasets: 3 seeds x (5+4) folds x 8 models = 216 tasks.
      Total: 2,016 training tasks.
    - Binary-input models: DT, RF-context, XGBoost-context and RC-OEO.
      Other configurations: majority-class baseline, RF-raw, XGBoost-raw, SVM-raw.
    - Performance comparison: full rule set and uniform voting.
      Sensitivity: full rule set with uniform/nonzero/support voting
      (uniform voting / nonzero-support filtering / support-weighted voting).
      Rule ablation: max_bins=20, uniform voting and seven rule configurations.
      Each model/bin/seed/fold combination is fitted once. Voting and ablation
      variants share that fit; they do not multiply the training task count.

    BABD: five maximum bin counts [5,10,20,30,50], eight models, three seeds,
    and uniform voting.
    - Models: RC-OEO, logistic regression, RF, XGBoost, CatBoost, LightGBM, SVM, MLP.
    - Balanced: seven classes; training objects per class
      [50,100,200,500,1000,2000,4000]; 1,000 test objects held out per class.
      7 training sizes x 5 bin settings x 8 models x 3 seeds = 840 tasks.
    - Imbalanced: eleven classes; total training sizes
      [500,1000,2000,5000,10000,20000,30000]; five-fold CV.
      7 training sizes x 5 bin settings x 8 models x 3 seeds x 5 folds = 4,200 tasks.
      Both studies together: 5,040 tasks. Do not multiply again by the class count.

    Elliptic: 5 bin settings [5,10,20,30,50] x 4 models x 3 seeds = 60 tasks.
    - Models: RC-OEO, RF, logistic regression and XGBoost; 93 local features.
      Time steps 1--34 are used for training and 35--49 for testing; no CV.
    - RC-OEO uses full, supporting-only and excluding-only rule configurations
      with uniform voting by default. These share fits, yielding 90 prediction
      configuration records across all models, not 90 fits.

    Completeness: one task for each of three categorical datasets
    (Balloons/Lenses/COVID19), plus the number of Iris bin settings.
    - Each task evaluates both direct and full_lattice extraction routes.
      The script defaults to Iris [5,8,10], giving 3+3=6 tasks.
      The paper's explicit --max-bins 5 setting gives four tasks.
    - Do not multiply by seeds, CV folds or the two extraction routes.

    Independent training-size study: 3 numerical datasets x 5 training fractions
    [.2,.4,.6,.8,1.0] x 3 bin settings [5,8,10] x 3 seeds x 1 model (RC-OEO)
    = 135 tasks; the holdout test fraction is .3.

    Paper configuration: 4 + 2,016 + 840 + 4,200 + 60 = 7,120 tasks.
    Current all defaults: 6 + 2,016 + 840 + 4,200 + 60 + 135 = 7,257 tasks.
    These are requested task counts, not counts of newly executed tasks.
    Successful checkpoints may be reused, reducing the required computation.
    Counts apply only to the grids above; recalculate after changing --models,
    --datasets, --seeds, --max-bins or other experimental settings.

5. Parameter locations, outputs and reproduction notes
    Precedence: configuration files -> experiment main()/script defaults
    -> explicit command-line arguments. In particular, UCI main() uses 20--100,
    whereas uci_config.py retains an earlier grid; do not rely on that file alone.
    Use commands such as python run_experiment.py uci --help for group-specific options.
    Workbooks and checkpoints are normally stored under results/classification/.
    Do not run multiple launcher instances writing to the same output location.
    For a subset or a new reproduction run, specify a separate --results-root,
    such as results/reproduction/uci, to avoid replacing a full report with a subset.
    Changes to source files or execution settings may change checkpoint identities.
    This guide does not execute any of the listed commands. After installation,
    oeo-classification uses the same unified entry; reinstall after updating it.
    Plotting only: python analysis/classification/plot_all.py
    (reads existing workbooks; does not train models).
    See docs/classification/reproduction.md and docs/classification/paper_artifacts.md.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from oeo_lab.experiments.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
