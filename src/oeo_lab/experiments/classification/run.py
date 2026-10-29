"""Retired entry point; retained only to explain migration."""


def main():
    raise SystemExit(
        "This legacy entry is retired. Use python run_experiment.py --no-watch "
        "<experiment> or oeo-classification --no-watch <experiment>. "
        "Run either launcher with --help for available experiments."
    )


if __name__ == "__main__":
    main()
