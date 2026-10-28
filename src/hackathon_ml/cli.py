from __future__ import annotations

import argparse

from hackathon_ml.config import ProjectConfig
from hackathon_ml.logging_utils import configure_logging
from hackathon_ml.pipeline import export, test, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon ML training and submission workflow")
    parser.add_argument("command", choices=["train", "test", "export", "prog_run"])
    parser.add_argument("--threshold", type=float, default=0.66, help="Minimum macro-F1 required by prog_run (default: 0.66)")
    parser.add_argument("--output", default=None, help="Submission output path for export/prog_run")
    parser.add_argument("--allow-below-threshold", action="store_true", help="Export even if prog_run misses threshold")
    args = parser.parse_args()
    config = ProjectConfig()
    logger = configure_logging(config.log_path)
    logger.info("COMMANDE: %s", args.command)

    if args.command == "train":
        train(config)
    elif args.command == "test":
        test(config)
    elif args.command == "export":
        export(config, args.output)
    else:
        score = test(config)
        if score < args.threshold and not args.allow_below_threshold:
            logger.warning("PROG_RUN: seuil non atteint | %.4f < %.4f", score, args.threshold)
            logger.warning("PROG_RUN: sauvegarde et export maintenus malgré le seuil non atteint")
        else:
            logger.info("PROG_RUN: seuil atteint | %.4f >= %.4f", score, args.threshold)
        train(config)
        export(config, args.output)
        logger.info("PROG_RUN: processus terminé")


if __name__ == "__main__":
    main()