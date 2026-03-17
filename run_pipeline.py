import argparse
import os
import sys

from src.pipeline import BuildingPipeline
from src.logger import setup_logger


def main():

    parser = argparse.ArgumentParser(
        description="Run building classification pipeline"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset directory"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory to store results"
    )

    args = parser.parse_args()

    # Validate dataset path
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset path does not exist: {args.data}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(args.output)

    logger.info("Starting Building Classification Pipeline")

    try:

        pipeline = BuildingPipeline(args.data, args.output, logger)

        pipeline.run()

        logger.info("Pipeline completed successfully")

    except Exception as e:

        logger.exception(f"Pipeline failed with error: {e}")

        sys.exit(1)


if __name__ == "__main__":
    main()
