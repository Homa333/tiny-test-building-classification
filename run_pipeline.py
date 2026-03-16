import argparse
from src.pipeline import BuildingPipeline
from src.logger import setup_logger


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    logger = setup_logger(args.output)

    pipeline = BuildingPipeline(args.data, args.output, logger)
    pipeline.run()


if __name__ == "__main__":
    main()
