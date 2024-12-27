#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Execute the data cleaning steps:
    - Download the raw dataset from W&B
    - Filter price outliers based on the given min and max price thresholds
    - Convert 'last_review' column to datetime
    - Save the cleaned dataset and log it back to W&B as a new artifact
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download and load the raw dataset
    logger.info(f"Downloading input artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Filter price outliers
    logger.info(
        f"Filtering out rows with price outside the range [{
            args.min_price}, {args.max_price}]"
    )
    min_price = args.min_price
    max_price = args.max_price
    idx = df["price"].between(min_price, max_price)
    df_clean = df[idx].copy()

    # Convert 'last_review' column to datetime
    logger.info('Converting "last_review" column to datetime')
    df_clean["last_review"] = pd.to_datetime(
        df_clean["last_review"], errors="coerce")

    # Drop rows in the dataset that are not in the proper geolocation
    idx = df_clean['longitude'].between(-74.25, -
                                        73.50) & df_clean['latitude'].between(40.5, 41.2)
    df_clean = df_clean[idx].copy()

    # Save the cleaned DataFrame locally
    output_file = "cleaned_data.csv"
    logger.info(f"Saving cleaned DataFrame to {output_file}")
    df_clean.to_csv(output_file, index=False)

    # Log the cleaned dataset as a new artifact
    logger.info(f"Logging artifact {args.output_artifact} to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the W&B artifact containing the raw dataset",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the artifact to store the cleaned dataset",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (e.g. 'cleaned_data')",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact content",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price threshold to filter out listings below this value",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price threshold to filter out listings below this value",
        required=True
    )

    args = parser.parse_args()

    go(args)
