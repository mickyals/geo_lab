"""
Precompute statistics for ERA5 data to avoid loading data twice during training.

This script computes and saves statistics for different time slice configurations.
Run this once before training to generate the statistics files.
"""

from pathlib import Path
from typing import Optional, List
import pickle
import xarray as xr
import argparse

from era5multi import ERA5MultiData


def compute_and_save_statistics(
    data_dir: Path,
    output_dir: Path,
    solution_vars: List[str],
    time_idx_range: Optional[List[int]],
    config_name: str,
    read_data_fn=None
):
    """
    Compute statistics for a specific configuration and save to disk.
    
    Args:
        data_dir: Directory containing ERA5 data
        output_dir: Directory to save statistics
        solution_vars: List of variable names to load
        time_idx_range: Time slice range [start, end] or None for all times
        config_name: Name for this configuration (e.g., "1_timeslice" or "all_timeslices")
        read_data_fn: Custom function to read data (defaults to xr.open_dataset)
    """
    print(f"\n{'='*60}")
    print(f"Computing statistics for: {config_name}")
    print(f"Time range: {time_idx_range}")
    print(f"Variables: {solution_vars}")
    print(f"{'='*60}\n")
    
    # Set default read function
    if read_data_fn is None:
        read_data_fn = xr.open_dataset
    
    # Initialize ERA5MultiData
    era5 = ERA5MultiData(
        data_dir=str(data_dir),
        read_data_fn=read_data_fn,
        variables=solution_vars
    )
    
    # Compute statistics (no virtual points needed for statistics)
    print("Loading data and computing statistics...")
    data, statistics = era5.run(
        time_idx_range=time_idx_range,
        pressure_idx_range=None,
        latitude_idx_range=None,
        longitude_idx_range=None,
        indexing='ij',
        num_samples=0,
        include_virtual=False,
        use_lhs=True
    )
    
    print(f"Statistics computed successfully!")
    print(f"Data keys: {list(data['data'].keys())}")
    print(f"Total samples: {data['count']}")
    print(f"Statistics keys: {list(statistics.keys())}")
    
    # Save statistics
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / f"statistics_{config_name}.pkl"
    
    with open(stats_file, 'wb') as f:
        pickle.dump(statistics, f)
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"File size: {stats_file.stat().st_size / 1024:.2f} KB")
    
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Precompute statistics for ERA5 troposphere data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing ERA5 data files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/statistics",
        help="Directory to save computed statistics (default: ./data/statistics)"
    )
    parser.add_argument(
        "--solution_vars",
        nargs="+",
        default=["u", "v", "w", "z"],
        help="List of solution variables (default: u v w z)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    solution_vars = args.solution_vars
    
    # Validate data directory
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Solution variables: {solution_vars}")
    
    # Compute statistics for both configurations
    configs = [
        {
            "time_idx_range": [0, 1],
            "config_name": "1_timeslice"
        },
        {
            "time_idx_range": None,
            "config_name": "all_timeslices"
        }
    ]
    
    all_statistics = {}
    
    for config in configs:
        try:
            stats = compute_and_save_statistics(
                data_dir=data_dir,
                output_dir=output_dir,
                solution_vars=solution_vars,
                time_idx_range=config["time_idx_range"],
                config_name=config["config_name"]
            )
            all_statistics[config["config_name"]] = stats
        except Exception as e:
            print(f"\nError computing statistics for {config['config_name']}: {str(e)}")
            raise
    
    print(f"\n{'='*60}")
    print("All statistics computed successfully!")
    print(f"{'='*60}")
    print(f"\nTo use these statistics in training, update your TroposphereDataModule")
    print(f"to load from: {output_dir}")


if __name__ == "__main__":
    main()