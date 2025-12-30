"""Utility script to reverse Min-Max normalization for synthesized NILM signals.

This script reads a real appliance power series to fit a :class:`MinMaxScaler`, then
applies that scaler to inverse-transform a synthetic sequence (``.npy`` or ``.csv``).
It also saves the restored series to ``.csv`` and plots a quick comparison with the
real sequence.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_power_series(path: Path) -> np.ndarray:
    """Load a power series from ``path`` and return it as a 2D array ``(n, 1)``.

    Accepts ``.npy`` arrays or ``.csv`` files containing a ``power`` column.
    """

    if not path.exists():
        hint = (
            "The file path passed to --real/--synthetic does not exist. "
            "If your file has a different name (e.g., `microwave_test.csv`), "
            "either rename it or pass the exact path via the flag."
        )
        raise FileNotFoundError(f"Input file not found: {path}. {hint}")

    if path.suffix == ".npy":
        data = np.load(path)
        return np.asarray(data).reshape(-1, 1)

    df = pd.read_csv(path)
    if "power" not in df.columns:
        raise ValueError(f"CSV file {path} must contain a 'power' column")

    return df["power"].to_numpy().reshape(-1, 1)


def inverse_transform(
    appliance_name: str,
    real_series_path: Path,
    synthetic_path: Path,
    output_csv: Path,
    plot_points: int = 20000,
) -> None:
    """Fit a scaler on the real series and denormalize the synthetic data."""

    real_series = read_power_series(real_series_path)
    scaler = MinMaxScaler()
    scaler.fit(real_series)

    print(f"[info] Loaded real series from {real_series_path} with {len(real_series)} samples.")

    synthetic_series = read_power_series(synthetic_path)
    restored = scaler.inverse_transform(synthetic_series)

    # Save restored data
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(restored.flatten(), columns=["power"]).to_csv(output_csv, index=False)
    print(f"[info] Restored synthetic data from {synthetic_path} -> {output_csv}")

    # Plot a quick comparison for visual inspection
    fig, axs = plt.subplots(2, figsize=(8, 8))
    axs[0].plot(restored.flatten()[:plot_points], linestyle="-", color="b", label="Generated (restored)")
    axs[0].set_title(f"generate {appliance_name}")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("power")
    axs[0].legend()

    axs[1].plot(real_series.flatten()[:plot_points], linestyle="-", color="r", label="Origin")
    axs[1].set_title(f"origin {appliance_name}")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("power")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse-transform generated appliance load data.")
    parser.add_argument(
        "--appliance",
        default="microwave",
        help="Appliance name for titles/labels (default: microwave)",
    )
    parser.add_argument(
        "--real",
        type=Path,
        default=Path("home2_microwave.csv"),
        help="Path to the real appliance CSV used to fit Min-Max scaler (expects a power column)",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("ddpm_fake_microwave.npy"),
        help="Path to the generated data to denormalize (.npy or CSV with a power column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generatedData/microwave_denormalized.csv"),
        help="Where to save the restored power series CSV",
    )
    parser.add_argument(
        "--plot-points",
        type=int,
        default=20000,
        help="Number of samples to show in the comparison plot",
    )

    args = parser.parse_args()
    print(
        "[提示] 如果默认文件名不存在（如 home2_microwave.csv / ddpm_fake_microwave.npy），"
        "请用 --real 和 --synthetic 指定真实的文件路径。"
    )
    inverse_transform(
        appliance_name=args.appliance,
        real_series_path=args.real,
        synthetic_path=args.synthetic,
        output_csv=args.output,
        plot_points=args.plot_points,
    )
    print(
        "[下一步] 生成的 CSV 已保存到上面的路径：可直接用于后续 NILM 预处理、"
        "训练或画图对比，不需要再运行 py_compile。"
    )


if __name__ == "__main__":
    main()
