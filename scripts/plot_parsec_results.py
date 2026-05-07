# -*- coding: utf-8 -*-
"""
plot_parsec_results.py

功能：
1. 读取 PARSEC 项目的实验 CSV 文件
2. 绘制 BJTaxi runtime 消融对比图
3. 绘制四个数据集上的 RMSE trade-off 对比图

运行方式：
    python scripts/plot_parsec_results.py

输入文件：
    results/ablation_bjtaxi.csv
    results/rmse_tradeoff.csv

输出文件：
    results/ablation_bjtaxi_runtime.png
    results/rmse_tradeoff.png
"""

from pathlib import Path
import csv
import matplotlib.pyplot as plt


# 项目根目录：脚本位于 scripts/ 下，所以父目录就是项目根目录
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"


def read_csv_as_dicts(csv_path: Path):
    """
    读取 CSV 文件，返回字典列表。
    每一行会被解析为一个 dict。
    """
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_bjtaxi_runtime():
    """
    绘制 BJTaxi 数据集上 Only divide 和 PARSEC 的累计 GPU runtime 对比。
    """
    csv_path = RESULTS_DIR / "ablation_bjtaxi.csv"
    rows = read_csv_as_dicts(csv_path)

    methods = []
    runtimes = []
    rmses = []

    for row in rows:
        methods.append(row["method"])
        runtimes.append(float(row["cumulative_gpu_runtime_s"]))
        rmses.append(float(row["final_rmse"]))

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, runtimes)

    # 在柱子上标注具体数值
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{runtime:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("Cumulative GPU Runtime (s)")
    plt.title("BJTaxi Runtime Ablation at Sampling Rate 0.5")
    plt.tight_layout()

    output_path = RESULTS_DIR / "ablation_bjtaxi_runtime.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[OK] Saved runtime figure to: {output_path}")


def plot_rmse_tradeoff():
    """
    绘制四个数据集上的 RMSE 对比图。
    每个数据集有两个方法：Only divide 和 PARSEC。
    """
    csv_path = RESULTS_DIR / "rmse_tradeoff.csv"
    rows = read_csv_as_dicts(csv_path)

    # 整理数据：dataset -> method -> rmse
    data = {}
    for row in rows:
        dataset = row["dataset"]
        method = row["method"]
        rmse = float(row["final_rmse"])

        if dataset not in data:
            data[dataset] = {}
        data[dataset][method] = rmse

    datasets = list(data.keys())
    only_divide_values = [data[d].get("Only divide", 0.0) for d in datasets]
    parsec_values = [data[d].get("PARSEC", 0.0) for d in datasets]

    x = list(range(len(datasets)))
    width = 0.35

    plt.figure(figsize=(8, 4.5))

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        only_divide_values,
        width,
        label="Only divide",
    )
    bars2 = plt.bar(
        [i + width / 2 for i in x],
        parsec_values,
        width,
        label="PARSEC",
    )

    # 标注数值，避免图像只有柱子没有灵魂
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, datasets)
    plt.ylabel("Final RMSE")
    plt.title("RMSE Trade-off at Sampling Rate 0.5")
    plt.legend()
    plt.tight_layout()

    output_path = RESULTS_DIR / "rmse_tradeoff.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[OK] Saved RMSE figure to: {output_path}")


def main():
    """
    主函数：依次生成两张图。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_bjtaxi_runtime()
    plot_rmse_tradeoff()

    print("[DONE] All figures generated successfully.")


if __name__ == "__main__":
    main()