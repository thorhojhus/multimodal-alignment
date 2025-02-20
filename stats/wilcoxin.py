from typing import Literal
import numpy as np
import argparse
from scipy.stats import wilcoxon

parser = argparse.ArgumentParser(description="Perform a Wilcoxin signed-rank test on two benchmarks.")
parser.add_argument("file1", type=str, help="Path to the first benchmark file.")
parser.add_argument("file2", type=str, help="Path to the second benchmark file.")
parser.add_argument("outfile", type=str, help="Path to the output file.")
args = parser.parse_args()

def load_benchmarks(path_to_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Load in a benchmark from the output of a benchmark file.
    
    Assume 'path_to_str' is relative to the working directory.
    Assume that the file has lines of form: 
        WER: [NUMBER]
        CER: [NUMBER]
    Times the number of 
    """

    with open(path_to_file, "r") as file:
        lines = file.readlines()
        wer = []
        cer = []
        for line in lines:
            if "WER: " in line:
                wer.append(float(line.split(" ")[1]))
            elif "CER: " in line:
                cer.append(float(line.split(" ")[1]))
    return np.asarray(wer), np.asanyarray(cer)

def append_test_results(output_file_path:str, type: Literal["CER", "WER"], p: float, statistic_value: float) -> None:
    """Append the wilcoxin test results to the output file."""
    with open(output_file_path, "a+") as file:
        file.write(f"[{type}] Test Statistic: {statistic_value} with P-value: {p}\n")

def wilcoxin_test(benchmark1: list[float], benchmark2: list[float]) -> float:
    """Perform a Wilcoxin signed-rank test on two benchmarks.
    
    Return the p-value of the test.
    """
    return wilcoxon(benchmark1, benchmark2)

def main(args: argparse.Namespace) -> None:
    # Perform Wilcoxin test on both the WER and CER benchmarks
    wer1, cer1 = load_benchmarks(args.file1)
    wer2, cer2 = load_benchmarks(args.file2)
    p_wer, statistic_wer = wilcoxin_test(wer1, wer2)
    p_cer, statistic_cer = wilcoxin_test(cer1, cer2)
    append_test_results(args.outfile, "WER", p_wer, statistic_wer)
    append_test_results(args.outfile, "CER", p_cer, statistic_cer)

if __name__ == "__main__":
    main(args)