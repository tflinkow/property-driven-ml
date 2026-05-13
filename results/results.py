from __future__ import annotations

import codecs
import textwrap
from collections import defaultdict, namedtuple
from datetime import timedelta
from pathlib import Path

import re

import pandas as pd


Result = namedtuple("Result", "p_acc c_acc c_sec_self c_sec_common")
VerificationResult = namedtuple("VerificationResult", "verified unknown")


def get_name_from_file(report: str) -> str:
    stem = Path(report).stem

    names = {
        "Goedel": "Gödel",
        "Lukasiewicz": r"{\L}ukasiewicz",
        "KleeneDienes": "Kleene-Dienes",
        "ReichenbachSigmoidal": "sig. Reichenbach",
    }

    name = names.get(stem, stem)

    if stem == "Goedel" and "robustness" in report:
        name = "Fuzzy Logic"

    return name


def format_mean_std(values: list[float]) -> str:
    s = pd.Series(values, dtype=float)

    mean = 100 * s.mean()
    std = 100 * s.std(ddof=1) if len(s) > 1 else 0.0

    return rf"{mean:.1f}+-{std:.1f}"


def format_optional_mean_std(values: list[float]) -> str:
    if not values:
        return r"\text{n/a}"
    return format_mean_std(values)


def parse_log_file(log_file: Path) -> VerificationResult:
    text = log_file.read_text()

    def extract(name: str) -> tuple[int, int]:
        match = re.search(rf"{re.escape(name)}:\s*(\d+)/(\d+)", text)

        if not match:
            raise ValueError(f"Could not find '{name}' in {log_file}")
        return int(match.group(1)), int(match.group(2))

    verified, total = extract("verified")
    timed_out, _ = extract("timed-out")
    errored, _ = extract("errored")

    return VerificationResult(
        verified=verified / total,
        unknown=(timed_out + errored) / total,
    )


def parse_logic_and_epsilon_from_log(log_file: Path) -> tuple[str, str]:
    stem = log_file.stem
    stem = re.sub(
        r"_chunk\d+$", "", stem
    )  # TODO: no longer generation chunks, change dice_to_idx.py and here

    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected log filename format: {log_file}")

    # last two parts are epsilon, some logic encode their softness param e.g. QLL_5
    eps = "_".join(parts[-2:])

    # everything before is logic
    logic_stem = "_".join(parts[:-2])

    logic = get_name_from_file(logic_stem)

    return logic, eps


# filenames like SomeLogic_0_5.log mean eps=0.5, SomeLogic_4_255.log means eps=4/255
# maybe it's okay to hard-code those two cases now
def eps_value(eps: str) -> float:
    if "_" in eps:
        num, denom = eps.split("_")

        if denom == "255":
            return float(num) / 255.0
        else:
            return float(f"{num}.{denom}")

    return float(eps)


def format_eps(eps: str) -> str:
    if eps.endswith("_255"):
        num = eps.removesuffix("_255")
        return rf"$\epsilon=\frac{{{num}}}{{255}}$"

    return rf"$\epsilon={eps.replace('_', '.')}$"


def read_final_result(csv_file: Path) -> Result:
    df = pd.read_csv(csv_file, comment="#")
    last = df.iloc[-1]

    return Result(
        p_acc=last["Test-P-Metric"],
        c_acc=last["Test-C-Acc"],
        c_sec_self=last["Test-C-Sec-self"],
        c_sec_common=last["Test-C-Sec-common"],
    )


def compute_time(csv_file: Path) -> float:
    df = pd.read_csv(csv_file, comment="#")

    total_test_time = df["Test-Time"].sum()
    total_train_time = df["Train-Time"].iloc[1:].sum()

    avg_test_time = df["Test-Time"].mean()
    avg_train_time = df["Train-Time"].iloc[1:].mean()

    print(
        f"file: {csv_file} "
        f"avg. train time [s]: {avg_train_time:.2f} "
        f"avg. test time [s]: {avg_test_time:.2f} "
        f"total train time [s]: {total_train_time:.2f} "
        f"total test time [s]: {total_test_time:.2f}"
    )

    return total_train_time + total_test_time


def mean(values: list[float]) -> float:
    return float(pd.Series(values, dtype=float).mean())


def bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def maybe_bold(s: str, do_bold: bool) -> str:
    return bold(s) if do_bold else s


# the best logic is the best product of PAcc and CSat (for the highest epsilon which is the one we train with)
def score_logic(
    values: list[Result],
    verifications_by_eps: dict[str, list[VerificationResult]],
    biggest_eps: str,
) -> float:
    verifications = verifications_by_eps.get(biggest_eps, [])

    if not verifications:
        return float("-inf")

    p_acc = mean([v.p_acc for v in values])
    verified = mean([v.verified for v in verifications])

    return p_acc * verified


def write_table_file(report_dir: Path, target_file: str) -> float:
    """
    Expects structure:

        constraint/dataset/seed/*.csv

    where each CSV corresponds to one logic.
    """

    csv_files = sorted(report_dir.glob("*/*.csv"))

    csv_files = [f for f in csv_files if not f.name.endswith("RegressionPlot.csv")]

    if not csv_files:
        return 0.0

    results_by_logic: dict[str, list[Result]] = defaultdict(list)
    verification_by_logic_eps: dict[str, dict[str, list[VerificationResult]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    all_eps: set[str] = set()

    total_seconds = 0.0

    for csv_file in csv_files:
        logic = get_name_from_file(str(csv_file))

        result = read_final_result(csv_file)
        results_by_logic[logic].append(result)

        total_seconds += compute_time(csv_file)

    log_files = sorted(report_dir.glob("*/*.log"))

    for log_file in log_files:
        logic, eps = parse_logic_and_epsilon_from_log(log_file)

        verification = parse_log_file(log_file)

        verification_by_logic_eps[logic][eps].append(verification)
        all_eps.add(eps)

    with codecs.open(target_file, "w", "utf-8") as file:
        eps_order = sorted(all_eps, key=eps_value)
        biggest_eps = eps_order[-1] if eps_order else None

        scores = (
            {
                logic: score_logic(
                    results_by_logic[logic],
                    verification_by_logic_eps.get(logic, {}),
                    biggest_eps,
                )
                for logic in results_by_logic
            }
            if biggest_eps is not None
            else {}
        )

        best_logic = max(scores, key=scores.get) if scores else None

        verification_cols = "".join("Q[c, mode=text]Q[c, mode=text]" for _ in eps_order)

        file.write(
            textwrap.dedent(rf"""
            \documentclass{{standalone}}
            \usepackage[utf8]{{inputenc}}
            \usepackage[T1]{{fontenc}}

            \usepackage{{siunitx}}
            \sisetup{{detect-all}}

            \usepackage{{amsmath}}
            \usepackage{{amssymb}}
            \usepackage{{nicefrac}}

            \usepackage{{tabularray}}
            \UseTblrLibrary{{booktabs}}

            \begin{{document}}
            \footnotesize
            \begin{{tblr}}
            {{
                colspec={{Q[l, mode=text]Q[c, mode=text]Q[c, mode=text]Q[c, mode=text]Q[c, mode=text]Q[c, mode=text]{verification_cols}}},
                row{{1}}={{font=\bfseries, mode=text}},
            }}
                \toprule
                Logic & Seeds & PAcc & CAcc & CSec (self) & CSec (common)
            """).strip()
            + "\n"
        )

        for eps in eps_order:
            file.write(rf" & Verified {format_eps(eps)} & Unknown {format_eps(eps)}")

        file.write(r" \\" + "\n")
        file.write(r"\midrule" + "\n")

        for logic in sorted(results_by_logic):
            values = results_by_logic[logic]
            is_best = logic == best_logic

            row = (
                rf"{maybe_bold(logic, is_best)} & "
                rf"{maybe_bold(str(len(values)), is_best)} & "
                rf"{maybe_bold(format_mean_std([v.p_acc for v in values]), is_best)} & "
                rf"{maybe_bold(format_mean_std([v.c_acc for v in values]), is_best)} & "
                rf"{maybe_bold(format_mean_std([v.c_sec_self for v in values]), is_best)} & "
                rf"{maybe_bold(format_mean_std([v.c_sec_common for v in values]), is_best)}"
            )

            for eps in eps_order:
                verifications = verification_by_logic_eps.get(logic, {}).get(eps, [])

                row += (
                    rf" & {maybe_bold(format_optional_mean_std([v.verified for v in verifications]), is_best)}"
                    rf" & {maybe_bold(format_optional_mean_std([v.unknown for v in verifications]), is_best)}"
                )

            file.write(row + r" \\" + "\n")

        file.write(
            textwrap.dedent(r"""
                \bottomrule
              \end{tblr}
            \end{document}
            """).strip()
            + "\n"
        )

    return total_seconds


def main():
    total_seconds = 0.0

    for constraint_dir in sorted(Path(".").iterdir()):
        if not constraint_dir.is_dir():
            continue

        if constraint_dir.name in {".git", "alsomitra"}:
            continue

        for dataset_dir in sorted(constraint_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            report_dir = dataset_dir

            table_file = f"table_{constraint_dir.name}_{dataset_dir.name}.tex"
            total_seconds += write_table_file(report_dir, table_file)

    total_time = timedelta(seconds=total_seconds)
    hours, remainder = divmod(total_time.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    print(f"Total time: {total_time.days} days {hours} hours {minutes} minutes")


if __name__ == "__main__":
    main()
