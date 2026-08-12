"""Exploratory analysis of the Telco customer churn dataset.

This script investigates patterns relevant to customer churn and publishes the
findings as a PDF report under ``reports``. It is deliberately separate from
``assess_data_quality.py``: that script asks whether the observed data conforms
to documented expectations, while this one investigates the data itself.

Investigation 1 — retention vs. churn baseline — is the only analysis
implemented so far. It measures how the target variable is distributed across
retained and churned customers, renders that distribution as a chart and a
table, and writes a strictly data-derived interpretation. Every figure in the
report is computed from the loaded dataset; none is written by hand.

The raw dataset is only read from. Nothing here cleans, imputes, encodes, or
otherwise modifies it.
"""

# --- Standard library imports ---
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

# --- Third-party imports ---
import matplotlib

# Select a non-interactive backend before importing pyplot, so the script
# renders to file without needing a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# --- Make the reusable "churner" package importable ---
# The script lives in ``<project_root>/scripts``, so the project root is one
# level up. The source code lives under ``<project_root>/src``. Adding that
# directory to ``sys.path`` lets us import the existing module without having
# the package installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_DIR))

# --- Import the existing, reusable data-loading function ---
# We deliberately reuse this function instead of calling ``pandas.read_csv``
# directly, so the analysis runs against the real loading logic.
from churner.data.load_dataset import load_dataset

# --- Resolve paths relative to the project root ---
# Building paths from ``PROJECT_ROOT`` avoids hardcoding an absolute path and
# keeps the script portable across machines.
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "churn_eda_report.pdf"

# --- Target variable ---
# The data dictionary documents ``Churn`` as the label column with the domain
# {Yes, No}, where "Yes" means the customer left during the last period. The
# domain is stated here as an expectation to compare against, not as a filter:
# any other value is reported rather than dropped.
TARGET_COLUMN = "Churn"
RETAINED_VALUE = "No"
CHURNED_VALUE = "Yes"

RETAINED_LABEL = "Retained"
CHURNED_LABEL = "Churned"
MISSING_LABEL = "Missing target value"

# --- Report layout ---
# A4 portrait in inches, so the pages print without rescaling.
PAGE_SIZE_INCHES = (8.27, 11.69)
PAGE_MARGIN = 0.09
TITLE_FONT_SIZE = 20
HEADING_FONT_SIZE = 14
BODY_FONT_SIZE = 10.5

# Character width used to wrap prose. Chosen so a line of the monospace body
# font fits between the page margins above.
TEXT_WRAP_WIDTH = 74

REPORT_TITLE = "Customer Churn — Exploratory Data Analysis"

# --- Chart colours ---
# One muted colour per status, kept consistent between the chart and any later
# investigation. Unexpected categories get a neutral grey so they read as
# "needs investigation" rather than as a normal outcome.
RETAINED_COLOR = "#2F6F9F"
CHURNED_COLOR = "#C1553B"
UNEXPECTED_COLOR = "#8C8C8C"


@dataclass(frozen=True)
class RetentionSummary:
    """Counts of the target variable for the retention vs. churn baseline.

    The counts are computed once and passed to the chart, the table, and the
    interpretation, so every part of the report describes the same numbers.
    ``unexpected_counts`` holds any target value outside the documented domain,
    keyed by the observed value; it is empty when the domain holds.
    """

    total_customers: int
    retained_count: int
    churned_count: int
    missing_count: int
    unexpected_counts: pd.Series

    def percent_of_total(self, count: int) -> float:
        """Express ``count`` as a percentage of all customers in the dataset."""
        if self.total_customers == 0:
            return 0.0
        return count / self.total_customers * 100

    @property
    def retained_percent(self) -> float:
        """Share of all customers that were retained."""
        return self.percent_of_total(self.retained_count)

    @property
    def churned_percent(self) -> float:
        """Share of all customers that churned."""
        return self.percent_of_total(self.churned_count)

    @property
    def has_unexpected_values(self) -> bool:
        """Whether the target holds missing or undocumented values."""
        return self.missing_count > 0 or not self.unexpected_counts.empty


def summarize_retention(df: pd.DataFrame) -> RetentionSummary:
    """Count retained, churned, missing, and undocumented target values.

    Percentages are derived from these counts rather than stored, so the report
    cannot drift from the data. The frame is only read from, and no row is
    excluded: values outside the documented domain are counted separately
    instead of being discarded.
    """
    target_values = df[TARGET_COLUMN]
    missing = target_values.isna()
    unexpected = ~target_values.isin([RETAINED_VALUE, CHURNED_VALUE]) & ~missing

    return RetentionSummary(
        total_customers=int(len(df)),
        retained_count=int((target_values == RETAINED_VALUE).sum()),
        churned_count=int((target_values == CHURNED_VALUE).sum()),
        missing_count=int(missing.sum()),
        unexpected_counts=target_values[unexpected].value_counts(),
    )


def build_status_counts(summary: RetentionSummary) -> list[tuple[str, int, str]]:
    """List each customer status with its count and chart colour.

    The chart and the table are both built from this single list, so they
    cannot disagree. Rows for missing or undocumented values appear only when
    such values exist, which keeps the reported statuses exhaustive: the counts
    always add up to the dataset size.
    """
    status_counts = [
        (RETAINED_LABEL, summary.retained_count, RETAINED_COLOR),
        (CHURNED_LABEL, summary.churned_count, CHURNED_COLOR),
    ]

    for observed_value, count in summary.unexpected_counts.items():
        label = f'Unexpected: "{observed_value}"'
        status_counts.append((label, int(count), UNEXPECTED_COLOR))

    if summary.missing_count > 0:
        status_counts.append((MISSING_LABEL, summary.missing_count, UNEXPECTED_COLOR))

    return status_counts


def format_count(count: int) -> str:
    """Format a customer count with thousands separators."""
    return f"{count:,}"


def format_percent(percent: float) -> str:
    """Format a percentage to two decimal places."""
    return f"{percent:.2f}%"


def build_summary_table_rows(summary: RetentionSummary) -> list[list[str]]:
    """Build the supporting table: one row per status, plus a total row.

    The total row restates the dataset size and 100%, both of which follow from
    the counts above it rather than being asserted independently.
    """
    rows = [
        [label, format_count(count), format_percent(summary.percent_of_total(count))]
        for label, count, _ in build_status_counts(summary)
    ]
    rows.append(["Total", format_count(summary.total_customers), format_percent(100.0)])
    return rows


def create_page_figure() -> Figure:
    """Create a blank portrait page sized for the report."""
    return plt.figure(figsize=PAGE_SIZE_INCHES)


def wrap_paragraphs(paragraphs: list[str]) -> str:
    """Wrap prose to the body width, keeping a blank line between paragraphs."""
    wrapped = [textwrap.fill(paragraph, width=TEXT_WRAP_WIDTH) for paragraph in paragraphs]
    return "\n\n".join(wrapped)


def draw_heading(figure: Figure, text: str, top: float) -> float:
    """Draw a section heading at ``top`` and return the y position below it.

    Positions are figure fractions measured from the bottom of the page, so the
    caller can stack blocks down the page without tracking font metrics.
    """
    figure.text(PAGE_MARGIN, top, text, fontsize=HEADING_FONT_SIZE, fontweight="bold", va="top")
    return top - 0.035


def draw_body_text(figure: Figure, text: str, top: float) -> None:
    """Draw monospace body text starting at ``top``.

    Monospace keeps the label/value lines of the overview section aligned
    without needing a table.
    """
    figure.text(
        PAGE_MARGIN,
        top,
        text,
        fontsize=BODY_FONT_SIZE,
        va="top",
        family="monospace",
        linespacing=1.7,
    )


def build_overview_lines(summary: RetentionSummary) -> list[str]:
    """Describe the dataset the report was generated from.

    The path is shown relative to the project root so the report identifies the
    input without leaking a machine-specific absolute path.
    """
    relative_dataset_path = DATASET_PATH.relative_to(PROJECT_ROOT).as_posix()
    return [
        f"{'Dataset:':<22}{relative_dataset_path}",
        f"{'Total records:':<22}{format_count(summary.total_customers)}",
        f"{'Target variable:':<22}{TARGET_COLUMN}",
        f"{'Analysis scope:':<22}Investigation 1 — retention vs. churn baseline",
        f"{'':<22}Distribution of the target variable only.",
        f"{'':<22}No other variable is examined in this report.",
        f"{'Data treatment:':<22}Read-only. The raw dataset is not modified,",
        f"{'':<22}cleaned, or transformed by this analysis.",
    ]


def create_title_and_overview_page(summary: RetentionSummary) -> Figure:
    """Render the title page with Section 1 — Dataset Overview."""
    figure = create_page_figure()

    top = 1 - PAGE_MARGIN
    figure.text(PAGE_MARGIN, top, REPORT_TITLE, fontsize=TITLE_FONT_SIZE, fontweight="bold", va="top")
    top -= 0.06

    top = draw_heading(figure, "1. Dataset Overview", top)
    draw_body_text(figure, "\n".join(build_overview_lines(summary)), top)

    return figure


def draw_retention_chart(axes: plt.Axes, summary: RetentionSummary) -> None:
    """Draw the retention vs. churn bar chart on ``axes``.

    A plain bar chart is used so the relative proportions are readable at a
    glance. Each bar is annotated with both its count and its share of the
    dataset, which keeps the chart self-explanatory for a business reader and
    exact for a technical one.
    """
    status_counts = build_status_counts(summary)
    labels = [label for label, _, _ in status_counts]
    counts = [count for _, count, _ in status_counts]
    colors = [color for _, _, color in status_counts]

    bars = axes.bar(labels, counts, color=colors, width=0.55)

    for bar, count in zip(bars, counts):
        annotation = f"{format_count(count)}\n{format_percent(summary.percent_of_total(count))}"
        axes.annotate(
            annotation,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    axes.set_title("Customers by status", fontsize=13, pad=14)
    axes.set_ylabel("Customer count")
    axes.tick_params(axis="x", labelsize=11)

    # Headroom so the annotations above the tallest bar stay inside the axes.
    axes.set_ylim(0, max(counts) * 1.18 if counts else 1)

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="y", linestyle=":", alpha=0.4)
    axes.set_axisbelow(True)


def draw_summary_table(axes: plt.Axes, summary: RetentionSummary) -> None:
    """Draw the supporting table on ``axes``."""
    axes.axis("off")

    column_labels = ["Customer Status", "Customer Count", "Percentage"]
    rows = build_summary_table_rows(summary)

    table = axes.table(
        cellText=rows,
        colLabels=column_labels,
        cellLoc="right",
        colLoc="right",
        loc="upper center",
        colWidths=[0.44, 0.26, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(BODY_FONT_SIZE)
    table.scale(1, 1.7)

    header_row = 0
    total_row = len(rows)
    status_column = 0

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        if column_index == status_column:
            cell.get_text().set_horizontalalignment("left")
            # Padding keeps the left-aligned labels off the cell border.
            cell.PAD = 0.04
        if row_index == header_row:
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#EDF1F5")
        elif row_index == total_row:
            cell.get_text().set_fontweight("bold")


def create_retention_page(summary: RetentionSummary) -> Figure:
    """Render Section 2 — Retention vs. Churn, with the chart and the table."""
    figure = create_page_figure()

    top = draw_heading(figure, "2. Retention vs. Churn", 1 - PAGE_MARGIN)

    figure.text(
        PAGE_MARGIN,
        top,
        "Counts and percentages are computed from the loaded dataset.",
        fontsize=BODY_FONT_SIZE,
        va="top",
        style="italic",
        color="#555555",
    )

    # Fixed rectangles rather than a grid, so the chart and the table keep the
    # same position on the page as later investigations are added.
    chart_axes = figure.add_axes((0.13, 0.47, 0.78, 0.34))
    draw_retention_chart(chart_axes, summary)

    table_axes = figure.add_axes((0.10, 0.20, 0.80, 0.22))
    draw_summary_table(table_axes, summary)

    return figure


def build_interpretation_paragraphs(summary: RetentionSummary) -> list[str]:
    """Describe what the counts show, without going beyond them.

    Every sentence restates a measured quantity or a direct arithmetic
    consequence of one. The interpretation deliberately makes no causal claim
    and offers no business explanation for churn, because this investigation
    examines the target variable alone and holds no evidence about why any
    customer left.
    """
    retained_text = (
        f"{format_count(summary.retained_count)} customers "
        f"({format_percent(summary.retained_percent)})"
    )
    churned_text = (
        f"{format_count(summary.churned_count)} customers "
        f"({format_percent(summary.churned_percent)})"
    )

    paragraphs = [
        f"Of the {format_count(summary.total_customers)} customers in the dataset, "
        f'{retained_text} are recorded as retained ({TARGET_COLUMN} = "{RETAINED_VALUE}") '
        f'and {churned_text} are recorded as churned ({TARGET_COLUMN} = "{CHURNED_VALUE}").',
        "Retained customers therefore outnumber churned customers by "
        f"{format_count(summary.retained_count - summary.churned_count)} customers, a gap of "
        f"{format_percent(summary.retained_percent - summary.churned_percent)} of the dataset.",
    ]

    if summary.has_unexpected_values:
        paragraphs.append(
            f"The target variable does not consist solely of the documented values "
            f'"{RETAINED_VALUE}" and "{CHURNED_VALUE}". The affected records are listed as '
            "separate rows in the table above and are reported here rather than removed or "
            "reassigned. They require investigation before the retention and churn "
            "percentages above are treated as covering the full dataset."
        )
    else:
        paragraphs.append(
            f'Every record carries one of the two documented target values, "{RETAINED_VALUE}" '
            f'or "{CHURNED_VALUE}"; the target contains no missing values and no other '
            "category. The two percentages above therefore account for the whole dataset."
        )

    paragraphs.append(
        "These proportions are the baseline for the exploratory analysis that follows. "
        "Subsequent investigations compare the churn rate within individual segments of the "
        "data against the overall rate reported here, and the same proportions describe the "
        "class balance any later model would be trained against."
    )
    paragraphs.append(
        "This investigation examines the distribution of the target variable only. It "
        "establishes how many customers churned, not why: no relationship between churn and "
        "any other variable has been measured at this stage, and no cause is inferred from "
        "these counts."
    )

    return paragraphs


def create_interpretation_page(summary: RetentionSummary) -> Figure:
    """Render Section 3 — Interpretation."""
    figure = create_page_figure()

    top = draw_heading(figure, "3. Interpretation", 1 - PAGE_MARGIN)
    draw_body_text(figure, wrap_paragraphs(build_interpretation_paragraphs(summary)), top)

    return figure


def write_report(summary: RetentionSummary, output_path: Path) -> None:
    """Write the report pages to ``output_path`` as a single PDF.

    Each figure is closed after it is written so repeated runs do not
    accumulate open figures.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_builders = (
        create_title_and_overview_page,
        create_retention_page,
        create_interpretation_page,
    )

    with PdfPages(output_path) as pdf:
        for build_page in page_builders:
            figure = build_page(summary)
            pdf.savefig(figure)
            plt.close(figure)


def print_console_summary(summary: RetentionSummary) -> None:
    """Print the same figures the report contains, for a run-time check."""
    print("Retention vs. churn baseline")
    print("-" * 40)
    print(f"{'Total customers:':<24}{format_count(summary.total_customers)}")
    for label, count, _ in build_status_counts(summary):
        percent = format_percent(summary.percent_of_total(count))
        print(f"{label + ':':<24}{format_count(count):>8}{percent:>10}")

    if summary.has_unexpected_values:
        print()
        print(
            f"Unexpected {TARGET_COLUMN} values detected. They are reported, not removed; "
            "see the report table."
        )


def main() -> None:
    """Run investigation 1 and write the EDA report."""
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    # Load the dataset using the reusable module. The DataFrame is only read
    # from; no operation below writes back to it.
    customer_churn_df = load_dataset(str(DATASET_PATH))

    if TARGET_COLUMN not in customer_churn_df.columns:
        raise SystemExit(
            f"Target column '{TARGET_COLUMN}' not found in {DATASET_PATH.name}. "
            f"Columns present: {list(customer_churn_df.columns)}"
        )

    summary = summarize_retention(customer_churn_df)

    print_console_summary(summary)

    write_report(summary, REPORT_PATH)

    print()
    print(f"Report written to: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
