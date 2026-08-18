"""Exploratory analysis of the Telco customer churn dataset.

This script investigates patterns relevant to customer churn and publishes the
findings as a PDF report under ``reports``. It is deliberately separate from
``assess_data_quality.py``: that script asks whether the observed data conforms
to documented expectations, while this one investigates the data itself.

Investigation 1 — retention vs. churn baseline — measures how the target
variable is distributed across retained and churned customers, renders that
distribution as a chart and a table, and writes a strictly data-derived
interpretation.

Investigation 2 — tenure vs. churn — compares the tenure distributions of the
two churn populations as a histogram and a box plot. Both are drawn from the
raw tenure observations rather than from any summary of them, because a summary
holds no individual observation to plot. The 6-month bars of the histogram are
a device for displaying a continuous variable; they are not analytical tenure
bands, and no churn rate is computed per bar.

Every figure in the report is computed from the loaded dataset; none is written
by hand. The raw dataset is only read from. Nothing here cleans, imputes,
encodes, or otherwise modifies it.
"""

# --- Standard library imports ---
import math
import sys
import textwrap
from dataclasses import dataclass
from functools import partial
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

# --- Tenure variable ---
# The data dictionary documents ``tenure`` as the number of months the customer
# has stayed with the company.
TENURE_COLUMN = "tenure"

# Width of the histogram bars, in months. Six months groups the observed range
# finely enough to show where observations concentrate while keeping each bar
# supported by enough customers to be readable. It is a display choice about
# this chart alone: no analytical tenure band is defined anywhere in this
# report, and no rate is computed per bar.
TENURE_BIN_WIDTH_MONTHS = 6

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
        f"{'':<22}Distribution of the target variable.",
        f"{'':<22}Investigation 2 — tenure vs. churn",
        f"{'':<22}Tenure distributions of the two churn",
        f"{'':<22}populations, described only. No other",
        f"{'':<22}variable is examined in this report.",
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
    """Render Section 3 — Interpretation of the retention vs. churn baseline."""
    figure = create_page_figure()

    top = draw_heading(figure, "3. Interpretation — Retention vs. Churn", 1 - PAGE_MARGIN)
    draw_body_text(figure, wrap_paragraphs(build_interpretation_paragraphs(summary)), top)

    return figure


@dataclass(frozen=True)
class TenureObservations:
    """The raw tenure observations of each churn population.

    Both the histogram and the box plot are drawn from this one object, so the
    two views cannot end up describing different data. The series hold the
    individual observations rather than a summary of them: a distribution
    cannot be reconstructed from means and quartiles.

    Customers whose tenure is missing are absent from these series, because a
    missing number has no position on an axis. They are excluded from the plot
    alone; nothing is imputed, and the loaded dataset keeps them. A customer
    whose target value is missing appears in neither population.
    """

    retained: pd.Series
    churned: pd.Series

    @property
    def combined(self) -> pd.Series:
        """Both populations' observations, for measuring their shared scale."""
        return pd.concat([self.retained, self.churned])

    @property
    def observed_range(self) -> tuple[float, float]:
        """Lowest and highest tenure observed across both populations.

        Read from the data rather than assumed, so the charts describe the
        range the dataset actually holds instead of one hardcoded here.
        """
        combined = self.combined
        return float(combined.min()), float(combined.max())


def collect_tenure_observations(df: pd.DataFrame) -> TenureObservations:
    """Split the tenure observations into the two churn populations.

    Selecting with a mask and dropping missing values both return new series,
    so the loaded frame is read from and never written back to.
    """
    tenure_values = df[TENURE_COLUMN]
    target_values = df[TARGET_COLUMN]

    return TenureObservations(
        retained=tenure_values[target_values == RETAINED_VALUE].dropna(),
        churned=tenure_values[target_values == CHURNED_VALUE].dropna(),
    )


def build_tenure_bin_edges(observations: TenureObservations) -> list[float]:
    """Build one set of histogram bin edges covering both populations.

    The edges are computed once from the combined observations and used for
    both populations, so the bars line up and the two distributions can be
    compared bar for bar. Edges start at a whole multiple of the bar width at
    or below the lowest observation and continue past the highest, so every
    observation falls inside a bar.
    """
    lowest, highest = observations.observed_range

    # No observation to cover: return a single empty bar of the usual width,
    # so the page still renders rather than failing on an undefined range.
    if math.isnan(lowest):
        return [0.0, float(TENURE_BIN_WIDTH_MONTHS)]

    first_edge = math.floor(lowest / TENURE_BIN_WIDTH_MONTHS) * TENURE_BIN_WIDTH_MONTHS
    edges = [float(first_edge)]
    while edges[-1] < highest or len(edges) < 2:
        edges.append(edges[-1] + TENURE_BIN_WIDTH_MONTHS)

    return edges


def build_tenure_populations(
    observations: TenureObservations,
) -> list[tuple[str, pd.Series, str]]:
    """List each churn population with its observations and chart colour.

    Both charts iterate this list, which keeps their colours, their labels, and
    the order they read in consistent with each other and with the retention
    chart in Section 2.
    """
    return [
        (RETAINED_LABEL, observations.retained, RETAINED_COLOR),
        (CHURNED_LABEL, observations.churned, CHURNED_COLOR),
    ]


def draw_tenure_histogram(
    axes: plt.Axes, observations: TenureObservations, bin_edges: list[float]
) -> None:
    """Draw both populations' tenure distributions as one overlaid histogram.

    The bars are semi-transparent and share one set of edges, so the region
    where the two distributions overlap stays visible instead of one hiding the
    other. Counts, rather than proportions, are plotted: the populations differ
    in size, and the chart reports what was observed rather than rescaling it.

    Each population's size is stated in the legend, since a taller bar in the
    larger population does not by itself mean a larger share of it.
    """
    for label, tenure_values, color in build_tenure_populations(observations):
        axes.hist(
            tenure_values,
            bins=bin_edges,
            label=f"{label} (n = {format_count(len(tenure_values))})",
            color=color,
            alpha=0.55,
            edgecolor=color,
            linewidth=1.1,
        )

    axes.set_title("Customers by tenure, in 6-month display bins", fontsize=13, pad=14)
    axes.set_xlabel("Tenure (months)")
    axes.set_ylabel("Customer count")

    # Tick every bar edge, so a bar can be read back to the months it covers.
    axes.set_xticks(bin_edges)
    axes.set_xlim(bin_edges[0], bin_edges[-1])
    axes.legend(frameon=False, fontsize=10)

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="y", linestyle=":", alpha=0.4)
    axes.set_axisbelow(True)


def draw_tenure_box_plot(axes: plt.Axes, observations: TenureObservations) -> None:
    """Draw both populations' tenure as box plots on one shared axis.

    Sharing the axis is the point of the chart: the medians, the quartiles, and
    the spread of the two populations are read against the same scale. Points
    beyond the whiskers are kept, since an exploratory chart should show the
    observations that sit apart from the bulk rather than trim them away.
    """
    populations = build_tenure_populations(observations)

    box_plot = axes.boxplot(
        [tenure_values for _, tenure_values, _ in populations],
        orientation="horizontal",
        tick_labels=[label for label, _, _ in populations],
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "#222222", "linewidth": 1.6},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "none",
            "markeredgecolor": "#555555",
            "alpha": 0.5,
        },
    )

    for box, (_, _, color) in zip(box_plot["boxes"], populations):
        box.set_facecolor(color)
        box.set_alpha(0.55)
        box.set_edgecolor(color)

    lowest, highest = observations.observed_range
    axes.set_title("Median, quartiles, and spread of tenure", fontsize=13, pad=14)
    axes.set_xlabel("Tenure (months)")
    axes.set_xlim(lowest - 2, highest + 2)

    # Read top to bottom in the same order as the histogram legend.
    axes.invert_yaxis()

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="x", linestyle=":", alpha=0.4)
    axes.set_axisbelow(True)


def draw_chart_caption(figure: Figure, text: str, top: float) -> None:
    """Draw the italic note that sits under a section heading."""
    figure.text(
        PAGE_MARGIN,
        top,
        text,
        fontsize=BODY_FONT_SIZE,
        va="top",
        style="italic",
        color="#555555",
    )


def create_tenure_histogram_page(observations: TenureObservations) -> Figure:
    """Render Section 4 — Tenure Distribution by Churn Status."""
    figure = create_page_figure()

    top = draw_heading(figure, "4. Tenure Distribution by Churn Status", 1 - PAGE_MARGIN)
    draw_chart_caption(
        figure,
        textwrap.fill(
            "Every recorded tenure observation is plotted. The 6-month bars display the "
            "continuous distribution; they are not analytical tenure bands, and no churn "
            "rate is computed for them.",
            width=88,
        ),
        top,
    )

    # Top edge kept level with the box plot in Section 5, so the two views of
    # the same observations sit in the same place from page to page.
    chart_axes = figure.add_axes((0.12, 0.24, 0.80, 0.52))
    draw_tenure_histogram(chart_axes, observations, build_tenure_bin_edges(observations))

    return figure


def create_tenure_box_plot_page(observations: TenureObservations) -> Figure:
    """Render Section 5 — Tenure Distribution by Churn Status, as a box plot."""
    figure = create_page_figure()

    top = draw_heading(
        figure, "5. Tenure Distribution by Churn Status — Box Plot", 1 - PAGE_MARGIN
    )
    draw_chart_caption(
        figure,
        textwrap.fill(
            "Drawn from the same tenure observations as Section 4. Boxes span the "
            "interquartile range, the line inside each box is the median, and points "
            "beyond the whiskers are shown rather than removed.",
            width=88,
        ),
        top,
    )

    chart_axes = figure.add_axes((0.14, 0.46, 0.78, 0.30))
    draw_tenure_box_plot(chart_axes, observations)

    return figure


def write_report(
    summary: RetentionSummary,
    tenure_observations: TenureObservations,
    output_path: Path,
) -> None:
    """Write the report pages to ``output_path`` as a single PDF.

    Each page is built only when it is about to be written, and its figure is
    closed afterwards, so repeated runs do not accumulate open figures.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_builders = (
        partial(create_title_and_overview_page, summary),
        partial(create_retention_page, summary),
        partial(create_interpretation_page, summary),
        partial(create_tenure_histogram_page, tenure_observations),
        partial(create_tenure_box_plot_page, tenure_observations),
    )

    with PdfPages(output_path) as pdf:
        for build_page in page_builders:
            figure = build_page()
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


def print_tenure_console_summary(observations: TenureObservations) -> None:
    """Print how many observations each tenure chart was drawn from.

    Reporting the counts and the observed range at run time makes it visible
    which data reached the charts, including how many customers were left out
    of them for holding no tenure value.
    """
    print()
    print("Tenure observations plotted, by churn status")
    print("-" * 44)
    for label, tenure_values, _ in build_tenure_populations(observations):
        print(f"{label + ':':<24}{format_count(len(tenure_values)):>8} observations")

    lowest, highest = observations.observed_range
    print(f"{'Observed range:':<24}{lowest:>8.0f} to {highest:.0f} months")


def main() -> None:
    """Run investigations 1 and 2, then write the EDA report."""
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    # Load the dataset using the reusable module. The DataFrame is only read
    # from; no operation below writes back to it.
    customer_churn_df = load_dataset(str(DATASET_PATH))

    missing_columns = [
        column_name
        for column_name in (TARGET_COLUMN, TENURE_COLUMN)
        if column_name not in customer_churn_df.columns
    ]
    if missing_columns:
        raise SystemExit(
            f"Required column(s) {missing_columns} not found in {DATASET_PATH.name}. "
            f"Columns present: {list(customer_churn_df.columns)}"
        )

    summary = summarize_retention(customer_churn_df)
    tenure_observations = collect_tenure_observations(customer_churn_df)

    print_console_summary(summary)
    print_tenure_console_summary(tenure_observations)

    write_report(summary, tenure_observations, REPORT_PATH)

    print()
    print(f"Report written to: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
