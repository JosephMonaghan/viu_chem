import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import chi2, false_discovery_control, ttest_ind, ttest_rel
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Patch

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rc('font', serif='Helvetica') 

def spectrum(mz:list[float], intensity:list[float],ax:plt.Axes=None, color:str="k",title:str=None, annotate_peaks:bool=True, annotate_percent:int=25, invert:bool = False) -> plt.Axes:
    """Plots a mass spectrum for a provided list of m/z and intensities. Draws in the specified ax object or generates it's own if none is provided.
    
    :param mz: List of m/z values
    :param intensity: List of intensity values
    :param ax: Target matplotlib axes object
    :param color: Color to draw the spectrum vertical lines
    :param title: Optional title string to draw over the plot
    :return ax: Returns the populated axes object"""
    if not ax:
        fig, ax = plt.subplots()

    if not invert:
        ax.vlines(mz,0,intensity,color=color)
    else:
        ax.vlines(mz,np.multiply(intensity,-1), 0,color=color)
    
    
    ax.set_xlabel("m/z",style='italic', fontweight='bold')
    ax.set_ylabel("Signal", fontweight='bold')
    ax.axhline(0, color='k')
    ylims = ax.get_ylim()


    if not invert:
        ax.set_ylim(ylims[0], ylims[1]*1.2)
    else:
        ax.set_ylim(ylims[0]*1.2, ylims[1])

    if title:
        ax.set_title(title, fontweight='bold')

    if annotate_peaks:
        data = pd.DataFrame({
            "mz": mz, 
            "intensity": intensity})
        
        max_signal = np.max(intensity)

        for i, (mz_ind, intensity_ind) in enumerate(zip(mz, intensity)):
            # Find peaks within ±1 m/z window
            window_mask = (data.mz >= mz_ind - 1) & (data.mz <= mz_ind + 1)
            window_data = data[window_mask].copy()
            window_data['intensity'] = abs(window_data['intensity'])
            max_idx = window_data['intensity'].idxmax()

            # Only label if current peak is the highest in the window
            if max_idx == data.index[i] and abs(window_data['intensity'].max()) > max_signal*annotate_percent/100:
                if not invert:
                    ax.annotate(f"{mz_ind:.4f}", xy=(mz_ind, intensity_ind), xytext=(0, 5),
                                    textcoords='offset points', ha='center', fontsize=8,
                                    fontweight='bold', fontstyle='italic', color=color)
                else:
                    ax.annotate(f"{mz_ind:.4f}", xy=(mz_ind, intensity_ind*-1), xytext=(0, -5),
                                    textcoords='offset points', ha='center', fontsize=8,
                                    fontweight='bold', fontstyle='italic', color=color)

    
    return ax


def cal_curve(x:list[float], y:list[float], ax:plt.Axes=None,xlabel:str="Your x label here!", ylabel:str="Your y label here!", color:str="#8C4FA4",slope_pos:tuple[float,float]=(0.02,0.95)) -> plt.Axes:
    """Generates a calibration curve for a given set of data x / y into the specified axes. If not axes provided it generates its own.
    
    :param x: List of x values
    :param y: List of y values
    :param ax: Target axes
    :param xlabel: String for the x axis label
    :param ylabel: String for the y axis label
    :param color: What color to draw the points
    :param slope_pos: Where to draw the slope and r2 text
    :return ax: Returns the populated axes object
    :return coeffs: Slope and intercept for the best-fit line
    :return r2: Coefficient of determination (R2) for the line"""
    if not ax:
        fig, ax = plt.subplots()
    
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    #Generate linear fit
    coeffs = np.polyfit(x, y, 1)
    poly_eq = np.poly1d(coeffs)
    x_fit = np.unique(x)
    y_fit = poly_eq(x_fit)

    y_pred = poly_eq(x)
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    
    ax.plot(x_fit, y_fit, color=color,linestyle='--')
    ax.scatter(x, y, marker='s',edgecolors='k', color=color)

    # Prepare the equation string
    slope, intercept = coeffs
    equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}"

    # Add text annotation for the equation
    ax.text(slope_pos[0], slope_pos[1], equation, transform=plt.gca().transAxes, va='top',ha='center', color=color)
    
    ax.set_xlabel(xlabel,fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    return ax, coeffs, r2


def volcano(data_numerator:pd.DataFrame,
            data_denom:pd.DataFrame,
            ax:plt.Axes = None,
            color_denom:str="#1C85C2",
            color_numer:str="#DA0000",
            marker_color="#BCBCBC",
            sig_cutoff:float=0.05,
            left_label:str="Denominator",
            right_label:str="Numerator",
            xlabel:str = None,
            ylabel:str = "p-value",
            paired:bool = False,
            cutoff_specifier:float=2.0,
            fdr_control:bool=False):
    """Generates a volcano plot comparing two dataframes into the specified axes. If no axes provided it generates its own.
    
    :param data_numerator: Dataframe containing numerator group values
    :param data_denom: Dataframe containing denominator group values
    :param ax: Target axes
    :param color_denom: Color for the denominator side gradient
    :param color_numer: Color for the numerator side gradient
    :param marker_color: Color to draw the data points
    :param sig_cutoff: P-value cutoff for significance
    :param cutoff_specifier: Absolute fold-change cutoff for significance (for example, 1.5 means 1.5-fold)
    :param left_label: Label for the denominator side of the plot
    :param right_label: Label for the numerator side of the plot
    :param xlabel: String for the x axis label
    :param ylabel: String for the y axis label
    :param paired: Pair samples by column position, using the mean matched log2 ratio and a paired t-test on log2 intensities
    :param fdr_control: Apply Benjamini-Hochberg correction and use adjusted p-values for plotting and significance
    :return ax: Returns the populated axes object
    :return return_df: Dataframe containing fold changes and p-values"""
    
    # Check that dataframes are coherent and match
    if not data_numerator.index.equals(data_denom.index):
        raise ValueError("Dataframes do not match!")
    if paired and data_numerator.shape[1] != data_denom.shape[1]:
        raise ValueError("Paired dataframes must contain the same number of samples!")
    cutoff_specifier = float(cutoff_specifier)
    if not np.isfinite(cutoff_specifier) or cutoff_specifier <= 1:
        raise ValueError("cutoff_specifier must be finite and greater than 1.")
    log2_cutoff = float(np.log2(cutoff_specifier))

    # If no axis specified, make one
    if not ax:
        fig, ax = plt.subplots()
    
    # Generate fold change and pval data
    index = data_numerator.index

    fold_changes = np.full(len(index), np.nan, dtype=float)
    pvals = np.full(len(index), np.nan, dtype=float)
    invalid = np.zeros(len(index), dtype=bool)

    for idx, mz in enumerate(index):
        local_numer = data_numerator.loc[mz]
        local_denom = data_denom.loc[mz]
        if paired:
            numer_values = np.asarray(local_numer, dtype=float)
            denom_values = np.asarray(local_denom, dtype=float)
            valid_pairs = (
                np.isfinite(numer_values)
                & np.isfinite(denom_values)
                & (numer_values > 0)
                & (denom_values > 0)
            )
            if np.count_nonzero(valid_pairs) < 2:
                invalid[idx] = True
                continue
            log_numer = np.log2(numer_values[valid_pairs])
            log_denom = np.log2(denom_values[valid_pairs])
            fold_changes[idx] = float(np.mean(log_numer - log_denom))
            _, p_val = ttest_rel(log_numer, log_denom)
            pvals[idx] = p_val
            continue

        numer_mean = local_numer.mean()
        denom_mean = local_denom.mean()
        if (denom_mean != 0) and (numer_mean != 0) and ((numer_mean / denom_mean) > 0):
            fold_changes[idx] = np.log2(numer_mean / denom_mean)
            _, p_val = ttest_ind(local_numer, local_denom, equal_var=False)
            pvals[idx] = p_val
        else:
            invalid[idx] = True

    

    # Actual plotting
    valid = (~invalid) & np.isfinite(fold_changes) & np.isfinite(pvals) & (pvals >= 0) & (pvals <= 1)
    result_columns = {
        "fold_change": fold_changes,
        "pval": pvals,
    }
    significance_pvals = pvals.copy()
    if fdr_control:
        adjusted_pvals = np.full(len(index), np.nan, dtype=float)
        if valid.any():
            adjusted_pvals[valid] = false_discovery_control(pvals[valid], method="bh")
        significance_pvals = adjusted_pvals
        result_columns["adjusted_pval"] = adjusted_pvals
    return_df = pd.DataFrame(result_columns, index=index)
    return_df = return_df[valid]

    plot_ylabel = "BH-adjusted p-value" if fdr_control and ylabel == "p-value" else ylabel

    if not valid.any():
        ax.set_yscale("log")
        ax.set_ylabel(plot_ylabel, fontweight='bold')
        if not xlabel:
            xlabel = f"log2({right_label} / {left_label})"
        ax.set_xlabel(xlabel, fontweight='bold')
        return ax, return_df

    plot_pvals = significance_pvals.copy()
    positive_pvals = plot_pvals[valid & (plot_pvals > 0)]
    if len(positive_pvals):
        pval_floor = np.min(positive_pvals) * 0.1
    else:
        pval_floor = max(sig_cutoff * 0.1, np.nextafter(0, 1))
    plot_pvals[valid & (plot_pvals <= 0)] = pval_floor

    sig = valid & (np.abs(fold_changes) > log2_cutoff) & (significance_pvals < sig_cutoff)
    ax.set_yscale("log")
    ax.scatter(
        fold_changes[valid],
        plot_pvals[valid],
        marker="s",
        edgecolors='k',
        facecolors=marker_color,
        zorder=10,
    )
    if sig.any():
        sig_colors = _volcano_sig_colors(
            fold_changes[sig],
            marker_color,
            color_denom,
            color_numer,
            log2_cutoff,
        )
        ax.scatter(
            fold_changes[sig],
            plot_pvals[sig],
            marker="s",
            edgecolors='k',
            facecolors=sig_colors,
            zorder=11,
        )

    occupied = []
    x_sep = 0.2
    y_sep = 0.2


    for change, pval, mz, is_sig in zip(fold_changes, plot_pvals, index, sig):
        mz = float(mz)
        if is_sig:
            too_close = any(
                abs(change - ox) < x_sep and abs(pval - oy) < y_sep
                for ox, oy in occupied
            )
            if not too_close and change > 0:
                ax.text(change-0.05, pval, f"{mz:.4f}", ha='left')
                occupied.append((change, pval))

            elif not too_close and change < 0:
                occupied.append((change, pval))
                ax.text(change+0.05, pval*0.9, f"{mz:.4f}", ha='right')


    overlay_col = "#6A6A6A"
    ax.axhline(sig_cutoff, color=overlay_col,linestyle="--")
    ax.axvline(-log2_cutoff,color=overlay_col,linestyle="--")
    ax.axvline(log2_cutoff, color=overlay_col,linestyle="--")


    xmin, xmax = np.min(fold_changes[valid])*1.2, np.max(fold_changes[valid])*1.2
    ymin, ymax = np.min(plot_pvals[valid]), np.max(plot_pvals[valid])

    if xmin == xmax:
        xmin, xmax = xmin - 1, xmax + 1
    if xmin > -log2_cutoff:
        xmin = -log2_cutoff * 1.2
    if xmax < log2_cutoff:
        xmax = log2_cutoff * 1.2
    if ymin == ymax:
        ymin = max(ymin * 0.5, np.nextafter(0, 1))
        ymax = ymax * 2

    # LEFT gradient (negative fold changes)
    _add_side_gradient(ax, xmin, 0, ymin, ymax, color_denom, direction="left", fade_y=sig_cutoff)

    # RIGHT gradient (positive fold changes)
    _add_side_gradient(ax, 0, xmax, ymin, ymax, color_numer, direction="right",fade_y=sig_cutoff)

    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(plot_ylabel, fontweight='bold')
    if not xlabel:
        xlabel = f"log2({right_label} / {left_label})"
    ax.set_xlabel(xlabel, fontweight='bold')

    ylim = ax.get_ylim()
    ax.text(xmax*0.95,ylim[1]*1.5,right_label, ha='right',fontweight='bold')
    ax.text(xmin*0.95,ylim[1]*1.5,left_label,ha='left',fontweight='bold')

    return ax, return_df


def _volcano_sig_colors(fold_changes, marker_color, color_denom, color_numer, log2_cutoff=1.0):
    """Returns marker fill colors that deepen with significant fold-change magnitude."""
    colors = []
    base = np.array(mcolors.to_rgb(marker_color))
    denom = np.array(mcolors.to_rgb(color_denom))
    numer = np.array(mcolors.to_rgb(color_numer))

    denom_max = np.max(np.abs(fold_changes[fold_changes < 0])) if np.any(fold_changes < 0) else 1
    numer_max = np.max(fold_changes[fold_changes > 0]) if np.any(fold_changes > 0) else 1

    for change in fold_changes:
        if change < 0:
            amount = np.clip(
                (abs(change) - log2_cutoff) / max(denom_max - log2_cutoff, 1),
                0,
                1,
            )
            colors.append(base + (denom - base) * amount)
        else:
            amount = np.clip(
                (change - log2_cutoff) / max(numer_max - log2_cutoff, 1),
                0,
                1,
            )
            colors.append(base + (numer - base) * amount)

    return colors


def _add_side_gradient(ax, x0, x1, y0, y1, color, direction="left", fade_y=0.05):
    """Draws a 2D side gradient onto an axes object.
    
    :param ax: Target matplotlib axes object
    :param x0: Gradient minimum x value
    :param x1: Gradient maximum x value
    :param y0: Gradient minimum y value
    :param y1: Gradient maximum y value
    :param color: Gradient color
    :param direction: Horizontal fade direction
    :param fade_y: Y value where the vertical fade reaches full color"""
    N = 256
    y0 = max(y0, np.nextafter(0, 1))
    if y1 <= y0:
        y1 = y0 * 10

    # Horizontal component
    if direction == "left":
        horiz = np.linspace(1, 0, N)  # white → color
    else:
        horiz = np.linspace(0, 1, N)  # color → white

    # Vertical component (white at bottom → color at top)

    y_vals = np.linspace(y0, y1, N)
    if fade_y <= y0:
        vert = np.zeros(N)
    else:
        vert = 1 - np.clip((y_vals - y0) / (fade_y - y0), 0, 1)



    # Outer product → 2D gradient map
    grad = np.outer(vert, horiz)   # shape = (N, N)

    # Build colormap from white → desired color
    rgba_color = mcolors.to_rgba(color)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "grad_cmap", [(1,1,1,0), rgba_color]
    )

    ax.imshow(
        grad,
        extent=[x0, x1, y0, y1],
        cmap=cmap,
        origin="lower",
        aspect="auto",
        zorder=0
    )

def scores_plot(results:dict, tgt_comp:tuple=(1,2), ax:plt.Axes|None=None,colors:list[str]|None=None):
    scores = results['scores']
    pc_x = f"PC{tgt_comp[0]}"
    pc_y = f"PC{tgt_comp[1]}"

    if not ax:
        fig, ax = plt.subplots()

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for idx, (class_type, group) in enumerate(scores.groupby("Classes")):
        if not colors:
            color = default_colors[idx % len(default_colors)]
        else:
            color = colors[idx]

        ax.scatter(
            group[pc_x],
            group[pc_y],
            label=class_type,
            edgecolors='k',
            zorder=10,
            color=color,
        )
        add_confidence_ellipse(ax,group[pc_x],group[pc_y], alpha=0.5, facecolor=color,edgecolor='none',zorder=5)


    ax.scatter(scores[pc_x],scores[pc_y])
    ax.axhline(0,linestyle='--', color='k',zorder=1)
    ax.axvline(0,linestyle='--', color='k',zorder=1)
    ax.legend()

    ax.set_xlabel(f"{pc_x} ({results['explained'][tgt_comp[0]-1]*100:.1f})%")
    ax.set_ylabel(f"{pc_y} ({results['explained'][tgt_comp[1]-1]*100:.1f})%")

    return ax

def loadings_plot(results,tgt_comp:tuple=(1,2),ax:plt.Axes | None = None, color:str="#7789E3",top_n:int=15):
    loadings = results['loadings']
    loadings['loading_strength'] = np.sqrt(loadings[tgt_comp[0]-1]**2 + loadings[tgt_comp[1]-1]**2)
    top_loadings = loadings.nlargest(top_n,'loading_strength')
    pc_x = f"PC{tgt_comp[0]}"
    pc_y = f"PC{tgt_comp[1]}"

    if not ax:
        fig, ax = plt.subplots()
    
    ax.scatter(loadings[tgt_comp[0]-1], loadings[tgt_comp[1]-1], edgecolors='k', color=color,zorder=10)
    ax.axhline(0,linestyle='--',color='k',zorder=1)
    ax.axvline(0,linestyle='--',color='k',zorder=1)

    for feature, row in top_loadings.iterrows():
        if isinstance(feature,float):
            text = f"{feature:.4f}"
        else:
            text = str(feature)
        ax.text(row[tgt_comp[0]-1],row[tgt_comp[1]-1],text,fontsize=8)

    ax.set_xlabel(f"{pc_x} ({results['explained'][tgt_comp[0]-1]*100:.1f})%")
    ax.set_ylabel(f"{pc_y} ({results['explained'][tgt_comp[1]-1]*100:.1f})%")

    return ax


def add_confidence_ellipse(ax, x, y, confidence=0.95, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 3:
        return

    cov = np.cov(x, y)

    # If covariance is singular or weird, skip
    if np.linalg.det(cov) <= 0:
        return

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(
        eigvecs[1, 0],
        eigvecs[0, 0]
    ))

    # 95% ellipse scale for 2D normal distribution
    scale = np.sqrt(chi2.ppf(confidence, df=2))

    width, height = 2 * scale * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        linewidth=2,
        **kwargs
    )

    ax.add_patch(ellipse)


def unpack_dataframe(
        data:pd.DataFrame,
        value:str,
        primary_group:str,
        secondary_group:str | None = None,
        dropna:bool=True) -> dict:
    """Converts a DataFrame into the dictionary layout used by boxplot and barchart.

    :param data: Source DataFrame
    :param value: Column containing the values to plot
    :param primary_group: Column used for the x-axis groups
    :param secondary_group: Optional column used for grouped series within each x-axis group
    :param dropna: Whether to remove NaN values from each plotted series
    :return plot_data: Dictionary suitable for boxplot or barchart
    """
    required_columns = [value, primary_group]
    if secondary_group is not None:
        required_columns.append(secondary_group)

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"DataFrame is missing required columns: {missing_columns}")

    plot_data = {}
    for primary_value, primary_data in data.groupby(primary_group, sort=False):
        if secondary_group is None:
            values = primary_data[value]
            if dropna:
                values = values.dropna()
            plot_data[primary_value] = values.to_list()
        else:
            plot_data[primary_value] = {}
            for secondary_value, secondary_data in primary_data.groupby(secondary_group, sort=False):
                values = secondary_data[value]
                if dropna:
                    values = values.dropna()
                plot_data[primary_value][secondary_value] = values.to_list()

    return plot_data


def _plot_group_structure(data):
    """Validate plot data and return its primary and secondary group order."""
    top_keys = list(data.keys())
    if not top_keys:
        raise ValueError("data must contain at least one primary group")

    nested = [isinstance(data[top_key], dict) for top_key in top_keys]
    if any(nested) and not all(nested):
        raise ValueError("primary groups must all be grouped or all be ungrouped")

    subkeys = []
    if all(nested):
        # Do not use the first primary group as the schema: later groups may have
        # additional secondary categories, and any category may be absent from a
        # particular primary group.
        for top_key in top_keys:
            for subkey in data[top_key]:
                if subkey not in subkeys:
                    subkeys.append(subkey)

    return top_keys, all(nested), subkeys


def _plot_color(colors, idx=0):
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        return default_colors[idx % len(default_colors)]
    if isinstance(colors, str):
        return colors
    if not colors:
        raise ValueError("colors must contain at least one color")
    return colors[idx % len(colors)]


def _group_offsets(num_subkeys):
    if num_subkeys == 0:
        return np.asarray([]), 0.55
    if num_subkeys == 1:
        return np.asarray([0.0]), 0.55
    group_span = min(0.8, 0.4 + 0.1 * max(num_subkeys - 2, 0))
    offsets = np.linspace(-group_span / 2, group_span / 2, num_subkeys)
    return offsets, offsets[1] - offsets[0]


def _normalized_group_values(data, top_keys, subkeys, autonormalize):
    """Copy nested values as float arrays and optionally normalize per primary group."""
    plot_data = {}
    for top_key in top_keys:
        present_values = {
            subkey: np.asarray(data[top_key][subkey], dtype=float)
            for subkey in subkeys
            if subkey in data[top_key]
        }
        finite_values = [
            values[np.isfinite(values)]
            for values in present_values.values()
            if np.isfinite(values).any()
        ]
        norm_limit = np.nanmax(np.concatenate(finite_values)) if finite_values else 1
        if not autonormalize or norm_limit == 0:
            norm_limit = 1
        plot_data[top_key] = {
            subkey: values / norm_limit
            for subkey, values in present_values.items()
        }
    return plot_data


def _violin_values(values):
    """Return finite values with enough variance for Matplotlib's KDE."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return values
    if len(values) == 1 or np.ptp(values) == 0:
        delta = max(abs(float(values[0])) * 1e-6, 1e-6)
        return values[0] + np.linspace(-delta, delta, max(len(values), 3))
    return values


def boxplot(
        data:dict | pd.DataFrame,
        ax:plt.Axes | None = None,
        colors:str | list[str] | None = None,
        autonormalize:bool=False,
        value:str | None = None,
        primary_group:str | None = None,
        secondary_group:str | None = None):
    if not ax:
        fig, ax = plt.subplots()

    if isinstance(data, pd.DataFrame):
        if value is None or primary_group is None:
            raise ValueError("value and primary_group must be supplied when data is a DataFrame")
        data = unpack_dataframe(data, value, primary_group, secondary_group)
    
    top_keys, subkeys_present, subkeys = _plot_group_structure(data)

    if subkeys_present:
        plot_data = _normalized_group_values(
            data, top_keys, subkeys, autonormalize
        )
        offsets, width = _group_offsets(len(subkeys))

        for idx, (subkey, offset) in enumerate(zip(subkeys, offsets)):
            available = [
                (top_idx, plot_data[top_key][subkey])
                for top_idx, top_key in enumerate(top_keys)
                if subkey in plot_data[top_key]
                and np.isfinite(plot_data[top_key][subkey]).any()
            ]
            if not available:
                continue
            key_data = [values[np.isfinite(values)] for _, values in available]
            positions = [top_idx + 1 + offset for top_idx, _ in available]
            color = _plot_color(colors, idx)

            ax.boxplot(key_data,
                       positions=positions,
                       widths=width,
                       label=str(subkey),
                       patch_artist=True,
                       boxprops={'facecolor':color, 'edgecolor':'k'},
                       medianprops={'color':'k'})
        
        ax.legend()
    
    else:
        color = _plot_color(colors)
        bp_data = [data[top_key] for top_key in top_keys]
        ax.boxplot(bp_data,
                   patch_artist=True,
                   boxprops={'facecolor':color,'edgecolor':'k'},
                   medianprops={'color':'k'})
    
    ax.set_xticks([x+1 for x in range(len(top_keys))],top_keys, rotation=45,ha='right')
    return ax


def violinplot(
        data:dict | pd.DataFrame,
        ax:plt.Axes | None = None,
        colors:str | list[str] | None = None,
        autonormalize:bool=False,
        value:str | None = None,
        primary_group:str | None = None,
        secondary_group:str | None = None):
    """Plot violins using the same dictionary or DataFrame workflow as boxplot.

    Secondary categories are discovered across every primary group. Missing
    primary/secondary combinations are omitted without shifting the remaining
    categories, so colors and horizontal positions stay consistent.
    """
    if not ax:
        fig, ax = plt.subplots()

    if isinstance(data, pd.DataFrame):
        if value is None or primary_group is None:
            raise ValueError("value and primary_group must be supplied when data is a DataFrame")
        data = unpack_dataframe(data, value, primary_group, secondary_group)

    top_keys, subkeys_present, subkeys = _plot_group_structure(data)
    if subkeys_present:
        plot_data = _normalized_group_values(
            data, top_keys, subkeys, autonormalize
        )
        offsets, width = _group_offsets(len(subkeys))
        legend_handles = []

        for idx, (subkey, offset) in enumerate(zip(subkeys, offsets)):
            available = [
                (top_idx, plot_data[top_key][subkey])
                for top_idx, top_key in enumerate(top_keys)
                if subkey in plot_data[top_key]
                and np.isfinite(plot_data[top_key][subkey]).any()
            ]
            if not available:
                continue
            datasets = [_violin_values(values) for _, values in available]
            positions = [top_idx + 1 + offset for top_idx, _ in available]
            color = _plot_color(colors, idx)
            artists = ax.violinplot(datasets, positions=positions, widths=width)
            for body in artists['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor('k')
                body.set_alpha(1)
            for name in ('cbars', 'cmins', 'cmaxes'):
                artists[name].set_color('k')
            legend_handles.append(Patch(facecolor=color, edgecolor='k', label=str(subkey)))

        if legend_handles:
            ax.legend(handles=legend_handles)
    else:
        color = _plot_color(colors)
        available = [
            (top_idx, _violin_values(data[top_key]))
            for top_idx, top_key in enumerate(top_keys)
        ]
        available = [(top_idx, values) for top_idx, values in available if len(values)]
        if available:
            artists = ax.violinplot(
                [values for _, values in available],
                positions=[top_idx + 1 for top_idx, _ in available],
                widths=0.55,
            )
            for body in artists['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor('k')
                body.set_alpha(1)
            for name in ('cbars', 'cmins', 'cmaxes'):
                artists[name].set_color('k')

    ax.set_xticks([x + 1 for x in range(len(top_keys))], top_keys, rotation=45, ha='right')
    return ax


def barchart(
        data:dict | pd.DataFrame,
        ax:plt.Axes | None = None,
        colors:str | list[str] | None = None,
        autonormalize:bool=False,
        error:str | None = "sd",
        draw_points: bool = True,
        draw_error:bool = True,
        point_color:str="k",
        point_size:float=20,
        point_alpha:float=0.8,
        value:str | None = None,
        primary_group:str | None = None,
        secondary_group:str | None = None):
    """Plots grouped bar charts with error bars and individual data points.

    Accepts the same data layouts as :func:`boxplot`: either ``{group: values}``
    or ``{group: {series: values}}``. DataFrames can be supplied by naming the
    value, primary_group, and optional secondary_group columns.

    :param data: Data to plot
    :param ax: Target axes
    :param colors: Single color or list of colors for grouped series
    :param autonormalize: Whether to normalize nested groups by their largest value
    :param error: Error bars to draw: "sd", "sem", or None
    :param draw_points: Whether to draw the individual observations
    :param draw_error: Whether to draw error bars
    :param point_color: Color for individual data points
    :param point_size: Marker size for individual data points
    :param point_alpha: Alpha for individual data points
    :param value: DataFrame column containing the values to plot
    :param primary_group: DataFrame column used for the x-axis groups
    :param secondary_group: Optional DataFrame column used for grouped series
    :return ax: Returns the populated axes object
    """
    if not ax:
        fig, ax = plt.subplots()

    if isinstance(data, pd.DataFrame):
        if value is None or primary_group is None:
            raise ValueError("value and primary_group must be supplied when data is a DataFrame")
        data = unpack_dataframe(data, value, primary_group, secondary_group)

    top_keys, subkeys_present, subkeys = _plot_group_structure(data)

    def prep_values(values):
        return np.asarray(values, dtype=float)

    def get_error(values):
        values = values[~np.isnan(values)]
        if error is None or len(values) <= 1:
            return 0
        if error.lower() == "sem":
            return np.std(values, ddof=1) / np.sqrt(len(values))
        if error.lower() == "sd":
            return np.std(values, ddof=1)
        raise ValueError('error must be "sem", "sd", or None')

    def point_positions(center, width, count):
        if count <= 1:
            return np.asarray([center])
        point_width = min(width * 0.55, 0.18)
        return center + np.linspace(-point_width / 2, point_width / 2, count)

    def directional_yerr(bar_values, errors):
        """Extend errors away from zero according to each bar's direction."""
        if not draw_error:
            return None
        bar_values = np.asarray(bar_values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        lower = np.where(bar_values < 0, errors, 0.0)
        upper = np.where(bar_values < 0, 0.0, errors)
        return np.vstack([lower, upper])

    if subkeys_present:
        plot_data = _normalized_group_values(
            data, top_keys, subkeys, autonormalize
        )
        offsets, width = _group_offsets(len(subkeys))

        for idx, (subkey, offset) in enumerate(zip(subkeys, offsets)):
            available = [
                (top_idx, top_key)
                for top_idx, top_key in enumerate(top_keys)
                if subkey in plot_data[top_key]
                and np.isfinite(plot_data[top_key][subkey]).any()
            ]
            if not available:
                continue
            centers = [top_idx + 1 + offset for top_idx, _ in available]
            bar_values = [
                np.nanmean(plot_data[top_key][subkey])
                for _, top_key in available
            ]
            errors = [
                get_error(plot_data[top_key][subkey])
                for _, top_key in available
            ]
            color = _plot_color(colors, idx)

            ax.bar(
                centers,
                bar_values,
                yerr=directional_yerr(bar_values, errors),
                width=width,
                label=str(subkey),
                color=color,
                edgecolor='k',
                capsize=4,
                zorder=2)

            if draw_points:
                for center, (_, top_key) in zip(centers, available):
                    values = plot_data[top_key][subkey]
                    ax.scatter(
                        point_positions(center, width, len(values)),
                        values,
                        color=point_color,
                        s=point_size,
                        alpha=point_alpha,
                        edgecolors='k',
                        linewidths=0.4,
                        zorder=3)

        ax.legend()

    else:
        bar_data = [prep_values(data[top_key]) for top_key in top_keys]
        centers = [x + 1 for x in range(len(top_keys))]
        bar_values = [np.nanmean(values) for values in bar_data]
        errors = [get_error(values) for values in bar_data]
        width = 0.55
        color = _plot_color(colors)

        ax.bar(
            centers,
            bar_values,
            yerr=directional_yerr(bar_values, errors),
            width=width,
            color=color,
            edgecolor='k',
            capsize=4,
            zorder=2)

        if draw_points:
            for center, values in zip(centers, bar_data):
                ax.scatter(
                    point_positions(center, width, len(values)),
                    values,
                    color=point_color,
                    s=point_size,
                    alpha=point_alpha,
                    edgecolors='k',
                    linewidths=0.4,
                    zorder=3)

    ax.set_xticks([x + 1 for x in range(len(top_keys))], top_keys, rotation=45, ha='right')
    return ax
