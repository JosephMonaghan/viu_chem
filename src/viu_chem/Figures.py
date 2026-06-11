import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import ttest_ind, chi2
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

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
            ylabel:str = "p-value"):
    """Generates a volcano plot comparing two dataframes into the specified axes. If no axes provided it generates its own.
    
    :param data_numerator: Dataframe containing numerator group values
    :param data_denom: Dataframe containing denominator group values
    :param ax: Target axes
    :param color_denom: Color for the denominator side gradient
    :param color_numer: Color for the numerator side gradient
    :param marker_color: Color to draw the data points
    :param sig_cutoff: P-value cutoff for significance
    :param left_label: Label for the denominator side of the plot
    :param right_label: Label for the numerator side of the plot
    :param xlabel: String for the x axis label
    :param ylabel: String for the y axis label
    :return ax: Returns the populated axes object
    :return return_df: Dataframe containing fold changes and p-values"""
    
    # Check that dataframes are coherent and match
    if not data_numerator.index.equals(data_denom.index):
        raise ValueError("Dataframes do not match!")

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
        numer_mean = local_numer.mean()
        denom_mean = local_denom.mean()

        if (denom_mean != 0) and (numer_mean != 0) and ((numer_mean / denom_mean) > 0):
            fold_changes[idx] = np.log2(numer_mean / denom_mean)
            _, p_val = ttest_ind(local_numer, local_denom, equal_var=False)
            pvals[idx] = p_val
        else:
            invalid[idx] = True

    

    # Actual plotting
    valid = (~invalid) & np.isfinite(fold_changes) & np.isfinite(pvals) & (pvals >= 0)
    return_df = pd.DataFrame({
        "fold_change": fold_changes,
        "pval": pvals,
    }, index=index)
    return_df = return_df[valid]

    if not valid.any():
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontweight='bold')
        if not xlabel:
            xlabel = f"log2({right_label} / {left_label})"
        ax.set_xlabel(xlabel, fontweight='bold')
        return ax, return_df

    plot_pvals = pvals.copy()
    positive_pvals = plot_pvals[valid & (plot_pvals > 0)]
    if len(positive_pvals):
        pval_floor = np.min(positive_pvals) * 0.1
    else:
        pval_floor = max(sig_cutoff * 0.1, np.nextafter(0, 1))
    plot_pvals[valid & (plot_pvals <= 0)] = pval_floor

    sig = valid & (np.abs(fold_changes) > 1) & (pvals < sig_cutoff)
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
    ax.axvline(-1,color=overlay_col,linestyle="--")
    ax.axvline(1, color=overlay_col,linestyle="--")


    xmin, xmax = np.min(fold_changes[valid])*1.2, np.max(fold_changes[valid])*1.2
    ymin, ymax = np.min(plot_pvals[valid]), np.max(plot_pvals[valid])

    if xmin == xmax:
        xmin, xmax = xmin - 1, xmax + 1
    if xmin > -1:
        xmin = -1.2
    if xmax < 1:
        xmax = 1.2
    if ymin == ymax:
        ymin = max(ymin * 0.5, np.nextafter(0, 1))
        ymax = ymax * 2

    # LEFT gradient (negative fold changes)
    _add_side_gradient(ax, xmin, 0, ymin, ymax, color_denom, direction="left", fade_y=sig_cutoff)

    # RIGHT gradient (positive fold changes)
    _add_side_gradient(ax, 0, xmax, ymin, ymax, color_numer, direction="right",fade_y=sig_cutoff)

    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(ylabel, fontweight='bold')
    if not xlabel:
        xlabel = f"log2({right_label} / {left_label})"
    ax.set_xlabel(xlabel, fontweight='bold')

    ylim = ax.get_ylim()
    ax.text(xmax*0.95,ylim[1]*1.5,right_label, ha='right',fontweight='bold')
    ax.text(xmin*0.95,ylim[1]*1.5,left_label,ha='left',fontweight='bold')

    return ax, return_df


def _volcano_sig_colors(fold_changes, marker_color, color_denom, color_numer):
    """Returns marker fill colors that deepen with significant fold-change magnitude."""
    colors = []
    base = np.array(mcolors.to_rgb(marker_color))
    denom = np.array(mcolors.to_rgb(color_denom))
    numer = np.array(mcolors.to_rgb(color_numer))

    denom_max = np.max(np.abs(fold_changes[fold_changes < 0])) if np.any(fold_changes < 0) else 1
    numer_max = np.max(fold_changes[fold_changes > 0]) if np.any(fold_changes > 0) else 1

    for change in fold_changes:
        if change < 0:
            amount = np.clip((abs(change) - 1) / max(denom_max - 1, 1), 0, 1)
            colors.append(base + (denom - base) * amount)
        else:
            amount = np.clip((change - 1) / max(numer_max - 1, 1), 0, 1)
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
    
    top_keys = list(data.keys())
    subkeys_present = isinstance(data[top_keys[0]],dict)

    if subkeys_present:
        subkeys = [key for key in data[top_keys[0]].keys()]

        if autonormalize:
            for key in top_keys:
                all_data = []
                for subkey in subkeys:
                    all_data.append(data[key][subkey])
                data[key]['_norm_limit'] = np.max(all_data)
                for subkey in subkeys:
                    data[key][subkey] = np.divide(data[key][subkey],[data[key]['_norm_limit']])

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        num_subkeys = len(subkeys)
        positions = np.linspace(0.8,1.2,num_subkeys)
        width = 0.4 * 3 / num_subkeys**2

            
        for idx, (key, pos) in enumerate(zip(subkeys,positions)):
            key_data = [data[top_key][key] for top_key in top_keys]
            positions = [pos+x for x in range(len(top_keys))]
            if not colors:
                color = default_colors[idx % len(default_colors)]
            else:
                color = colors[idx]

            ax.boxplot(key_data,
                       positions=positions,
                       widths=width,
                       label=key, 
                       patch_artist=True,
                       boxprops={'facecolor':color, 'edgecolor':'k'},
                       medianprops={'color':'k'})
        
        ax.legend()
    
    else:
        if not colors:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        
        bp_data = [data[top_key] for top_key in top_keys]
        ax.boxplot(bp_data,
                   patch_artist=True,
                   boxprops={'facecolor':color,'edgecolor':'k'},
                   medianprops={'color':'k'})
    
    ax.set_xticks([x+1 for x in range(len(top_keys))],top_keys, rotation=45,ha='right')


def barchart(
        data:dict | pd.DataFrame,
        ax:plt.Axes | None = None,
        colors:str | list[str] | None = None,
        autonormalize:bool=False,
        error:str | None = "sd",
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

    top_keys = list(data.keys())
    subkeys_present = isinstance(data[top_keys[0]], dict)

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

    def get_color(idx=0):
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if colors is None:
            return default_colors[idx % len(default_colors)]
        if isinstance(colors, str):
            return colors
        return colors[idx]

    if subkeys_present:
        subkeys = [key for key in data[top_keys[0]].keys()]
        plot_data = {}

        for top_key in top_keys:
            plot_data[top_key] = {}
            norm_limit = 1
            if autonormalize:
                all_values = [
                    prep_values(data[top_key][subkey])
                    for subkey in subkeys
                ]
                norm_limit = np.nanmax(np.concatenate(all_values))

            for subkey in subkeys:
                values = prep_values(data[top_key][subkey])
                if autonormalize and norm_limit != 0:
                    values = values / norm_limit
                plot_data[top_key][subkey] = values

        num_subkeys = len(subkeys)
        group_span = min(0.8, 0.4 + 0.1 * max(num_subkeys - 2, 0))
        positions = np.linspace(1 - group_span / 2.5, 1 + group_span / 2.5, num_subkeys)
        if num_subkeys > 1:
            width = (positions[1] - positions[0]) * 1
        else:
            width = 0.55

        for idx, (subkey, pos) in enumerate(zip(subkeys, positions)):
            centers = [pos + x for x in range(len(top_keys))]
            bar_values = [
                np.nanmean(plot_data[top_key][subkey])
                for top_key in top_keys
            ]
            errors = [
                get_error(plot_data[top_key][subkey])
                for top_key in top_keys
            ]
            color = get_color(idx)

            ax.bar(
                centers,
                bar_values,
                yerr=errors,
                width=width,
                label=subkey,
                color=color,
                edgecolor='k',
                capsize=4,
                zorder=2)

            for center, top_key in zip(centers, top_keys):
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
        color = get_color()

        ax.bar(
            centers,
            bar_values,
            yerr=errors,
            width=width,
            color=color,
            edgecolor='k',
            capsize=4,
            zorder=2)

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
        


    
