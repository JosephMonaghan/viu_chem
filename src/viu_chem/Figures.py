import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.colors as mcolors


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


def cal_curve(x:list[float], y:list[float], ax:plt.Axes=None,xlabel:str="Your x label here!", ylabel:str="Your y label here!", color:str="#8C4FA4") -> plt.Axes:
    """Generates a calibration curve for a given set of data x / y into the specified axes. If not axes provided it generates its own.
    
    :param x: List of x values
    :param y: List of y values
    :param ax: Target axes
    :param xlabel: String for the x axis label
    :param ylabel: String for the y axis label
    :param color: What color to draw the points
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
    ax.text(0.2, 0.95, equation, transform=plt.gca().transAxes, va='top',ha='center', color=color)
    
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

    
    # Check that dataframes are coherent and match
    if not data_numerator.index.equals(data_denom.index):
        raise ValueError("Dataframes do not match!")

    # If no axis specified, make one
    if not ax:
        fig, ax = plt.subplots()
    
    # Generate fold change and pval data
    index = data_numerator.index

    fold_changes = np.zeros_like(index)
    pvals = np.zeros_like(index)
    invalid = np.zeros_like(index)

    for idx, mz in enumerate(index):
        local_numer = data_numerator.loc[mz]
        local_denom = data_denom.loc[mz]

        if (local_denom.mean() != 0) and (local_numer.mean()!= 0):
            fold_changes[idx] = np.log2(local_numer.mean() / local_denom.mean())
            _, p_val = ttest_ind(local_numer, local_denom, equal_var=False)
            pvals[idx] = p_val
        else:
            invalid[idx] = 1

    

    # Actual plotting
    ax.semilogy(fold_changes, pvals, linestyle="none", marker="s",markeredgecolor='k',markerfacecolor=marker_color)

    occupied = []
    x_sep = 0.2
    y_sep = 0.2


    for change, pval, mz in zip(fold_changes, pvals, index):
        mz = float(mz)
        if abs(change) > 1 and pval < sig_cutoff:
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


    xmin, xmax = np.min(fold_changes)*1.2, np.max(fold_changes)*1.2
    ymin, ymax = np.min(pvals), np.max(pvals)

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

    return_df = pd.DataFrame({
        "fold_change": fold_changes,
        "pval": pvals,
    }, index=index)

    return_df = return_df[invalid==0]

    return ax, return_df


def _add_side_gradient(ax, x0, x1, y0, y1, color, direction="left", fade_y=0.05):
    """
    Draws a 2D gradient:
        - Horizontal fade (white <-> color)
        - Vertical fade (white at bottom, strong color at top)
    """
    N = 256

    # Horizontal component
    if direction == "left":
        horiz = np.linspace(1, 0, N)  # white → color
    else:
        horiz = np.linspace(0, 1, N)  # color → white

    # Vertical component (white at bottom → color at top)

    y_vals = np.linspace(y0, y1, N)
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

    



