import pyimzml.ImzMLParser as ImzMLParser
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import numpy as np
import os
import imzml_writer.utils as iw_utils
import cv2 as cv
import scipy.ndimage
import matplotlib.colors as mcolors


def convert_from_RAW(dir:str,mode:str="Centroid",x_speed:float=40.0,y_step:float=150.0,filetype:str="raw", stop_at_mzML:bool=False):
    """Converts a directory of RAW files to mzML and then imzML with processed metadata.
    
    :param dir: Path to the directory containing RAW files
    :param mode: Conversion mode to use when writing mzML files
    :param x_speed: X-axis scan speed used for imzML metadata processing
    :param y_step: Y-axis step size used for imzML metadata processing
    :param filetype: File extension for the raw data files
    :param stop_at_mzML: Whether to stop after mzML conversion"""
    iw_utils.RAW_to_mzML(dir,write_mode=mode)

    iw_utils.clean_raw_files(dir,filetype)
    if stop_at_mzML:
        return
    mzML_path = os.path.join(dir,"Output mzML Files")
    iw_utils.mzML_to_imzML_convert(PATH=mzML_path)

    iw_utils.imzML_metadata_process(
        model_files=mzML_path,
        x_speed=x_speed,
        y_step=y_step,
        path=dir
        )
    

def get_image_matrix(src:str, mz:list | float = 104.1070,tol: list | float = 10.0):
    """Retrieves the requested ion image as a numpy array
    
    :param src: File path to the imzML source
    :param mz: m/z or list of m/z to retrieve images for
    :param tol: Tolerance with which to retrieve the images
    :return img_raw: Ion image array or list of ion image arrays"""

    with warnings.catch_warnings(action="ignore"):
        with ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml') as img:
            if isinstance(mz,float):
                tolerance = mz * tol / 1e6
                img_raw = ImzMLParser.getionimage(img, mz, tolerance)
            elif isinstance(mz,list):
                img_raw = []
                for idx, spp in enumerate(mz):
                    if isinstance(tol,float) or isinstance(tol,int):
                        tolerance = spp * tol / 1e6
                    elif isinstance(tol,list):
                        tolerance = spp * tol[idx] / 1e6
                    img_raw.append(ImzMLParser.getionimage(img,spp,tolerance))
                
    return img_raw


def get_TIC_image(src:str):
    """Retrieves the total ion current image from an imzML file.
    
    :param src: File path to the imzML source
    :return tic_image: Total ion current image as a numpy array"""
    with warnings.catch_warnings(action='ignore'):
        with ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml') as img:
            tic_image = ImzMLParser.getionimage(img,500,9999)
    
    return tic_image

def get_weighted_median_image(src:str):
    """Retrieves a weighted median image from an imzML file.
    
    :param src: File path to the imzML source
    :return wmi: Weighted median image as a numpy array"""

    def wmi_reduce_func(seq):
        """Calculates the median of nonzero values in an intensity sequence.
        
        :param seq: Intensity sequence
        :return: Median of nonzero intensity values"""
        no_zeros = seq[seq!=0]
        return np.median(no_zeros)

    with warnings.catch_warnings(action='ignore'):
        with ImzMLParser.ImzMLParser(filename=src, parse_lib='lxml') as img:
            wmi = ImzMLParser.getionimage(img,500,9999,reduce_func=wmi_reduce_func)  
    
    return wmi


def get_scale(src:str):
    """Returns the dimensions of the image in µm
    :param src: Path to the imzML
    :return: Tuple of form (scale_x, scale_y)"""
    with warnings.catch_warnings(action="ignore"):
        img = ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml')
        metadata = img.metadata.pretty()
        scan_settings = metadata["scan_settings"]["scanSettings1"]
        for key in scan_settings.keys():
            if key == "max dimension x":
                scale_x = scan_settings[key]
            elif key == "max dimension y":
                scale_y = scan_settings[key]
        return scale_x, scale_y

def get_aspect_ratio(src:str):
    """Calculates the image aspect ratio from imzML pixel size metadata.
    
    :param src: File path to the imzML source
    :return: Ratio of y pixel size to x pixel size"""
    with warnings.catch_warnings(action="ignore"):
        img = ImzMLParser.ImzMLParser(filename=src,parse_lib='lxml')
        metadata = img.metadata.pretty()
        scan_settings = metadata["scan_settings"]["scanSettings1"]
        for key in scan_settings.keys():
            if key == "pixel size (x)" or key == "pixel size x":
                x_pix = scan_settings[key]
            elif key == "pixel size y":
                y_pix = scan_settings[key]
        
        return y_pix / x_pix


def draw_ion_image(data:np.array, cmap:str="viridis",mode:str = "draw", path:str = None, cut_offs:tuple=(5, 95),quality:int=100, asp:float=1,scale:float=1,NL_override=None, custom_size:tuple=None):
    """Draws or saves an ion image from a numpy array with percentile-based intensity cutoffs.
    
    :param data: Ion image data as a numpy array
    :param cmap: Matplotlib colormap used to draw the image
    :param mode: Whether to draw the image or save it to disk
    :param path: Output path used when mode is save
    :param cut_offs: Lower and upper percentile cutoffs for image intensity
    :param quality: DPI to use when saving the image
    :param asp: Aspect ratio to use when drawing the image
    :param scale: Figure scale multiplier
    :param NL_override: Optional override for the image maximum intensity
    :param custom_size: Optional figure size override"""
    mpl.rcParams['savefig.pad_inches'] = 0
    up_cut = np.percentile(data,max(cut_offs))
    down_cut = np.percentile(data,min(cut_offs))

    img_cutoff = np.where(data > up_cut,up_cut,data)
    img_cutoff = np.where(data < down_cut,0,data)

    fig = plt.figure()
    _plt = plt.subplot()
    _plt.axis('off')
    if NL_override == None:
        _plt.imshow(img_cutoff,aspect=asp,interpolation="none",cmap=cmap,vmax=up_cut,vmin=0)
    else:
        _plt.imshow(img_cutoff,aspect=asp,interpolation="none",cmap=cmap,vmax=NL_override,vmin=0)
    size = fig.get_size_inches()
    scaled_size = size * scale
    fig.set_size_inches(scaled_size)

    if custom_size:
        fig.set_size_inches(custom_size)


    if mode == "draw":
        plt.show()
    elif mode == "save":
        if path is None:
            raise Exception("No file name specified")
        else:
            fig.savefig(path, dpi=quality,pad_inches=0,bbox_inches='tight')
            plt.close(fig)
    
def unsharp_mask(image, kernel_size=(5, 5), sigmaX=1.0, sigmaY=1.0, amount=1.0, threshold=0):
    """Returns a sharpened image using an unsharp mask.
    
    :param image: Image array to sharpen
    :param kernel_size: Gaussian blur kernel size
    :param sigmaX: Gaussian blur sigma in the x direction
    :param sigmaY: Gaussian blur sigma in the y direction
    :param amount: Sharpening amount
    :param threshold: Low-contrast threshold below which original pixels are preserved
    :return: Sharpened image array"""
    blurred = cv.GaussianBlur(image, kernel_size, sigmaX=sigmaX, sigmaY=sigmaY)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, np.max(image) * np.ones(sharpened.shape))
    # sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def smooth_image(img_data,asp:float, factor:int=3,base_sigma:float=10,weight_factor:float=0.5):
    """Smooths and sharpens an ion image after zooming it by the specified factor.
    
    :param img_data: Ion image data as a numpy array
    :param asp: Aspect ratio used to scale smoothing in the y direction
    :param factor: Zoom factor applied before sharpening
    :param base_sigma: Base Gaussian blur sigma for the unsharp mask
    :param weight_factor: Amount of sharpening to apply
    :return sharpened_img: Smoothed and sharpened image data"""
    zoomed_img = scipy.ndimage.zoom(img_data,factor)
    sharpened_img = unsharp_mask(zoomed_img, sigmaX=base_sigma, sigmaY=base_sigma/asp, kernel_size=(9,9), amount=weight_factor)
    return sharpened_img


def find_matching_ROI(ROI_files:list,match_folder:str, ROI_folder:str):
    """Finds and loads an ROI mask matching a data folder name.
    
    :param ROI_files: List of ROI npz filenames
    :param match_folder: Folder name to match against ROI filenames
    :param ROI_folder: Path to the folder containing ROI npz files
    :return roi_mask: ROI mask loaded from the matching npz file"""
    matching_npz = None

    if f"{match_folder}.npz" in ROI_files:
        matching_npz = os.path.join(ROI_folder, f"{match_folder}.npz")
        all_data = np.load(matching_npz)
        roi_mask = all_data['roi_mask']
        return roi_mask

    for file in ROI_files:
        file_string = file.split(".npz")[0]
        if file_string in match_folder:
            matching_npz = os.path.join(ROI_folder, file)
            break
    
    if matching_npz is not None:
        all_data = np.load(matching_npz)
        roi_mask = all_data['roi_mask']
        return roi_mask
    else:
        print(f"No matching file found! folder name: {match_folder}")
        raise
        
            
def find_data_filt_string(path: str, search_pattern: str):
    """Finds the first file containing a search pattern while skipping known raw-data folders.
    
    :param path: Directory path to search recursively
    :param search_pattern: String pattern to match in candidate filenames
    :return: Path to the matching file, or None if no match is found"""
    bad_options = {"Initial RAW files", "Output mzML Files"}

    for root, dirs, files in os.walk(path):
        # Modify dirs *in-place* to prevent descending into bad folders
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in bad_options
        ]

        for file in files:
            if search_pattern in file:
                return os.path.join(root, file)

    return None




def drawGrid(
    images:list[np.array],
    dims:tuple[int,int]=None,
    cut_off:float=95,
    lower_cut_off:float=None,
    title:str=None,
    aspects:list[float]=None,
    names:list[str]=None,
    groups:list[str]=None,
    group_axis:str="row",
    secondary_groups:list[str]=None,
    group_order:list[str]=None,
    secondary_group_order:list[str]=None,
    image_dimensions:list[tuple[float,float]]=None,
    scale_bar:float=None,
    scale_units:str="µm",
    color_scale:str="global",
    ignore_zeros:bool=True,
    cmap:str | mcolors.Colormap="viridis",
):
    """Draws a grid of ion images with a shared intensity scale and colorbar.
    
    :param images: List of numpy image arrays
    :param dims: Dimensions to draw the ion images in (height, width)
    :param cut_off: Percentile cutoff to use for the global dataset
    :param lower_cut_off: Optional lower percentile cutoff; defaults to zero intensity
    :param title: Title to draw above the entire image
    :param aspects: Aspect values for each image
    :param names: List of names to draw above each image
    :param groups: Group label for each image
    :param group_axis: Axis on which to arrange groups, either row or column
    :param secondary_groups: Labels for the sample positions perpendicular to group_axis
    :param group_order: Optional preferred ordering for primary group labels
    :param secondary_group_order: Optional preferred ordering for secondary group labels
    :param image_dimensions: Physical (width, height) of each image
    :param scale_bar: Physical length of a global scale bar
    :param scale_units: Units displayed on the global scale bar
    :param color_scale: Shared color scaling scope: global, row, column, or image/none
    :param ignore_zeros: Whether to exclude zero-valued pixels from percentile scaling
    :param cmap: Matplotlib colormap name or colormap object
    :return fig: Populated matplotlib figure"""

    def format_cbar_value(value:float, overflow:bool=False):
        """Formats colorbar values with readable scientific notation when needed."""
        if value != 0 and (abs(value) >= 1e4 or abs(value) < 1e-3):
            exponent = int(np.floor(np.log10(abs(value))))
            mantissa = value / 10 ** exponent
            suffix = "+" if overflow else ""
            return rf"${mantissa:.2f} \times 10^{{{exponent}}}{suffix}$"
        return f"{value:.2g}{'+' if overflow else ''}"

    def add_cbar(
        cbar_ax:plt.Axes,
        lower_cbar_cutoff:float=None,
        upper_cbar_cutoff:float=95,
        actual_limits:tuple[float,float]=None,
        has_overflow:bool=False,
    ):
        """Adds an intensity or percentile colorbar to an ion image grid.
        
        :param cbar_ax: Reserved axes that receives the colorbar
        :param lower_cbar_cutoff: Lower percentile label for the colorbar
        :param upper_cbar_cutoff: Upper percentile label for the colorbar
        :param actual_limits: Shared intensity limits to label instead of percentiles
        :param has_overflow: Whether to show a cap for values above the displayed range"""
        bar_top = 0.92 if has_overflow else 1
        cbar_ax.set_facecolor(background_color)
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        cbar_ax.imshow(
            gradient,
            aspect="auto",
            origin="lower",
            extent=(0, 1, 0, bar_top),
            cmap=cmap_obj,
        )
        if has_overflow:
            cbar_ax.add_patch(mpl.patches.Polygon(
                [(0, bar_top), (0.5, 1), (1, bar_top)],
                facecolor=cmap_obj(1.0),
                edgecolor="none",
            ))
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)
        label_font_size = mpl.font_manager.FontProperties(
            size=mpl.rcParams["axes.labelsize"]
        ).get_size_in_points()
        cbar_ax.set_ylabel(
            'Intensity',
            color=foreground_color,
            rotation=270,
            va='center',
            labelpad=24,
            fontsize=label_font_size,
        )
        cbar_ax.yaxis.set_label_position('right')
        cbar_ax.yaxis.set_tick_params(
            color=foreground_color,
            pad=7,
            labelsize=max(label_font_size - 2, 1),
        )
        cbar_ax.yaxis.set_ticks_position('right')
        cbar_ax.set_xticks([])

        if actual_limits is None:
            labels = ["0%", "50%", "100%"]
        else:
            lower_limit, upper_limit = actual_limits
            midpoint = (lower_limit + upper_limit) / 2
            labels = [
                format_cbar_value(lower_limit),
                format_cbar_value(midpoint),
                format_cbar_value(upper_limit, overflow=has_overflow),
            ]
        if has_overflow and actual_limits is None:
            labels[-1] += "+"
        cbar_ax.set_yticks(
            [0, bar_top / 2, bar_top],
            labels=labels,
            color=foreground_color,
        )
        cbar_ax.set_ylim(0, 1)

    def add_scale_bar(scale_ax:plt.Axes, length:float, units:str):
        """Adds the artists used for a dynamically calibrated scale bar."""
        center_x = 0.5
        y = 0.65
        line = mpl.lines.Line2D(
            [center_x, center_x],
            [y, y],
            transform=scale_ax.transAxes,
            color=foreground_color,
            linewidth=3,
            clip_on=False,
        )
        scale_ax.add_line(line)
        scale_ax.text(center_x, y - 0.15, f"{length:g} {units}", transform=scale_ax.transAxes,
                      color=foreground_color, ha="center", va="top")
        return line

    def update_physical_layout(physical_axes:list[plt.Axes], scale_line:mpl.lines.Line2D=None):
        """Fits physical image coordinates to the current axes sizes."""
        changed = False
        canvas_center_x = max_width / 2
        canvas_center_y = max_height / 2
        canvas_ratio = max_width / max_height

        for image_ax in physical_axes:
            axes_ratio = image_ax.bbox.width / image_ax.bbox.height
            if axes_ratio > canvas_ratio:
                visible_width = max_height * axes_ratio
                visible_height = max_height
            else:
                visible_width = max_width
                visible_height = max_width / axes_ratio
            x_limits = (
                canvas_center_x - visible_width / 2,
                canvas_center_x + visible_width / 2,
            )
            y_limits = (
                canvas_center_y + visible_height / 2,
                canvas_center_y - visible_height / 2,
            )
            if not np.allclose(image_ax.get_xlim(), x_limits):
                image_ax.set_xlim(x_limits)
                changed = True
            if not np.allclose(image_ax.get_ylim(), y_limits):
                image_ax.set_ylim(y_limits)
                changed = True

        if scale_line is not None:
            reference_ax = physical_axes[0]
            start_px = reference_ax.transData.transform((0, 0))[0]
            end_px = reference_ax.transData.transform((scale_bar, 0))[0]
            bar_width = abs(end_px - start_px) / scale_ax.bbox.width
            x_data = [0.5 - bar_width / 2, 0.5 + bar_width / 2]
            if not np.allclose(scale_line.get_xdata(), x_data):
                scale_line.set_xdata(x_data)
                changed = True
        return changed

    images = list(images)
    aspects = list(aspects) if aspects is not None else None
    names = list(names) if names is not None else None
    groups = list(groups) if groups is not None else None
    secondary_groups = list(secondary_groups) if secondary_groups is not None else None
    group_order = list(group_order) if group_order is not None else None
    secondary_group_order = list(secondary_group_order) if secondary_group_order is not None else None
    image_dimensions = list(image_dimensions) if image_dimensions is not None else None
    cmap_obj = mpl.colormaps.get_cmap(cmap)
    background_color = cmap_obj(0.0)
    background_rgb = np.asarray(background_color[:3])
    linear_rgb = np.where(
        background_rgb <= 0.04045,
        background_rgb / 12.92,
        ((background_rgb + 0.055) / 1.055) ** 2.4,
    )
    background_luminance = np.dot(linear_rgb, [0.2126, 0.7152, 0.0722])
    foreground_color = "black" if background_luminance > 0.45 else "white"

    if len(images) == 0:
        raise ValueError("images must contain at least one image")

    image_count = len(images)
    for values, label in (
        (aspects, "aspects"),
        (names, "names"),
        (groups, "groups"),
        (secondary_groups, "secondary_groups"),
        (image_dimensions, "image_dimensions"),
    ):
        if values is not None and len(values) != image_count:
            raise ValueError(f"{label} must contain one value per image")

    if not 0 <= cut_off <= 100:
        raise ValueError("cut_off must be between 0 and 100")
    if lower_cut_off is not None and not 0 <= lower_cut_off < cut_off:
        raise ValueError("lower_cut_off must be between 0 and cut_off")
    if group_axis not in {"row", "column"}:
        raise ValueError("group_axis must be either 'row' or 'column'")
    if color_scale == "none":
        color_scale = "image"
    if color_scale not in {"global", "row", "column", "image"}:
        raise ValueError("color_scale must be one of: global, row, column, image, none")
    if scale_bar is not None and image_dimensions is None:
        raise ValueError("image_dimensions must be supplied when using scale_bar")
    if scale_bar is not None and scale_bar <= 0:
        raise ValueError("scale_bar must be greater than zero")
    if group_order is not None and groups is None:
        raise ValueError("groups must be supplied when using group_order")
    if secondary_group_order is not None and secondary_groups is None:
        raise ValueError("secondary_groups must be supplied when using secondary_group_order")

    def ordered_labels(values:list[str], preferred_order:list[str]=None):
        """Returns unique labels with an optional preferred prefix ordering."""
        encountered = list(dict.fromkeys(values))
        if preferred_order is None:
            return encountered
        preferred = list(dict.fromkeys(preferred_order))
        unknown = [label for label in preferred if label not in encountered]
        if unknown:
            raise ValueError(f"group order contains unknown labels: {unknown}")
        return preferred + [label for label in encountered if label not in preferred]

    def wrap_row_label(label:str, max_line_length:int=13):
        """Wraps long row labels into two balanced lines."""
        label = str(label)
        if "\n" in label or len(label) <= max_line_length:
            return label
        words = label.split()
        if len(words) < 2:
            return label
        candidates = [
            (" ".join(words[:split_idx]), " ".join(words[split_idx:]))
            for split_idx in range(1, len(words))
        ]
        first, second = min(
            candidates,
            key=lambda lines: (max(map(len, lines)), abs(len(lines[0]) - len(lines[1]))),
        )
        return f"{first}\n{second}"

    placements = []
    group_labels = []
    if groups is not None:
        group_labels = ordered_labels(groups, group_order)
        if secondary_groups is not None:
            secondary_label_order = ordered_labels(secondary_groups, secondary_group_order)
            minimum_dims = (
                (len(group_labels), len(secondary_label_order))
                if group_axis == "row"
                else (len(secondary_label_order), len(group_labels))
            )
        else:
            grouped_indices = [[idx for idx, group in enumerate(groups) if group == label] for label in group_labels]
            largest_group = max(len(indices) for indices in grouped_indices)
            minimum_dims = (
                (len(group_labels), largest_group)
                if group_axis == "row"
                else (largest_group, len(group_labels))
            )
        if dims is None:
            dims = minimum_dims
        elif dims[0] < minimum_dims[0] or dims[1] < minimum_dims[1]:
            raise ValueError(f"dims must be at least {minimum_dims} for the requested grouping")

        if secondary_groups is not None:
            occupied_positions = set()
            for image_idx, (group, secondary_group) in enumerate(zip(groups, secondary_groups)):
                group_idx = group_labels.index(group)
                secondary_idx = secondary_label_order.index(secondary_group)
                row, column = (
                    (group_idx, secondary_idx)
                    if group_axis == "row"
                    else (secondary_idx, group_idx)
                )
                if (row, column) in occupied_positions:
                    raise ValueError("each groups and secondary_groups pair must be unique")
                placements.append((image_idx, row, column))
                occupied_positions.add((row, column))
        else:
            for group_idx, indices in enumerate(grouped_indices):
                for sample_idx, image_idx in enumerate(indices):
                    row, column = (
                        (group_idx, sample_idx)
                        if group_axis == "row"
                        else (sample_idx, group_idx)
                    )
                    placements.append((image_idx, row, column))
    else:
        if dims is None:
            dims = (4, int(np.ceil(image_count / 4)))
        if dims[0] * dims[1] < image_count:
            raise ValueError("dims does not contain enough cells for all images")
        placements = [
            (image_idx, image_idx // dims[1], image_idx % dims[1])
            for image_idx in range(image_count)
        ]

    secondary_labels = {}
    if secondary_groups is not None:
        secondary_axis = 1 if group_axis == "row" else 0
        for image_idx, row, column in placements:
            position = (row, column)[secondary_axis]
            label = secondary_groups[image_idx]
            secondary_labels[position] = label

    if image_dimensions is not None:
        if any(width <= 0 or height <= 0 for width, height in image_dimensions):
            raise ValueError("image_dimensions values must be greater than zero")
        max_width = max(width for width, _ in image_dimensions)
        max_height = max(height for _, height in image_dimensions)
        if scale_bar is not None and scale_bar > max_width:
            raise ValueError("scale_bar cannot be wider than the largest image")

    if aspects is None:
        aspects = [1 for _ in images]

    fig = plt.figure()
    has_column_headers = (
        (groups is not None and group_axis == "column")
        or (secondary_groups is not None and group_axis == "row")
    )
    has_row_labels = (
        (groups is not None and group_axis == "row")
        or (secondary_groups is not None and group_axis == "column")
    )
    header_rows = 1 if has_column_headers else 0
    title_rows = 1 if names is not None else 0
    row_block = title_rows + 5
    image_grid_rows = dims[0] * row_block
    grid_rows = image_grid_rows + header_rows
    data_start_column = 1 if has_row_labels else 0
    legend_start_column = data_start_column + dims[1]
    column_header_ratio = 0.8
    sample_title_ratio = 0.65
    height_ratios = [column_header_ratio] if has_column_headers else []
    for _ in range(dims[0]):
        if names is not None:
            height_ratios.append(sample_title_ratio)
        height_ratios.extend([1] * 5)
    outer_left = 0.03
    outer_right = 0.94
    grid = fig.add_gridspec(
        grid_rows,
        data_start_column + dims[1] + 3,
        width_ratios=([0.5] if has_row_labels else []) + [1] * dims[1] + [0.2, 0.18, 0.45],
        height_ratios=height_ratios,
        left=outer_left,
        right=outer_right,
    )
    ax = np.array([
        [
            fig.add_subplot(
                grid[
                    header_rows + row * row_block + title_rows:
                    header_rows + (row + 1) * row_block,
                    data_start_column + column,
                ]
            )
            for column in range(dims[1])
        ]
        for row in range(dims[0])
    ])

    title_axes = {}
    if names is not None:
        image_idx_by_position = {(row, column): image_idx for image_idx, row, column in placements}
        for row in range(dims[0]):
            for column in range(dims[1]):
                image_idx = image_idx_by_position.get((row, column))
                if image_idx is None:
                    continue
                title_ax = fig.add_subplot(
                    grid[header_rows + row * row_block, data_start_column + column]
                )
                title_ax.text(
                    0.5, 0.5, str(names[image_idx]), transform=title_ax.transAxes,
                    color=foreground_color, ha="center", va="center",
                )
                title_ax.set_axis_off()
                title_axes[(row, column)] = title_ax

    header_axes = {}
    if has_column_headers:
        column_labels = group_labels if group_axis == "column" else secondary_labels
        column_label_items = (
            enumerate(column_labels)
            if isinstance(column_labels, list)
            else sorted(column_labels.items())
        )
        for position, column_label in column_label_items:
            header_ax = fig.add_subplot(grid[0, data_start_column + position])
            header_ax.text(
                0.5, 0.8, str(column_label), transform=header_ax.transAxes,
                color=foreground_color, weight="bold", ha="center", va="center",
            )
            header_ax.set_axis_off()
            header_axes[position] = header_ax

    row_label_axes = {}
    if has_row_labels:
        row_labels = group_labels if group_axis == "row" else secondary_labels
        row_label_items = (
            enumerate(row_labels)
            if isinstance(row_labels, list)
            else sorted(row_labels.items())
        )
        for position, row_label in row_label_items:
            label_ax = fig.add_subplot(
                grid[
                    header_rows + position * row_block:
                    header_rows + (position + 1) * row_block,
                    0,
                ]
            )
            wrapped_label = wrap_row_label(row_label)
            label_ax.text(
                0.5, 0.5, wrapped_label, transform=label_ax.transAxes,
                color=foreground_color, weight="bold", rotation=90, ha="center", va="center",
            )
            label_ax.set_axis_off()
            row_label_axes[position] = label_ax

    cbar_start = header_rows + int(image_grid_rows * 0.1)
    cbar_end = max(cbar_start + 1, header_rows + int(image_grid_rows * 0.65))
    scale_start = max(cbar_end, header_rows + int(image_grid_rows * 0.75))
    cbar_ax = fig.add_subplot(grid[cbar_start:cbar_end, legend_start_column + 1])
    scale_ax = fig.add_subplot(grid[scale_start:, legend_start_column:])
    scale_ax.set_axis_off()
    panel_width = 2
    panel_height = panel_width
    if image_dimensions is not None:
        panel_height = panel_width * max_height / max_width
    legend_width = 1.8
    row_label_width = 0.8 if has_row_labels else 0
    title_height = 0.5 if title else 0
    image_row_height = panel_height
    sample_title_height = (
        dims[0] * image_row_height * sample_title_ratio / 5
        if names is not None else 0
    )
    column_label_height = (
        image_row_height * column_header_ratio / 5
        if has_column_headers else 0
    )
    default_horizontal_span = (
        mpl.rcParams["figure.subplot.right"] - mpl.rcParams["figure.subplot.left"]
    )
    horizontal_size_factor = default_horizontal_span / (outer_right - outer_left)
    fig.set_size_inches(
        (dims[1] * panel_width + legend_width + row_label_width) * horizontal_size_factor,
        dims[0] * panel_height + sample_title_height + title_height + column_label_height,
    )
    fig.set_facecolor(background_color)
    if title:
        fig.suptitle(title,color=foreground_color)
    
    def get_scope_limits(image_indices:list[int]):
        """Returns pooled lower and upper percentile limits for a scope."""
        scope_values = np.concatenate([np.asarray(images[idx]).ravel() for idx in image_indices])
        finite_values = scope_values[np.isfinite(scope_values)]
        nonzero_values = finite_values[finite_values != 0]
        percentile_values = nonzero_values if ignore_zeros and nonzero_values.size else finite_values
        if percentile_values.size == 0:
            return 0, 1
        lower_limit = 0 if lower_cut_off is None else np.percentile(percentile_values, lower_cut_off)
        upper_limit = np.percentile(percentile_values, cut_off)
        if upper_limit == 0:
            upper_limit = np.max(percentile_values)
        if lower_limit >= upper_limit:
            lower_limit = min(float(np.min(percentile_values)), 0.0)
        if lower_limit >= upper_limit:
            lower_limit = float(upper_limit) - 1e-9
        return lower_limit, upper_limit

    image_limits = [get_scope_limits([idx]) for idx in range(image_count)]
    global_limits = get_scope_limits(list(range(image_count)))
    global_has_overflow = any(
        np.any(np.asarray(image)[np.isfinite(image)] > global_limits[1])
        for image in images
    )
    row_limits = {
        row: get_scope_limits([idx for idx, image_row, _ in placements if image_row == row])
        for row in range(dims[0])
        if any(image_row == row for _, image_row, _ in placements)
    }
    column_limits = {
        column: get_scope_limits([idx for idx, _, image_column in placements if image_column == column])
        for column in range(dims[1])
        if any(image_column == column for _, _, image_column in placements)
    }
    
    used_axes = set()
    physical_axes = []
    for idx, row, column in placements:
        image_ax = ax[row, column]
        image = images[idx]
        lower_limit, upper_limit = {
            "global": global_limits,
            "row": row_limits[row],
            "column": column_limits[column],
            "image": image_limits[idx],
        }[color_scale]
        if image_dimensions is None:
            image_ax.imshow(
                image,
                aspect=aspects[idx],
                vmin=lower_limit,
                vmax=upper_limit,
                cmap=cmap_obj,
            )
        else:
            width, height = image_dimensions[idx]
            x_offset = (max_width - width) / 2
            y_offset = (max_height - height) / 2
            image_ax.imshow(
                image,
                extent=(x_offset, x_offset + width, y_offset + height, y_offset),
                aspect="auto",
                vmin=lower_limit,
                vmax=upper_limit,
                cmap=cmap_obj,
            )
            physical_axes.append(image_ax)
        image_ax.set_axis_off()
        used_axes.add((row, column))
    
    for row in range(dims[0]):
        for column in range(dims[1]):
            if (row, column) not in used_axes:
                ax[row, column].set_axis_off()
            ax[row, column].set_facecolor(background_color)

    add_cbar(
        cbar_ax,
        lower_cut_off,
        cut_off,
        actual_limits=global_limits if color_scale == "global" else None,
        has_overflow=global_has_overflow if color_scale == "global" else False,
    )
    scale_line = None
    if scale_bar is not None:
        scale_line = add_scale_bar(scale_ax, scale_bar, scale_units)

    if image_dimensions is not None:
        def on_draw(_event):
            if update_physical_layout(physical_axes, scale_line):
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("draw_event", on_draw)
        fig.canvas.draw()

    return fig


def grid_image(dir:str, dims:tuple=None,names:list=None,ext:str=".tif", title_string:str=None, savepath:str=None, cbar_cutoffs:tuple=(5, 95)):
    """Takes in a folder of images and makes a grid from them to display them all at once.
    
    :param dir: Path to the directory containing image files
    :param dims: Tuple of form height, width
    :param names: List of names used to select and label images
    :param ext: File extension for images to include
    :param title_string: Title string to draw into the grid
    :param savepath: Optional path to save the figure instead of displaying it
    :param cbar_cutoffs: Lower and upper percentile labels for the colorbar"""
    all_images = [image for image in os.listdir(dir) if image.endswith(ext)]
    all_images = [image for name in names for image in all_images if name in image]

    fig, axarr = plt.subplots(dims[0], dims[1])
    fig.set_size_inches(dims[1]*1.5,dims[0]*1.5)
    fig.set_facecolor("#440154")
    my_axes = axarr.ravel()

    for idx, image in enumerate(all_images):
        local_img = plt.imread(os.path.join(dir,image))
        my_axes[idx].imshow(local_img)
        my_axes[idx].set_title(names[idx], color='white')
        my_axes[idx].set_axis_off()
    
    ticker = 0
    while idx < len(my_axes)-1:
        ticker +=1
        idx += 1
        my_axes[idx].set_axis_off()
        my_axes[idx].set_facecolor("#440154")
        if ticker==1:
            my_axes[idx].text(0.5, 0.5, title_string, color='white', weight='bold', ha='center', va='center')
        if ticker==2:
            plt.tight_layout()
            norm = mcolors.Normalize(vmin=0, vmax=100)
            ax_pos = my_axes[idx].get_position()
            width = 0.2
            height = 0.7
            cbar_position = [ax_pos.x0 + 0.05, ax_pos.y0, ax_pos.width * width, ax_pos.height * height]
            cbar_ax = fig.add_axes(cbar_position)  # Add the colorbar axes at the defined position
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                cax=cbar_ax,
                orientation='vertical'
            )
            cbar.set_label('Intensity', color='white', rotation=270, va='center')  # Set your label if needed
            cbar.ax.yaxis.set_tick_params(color='white')  # Set color for colorbar ticks
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.set_ticks([0, 100])
            cbar.set_ticklabels([f"{cbar_cutoffs[0]}th", f"{cbar_cutoffs[1]}th"])
            cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), color='white')  # Set tick labels to white
    
    if not savepath:
        plt.show()
    else:
        plt.savefig(savepath)
        plt.close()





def bulk_image_export(dir:str,search_pattern:str, save_path:str, mz_list:list, target_list:list, include_codes:list=None, tolerance:float=10, uniform_scale:bool=False,smooth:bool=False,universal_cutoff:float=80, ROI_files:list=None, ROI_path:str=None):
    """Convenient API to convert a folder full of imzML files into ion images for a provided list of metabolites.
    
    :param dir: Path to the directory containing the imzML files (organized by experiment - each one in its own subfolder)
    :param search_pattern: Search string for the scan filter at the end of the imzML
    :param save_path: Path to a folder where images should be saved.
    :param mz_list: List of m/z to generate images for
    :param target_list: List of names matching the m/z list
    :param include_codes: List of strings that must be included to produce an output (when you want to subset a larger campaign)
    :param tolerance: Tolerance (in ppm) with which to extract the images
    :param uniform_scale: Optional argument for whether images should be scaled to self (False; default) or normalized to the most intense image (True)
    :param smooth: Should the resulting images be smoothed
    :param universal_cutoff: Fudge factor for the uniform_scaling - where should the intensity cutoff percentile be
    :param ROI_files: List of ROI filenames to match with folder names (based on sample codes for example) if image should only show a subset of pixels
    :param ROI_path: Path where the ROI files are located"""

    roi_mask = None
    all_folders = os.listdir(dir)
    data_folders = []
    for folder in all_folders:
        if not folder.startswith("."):
            if not include_codes:
                data_folders.append(folder)
            elif any(code.lower() in folder.lower() for code in include_codes):
                data_folders.append(folder)
    
    for target in target_list:
        path = os.path.join(save_path, "images", target)
        os.makedirs(path, exist_ok=True)
    
    scale = None
    NLs = [0 for _ in range(len(mz_list))]

    data_list = []
    asp_list = []
    TIC_list = []
    scale_list = []
    roi_list = []
    real_dims = [] #Actual image dimensions to draw in inches

    #Check all the scales etc. - hold the data in memory so you don't have to retrieve fresh
    for file_idx, folder in enumerate(data_folders):
        print(f"Starting file {file_idx+1} / {len(data_folders)} - {folder}")
        image = find_data_filt_string(os.path.join(dir,folder),search_pattern=search_pattern)
        aspect_ratio = get_aspect_ratio(image)
        data = get_image_matrix(image, mz_list,tol=tolerance)
        TIC_image = get_TIC_image(image)

        if ROI_path is not None and ROI_files is not None:
            roi_mask = find_matching_ROI(ROI_files, folder, ROI_path)

        ##TODO Fix y-scaling too so they actually come out to scale!
        x_scale, y_scale = get_scale(image)
        if scale == None:
            scale = 1
            full_scale_x = 1
            full_scale_y = 1
            norm_factor = x_scale
        else:
            scale = x_scale / norm_factor
        
        for idx, img in enumerate(data):
            normalized = np.divide(img, TIC_image, out=np.zeros_like(img), where=TIC_image!=0)
            if roi_mask is not None:
                normalized = normalized * roi_mask
            
            top_cutoff = np.percentile(normalized, universal_cutoff)
            if top_cutoff > NLs[idx]:
                NLs[idx] = top_cutoff

        data_list.append(data)
        asp_list.append(aspect_ratio)
        TIC_list.append(TIC_image)
        scale_list.append(scale)
        roi_list.append(roi_mask)
        real_dims.append((x_scale/2540, y_scale/2540))

    print("Saving images...")
    for file_idx, folder in enumerate(data_folders):
        data = data_list[file_idx]
        aspect_ratio = asp_list[file_idx]
        TIC_image = TIC_list[file_idx]
        scale = scale_list[file_idx]
        roi_mask = roi_list[file_idx]

        for idx, img in enumerate(data):
            path = os.path.join(save_path,"images",target_list[idx], f"{folder}-{target_list[idx]}.tif")
            normalized = np.divide(img, TIC_image, out=np.zeros_like(img), where=TIC_image!=0)

            if roi_mask is not None:
                normalized = normalized * roi_mask

            if smooth:
                normalized = smooth_image(normalized, aspect_ratio, factor=10)
            
            
            if uniform_scale:
                draw_ion_image(normalized, 'viridis', mode='save', path=path, asp=aspect_ratio, cut_offs=(20, 95), scale=scale, NL_override=NLs[idx], custom_size=real_dims[file_idx])
            else:
                draw_ion_image(normalized, cmap='viridis', mode='save', path=path, asp=aspect_ratio, cut_offs=(20, 95), scale=scale, custom_size=real_dims[file_idx])






        





    
