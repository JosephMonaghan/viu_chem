
from __future__ import annotations
import warnings
import pyimzml.ImzMLParser as ImzMLParser
import numpy as np
from bisect import bisect_left
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import imzml_writer.utils as iw_utils
import pyimzml.ImzMLWriter as imzmlw


from dataclasses import dataclass
from typing import Iterable
from scipy.signal import find_peaks, savgol_filter

def _iter_paths(image_path: Path | list[Path]) -> Iterable[Path]:
    if isinstance(image_path, Path):
        yield image_path
    else:
        yield from image_path

def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = int(min(max(k, 0), n))
    if k == 0:
        return np.array([], dtype=int)
    return rng.choice(n, size=k, replace=False)


def _topk_peaks(mz: np.ndarray, inten: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if mz.size <= k:
        o = np.argsort(mz)
        return mz[o], inten[o]
    idx = np.argpartition(inten, -k)[-k:]
    o = np.argsort(mz[idx])
    return mz[idx][o], inten[idx][o]


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to ~sigma for normal dist."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def _ppm_window(mz: float, tol_ppm: float) -> float:
    return mz * tol_ppm * 1e-6

@dataclass
class _Cluster:
    mz: float
    n: int

def estimate_tol_ppm_from_jitter(
    image_paths: Path | list[Path],
    *,
    init_tol_ppm: float = 3.0,          # conservative starting tol for forming clusters
    samples_per_file: int = 150,
    topk_per_spectrum: int = 300,       # only strong peaks
    min_cluster_size: int = 20,         # only use well-supported clusters
    rng_seed: int = 0,
    k_sigma: float = 5.0,               # tol = k_sigma * robust_sigma_ppm
    min_tol_ppm: float = 0.75,
    max_tol_ppm: float = 5.0,
) -> float:
    """
    Estimate alignment tolerance from observed centroid peak jitter.
    Returns a single scalar tol_ppm.
    """
    rng = np.random.default_rng(rng_seed)

    centers: list[_Cluster] = []
    residuals_ppm: list[float] = []

    # We store member residuals only after clusters stabilize a bit:
    # simplest approach: two-pass within the sampled peaks.
    sampled_peaks: list[np.ndarray] = []  # store mz arrays of top peaks only (small!)

    # Pass 1: collect sampled top peaks
    for p in _iter_paths(image_paths):
        with warnings.catch_warnings(action="ignore"):
            imz = ImzMLParser.ImzMLParser(str(p), parse_lib="lxml")
        idxs = _sample_indices(len(imz.coordinates), samples_per_file, rng)

        for idx in idxs:
            mz, inten = imz.getspectrum(int(idx))
            mz = np.asarray(mz, float)
            inten = np.asarray(inten, float)
            if mz.size == 0:
                continue
            mz, _ = _topk_peaks(mz, inten, topk_per_spectrum)
            sampled_peaks.append(mz)

    if not sampled_peaks:
        raise ValueError("No sampled peaks available to estimate jitter.")

    # Pass 2: build clusters (centers only, count only)
    for mz_arr in sampled_peaks:
        for mz in mz_arr:
            if not centers:
                centers.append(_Cluster(mz=float(mz), n=1))
                continue

            c_mz = [c.mz for c in centers]
            j = bisect_left(c_mz, mz)
            candidates = []
            if j > 0:
                candidates.append(j - 1)
            if j < len(centers):
                candidates.append(j)

            best = None
            best_dmz = None
            for k in candidates:
                dmz = abs(mz - centers[k].mz)
                if dmz <= _ppm_window(centers[k].mz, init_tol_ppm):
                    if best_dmz is None or dmz < best_dmz:
                        best_dmz = dmz
                        best = k

            if best is None:
                centers.insert(j, _Cluster(mz=float(mz), n=1))
            else:
                # update center as running mean (count-weighted)
                c = centers[best]
                c.mz = (c.mz * c.n + float(mz)) / (c.n + 1)
                c.n += 1
                # keep list sorted locally
                k = best
                while k > 0 and centers[k].mz < centers[k-1].mz:
                    centers[k], centers[k-1] = centers[k-1], centers[k]
                    k -= 1
                while k+1 < len(centers) and centers[k].mz > centers[k+1].mz:
                    centers[k], centers[k+1] = centers[k+1], centers[k]
                    k += 1

    # Keep only “real” clusters
    good_centers = np.array([c.mz for c in centers if c.n >= min_cluster_size], dtype=float)
    if good_centers.size < 10:
        # fallback: not enough stable clusters
        return float(np.clip(init_tol_ppm, min_tol_ppm, max_tol_ppm))

    good_centers.sort()

    # Pass 3: compute residuals ppm of peaks to nearest good center (within init_tol)
    for mz_arr in sampled_peaks:
        j = np.searchsorted(good_centers, mz_arr)
        j = np.clip(j, 1, good_centers.size - 1)
        left = j - 1
        right = j
        choose_right = np.abs(good_centers[right] - mz_arr) < np.abs(good_centers[left] - mz_arr)
        jj = np.where(choose_right, right, left)

        dmz = mz_arr - good_centers[jj]
        ok = np.abs(dmz) <= _ppm_window(good_centers[jj], init_tol_ppm)
        ppm = dmz[ok] / good_centers[jj][ok] * 1e6
        residuals_ppm.extend(ppm.tolist())

    if len(residuals_ppm) < 100:
        return float(np.clip(init_tol_ppm, min_tol_ppm, max_tol_ppm))

    sigma_ppm = _mad(np.array(residuals_ppm, dtype=float))
    tol_ppm = k_sigma * sigma_ppm
    return float(np.clip(tol_ppm, min_tol_ppm, max_tol_ppm))


def _sample_spectrum_indices(n_spectra: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample k unique indices from [0, n_spectra), safe if k > n_spectra."""
    k = int(min(max(k, 0), n_spectra))
    if k == 0:
        return np.array([], dtype=int)
    return rng.choice(n_spectra, size=k, replace=False)


# Dataclass to keep campaign formatted
@dataclass(frozen=True)
class CampaignRef:
    domain_mz: np.ndarray        # shared domain axis
    mean_intensity: np.ndarray   # mean spectrum on that axis
    ref_mz: np.ndarray           # peak-picked reference m/z values
    ref_intensity: np.ndarray    # corresponding mean intensities
    step_ppm_est: float          # estimated domain step (ppm)
    n_pixels: int                # total number of spectra averaged
    domain_detect_freq: np.ndarray | None = None  # per-domain detection frequency (0..1)
    ref_detect_freq: np.ndarray | None = None     # detection frequency for ref peaks (0..1)

def compute_campaign_mean_and_ref(
    image_paths: Path | list[Path],
    *,
    tol_ppm: float | None = None,
    # domain estimation
    samples_per_file: int = 150,
    topk_peaks_per_spectrum: int | None = 1500,
    trim_quantiles: tuple[float, float] = (0.01, 0.99),
    rng_seed: int = 0,
    # domain step clamp: helps keep domain size sane and reproducible
    min_step_ppm: float = 0.5,
    max_step_ppm: float = 2.0,
    # peak picking on mean spectrum
    presmooth_width: int = 201,
    presmooth_order: int = 3,
    prominence: float | None = None,
    height_fraction: float = 1e-5,   # keep peaks >= fraction of max mean intensity
) -> CampaignRef:
    """
    Compute a shared mean spectrum and a shared reference peak list for a campaign of centroid Orbitrap imzML files.
    """

    # 1) estimate a domain step from a sample
    step_ppm_est, mz_min, mz_max = _estimate_step_ppm_from_sample(
        image_paths,
        samples_per_file=samples_per_file,
        topk_peaks_per_spectrum=topk_peaks_per_spectrum,
        trim_quantiles=trim_quantiles,
        rng_seed=rng_seed,
    )

    # clamp the step so the domain is neither absurdly dense nor too coarse
    step_ppm = float(np.clip(step_ppm_est, min_step_ppm, max_step_ppm))
    print(f"estimated step_ppm={step_ppm_est:.3f}, using step_ppm={step_ppm:.3f}; mz range ~[{mz_min:.2f}, {mz_max:.2f}]")

    # 2) build shared domain
    domain_mz = _build_domain_ppm(mz_min, mz_max, step_ppm)

    # m/z jitter
    if tol_ppm is None:
        tol_ppm = 1.8 * estimate_tol_ppm_from_jitter(
            image_paths,
            init_tol_ppm=20.0,
            samples_per_file=150,
            topk_per_spectrum=300,
            min_cluster_size=20,
            k_sigma=5.0,
            min_tol_ppm=0.75,
            max_tol_ppm=5.0,
        )
    print(f"Estimated ppm tolerance for {tol_ppm}")


    # 3) stream mean spectrum on that domain
    mean_int, domain_detect_freq, n_pixels = _accumulate_mean_on_domain(
        image_paths,
        domain_mz,
        tol_ppm=tol_ppm,
    )

    # 4) smooth + peak-pick the mean spectrum
    # SavGol requires odd window <= len
    if presmooth_width % 2 == 0:
        presmooth_width += 1
    presmooth_width = min(presmooth_width, (len(mean_int) // 2) * 2 - 1)  # largest odd < len
    presmooth_width = max(presmooth_width, 5)

    mean_smooth = savgol_filter(mean_int, presmooth_width, presmooth_order)

    peak_kwargs = {}
    if prominence is not None:
        peak_kwargs["prominence"] = prominence

    peak_idx, _ = find_peaks(mean_smooth, **peak_kwargs)

    ref_mz = domain_mz[peak_idx]
    ref_int = mean_smooth[peak_idx]
    ref_detect_freq = domain_detect_freq[peak_idx]

    # 5) threshold peaks by relative height
    if ref_int.size:
        thr = ref_int.max() * height_fraction
        keep = ref_int >= thr
        ref_mz = ref_mz[keep]
        ref_int = ref_int[keep]
        ref_detect_freq = ref_detect_freq[keep]
    # Recompute ref detection frequency and aggregate ref signal using nearby domain bins
    if ref_mz.size:
        ref_detect_freq = _windowed_ref_detect_freq(
            domain_mz, domain_detect_freq, ref_mz, tol_ppm
        )
        ref_mz, ref_int = _windowed_ref_signal(
            domain_mz, mean_int, ref_mz, tol_ppm
        )

    return CampaignRef(
        domain_mz=domain_mz,
        mean_intensity=mean_int,
        domain_detect_freq=domain_detect_freq,
        ref_mz=ref_mz,
        ref_intensity=ref_int,
        ref_detect_freq=ref_detect_freq,
        step_ppm_est=step_ppm_est,
        n_pixels=n_pixels,
    )


def _windowed_ref_detect_freq(
    domain_mz: np.ndarray,
    domain_detect_freq: np.ndarray,
    ref_mz: np.ndarray,
    detect_tol_ppm: float,
) -> np.ndarray:
    """Aggregate detection frequency in a ppm window around each ref m/z."""
    out = np.zeros_like(ref_mz, dtype=float)
    for i, m in enumerate(ref_mz):
        dmz = m * detect_tol_ppm * 1e-6
        left = np.searchsorted(domain_mz, m - dmz, side="left")
        right = np.searchsorted(domain_mz, m + dmz, side="right")
        if right <= left:
            out[i] = 0.0
        else:
            out[i] = float(np.minimum(1.0, np.sum(domain_detect_freq[left:right])))
    return out


def _windowed_ref_signal(
    domain_mz: np.ndarray,
    mean_intensity: np.ndarray,
    ref_mz: np.ndarray,
    detect_tol_ppm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate mean intensity in a ppm window and compute intensity-weighted mean m/z."""
    mz_out = np.zeros_like(ref_mz, dtype=float)
    int_out = np.zeros_like(ref_mz, dtype=float)
    for i, m in enumerate(ref_mz):
        dmz = m * detect_tol_ppm * 1e-6
        left = np.searchsorted(domain_mz, m - dmz, side="left")
        right = np.searchsorted(domain_mz, m + dmz, side="right")
        if right <= left:
            mz_out[i] = m
            int_out[i] = 0.0
            continue
        mz_slice = domain_mz[left:right]
        int_slice = mean_intensity[left:right]
        total = float(np.sum(int_slice))
        int_out[i] = total
        if total > 0:
            mz_out[i] = float(np.sum(mz_slice * int_slice) / total)
        else:
            mz_out[i] = float(np.mean(mz_slice))
    return mz_out, int_out


def _take_topk(mz: np.ndarray, inten: np.ndarray, topk: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Optionally downselect to topk intensities (faster for domain estimation)."""
    if topk is None or mz.size <= topk:
        return mz, inten
    idx = np.argpartition(inten, -topk)[-topk:]
    # sort by mz for diff() stability
    o = np.argsort(mz[idx])
    return mz[idx][o], inten[idx][o]


def _estimate_step_ppm_from_sample(
    image_paths: Path | list[Path],
    *,
    samples_per_file: int = 150,
    topk_peaks_per_spectrum: int | None = 1500,
    trim_quantiles: tuple[float, float] = (0.01, 0.99),
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """
    Estimate a representative spacing in ppm from centroid m/z arrays, plus global mz_min/mz_max.
    This is NOT the instrument mass accuracy; it's a domain step estimate driven by peak density.
    """
    rng = np.random.default_rng(rng_seed)

    all_dppm = []
    mz_min = np.inf
    mz_max = -np.inf

    for _, p in enumerate(_iter_paths(image_paths), start=1):
        with warnings.catch_warnings(action="ignore"):
            imz = ImzMLParser.ImzMLParser(str(p), parse_lib="lxml")

        n = len(imz.coordinates)
        idxs = _sample_spectrum_indices(n, samples_per_file, rng)

        for idx in idxs:
            mz, inten = imz.getspectrum(int(idx))
            mz = np.asarray(mz, dtype=float)
            inten = np.asarray(inten, dtype=float)
            if mz.size < 2:
                continue

            mz, inten = _take_topk(mz, inten, topk_peaks_per_spectrum)

            mz_min = min(mz_min, float(mz[0]))
            mz_max = max(mz_max, float(mz[-1]))

            dppm = np.diff(mz) / mz[1:] * 1e6
            if dppm.size:
                all_dppm.append(dppm)

    if not all_dppm:
        raise ValueError("Could not estimate domain: no usable sampled spectra.")

    dppm = np.concatenate(all_dppm)
    lo, hi = np.quantile(dppm, trim_quantiles)
    dppm = dppm[(dppm >= lo) & (dppm <= hi)]

    step_ppm = float(np.median(dppm))
    return step_ppm, float(mz_min), float(mz_max)

# Generates shared domain (step_ppm bins)
def _build_domain_ppm(mz_min: float, mz_max: float, step_ppm: float) -> np.ndarray:
    """Multiplicative ppm grid: m_{k+1} = m_k * (1 + step_ppm*1e-6)."""
    if mz_min <= 0 or mz_max <= mz_min or step_ppm <= 0:
        raise ValueError("Invalid mz_min/mz_max/step_ppm for domain construction.")
    r = 1.0 + step_ppm * 1e-6
    out = [mz_min]
    while out[-1] < mz_max:
        out.append(out[-1] * r)
        if len(out) > 10_000_000:
            raise RuntimeError("Domain got unreasonably large; increase step_ppm.")
    return np.asarray(out, dtype=float)


def _accumulate_mean_on_domain(
    image_paths: Path | list[Path],
    ref_mz: np.ndarray,
    *,
    tol_ppm: float = 3.0,
    detect_tol_ppm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Stream over all spectra in all files, project centroid peaks onto ref_mz bins,
    accumulate sum, then mean = sum / n_pixels.
    """
    if detect_tol_ppm is None:
        detect_tol_ppm = tol_ppm
    print(ref_mz.size)
    sum_int = np.zeros(ref_mz.size, dtype=np.float64)
    detect_count = np.zeros(ref_mz.size, dtype=np.int32)
    n_pixels = 0

    for p_i, p in enumerate(_iter_paths(image_paths), start=1):
        print(f"mean spectrum: starting file {p_i}")
        with warnings.catch_warnings(action="ignore"):
            imz = ImzMLParser.ImzMLParser(str(p), parse_lib="lxml")

        for sp_i in range(len(imz.coordinates)):
            mz, inten = imz.getspectrum(int(sp_i))
            mz = np.asarray(mz, dtype=float)
            inten = np.asarray(inten, dtype=float)
            if mz.size == 0:
                n_pixels += 1
                continue

            # nearest neighbor in ref axis
            j = np.searchsorted(ref_mz, mz)
            j = np.clip(j, 1, ref_mz.size - 1)

            left = j - 1
            right = j
            choose_right = (np.abs(ref_mz[right] - mz) < np.abs(ref_mz[left] - mz))
            jj = np.where(choose_right, right, left)

            # ppm tolerance relative to ref center
            ok_align = np.abs(mz - ref_mz[jj]) <= (ref_mz[jj] * tol_ppm * 1e-6)
            ok_detect = np.abs(mz - ref_mz[jj]) <= (ref_mz[jj] * detect_tol_ppm * 1e-6)

            # sum intensity into bins (if multiple peaks map to same bin, they add)
            if np.any(ok_align):
                np.add.at(sum_int, jj[ok_align], inten[ok_align])
            # detection count: once per spectrum per bin
            if np.any(ok_detect):
                uniq = np.unique(jj[ok_detect])
                np.add.at(detect_count, uniq, 1)

            n_pixels += 1

    mean_int = sum_int / max(n_pixels, 1)
    detect_freq = detect_count / max(n_pixels, 1)
    return mean_int, detect_freq, n_pixels


def rewrite_imzml_with_common_mz(
    image_paths: list[Path],
    common_mz: np.ndarray,
    target_dir: Path,
    *,
    tol_ppm: float = 5.0,
    suffix: str = "-common",
    overwrite: bool = True,
    annotate_imzML: bool = True
) -> list[Path]:
    """
    Rewrite imzML files so each spectrum shares a common m/z axis.
    Metadata is preserved by annotating from the original imzML.

    :param image_paths: list of source imzML paths
    :param common_mz: shared m/z axis to enforce
    :param target_dir: directory for rewritten files
    :param tol_ppm: ppm tolerance used to map peaks to common_mz
    :param suffix: suffix appended to output filename stem
    :param overwrite: allow overwriting existing outputs
    :param annotate_imzML: Whether or not to annotate resulting imzML files based on the source
    :return: list of output imzML paths
    """
    if not image_paths:
        return []

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    common_mz = np.asarray(common_mz, dtype=float)
    if common_mz.ndim != 1 or common_mz.size == 0:
        raise ValueError("common_mz must be a non-empty 1D array.")
    if not np.all(np.diff(common_mz) >= 0):
        common_mz = np.sort(common_mz)

    out_paths: list[Path] = []

    for p in image_paths:
        p = Path(p)
        out_path = target_dir / f"{p.stem}{suffix}.imzML"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {out_path}")

        with warnings.catch_warnings(action="ignore"):
            with ImzMLParser.ImzMLParser(str(p), parse_lib="lxml") as img:
                with imzmlw.ImzMLWriter(str(out_path), mode="continuous") as out_imzml:
                    for sp_i, (x, y, z) in enumerate(img.coordinates):
                        mz, inten = img.getspectrum(int(sp_i))
                        mz = np.asarray(mz, dtype=float)
                        inten = np.asarray(inten, dtype=float)

                        if mz.size == 0:
                            out_int = np.zeros_like(common_mz)
                            out_imzml.addSpectrum(common_mz, out_int, (x, y, z))
                            continue

                        j = np.searchsorted(common_mz, mz)
                        j = np.clip(j, 1, common_mz.size - 1)
                        left = j - 1
                        right = j
                        choose_right = np.abs(common_mz[right] - mz) < np.abs(common_mz[left] - mz)
                        jj = np.where(choose_right, right, left)

                        ok = np.abs(mz - common_mz[jj]) <= (common_mz[jj] * tol_ppm * 1e-6)

                        out_int = np.zeros_like(common_mz)
                        if np.any(ok):
                            np.add.at(out_int, jj[ok], inten[ok])

                        out_imzml.addSpectrum(common_mz, out_int, (x, y, z))

        if annotate_imzML:
            iw_utils.annotate_from_model_imzML(str(p), str(out_path))
        out_paths.append(out_path)

    return out_paths





def _pool_spectra(image_path:Path | list[Path]):
    full_mz, full_intensity = [], []
    if isinstance(image_path, Path):
        with warnings.catch_warnings(action="ignore"):
            image = ImzMLParser.ImzMLParser(image_path,parse_lib='lxml')
        
        for idx, _ in enumerate(image.coordinates):
            local_mz, local_intensity = image.getspectrum(idx)
            full_mz.extend(local_mz)
            full_intensity.extend(local_intensity)
    elif isinstance(image_path,list):
        for img_idx, img_loc in enumerate(image_path):
            print(f"starting img #{img_idx+1} / {len(image_path)}")
            with warnings.catch_warnings(action="ignore"):
                image = ImzMLParser.ImzMLParser(img_loc,parse_lib='lxml')
        
            for idx, _ in enumerate(image.coordinates):
                local_mz, local_intensity = image.getspectrum(idx)
                full_mz.extend(np.asarray(local_mz))
                full_intensity.extend(np.asarray(local_intensity))
    
    return np.concatenate(full_mz), np.concatenate(full_intensity)




def generate_ref_list(image_path:Path | list[Path],tol:float=5, percentage_cutoff:float=0.0005,presmooth_width:int=201, presmooth_order:int=3):

    full_mz, full_intensity = _pool_spectra(image_path)
    
    mz_sort = np.argsort(full_mz)
    mz_raw = np.array(full_mz)[mz_sort]
    int_raw = np.array(full_intensity)[mz_sort]
    int_smooth = savgol_filter(int_raw, presmooth_width, presmooth_order)

    peak_idx, _ = find_peaks(int_smooth)
    mz_peaks = mz_raw[peak_idx]
    int_peaks = int_smooth[peak_idx]

    delta_mz = np.diff(mz_peaks)
    delta_ppm = delta_mz / mz_peaks[1:] * 1e6

    new_cluster = np.concatenate(([True], delta_ppm >= 5))
    cluster_ids = np.cumsum(new_cluster) - 1

    order = np.lexsort((-int_peaks, cluster_ids))
    sorted_clusters = cluster_ids[order]

    _, first_idx = np.unique(sorted_clusters, return_index=True)
    keep = order[first_idx]

    keep.sort()
    mz_keep = mz_peaks[keep]
    int_keep = int_peaks[keep]

    intensity_threshold = np.max(int_keep) * percentage_cutoff
    intensity_high = int_keep >= intensity_threshold
    mz_final = mz_keep[intensity_high]
    intensity_final = int_keep[intensity_high]
    return mz_final, intensity_final
