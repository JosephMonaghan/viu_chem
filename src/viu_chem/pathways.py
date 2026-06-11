"""Plot metabolic pathways with structures and embedded matplotlib axes.

The functions in this module keep RDKit and networkx optional. If RDKit is
installed, metabolites with SMILES strings are rendered as ACS-style chemical
structure insets; otherwise the node label is drawn as text. If networkx is
installed, :meth:`Pathway.to_networkx` returns a graph for analysis or custom
layout work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from typing import Any, Literal

import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

Position = tuple[float, float]
NodeKind = Literal["metabolite", "axis"]


@dataclass(frozen=True)
class PathwayNode:
    """A pathway node.

    Parameters
    ----------
    id:
        Unique stable identifier used by reactions.
    label:
        Human-readable label drawn below the structure or inside the node.
    smiles:
        Optional SMILES string used for RDKit structure rendering.
    pos:
        Optional ``(x, y)`` position in pathway coordinates.
    kind:
        ``"metabolite"`` for a structure/text node or ``"axis"`` for a blank
        matplotlib inset axis reserved for arbitrary plots.
    data:
        Extra user metadata carried through to graph exports.
    """

    id: str
    label: str | None = None
    smiles: str | None = None
    pos: Position | None = None
    kind: NodeKind = "metabolite"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathwayEdge:
    """A directed connection between pathway nodes."""

    source: str
    target: str
    label: str | None = None
    reversible: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pathway:
    """A metabolic pathway definition."""

    name: str
    nodes: tuple[PathwayNode, ...]
    edges: tuple[PathwayEdge, ...]
    data: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, PathwayNode]:
        """Return nodes keyed by their identifiers."""

        return {node.id: node for node in self.nodes}

    def with_axes(self, axes: Iterable[PathwayNode | Mapping[str, Any]]) -> "Pathway":
        """Return a copy with additional axis placeholder nodes."""

        axis_nodes = tuple(_coerce_node(axis, default_kind="axis") for axis in axes)
        return replace(self, nodes=(*self.nodes, *axis_nodes))

    def to_networkx(self):
        """Return this pathway as a ``networkx.DiGraph``.

        Raises
        ------
        ImportError
            If networkx is not installed.
        """

        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError("Install networkx to export pathways as graphs.") from exc

        graph = nx.DiGraph(name=self.name, **self.data)
        for node in self.nodes:
            graph.add_node(
                node.id,
                label=node.label,
                smiles=node.smiles,
                pos=node.pos,
                kind=node.kind,
                **node.data,
            )
        for edge in self.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                label=edge.label,
                reversible=edge.reversible,
                **edge.data,
            )
        return graph


@dataclass
class PathwayPlot:
    """Return object for :func:`plot_pathway`."""

    figure: Figure
    axis: Axes
    node_axes: dict[str, Axes]


def available_presets() -> tuple[str, ...]:
    """Return names accepted by :func:`get_preset`."""

    return tuple(_PRESETS)


def get_preset(name: str) -> Pathway:
    """Return a built-in pathway definition.

    Presets are intentionally modest, editable starting points rather than
    exhaustive biochemical maps.
    """

    key = _normalize_preset_name(name)
    if key not in _PRESETS:
        options = ", ".join(available_presets())
        raise ValueError(f"Unknown pathway preset {name!r}. Available presets: {options}.")
    return _PRESETS[key]()


def make_pathway(
    name: str,
    nodes: Iterable[PathwayNode | Mapping[str, Any]],
    edges: Iterable[PathwayEdge | tuple[str, str] | Mapping[str, Any]],
    *,
    data: Mapping[str, Any] | None = None,
) -> Pathway:
    """Build a :class:`Pathway` from dataclasses, dictionaries, or edge tuples."""

    coerced_nodes = tuple(_coerce_node(node) for node in nodes)
    ids = [node.id for node in coerced_nodes]
    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
        raise ValueError(f"Pathway node ids must be unique; duplicates: {duplicates}.")

    coerced_edges = tuple(_coerce_edge(edge) for edge in edges)
    known_ids = set(ids)
    unknown = sorted(
        {
            endpoint
            for edge in coerced_edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in known_ids
        }
    )
    if unknown:
        raise ValueError(f"Pathway edges reference unknown node ids: {unknown}.")

    return Pathway(name=name, nodes=coerced_nodes, edges=coerced_edges, data=dict(data or {}))


def layout_axis_nodes(
    pathway: Pathway,
    axes: Iterable[PathwayNode | Mapping[str, Any]],
    *,
    anchors: Mapping[str, Sequence[str]],
    positions: Mapping[str, Position] | None = None,
    node_size: tuple[float, float] = (0.16, 0.12),
    axis_size: tuple[float, float] = (0.22, 0.16),
    same_size: bool = True,
    dynamic: bool = True,
    min_scale: float = 0.65,
    max_anchor_distance: float = 0.36,
    label_size: tuple[float, float] | None = None,
    label_y_shift: float = -0.105,
    edge_clearance: float = 0.025,
    edge_connectionstyle: str | Mapping[tuple[str, str], str] | None = None,
) -> tuple[PathwayNode, ...]:
    """Lay out arbitrary axis placeholder nodes near pathway anchors.

    This is useful for placing ion images, chromatograms, boxplots, or other
    matplotlib insets near metabolite structures without hand-tuning every
    panel. The returned nodes can be passed to :meth:`Pathway.with_axes`.

    Parameters
    ----------
    pathway:
        Pathway whose metabolite positions and edges define obstacles.
    axes:
        Axis placeholder nodes or dictionaries. Each axis id must have an entry
        in ``anchors``.
    anchors:
        Mapping from axis id to one or more pathway node ids that the axis
        should stay near.
    positions:
        Optional pathway position overrides.
    node_size, axis_size:
        Structure/text node and requested axis sizes in parent-axis fractions.
    same_size:
        If true, choose the largest single axis size that all panels can share.
        If false, each panel gets its own largest local size.
    dynamic:
        If false, return the supplied axis nodes unchanged.
    min_scale:
        Smallest fallback scale relative to ``axis_size``.
    max_anchor_distance:
        Maximum distance from an axis to the mean of its anchor nodes when
        searching for dynamic placements. Increase this for very sparse layouts.
    label_size, label_y_shift:
        Approximate label obstacle size and vertical shift for metabolite labels.
    edge_clearance:
        Minimum clearance from edge paths.
    edge_connectionstyle:
        Optional connection style matching :func:`plot_pathway`. ``arc3`` curve
        radii are sampled so axis placement avoids the same curved arrows that
        will be drawn.
    """

    axis_nodes = tuple(_coerce_node(axis, default_kind="axis") for axis in axes)
    if not dynamic:
        return axis_nodes

    pos = _resolve_positions(pathway, positions)
    node_ids = set(pos)
    missing = sorted(
        anchor
        for axis in axis_nodes
        for anchor in anchors.get(axis.id, ())
        if anchor not in node_ids
    )
    if missing:
        raise ValueError(f"Axis anchors reference unknown node ids: {missing}.")

    label_size = label_size or (node_size[0] * 1.25, 0.055)
    context = {
        "positions": pos,
        "node_size": node_size,
        "axis_size": axis_size,
        "anchors": anchors,
        "edge_paths": _layout_edge_paths(pathway.edges, pos, edge_connectionstyle),
        "metabolite_obstacles": _layout_metabolite_obstacles(
            pathway,
            pos,
            node_size=node_size,
            label_size=label_size,
            label_y_shift=label_y_shift,
        ),
        "max_anchor_distance": max_anchor_distance,
        "edge_clearance": edge_clearance,
    }

    if same_size:
        for scale in _layout_axis_scales(axis_size, min_scale=min_scale):
            size = (axis_size[0] * scale, axis_size[1] * scale)
            laid_out = _try_layout_axes(axis_nodes, size, context)
            if laid_out is not None:
                return laid_out
        fallback_size = (axis_size[0] * min_scale, axis_size[1] * min_scale)
        return _try_layout_axes(axis_nodes, fallback_size, context, require_clearance=False) or axis_nodes

    laid_out = []
    obstacles = list(context["metabolite_obstacles"])
    for axis in axis_nodes:
        for scale in _layout_axis_scales(axis_size, min_scale=min_scale):
            size = (axis_size[0] * scale, axis_size[1] * scale)
            placed = _place_axis_node(axis, size, obstacles, context)
            if placed is not None:
                laid_out.append(placed)
                obstacles.append(_layout_rect(placed.pos or (0.5, 0.5), size, pad=0.015))
                break
        else:
            laid_out.append(axis)
    return tuple(laid_out)


def plot_pathway(
    pathway: str | Pathway,
    *,
    ax: Axes | None = None,
    positions: Mapping[str, Position] | None = None,
    node_size: tuple[float, float] = (0.16, 0.12),
    axis_size: tuple[float, float] = (0.22, 0.16),
    structure: bool = True,
    show_labels: bool = True,
    show_edge_labels: bool = True,
    node_colors: Mapping[str, Any] | None = None,
    parent_aspect: str | float = "equal",
    label_font_size: float = 8,
    label_offsets: Mapping[str, tuple[float, float]] | None = None,
    edge_connectionstyle: str | Mapping[tuple[str, str], str] | None = None,
    edges_behind_nodes: bool = True,
    edge_direction_markers: bool = False,
    edge_direction_position: float | Mapping[tuple[str, str], float] = 0.62,
    structure_background: Any = "white",
    axis_factory: Callable[[Axes, PathwayNode], None] | Mapping[str, Callable[[Axes], None]] | None = None,
    edge_kwargs: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> PathwayPlot:
    """Draw a pathway.

    Parameters
    ----------
    pathway:
        A :class:`Pathway` object or built-in preset name.
    ax:
        Existing matplotlib axis to draw into. A new figure is created if omitted.
    positions:
        Optional position overrides keyed by node id.
    node_size, axis_size:
        Inset sizes as fractions of the parent axis.
    structure:
        If true, attempt RDKit structure rendering for nodes with SMILES.
    show_labels:
        Draw labels for metabolite nodes.
    show_edge_labels:
        Draw labels attached to reaction arrows.
    node_colors:
        Optional metabolite colors keyed by node id. RDKit structures, fallback
        labels, and metabolite labels use these colors.
    parent_aspect:
        Aspect ratio for the parent pathway axis. Use ``"auto"`` for denser
        rectangular layouts.
    label_font_size:
        Font size for metabolite labels drawn below structures.
    label_offsets:
        Optional label offsets keyed by metabolite id, in inset axis
        coordinates. Defaults to centered just below each structure.
    edge_connectionstyle:
        Matplotlib connection style for arrows. Pass a single style string or a
        mapping keyed by ``(source, target)`` for per-edge curves.
    edges_behind_nodes:
        Draw reaction arrows before node panels when true. Set false when using
        large shrink distances or curved arrows around large structures.
    edge_direction_markers:
        Add small arrowheads along connector paths. This is useful when large
        node panels hide endpoint arrowheads.
    edge_direction_position:
        Fractional position along each connector for direction markers. Pass a
        mapping keyed by ``(source, target)`` for per-edge positions.
    structure_background:
        Background color composited behind RDKit structure images.
    axis_factory:
        Callback(s) used to populate ``kind="axis"`` placeholder nodes.
    edge_kwargs:
        Matplotlib arrow style overrides.
    title:
        Optional plot title. Defaults to the pathway name.
    """

    pathway_obj = get_preset(pathway) if isinstance(pathway, str) else pathway
    fig, ax = _get_figure_axis(ax)
    pos = _resolve_positions(pathway_obj, positions)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(parent_aspect, adjustable="box")
    ax.axis("off")
    ax.set_title(pathway_obj.name if title is None else title)

    if edges_behind_nodes:
        _draw_edges(
            ax,
            pathway_obj.edges,
            pos,
            edge_kwargs=edge_kwargs,
            show_labels=show_edge_labels,
            edge_connectionstyle=edge_connectionstyle,
            edge_direction_markers=edge_direction_markers,
            edge_direction_position=edge_direction_position,
        )
    node_axes = _draw_nodes(
        ax,
        pathway_obj,
        pos,
        node_size=node_size,
        axis_size=axis_size,
        structure=structure,
        show_labels=show_labels,
        label_font_size=label_font_size,
        label_offsets=label_offsets,
        node_colors=node_colors,
        structure_background=structure_background,
        axis_factory=axis_factory,
    )
    if not edges_behind_nodes:
        _draw_edges(
            ax,
            pathway_obj.edges,
            pos,
            edge_kwargs=edge_kwargs,
            show_labels=show_edge_labels,
            edge_connectionstyle=edge_connectionstyle,
            edge_direction_markers=edge_direction_markers,
            edge_direction_position=edge_direction_position,
        )

    return PathwayPlot(figure=fig, axis=ax, node_axes=node_axes)


def _draw_nodes(
    ax: Axes,
    pathway: Pathway,
    positions: Mapping[str, Position],
    *,
    node_size: tuple[float, float],
    axis_size: tuple[float, float],
    structure: bool,
    show_labels: bool,
    label_font_size: float,
    label_offsets: Mapping[str, tuple[float, float]] | None,
    node_colors: Mapping[str, Any] | None,
    structure_background: Any,
    axis_factory: Callable[[Axes, PathwayNode], None] | Mapping[str, Callable[[Axes], None]] | None,
) -> dict[str, Axes]:
    node_axes: dict[str, Axes] = {}
    for node in pathway.nodes:
        x, y = positions[node.id]
        if node.kind == "axis":
            width, height = node.data.get("axis_size", axis_size)
        else:
            width, height = node_size
        inset = ax.inset_axes([x - width / 2, y - height / 2, width, height], transform=ax.transAxes)
        node_axes[node.id] = inset

        if node.kind == "axis":
            _populate_axis_node(inset, node, axis_factory)
            continue

        inset.axis("off")
        color = _node_color(node, node_colors)
        _draw_metabolite_node(
            inset,
            node,
            structure=structure,
            show_labels=show_labels,
            label_font_size=label_font_size,
            label_offset=(label_offsets or {}).get(node.id, (0.5, -0.04)),
            background=structure_background,
            color=color,
        )

    return node_axes


def _draw_metabolite_node(
    ax: Axes,
    node: PathwayNode,
    *,
    structure: bool,
    show_labels: bool,
    label_font_size: float,
    label_offset: tuple[float, float],
    background: Any,
    color: Any,
) -> None:
    if structure and node.smiles:
        image = _mol_image(
            node.smiles,
            color=color,
            background=background,
            projection=node.data.get("projection"),
        )
        if image is not None:
            ax.imshow(image, interpolation="lanczos")
        else:
            _draw_label_box(ax, node.label or node.id, color=color, background=background)
    else:
        _draw_label_box(ax, node.label or node.id, color=color, background=background)

    if show_labels and node.smiles:
        ax.text(
            label_offset[0],
            label_offset[1],
            node.label or node.id,
            ha="center",
            va="top",
            fontsize=label_font_size,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
        )


def _draw_label_box(ax: Axes, label: str, *, color: Any = "black", background: Any = "white") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.22),
            0.92,
            0.56,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=background,
            edgecolor=color,
            linewidth=1,
        )
    )
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=8, color=color, wrap=True)


def _layout_metabolite_obstacles(
    pathway: Pathway,
    positions: Mapping[str, Position],
    *,
    node_size: tuple[float, float],
    label_size: tuple[float, float],
    label_y_shift: float,
) -> list[tuple[float, float, float, float]]:
    obstacles = []
    for node in pathway.nodes:
        if node.kind != "metabolite" or node.id not in positions:
            continue
        obstacles.append(_layout_rect(positions[node.id], (node_size[0] * 1.08, node_size[1] * 1.05), pad=0.012))
        if node.label:
            obstacles.append(_layout_rect(positions[node.id], label_size, y_shift=label_y_shift, pad=0.006))
    return obstacles


def _layout_axis_scales(axis_size: tuple[float, float], *, min_scale: float) -> np.ndarray:
    width_limit = 0.84 / axis_size[0]
    height_limit = 0.82 / axis_size[1]
    return np.linspace(min(width_limit, height_limit), min_scale, 120)


def _layout_edge_paths(
    edges: Sequence[PathwayEdge],
    positions: Mapping[str, Position],
    edge_connectionstyle: str | Mapping[tuple[str, str], str] | None,
) -> list[list[Position]]:
    paths = []
    for edge in edges:
        start = positions[edge.source]
        end = positions[edge.target]
        paths.append(_layout_edge_path(start, end, _edge_connectionstyle(edge, edge_connectionstyle)))
    return paths


def _layout_edge_path(start: Position, end: Position, connectionstyle: str | None) -> list[Position]:
    rad = _layout_arc3_rad(connectionstyle)
    if rad is None or rad == 0:
        return [start, end]

    start_array = np.array(start, dtype=float)
    end_array = np.array(end, dtype=float)
    midpoint = (start_array + end_array) / 2
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length == 0:
        return [start, end]

    normal = np.array([-delta[1], delta[0]]) / length
    control = midpoint + normal * rad * length
    return [
        tuple((1 - t) ** 2 * start_array + 2 * (1 - t) * t * control + t**2 * end_array)
        for t in np.linspace(0, 1, 36)
    ]


def _layout_arc3_rad(connectionstyle: str | None) -> float | None:
    if not connectionstyle or not connectionstyle.startswith("arc3"):
        return None
    for part in connectionstyle.split(","):
        key, _, value = part.partition("=")
        if key.strip() == "rad" and value:
            try:
                return float(value)
            except ValueError:
                return None
    return 0.0


def _try_layout_axes(
    axis_nodes: Sequence[PathwayNode],
    axis_size: tuple[float, float],
    context: Mapping[str, Any],
    *,
    require_clearance: bool = True,
) -> tuple[PathwayNode, ...] | None:
    positioned: dict[str, PathwayNode] = {}
    obstacles = list(context["metabolite_obstacles"])
    remaining = {axis.id: axis for axis in axis_nodes}

    while remaining:
        options = []
        for axis in remaining.values():
            viable = _layout_axis_candidates(axis, axis_size, obstacles, context, require_clearance=require_clearance)
            if viable:
                options.append((len(viable), axis, viable))
        if not options:
            return None

        _, axis, viable = min(options, key=lambda option: option[0])
        pos = min(viable, key=lambda candidate: _layout_axis_score(axis, candidate, axis_size, obstacles, context))
        data = dict(axis.data)
        data["axis_size"] = axis_size
        placed = replace(axis, pos=pos, data=data)
        positioned[axis.id] = placed
        obstacles.append(_layout_rect(pos, axis_size, pad=0.015))
        del remaining[axis.id]

    return tuple(positioned[axis.id] for axis in axis_nodes)


def _place_axis_node(
    axis: PathwayNode,
    axis_size: tuple[float, float],
    obstacles: list[tuple[float, float, float, float]],
    context: Mapping[str, Any],
) -> PathwayNode | None:
    viable = _layout_axis_candidates(axis, axis_size, obstacles, context, require_clearance=True)
    if not viable:
        return None
    pos = min(viable, key=lambda candidate: _layout_axis_score(axis, candidate, axis_size, obstacles, context))
    data = dict(axis.data)
    data["axis_size"] = axis_size
    return replace(axis, pos=pos, data=data)


def _layout_axis_candidates(
    axis: PathwayNode,
    axis_size: tuple[float, float],
    obstacles: list[tuple[float, float, float, float]],
    context: Mapping[str, Any],
    *,
    require_clearance: bool,
) -> list[Position]:
    candidates = _layout_candidate_positions(axis, axis_size, context)
    return [
        candidate
        for candidate in candidates
            if _layout_axis_is_clear(axis, candidate, axis_size, obstacles, context, require_clearance=require_clearance)
    ]


def _layout_candidate_positions(
    axis: PathwayNode,
    axis_size: tuple[float, float],
    context: Mapping[str, Any],
) -> list[Position]:
    positions = context["positions"]
    anchors = context["anchors"].get(axis.id, ())
    anchor_positions = np.array([positions[anchor] for anchor in anchors], dtype=float)
    min_x, min_y = anchor_positions.min(axis=0)
    max_x, max_y = anchor_positions.max(axis=0)
    center_x, center_y = anchor_positions.mean(axis=0)
    node_size = context["node_size"]
    x_gap = axis_size[0] / 2 + node_size[0] / 2 + 0.035
    y_gap = axis_size[1] / 2 + node_size[1] / 2 + 0.055

    candidates = [
        (center_x, max_y + y_gap),
        (center_x, max_y + 1.45 * y_gap),
        (center_x, min_y - y_gap),
        (center_x, min_y - 1.35 * y_gap),
        (min_x - x_gap, center_y),
        (max_x + x_gap, center_y),
        (min_x - x_gap, max_y + 0.55 * y_gap),
        (max_x + x_gap, max_y + 0.55 * y_gap),
        (min_x - x_gap, min_y - 0.55 * y_gap),
        (max_x + x_gap, min_y - 0.55 * y_gap),
        (center_x - x_gap, center_y + 0.35 * y_gap),
        (center_x + x_gap, center_y + 0.35 * y_gap),
        (center_x - x_gap, center_y - 0.35 * y_gap),
        (center_x + x_gap, center_y - 0.35 * y_gap),
    ]
    candidates.extend(
        (float(x), float(y))
        for x in np.linspace(0.07, 0.93, 13)
        for y in np.linspace(0.1, 0.9, 9)
    )
    return list(dict.fromkeys((_layout_clamp(x, 0.045, 0.955), _layout_clamp(y, 0.075, 0.925)) for x, y in candidates))


def _layout_axis_is_clear(
    axis: PathwayNode,
    pos: Position,
    axis_size: tuple[float, float],
    obstacles: list[tuple[float, float, float, float]],
    context: Mapping[str, Any],
    *,
    require_clearance: bool,
) -> bool:
    if _layout_anchor_distance(pos, context, axis=axis) > context["max_anchor_distance"]:
        return False
    rect = _layout_rect(pos, axis_size, pad=0.012)
    if _layout_bounds_penalty(rect):
        return False
    if any(_layout_rect_overlap_area(rect, obstacle) for obstacle in obstacles):
        return False
    if require_clearance and any(_layout_path_rect_distance(path, rect) < context["edge_clearance"] for path in context["edge_paths"]):
        return False
    return True


def _layout_axis_score(
    axis: PathwayNode,
    pos: Position,
    axis_size: tuple[float, float],
    obstacles: list[tuple[float, float, float, float]],
    context: Mapping[str, Any],
) -> float:
    rect = _layout_rect(pos, axis_size, pad=0.01)
    anchor_distance = _layout_anchor_distance(pos, context, axis=axis)
    score = 12.0 * anchor_distance + _layout_bounds_penalty(rect)
    for obstacle in obstacles:
        overlap = _layout_rect_overlap_area(rect, obstacle)
        if overlap:
            score += 100000 * overlap + 400
    for path in context["edge_paths"]:
        distance = _layout_path_rect_distance(path, rect)
        if distance < context["edge_clearance"] + 0.015:
            score += (context["edge_clearance"] + 0.015 - distance) * 220
    return score


def _layout_anchor_distance(pos: Position, context: Mapping[str, Any], axis: PathwayNode | None = None) -> float:
    if axis is None:
        return 0.0
    positions = context["positions"]
    anchors = context["anchors"].get(axis.id, ())
    anchor_positions = np.array([positions[anchor] for anchor in anchors], dtype=float)
    return float(np.linalg.norm(np.array(pos) - anchor_positions.mean(axis=0)))


def _layout_rect(
    pos: Position,
    size: tuple[float, float],
    *,
    y_shift: float = 0.0,
    pad: float = 0.0,
) -> tuple[float, float, float, float]:
    x, y = pos
    width, height = size
    y += y_shift
    return (x - width / 2 - pad, x + width / 2 + pad, y - height / 2 - pad, y + height / 2 + pad)


def _layout_bounds_penalty(rect: tuple[float, float, float, float]) -> float:
    left, right, bottom, top = rect
    overflow = max(0.0, 0.02 - left) + max(0.0, right - 0.98)
    overflow += max(0.0, 0.035 - bottom) + max(0.0, top - 0.93)
    return overflow * 10000


def _layout_rect_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    bottom = max(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def _layout_path_rect_distance(path: Sequence[Position], rect: tuple[float, float, float, float]) -> float:
    return min(_layout_point_rect_distance(point, rect) for point in path)


def _layout_point_rect_distance(point: Position, rect: tuple[float, float, float, float]) -> float:
    x, y = point
    left, right, bottom, top = rect
    if left <= x <= right and bottom <= y <= top:
        return 0.0
    dx = max(left - x, 0.0, x - right)
    dy = max(bottom - y, 0.0, y - top)
    return float(np.hypot(dx, dy))


def _layout_clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _mol_image(
    smiles: str,
    *,
    color: Any = "black",
    background: Any = "white",
    projection: str | None = None,
):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Geometry import Point2D
    except ImportError:  # pragma: no cover - depends on optional dep
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if projection == "haworth":
        coord_map = _haworth_coord_map(mol, Point2D)
        if coord_map:
            rdDepictor.Compute2DCoords(mol, coordMap=coord_map, canonOrient=False)
        else:
            rdDepictor.Compute2DCoords(mol)
    else:
        rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 820)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.025
    if hasattr(options, "useBWAtomPalette"):
        options.useBWAtomPalette()
    if hasattr(rdMolDraw2D, "SetACS1996Mode"):
        rdMolDraw2D.SetACS1996Mode(options, True)
    rgb = mcolors.to_rgb(color)
    palette = {
        atomic_num: rgb
        for atomic_num in {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    }
    options.setAtomPalette(palette)
    options.setSymbolColour(rgb)
    options.setAnnotationColour(rgb)
    options.setBondNoteColour(rgb)
    options.bondLineWidth = 2.0
    options.fixedFontSize = -1
    options.baseFontSize = 0.28
    options.minFontSize = 10
    options.maxFontSize = 28
    options.multipleBondOffset = 0.16
    if hasattr(options, "singleColourWedgeBonds"):
        options.singleColourWedgeBonds = True

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    image = mpimg.imread(BytesIO(drawer.GetDrawingText()), format="png")
    return _opaque_background(_crop_structure_image(_recolor_structure_image(image, color)), background)


def _haworth_coord_map(mol, point_factory) -> dict[int, Any]:
    ring = _sugar_ring_atoms(mol)
    if ring is None:
        return {}

    oxygen_index = next(index for index in ring if mol.GetAtomWithIdx(index).GetSymbol() == "O")
    oxygen_position = ring.index(oxygen_index)
    ordered_ring = list(ring[oxygen_position:]) + list(ring[:oxygen_position])

    if len(ordered_ring) == 5:
        ring_coords = [
            (0.9, 0.18),
            (0.34, -0.58),
            (-0.55, -0.38),
            (-0.72, 0.5),
            (0.18, 0.86),
        ]
    else:
        ring_coords = [
            (1.0, 0.25),
            (0.55, -0.55),
            (-0.55, -0.55),
            (-1.0, 0.05),
            (-0.55, 0.75),
            (0.55, 0.75),
        ]
    coord_map = {
        atom_index: point_factory(x, y)
        for atom_index, (x, y) in zip(ordered_ring, ring_coords, strict=True)
    }

    ring_set = set(ordered_ring)
    center = np.mean(np.array(ring_coords), axis=0)
    for atom_index, (x, y) in zip(ordered_ring, ring_coords, strict=True):
        substituents = [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
            if neighbor.GetIdx() not in ring_set
        ]
        if not substituents:
            continue
        direction = np.array([x, y]) - center
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction = np.array([0.0, -1.0])
        else:
            direction = direction / norm
        perpendicular = np.array([-direction[1], direction[0]])
        for offset, substituent in enumerate(substituents):
            spread = (offset - (len(substituents) - 1) / 2) * 0.32
            point = np.array([x, y]) + 0.72 * direction + spread * perpendicular
            coord_map[substituent] = point_factory(float(point[0]), float(point[1]))

    return coord_map


def _sugar_ring_atoms(mol) -> tuple[int, ...] | None:
    for ring in mol.GetRingInfo().AtomRings():
        symbols = [mol.GetAtomWithIdx(index).GetSymbol() for index in ring]
        if len(ring) == 6 and symbols.count("O") == 1 and symbols.count("C") == 5:
            return ring
        if len(ring) == 5 and symbols.count("O") == 1 and symbols.count("C") == 4:
            return ring
    return None


def _recolor_structure_image(image, color: Any):
    colored = np.array(image, copy=True)
    rgb = np.array(mcolors.to_rgb(color))

    if colored.ndim != 3:
        return colored

    if colored.shape[2] == 4:
        ink = colored[:, :, 3] > 0.01
    else:
        ink = (colored[:, :, :3] < 0.95).any(axis=2)

    colored[ink, :3] = rgb
    return colored


def _crop_structure_image(image, padding: int = 8):
    if image.ndim != 3:
        return image

    if image.shape[2] == 4:
        mask = image[:, :, 3] > 0.01
    else:
        mask = (abs(image[:, :, :3] - 1.0) > 0.02).any(axis=2)

    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return image

    row_start = max(int(rows.min()) - padding, 0)
    row_stop = min(int(rows.max()) + padding + 1, image.shape[0])
    col_start = max(int(cols.min()) - padding, 0)
    col_stop = min(int(cols.max()) + padding + 1, image.shape[1])
    return image[row_start:row_stop, col_start:col_stop]


def _opaque_background(image, background: Any):
    if image.ndim != 3 or image.shape[2] != 4:
        return image

    alpha = image[:, :, 3:4]
    background_rgb = np.array(mcolors.to_rgb(background))
    rgb = image[:, :, :3] * alpha + background_rgb * (1 - alpha)
    return rgb


def _populate_axis_node(
    ax: Axes,
    node: PathwayNode,
    axis_factory: Callable[[Axes, PathwayNode], None] | Mapping[str, Callable[[Axes], None]] | None,
) -> None:
    if callable(axis_factory):
        axis_factory(ax, node)
        return
    if isinstance(axis_factory, Mapping) and node.id in axis_factory:
        axis_factory[node.id](ax)
        return

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)
    ax.text(0.5, 0.5, node.label or node.id, ha="center", va="center", fontsize=8, transform=ax.transAxes)


def _draw_edges(
    ax: Axes,
    edges: Sequence[PathwayEdge],
    positions: Mapping[str, Position],
    *,
    edge_kwargs: Mapping[str, Any] | None,
    show_labels: bool,
    edge_connectionstyle: str | Mapping[tuple[str, str], str] | None,
    edge_direction_markers: bool,
    edge_direction_position: float | Mapping[tuple[str, str], float],
) -> None:
    arrow_style = {
        "arrowstyle": "-|>",
        "mutation_scale": 12,
        "linewidth": 1.2,
        "color": "0.15",
        "shrinkA": 18,
        "shrinkB": 18,
    }
    arrow_style.update(edge_kwargs or {})

    for edge in edges:
        start = positions[edge.source]
        end = positions[edge.target]
        local_style = dict(arrow_style)
        connectionstyle = _edge_connectionstyle(edge, edge_connectionstyle)
        if connectionstyle is not None:
            local_style["connectionstyle"] = connectionstyle
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, **local_style))
        if edge_direction_markers:
            _draw_edge_direction_marker(
                ax,
                start,
                end,
                connectionstyle=connectionstyle,
                color=local_style.get("color", "0.15"),
                position=_edge_direction_position(edge, edge_direction_position),
                mutation_scale=local_style.get("mutation_scale", 12),
            )
        if edge.reversible:
            reverse_style = {**local_style, "alpha": local_style.get("alpha", 1) * 0.6}
            ax.add_patch(FancyArrowPatch(end, start, transform=ax.transAxes, **reverse_style))
        if show_labels and edge.label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                edge.label,
                ha="center",
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                transform=ax.transAxes,
            )


def _edge_connectionstyle(
    edge: PathwayEdge,
    edge_connectionstyle: str | Mapping[tuple[str, str], str] | None,
) -> str | None:
    if isinstance(edge_connectionstyle, str) or edge_connectionstyle is None:
        return edge_connectionstyle
    return edge_connectionstyle.get((edge.source, edge.target))


def _edge_direction_position(
    edge: PathwayEdge,
    edge_direction_position: float | Mapping[tuple[str, str], float],
) -> float:
    if isinstance(edge_direction_position, Mapping):
        return edge_direction_position.get((edge.source, edge.target), 0.62)
    return edge_direction_position


def _draw_edge_direction_marker(
    ax: Axes,
    start: Position,
    end: Position,
    *,
    connectionstyle: str | None,
    color: Any,
    position: float,
    mutation_scale: float,
) -> None:
    position = min(max(position, 0.05), 0.95)
    delta = 0.035
    p0 = _edge_point(ax, start, end, connectionstyle, max(position - delta, 0.0))
    p1 = _edge_point(ax, start, end, connectionstyle, min(position + delta, 1.0))
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=0.8,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _edge_point(ax: Axes, start: Position, end: Position, connectionstyle: str | None, t: float) -> Position:
    start_array = ax.transAxes.transform(start)
    end_array = ax.transAxes.transform(end)
    rad = _arc3_rad(connectionstyle)
    if rad == 0:
        point = start_array + t * (end_array - start_array)
        axes_point = ax.transAxes.inverted().transform(point)
        return float(axes_point[0]), float(axes_point[1])

    delta = end_array - start_array
    midpoint = (start_array + end_array) / 2
    control = midpoint + rad * np.array([-delta[1], delta[0]])
    point = (1 - t) ** 2 * start_array + 2 * (1 - t) * t * control + t**2 * end_array
    axes_point = ax.transAxes.inverted().transform(point)
    return float(axes_point[0]), float(axes_point[1])


def _arc3_rad(connectionstyle: str | None) -> float:
    if not connectionstyle or not connectionstyle.startswith("arc3"):
        return 0.0
    for part in connectionstyle.split(","):
        key, _, value = part.partition("=")
        if key.strip() == "rad":
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def _node_color(node: PathwayNode, node_colors: Mapping[str, Any] | None) -> Any:
    if node_colors and node.id in node_colors:
        return node_colors[node.id]
    return node.data.get("color", "black")


def _resolve_positions(pathway: Pathway, overrides: Mapping[str, Position] | None) -> dict[str, Position]:
    positions = {node.id: node.pos for node in pathway.nodes if node.pos is not None}
    if overrides:
        positions.update(overrides)

    missing = [node.id for node in pathway.nodes if node.id not in positions]
    if missing:
        positions.update(_fallback_layout(pathway, missing))
    return positions


def _fallback_layout(pathway: Pathway, missing: Sequence[str]) -> dict[str, Position]:
    try:
        graph = pathway.to_networkx()
        layout = __import__("networkx").spring_layout(graph, seed=2)
    except ImportError:
        layout = _circle_layout([node.id for node in pathway.nodes])

    normalized = _normalize_layout(layout)
    return {node_id: normalized[node_id] for node_id in missing}


def _circle_layout(node_ids: Sequence[str]) -> dict[str, Position]:
    import math

    radius = 0.38
    center = (0.5, 0.5)
    count = max(len(node_ids), 1)
    return {
        node_id: (
            center[0] + radius * math.cos(2 * math.pi * index / count),
            center[1] + radius * math.sin(2 * math.pi * index / count),
        )
        for index, node_id in enumerate(node_ids)
    }


def _normalize_layout(layout: Mapping[str, Sequence[float]]) -> dict[str, Position]:
    xs = [float(value[0]) for value in layout.values()]
    ys = [float(value[1]) for value in layout.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value: float, low: float, high: float) -> float:
        if high == low:
            return 0.5
        return 0.12 + 0.76 * (value - low) / (high - low)

    return {
        node_id: (scale(float(value[0]), min_x, max_x), scale(float(value[1]), min_y, max_y))
        for node_id, value in layout.items()
    }


def _get_figure_axis(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, new_ax = plt.subplots(figsize=(8, 6))
    return fig, new_ax


def _coerce_node(node: PathwayNode | Mapping[str, Any], default_kind: NodeKind = "metabolite") -> PathwayNode:
    if isinstance(node, PathwayNode):
        return node if node.kind else replace(node, kind=default_kind)
    values = dict(node)
    values.setdefault("kind", default_kind)
    return PathwayNode(**values)


def _coerce_edge(edge: PathwayEdge | tuple[str, str] | Mapping[str, Any]) -> PathwayEdge:
    if isinstance(edge, PathwayEdge):
        return edge
    if isinstance(edge, tuple):
        return PathwayEdge(source=edge[0], target=edge[1])
    return PathwayEdge(**dict(edge))


def _normalize_preset_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def _linear_pathway(name: str, rows: Sequence[tuple[str, str, str]]) -> Pathway:
    node_count = len(rows)
    nodes = []
    for index, (node_id, label, smiles) in enumerate(rows):
        x = 0.1 + 0.8 * index / max(node_count - 1, 1)
        data = {"projection": "haworth"} if node_id in {"glucose", "g6p"} else {}
        nodes.append(PathwayNode(node_id, label, smiles, (x, 0.5), data=data))
    edges = [PathwayEdge(rows[index][0], rows[index + 1][0]) for index in range(node_count - 1)]
    return make_pathway(name, nodes, edges)


def _glycolysis() -> Pathway:
    nodes = [
        PathwayNode("glucose", "Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O", (0.08, 0.68), data={"projection": "haworth"}),
        PathwayNode(
            "g6p",
            "Glucose-6-phosphate",
            "C(C1C(C(C(C(O1)O)O)O)OP(=O)(O)O)O",
            (0.22, 0.68),
            data={"projection": "haworth"},
        ),
        PathwayNode(
            "f6p",
            "Fructose-6-phosphate",
            "O=P(O)(O)OCC1OC(O)(CO)C(O)C1O",
            (0.36, 0.68),
            data={"projection": "haworth"},
        ),
        PathwayNode(
            "fbp",
            "Fructose-1,6-bisphosphate",
            "O=P(O)(O)OCC1OC(O)(COP(=O)(O)O)C(O)C1O",
            (0.5, 0.68),
            data={"projection": "haworth"},
        ),
        PathwayNode("dhap", "Dihydroxyacetone phosphate", "C(C(=O)COP(=O)(O)O)O", (0.5, 0.34)),
        PathwayNode("g3p", "Glyceraldehyde-3-phosphate", "C(C(C=O)O)OP(=O)(O)O", (0.64, 0.34)),
        PathwayNode("13bpg", "1,3-Bisphosphoglycerate", "O=P(O)(O)OCC(O)C(=O)OP(=O)(O)O", (0.78, 0.34)),
        PathwayNode("3pg", "3-Phosphoglycerate", "O=P(O)(O)OCC(O)C(=O)O", (0.92, 0.34)),
        PathwayNode("2pg", "2-Phosphoglycerate", "O=P(O)(O)OC(CO)C(=O)O", (0.64, 0.08)),
        PathwayNode("pep", "Phosphoenolpyruvate", "C=C(OP(=O)(O)O)C(=O)O", (0.78, 0.08)),
        PathwayNode("pyruvate", "Pyruvate", "CC(=O)C(=O)O", (0.92, 0.08)),
    ]
    edges = [
        ("glucose", "g6p"),
        ("g6p", "f6p"),
        ("f6p", "fbp"),
        ("fbp", "g3p"),
        ("fbp", "dhap"),
        PathwayEdge("dhap", "g3p", reversible=True),
        ("g3p", "13bpg"),
        ("13bpg", "3pg"),
        ("3pg", "2pg"),
        ("2pg", "pep"),
        ("pep", "pyruvate"),
    ]
    return make_pathway("Glycolysis", nodes, edges)


def _citric_acid_cycle() -> Pathway:
    nodes = [
        PathwayNode("citrate", "Citrate", "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", (0.5, 0.88)),
        PathwayNode("isocitrate", "Isocitrate", "C(C(C(CC(=O)O)C(=O)O)O)C(=O)O", (0.82, 0.68)),
        PathwayNode("akg", "alpha-Ketoglutarate", "C(CC(=O)C(=O)O)C(=O)O", (0.82, 0.32)),
        PathwayNode("succinyl_coa", "Succinyl-CoA", "C(CC(=O)S)C(=O)O", (0.5, 0.12)),
        PathwayNode("succinate", "Succinate", "C(CC(=O)O)C(=O)O", (0.18, 0.32)),
        PathwayNode("fumarate", "Fumarate", "C(=CC(=O)O)C(=O)O", (0.18, 0.68)),
        PathwayNode("malate", "Malate", "C(C(C(=O)O)O)C(=O)O", (0.34, 0.8)),
        PathwayNode("oxaloacetate", "Oxaloacetate", "C(C(=O)C(=O)O)C(=O)O", (0.66, 0.8)),
    ]
    edges = [
        ("citrate", "isocitrate"),
        ("isocitrate", "akg"),
        ("akg", "succinyl_coa"),
        ("succinyl_coa", "succinate"),
        ("succinate", "fumarate"),
        ("fumarate", "malate"),
        ("malate", "oxaloacetate"),
        ("oxaloacetate", "citrate"),
    ]
    return make_pathway("Citric Acid Cycle", nodes, edges)


def _methionine_cycle() -> Pathway:
    nodes = [
        PathwayNode("methionine", "Methionine", "CSCC[C@H](N)C(=O)O", (0.5, 0.86)),
        PathwayNode(
            "sam",
            "SAM",
            "C[S+](CC[C@H](N)C(=O)O)C[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1O",
            (0.82, 0.62),
        ),
        PathwayNode(
            "sah",
            "SAH",
            "N[C@@H](CCSC[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1O)C(=O)O",
            (0.7, 0.24),
        ),
        PathwayNode("homocysteine", "Homocysteine", "N[C@@H](CCS)C(=O)O", (0.3, 0.24)),
        PathwayNode("methyl_acceptor", "Methyl acceptor axis", None, (0.22, 0.62), "axis"),
    ]
    edges = [
        PathwayEdge("methionine", "sam", "ATP"),
        PathwayEdge("sam", "sah", "methylation"),
        PathwayEdge("sah", "homocysteine"),
        PathwayEdge("homocysteine", "methionine", "5-methyl-THF"),
        PathwayEdge("sam", "methyl_acceptor"),
    ]
    return make_pathway("Methionine Cycle", nodes, edges)


_PRESETS: dict[str, Callable[[], Pathway]] = {
    "glycolysis": _glycolysis,
    "citric acid cycle": _citric_acid_cycle,
    "tca cycle": _citric_acid_cycle,
    "methionine cycle": _methionine_cycle,
}
