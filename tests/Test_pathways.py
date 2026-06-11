import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from viu_chem import pathways


def test_available_presets_include_common_pathways():
    assert "glycolysis" in pathways.available_presets()
    assert "citric acid cycle" in pathways.available_presets()
    assert "methionine cycle" in pathways.available_presets()


def test_get_preset_accepts_aliases_and_rejects_unknown_names():
    tca = pathways.get_preset("TCA-cycle")

    assert tca.name == "Citric Acid Cycle"
    assert "citrate" in tca.node_map()
    with pytest.raises(ValueError, match="Unknown pathway preset"):
        pathways.get_preset("pentose phosphate party")


def test_make_pathway_validates_duplicate_and_unknown_nodes():
    with pytest.raises(ValueError, match="duplicates"):
        pathways.make_pathway(
            "Bad",
            [{"id": "a"}, {"id": "a"}],
            [],
        )

    with pytest.raises(ValueError, match="unknown node ids"):
        pathways.make_pathway(
            "Bad",
            [{"id": "a"}],
            [("a", "b")],
        )


def test_plot_pathway_draws_text_fallback_and_axis_placeholder():
    pathway = pathways.make_pathway(
        "Tiny",
        [
            {"id": "a", "label": "A", "pos": (0.25, 0.5)},
            {"id": "b", "label": "B", "pos": (0.75, 0.5)},
            {"id": "plot", "label": "Plot slot", "kind": "axis", "pos": (0.5, 0.2)},
        ],
        [("a", "b")],
    )

    result = pathways.plot_pathway(pathway, structure=False)

    assert set(result.node_axes) == {"a", "b", "plot"}
    assert result.axis.get_title() == "Tiny"
    assert any(text.get_text() == "A" for text in result.node_axes["a"].texts)
    assert any(text.get_text() == "Plot slot" for text in result.node_axes["plot"].texts)
    plt.close(result.figure)


def test_plot_pathway_axis_factory_can_populate_placeholder_axes():
    pathway = pathways.get_preset("methionine cycle")

    def draw_placeholder(ax):
        ax.plot([0, 1], [1, 0])

    result = pathways.plot_pathway(
        pathway,
        structure=False,
        axis_factory={"methyl_acceptor": draw_placeholder},
    )

    assert len(result.node_axes["methyl_acceptor"].lines) == 1
    plt.close(result.figure)


def test_pathway_with_axes_returns_copy_with_axis_nodes():
    pathway = pathways.get_preset("glycolysis")

    updated = pathway.with_axes([{"id": "kinetics", "label": "Kinetics", "pos": (0.5, 0.75)}])

    assert "kinetics" not in pathway.node_map()
    assert updated.node_map()["kinetics"].kind == "axis"


def test_glycolysis_preset_includes_major_intermediates():
    pathway = pathways.get_preset("glycolysis")

    assert set(pathway.node_map()) >= {
        "glucose",
        "g6p",
        "f6p",
        "fbp",
        "dhap",
        "g3p",
        "13bpg",
        "3pg",
        "2pg",
        "pep",
        "pyruvate",
    }
    assert ("dhap", "g3p") in {(edge.source, edge.target) for edge in pathway.edges}


def test_layout_axis_nodes_places_axis_near_anchor_with_shared_size():
    pathway = pathways.make_pathway(
        "Layout",
        [
            {"id": "a", "label": "A", "pos": (0.3, 0.5)},
            {"id": "b", "label": "B", "pos": (0.7, 0.5)},
        ],
        [("a", "b")],
    )

    axes = pathways.layout_axis_nodes(
        pathway,
        [
            {"id": "a_plot", "kind": "axis", "pos": (0.1, 0.1)},
            {"id": "b_plot", "kind": "axis", "pos": (0.9, 0.1)},
        ],
        anchors={"a_plot": ("a",), "b_plot": ("b",)},
        node_size=(0.12, 0.12),
        axis_size=(0.1, 0.08),
        same_size=True,
        max_anchor_distance=0.35,
        edge_connectionstyle={("a", "b"): "arc3,rad=0.2"},
    )

    assert {axis.id for axis in axes} == {"a_plot", "b_plot"}
    assert axes[0].data["axis_size"] == axes[1].data["axis_size"]
    assert axes[0].pos != (0.1, 0.1)
