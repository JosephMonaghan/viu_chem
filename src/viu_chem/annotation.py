from lamp import anno
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from enum import Enum

try:
    from lamp import anno
except ImportError:
    raise RuntimeError("This feature requires the optional dependency: lamp")


ref_dict = {
    "pos": anno.read_ref("",ion_mode="pos", calc=False),
    "neg": anno.read_ref("",ion_mode="neg", calc=True)
}


@dataclass(frozen=True)
class Adduct:
    label:str
    exact_mass: float
    charge: int
    ion_mode: Literal["pos", "neg"]


class AdductLabel(str, Enum):
    M = "M+"
    M_H = "[M+H]+"
    M_Na = "[M+Na]+"
    M_K = "[M+K]+"
    M_H_neg = "[M-H]-"
    M_Cl = "[M+Cl]-"
    M_FA = "[M+FA]-"
    M_Ac = "[M+Ac]-"



ALL_ADDUCTS: dict[str, Adduct] = {
    "M+": Adduct("M+",   0.0,       1, "pos"),
    "[M+H]+": Adduct("M+H", 1.00783,   1, "pos"),
    "[M+Na]+": Adduct("M+Na",22.98977,  1, "pos"),
    "[M+K]+": Adduct("M+K", 38.96371,  1, "pos"),
    "[M+NH4]+": Adduct("M+NH4", 18.033823, 1, "pos"),

    "[M-H]-": Adduct("M-H", -1.00783, 1, "neg"),
    "[M+Cl]-": Adduct("M+Cl", 34.9694, 1, "neg"),
    "[M+FA]-": Adduct("M+FA", 44.99820,   1, "neg"),
    "[M+Ac]-": Adduct("M+Ac", 58.00548,   1, "neg"),
}


DEFAULT_POSITIVE = [AdductLabel.M, AdductLabel.M_H, AdductLabel.M_Na, AdductLabel.M_K]
DEFAULT_NEGATIVE= [AdductLabel.M_H_neg, AdductLabel.M_Cl]

def resolve_adduct_labels(labels: list[AdductLabel]) -> list[Adduct]:
    return [ALL_ADDUCTS[lbl.value] for lbl in labels]


def adducts_to_df(adduct_list: list[Adduct]) -> pd.DataFrame:
    return pd.DataFrame([vars(a) for a in adduct_list])

def annotate_mz(mz:float,polarity:Literal["pos", "neg"]="pos",adducts:list[AdductLabel] | None=None,tol:float=5) ->dict:
    temp_df = pd.DataFrame([["tmp", mz, 1]],columns=["name", "mz", "rt"])

    if adducts is None:
        adducts = (
            DEFAULT_POSITIVE if polarity == "pos" else DEFAULT_NEGATIVE
        )
    adduct_labels = {a.value for a in adducts}
    adducts = resolve_adduct_labels(adducts)    
    adducts_lib = adducts_to_df(adducts)

    # Try searching for matches, return none if no matches found
    try:
        result = anno.comp_match_mass_add(temp_df,tol,ref_dict[polarity], adducts_lib)
    except KeyError:
        return {}
    
    print(result)
    print(result.columns)
    return_dict = {}
    for _, possible_annotation in result.iterrows():
        print(f"ion_type: {possible_annotation.ion_type}") 
        print(f"Adduct {possible_annotation.adduct}")     
        if possible_annotation.ion_type in adduct_labels:          
            return_dict[possible_annotation["compound_name"]] = {
                "name": possible_annotation.compound_name,
                "formula": possible_annotation.molecular_formula,
                "exact_mass": possible_annotation.exact_mass,
                "hmdb_id": possible_annotation.hmdb_id,
                "kegg_id": possible_annotation.kegg_id,
                "adduct": possible_annotation.ion_type,
                "ppm_error":possible_annotation.ppm_error
            }
    
    return return_dict
