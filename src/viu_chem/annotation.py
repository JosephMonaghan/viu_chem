
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from enum import Enum
from importlib.resources import files


path = files("viu_chem") / "hmdb_metabolites.csv.gz"
hmdb_metabolites = pd.read_csv(path)


# hmdb_metabolites = pd.read_csv("hmdb_metabolites.csv.gz")


@dataclass(frozen=True)
class Adduct:
    """Container describing an ion adduct.
    
    :param label: Display label for the adduct
    :param exact_mass: Exact mass shift for the adduct
    :param charge: Charge state for the adduct
    :param ion_mode: Ionization mode for the adduct"""
    label:str
    exact_mass: float
    charge: int
    ion_mode: Literal["pos", "neg"]


class AdductLabel(str, Enum):
    """Enumeration of supported adduct labels."""
    M = "M+"
    M_H = "[M+H]+"
    M_Na = "[M+Na]+"
    M_K = "[M+K]+"
    M_H_neg = "[M-H]-"
    M_Cl = "[M+Cl]-"
    M_FA = "[M+FA]-"
    M_Ac = "[M+Ac]-"
    M_Li = "[M+Li]+"
    M_2Li = "[M-H+2Li]+"



ALL_ADDUCTS: dict[str, Adduct] = {
    "M+": Adduct("M+",   0.0,       1, "pos"),
    "[M+H]+": Adduct("M+H", 1.00783,   1, "pos"),
    "[M+Na]+": Adduct("M+Na",22.98977,  1, "pos"),
    "[M+K]+": Adduct("M+K", 38.96371,  1, "pos"),
    "[M+NH4]+": Adduct("M+NH4", 18.033823, 1, "pos"),
    "[M+Li]+": Adduct("M+Li", 7.016,1, "pos"),
    "[M-H+2Li]+": Adduct("M-H+2Li",13.0242,1,"pos"),

    "[M-H]-": Adduct("M-H", -1.00783, 1, "neg"),
    "[M+Cl]-": Adduct("M+Cl", 34.9694, 1, "neg"),
    "[M+FA]-": Adduct("M+FA", 44.99820,   1, "neg"),
    "[M+Ac]-": Adduct("M+Ac", 58.00548,   1, "neg"),
}


DEFAULT = [AdductLabel.M, AdductLabel.M_H, AdductLabel.M_Na, AdductLabel.M_K]

def resolve_adduct_labels(labels: list[AdductLabel]) -> list[Adduct]:
    """Resolves adduct label enum values to their adduct definitions.
    
    :param labels: List of adduct labels to resolve
    :return: List of matching adduct definitions"""
    return [ALL_ADDUCTS[lbl.value] for lbl in labels]


def adducts_to_df(adduct_list: list[Adduct]) -> pd.DataFrame:
    """Converts a list of adduct definitions into a dataframe.
    
    :param adduct_list: List of adduct definitions
    :return: Dataframe containing adduct labels, masses, charges, and ion modes"""
    return pd.DataFrame([vars(a) for a in adduct_list])

def tol_range(mz:float, tol:float=5):
    """Calculates an m/z tolerance window in daltons from a ppm tolerance.
    
    :param mz: Center m/z value
    :param tol: Tolerance in ppm
    :return: Tolerance window in daltons"""
    return mz*tol/1e6

def annotate_mz(mz:float,adducts:list[AdductLabel] | None=None,tol:float=5) ->pd.DataFrame:
    """Annotates a mz based on matching within a tolerance to the HMDB database for specified adducts.
    
    :param mz: m/z value to search for (as observed, do not compensate for adduct weight)
    :param adducts: Which m/z adducts to search for, specified as a list of AdductLabel.[adduct] objects (typehinting should help here)
    :param tol: m/z tolerance to search the database with
    
    :return: Dataframe containing matching metabolites with their respective adducts, sorted by their ppm offset"""
    if adducts is None:
        adducts = DEFAULT

    adducts = resolve_adduct_labels(adducts)    
    adducts_lib = adducts_to_df(adducts)


    match_lib = {}
    # m/z range
    window = tol_range(mz, tol)
    for _, row in adducts_lib.iterrows():
        # Convert observed m/z into neutral mass by subtracting adduct mass
        neutral_mass = mz - row.exact_mass

        # Filter HMDB within tolerance
        subset = hmdb_metabolites[
            (hmdb_metabolites["exact_mass"] >= (neutral_mass - window)) &
            (hmdb_metabolites["exact_mass"] <= (neutral_mass + window))
        ].copy()

        if len(subset) == 0:
            continue

        # Label the adduct
        subset["adduct"] = row.label
        subset["ppm_offset"] = abs((mz - (subset['exact_mass'] + row.exact_mass)) * 1e6 / mz)

        # Store in dictionary
        match_lib[row.label] = subset

    # If nothing matched
    if not match_lib:
        return pd.DataFrame()  # Return empty DF

    # Combine everything into one annotation table
    result = pd.concat(match_lib.values(), ignore_index=True)
    result = result.sort_values(by="ppm_offset").reset_index(drop=True)

    return result




        


    
    # return return_dict
