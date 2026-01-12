"""
WormAtlas mapper for cell type to lineage name conversion.

This module provides mappings between terminal cell names (e.g., "ADAL", "BWM")
and their lineage origins (e.g., "AB plapaaaapp") using data from WormAtlas.

This is essential for bridging the gap between:
- Large2025 cell_type annotations (terminal differentiated names)
- WormGUIDES spatial data (lineage-based cell names)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Hardcoded mapping of common cell types to their lineage(s)
# Based on WormAtlas cell list (Sulston & White 1988)
# Format: "cell_name": ["lineage1", "lineage2", ...]
# Note: Some cells have multiple lineages due to L/R symmetry or cell fusions

CELL_TYPE_TO_LINEAGE = {
    # Neurons - Amphid
    "ADAL": ["ABplapaaaapp"],
    "ADAR": ["ABprapaaaapp"],
    "ADEL": ["ABplapaaaapa"],
    "ADER": ["ABprapaaaapa"],
    "ADFL": ["ABalpppppaa"],
    "ADFR": ["ABpraaappaa"],
    "ADLL": ["ABalppppaad"],
    "ADLR": ["ABpraaapaad"],
    "AFDL": ["ABalpppapav"],
    "AFDR": ["ABpraaaapav"],
    "AIAL": ["ABplppaappa"],
    "AIAR": ["ABprppaappa"],
    "AIBL": ["ABplaapappa"],
    "AIBR": ["ABpraapappa"],
    "AIML": ["ABplpaapppa"],
    "AIMR": ["ABprpaapppa"],
    "AINL": ["ABalaaaalal"],
    "AINR": ["ABalaapaaar"],
    "AIYL": ["ABplpapaaap"],
    "AIYR": ["ABprpapaaap"],
    "AIZL": ["ABplapaaapav"],
    "AIZR": ["ABprapaaapav"],
    "ALA": ["ABalapppaaa"],
    "ALML": ["ABarppaappa"],
    "ALMR": ["ABarpppappa"],
    "ALNL": ["ABplapappppap"],
    "ALNR": ["ABprapappppap"],
    "ASEL": ["ABalppppppaa"],
    "ASER": ["ABpraaapppaa"],
    "ASGL": ["ABplaapapap"],
    "ASGR": ["ABpraapapap"],
    "ASHL": ["ABplpaappaa"],
    "ASHR": ["ABprpaappaa"],
    "ASIL": ["ABplaapapppa"],
    "ASIR": ["ABpraapapppa"],
    "ASJL": ["ABalpppppppa"],
    "ASJR": ["ABpraaappppa"],
    "ASKL": ["ABalpppapppa"],
    "ASKR": ["ABpraaaapppa"],
    "AUAL": ["ABalpppppppp"],
    "AUAR": ["ABpraaappppp"],
    "AVAL": ["ABalppaaapa"],
    "AVAR": ["ABalaappapa"],
    "AVBL": ["ABplpaapaap"],
    "AVBR": ["ABprpaapaap"],
    "AVDL": ["ABalaaapalr"],
    "AVDR": ["ABalaaapprl"],
    "AVEL": ["ABalpppaaaa"],
    "AVER": ["ABpraaaaaaa"],
    "AVG": ["ABprpapppap"],
    "AVHL": ["ABalapaaaaaa"],
    "AVHR": ["ABapalppapaa"],
    "AVJL": ["ABaalapapppa"],
    "AVJR": ["ABalapppppa"],
    "AVKL": ["ABplpapapap"],
    "AVKR": ["ABprpapapap"],
    "AVL": ["ABprpappaap"],
    "AWAL": ["ABplaapapaa"],
    "AWAR": ["ABpraapapaa"],
    "AWBL": ["ABalpppppap"],
    "AWBR": ["ABpraaappap"],
    "AWCL": ["ABplpaaaaap"],
    "AWCR": ["ABprpaaaaap"],
    # Pharyngeal neurons
    "I1L": ["ABalpapppaa"],
    "I1R": ["ABaapappaa"],
    "I2L": ["ABalpappaapa"],
    "I2R": ["ABapapapaapa"],
    "I3": ["MSaaaaapaa"],
    "I4": ["MSaaaapaa"],
    "I5": ["ABaapapapp"],
    "I6": ["MSpaaapaa"],
    "M1": ["MSpaapaaa"],
    "M2L": ["ABaaapappa"],
    "M2R": ["ABaaappppa"],
    "M3L": ["ABaaapappp"],
    "M3R": ["ABaaappppp"],
    "M4": ["MSpaaaaaa"],
    "M5": ["MSpaaapap"],
    "MI": ["ABaaappaaa"],
    "NSML": ["ABaaapapaav"],
    "NSMR": ["ABaaapppaav"],
    # Motor neurons
    "DA1": ["ABprppapaap"],
    "DA2": ["ABplppapapa"],
    "DA3": ["ABprppapapa"],
    "DA4": ["ABplppapapp"],
    "DA5": ["ABprppapapp"],
    "DA6": ["ABplpppaaap"],
    "DA7": ["ABprpppaaap"],
    "DA8": ["ABprpapappp"],
    "DA9": ["ABplpppaaaa"],
    "DB1": ["ABplpaaaapp"],
    "DB2": ["ABaappappa"],
    "DB3": ["ABprpaaaapp"],
    "DB4": ["ABprpappapp"],
    "DB5": ["ABplpapappp"],
    "DB6": ["ABplppaappp"],
    "DB7": ["ABprppaappp"],
    "DD1": ["ABplppappap"],
    "DD2": ["ABprppappap"],
    "DD3": ["ABplppapppa"],
    "DD4": ["ABprppapppa"],
    "DD5": ["ABplppapppp"],
    "DD6": ["ABprppapppp"],
    # Ventral cord interneurons
    "PVCL": ["ABplpppaapaa"],
    "PVCR": ["ABprpppaapaa"],
    "DVA": ["ABprppppapp"],
    "DVC": ["Caapaa"],
    "PVR": ["Caappv"],
    "PVT": ["ABplpappppa"],
    # Touch receptor neurons
    "AVM": ["QRpaa"],
    "PVM": ["QLpaa"],
    "PLML": ["ABplapappppaa"],
    "PLMR": ["ABprapappppaa"],
    # HSN (hermaphrodite specific)
    "HSNL": ["ABplapppappa"],
    "HSNR": ["ABprapppappa"],
    # Amphid sheath/socket
    "AMshL": ["ABplaapaapp"],
    "AMshR": ["ABpraapaapp"],
    "AMsoL": ["ABplpaapapa"],
    "AMsoR": ["ABprpaapapa"],
    # Phasmid neurons
    "PHAL": ["ABplpppaapp"],
    "PHAR": ["ABprpppaapp"],
    "PHBL": ["ABplapppappp"],
    "PHBR": ["ABprapppappp"],
    # Body wall muscle (examples - many more exist)
    "mu_bod_01": ["Capaaaa"],
    "mu_bod_02": ["Capaaap"],
    "mu_bod_03": ["Capaapa"],
    "mu_bod_04": ["Capaapp"],
    # Intestine cells (E lineage)
    "int_1": ["Ealaad"],
    "int_2": ["Ealaav"],
    "int_3": ["Ealap"],
    "int_4": ["Ealpa"],
    "int_5": ["Ealpp"],
    # Hypodermis
    "hyp7": [
        "Caaaaaa",
        "Caaaap",
        "Caaapa",
        "Caaapp",
        "Caappd",
        "Cpaaaa",
        "Cpaaap",
        "Cpaapa",
        "Cpaapp",
        "Cpapaa",
        "Cpapap",
        "Cpappd",
    ],  # Syncytium
    # Rectal cells
    "K": ["ABplpapppaa"],
    "B": ["ABprppppapa"],
    "F": ["ABplppppapp"],
    "U": ["ABplppppapa"],
    "Y": ["ABprpppaaaa"],
    # Germline precursors
    "Z2": ["P4p"],
    "Z3": ["P4a"],
    # Coelomocytes
    "ccAL": ["MSapapaaa"],
    "ccAR": ["MSppapaaa"],
    "ccPL": ["MSapapaap"],
    "ccPR": ["MSppapaap"],
    # Excretory system
    "exc_cell": ["ABplpappaap"],
    "exc_duct": ["ABplpaaaapa"],
    "exc_gl_L": ["ABplpapapaa"],
    "exc_gl_R": ["ABprpapapaa"],
}


# Cell type categories for grouping
CELL_TYPE_CATEGORIES = {
    "neuron": [
        "ADAL",
        "ADAR",
        "ADEL",
        "ADER",
        "ADFL",
        "ADFR",
        "ASEL",
        "ASER",
        "AIYL",
        "AIYR",
        "AVAL",
        "AVAR",
        "AVBL",
        "AVBR",
        "DA1",
        "DA2",
        "DB1",
        "DB2",
        "DD1",
        "DD2",
        "PVCL",
        "PVCR",
        "DVA",
        "AVM",
        "PVM",
        "PLML",
        "PLMR",
        "HSNL",
        "HSNR",
    ],
    "pharynx": [
        "I1L",
        "I1R",
        "I2L",
        "I2R",
        "I3",
        "I4",
        "M1",
        "M2L",
        "M2R",
        "M3L",
        "M3R",
        "M4",
        "MI",
    ],
    "muscle": ["mu_bod"],
    "intestine": ["int"],
    "hypodermis": ["hyp"],
    "germline": ["Z2", "Z3"],
    "excretory": ["exc_cell", "exc_duct", "exc_gl"],
}


class WormAtlasMapper:
    """
    Maps cell types to lineage names using WormAtlas data.

    This enables bridging between:
    - Transcriptome data with cell_type annotations
    - Spatial data with lineage-based naming

    Usage:
        >>> mapper = WormAtlasMapper()
        >>> mapper.celltype_to_lineage("ADAL")
        ['ABplapaaaapp']
        >>> mapper.find_matching_lineages("BWM_head")
        ['Capaaaa', 'Capaaap', ...]
    """

    def __init__(
        self,
        data_dir: str = "dataset/raw",
        custom_mappings: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the WormAtlas mapper.

        Args:
            data_dir: Base directory containing data files.
            custom_mappings: Additional cell type to lineage mappings.
        """
        self.data_dir = Path(data_dir)
        self._mappings = CELL_TYPE_TO_LINEAGE.copy()

        if custom_mappings:
            self._mappings.update(custom_mappings)

        # Build reverse mapping (lineage -> cell types)
        self._lineage_to_celltype: Dict[str, List[str]] = {}
        for celltype, lineages in self._mappings.items():
            for lineage in lineages:
                if lineage not in self._lineage_to_celltype:
                    self._lineage_to_celltype[lineage] = []
                self._lineage_to_celltype[lineage].append(celltype)

        logger.info(
            f"Initialized WormAtlasMapper with {len(self._mappings)} cell types"
        )

    def celltype_to_lineage(self, celltype: str) -> List[str]:
        """
        Get lineage name(s) for a cell type.

        Args:
            celltype: Cell type name (e.g., "ADAL", "BWM_head").

        Returns:
            List of matching lineage names, or empty list if not found.
        """
        # Exact match
        if celltype in self._mappings:
            return self._mappings[celltype]

        # Try uppercase
        celltype_upper = celltype.upper()
        if celltype_upper in self._mappings:
            return self._mappings[celltype_upper]

        # Try without L/R suffix
        if celltype.endswith("L") or celltype.endswith("R"):
            base = celltype[:-1]
            if base + "L" in self._mappings:
                return self._mappings[base + "L"] + self._mappings.get(base + "R", [])

        return []

    def lineage_to_celltype(self, lineage: str) -> List[str]:
        """
        Get cell type name(s) for a lineage.

        Args:
            lineage: Lineage name (e.g., "ABplapaaaapp").

        Returns:
            List of matching cell type names.
        """
        # Normalize lineage (remove spaces, periods)
        lineage_clean = lineage.replace(" ", "").replace(".", "")

        if lineage_clean in self._lineage_to_celltype:
            return self._lineage_to_celltype[lineage_clean]

        # Try case-insensitive
        lineage_lower = lineage_clean.lower()
        for lin, types in self._lineage_to_celltype.items():
            if lin.lower() == lineage_lower:
                return types

        return []

    def find_matching_lineages(
        self,
        celltype_pattern: str,
        regex: bool = False,
    ) -> List[str]:
        """
        Find lineages matching a cell type pattern.

        Args:
            celltype_pattern: Pattern to match (e.g., "BWM", "hyp", "neuron").
            regex: If True, treat pattern as regex.

        Returns:
            List of matching lineage names.
        """
        matches = []

        if regex:
            pattern = re.compile(celltype_pattern, re.IGNORECASE)
            for celltype, lineages in self._mappings.items():
                if pattern.search(celltype):
                    matches.extend(lineages)
        else:
            pattern_lower = celltype_pattern.lower()
            for celltype, lineages in self._mappings.items():
                if pattern_lower in celltype.lower():
                    matches.extend(lineages)

        return list(set(matches))

    def get_category(self, celltype: str) -> str:
        """
        Get the category for a cell type.

        Args:
            celltype: Cell type name.

        Returns:
            Category name (e.g., 'neuron', 'muscle') or 'unknown'.
        """
        celltype_upper = celltype.upper()

        for category, members in CELL_TYPE_CATEGORIES.items():
            for member in members:
                if celltype_upper.startswith(member.upper()):
                    return category

        return "unknown"

    def get_all_celltypes(self) -> List[str]:
        """Get list of all known cell types."""
        return list(self._mappings.keys())

    def get_all_lineages(self) -> Set[str]:
        """Get set of all known lineages."""
        all_lineages = set()
        for lineages in self._mappings.values():
            all_lineages.update(lineages)
        return all_lineages

    def match_large2025_celltypes(
        self,
        cell_types: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Match Large2025 cell_type annotations to lineages.

        Args:
            cell_types: Series of cell type annotations from Large2025.

        Returns:
            Tuple of (matched_lineages, match_success) Series.
        """
        matched_lineages = []
        match_success = []

        for ct in cell_types:
            if pd.isna(ct) or ct == "unassigned":
                matched_lineages.append("")
                match_success.append(False)
                continue

            # Try direct match
            lineages = self.celltype_to_lineage(ct)
            if lineages:
                matched_lineages.append(lineages[0])  # Take first match
                match_success.append(True)
            else:
                # Try partial match
                lineages = self.find_matching_lineages(ct.split("_")[0])
                if lineages:
                    matched_lineages.append(lineages[0])
                    match_success.append(True)
                else:
                    matched_lineages.append("")
                    match_success.append(False)

        logger.info(
            f"Matched {sum(match_success)}/{len(cell_types)} cell types to lineages"
        )

        return pd.Series(matched_lineages), pd.Series(match_success)

    def load_from_json(self, json_path: Path) -> None:
        """
        Load additional mappings from a JSON file.

        Args:
            json_path: Path to JSON file with cell type -> lineage mappings.
        """
        with open(json_path, "r") as f:
            additional = json.load(f)

        self._mappings.update(additional)

        # Rebuild reverse mapping
        for celltype, lineages in additional.items():
            for lineage in lineages:
                if lineage not in self._lineage_to_celltype:
                    self._lineage_to_celltype[lineage] = []
                self._lineage_to_celltype[lineage].append(celltype)

        logger.info(f"Loaded {len(additional)} additional mappings from {json_path}")

    def save_to_json(self, json_path: Path) -> None:
        """
        Save current mappings to a JSON file.

        Args:
            json_path: Output path for JSON file.
        """
        with open(json_path, "w") as f:
            json.dump(self._mappings, f, indent=2, sort_keys=True)

        logger.info(f"Saved {len(self._mappings)} mappings to {json_path}")

    def get_founder_distribution(self) -> Dict[str, int]:
        """
        Get distribution of lineages by founder cell.

        Returns:
            Dict mapping founder name to count of lineages.
        """
        distribution = {"AB": 0, "MS": 0, "E": 0, "C": 0, "D": 0, "P4": 0, "other": 0}

        for lineages in self._mappings.values():
            for lineage in lineages:
                if lineage.startswith("AB"):
                    distribution["AB"] += 1
                elif lineage.startswith("MS"):
                    distribution["MS"] += 1
                elif lineage.startswith("E"):
                    distribution["E"] += 1
                elif lineage.startswith("C"):
                    distribution["C"] += 1
                elif lineage.startswith("D"):
                    distribution["D"] += 1
                elif lineage.startswith("P4") or lineage.startswith("Z"):
                    distribution["P4"] += 1
                else:
                    distribution["other"] += 1

        return distribution

    def summary(self) -> str:
        """
        Get a summary of the mapper's contents.

        Returns:
            Human-readable summary string.
        """
        founder_dist = self.get_founder_distribution()
        total_lineages = sum(len(v) for v in self._mappings.values())

        lines = [
            "WormAtlasMapper Summary",
            "=" * 40,
            f"Total cell types: {len(self._mappings)}",
            f"Total lineages: {total_lineages}",
            "",
            "Lineages by founder:",
        ]

        for founder, count in sorted(founder_dist.items(), key=lambda x: -x[1]):
            lines.append(f"  {founder}: {count}")

        return "\n".join(lines)
