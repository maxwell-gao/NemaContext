"""
Constants and configuration for NemaContext data downloaders.

All URLs, file definitions, and configuration values are centralized here
for easy maintenance and updates.
"""

# =============================================================================
# Global Configuration
# =============================================================================

DEFAULT_DATA_DIR = "dataset/raw"
DEFAULT_TIMEOUT = 60
DEFAULT_MIN_FILE_SIZE = 100  # bytes
DOWNLOAD_CHUNK_SIZE = 8192

# =============================================================================
# Packer et al. 2019 (GSE126954) - Single-cell Transcriptomics
# =============================================================================

PACKER_SUBDIR = "packer2019"
PACKER_TIMEOUT = 120  # Larger timeout for big files

PACKER_FILES = {
    "matrix": {
        "filename": "GSE126954_gene_by_cell_count_matrix.txt.gz",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126954&format=file&file=GSE126954%5Fgene%5Fby%5Fcell%5Fcount%5Fmatrix%2Etxt%2Egz",
        "description": "Gene-by-cell count matrix (~250MB)",
    },
    "cell_annotation": {
        "filename": "GSE126954_cell_annotation.csv.gz",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126954&format=file&file=GSE126954%5Fcell%5Fannotation%2Ecsv%2Egz",
        "description": "Cell annotations with lineage, time, cell type",
    },
    "gene_annotation": {
        "filename": "GSE126954_gene_annotation.csv.gz",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126954&format=file&file=GSE126954%5Fgene%5Fannotation%2Ecsv%2Egz",
        "description": "Gene annotations",
    },
}

# =============================================================================
# OpenWorm/c302 - Connectome & Neuron Data
# =============================================================================

OPENWORM_SUBDIR = "openworm"
OPENWORM_BASE_URL = "https://raw.githubusercontent.com/openworm/c302/master/c302/data"

OPENWORM_FILES = {
    # Connectome data
    "connectome_white_whole": {
        "filename": "aconnectome_white_1986_whole.csv",
        "description": "White et al. 1986 connectome (whole animal)",
    },
    "connectome_white_adult": {
        "filename": "aconnectome_white_1986_A.csv",
        "description": "White et al. 1986 connectome (adult)",
    },
    "connectome_white_l4": {
        "filename": "aconnectome_white_1986_L4.csv",
        "description": "White et al. 1986 connectome (L4 larva)",
    },
    "connectome_edgelist": {
        "filename": "herm_full_edgelist.csv",
        "description": "Full hermaphrodite connectome edge list",
    },
    # Neuron information
    "neuron_tables": {
        "filename": "CElegansNeuronTables.xls",
        "description": "C. elegans neuron tables (Excel)",
    },
    "neuron_connect": {
        "filename": "NeuronConnectFormatted.xlsx",
        "description": "Formatted neuron connectivity (Excel)",
    },
    # Expression data
    "expression_bentley": {
        "filename": "Bentley_et_al_2016_expression.csv",
        "description": "Bentley et al. 2016 gene expression",
    },
    # OpenWorm metadata cache
    "owmeta_cache": {
        "filename": "owmeta_cache.json",
        "description": "OpenWorm metadata cache (cell info)",
    },
}

# =============================================================================
# WormBase - Lineage Tree Data
# =============================================================================

WORMBASE_SUBDIR = "wormbase"
WORMBASE_PARTSLIST_URL = "https://raw.githubusercontent.com/zhirongbaolab/WormGUIDES/master/src/application_src/application_model/data/CElegansData/PartsList/partslist.txt"

# Founder cells for lineage tree construction
WORMBASE_FOUNDER_CELLS = {
    "P0": {"children": ["AB", "P1"], "parent": None},
    "AB": {"children": ["ABa", "ABp"], "parent": "P0"},
    "P1": {"children": ["EMS", "P2"], "parent": "P0"},
    "EMS": {"children": ["MS", "E"], "parent": "P1"},
    "P2": {"children": ["C", "P3"], "parent": "P1"},
    "MS": {"children": ["MSa", "MSp"], "parent": "EMS"},
    "E": {"children": ["Ea", "Ep"], "parent": "EMS"},
    "C": {"children": ["Ca", "Cp"], "parent": "P2"},
    "P3": {"children": ["D", "P4"], "parent": "P2"},
    "D": {"children": ["Da", "Dp"], "parent": "P3"},
    "P4": {"children": ["Z2", "Z3"], "parent": "P3"},
}

# Cell fate keywords for classification
CELL_FATE_KEYWORDS = {
    "Neuron": ["neuron", "interneuron"],
    "Muscle": ["muscle"],
    "Hypodermis": ["hypoderm", "skin"],
    "Intestine": ["intestin", "gut"],
    "Pharynx": ["pharyn"],
    "Germline": ["gonad", "germ"],
    "Sheath": ["sheath"],
    "Socket": ["socket"],
    "Glia": ["glia"],
    "Epithelial": ["epithe"],
    "Excretory": ["excret", "canal"],
    "Seam": ["seam"],
    "Rectal": ["rectal", "rect"],
    "Vulva": ["vulva"],
    "Uterine": ["uterine"],
    "Sensory": ["sensory", "receptor"],
    "Motor": ["motor"],
}

# =============================================================================
# WormGUIDES - 4D Embryo & Cell Data
# =============================================================================

WORMGUIDES_SUBDIR = "wormguides"
WORMGUIDES_BASE_URL = "https://raw.githubusercontent.com/zhirongbaolab/WormGUIDES/master/src/application_src/application_model/data/CElegansData"

WORMGUIDES_FILES = {
    "anatomy": {
        "filename": "anatomy.csv",
        "url_path": "Anatomy/anatomy.csv",
        "description": "Cell anatomy annotations",
    },
    "cell_deaths": {
        "filename": "CellDeaths.csv",
        "url_path": "CellDeaths/CellDeaths.csv",
        "description": "Programmed cell death data",
    },
    "connectome": {
        "filename": "NeuronConnect.csv",
        "url_path": "Connectome/NeuronConnect.csv",
        "description": "Neuron connectivity data",
    },
    "parts_list": {
        "filename": "partslist.txt",
        "url_path": "PartsList/partslist.txt",
        "description": "Complete parts list of C. elegans cells",
    },
    "analogous_cells": {
        "filename": "EmbryonicAnalogousCells.csv",
        "url_path": "AnalogousCells/EmbryonicAnalogousCells.csv",
        "description": "Embryonic analogous cells mapping",
    },
}

# =============================================================================
# Display Messages
# =============================================================================

MESSAGES = {
    "packer_header": "📊 Packer et al. 2019 - Single-cell Transcriptomics",
    "openworm_header": "🧠 OpenWorm/c302 - Connectome & Neuron Data",
    "wormbase_header": "🌳 WormBase - Lineage Tree Data",
    "wormguides_header": "🔬 WormGUIDES - 4D Embryo & Cell Data",
    "main_header": "🧬 NemaContext Data Downloader",
    "summary_header": "📊 Download Summary",
    "download_complete": "✨ Download complete!",
}
