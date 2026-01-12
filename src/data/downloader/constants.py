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
# Large et al. 2025 (GSE292756) - Lineage-Resolved Embryo Atlas
# =============================================================================
# Science 2025, PMID: 40536976
# This is the RECOMMENDED dataset for transcriptome-spatial-lineage integration.
# Supersedes Packer et al. 2019 with improved lineage annotations.

LARGE2025_SUBDIR = "large2025"
LARGE2025_TIMEOUT = 300  # Large files, need longer timeout

LARGE2025_FILES = {
    "expression_matrix": {
        "filename": "GSE292756_expression_matrix.mtx.gz",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE292756&format=file&file=GSE292756%5Fexpression%5Fmatrix%2Emtx%2Egz",
        "description": "Expression matrix in MTX format (~1.1 GB) - C. elegans + C. briggsae",
    },
    "cell_annotations": {
        "filename": "GSE292756_cell_annotations.csv.gz",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE292756&format=file&file=GSE292756%5Fcell%5Fannotations%2Ecsv%2Egz",
        "description": "Cell annotations with lineage-resolved cell types (~11.7 MB)",
    },
    "gene_annotations": {
        "filename": "GSE292756_gene_annotations.csv.gz",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE292756&format=file&file=GSE292756%5Fgene%5Fannotations%2Ecsv%2Egz",
        "description": "Gene annotations (~364 KB)",
    },
}

# Additional resources (not downloaded, for reference)
LARGE2025_RESOURCES = {
    "github": "https://github.com/livinlrg/C.elegans_C.briggsae_Embryo_Single_Cell",
    "shiny_app": "https://cello.shinyapps.io/cel_cbr_embryo_single_cell/",
    "gexplore": "https://genome.science.sfu.ca/gexplore",
    "dryad": "https://doi.org/10.5061/dryad.1rn8pk15n",
    "zenodo": "https://doi.org/10.5281/zenodo.15091632",
    "paper_doi": "10.1126/science.adu8249",
    "pmid": "40536976",
}

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

# Nuclei 4D data (spatiotemporal positions)
WORMGUIDES_NUCLEI_BASE_URL = "https://raw.githubusercontent.com/zhirongbaolab/WormGUIDES/master/src/atlas_model/data/nuclei_files"
WORMGUIDES_TOTAL_TIMEPOINTS = 360
WORMGUIDES_TIME_RESOLUTION_SEC = 60  # seconds per timepoint
WORMGUIDES_START_TIME_MIN = 20  # minutes from first cleavage

# Nuclei file column indices (0-based)
NUCLEI_COL_ID = 0
NUCLEI_COL_FLAG = 1
NUCLEI_COL_X = 5
NUCLEI_COL_Y = 6
NUCLEI_COL_Z = 7
NUCLEI_COL_DIAMETER = 8
NUCLEI_COL_CELL_NAME = 9

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
# Lineage Timing Data
# =============================================================================

# Classic Sulston lineage division times (minutes after first cleavage)
# Based on Sulston et al. 1983 and refined measurements
SULSTON_DIVISION_TIMES = {
    # Founder cells
    "P0": {"birth": 0, "division": 0},
    "AB": {"birth": 0, "division": 15},
    "P1": {"birth": 0, "division": 20},
    # 2-cell to 4-cell
    "ABa": {"birth": 15, "division": 25},
    "ABp": {"birth": 15, "division": 25},
    "EMS": {"birth": 20, "division": 30},
    "P2": {"birth": 20, "division": 35},
    # 4-cell to 8-cell
    "ABal": {"birth": 25, "division": 35},
    "ABar": {"birth": 25, "division": 35},
    "ABpl": {"birth": 25, "division": 35},
    "ABpr": {"birth": 25, "division": 35},
    "MS": {"birth": 30, "division": 45},
    "E": {"birth": 30, "division": 50},
    "C": {"birth": 35, "division": 50},
    "P3": {"birth": 35, "division": 55},
    # 8-cell to 16-cell
    "ABala": {"birth": 35, "division": 50},
    "ABalp": {"birth": 35, "division": 50},
    "ABara": {"birth": 35, "division": 50},
    "ABarp": {"birth": 35, "division": 50},
    "ABpla": {"birth": 35, "division": 50},
    "ABplp": {"birth": 35, "division": 50},
    "ABpra": {"birth": 35, "division": 50},
    "ABprp": {"birth": 35, "division": 50},
    "MSa": {"birth": 45, "division": 60},
    "MSp": {"birth": 45, "division": 60},
    "Ea": {"birth": 50, "division": 90},
    "Ep": {"birth": 50, "division": 90},
    "Ca": {"birth": 50, "division": 65},
    "Cp": {"birth": 50, "division": 65},
    "D": {"birth": 55, "division": 90},
    "P4": {"birth": 55, "division": None},  # Germline precursor, divides much later
    # Germline
    "Z2": {"birth": 100, "division": None},  # Post-embryonic
    "Z3": {"birth": 100, "division": None},  # Post-embryonic
}

# =============================================================================
# Display Messages
# =============================================================================

MESSAGES = {
    "large2025_header": "🧬 Large et al. 2025 - Lineage-Resolved Embryo Atlas (RECOMMENDED)",
    "packer_header": "📊 Packer et al. 2019 - Single-cell Transcriptomics (Legacy)",
    "openworm_header": "🧠 OpenWorm/c302 - Connectome & Neuron Data",
    "wormbase_header": "🌳 WormBase - Lineage Tree Data",
    "wormguides_header": "🔬 WormGUIDES - 4D Embryo & Cell Data",
    "main_header": "🧬 NemaContext Data Downloader",
    "summary_header": "📊 Download Summary",
    "download_complete": "✨ Download complete!",
}
