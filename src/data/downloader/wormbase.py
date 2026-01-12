"""
WormBase-style lineage data downloader.

Extracts lineage information from WormGUIDES partslist and generates
structured lineage tree data.
"""

import json
from pathlib import Path

from .base import BaseDownloader
from .constants import (
    CELL_FATE_KEYWORDS,
    MESSAGES,
    WORMBASE_FOUNDER_CELLS,
    WORMBASE_PARTSLIST_URL,
    WORMBASE_SUBDIR,
)


class WormBaseDownloader(BaseDownloader):
    """
    Downloader for WormBase-style lineage data.
    Extracts lineage information from WormGUIDES partslist and generates
    structured lineage tree data.
    """

    def download(self) -> None:
        """Download and process lineage data."""
        self._print_header(MESSAGES["wormbase_header"])

        # Create output directory
        save_dir = self.data_dir / WORMBASE_SUBDIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Download partslist if not exists
        partslist_path = save_dir / "partslist.txt"
        if not partslist_path.exists():
            self._download_file(
                url=WORMBASE_PARTSLIST_URL,
                filename="partslist.txt",
                subdir=WORMBASE_SUBDIR,
            )

        # Parse partslist and generate lineage data
        self._generate_cell_lineage_map(save_dir, partslist_path)
        self._generate_lineage_tree(save_dir)
        self._generate_cell_fates(save_dir, partslist_path)

    def _generate_cell_lineage_map(self, save_dir: Path, partslist_path: Path) -> None:
        """Generate cell-to-lineage mapping from partslist."""
        output_path = save_dir / "cell_lineage_map.json"
        if output_path.exists():
            print("✅ cell_lineage_map.json already exists. Skipping...")
            return

        print("📝 Generating cell_lineage_map.json from partslist...")

        cell_lineage = {}
        if partslist_path.exists():
            with open(partslist_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        cell_name = parts[0]
                        lineage = parts[1]
                        description = parts[2] if len(parts) > 2 else ""
                        cell_lineage[cell_name] = {
                            "lineage": lineage,
                            "description": description,
                        }

        with open(output_path, "w") as f:
            json.dump(cell_lineage, f, indent=2)
        print(f"✅ Created cell_lineage_map.json ({len(cell_lineage)} cells)")

    def _generate_lineage_tree(self, save_dir: Path) -> None:
        """Generate lineage tree with parent-child relationships."""
        tree_path = save_dir / "lineage_tree.json"
        if tree_path.exists():
            print("✅ lineage_tree.json already exists. Skipping...")
            return

        print("📝 Generating lineage_tree.json...")

        # Build tree from lineage naming convention
        # In C. elegans lineage names, each character after the founder
        # represents a cell division (a=anterior, p=posterior, l=left, r=right, d=dorsal, v=ventral)
        tree = {}

        # Add founder cells to tree
        for cell, info in WORMBASE_FOUNDER_CELLS.items():
            tree[cell] = info.copy()

        # Read cell_lineage_map to extract all lineage names
        lineage_map_path = save_dir / "cell_lineage_map.json"
        if lineage_map_path.exists():
            with open(lineage_map_path) as f:
                cell_lineage = json.load(f)

            # Collect all unique lineage names
            lineages = set()
            for info in cell_lineage.values():
                lineage = info.get("lineage", "")
                if lineage:
                    lineages.add(lineage)
                    # Add all parent lineages
                    for i in range(1, len(lineage)):
                        lineages.add(lineage[:i])

            # Build tree from lineage names
            for lineage in sorted(lineages, key=len):
                if lineage not in tree and len(lineage) > 0:
                    # Find parent (lineage without last character)
                    parent = lineage[:-1] if len(lineage) > 1 else None

                    # Special handling for founder-derived cells
                    if lineage.startswith("AB") and len(lineage) > 2:
                        parent = lineage[:-1]
                    elif lineage in ["ABa", "ABp"]:
                        parent = "AB"

                    tree[lineage] = {
                        "children": [],
                        "parent": parent,
                    }

                    # Add this cell to parent's children
                    if parent and parent in tree:
                        if lineage not in tree[parent]["children"]:
                            tree[parent]["children"].append(lineage)

        with open(tree_path, "w") as f:
            json.dump(tree, f, indent=2)
        print(f"✅ Created lineage_tree.json ({len(tree)} lineage nodes)")

    def _generate_cell_fates(self, save_dir: Path, partslist_path: Path) -> None:
        """Generate cell fate annotations grouped by tissue type."""
        fates_path = save_dir / "cell_fates.json"
        if fates_path.exists():
            print("✅ cell_fates.json already exists. Skipping...")
            return

        print("📝 Generating cell_fates.json...")

        cell_fates: dict[str, list[str]] = {}

        if partslist_path.exists():
            with open(partslist_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        cell_name = parts[0]
                        description = parts[2].lower()

                        # Categorize by description keywords
                        fate = self._classify_cell_fate(description)

                        if fate not in cell_fates:
                            cell_fates[fate] = []
                        cell_fates[fate].append(cell_name)

        # Sort cells in each category
        for fate in cell_fates:
            cell_fates[fate] = sorted(set(cell_fates[fate]))

        with open(fates_path, "w") as f:
            json.dump(cell_fates, f, indent=2)

        total_cells = sum(len(cells) for cells in cell_fates.values())
        print(
            f"✅ Created cell_fates.json ({len(cell_fates)} fates, {total_cells} cells)"
        )

    @staticmethod
    def _classify_cell_fate(description: str) -> str:
        """Classify cell fate based on description keywords."""
        for fate, keywords in CELL_FATE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description:
                    return fate
        return "Other"
