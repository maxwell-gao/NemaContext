"""
Large et al. 2025 (GSE292756) downloader.

Lineage-resolved analysis of embryonic gene expression evolution in C. elegans and C. briggsae.
Published in Science, June 2025 (PMID: 40536976).

This dataset provides:
- >200,000 C. elegans cells with lineage-resolved annotations
- >175,000 C. briggsae cells for evolutionary comparison
- 429 annotated progenitor and terminal cell types
- Direct cell-to-lineage mapping (solves Packer 2019 annotation ambiguity)
- Gene regulatory network inference results

This is the recommended dataset for transcriptome-spatial-lineage integration,
as it supersedes and improves upon Packer et al. 2019 (GSE126954).
"""

from .base import BaseDownloader
from .constants import (
    LARGE2025_FILES,
    LARGE2025_SUBDIR,
    LARGE2025_TIMEOUT,
    MESSAGES,
)


class Large2025Downloader(BaseDownloader):
    """
    Downloader for Large et al. 2025 (GSE292756).

    Lineage-resolved single-cell transcriptomics of C. elegans and C. briggsae
    embryogenesis. This dataset provides improved lineage annotations compared
    to Packer et al. 2019, with direct cell-to-lineage mapping.

    Key improvements over Packer 2019:
    - 2-3x more cells (>375,000 total vs ~86,000)
    - Lineage-resolved annotations (vs ~10% clean annotations)
    - Cross-species comparison with C. briggsae
    - Gene regulatory network inference included

    Data sources:
    - GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE292756
    - GitHub: https://github.com/livinlrg/C.elegans_C.briggsae_Embryo_Single_Cell
    - Dryad: https://doi.org/10.5061/dryad.1rn8pk15n
    - Zenodo: https://doi.org/10.5281/zenodo.15091632
    """

    def download(self) -> None:
        """Download all Large et al. 2025 files from GEO."""
        self._print_header(MESSAGES["large2025_header"])

        print("📋 Dataset info:")
        print("   - >200,000 C. elegans cells + >175,000 C. briggsae cells")
        print("   - 429 annotated progenitor and terminal cell types")
        print("   - Lineage-resolved annotations (improved over Packer 2019)")
        print("   - Time range: 120-600 minutes post-cleavage")
        print()

        for key, info in LARGE2025_FILES.items():
            success = self._download_file(
                url=info["url"],
                filename=info["filename"],
                subdir=LARGE2025_SUBDIR,
                timeout=LARGE2025_TIMEOUT,
            )
            if success and "description" in info:
                print(f"   ℹ️  {info['description']}")

        self._print_additional_resources()

    def _print_additional_resources(self) -> None:
        """Print information about additional data resources."""
        print("\n" + "-" * 50)
        print("📚 Additional Resources (not auto-downloaded):")
        print("-" * 50)
        print()
        print("🔗 Interactive visualization:")
        print("   https://cello.shinyapps.io/cel_cbr_embryo_single_cell/")
        print()
        print("🔗 Gene expression browser (GExplore):")
        print("   https://genome.science.sfu.ca/gexplore")
        print()
        print("🔗 GitHub repository (code & processed data):")
        print("   https://github.com/livinlrg/C.elegans_C.briggsae_Embryo_Single_Cell")
        print()
        print("🔗 Dryad archive (full dataset):")
        print("   https://doi.org/10.5061/dryad.1rn8pk15n")
        print()
        print("🔗 Zenodo archive (code & data):")
        print("   https://doi.org/10.5281/zenodo.15091632")
        print()
        print("📖 Citation:")
        print("   Large CRL et al. (2025) Science 388:eadu8249")
        print("   DOI: 10.1126/science.adu8249 | PMID: 40536976")
