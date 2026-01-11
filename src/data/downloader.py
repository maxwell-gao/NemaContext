import os

import requests
from tqdm import tqdm


class PackerDownloader:
    """
    Chronos-CE Data Downloader for C. elegans embryogenesis (GSE126954).
    Targeting ~86,024 single cells from 100-650 min post-cleavage.
    """

    # Updated URLs based on the verified GEO Accession viewer
    FILES = {
        "matrix": {
            "filename": "GSE126954_gene_by_cell_count_matrix.txt.gz",
            "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126954&format=file&file=GSE126954%5Fgene%5Fby%5Fcell%5Fcount%5Fmatrix%2Etxt%2Egz",
        },
        "annotation": {
            "filename": "GSE126954_cell_annotation.csv.gz",
            "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126954&format=file&file=GSE126954%5Fcell%5Fannotation%2Ecsv%2Egz",
        },
        "gene_anno": {
            "filename": "GSE126954_gene_annotation.csv.gz",
            "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126954&format=file&file=GSE126954%5Fgene%5Fannotation%2Ecsv%2Egz",
        },
    }

    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download(self):
        for key, info in self.FILES.items():
            save_path = os.path.join(self.data_dir, info["filename"])

            if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                print(f"✅ {info['filename']} already exists. Skipping...")
                continue

            print(f"🚀 Downloading {info['filename']}...")

            try:
                # Using stream=True to handle the ~250MB matrix file
                response = requests.get(info["url"], stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with (
                    open(save_path, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=info["filename"],
                    ) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            except Exception as e:
                print(f"❌ Failed to download {info['filename']}: {e}")
                if os.path.exists(save_path):
                    os.remove(save_path)  # Clean up partial downloads


if __name__ == "__main__":
    downloader = PackerDownloader()
    downloader.download()
