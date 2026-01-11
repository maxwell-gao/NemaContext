import os

import requests
from tqdm import tqdm


class PackerDownloader:
    BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126954/suppl/"
    FILES = {
        "matrix": "GSE126954_gene_by_cell.csv.gz",
        "annotation": "GSE126954_cell_annotation.csv.gz",
    }

    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download(self):
        for key, filename in self.FILES.items():
            save_path = os.path.join(self.data_dir, filename)
            if os.path.exists(save_path):
                print(f"File {filename} already exists. Skipping...")
                continue

            url = self.BASE_URL + filename
            print(f"Downloading {filename} from GEO...")

            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(save_path, "wb") as f,
                tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=filename
                ) as pbar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))


if __name__ == "__main__":
    downloader = PackerDownloader()
    downloader.download()
