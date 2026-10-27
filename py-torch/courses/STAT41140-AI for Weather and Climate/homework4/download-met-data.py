"""
Download hourly weather data for all Met Éireann stations.
Files are saved to homework4/data/ as hly<stno>.csv
"""

import requests
import time
from pathlib import Path

STATIONS = {
    1875: "Athenry",
    675: "Ballyhaise",
    2375: "Belmullet",
    3723: "Casement",
    2175: "Claremorris",
    3904: "Cork Airport",
    532: "Dublin Airport",
    1375: "Dunsany",
    2075: "Finner",
    1475: "Gurteen",
    1775: "Johnstown Castle 2",
    4935: "Knock Airport",
    275: "Mace Head",
    1575: "Malin Head",
    1275: "Markree",
    575: "Moore Park",
    1975: "Mt Dillon",
    875: "Mullingar",
    1175: "Newport",
    375: "Oak Park",
    175: "Phoenix Park",
    1075: "Roches Point",
    518: "Shannon Airport",
    775: "Sherkinisland",
    2275: "Valentia Observatory",
}

BASE_URL = "https://clidata.met.ie/cli/climate_data/showdata.php"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_station(stno: int, name: str) -> None:
    url = f"{BASE_URL}?action=download&file=hly{stno}.csv"
    out_path = OUTPUT_DIR / f"hly{stno}.csv"

    if out_path.exists():
        print(f"[SKIP] {name} ({stno}) — already downloaded")
        return

    try:
        print(f"[DOWNLOADING] {name} ({stno}) ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        print(f"[OK] {name} ({stno}) — {len(response.content) / 1024:.1f} KB")
    except requests.RequestException as e:
        print(f"[ERROR] {name} ({stno}) — {e}")

def main():
    print(f"Downloading {len(STATIONS)} stations to {OUTPUT_DIR.resolve()}\n")
    for stno, name in STATIONS.items():
        download_station(stno, name)
        time.sleep(1)  # be polite to the server
    print("\nDone.")

if __name__ == "__main__":
    main()