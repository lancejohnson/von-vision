"""Download a small, openly licensed visual-scoring demo set from Wikimedia Commons."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


FILES = {
    "mountain_lake": "File:Lake Abert OR - Abert Rim and Lake Abert on the Three Flags Highway, Shasta Cascade Wonderland (NBY 432290).jpg",
    "ocean_balcony": "File:Hawaii Hotel Balcony - Ocean View - Poipu Beach Kauai (53659308081).jpg",
    "city_skyline": "File:Quebec City skyline from a park.jpg",
    "forest": "File:John William North - Forest Landscape - B2015.18.11 - Yale Center for British Art.jpg",
    "bedroom": "File:American homes and gardens (1911) (17971440349).jpg",
    "parking_lot": "File:Parking lot, Gross-Gerau (P1090665).jpg",
}

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "von-vision-demo/0.1 (https://github.com/lancejohnson/von-vision)"


def plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def main() -> None:
    destination = Path("data/demo-images")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []

    for slug, title in FILES.items():
        query = urllib.parse.urlencode(
            {
                "action": "query",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 960,
                "format": "json",
            }
        )
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            page = next(iter(json.load(response)["query"]["pages"].values()))

        info = page["imageinfo"][0]
        metadata = info.get("extmetadata", {})
        image_url = info.get("thumburl", info["url"])
        suffix = Path(urllib.parse.urlparse(image_url).path).suffix or ".jpg"
        output = destination / f"{slug}{suffix}"
        download = urllib.request.Request(image_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(download, timeout=60) as response:
            output.write_bytes(response.read())

        item = {
            "slug": slug,
            "file": output.name,
            "title": page["title"],
            "source": info["descriptionurl"],
            "license": plain(metadata.get("LicenseShortName", {}).get("value", "Unknown")),
            "license_url": metadata.get("LicenseUrl", {}).get("value"),
            "artist": plain(metadata.get("Artist", {}).get("value", "Unknown")),
        }
        manifest.append(item)
        print(f"Downloaded {output} ({item['license']})")

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
