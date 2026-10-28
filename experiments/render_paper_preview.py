# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Render a compiled paper to page PNGs for visual QA; does not change the PDF."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-dir", type=Path, help="Isolated pypdfium2 install directory")
    args = parser.parse_args()
    if args.tool_dir:
        sys.path.insert(0, str(args.tool_dir.resolve()))
    import pypdfium2 as pdfium  # pylint: disable=import-outside-toplevel
    from PIL import Image, ImageDraw  # pylint: disable=import-outside-toplevel

    args.output.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    with pdfium.PdfDocument(str(args.pdf)) as document:
        page_count = len(document)
        for index in range(page_count):
            page = document[index]
            bitmap = page.render(scale=1.5)
            rendered = bitmap.to_pil().convert("RGB")
            rendered.save(args.output / f"page-{index + 1:02d}.png")
            rendered.thumbnail((360, 510))
            thumb = Image.new("RGB", (380, 550), "#dddddd")
            thumb.paste(rendered, ((380 - rendered.width) // 2, 25))
            ImageDraw.Draw(thumb).text((10, 5), f"Page {index + 1}", fill="black")
            thumbnails.append(thumb)
            bitmap.close()
            page.close()
    sheet = Image.new("RGB", (380 * 3, 550 * ((page_count + 2) // 3)), "white")
    for index, thumb in enumerate(thumbnails):
        sheet.paste(thumb, ((index % 3) * 380, (index // 3) * 550))
    sheet.save(args.output / "contact-sheet.png")
    manifest = {
        "pdf": str(args.pdf.resolve()),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "pages": page_count,
        "renderer": "pypdfium2",
        "scale": 1.5,
        "note": "Rendering completed; human/agent visual review is a separate check.",
    }
    (args.output / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
