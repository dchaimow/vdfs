"""Usage: build_report.py <results dir>
Inlines the figures and data from <results dir>/report into report_template.html and writes
<results dir>/report/vdfs_equivolume_report.html (a single self-contained file)."""
import base64
import re
import sys
from pathlib import Path

report = Path(sys.argv[1]) / "report"
html = (Path(__file__).parent / "report_template.html").read_text()
html = re.sub(r"\{\{IMG:(\w+)\}\}", lambda m: "data:image/png;base64,"
              + base64.b64encode((report / f"{m.group(1)}.png").read_bytes()).decode(), html)
html = html.replace("{{DATA}}", (report / "report_data.json").read_text())
assert "{{" not in html, "unreplaced placeholder in template"
(report / "vdfs_equivolume_report.html").write_text(html)
print("written", report / "vdfs_equivolume_report.html", f"({len(html) / 1e6:.2f} MB)")
