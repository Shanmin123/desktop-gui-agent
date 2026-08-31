"""把 Markdown 文档转成 PDF。

走 pandoc 生成 HTML，再用 Edge 或 Chrome 的无头模式打印。这台机器上没有 LaTeX
引擎，pandoc 不能直接输出 PDF。

用法：
    python scripts/md2pdf.py docs/第1周实验报告.md
    python scripts/md2pdf.py docs/*.md -o ../交付物
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body{ font-family:"Microsoft YaHei","PingFang SC",sans-serif; color:#1a1a1a;
      line-height:1.7; font-size:10.5pt; }
h1{ font-size:19pt; margin:0 0 4pt; padding-bottom:6pt; border-bottom:2px solid #2b5fa8; }
h2{ font-size:13pt; margin:18pt 0 8pt; color:#1c3f6e; page-break-after:avoid; }
h3{ font-size:11pt; margin:12pt 0 6pt; color:#243447; page-break-after:avoid; }
p{ margin:6pt 0; }
table{ border-collapse:collapse; width:100%; margin:8pt 0; font-size:9.5pt;
       page-break-inside:avoid; }
th,td{ border:1px solid #d5d9de; padding:4pt 7pt; text-align:left; vertical-align:top; }
th{ background:#f2f5f8; font-weight:600; }
code{ font-family:Consolas,monospace; font-size:9pt; background:#f4f6f8;
      padding:0 3pt; border-radius:2px; }
pre{ background:#f4f6f8; border:1px solid #e0e4e8; padding:6pt 9pt;
     font-size:9pt; page-break-inside:avoid; }
pre code{ background:none; padding:0; }
ul,ol{ padding-left:18pt; } li{ margin:3pt 0; }
hr{ border:none; border-top:1px solid #e0e4e8; margin:14pt 0; }
a{ color:#2b5fa8; text-decoration:none; }
"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    found = shutil.which("msedge") or shutil.which("chrome")
    if not found:
        raise SystemExit("找不到 Edge 或 Chrome，无法打印 PDF")
    return found


def convert(md: Path, out_pdf: Path, browser: str) -> None:
    """中间文件放在 ASCII 临时目录，避免浏览器处理中文路径出问题。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        css, html, pdf = tmp / "s.css", tmp / "d.html", tmp / "d.pdf"
        css.write_text(CSS, encoding="utf-8")

        subprocess.run(
            ["pandoc", str(md), "-f", "gfm", "-t", "html5", "-s",
             "--metadata", "title=", "-c", str(css),
             "--embed-resources", "--standalone", "-o", str(html)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf}", html.as_uri()],
            check=True, capture_output=True, timeout=120,
        )
        if not pdf.exists():
            raise SystemExit(f"{md.name} 打印失败")

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="要转换的 Markdown 文件")
    ap.add_argument("-o", "--out-dir", default=None, help="输出目录，默认与源文件同级")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        raise SystemExit("找不到 pandoc")
    browser = find_browser()

    for f in args.files:
        md = Path(f)
        if not md.exists():
            print(f"  跳过，文件不存在：{md}")
            continue
        out = (Path(args.out_dir) if args.out_dir else md.parent) / (md.stem + ".pdf")
        convert(md, out, browser)
        print(f"  {md.name}  ->  {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
