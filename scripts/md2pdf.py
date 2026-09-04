"""把 Markdown 文档转成 PDF。

走 pandoc 生成 HTML，再用 Edge 或 Chrome 的无头模式打印。这台机器上没有 LaTeX
引擎，pandoc 不能直接输出 PDF。

用法：
    python scripts/md2pdf.py docs/第1周实验报告.md
    python scripts/md2pdf.py docs/*.md -o ../pdf
    python scripts/md2pdf.py a.md b.md --merge 合集.pdf   # 合成一份
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body{ font-family:"Microsoft YaHei","PingFang SC",sans-serif; color:#1a1a1a;
      line-height:1.7; font-size:10.5pt; }
h1{ font-size:19pt; margin:0 0 4pt; padding-bottom:6pt; border-bottom:2px solid #2b5fa8;
    page-break-before:always; }
h1:first-of-type{ page-break-before:auto; }
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
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browsers() -> list:
    """按顺序返回所有可用的浏览器。

    不只挑第一个：无头打印有时会静默失败，退出码 0 但不产出 PDF。实测遇到过
    Edge 对所有文档都这样而 Chrome 正常，所以逐个试到能出 PDF 为止。
    """
    found = [p for p in BROWSERS if Path(p).exists()]
    for name in ("chrome", "msedge"):
        w = shutil.which(name)
        if w and w not in found:
            found.append(w)
    if not found:
        raise SystemExit("找不到 Chrome 或 Edge，无法打印 PDF")
    return found


def convert(sources, out_pdf: Path, browsers: list) -> None:
    """把一个或多个 Markdown 转成一份 PDF。

    中间文件放在 ASCII 临时目录，避免浏览器处理中文路径出问题。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        css, html, pdf = tmp / "s.css", tmp / "d.html", tmp / "d.pdf"
        css.write_text(CSS, encoding="utf-8")

        src = tmp / "in.md"
        src.write_text(
            "\n\n".join(Path(f).read_text(encoding="utf-8").strip() for f in sources),
            encoding="utf-8",
        )
        subprocess.run(
            ["pandoc", str(src), "-f", "gfm", "-t", "html5", "-s",
             "--metadata", "title=", "-c", str(css),
             "--embed-resources", "--standalone", "-o", str(html)],
            check=True, capture_output=True,
        )
        # 单独的用户数据目录：用户已经开着浏览器时，无头实例复用默认目录会起不来
        for i, browser in enumerate(browsers):
            subprocess.run(
                [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                 f"--user-data-dir={tmp / f'profile{i}'}",
                 f"--print-to-pdf={pdf}", html.as_uri()],
                capture_output=True, timeout=120,
            )
            if pdf.exists():
                break
        else:
            raise SystemExit(
                f"打印失败：{out_pdf.name}，试过 " + "、".join(Path(b).name for b in browsers)
            )

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="要转换的 Markdown 文件")
    ap.add_argument("-o", "--out-dir", default=None, help="输出目录，默认与源文件同级")
    ap.add_argument("--merge", metavar="PDF", default=None, help="合成一份 PDF，按给定顺序拼接")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        raise SystemExit("找不到 pandoc")
    browsers = find_browsers()

    files = [Path(f) for f in args.files]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit("文件不存在：" + "、".join(str(m) for m in missing))

    if args.merge:
        out = Path(args.merge)
        if args.out_dir:
            out = Path(args.out_dir) / out.name
        convert(files, out, browsers)
        print(f"  {len(files)} 份合成  ->  {out}  ({out.stat().st_size // 1024} KB)")
        return

    for md in files:
        out = (Path(args.out_dir) if args.out_dir else md.parent) / (md.stem + ".pdf")
        convert([md], out, browsers)
        print(f"  {md.name}  ->  {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
