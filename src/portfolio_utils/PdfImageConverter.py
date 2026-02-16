from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Literal, Sequence, cast
import subprocess


ImageFormat = Literal["png", "jpg"]
RendererName = Literal["pdftoppm", "pdftocairo"]
RendererMode = Literal["auto", "pdftoppm", "pdftocairo"]


@dataclass(frozen=True, slots=True)
class PdfToImageOptions:
    """
    Immutable conversion options used by the functional pipeline.

    Attributes:
        image_format: Output format ("png" or "jpg").
        dpi: Rendering resolution.
        quality: JPG quality in range 1..100.
        pages: Optional list of 1-based page numbers. If omitted, converts all pages.
        output_stem: Optional output filename stem (without extension).
        overwrite: Replace files if they already exist.
        renderer: Renderer strategy. "auto" picks the first available backend.
    """

    image_format: ImageFormat = "png"
    dpi: int = 200
    quality: int = 90
    pages: Sequence[int] | None = None
    output_stem: str | None = None
    overwrite: bool = False
    renderer: RendererMode = "auto"


@dataclass(frozen=True, slots=True)
class PageExportTask:
    """Single-page render task produced during planning."""

    page_number: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class PdfImageConversionReport:
    """Structured output from a completed conversion run."""

    source_pdf: Path
    output_dir: Path
    image_format: ImageFormat
    dpi: int
    renderer: RendererName
    exported_pages: tuple[int, ...]
    files: tuple[Path, ...]


def convert_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    options: PdfToImageOptions | None = None,
) -> PdfImageConversionReport:
    """
    End-to-end conversion API: plan first, execute second.

    Args:
        pdf_path: Source PDF.
        output_dir: Destination folder. Defaults to "<pdf_name>_images" next to the PDF.
        options: Optional immutable conversion options.

    Returns:
        PdfImageConversionReport with produced files and metadata.
    """
    source_pdf, target_dir, safe_options, tasks = plan_pdf_to_image_conversion(
        pdf_path=pdf_path,
        output_dir=output_dir,
        options=options,
    )
    return execute_pdf_to_image_conversion(
        source_pdf=source_pdf,
        output_dir=target_dir,
        tasks=tasks,
        options=safe_options,
    )


def pdf_to_png(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    dpi: int = 200,
    pages: Sequence[int] | None = None,
    output_stem: str | None = None,
    overwrite: bool = False,
    renderer: RendererMode = "auto",
) -> PdfImageConversionReport:
    """Convenience wrapper to export PDF pages as PNG images."""
    options = PdfToImageOptions(
        image_format="png",
        dpi=dpi,
        pages=pages,
        output_stem=output_stem,
        overwrite=overwrite,
        renderer=renderer,
    )
    return convert_pdf_to_images(pdf_path, output_dir, options=options)


def pdf_to_jpg(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    dpi: int = 200,
    quality: int = 90,
    pages: Sequence[int] | None = None,
    output_stem: str | None = None,
    overwrite: bool = False,
    renderer: RendererMode = "auto",
) -> PdfImageConversionReport:
    """Convenience wrapper to export PDF pages as JPG images."""
    options = PdfToImageOptions(
        image_format="jpg",
        dpi=dpi,
        quality=quality,
        pages=pages,
        output_stem=output_stem,
        overwrite=overwrite,
        renderer=renderer,
    )
    return convert_pdf_to_images(pdf_path, output_dir, options=options)


def plan_pdf_to_image_conversion(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    options: PdfToImageOptions | None = None,
) -> tuple[Path, Path, PdfToImageOptions, tuple[PageExportTask, ...]]:
    """
    Pure planning function: validates inputs and creates render tasks.
    """
    safe_options = _validate_options(options or PdfToImageOptions())
    source_pdf = _resolve_pdf_path(pdf_path)
    target_dir = _resolve_output_dir(source_pdf, output_dir)

    page_count = get_pdf_page_count(source_pdf)
    selected_pages = _normalize_pages(safe_options.pages, page_count)
    output_stem = _resolve_output_stem(source_pdf, safe_options.output_stem)
    suffix = ".png" if safe_options.image_format == "png" else ".jpg"

    tasks = tuple(
        PageExportTask(
            page_number=page,
            output_path=target_dir / f"{output_stem}_p{page:04d}{suffix}",
        )
        for page in selected_pages
    )
    _validate_overwrite(tasks, safe_options.overwrite)
    return source_pdf, target_dir, safe_options, tasks


def execute_pdf_to_image_conversion(
    source_pdf: str | Path,
    output_dir: str | Path,
    tasks: Sequence[PageExportTask],
    options: PdfToImageOptions,
) -> PdfImageConversionReport:
    """
    Side-effect function that executes a previously built conversion plan.
    """
    if not tasks:
        raise ValueError("No conversion tasks found.")

    safe_options = _validate_options(options)
    source = _resolve_pdf_path(source_pdf)
    destination = Path(output_dir).expanduser().resolve()
    renderer = _resolve_renderer(safe_options.renderer)
    _validate_overwrite(tasks, safe_options.overwrite)

    destination.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []

    for task in tasks:
        prefix = task.output_path.with_suffix("")
        if safe_options.overwrite and task.output_path.exists():
            task.output_path.unlink()

        command = _build_render_command(
            renderer=renderer,
            source_pdf=source,
            page_number=task.page_number,
            output_prefix=prefix,
            options=safe_options,
        )
        _run_command(command)

        rendered_file = _resolve_rendered_file(task.output_path)
        if rendered_file != task.output_path:
            if safe_options.overwrite and task.output_path.exists():
                task.output_path.unlink()
            rendered_file.replace(task.output_path)

        exported_paths.append(task.output_path)

    return PdfImageConversionReport(
        source_pdf=source,
        output_dir=destination,
        image_format=safe_options.image_format,
        dpi=safe_options.dpi,
        renderer=renderer,
        exported_pages=tuple(task.page_number for task in tasks),
        files=tuple(exported_paths),
    )


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Return total pages using `pdfinfo`."""
    _ensure_binary("pdfinfo")
    source_pdf = _resolve_pdf_path(pdf_path)
    command = ["pdfinfo", str(source_pdf)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            value = line.split(":", maxsplit=1)[1].strip()
            try:
                pages = int(value)
            except ValueError as error:
                raise RuntimeError("Failed to parse PDF page count.") from error
            if pages < 1:
                raise ValueError("PDF must contain at least one page.")
            return pages

    raise RuntimeError("Could not find page count in `pdfinfo` output.")


def _build_render_command(
    renderer: RendererName,
    source_pdf: Path,
    page_number: int,
    output_prefix: Path,
    options: PdfToImageOptions,
) -> list[str]:
    command = [
        renderer,
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-singlefile",
        "-r",
        str(options.dpi),
    ]

    if options.image_format == "png":
        command.append("-png")
    else:
        command.append("-jpeg")
        if renderer == "pdftoppm":
            command.extend(
                [
                    "-jpegopt",
                    f"quality={options.quality},progressive=y,optimize=y",
                ]
            )
        else:
            command.extend(["-jpegopt", f"quality={options.quality}"])

    command.extend([str(source_pdf), str(output_prefix)])
    return command


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        if stderr:
            raise RuntimeError(stderr) from error
        raise RuntimeError(f"Command failed: {' '.join(command)}") from error


def _resolve_renderer(renderer: RendererMode) -> RendererName:
    if renderer == "auto":
        for candidate in ("pdftoppm", "pdftocairo"):
            if which(candidate):
                return cast(RendererName, candidate)
        raise RuntimeError(
            "No PDF renderer available. Install poppler-utils "
            "(requires `pdftoppm` or `pdftocairo`)."
        )

    _ensure_binary(renderer)
    return cast(RendererName, renderer)


def _validate_options(options: PdfToImageOptions) -> PdfToImageOptions:
    image_format = _normalize_image_format(options.image_format)
    if options.dpi < 50:
        raise ValueError("dpi must be >= 50.")
    if options.quality < 1 or options.quality > 100:
        raise ValueError("quality must be within 1..100.")
    if options.renderer not in {"auto", "pdftoppm", "pdftocairo"}:
        raise ValueError(
            "renderer must be one of: 'auto', 'pdftoppm', 'pdftocairo'."
        )

    cleaned_pages = None
    if options.pages is not None:
        cleaned_pages = tuple(options.pages)
        if not cleaned_pages:
            raise ValueError("pages cannot be an empty sequence.")

    return PdfToImageOptions(
        image_format=image_format,
        dpi=options.dpi,
        quality=options.quality,
        pages=cleaned_pages,
        output_stem=options.output_stem,
        overwrite=options.overwrite,
        renderer=options.renderer,
    )


def _normalize_image_format(image_format: str) -> ImageFormat:
    normalized = image_format.strip().lower()
    if normalized == "jpeg":
        normalized = "jpg"
    if normalized not in {"png", "jpg"}:
        raise ValueError("image_format must be one of: 'png', 'jpg', 'jpeg'.")
    return cast(ImageFormat, normalized)


def _normalize_pages(pages: Sequence[int] | None, page_count: int) -> tuple[int, ...]:
    if pages is None:
        return tuple(range(1, page_count + 1))

    valid_pages: list[int] = []
    for page in pages:
        if isinstance(page, bool) or not isinstance(page, int):
            raise TypeError("pages must contain integers.")
        if page < 1 or page > page_count:
            raise ValueError(f"Page {page} is out of bounds. Valid range: 1..{page_count}")
        valid_pages.append(page)

    unique_sorted = sorted(set(valid_pages))
    if not unique_sorted:
        raise ValueError("pages must contain at least one valid page number.")
    return tuple(unique_sorted)


def _validate_overwrite(tasks: Sequence[PageExportTask], overwrite: bool) -> None:
    if overwrite:
        return

    conflicts = [str(task.output_path) for task in tasks if task.output_path.exists()]
    if conflicts:
        preview = ", ".join(conflicts[:3])
        if len(conflicts) > 3:
            preview = f"{preview}, ..."
        raise FileExistsError(
            "Output files already exist. Use overwrite=True or change output directory. "
            f"Conflicts: {preview}"
        )


def _resolve_output_stem(source_pdf: Path, output_stem: str | None) -> str:
    stem = output_stem.strip() if output_stem else source_pdf.stem
    if not stem:
        raise ValueError("output_stem cannot be empty.")
    if "/" in stem or "\\" in stem:
        raise ValueError("output_stem cannot contain path separators.")
    return stem


def _resolve_pdf_path(pdf_path: str | Path) -> Path:
    source_pdf = Path(pdf_path).expanduser().resolve()
    if not source_pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {source_pdf}")
    if source_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a .pdf file: {source_pdf}")
    return source_pdf


def _resolve_output_dir(source_pdf: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return source_pdf.parent / f"{source_pdf.stem}_images"
    return Path(output_dir).expanduser().resolve()


def _resolve_rendered_file(expected_path: Path) -> Path:
    if expected_path.exists():
        return expected_path

    alternate = expected_path.with_suffix(".jpeg")
    if alternate.exists():
        return alternate

    raise FileNotFoundError(f"Renderer did not produce expected file: {expected_path}")


def _ensure_binary(binary_name: str) -> None:
    if which(binary_name) is None:
        raise RuntimeError(
            f"Missing dependency '{binary_name}'. Install poppler-utils and retry."
        )


__all__ = [
    "PdfImageConversionReport",
    "PdfToImageOptions",
    "PageExportTask",
    "convert_pdf_to_images",
    "execute_pdf_to_image_conversion",
    "get_pdf_page_count",
    "pdf_to_jpg",
    "pdf_to_png",
    "plan_pdf_to_image_conversion",
]


def __main():
    path = "../../docs/01/"
    name = "portfolio_weights_vs_benchmark"
    
    file = "./portfolio_weights_vs_benchmark"
    
    #png_report = pdf_to_png(
    #    f"{path}/{name}.pdf",
    #    f"{path}/{name}.png", 
    #    dpi=200, 
    #    pages=[1, 2], 
    #    overwrite=True)
    
    png_report = pdf_to_png(f"{file}.pdf", f"{file}.png")
    
    return

if __name__ == "__main__":
    __main()
    
    pass