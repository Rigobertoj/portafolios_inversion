"""Snake-case compatibility wrapper for PDF image conversion helpers.

This module re-exports the public API from `PdfImageConverter.py` under a
snake-case module name for newer imports.
"""

from .PdfImageConverter import (
    PageExportTask,
    PdfImageConversionReport,
    PdfToImageOptions,
    convert_pdf_to_images,
    execute_pdf_to_image_conversion,
    get_pdf_page_count,
    pdf_to_jpg,
    pdf_to_png,
    plan_pdf_to_image_conversion,
)

__all__ = [
    "PageExportTask",
    "PdfImageConversionReport",
    "PdfToImageOptions",
    "convert_pdf_to_images",
    "execute_pdf_to_image_conversion",
    "get_pdf_page_count",
    "pdf_to_jpg",
    "pdf_to_png",
    "plan_pdf_to_image_conversion",
]
