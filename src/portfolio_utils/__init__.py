"""Public API for portfolio utility helpers.

The package currently exposes PDF-to-image conversion helpers through both the
legacy CamelCase module and the newer snake-case wrapper.
"""

from .pdf_image_converter import (
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
