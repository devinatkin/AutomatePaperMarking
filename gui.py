"""Qt GUI for AutomatePaperMarking utilities.

This interface provides two workflows:
1. Stamp ArUco markers onto an input PDF (using ``stamp_aruco_corners.process_pdf``).
2. Align and merge marked PDFs (using helpers from ``align_and_merge``).

Run with:
    python gui.py

Dependencies:
    - PyQt5 (for the GUI)
    - Runtime dependencies of the CLI tools (PyMuPDF, OpenCV with ArUco, Pillow, numpy)
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Callable, Iterable

from PyQt5 import QtCore, QtWidgets

from align_and_merge import (
    align_page_images,
    convert_images_to_pdf,
    convert_pdf_to_images,
    get_aruco_corners,
    merge_aligned_page_images,
    sort_pages,
)
from stamp_aruco_corners import ARUCO_DICT_NAMES, process_pdf


class TaskWorker(QtCore.QThread):
    """Run a blocking task on a background thread."""

    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, task: Callable[[Callable[[str], None]], None]):
        super().__init__()
        self.task = task

    def run(self) -> None:  # type: ignore[override]
        try:
            self.task(self.progress.emit)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self.finished.emit(False, str(exc))
        else:
            self.finished.emit(True, "Task completed successfully.")


def stamp_task(
    input_pdf: str,
    output_pdf: str,
    marker_mm: float,
    inset_mm: float,
    dictionary_name: str,
    unique_ids: bool,
    password: str | None,
    render_px: int,
    border_bits: int,
    opacity: float,
) -> TaskWorker:
    def _run(progress: Callable[[str], None]) -> None:
        progress(f"Stamping ArUco markers on '{input_pdf}'...")
        process_pdf(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            marker_mm=marker_mm,
            inset_mm=inset_mm,
            dictionary_name=dictionary_name,
            per_page_unique_ids=unique_ids,
            password=password,
            marker_render_px=render_px,
            border_bits=border_bits,
            opacity=opacity,
        )
        progress(f"Saved stamped PDF to '{output_pdf}'.")

    return TaskWorker(_run)


def _cleanup_temp(paths: Iterable[str]) -> None:
    for path in paths:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                try:
                    os.remove(os.path.join(path, filename))
                except OSError:
                    pass
            try:
                os.rmdir(path)
            except OSError:
                pass
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def align_task(input_pdf: str, output_pdf: str, dpi: int = 300) -> TaskWorker:
    def _run(progress: Callable[[str], None]) -> None:
        if not os.path.isfile(input_pdf):
            raise FileNotFoundError(f"Input file '{input_pdf}' not found.")

        temp_dir = tempfile.mkdtemp(prefix="apm_gui_")
        merged_image_paths: list[str] = []

        try:
            progress("Converting PDF to images...")
            images = convert_pdf_to_images(input_pdf, temp_dir, dpi=dpi)
            progress(f"Converted {len(images)} pages.")

            corner_image_list = []
            for idx, image in enumerate(images, start=1):
                corners, ids = get_aruco_corners(image)
                if corners is not None and ids is not None:
                    progress(f"Detected ArUco markers on page image {idx}.")
                    corner_image_list.append((image, corners, ids))
                else:
                    progress(f"No ArUco markers detected on page image {idx}; skipping.")

            sorted_pages = sort_pages(corner_image_list)
            if not sorted_pages:
                raise RuntimeError("No alignable pages detected. Ensure stamped markers are present.")

            aligned_pages = []
            for page_index, page in enumerate(sorted_pages, start=1):
                progress(f"Aligning page group {page_index} with {len(page)} images...")
                aligned_pages.append(align_page_images(page))

            for page_index, page_images in enumerate(aligned_pages, start=1):
                progress(f"Merging aligned images for page {page_index}...")
                merged_image = merge_aligned_page_images(page_images)
                merged_image_path = os.path.join(temp_dir, f"merged_page_{page_index}.jpg")
                merged_image.save(merged_image_path, "JPEG", quality=95, optimize=True)
                merged_image_paths.append(merged_image_path)

            progress("Assembling merged PDF...")
            convert_images_to_pdf(merged_image_paths, output_pdf)
            progress(f"Saved merged PDF to '{output_pdf}'.")
        finally:
            _cleanup_temp([temp_dir])

    return TaskWorker(_run)


class MainWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Automate Paper Marking GUI")
        self.resize(700, 550)
        self._worker: TaskWorker | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        layout.addWidget(self.log)

        self.tabs.addTab(self._build_stamp_tab(), "Stamp ArUco Corners")
        self.tabs.addTab(self._build_align_tab(), "Align and Merge")

    # ---- UI builders ----
    def _build_stamp_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.stamp_input = QtWidgets.QLineEdit()
        self.stamp_output = QtWidgets.QLineEdit()
        self._add_file_row(form, "Input PDF", self.stamp_input, select_output=False)
        self._add_file_row(form, "Output PDF", self.stamp_output, select_output=True)

        self.marker_mm = QtWidgets.QDoubleSpinBox()
        self.marker_mm.setRange(1.0, 100.0)
        self.marker_mm.setValue(8.0)
        form.addRow("Marker size (mm)", self.marker_mm)

        self.inset_mm = QtWidgets.QDoubleSpinBox()
        self.inset_mm.setRange(0.0, 100.0)
        self.inset_mm.setValue(6.0)
        form.addRow("Inset from edge (mm)", self.inset_mm)

        self.dictionary = QtWidgets.QComboBox()
        self.dictionary.addItems(sorted(ARUCO_DICT_NAMES.keys()))
        self.dictionary.setCurrentText("4X4_1000")
        form.addRow("Dictionary", self.dictionary)

        self.unique_ids = QtWidgets.QCheckBox("Use unique IDs per page")
        self.unique_ids.setChecked(True)
        form.addRow("Unique IDs", self.unique_ids)

        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        form.addRow("PDF password", self.password)

        self.render_px = QtWidgets.QSpinBox()
        self.render_px.setRange(64, 2000)
        self.render_px.setValue(300)
        form.addRow("Render size (px)", self.render_px)

        self.border_bits = QtWidgets.QSpinBox()
        self.border_bits.setRange(0, 10)
        self.border_bits.setValue(1)
        form.addRow("Border bits", self.border_bits)

        self.opacity = QtWidgets.QDoubleSpinBox()
        self.opacity.setDecimals(2)
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(1.0)
        form.addRow("Opacity", self.opacity)

        self.stamp_button = QtWidgets.QPushButton("Run Stamping")
        self.stamp_button.clicked.connect(self._start_stamp)
        form.addRow(self.stamp_button)

        return widget

    def _build_align_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self.align_input = QtWidgets.QLineEdit()
        self.align_output = QtWidgets.QLineEdit()
        self._add_file_row(form, "Input PDF", self.align_input, select_output=False)
        self._add_file_row(form, "Output PDF", self.align_output, select_output=True)

        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        form.addRow("Render DPI", self.dpi_spin)

        self.align_button = QtWidgets.QPushButton("Align and Merge")
        self.align_button.clicked.connect(self._start_align)
        form.addRow(self.align_button)

        return widget

    def _add_file_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        line_edit: QtWidgets.QLineEdit,
        select_output: bool,
    ) -> None:
        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)

        btn = QtWidgets.QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_file(line_edit, select_output))
        hbox.addWidget(btn)

        form.addRow(label, container)

    # ---- Task handling ----
    def _start_stamp(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        input_pdf = self.stamp_input.text().strip()
        output_pdf = self.stamp_output.text().strip()
        if not input_pdf or not output_pdf:
            self._log("Please specify both input and output PDF paths for stamping.")
            return

        worker = stamp_task(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            marker_mm=self.marker_mm.value(),
            inset_mm=self.inset_mm.value(),
            dictionary_name=self.dictionary.currentText(),
            unique_ids=self.unique_ids.isChecked(),
            password=self.password.text().strip() or None,
            render_px=self.render_px.value(),
            border_bits=self.border_bits.value(),
            opacity=self.opacity.value(),
        )
        self._start_worker(worker)

    def _start_align(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        input_pdf = self.align_input.text().strip()
        output_pdf = self.align_output.text().strip()
        if not input_pdf or not output_pdf:
            self._log("Please specify both input and output PDF paths for alignment.")
            return

        worker = align_task(input_pdf=input_pdf, output_pdf=output_pdf, dpi=self.dpi_spin.value())
        self._start_worker(worker)

    def _start_worker(self, worker: TaskWorker) -> None:
        self._worker = worker
        worker.progress.connect(self._log)
        worker.finished.connect(self._task_finished)
        self._set_buttons_enabled(False)
        worker.start()

    def _task_finished(self, success: bool, message: str) -> None:
        self._log(message)
        self._set_buttons_enabled(True)
        self._worker = None

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.stamp_button.setEnabled(enabled)
        self.align_button.setEnabled(enabled)

    # ---- Helpers ----
    def _browse_file(self, line_edit: QtWidgets.QLineEdit, select_output: bool) -> None:
        if select_output:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output PDF", filter="PDF Files (*.pdf)")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input PDF", filter="PDF Files (*.pdf)")
        if path:
            line_edit.setText(path)

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
