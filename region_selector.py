import sys
import threading
from PySide6.QtCore import QRect, Qt, Signal, QEventLoop
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QApplication, QWidget


class RegionSelectionOverlay(QWidget):
    """Fullscreen transparent overlay for drawing a selection rectangle."""

    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self, screen_geometry):
        super().__init__()
        self._start = None
        self._current = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(screen_geometry)

    def showEvent(self, event):
        super().showEvent(event)
        self.grabMouse()
        self.grabKeyboard()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.globalPosition().toPoint()
            self._current = self._start
            self.update()

    def mouseMoveEvent(self, event):
        if self._start:
            self._current = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._start:
            self._current = event.globalPosition().toPoint()
            rect = QRect(self._start, self._current).normalized()
            if rect.width() >= 10 and rect.height() >= 10:
                self.selected.emit(rect)
            else:
                self.cancelled.emit()
            self.releaseMouse()
            self.releaseKeyboard()
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.releaseMouse()
            self.releaseKeyboard()
            self.close()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self._start and self._current:
            selected = QRect(self._start, self._current).normalized()
            geo = self.geometry()
            local = selected.translated(-geo.x(), -geo.y())

            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(local, Qt.GlobalColor.transparent)

            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(74, 222, 128), 2))
            painter.drawRect(local.adjusted(0, 0, -1, -1))

            painter.setPen(QColor(74, 222, 128))
            font = QFont("Consolas", 11)
            painter.setFont(font)
            label = f"{selected.width()} x {selected.height()}"
            painter.drawText(local.x() + 6, local.y() - 6, label)
        else:
            painter.setPen(QColor(255, 255, 255, 180))
            font = QFont("Segoe UI", 14)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Drag to select a screen region\nPress ESC to cancel")

        painter.end()


def _ensure_app():
    """Get or create QApplication (must be called from main thread)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_result_lock = threading.Lock()
_result = None
_completed = threading.Event()


def _on_selected(rect):
    global _result
    with _result_lock:
        _result = (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
    _completed.set()


def _on_cancelled():
    global _result
    with _result_lock:
        _result = None
    _completed.set()


def select_region():
    """
    Launch overlay and return (left, top, right, bottom) or None.
    MUST be called from the main thread.
    """
    global _result
    _result = None
    _completed.clear()

    app = _ensure_app()
    screen_geo = app.primaryScreen().geometry()
    overlay = RegionSelectionOverlay(screen_geo)
    overlay.selected.connect(_on_selected)
    overlay.cancelled.connect(_on_cancelled)
    overlay.show()

    # Process events until selection is complete
    while not _completed.is_set():
        app.processEvents()
        _completed.wait(timeout=0.01)

    return _result
