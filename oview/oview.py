"""OView: Multi-purpose image and data viewer for Omega data files.

keyboard shortcuts:
    F1: help window
    l: open look-up-table window
    o: open image file (supported formats: hdf, tif, png, jpg, bmp)
    h: reset plot limits
    Esc: close window
""" # module docstring doubles as help window text

__author__ = "J. Ryan Rygg"
__version__ = "0.2.2"

#==============================================================================
import os.path as osp
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QDesktopWidget, QFileDialog,
        QLabel, QTextEdit,
)
from PyQt5.QtGui import QIcon

from .io import read_hdf

# lazy import: won't import until needed (gives tiny speedup on startup)
sp_misc = None # import scipy.misc

#==============================================================================
# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

FILE_PATH = osp.abspath(osp.dirname(__file__))
ICON_PATH = osp.abspath(osp.join(FILE_PATH, '..', 'oview_data', 'oview48.png'))
STARTDIR = None # TODO: allow user to specify default starting directory

#==============================================================================
class OViewMainWindow(QMainWindow):
    """OView MainWindow"""
    WINTITLE_PREFIX = "OView"

    def __init__(self, filename=None):
        QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._init_UI()
        self._init_shortcuts()
        self.show()

        if filename is None:
            self.openDialog()
        else:
            self.open(filename)

        self.lut.show()
        self.lut.move_to_parent()
        self.activateWindow() # return active window state to self

    def _init_UI(self):
        self.setWindowTitle(self.WINTITLE_PREFIX)
        self.setWindowIcon(QIcon(ICON_PATH))
        self.resize_maxrect()

        #--- pyqtgraph widgets ---
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.ci.layout.setContentsMargins(2, 2, 2, 2)
        self.setCentralWidget(self.graphics_widget)

        self.statusbar_right = QLabel("")
        self.statusBar().addPermanentWidget(self.statusbar_right)

        # A plot area (ViewBox + axes) for displaying the image
        self.plot1 = self.graphics_widget.addPlot()
        self.plot1.setAspectLocked(True)

        # Item for displaying image data
        self.img1 = pg.ImageItem()
#        self.img1.setImage(np.ones((2,2)))
        self.plot1.addItem(self.img1)

        # Contrast/color control
        self.lut = LUTWindow(self)
        self.hist = self.lut.hist

        # mouse move updates
        self.proxy = pg.SignalProxy(self.plot1.scene().sigMouseMoved,
                                    rateLimit=60, slot=self.mouse_moved)

    def _init_shortcuts(self):
        """initialize keyboard shortcuts"""
        QShortcut = QtGui.QShortcut
        Qt = QtCore.Qt

        QShortcut(Qt.Key_F1, self, self.helpDialog)
        QShortcut(Qt.Key_O, self, self.openDialog)
        QShortcut(Qt.Key_L, self, self.lutDialog)
        QShortcut(Qt.Key_1, self, self.reset_plotview)
        QShortcut(Qt.Key_H, self, self.reset_plotview)

        # add various shortcuts to close the window
        QShortcut(Qt.Key_Escape, self, self.close)
        for ks in ("Ctrl+w", "Ctrl+F4", "Ctrl+q"):
            QShortcut(QtGui.QKeySequence(ks), self, self.close)

    def resize_maxrect(self, mult=0.9, aspect=1.0):
        """resize frame to a multiple of the maximum-size square

        mult: scale vs max
        aspect: width/height aspect ratio
        """
        self.move(0, 0)
        desktop = QDesktopWidget().availableGeometry()
        frame = self.frameGeometry()

        # set size to square that is mult of shorter side of desktop
        wmax = desktop.width() + self.width() - frame.width()
        hmax = desktop.height() + self.height() - frame.height()

        if aspect * hmax < wmax: # height-limited
            d1 = mult * hmax
            self.resize(d1 * aspect, d1)
        else: # width-limited
            d1 = mult * wmax
            self.resize(d1, d1 / aspect)

    def reset_plotview(self):
        """Reset the plot view (slightly different than autoscale)."""
        if not hasattr(self, "data"):
            return
        Ny, Nx = np.shape(self.data)
        self.plot1.setXRange(0, Nx, padding=0)
        self.plot1.setYRange(0, Ny, padding=0)

    def mouse_moved(self, evt):
        """update statusbar with x,y,z of mouse event"""
        if not self.plot1.sceneBoundingRect().contains(evt[0]):
            return
        if not hasattr(self, "data",):
            return

        mousePoint = self.plot1.vb.mapSceneToView(evt[0])
        x, y = int(mousePoint.x()), int(mousePoint.y())
        Ny, Nx = np.shape(self.data)
        if (0 < y < Ny) and (0 < x < Nx):
            z = "{:.4g}".format(self.data[y, x])
        else:
            z = "n/a"
        self.statusBar().showMessage(
            "x,y,z = {:.4g}, {:.4g}, {}".format(x, y, z))

    def openDialog(self):
        """Instantiate an Open File Dialog."""
        if hasattr(self, "opendialog_called"):
            startdir = None
        else:
            startdir = STARTDIR
            self.opendialog_called = True

        title = "{}: Open File".format(self.WINTITLE_PREFIX)
        filename = QFileDialog.getOpenFileName(self, title, startdir)[0]
        if filename:
            self.open(filename)

    def open(self, filename):
        """open image file and update displayed image data"""
        base, ext = osp.splitext(filename)
        if ext.lower() == '.hdf':
            hdat = read_hdf.Hdf(filename)
            A = np.rot90(hdat.sig, 2)
        elif ext[1:].lower() in ('tif', 'tiff', 'png', 'bmp', 'jpg', 'jpeg'):
            global sp_misc
            if sp_misc is None:
                import scipy.misc as sp_misc
            A = np.flipud(sp_misc.imread(filename, flatten=True))
        else:
            self.statusBar().showMessage("{} files currently unsupported".format(ext))
            return

        # TODO: enable toggling between linear and log imagescale
        if False:
            self.logdata = np.empty_like(A)
            self.logdata[A>0] = np.log10(A[A>0])
            self.logdata[A<=0] = np.nan
            self.data = self.logdata
        else:
            self.data = self.lindata = A

        self.img1.setImage(self.data)
        self.resize_maxrect(aspect=A.shape[1]/A.shape[0])
        self.reset_plotview()
        self.lut.move_to_parent()
        self.lut.setLevels()

        self.filename = filename
        self.statusBar().showMessage("opened file: {}".format(filename))
        self.statusbar_right.setText(osp.split(filename)[1])
        self.setWindowTitle("{}: {}".format(self.WINTITLE_PREFIX, filename))

    def helpDialog(self):
        if not hasattr(self, "help_window"):
            self.help_window = HelpWindow()
            self.help_window.move(self.x() + 24, self.y() + 32)
        self.help_window.show()

    def lutDialog(self):
        """Show the LUT Window, and arrange close to self."""
        self.lut.show()
        self.lut.move_to_parent()


class LUTWindow(QMainWindow):
    """Window for lookuptable"""
    def __init__(self, parent, *args, **kws):
        QMainWindow.__init__(self, parent=parent, *args, **kws)
        self.parent_window = parent
        self.setWindowTitle("LUT")
        self.setWindowIcon(QIcon(ICON_PATH))

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphics_widget)

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.parent_window.img1)
        self.graphics_widget.addItem(self.hist)
        self.hist.gradient.loadPreset('thermal') # pyqtgraph preset gradient

        # shortcuts
        QtGui.QShortcut(QtCore.Qt.Key_L, self, self.move_to_parent)

    def move_to_parent(self):
        parent_geom = self.parent_window.frameGeometry()
        self.move(parent_geom.x() + parent_geom.width(), parent_geom.y())
        self.resize(375, self.parent_window.height())

    def setLevels(self, data=None, nsigma=3):
        """Set color histogram levels to mean value +/- nsigma of data."""
        if data is None:
            data = self.hist.imageItem().image.view(np.ndarray)

        z0 = np.nanmean(data)
        dz = np.nanstd(data)
        zmin = max(np.nanmin(data), z0 - nsigma*dz)
        zmax = min(np.nanmax(data), z0 + nsigma*dz)
        self.hist.setLevels(zmin, zmax)


class HelpWindow(QTextEdit):
    """RyView Help Window"""
    def __init__(self, *args, **kws):
        QTextEdit.__init__(self, *args, **kws)
        # add various shortcuts to close the window
        QtGui.QShortcut(QtCore.Qt.Key_Escape, self, self.close)
        for ks in ("Ctrl+w", "Ctrl+F4", "Ctrl+q"):
            QtGui.QShortcut(QtGui.QKeySequence(ks), self, self.close)

        self.setWindowTitle("{} Help".format(OViewMainWindow.WINTITLE_PREFIX))
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setText("{}\n\nversion: {}".format(__doc__, __version__))
        self.setReadOnly(True)
        self.resize(600, 400)


#==============================================================================
def main():
    app = QApplication(sys.argv)
    mywindow = OViewMainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import oview
    main()
