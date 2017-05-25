"""
Read and parse hdf4 files produced at Omega-60 and Omega-EP.

author: J. R. Rygg
history:
    2017-05-19 basic support for p510_data summary hdf
    2017-04-13 created
"""

import os.path as osp
import sys

import numpy as np
import pyhdf.SD as sd

# lazy import: won't import until needed (gives tiny speedup on startup)
plt = None # import matplotlib.pyplot as plt


class Hdf():
    """Hdf file data from Omega-60 and Omega-EP data sources."""
    KNOWN_DATA_SOURCES = ('p510', 'asbo', 'sop', 'pxrdip') # known hdf file sources

    PXRDIP_SCAN_ORDER_DEFAULT = ('U', 'L', 'D', 'R', 'B') # default pxrdip scan order
    PXRDIP_DISPLAY_ORDER = { # display order for PXRDIP plates on omega-60 or ep
        'omega-60': ('R', 'D', 'L', 'U', 'B'),
        'omega-ep': ('L', 'U', 'R', 'D', 'B'),
    }

    def __init__(self, handle, pathlist=None, hint=None):
        self.handle = handle # filename or (TODO: implement) other keyword
        self.pathlist = pathlist # list of possible paths
        self.hint = hint # hint to preferred filename

        self.filename = self.handle#self.find_filename(self.handle, pathlist)
        self.title, self.fileext = osp.splitext(osp.split(self.filename)[1])

        self.facility = 'omega-ep' # TODO: omega-60 or NIF or other or None
        self.attributes = None
        self.datasets = None

        self.datasource = self.determine_datasource()
        self.read_file()

    def determine_datasource(self):
        """Determine data type of given file based on filename."""
        filename = self.filename.lower()
        for datasource in self.KNOWN_DATA_SOURCES:
            if datasource in filename:
                return datasource

    def read_file(self):
        """Read datafile at self.filename."""
        if self.fileext == '.hdf':
            filedata = sd.SD(self.filename, sd.SDC.READ)
            self.parse_filedata(filedata)
            print_hdf_info(filedata)
            filedata.end() # close hdf file

    def parse_filedata(self, filedata):
        self.attributes = filedata.attributes()
        self.datasets = filedata.datasets()

        if self.fileext == '.hdf' and self.datasource in self.KNOWN_DATA_SOURCES:
            parse_method = getattr(self, "parse_" + self.datasource)
            parse_method(filedata)

    def parse_p510(self, filedata):
        """Parse Omega p510 summary file (laser power results)."""
        self.shotnum = self.attributes['LOG_NUM']

        self.time = filedata.select('SUMMARY_DATA_TIME').get() # time [ns]
        self.power = filedata.select('SUMMARY_DATA_AVERAGE')
        self.beam_power = filedata.select('SUMMARY_DATA_UV').get()
#        self.beams = self.power.attributes()['BEAMS'] # list of beams
        self.power = self.power.get() # per-beam powers [TW]

    def parse_asbo(self, filedata):
        """Parse Omega ASBO hdf (Active Shock BreakOut, = Omega VISAR)."""
        dat = filedata.select("Streak_array")
        ref = filedata.select("Reference")
        att = self.attributes
        self.shotnum = ""
        self.location = att['Location']
        self.unit = att['Unit']

        att_d = dat.attributes()
        self.sweep_setting = att_d['SweepSpeed']
        self.timestamp = att_d['TimeStamp']

        A = dat.get().astype(float)
        self.sig = np.rot90(A[0] - A[1], 2)
        self.ref = np.rot90(ref.get().astype(float) - A[1], 2)
        self.bg = np.rot90(A[1], 2)

    def parse_sop(self, filedata):
        """Parse Omega SOP hdf (SOP = streaked optical pyrometer)."""
        dat = filedata.select("Streak_array")
        att = self.attributes
        self.shotnum = ""
        self.location = att['Location']
        self.unit = att['Unit']

        att_d = dat.attributes()
        self.sweep_setting = att_d['SweepSpeed']
        self.timestamp = att_d['TimeStamp']

        A = dat.get().astype(float)
        self.sig = np.rot90(A[0] - A[1], 2)
        self.bg = np.rot90(A[1], 2)

    def parse_pxrdip(self, filedata):
        """Parse PXRDIP hdf (Powder X-Ray Diffraction Image Plate)."""
        psl = filedata.select("PSL_per_px")
        psl_attr = psl.attributes()
        self.scan_img = filedata.select("PSL_per_px").get()
        self.shotnum = str(int(psl_attr['shotNum']))
        self.instrument = psl_attr['instrument'] # scanner
        self.scandelay = psl_attr['scanDelaySeconds']
        self.latitude = psl_attr['latitude']
        self.scale = psl_attr['pixelSizeX']
        self.scale_units = 'um per pixel'
        if psl_attr['pixelSizeY'] != self.scale:
            sys.stdout.write("anisotropic IP scan\n")
            print(self.scale, psl_attr['pixelSizeX'], psl_attr['pixelSizeY'])

        self.ip_scan_order = self.PXRDIP_SCAN_ORDER_DEFAULT
        self.ip_display_order = self.PXRDIP_DISPLAY_ORDER[self.facility]
        self.ijs = self.clip_pxrdip_plates()
        self.shuffle_pxrdip_plates(self.ijs)

    def clip_pxrdip_plates(self, di=40):
        """Return slices for different image plates.

        di = range in pixels over which to search for boundaries
        """
        scale = self.scale / 1000 # convert to mm
        L = 75 # mm approximate pxrdip ip length
        W = 50 # mm approximate pxrdip ip width

        yy = self.scan_img.mean(1) + 1e-3 # vertical lineout
        step = np.array((-1,-1,-1,1,1,1)) # step kernel
        j1 = np.argmax(np.convolve(yy, step, mode='same') / yy)
        jslice = slice(0, j1+5) # "vertical" slice

        xx = self.scan_img.mean(0) + 1e-3 # horizontal lineout
        i0 = int(L / scale) * np.array((1, 2, 3, 4, 5), dtype=int)
        i0[-1] -= int((L - W) / scale)
        xconv = np.convolve(xx, step, mode='same') / xx
        i1 = []
        for i in i0:
            imin = np.argmin(xconv[i-di:i+di])
            i1.append(i-di + np.argmax(xconv[i-di:i-di+imin]))

        ijs = [(jslice, slice(0, i1[0]))]
        ijs.extend([(jslice, slice(i1[k-1],i1[k])) for k in range(1,len(i1))])

        return ijs

    def shuffle_pxrdip_plates(self, ijs, pattern='all'):
        """Shuffle the pxrdip plates into desired order."""
        self.sig = B = np.zeros((2000, 1250))# + np.nan
        display_lookup = {ip: k for k, ip in enumerate(self.ip_display_order)}

        for k, ij in enumerate(ijs):
            Ak = self.scan_img[ij]
            Ny, Nx = np.shape(Ak)
            k2 = display_lookup[self.ip_scan_order[k]]
            if k < len(ijs) - 1:
                B[500*k2:500*k2+Ny,0:Nx] = Ak
            else:
                B[1500:1500+Ny,-Nx:] = Ak

    def imshow(self, **kws):
        """Generate matplotlib figure of image data."""
        global plt
        if plt is None:
            import matplotlib.pyplot as plt

        if self.datatype == 'pxrdip':
            figsize = (3.75, 6)
        else:
            figsize = (6, 6)

        A = self.sig
        kws['origin'] = 'lower'
        if 'cmap' not in kws:
            kws['cmap'] = 'viridis'
        if 'vmin' not in kws and 'vmax' not in kws:
            z0 = np.mean(A)
            dz = np.std(A)
            kws['vmin'] = max(A.min(), z0 - 3*dz)
            kws['vmax'] = min(A.max(), z0 + 3*dz)
            kws['vmin'], kws['vmax'] = thresh_vlim(A, thresh=0.05, Nbin=2000)

        fig = plt.figure(figsize=figsize, dpi=250)
        fig.canvas.window().move(0, 0)
        ax = fig.add_axes((0,0,1,1))
        im = ax.imshow(np.rot90(A,2), **kws)

        if kws['cmap'] == 'viridis':
            textcolor = 'w'
        else:
            textcolor = 'k'

        if self.datatype == 'pxrdip':
            fig.text(0.02, 0.985, "PXRDIP - shot {}".format(self.shotnum),
                     ha='left', va='top', color=textcolor)
            cax = fig.add_axes((0.2, 0.51, 0.03, 0.24))
            fig.colorbar(im, cax=cax)
            cax.set_ylabel("signal (PSL)", color=textcolor)
            for item in cax.get_yticklabels():
                item.set_color(textcolor)
                item.set_fontsize('small')
        elif self.datatype == 'asbo':
            if kws['cmap'] == 'viridis':
                color = 'w'
            else:
                color = 'k'
            fig.text(0.02, 0.985, "{}".format(self.unit.split()[0]),
                     ha='left', va='top', color=color)
            cax = fig.add_axes((0.4, 0.97, 0.57, 0.02))
            fig.colorbar(im, cax=cax, orientation='horizontal')

            for item in cax.get_xticklabels():
                item.set_color(textcolor)
                item.set_fontsize('small')


def thresh_vlim(data, thresh=0.01, Nbin=200):
    """Calculate colormap vlim based on a histogram threshold.
    Excludes fraction on top and bottom given by scalar or 2-tuple thresh.
    """
    thresh = thresh*np.ones(2) if np.size(thresh) < 2 else thresh
    H, bin_edges = np.histogram(data[~np.isnan(data)], Nbin)
    vmin, vmax = bin_edges[0], bin_edges[-1]
    csH = H.cumsum() / np.nansum(H) # fractional cumulative histogram sum
    ilo = np.where(csH < thresh[0])[0]
    ihi = np.where(csH > 1.0 - thresh[1])[0]
    if len(ilo) > 0: vmin = bin_edges[np.nanmax(ilo) + 1]
    if len(ihi) > 0: vmax = bin_edges[np.nanmin(ihi) + 1]
    return vmin, vmax


def print_hdf_info(h):
    sow = sys.stdout.write
    sow("Number of (datasets, attributes) = {}\n".format(h.info()))
    print(h.datasets())
    print(h.attributes())
    for d in h.datasets():
        sow("---- {} ----\n".format(d))
        d = h.select(d)
        sow(str(d.info()))
        sow('\n')
        att = d.attributes()
        for k in sorted(att.keys()):
            sow("{}\t{}\n".format(k, att[k]))

