from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe

import astropy
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc

import myutils
import scipy.stats
import pandas as pd
import scipy.interpolate
import scipy.ndimage.filters as filters
import triangle

#plt.style.use('talk')
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'Dosis'

red = '#d62728'
blue = '#1f77b4'
lblue = '#4ca3e0'
dblue = '#0c1344'

#red = 'darkorange'
#red = 'orangered'
#lblue = 'cornflowerblue'
#dblue = 'midnightblue'

class Survey():
    def __init__(self, name, qrel=1, qrv=50, nonnegative=True, crtheta=90, observed=True, vdisk=220*u.km/u.s, sdisk=180*u.km/u.s, label=''):
        self.name = name
        self._data = Table.read('/home/ana/projects/mrich_halo/data/{}_abb.fits'.format(self.name))
        self.vdisk = vdisk
        self.sdisk = sdisk
        if len(label)==0:
            label = name
        self.label = label
        
        self.quality_cut(qrel=qrel, qrv=qrv, nonnegative=nonnegative, observed=observed)
        self.configuration_coords(observed=observed)
        self.phase_coords()
        self.define_counterrot(crtheta)
        self.toomre(vdisk, sdisk)
        
    def quality_cut(self, qrel=1, qrv=50, nonnegative=True, observed=True):
        if observed:
            quality = ((np.abs(self._data['pmra_error']/self._data['pmra'])<qrel) 
                & (np.abs(self._data['pmdec_error']/self._data['pmdec'])<qrel) 
                & (np.abs(self._data['hrv_error'])<qrv)
                & (np.abs(self._data['parallax_error']/self._data['parallax'])<qrel))
            if nonnegative:
                quality = quality & (self._data['parallax']>0) & np.isfinite(self._data['pmra'])
            self.data = self._data[quality]
        else:
            self.data = self._data
    
    def configuration_coords(self, observed=True):
        if observed:
            c = coord.SkyCoord(ra=np.array(self.data['ra'])*u.deg, dec=np.array(self.data['dec'])*u.deg, distance=1/np.array(self.data['parallax'])*u.kpc)
            cgal = c.transform_to(coord.Galactocentric) 

            self.x = np.transpose([cgal.x, cgal.y, cgal.z])*u.kpc
            self.v = np.transpose(gc.vhel_to_gal(c.icrs, rv=self.data['hrv']*u.km/u.s, pm=[np.array(self.data['pmra']), np.array(self.data['pmdec'])]*u.mas/u.yr))
        else:
            self.x = -self.data['x']*u.kpc
            self.v = self.data['v']*u.km/u.s
    
    def phase_coords(self):
        self.Ek = 0.5*np.linalg.norm(self.v, axis=1)**2
        self.vtot = np.linalg.norm(self.v, axis=1)

        self.L = np.cross(self.x, self.v, axis=1)
        self.L2 = np.linalg.norm(self.L, axis=1)
        self.Lperp = np.sqrt(self.L[:,0]**2 + self.L[:,1]**2)
        self.ltheta = np.degrees(np.arccos(self.L[:,2]/self.L2))
        self.lx = np.degrees(np.arccos(self.L[:,0]/self.L2))
        self.ly = np.degrees(np.arccos(self.L[:,1]/self.L2))
    
    def define_counterrot(self, crtheta):
        self.crtheta = crtheta
        self.counter_rotating = self.ltheta < self.crtheta
        self.rotating = self.ltheta > 180 - self.crtheta
    
    def toomre(self, vdisk, sdisk):
        self.vxz = np.sqrt(self.v[:,0]**2 + self.v[:,2]**2)
        self.vy = self.v[:,1]
        
        self.halo = np.sqrt(self.vxz**2 + (self.vy - vdisk)**2)>sdisk
        self.disk = ~self.halo

def load_survey(survey):
    """"""
    if survey=='raveon':
        s = Survey('raveon', label='RAVE-on', sdisk=220*u.km/u.s, qrel=1, qrv=20)
    elif survey=='apogee':
        s = Survey('apogee', label='APOGEE', sdisk=220*u.km/u.s, qrel=1, qrv=20)
    elif survey=='lattemdif':
        s = Survey('lattemdif', label='Latte', observed=False, vdisk=250*u.km/u.s, sdisk=220*u.km/u.s)
    else:
        s = Survey(survey, label=survey, observed=False, vdisk=250*u.km/u.s, sdisk=220*u.km/u.s)
        
    return s


def toomre(survey='raveon'):
    """"""
    mpl.rcParams['axes.linewidth'] = 1.5
    
    s = load_survey(survey)

    bx = np.arange(-400,401,10)
    by = np.arange(0,401,10)
    xc = myutils.bincen(bx)
    yc = myutils.bincen(by)
    xg, yg = np.meshgrid(xc, yc)
    xg = xg.transpose()
    yg = yg.transpose()
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,3.5))

    # number density
    plt.sca(ax[0])
    cnt, xe, ye, im = plt.hist2d(s.vy, s.vxz, bins=(bx, by), cmap='bone_r', norm=mpl.colors.LogNorm(), vmin=1e-1, vmax=10000)
    
    vh_y = np.linspace(0,400,400)*u.km/u.s
    vh_xz = np.sqrt(s.sdisk**2 - (vh_y - s.vdisk)**2)
    plt.plot(vh_y, vh_xz, 'k-', lw=2)
    
    plt.xlim(-400,400)
    plt.ylim(0,400)
    plt.xlabel('$V_Y$ (km/s)')
    plt.ylabel('$V_{XZ}$ (km/s)')
        
    t = plt.text(0.98, 0.1, 'Disk', transform=ax[0].transAxes, ha='right', fontsize='medium')
    t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    t = plt.text(0.98, 0.6, 'Halo', transform=ax[0].transAxes, ha='right', fontsize='medium')
    t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=10**np.arange(-1,4.1,1))
    plt.ylabel('Density (km/s)$^{-2}$')
    
    # metallicity
    plt.sca(ax[1])
    finite = np.isfinite(s.data['feh'])
    
    im = plt.scatter(s.vy[s.halo & finite], s.vxz[s.halo & finite], c=s.data['feh'][s.halo & finite], s=20, cmap='magma', vmin=-2.5, vmax=0.5, edgecolors='none', rasterized=True)
    
    feh, xe, ye, nb = scipy.stats.binned_statistic_2d(s.vy[finite], s.vxz[finite], s.data['feh'][finite], statistic=np.mean, bins=(bx, by))
    
    halo_mask = np.zeros(np.shape(xg), dtype=bool)
    ind = yg**2 + (xg-220)**2 < 220**2
    halo_mask[ind] = True
    feh[~halo_mask] = np.nan
    
    data = np.ma.masked_invalid(feh)
    im = plt.imshow(data.filled(data.mean()).T, origin='lower', vmin=-2.5, vmax=0.5, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect='auto', interpolation='gaussian', cmap='magma', rasterized=True)
    bad_data = np.ma.masked_where(~data.mask, data.mask)
    plt.imshow(bad_data.T, origin='lower', vmin=2, vmax=3, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect='auto', interpolation='none', cmap=mpl.cm.gray_r)
    
    vh_y = np.linspace(0,400,400)*u.km/u.s
    vh_xz = np.sqrt(s.sdisk**2 - (vh_y - s.vdisk)**2)
    plt.plot(vh_y, vh_xz, 'k-', lw=2)
    
    plt.xlim(-400,400)
    plt.ylim(0,400)
    plt.xlabel('$V_Y$ (km/s)')
    plt.ylabel('$V_{XZ}$ (km/s)')
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=np.arange(-2.5,0.51,0.5))
    plt.ylabel('[Fe/H]')
        
    plt.tight_layout()

    #pos = plt.gca().get_position()
    #cax = plt.axes([0.97,pos.y0,0.02,pos.y1 - pos.y0])
    #plt.colorbar(im, cax=cax, ticks=np.arange(-2.5,0.51,0.5))
    #plt.ylabel('[Fe/H]')

    plt.savefig('../plots/paper/toomre.pdf', bbox_inches='tight')
    
    mpl.rcParams['axes.linewidth'] = 2
    
def mdf():
    """"""
    
    raveon = load_survey('raveon')
    apogee = load_survey('apogee')
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(5, 8))
    bx = np.arange(-2.5,0.5,0.2)

    for i, s in enumerate([raveon, apogee]):
        plt.sca(ax[i])
        finite = np.isfinite(s.data['feh'])
        
        plt.hist(s.data['feh'][finite & s.disk], bins=bx, histtype='stepfilled', color=red, normed=True, lw=0, alpha=0.8, label='Disk')
        plt.hist(s.data['feh'][finite & s.halo], bins=bx, histtype='stepfilled', color=blue, normed=True, lw=0, alpha=0.8, label='Halo')
        print(s.label, np.median(s.data['feh'][finite & s.halo]), np.sum(s.data['feh'][finite & s.halo]>-1)/np.sum(finite & s.halo))
        
        plt.axvline(-1, ls='--', lw=2, color='0.2')
        plt.xlabel('[Fe/H]')
        plt.ylabel('Probability density (dex$^{-1}$)')
        #plt.title(s.label)
        plt.text(0.1,0.85, s.label, transform=ax[i].transAxes)
        if i==0:
            plt.legend(frameon=False, loc=(0.1,0.55), borderpad=0)

    plt.tight_layout(h_pad=1)
    plt.savefig('../plots/paper/mdf.pdf', bbox_inches='tight')

def afeh():
    """"""
    
    raveon = load_survey('raveon')
    apogee = load_survey('apogee')
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(5, 8))
    bx = np.linspace(-2.5,0.5,100)
    by = np.linspace(-0.1,0.4,100)

    for i, s in enumerate([raveon, apogee]):
        plt.sca(ax[i])

        finite = np.isfinite(s.data['feh'])
        plt.plot(-3,0, 's', ms=6, color=mpl.cm.Reds(0.5), label='Disk', mec='none')
        plt.hist2d(s.data['feh'][finite & s.disk], s.data['afe'][finite & s.disk], bins=(bx,by), cmap='Reds', norm=mpl.colors.LogNorm(), rasterized=True)

        plt.plot(s.data['feh'][finite & s.halo], s.data['afe'][finite & s.halo], 'o', color=blue, mec='none', label='Halo', ms=5, rasterized=False)
        
        plt.xlim(-2.5,0.5)
        plt.ylim(-0.1,0.4)

        plt.xlabel('[Fe/H]')
        plt.ylabel('[$\\alpha$/Fe]')
        #plt.title(s.label, fontsize=24)
        if i==0:
            plt.legend(loc=3, frameon=False, numpoints=1, handlelength=0.2)
            #plt.plot(0.1, 0.9, 'ko', ms=4, label='', transform=ax[i].transAxes)
            rfeherr = np.median(s.data['E_FE_H'][np.isfinite(s.data['E_FE_H'])])
            rafeerr = np.median([np.median(s.data['E_CA_H'][np.isfinite(s.data['E_CA_H'])]), np.median(s.data['E_MG_H'][np.isfinite(s.data['E_MG_H'])]), np.median(s.data['E_SI_H'][np.isfinite(s.data['E_SI_H'])])])
            plt.errorbar(0.1, 0.9, yerr=rafeerr, xerr=rfeherr, label='', transform=ax[i].transAxes, fmt='none', ecolor='k')
        else:
            #plt.plot(0.1, 0.1, 'ko', ms=4, label='', transform=ax[i].transAxes)
            plt.errorbar(0.1, 0.1, yerr=0.05, xerr=0.05, label='', transform=ax[i].transAxes, fmt='none', ecolor='k')
        
        plt.text(0.93,0.85, s.label, transform=ax[i].transAxes, ha='right')

    plt.tight_layout(h_pad=0.5)
    plt.savefig('../plots/paper/afeh.pdf', bbox_inches='tight')

def ltheta():
    """"""
    raveon = load_survey('raveon')
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(5.5,5))
    bx = np.linspace(0,180,10)
    flag_norm = True
    fehacc = -1

    finite = np.isfinite(raveon.data['feh'])
    accreted = raveon.data['feh']<=fehacc
    
    plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color=red, label='Disk', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.7)
    plt.hist(raveon.ltheta[raveon.halo & finite & ~accreted], bins=bx, color=lblue, label='Halo: $[Fe/H]>-1$', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.7)
    plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color=dblue, label='Halo: $[Fe/H]\leq-1$', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.7)
    
    plt.axvline(90, ls='--', color='0.2', lw=2)
    ax.set_xticks(np.arange(0,181,45))
    
    plt.xlim(0, 180)
    plt.ylim(1e-3, 0.1)
    ax.set_yscale('log')
    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('Probability density (deg$^{-1}$)')

    #leg = plt.legend(loc=2, frameon=False, fontsize='small')
    leg = plt.legend(loc=2, frameon=True, fontsize='small', framealpha=0.95, edgecolor='w')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    #leg.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    
    label_pad = 0.04
    plt.text(label_pad, 1.04, 'Retrograde', fontsize='small', transform=ax.transAxes, ha='left', va='center')
    plt.text(1-label_pad, 1.04, 'Prograde', fontsize='small', transform=ax.transAxes, ha='right', va='center')
    
    ax.set_axisbelow(False)

    plt.tight_layout()
    plt.savefig('../plots/paper/ltheta.pdf', bbox_inches='tight')

def ltheta_empty():
    """"""
    raveon = load_survey('raveon')
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    bx = np.linspace(0,180,10)
    flag_norm = True
    fehacc = -1

    finite = np.isfinite(raveon.data['feh'])
    accreted = raveon.data['feh']<=fehacc
    
    #N = 20
    #lw = 1
    #fancy_histogram(raveon.ltheta[raveon.disk & finite], bx, color=red, label='Disk', dx=(0.15,0.995), log=(False,True), normed=flag_norm, zorder=4, N=N, lw=lw)
    #fancy_histogram(raveon.ltheta[raveon.halo & finite & ~accreted], bx, color=lblue, label='Halo: $[Fe/H]>-1$', dx=(0.15,0.995), log=(False,True), normed=flag_norm, zorder=2, N=N, lw=lw)
    #fancy_histogram(raveon.ltheta[raveon.halo & finite & accreted], bx, color=dblue, label='Halo: $[Fe/H]\leq-1$', dx=(0.15,0.995), log=(False,True), normed=flag_norm, zorder=0, N=N, lw=lw)
    
    plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color=red, label='', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.2)
    plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color='orange', label='', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.2)
    plt.hist(raveon.ltheta[raveon.halo & finite & ~accreted], bins=bx, color=lblue, label='', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.4)
    plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color=dblue, label='', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.2)
    plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color=lblue, label='', normed=flag_norm, lw=0, histtype='stepfilled', alpha=0.2)
    
    plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color=red, label='Disk', normed=flag_norm, lw=3, histtype='step', alpha=0.6)
    plt.hist(raveon.ltheta[raveon.halo & finite & ~accreted], bins=bx, color=lblue, label='Halo: $[Fe/H]>-1$', normed=flag_norm, lw=3, histtype='step', alpha=0.6)
    plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color=dblue, label='Halo: $[Fe/H]\leq-1$', normed=flag_norm, lw=3, histtype='step', alpha=0.6)

    #awhite = 0.7
    
    #plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color=red, label='Disk', normed=flag_norm, lw=4, histtype='step')
    #plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color='w', alpha=awhite, label='', normed=flag_norm, lw=1., histtype='step')
    ##plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, color=red, alpha=0.3, label='', normed=flag_norm, lw=1., histtype='step')
    
    #plt.hist(raveon.ltheta[raveon.halo & finite & ~accreted], bins=bx, color=lblue, label='Halo: $[Fe/H]>-1$', normed=flag_norm, lw=4, histtype='step')
    #plt.hist(raveon.ltheta[raveon.halo & finite & ~accreted], bins=bx, color='w', alpha=awhite, label='', normed=flag_norm, lw=1., histtype='step')
    ##plt.hist(raveon.ltheta[raveon.halo & finite & ~accreted], bins=bx, color=lblue, alpha=0.3, label='', normed=flag_norm, lw=1., histtype='step')
    
    #plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color=dblue, label='Halo: $[Fe/H]\leq-1$', normed=flag_norm, lw=4, histtype='step')
    #plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color='w', alpha=awhite, label='', normed=flag_norm, lw=1., histtype='step')
    ##plt.hist(raveon.ltheta[raveon.halo & finite & accreted], bins=bx, color=dblue, alpha=0.3, label='', normed=flag_norm, lw=1., histtype='step')
    
    
    plt.xlim(0, 180)
    plt.ylim(1e-3, 0.1)
    ax.set_yscale('log')
    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('Probability density (deg$^{-1}$)')
    leg = plt.legend(loc=2, frameon=False, fontsize='small')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    
    ax.set_axisbelow(False)

    plt.tight_layout()
    #plt.savefig('../plots/paper/ltheta.pdf', bbox_inches='tight')

def fancy_histogram(x, bx, color='k', normed=False, label='', ax=None, N=20, dx=(0.15, 0.15), log=(False, False), zorder=0, lw=1):
    """Plot fancy histogram with inner shading"""
    
    if ax==None:
        ax = plt.gca()
    plt.sca(ax)
    
    #print(x)
    n, bins, patches = plt.hist(x, color=color, histtype='step', bins=bx, lw=lw, normed=normed, label=label, zorder=zorder+1)
    
    vertices_ = np.copy(patches[0].get_xy())
    poly0 = mpl.patches.Polygon(vertices_)
    Nv = np.shape(vertices_)[0]
    vertices0 = np.empty(np.shape(vertices_))
    vertices0[0] = vertices_[0]
    for i in range(1,Nv):
        vertices0[i] = vertices_[-i]
    
    vertices = np.copy(patches[0].get_xy())
    
    direction_ = np.zeros(Nv)
    direction_[1:] = np.sign(vertices[1:,1] - vertices[:-1,1])
    iszero = (direction_[:-1]==0) & (direction_[1:]>0)
    direction_[iszero] += 1
    iszero = (direction_[:-1]==0) & (direction_[1:]<0)
    direction_[iszero] -= 1
    direction_[1] = 0
    direction_[Nv-2] = 0
    
    direction = (direction_, np.ones(Nv))
    
    for i in range(N):
        for j in range(2):
            if log[j]:
                vertices[:,j] *= dx[j]*direction[j]
            else:
                vertices[:,j] += dx[j]*direction[j]

        filled01 = [vertices0[:].tolist() + vertices[:].tolist()]
        
        kinds1 = [2]*(Nv)
        kinds1[0] = 1
        kinds01 = [kinds1 + kinds1]

        cs = mpl.contour.ContourSet(ax, [0, 1], [filled01], [kinds01], filled=True, colors=color, alpha=1/N, zorder=zorder+1)
    
    if N>0:
        cs = mpl.contour.ContourSet(ax, [0, 1], [filled01], [kinds01], filled=True, colors='w', alpha=1, zorder=zorder)

def hole_test():
    """"""
    plt.close()
    fig, ax = plt.subplots(1,1)

    poly_patch = mpl.patches.Polygon(((1,1),(1,3),(3,3), (3,1))) 
    hole_patch = mpl.patches.Polygon(((1.1,1.1),(1.1,2),(2.9,2), (2.9,1.1))) 
    hole_patch = mpl.patches.Circle((2,2), radius=.5, facecolor='red')
    poly_with_hole_patch = exclude_path(poly_patch, hole_patch)

    ax.add_patch(poly_with_hole_patch)

    ax.set_xlim(0, 4) 
    ax.set_ylim(0, 4)

def exclude_path(poly_patch, hole_patch, **kwargs):
    """"""
    
    poly_path = mpl.bezier.make_path_regular(poly_patch.get_path())
    closed_poly = mpl.path.Path(list(poly_path.vertices)+[(0,0)], list(poly_path.codes)+[mpl.path.Path.CLOSEPOLY])
    
    hole_path = hole_patch.get_path()
    hole_path = mpl.bezier.make_path_regular(hole_patch.get_path())
    closed_hole = mpl.path.Path(list(hole_path.vertices)+[(0,0)], list(hole_path.codes)+[mpl.path.Path.CLOSEPOLY])
    transformed_hole = hole_patch.get_patch_transform().transform_path(hole_path)
    
    print(hole_path, closed_hole)
    
    holedpath = mpl.bezier.concatenate_paths([closed_poly, transformed_hole])
    holedpatch = mpl.patches.PathPatch(holedpath, **kwargs)
    
    return holedpatch

def contour_hole_test():
    """"""
    
    plt.close()
    plt.figure()
    filled01 = [[[0, 0], [3, 0], [3, 3], [0, 3], [1, 1], [1, 2], [2, 2], [2, 1]]]
    filled01 = [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 1], [0, 3], [3, 3], [3, 1]]]
    filled01 = [[[0.0, 0.0], [0.0, 0.005555555555555555], [180.0, 0.005555555555555555], [180.0, 0.0], [0.0, 0.0], [0.0, 0.0038888888888888883], [180.0, 0.0038888888888888883], [180.0, 0.0]]]
    filled01 = [[[0.0, 0.0], [180.0, 0.0], [180.0, 0.005555555555555555], [0.0, 0.005555555555555555], [0.0, 0.0], [0.0, 0.0038888888888888883], [180.0, 0.0038888888888888883], [180.0, 0.0]]]
    kinds01 = [[1, 2, 2, 2, 1, 2, 2, 2]]
    cs = mpl.contour.ContourSet(plt.gca(), [0, 1], [filled01], [kinds01], filled=True, colors=dblue, alpha=0.2)

    #plt.axis([-0.5, 3.5, -0.5, 3.5])
    plt.axis([0,180,0,0.008])
    
    print(filled01, len(filled01[0]))
    print(kinds01)

def get_toy_model(survey, seed=205, vd=np.array([0, 220 - 15, 0]), vtd=np.array([0, 220 - 46, 0]), vh=np.array([0, 0, 0]), sd=np.array([35,20,16]), std=np.array([67,38,35]), sh=np.array([160,90,90]), ftd=0.3, N_=None):
    """ Produce toy model with arbitrary velocity components (defaults to Bensby+2003)
    Bensby et al. (2003) components
    vd = np.array([0,220-15,0])
    vtd = np.array([0,220-46,0])
    vh = np.array([0,0,0])

    sd = np.array([35, 20, 16])**2
    std = np.array([67, 38, 35])**2
    sh = np.array([160, 90, 90])**2
    """
    
    np.random.seed(seed)

    if N_==None:
        N = len(survey.data)
    else:
        N = N_
    Ncr = np.sum(survey.counter_rotating)
    Nh = Ncr*2
    Nd = N - Nh

    # pick positions from observed sample
    if N_==None:
        mixedx = np.random.permutation(survey.x)
    else:
        ind = np.random.choice(np.shape(survey.x)[0], size=N)
        mixedx = survey.x[ind]
    
    ## halo
    hx = mixedx[:Nh]

    # velocity direction
    hu_ = np.random.random(Nh)
    hv_ = np.random.random(Nh)
    htheta = np.arccos(2*hu_ - 1)
    hphi = 2 * np.pi * hv_
    hd = np.array([np.sin(htheta) * np.cos(hphi),
                np.sin(htheta) * np.sin(hphi),
                np.cos(htheta)]).T

    # velocity magnitude
    hv = sh[np.newaxis,:] * np.random.randn(Nh, 3) + vh[np.newaxis,:]

    # angular momentum
    hl = np.cross(hx, hv, axis=1)
    hl2 = np.linalg.norm(hl, axis=1)
    hlperp = np.sqrt(hl[:,0]**2 + hl[:,1]**2)
    hltheta = np.degrees(np.arccos(hl[:,2]/hl2))

    ## disk
    dx = mixedx[Nh:]

    # velocities
    dvt = sd[np.newaxis,:] * np.random.randn(Nd, 3) + vd[np.newaxis,:]
    dvth = std[np.newaxis,:] * np.random.randn(Nd, 3) + vtd[np.newaxis,:]
    dv = dvt

    td_ind = np.random.choice(np.arange(Nd), size=int(ftd*Nd), replace=False)
    dv[td_ind] = dvth[td_ind]

    # angular momentum
    dl = np.cross(dx, dv, axis=1)
    dl2 = np.linalg.norm(dl, axis=1)
    dlperp = np.sqrt(dl[:,0]**2 + dl[:,1]**2)
    dltheta = np.degrees(np.arccos(dl[:,2]/dl2))

    tv = np.concatenate([hv, dv])
    theta = np.concatenate([hltheta, dltheta])
    
    thalo = np.zeros(N, dtype=bool)
    thalo[:Nh] = True
    tdisk = np.zeros(N, dtype=bool)
    tdisk[Nh:] = True
    tddisk = td_ind
    
    return (tv, theta, thalo, tdisk, tddisk)

def toy_model(seed=205):
    """"""
    
    raveon = load_survey('raveon')
    vh_y = np.linspace(0,400,400)*u.km/u.s
    vh_xz = np.sqrt(raveon.sdisk**2 - (vh_y - raveon.vdisk)**2)
    
    tv, theta, thalo, tdisk, tddisk = get_toy_model(raveon, seed=seed)
    mock_halo = np.sqrt(tv[:,0]**2 + tv[:,2]**2 + (tv[:,1] - raveon.vdisk.value)**2)>raveon.sdisk.value
    #mock_halo = np.sqrt(tv[:,0]**2 + tv[:,2]**2 + (tv[:,1] - raveon.vdisk.value)**2)>180
    print('halo purity', np.sum(mock_halo & thalo)/np.sum(mock_halo))
    print('halo completeness', np.sum(mock_halo & thalo)/np.sum(thalo))
    
    plt.close()
    fig, ax = plt.subplots(3, 1, figsize=(5.5,12.5), gridspec_kw = {'height_ratios':[1.02,1.5,1.5]})
    
    plt.sca(ax[0])
    plt.plot(vh_y, vh_xz, 'k-', lw=3, zorder=10)

    txz = np.sqrt(tv[:,0]**2 + tv[:,2]**2)
    plt.plot(tv[:,1][tdisk], txz[tdisk], 'o', color=red, ms=2, rasterized=True, label='Toy disk')
    plt.plot(tv[:,1][thalo], txz[thalo], 'o', color=blue, ms=2, rasterized=True, label='Toy halo')
    
    t = plt.text(0.98, 0.1, 'Disk', transform=ax[0].transAxes, ha='right', fontsize='medium')
    t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    t = plt.text(0.98, 0.6, 'Halo', transform=ax[0].transAxes, ha='right', fontsize='medium')
    t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    
    plt.legend(loc=2, frameon=False, fontsize='small', handlelength=0.2, markerscale=2)
    plt.xlim(-400,400)
    plt.ylim(0,400)
    plt.xlabel('$V_Y$ (km/s)')
    plt.ylabel('$V_{XZ}$ (km/s)')
    title = plt.title('Toy model')
    title.set_position([.5, 1.05])
    
    plt.sca(ax[1])
    bx = np.linspace(0, 180, 15)
    plt.hist(theta[tdisk], bins=bx, histtype='stepfilled', color=red, alpha=0.8, lw=4, label='Toy disk')
    plt.hist(theta[thalo], bins=bx, histtype='stepfilled', color=blue, alpha=0.8, lw=4, label='Toy halo')

    plt.hist(theta, bins=bx, histtype='step', color='0.5', lw=2, label='')
    plt.plot([-1,0], [-10,-11], color='0.5', lw=2, alpha=1, label='Toy model')
    
    plt.hist(raveon.ltheta, bins=bx, histtype='step', color='0.9', lw=4, label='')
    plt.hist(raveon.ltheta, bins=bx, histtype='step', color='0.2', lw=2, label='')
    plt.plot([-1,0], [-10,-11], color='0.2', lw=2, alpha=1, label='Milky Way', path_effects=[pe.Stroke(linewidth=5, foreground='0.9'), pe.Normal()])

    plt.xlim(0,180)
    plt.ylim(9e-1, 3e5)
    plt.gca().set_yscale('log')
    ax[1].set_xticks(np.arange(0,181,45))

    plt.legend(loc=2, frameon=False, fontsize='small')
    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('Number of stars')
    plt.title('True membership', fontsize='medium')
    
    h_, l_ = ax[1].get_legend_handles_labels()
    hcorr = h_[2:] + h_[:2]
    lcorr = l_[2:] + l_[:2]
    ax[1].legend(hcorr, lcorr, frameon=False, loc=2, fontsize='small')
    
    plt.sca(ax[2])
    plt.hist(theta[thalo & mock_halo], bins=bx, histtype='stepfilled', color=blue, alpha=0.8, lw=4, label='Isotropic toy halo', normed=True)
    
    finite = np.isfinite(raveon.data['feh'])
    mrich = raveon.data['feh']>-1
    bxmod = np.copy(bx)
    bxmod[0] -= 1
    bxmod[-1] += 2
    
    plt.hist(raveon.ltheta[raveon.halo & finite & ~mrich], bins=bxmod, histtype='step', color='0.9', lw=4, label='', normed=True)
    plt.hist(raveon.ltheta[raveon.halo & finite & ~mrich], bins=bxmod, histtype='step', color=dblue, lw=2, label='', normed=True)
    plt.plot([-1,0], [-10,-11], color=dblue, lw=2, alpha=1, label='Milky Way, metal-poor halo', path_effects=[pe.Stroke(linewidth=5, foreground='0.9'), pe.Normal()])
    
    plt.hist(raveon.ltheta[raveon.halo & finite & mrich], bins=bxmod, histtype='step', color='0.9', lw=4, label='', normed=True)
    plt.hist(raveon.ltheta[raveon.halo & finite & mrich], bins=bxmod, histtype='step', color=lblue, lw=2, label='', normed=True)
    plt.plot([-1,0], [-10,-11], color=lblue, lw=2, alpha=1, label='Milky Way, metal-rich halo', path_effects=[pe.Stroke(linewidth=5, foreground='0.9'), pe.Normal()])
    
    #plt.hist(raveon.ltheta[raveon.halo & finite & mrich], bins=bxmod, histtype='step', color='w', lw=4, label='', normed=True)
    #plt.hist(raveon.ltheta[raveon.halo & finite & mrich], bins=bxmod, histtype='step', color=lblue, lw=2, label='Metal-rich MW halo', normed=True)
    
    plt.xlim(0,180)
    plt.gca().set_yscale('log')
    ax[2].set_xticks(np.arange(0,181,45))

    plt.legend(loc=2, frameon=False, fontsize='small')
    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('Probability density (deg$^{-1}$)')
    plt.ylim(3e-4, 1.4e-1)
    plt.title('Kinematic selection', fontsize='medium')
    
    h_, l_ = ax[2].get_legend_handles_labels()
    hcorr = [h_[2]] + h_[:2]
    lcorr = [l_[2]] + l_[:2]
    ax[2].legend(hcorr, lcorr, frameon=False, loc=2, fontsize='small')
    
    plt.tight_layout()
    plt.savefig('../plots/paper/toy_model.pdf', bbox_inches='tight')

def latte(mwcomp=False):
    """Properties of a solar neighborhood in Latte"""
    
    mpl.rcParams['axes.linewidth'] = 1.5
    s = load_survey('lattemdif')
    raveon = load_survey('raveon')
    fehacc = -1
    accreted = s.data['feh']<=fehacc
    
    plt.close()
    fig, ax = plt.subplots(3, 1, figsize=(5.5,13), gridspec_kw = {'height_ratios':[1,1.5,1.5]})
    
    plt.sca(ax[0])
    im = plt.scatter(s.vy, s.vxz, c=s.data['feh'], s=10, cmap='magma', vmin=-2.5, vmax=0.5, edgecolors='none', rasterized=True)
    #plt.plot(s.vy[~accreted], s.vxz[~accreted], 'o', color=lblue, ms=1, rasterized=True)
    #plt.plot(s.vy[accreted], s.vxz[accreted], 'o', color=dblue, ms=1, rasterized=True)
    
    vh_y = np.linspace(0,400,400)*u.km/u.s
    vh_xz = np.sqrt(s.sdisk**2 - (vh_y - s.vdisk)**2)
    plt.plot(vh_y, vh_xz, 'k-', lw=2)
    print(np.mean(s.vy), np.mean(raveon.vy))
    
    t = plt.text(0.98, 0.1, 'Disk', transform=ax[0].transAxes, ha='right', fontsize='medium')
    t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    t = plt.text(0.98, 0.6, 'Halo', transform=ax[0].transAxes, ha='right', fontsize='medium')
    t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
    
    plt.xlim(-400,400)
    plt.ylim(0,400)
    plt.xlabel('$V_Y$ (km/s)')
    plt.ylabel('$V_{XZ}$ (km/s)')
    title = plt.title('Latte simulation')
    title.set_position([.5, 1.4])
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("top", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=np.arange(-2.5,0.51,0.5), orientation='horizontal')
    
    cax.xaxis.tick_top()
    cax.tick_params(labelsize='small')
    cax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('[Fe/H]', fontsize='small')
    cax.xaxis.set_label_position('top') 
    
    plt.sca(ax[1])
    bx = np.arange(-2.5,0.5,0.2)
    
    # overplot MW
    finite = np.isfinite(raveon.data['feh'])
    plt.hist(s.data['feh'][s.disk], bins=bx, histtype='stepfilled', color=red, normed=True, lw=0, alpha=0.8, label='Disk')
    if mwcomp:
        plt.hist(raveon.data['feh'][raveon.disk & finite], bins=bx, histtype='step', color='0.9', normed=True, lw=4, alpha=1, label='')
        plt.hist(raveon.data['feh'][raveon.disk & finite], bins=bx, histtype='step', color=red, normed=True, lw=2, alpha=1, label='')

    plt.hist(s.data['feh'][s.halo], bins=bx, histtype='stepfilled', color=blue, normed=True, lw=0, alpha=0.8, label='Halo')
    if mwcomp:
        plt.hist(raveon.data['feh'][raveon.halo & finite], bins=bx, histtype='step', color='0.9', normed=True, lw=4, alpha=1, label='')
        plt.hist(raveon.data['feh'][raveon.halo & finite], bins=bx, histtype='step', color=blue, normed=True, lw=2, alpha=1, label='')
        plt.plot([-1,0], [-10,-11], color='0.2', lw=2, alpha=1, label='Milky Way', path_effects=[pe.Stroke(linewidth=5, foreground='0.9'), pe.Normal()])
        
    
    plt.axvline(-1, ls='--', lw=2, color='0.2')
    plt.xlabel('[Fe/H]')
    plt.ylabel('Probability density (dex$^{-1}$)')
    plt.title('Kinematic selection', fontsize='medium')
    #ax[1].set_xticks(np.arange(-2,0.5,1))
    #plt.xlim(-2,0.5)
    ax[1].set_ylim(bottom=0)

    leg = plt.legend(frameon=False, loc=2, fontsize='small')
    #h_, l_ = ax[1].get_legend_handles_labels()
    #hcorr = [h_[1], h_[2], h_[0]]
    #lcorr = [l_[1], l_[2], l_[0]]
    #ax[1].legend(hcorr, lcorr, frameon=False, loc=2, fontsize='small')
    
    plt.sca(ax[2])
    bx = np.linspace(0,180,10)
    flag_norm = True
    
    # disk
    plt.hist(s.ltheta[s.disk], color=red, histtype='stepfilled', alpha=0.8, bins=bx, zorder=0, lw=2, normed=flag_norm, label='Disk')
    
    if mwcomp:
        plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, histtype='step', color='0.9', lw=4, label='', normed=True)
        plt.hist(raveon.ltheta[raveon.disk & finite], bins=bx, histtype='step', color=red, lw=2, label='', normed=True)
        plt.plot([10,20], [-10,-11], color='0.2', lw=2, alpha=1, label='Milky Way', path_effects=[pe.Stroke(linewidth=5, foreground='0.9'), pe.Normal()])

    # halo
    plt.hist(s.ltheta[s.halo & ~accreted], color=lblue, histtype='stepfilled', alpha=0.7, bins=bx, lw=2, normed=flag_norm, 
            label='Metal-rich halo')
    # overplot MW
    if mwcomp:
        finite = np.isfinite(raveon.data['feh'])
        mrich = raveon.data['feh']>-1
        plt.hist(raveon.ltheta[raveon.halo & finite & mrich], bins=bx, histtype='step', color='0.9', lw=4, label='', normed=True)
        plt.hist(raveon.ltheta[raveon.halo & finite & mrich], bins=bx, histtype='step', color='royalblue', lw=2, label='', normed=True)

    plt.hist(s.ltheta[s.halo & accreted], color=dblue, histtype='stepfilled', alpha=0.7, bins=bx, lw=2, normed=flag_norm, 
            label='Metal-poor halo')
    if mwcomp:
        plt.hist(raveon.ltheta[raveon.halo & finite & ~mrich], bins=bx, histtype='step', color='0.9', lw=4, label='', normed=True)
        plt.hist(raveon.ltheta[raveon.halo & finite & ~mrich], bins=bx, histtype='step', color='navy', lw=2, label='', normed=True)

    
    plt.xlim(0, 180)
    plt.ylim(1e-3, 0.1)
    ax[2].set_yscale('log')
    ax[2].set_xticks(np.arange(0,181,45))
    
    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('Probability density (deg$^{-1}$)')
    plt.title('Kinematic selection', fontsize='medium')

    plt.legend(loc=2, frameon=False, fontsize='small')
    
    for i in [1,2]:
        h_, l_ = ax[i].get_legend_handles_labels()
        hcorr = h_[1:] + [h_[0]]
        lcorr = l_[1:] + [l_[0]]
        ax[i].legend(hcorr, lcorr, frameon=False, loc=2, fontsize='small')

    
    
    plt.tight_layout()
    if mwcomp:
        plt.savefig('../plots/paper/latte_mwcomp.pdf', bbox_inches='tight')
    else:
        plt.savefig('../plots/paper/latte.pdf', bbox_inches='tight')
    mpl.rcParams['axes.linewidth'] = 2



def latte_dform():
    """Formation properties of Latte star particles"""
    
    latte = load_survey('lattemdif')
    
    plt.close()
    plt.figure(figsize=(6,5))

    dacc = 20
    accreted = latte.data['dform']>dacc

    im = plt.scatter(latte.data['age'], latte.data['dform'], c=latte.data['feh'], edgecolors='none', cmap='magma', vmin=-2.5, vmax=0.5, rasterized=True)

    plt.axhline(dacc, ls='-', color='k', lw=2, zorder=0)
    plt.axhspan(5, 11, color='0.5', alpha=0.2, zorder=2)
    plt.text(0.7,0.8, 'Accreted', transform=plt.gca().transAxes, ha='left', va='top', fontsize=18)
    plt.text(0.7,0.25, 'In situ', transform=plt.gca().transAxes, ha='left', va='bottom', fontsize=18)
    
    plt.ylim(1e-1,500)
    plt.xlim(13.8,0)
    plt.gca().set_yscale('log')
    plt.xlabel('Age (Gyr)')
    plt.ylabel('Formation distance (kpc)')

    plt.tight_layout()
    
    pos = plt.gca().get_position()
    cax = plt.axes([0.95,pos.y0,0.02,pos.y1 - pos.y0])
    plt.colorbar(im, cax=cax)
    plt.ylabel('[Fe/H]')
    
    plt.savefig('../plots/paper/latte_dform.pdf', bbox_inches='tight')
    
def latte_dform2():
    """Formation properties of Latte star particles"""
    
    latte = load_survey('lattemdif')
    
    plt.close()
    plt.figure(figsize=(6,5))

    dacc = 20
    accreted = latte.data['feh']<=-1
    lw = 0.3
    ms = 5

    plt.plot(latte.data['age'][latte.disk], latte.data['dform'][latte.disk], 'o', ms=ms, c=red, mec='w', mew=lw, rasterized=True, label='Disk')
    plt.plot(latte.data['age'][latte.halo & accreted], latte.data['dform'][latte.halo & accreted], 'o', ms=ms, c=dblue, mec='w', mew=lw, rasterized=True, label='Metal-poor halo')
    plt.plot(latte.data['age'][latte.halo & ~accreted], latte.data['dform'][latte.halo & ~accreted], 'o', ms=ms, c=lblue, mec='w', mew=lw, rasterized=True, label='Metal-rich halo')

    plt.axhline(dacc, ls='-', color='k', lw=2, zorder=0)
    plt.axhspan(5, 11, color='0.5', alpha=0.2, zorder=2)
    plt.text(0.75,0.8, 'Accreted', transform=plt.gca().transAxes, ha='left', va='top', fontsize='medium')
    plt.text(0.75,0.25, 'In situ', transform=plt.gca().transAxes, ha='left', va='bottom', fontsize='medium')
    
    plt.ylim(1e-1,500)
    plt.xlim(13.8,0)
    plt.gca().set_yscale('log')
    plt.xlabel('Age (Gyr)')
    plt.ylabel('Formation distance (kpc)')
    plt.legend(loc=4, fontsize='small', frameon=False, handlelength=0.2)

    plt.tight_layout()
    plt.savefig('../plots/paper/latte_dform2.pdf', bbox_inches='tight')

def accreted_fraction(x, dacc=20):
    if np.size(x):
        acc = x>dacc
        return np.sum(acc)/np.size(acc)
    else:
        return np.nan

def latte_facc():
    """"""
    s = load_survey('lattemdif')
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(5.5,5))
    by = np.linspace(-3,0.5,10)
    bx = np.linspace(0,180,10)

    facc, xe, ye, nb = scipy.stats.binned_statistic_2d(s.ltheta[s.halo], s.data['feh'][s.halo], s.data['dform'][s.halo], 
                                                    statistic=accreted_fraction, bins=(bx, by))
    
    # set nans to the avg of the nearest finite pixels
    facc_interp = pd.DataFrame(facc).interpolate(method='cubic', axis=1).values
    facc_interp = pd.DataFrame(facc_interp).interpolate(method='cubic', axis=0).values
    
    xc = myutils.bincen(xe)
    yc = myutils.bincen(ye)
    oldgrid_x, oldgrid_y = np.meshgrid(xc, yc)
    points = np.array([np.ravel(oldgrid_x), np.ravel(oldgrid_y)]).T
    values = np.ravel(facc_interp)
    
    grid_x, grid_y = np.mgrid[0:180:1000j, -3:0.5:1000j]
    grid_z = scipy.interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
    
    facc_smooth = filters.gaussian_filter(grid_z.T, 100)

    
    ratio = (xe[-1] - xe[0]) / (ye[-1] - ye[0])
    im = plt.imshow(facc_smooth.T, origin='lower', vmin=0, vmax=1, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect='auto', interpolation='gaussian', cmap='viridis')
    #im = plt.imshow(grid_z, origin='lower', vmin=0, vmax=1, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect=ratio, interpolation='none', cmap='bone')
    
    cs = plt.contour(facc_smooth.T, extent=(xe[0], xe[-1], ye[0], ye[-1]), levels=(0.1,0.5,0.9), colors='0.9')
    
    fmt = {}
    for i, l in enumerate(cs.levels):
        fmt[l] = '{:.0f}% accreted'.format(l*100)
    labels = plt.clabel(cs, inline=True, fontsize='small', fmt=fmt, colors='w')

    ax.set_xticks(np.arange(0,181,45))
    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('[Fe/H]')

    plt.tight_layout()

    pos = plt.gca().get_position()
    cax = plt.axes([0.97,pos.y0,0.05,pos.y1 - pos.y0])
    plt.colorbar(im, cax=cax) #, ticks=np.arange(-2.5,0.51,0.5))
    plt.ylabel('Accreted fraction')
    
    plt.savefig('../plots/paper/latte_facc.pdf', bbox_inches='tight')
    
def latte_med_dform():
    """"""
    s = load_survey('lattemdif')
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    by = np.linspace(-3,0.5,10)
    bx = np.linspace(0,180,10)

    facc, xe, ye, nb = scipy.stats.binned_statistic_2d(s.ltheta[s.halo], s.data['feh'][s.halo], s.data['dform'][s.halo], 
                                                    statistic=np.median, bins=(bx, by))
    
    # set nans to the avg of the nearest finite pixels
    facc_interp = pd.DataFrame(facc).interpolate(method='cubic', axis=1).values
    facc_interp = pd.DataFrame(facc_interp).interpolate(method='cubic', axis=0).values
    
    xc = myutils.bincen(xe)
    yc = myutils.bincen(ye)
    oldgrid_x, oldgrid_y = np.meshgrid(xc, yc)
    points = np.array([np.ravel(oldgrid_x), np.ravel(oldgrid_y)]).T
    values = np.ravel(facc_interp)
    
    grid_x, grid_y = np.mgrid[0:180:1000j, -3:0.5:1000j]
    grid_z = scipy.interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
    
    facc_smooth = filters.gaussian_filter(grid_z.T, 100)

    
    ratio = (xe[-1] - xe[0]) / (ye[-1] - ye[0])
    im = plt.imshow(facc_smooth.T, origin='lower', vmin=1, vmax=100, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect=ratio, interpolation='gaussian', cmap='viridis', norm=mpl.colors.LogNorm())
    #im = plt.imshow(grid_z, origin='lower', vmin=0, vmax=1, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect=ratio, interpolation='none', cmap='bone')

    plt.xlabel('$\\vec{L}$ orientation (deg)')
    plt.ylabel('[Fe/H]')

    plt.tight_layout()

    pos = plt.gca().get_position()
    cax = plt.axes([0.97,pos.y0,0.05,pos.y1 - pos.y0])
    plt.colorbar(im, cax=cax) #, ticks=np.arange(-2.5,0.51,0.5))
    plt.ylabel('Median formation distance')
    
    plt.savefig('../plots/paper/latte_med_dform.pdf', bbox_inches='tight')
    
def latte_ltheta_origin():
    """"""
    latte = load_survey('lattemdif')
    accreted = latte.data['dform']>20
    mrich = latte.data['feh']>-1
    bx = np.linspace(0,180,10)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.hist(latte.ltheta[latte.halo & accreted], bins=bx, color=dblue, alpha=0.5, normed=True)
    plt.hist(latte.ltheta[latte.halo & ~accreted], bins=bx, color=lblue, alpha=0.5, normed=True)
    ax[0].set_yscale('log')
    
    plt.sca(ax[1])
    plt.hist(latte.ltheta[latte.halo & ~mrich], bins=bx, color=dblue, alpha=0.5, normed=True)
    plt.hist(latte.ltheta[latte.halo & mrich], bins=bx, color=lblue, alpha=0.5, normed=True)
    ax[1].set_yscale('log')
    
    plt.tight_layout()



def tdcontamination(chicalc=False):
    """"""
    raveon = load_survey('raveon')
    tv, theta, thalo, tdisk, tddisk = get_toy_model(raveon, N_=5000000)
    txz = np.sqrt(tv[:,0]**2 + tv[:,2]**2)
    
    extents = [[-400,400], [0,600]]

    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(12,5), gridspec_kw = {'width_ratios':[2, 1]})

    plt.sca(ax[0])
    V = 1.0 - np.exp(-0.5 * np.arange(1, 4.1, 1) ** 2)
    
    tdx = tv[:,1][tdisk][tddisk]
    tdy = txz[tdisk][tddisk]
    bins = 50
    X = np.linspace(extents[0][0], extents[0][1], bins + 1)
    Y = np.linspace(extents[1][0], extents[1][1], bins + 1)
    H, X, Y = np.histogram2d(tdx.flatten(), tdy.flatten(), bins=(X, Y))

    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    V_ = V*0
    for i, v0 in enumerate(V):
        try:
            V_[i] = Hflat[sm <= v0][-1]
        except:
            V_[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    cs = plt.contour(X1, Y1, H.T, np.sort(V_), colors='orange', linewidths=3)
    
    fmt = {}
    for i, l in enumerate(cs.levels):
        fmt[l] = '{:d} $\sigma$'.format(4-i)
    labels = plt.clabel(cs, inline=True, fontsize=15, fmt=fmt)
    
    for l in labels:
        l.set_rotation(0)
    
    #triangle.hist2d(tv[:,1][tdisk][tddisk], txz[tdisk][tddisk], ax=ax[0], plot_contours=True, plot_datapoints=False, plot_hist2d=False,
                    #extent=[extents[0], extents[1]], color='orange', linewidths=3, levels=V)


    plt.plot(-100, -100, '-', color='orange', lw=3, label='Thick disk (toy model)')

    finite = np.isfinite(raveon.data['feh'])
    mrich = raveon.data['feh']>-1

    plt.plot(raveon.vy[finite & raveon.halo & ~mrich], raveon.vxz[finite & raveon.halo & ~mrich], 's', color='navy', ms=4, alpha=1, zorder=1,
            label='Milky Way halo: $[Fe/H]\leq-1$')
    plt.plot(raveon.vy[finite & raveon.halo & mrich], raveon.vxz[finite & raveon.halo & mrich], 'o', color='royalblue', ms=4, alpha=1, zorder=1,
            label='Milky Way halo: $[Fe/H]>-1$')

    plt.legend(handlelength=0.6, handletextpad=0.5, markerscale=1.5, ncol=1, framealpha=0.97, fontsize='small')
    plt.xlim(extents[0])
    plt.ylim(extents[1])

    plt.xlabel('$V_Y$ (km/s)')
    plt.ylabel('$V_{XZ}$ (km/s)')
    
    plt.sca(ax[1])
    
    vraveon = np.array([raveon.v[:,0], raveon.v[:,1], raveon.v[:,2]])
    std = np.array([67,38,35])**2
    vtd=np.array([0, 220 - 46, 0])
    
    if chicalc:
        N = len(raveon.data)
        chitd_man = np.zeros(N)
        for i_ in range(N):
            std_ = np.diag(std) + np.diag(raveon.data['gc_error'][i_,3:])**2
            aux = np.matmul(np.linalg.inv(std_), (vraveon[:,i_]-vtd))
            chitd_man[i_] = np.matmul((vraveon[:,i_]-vtd).T, aux)
        np.save('../data/chitd', chitd_man)
    else:
        chitd_man = np.load('../data/chitd.npy')
    
    bx = np.logspace(-3,3,200)
    mrich = raveon.data['feh']>-1
    cfinite = np.isfinite(chitd_man)
    rv = scipy.stats.chi2(3)
    
    hs, be = np.histogram(chitd_man[raveon.halo & ~mrich & cfinite], bins=bx)
    bc = myutils.bincen(bx)
    plt.plot(1-rv.cdf(bc), np.cumsum(hs)/np.sum(hs), '-', color='navy', lw=3, label='Metal-poor halo')
    
    hs, be = np.histogram(chitd_man[raveon.halo & mrich & cfinite], bins=bx)
    bc = myutils.bincen(bx)
    plt.plot(1-rv.cdf(bc), np.cumsum(hs)/np.sum(hs), '-', color='royalblue', lw=3, label='Metal-rich halo')
    
    print('mrich', 1-rv.cdf(np.median(chitd_man[raveon.halo & mrich & cfinite])))
    print('mpoor', 1-rv.cdf(np.median(chitd_man[raveon.halo & ~mrich & cfinite])))

    plt.axvline(1e-2, color='k')
    plt.gca().set_xscale('log')
    plt.xlim(1e-6,1)
    plt.ylim(0,1)

    plt.xlabel('Probability')
    plt.ylabel('Cumulative fraction')
    plt.title('Thick disk misclassification', fontsize='medium')
    plt.legend(frameon=True, loc=1, fontsize='small', framealpha=1)

    plt.tight_layout()
    plt.savefig('../plots/paper/tdcontamination.pdf', bbox_inches='tight')

def quantify_contamination(genmod=False):
    """"""
    raveon = load_survey('raveon')
    
    if genmod:
        tv, theta, thalo, tdisk, tddisk = get_toy_model(raveon, N_=5000000) #, std=np.array([53, 51, 35])
        txz = np.sqrt(tv[:,0]**2 + tv[:,2]**2)
        
        extents = [[-400,400], [0,600]]

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8,5))

        V = 1.0 - np.exp(-0.5 * np.arange(4, 4.1, 1) ** 2)
        
        tdx = tv[:,1][tdisk][tddisk]
        tdy = txz[tdisk][tddisk]
        bins = 50
        X = np.linspace(extents[0][0], extents[0][1], bins + 1)
        Y = np.linspace(extents[1][0], extents[1][1], bins + 1)
        H, X, Y = np.histogram2d(tdx.flatten(), tdy.flatten(), bins=(X, Y))

        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]

        V_ = V*0
        for i, v0 in enumerate(V):
            try:
                V_[i] = Hflat[sm <= v0][-1]
            except:
                V_[i] = Hflat[0]

        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
        cs = plt.contour(X1, Y1, H.T, np.sort(V_), colors='orange', linewidths=3)
        
        p = cs.collections[0].get_paths()[0]
        vert = p.vertices
        np.save('../data/td_4sigma_vertices', vert)
    else:
        vert = np.load('../data/td_4sigma_vertices.npy')
    
    
    path_td = mpl.path.Path(vert)
    
    finite = np.isfinite(raveon.data['feh'])
    mrich = raveon.data['feh']>-1
    points = np.array([raveon.vy[finite & raveon.halo & mrich], raveon.vxz[finite & raveon.halo & mrich]]).T
    tdbox = path_td.contains_points(points)
    print(np.size(tdbox) - np.sum(tdbox), np.sum(tdbox), np.shape(points))
    print('metal-rich, within 4sig', np.sum(tdbox)/np.size(tdbox))
    
    points = np.array([raveon.vy[finite & raveon.halo & ~mrich], raveon.vxz[finite & raveon.halo & ~mrich]]).T
    tdbox = path_td.contains_points(points)
    print(np.size(tdbox) - np.sum(tdbox), np.sum(tdbox), np.shape(points))
    print('metal-poor, within 4sig', np.sum(tdbox)/np.size(tdbox))
    
    print('thick disk, between 3 and 4 sig', np.sum(raveon.disk)*0.3*(np.exp(-0.5*3**2) - np.exp(-0.5*4**2)))
    print('thick disk, outside 4sig', np.sum(raveon.disk)*0.3*np.exp(-0.5*4**2))
    
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    plt.plot(raveon.vy[finite & raveon.halo & mrich], raveon.vxz[finite & raveon.halo & mrich], 'ko')
    plt.plot(vert[:,0], vert[:,1], '-', color='orange', lw=2)

def tdchem():
    """"""
    raveon = load_survey('raveon')
    finite = np.isfinite(raveon.data['feh'])
    inter = (raveon.data['feh']>-0.8) & (raveon.data['feh']<-0.2)
    
    plt.close()
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    im = plt.hist2d(raveon.vy[finite & inter], raveon.vxz[finite & inter], bins=(np.linspace(-400,400,30), np.linspace(0,400,15)), norm=mpl.colors.LogNorm(), vmax=1e5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im[-1], cax=cax)


def latte_ages():
    """"""
    latte = load_survey('lattemdif')
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    Nb = 14
    bx = np.linspace(0, 14, Nb)
    bc = myutils.bincen(bx)

    dacc = 20
    accreted = latte.data['dform']>dacc

    indices = [latte.disk, latte.halo & ~accreted, latte.halo & accreted]
    colors = [red, lblue, dblue]
    labels = ['Disk', 'In situ halo', 'Accreted halo']

    for i in range(3):
        idx = np.digitize(latte.data['age'][indices[i]], bx)
        rmed = np.array([np.median(latte.data['feh'][indices[i]][idx==k]) for k in range(Nb)])[1:]
        rup = np.array([np.percentile(latte.data['feh'][indices[i]][idx==k], [84,]) if np.sum(idx==k) else np.nan for k in range(Nb)])[1:]
        rdn = np.array([np.percentile(latte.data['feh'][indices[i]][idx==k], [16,]) if np.sum(idx==k) else np.nan for k in range(Nb)])[1:]

        plt.fill_between(bc, rup, rdn, color=colors[i], alpha=0.7, label=labels[i])
        
    plt.ylim(-2.5,0.5)
    plt.xlim(13.8,0)
    plt.xlabel('Age (Gyr)')
    plt.ylabel('[Fe/H]')
    plt.legend(loc=4, fontsize='medium', frameon=False)

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/paper/latte_ages.pdf', bbox_inches='tight')


def vy_mode():
    """"""
    
    latte = load_survey('lattemdif')
    raveon = load_survey('raveon')
    
    print(np.sum(raveon.vy>220*u.km/u.s)/np.size(raveon.vy))
    print(np.sum(latte.vy>248*u.km/u.s)/np.size(latte.vy))
    
    bx = np.linspace(150,300,100)
    
    plt.close()
    plt.figure()
    
    plt.hist(latte.vy, bins=bx, normed=True, label='Latte', histtype='step')
    plt.hist(raveon.vy, bins=bx, normed=True, label='Milky Way', histtype='step')
    
    plt.legend()
    
def age_ltheta():
    """"""
    latte = load_survey('lattemdif')
    mrich = latte.data['feh']>-1
    
    Nage = 2
    bins_age = np.linspace(8,14,Nage+1)
    bins_ltheta = np.linspace(0,181,10)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.plot(latte.data['age'][latte.halo & mrich], latte.ltheta[latte.halo & mrich], 'ko')
    
    plt.sca(ax[1])
    for i in range(Nage):
        ind = (latte.data['age']>=bins_age[i]) & (latte.data['age']<bins_age[i+1])
        plt.hist(latte.ltheta[latte.halo & mrich & ind], bins=bins_ltheta, histtype='step', color=mpl.cm.magma(i/Nage), normed=True, lw=2, label='{}'.format(bins_age[i]))
    
    plt.legend()



### Latte comparison ###
# cross-check of all the plots in metal-diffusion, fiducial and high-res runs

def comp_latte():
    """"""
    mpl.rcParams['axes.linewidth'] = 1.5
    
    plt.close()
    fig, ax = plt.subplots(3, 3, figsize=(16.5,13), gridspec_kw = {'height_ratios':[1,1.5,1.5]})
    
    for i, survey in enumerate(['lattemdif', 'lattefid', 'lattehr']):
        s = load_survey(survey)
        fehacc = -1
        accreted = s.data['feh']<=fehacc
        
        plt.sca(ax[0][i])
        im = plt.scatter(s.vy, s.vxz, c=s.data['feh'], s=10, cmap='magma', vmin=-2.5, vmax=0.5, edgecolors='none', rasterized=True)

        vh_y = np.linspace(0,400,400)*u.km/u.s
        vh_xz = np.sqrt(s.sdisk**2 - (vh_y - s.vdisk)**2)
        plt.plot(vh_y, vh_xz, 'k-', lw=2)
        
        t = plt.text(0.98, 0.1, 'Disk', transform=ax[0][i].transAxes, ha='right', fontsize='medium')
        t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
        t = plt.text(0.98, 0.6, 'Halo', transform=ax[0][i].transAxes, ha='right', fontsize='medium')
        t.set_bbox(dict(fc='w', alpha=0.5, ec='none'))
        
        plt.xlim(-400,400)
        plt.ylim(0,400)
        plt.xlabel('$V_Y$ (km/s)')
        plt.ylabel('$V_{XZ}$ (km/s)')
        title = plt.title(survey)
        title.set_position([.5, 1.4])
        
        divider = make_axes_locatable(ax[0][i])
        cax = divider.append_axes("top", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax, ticks=np.arange(-2.5,0.51,0.5), orientation='horizontal')
        
        cax.xaxis.tick_top()
        cax.tick_params(labelsize='small')
        cax.tick_params(axis=u'both', which=u'both',length=0)
        plt.xlabel('[Fe/H]', fontsize='small')
        cax.xaxis.set_label_position('top') 
        
        plt.sca(ax[1][i])
        bx = np.arange(-2.5,0.5,0.2)
        
        plt.hist(s.data['feh'][s.disk], bins=bx, histtype='stepfilled', color=red, normed=True, lw=0, alpha=0.8, label='Disk')
        plt.hist(s.data['feh'][s.halo], bins=bx, histtype='stepfilled', color=blue, normed=True, lw=0, alpha=0.8, label='Halo')

        plt.axvline(-1, ls='--', lw=2, color='0.2')
        plt.xlabel('[Fe/H]')
        plt.ylabel('Probability density (dex$^{-1}$)')
        plt.title('Kinematic selection', fontsize='medium')

        ax[1][i].set_ylim(bottom=0)

        leg = plt.legend(frameon=False, loc=2, fontsize='small')

        plt.sca(ax[2][i])
        bx = np.linspace(0,180,10)
        flag_norm = True
        
        # disk
        plt.hist(s.ltheta[s.disk], color=red, histtype='stepfilled', alpha=0.8, bins=bx, zorder=0, lw=2, normed=flag_norm, label='Disk')

        # halo
        plt.hist(s.ltheta[s.halo & ~accreted], color=lblue, histtype='stepfilled', alpha=0.7, bins=bx, lw=2, normed=flag_norm, 
                label='Metal-rich halo')

        plt.hist(s.ltheta[s.halo & accreted], color=dblue, histtype='stepfilled', alpha=0.7, bins=bx, lw=2, normed=flag_norm, 
                label='Metal-poor halo')
        
        plt.xlim(0, 180)
        plt.ylim(1e-3, 0.1)
        ax[2][i].set_yscale('log')
        ax[2][i].set_xticks(np.arange(0,181,45))
        
        plt.xlabel('$\\vec{L}$ orientation (deg)')
        plt.ylabel('Probability density (deg$^{-1}$)')
        plt.title('Kinematic selection', fontsize='medium')

        plt.legend(loc=2, frameon=False, fontsize='small')
        
        for j in [1,2]:
            h_, l_ = ax[j][i].get_legend_handles_labels()
            hcorr = h_[1:] + [h_[0]]
            lcorr = l_[1:] + [l_[0]]
            ax[j][i].legend(hcorr, lcorr, frameon=False, loc=2, fontsize='small')

    
    plt.tight_layout()
    plt.savefig('../plots/comparison_latte.pdf', bbox_inches='tight')
    mpl.rcParams['axes.linewidth'] = 2

def comp_latte_dform2():
    """"""
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15 ,5))
    
    for i, survey in enumerate(['lattemdif', 'lattefid', 'lattehr']):
        latte = load_survey(survey)

        dacc = 20
        accreted = latte.data['feh']<=-1
        lw = 0.3
        ms = 5
        
        plt.sca(ax[i])
        plt.plot(latte.data['age'][latte.disk], latte.data['dform'][latte.disk], 'o', ms=ms, c=red, mec='w', mew=lw, rasterized=True, label='Disk')
        plt.plot(latte.data['age'][latte.halo & accreted], latte.data['dform'][latte.halo & accreted], 'o', ms=ms, c=dblue, mec='w', mew=lw, rasterized=True, label='Metal-poor halo')
        plt.plot(latte.data['age'][latte.halo & ~accreted], latte.data['dform'][latte.halo & ~accreted], 'o', ms=ms, c=lblue, mec='w', mew=lw, rasterized=True, label='Metal-rich halo')

        plt.axhline(dacc, ls='-', color='k', lw=2, zorder=0)
        plt.axhspan(5, 11, color='0.5', alpha=0.2, zorder=2)
        plt.text(0.75,0.8, 'Accreted', transform=plt.gca().transAxes, ha='left', va='top', fontsize='medium')
        plt.text(0.75,0.25, 'In situ', transform=plt.gca().transAxes, ha='left', va='bottom', fontsize='medium')
        
        plt.ylim(1e-1,500)
        plt.xlim(13.8,0)
        plt.gca().set_yscale('log')
        plt.xlabel('Age (Gyr)')
        plt.ylabel('Formation distance (kpc)')
        plt.legend(loc=4, fontsize='small', frameon=False, handlelength=0.2)
        plt.title(survey)

    plt.tight_layout()
    plt.savefig('../plots/comparison_latte_dform2.pdf', bbox_inches='tight')

def comp_latte_facc():
    """"""
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(16.5,5))
    by = np.linspace(-3,0.5,10)
    bx = np.linspace(0,180,10)

    for i, survey in enumerate(['lattemdif', 'lattefid', 'lattehr']):
        s = load_survey(survey)
        
        if i>0:
            by = np.linspace(-4,0.5,10)
        
        facc, xe, ye, nb = scipy.stats.binned_statistic_2d(s.ltheta[s.halo], s.data['feh'][s.halo], s.data['dform'][s.halo], 
                                                        statistic=accreted_fraction, bins=(bx, by))
        
        # set nans to the avg of the nearest finite pixels
        facc_interp = pd.DataFrame(facc).interpolate(method='cubic', axis=1).values
        facc_interp = pd.DataFrame(facc_interp).interpolate(method='cubic', axis=0).values
        
        xc = myutils.bincen(xe)
        yc = myutils.bincen(ye)
        oldgrid_x, oldgrid_y = np.meshgrid(xc, yc)
        points = np.array([np.ravel(oldgrid_x), np.ravel(oldgrid_y)]).T
        values = np.ravel(facc_interp)
        
        grid_x, grid_y = np.mgrid[0:180:1000j, -3:0.5:1000j]
        grid_z = scipy.interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
        
        facc_smooth = filters.gaussian_filter(grid_z.T, 100)
        
        ratio = (xe[-1] - xe[0]) / (ye[-1] - ye[0])
        
        plt.sca(ax[i])
        im = plt.imshow(facc_smooth.T, origin='lower', vmin=0, vmax=1, extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect='auto', interpolation='gaussian', cmap='viridis')
        
        cs = plt.contour(facc_smooth.T, extent=(xe[0], xe[-1], ye[0], ye[-1]), levels=(0.1,0.5,0.9), colors='0.9')
        
        fmt = {}
        for j, l in enumerate(cs.levels):
            fmt[l] = '{:.0f}% accreted'.format(l*100)
        labels = plt.clabel(cs, inline=True, fontsize='small', fmt=fmt, colors='w')

        ax[i].set_xticks(np.arange(0,181,45))
        plt.xlabel('$\\vec{L}$ orientation (deg)')
        plt.ylabel('[Fe/H]')
        plt.title(survey)

        pos = ax[i].get_position()
        print(i, pos)
        if i==2:
            cax = plt.axes([0.99, pos.y0, 0.02, pos.y1 - pos.y0])
            plt.colorbar(im, cax=cax)
            plt.ylabel('Accreted fraction')
    
    plt.tight_layout()
    plt.savefig('../plots/comparison_latte_facc.pdf', bbox_inches='tight')

def comp_latte_ages():
    """"""
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    Nb = 14
    bx = np.linspace(0, 14, Nb)
    bc = myutils.bincen(bx)
    
    for i, survey in enumerate(['lattemdif', 'lattefid', 'lattehr']):
        latte = load_survey(survey)
        plt.sca(ax[i])

        dacc = 20
        accreted = latte.data['dform']>dacc

        indices = [latte.disk, latte.halo & ~accreted, latte.halo & accreted]
        colors = [red, lblue, dblue]
        labels = ['Disk', 'In situ halo', 'Accreted halo']

        for j in range(3):
            idx = np.digitize(latte.data['age'][indices[j]], bx)
            rmed = np.array([np.median(latte.data['feh'][indices[j]][idx==k]) for k in range(Nb)])[1:]
            rup = np.array([np.percentile(latte.data['feh'][indices[j]][idx==k], [84,]) if np.sum(idx==k) else np.nan for k in range(Nb)])[1:]
            rdn = np.array([np.percentile(latte.data['feh'][indices[j]][idx==k], [16,]) if np.sum(idx==k) else np.nan for k in range(Nb)])[1:]

            plt.fill_between(bc, rup, rdn, color=colors[j], alpha=0.7, label=labels[j])
            
        plt.ylim(-2.5,0.5)
        plt.xlim(13.8,0)
        plt.xlabel('Age (Gyr)')
        plt.ylabel('[Fe/H]')
        plt.legend(loc=4, fontsize='medium', frameon=False)
        plt.title(survey)

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/comparison_latte_ages.pdf', bbox_inches='tight')
