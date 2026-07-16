"""Compute power spectra

Initially written by Olivier Iffrig
Adapted by Noé Brucy.
"""

import argparse

import textwrap
from builtins import range

import numpy as np
from numba import jit
import yt

import tables as T
from numpy.fft import fftn, ifft

__generator__ = "pspec.py"
__version__ = "0.3"


def calc_k(n, nbinsk, nbig, dkbig, dim=3, saxis=2):
    """Make cubes containing the wave vectors, a list of bins and the
    associated normalization factors

    The wave vector bins are first linear, then logarithmic.

    Parameters:
    -----------
    n
        (int) number of points in each direction
    nbinsk
        (int) number of wave vector bins
    nbig
        (int) number of linear bins
    dkbig
        (float) linear bin width
    dim
        (int, default 3) dimension of the Fourier transform to be performed,
        should be either 2 or 3
    saxis
        (int, default 2) for 2D Fourier transforms, the slicing axis (i.e. the
        axis where the transform is NOT done), should be 0, 1 or 2

    Return value:
    -------------
    cubes_k
        (dict) Dictionary containing the wave vector cubes:
        'kx', 'ky', 'kz'
            Components of the wave vectors (in 2D, the one corresponding to
            saxis is always 0)
        'k'
            Norm of the wave vector
    kbins
        (numpy.array) Wave vector bins (length = nbinsk + 1): bin `i` is
        between `kbins[i]` and `kbins[i+1]`.
    norm
        (numpy.array) Number of points of the Fourier space in each wave vector
        bin (length = nbinsk)
    """

    k_alias = np.arange(n, dtype=np.float64)
    k = np.where(k_alias >= n // 2, k_alias - n, k_alias)

    if dim == 3:
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    else:
        a = [k, k, k]
        a[saxis] = np.zeros_like(k)
        kx, ky, kz = np.meshgrid(a[0], a[1], a[2], indexing="ij")

    cube_k = np.sqrt(kx**2 + ky**2 + kz**2)

    cubes_k = {"kx": kx, "ky": ky, "kz": kz, "k": cube_k}

    kmin = 1.0
    kmax = n // 2

    nbins2 = nbinsk - nbig
    kmid = kmin + nbig * dkbig

    kbins = np.concatenate(
        (
            kmin + (kmid - kmin) * (np.arange(nbig, dtype=float) / nbig),
            kmid * (kmax / kmid) ** (np.arange(nbins2 + 1, dtype=float) / nbins2),
        )
    )

    norm, _ = np.histogram(cube_k, bins=kbins)

    return cubes_k, kbins, norm


def helmholtz(cubes, var, update=False):
    r"""Perform a Helmholtz decomposition of a vector field

    The Helmholtz decomposition of a vector field $\vec{v}$ is the couple of
    vector fields $\left(\vec{v}_c, \vec{v}_s\right)$, where $\vec{\nabla}
    \cdot \vec{v}_s = 0$ and $\vec{\nabla} \cdot \vec{v}_c = \vec{0}$. 'c'
    stands for compressive and 's' for solenoidal.

    This routine uses the Fourier-domain projection method, where
    $\hat{\vec{v}}_c$ is proportional to the wave vector $\vec{k}$, and
    $\hat{\vec{v}}_s$ is orthogonal to $\vec{k}$. For the compressive
    component, only the projection of the field on the wave vector is computed

    Parameters:
    -----------
    cubes
        (dict) 3D Fourier space data cubes containing the vector field (see
        `var`) and the wave numbers (`'kx'`, `'ky'`, `'kz'`)
    var
        (str) name of the vector field, the components are assumed to
        correspond to the `var + dim` key, where `dim` is `'x'`, `'y'` or `'z'`
    update
        (bool) if True, `cubes` is updated with the data in `hcubes`

    Return value:
    -------------
    hcubes
        (dict) the components of the compressive and solenoidal fields, in
        Fourier space:
        var + 'c'
            (numpy.ndarray, ndim=3) the projection of the compressive component
            on the unit wave vector
        var + 's' + dim
            (numpy.ndarray, ndim=3) the solenoidal component, for each `dim` in
            `['x', 'y', 'z']`
    """

    knorm = np.zeros_like(cubes["k"])
    vpar = np.zeros_like(cubes[var + "x"])
    for d in ["x", "y", "z"]:
        vpar += cubes[var + d] * cubes["k" + d]
        knorm += cubes["k" + d] ** 2
    knorm = np.sqrt(knorm)

    # Prevent NaNs
    knorm[0, 0, 0] = 1.0
    vpar[0, 0, 0] = 0.0

    vpar /= knorm

    hcubes = {}
    hcubes[var + "c"] = vpar

    for d in ["x", "y", "z"]:
        hcubes[var + "s" + d] = cubes[var + d] - vpar * cubes["k" + d] / knorm

    if update:
        cubes.update(hcubes)

    return hcubes


def avg_vect(cubes, var, dim=3, saxis=2):
    """Average a vector in the cube

    In 3D, the output is a (3,)-array (global average).
    In 2D, it is a (n, 3)-array (average by slices).

    Parameters:
    -----------
    cubes
        (dict) 3D data cubes containing the vector field (see `var`)
    var
        (str) name of the vector field, the components are assumed to
        correspond to the `var + dim` key, where `dim` is `'x'`, `'y'` or `'z'`
    dim
        (int, default 3) dimension of the Fourier transform to be performed,
        should be either 2 or 3
    saxis
        (int, default 2) for 2D Fourier transforms, the slicing axis (i.e. the
        axis where the transform is NOT done), should be 0, 1 or 2

    Return value:
    -------------
    avg
        (numpy.ndarray) averaged vector, shape is (3,) in 3D, (n, 3) in 2D (n =
        number of points along the slicing axis)
    """
    if dim == 3:
        vavg = np.zeros(3, dtype=np.float64)
        for i, d in enumerate(["x", "y", "z"]):
            vavg[i] = cubes[var + d].mean()
    else:
        vavg = np.zeros((cubes[var + "x"].shape[saxis], 3), dtype=np.float64)
        axes = [0, 1, 2]
        axes.remove(saxis)
        for i, d in enumerate(["x", "y", "z"]):
            vavg[:, i] = cubes[var + d].mean(axis=tuple(axes))
    return vavg


def proj_B(cubes_k, kbins, vec, var="", dim=3, saxis=2, update=False):
    r"""Project wave vectors parallel and perpendicular to a given vector

    If the mean field value is zero, the parallel component is set to 0 and the
    perpendicular component is the wave vector itself.
    In 2D, `vec` can be either a (n, 3) or a (3,) shaped array, the additional
    dimension is taken along the slice axis.

    Parameters:
    -----------
    cubes_k
        (dict) 3D data cube containing the wave numbers (`'kx'`, `'ky'`, `'kz'`)
    kbins
        (numpy.array) wave vector bin edges (length nbins + 1), see `calc_k`
    vec
        (numpy.array) 3D: vector to project on, 2D: vector or array of vectors
    var
        (str) name of the vector in the output
    dim
        (int, default 3) dimension of the Fourier transform to be performed,
        should be either 2 or 3
    saxis
        (int, default 2) for 2D Fourier transforms, the slicing axis (i.e. the
        axis where the transform is NOT done), should be 0, 1 or 2
    update
        (bool, default False) if True, `cubes_k` is updated with the data in
        `proj_cubes`

    Return value:
    -------------
    proj_cubes
        (dict) the magnitudes of the projected wave vectors:
        'k' + var + 'par'
            (numpy.ndarray, ndim=3) the projection of the wave vector parallel
            to the mean field
        'k' + var + 'perp'
            (numpy.ndarray, ndim=3) the projection of the wave vector
            perpendicular to the mean field
    knorm_par
        (numpy.array) number of wave vectors in each bin for the parallel
        component (length nbins), see `calc_k`
    knorm_perp
        (numpy.array) number of wave vectors in each bin for the perpendicular
        component (length nbins), see `calc_k`
    """

    # we want vec_z[..., saxis] to be set to 0 without changing the argument
    vec_z = vec.copy()
    if dim == 2:
        vec_z[..., saxis] = 0.0

    # we need vec_z to be indexed correctly: we create dummy axes in the FT
    # directions
    vind = [np.newaxis, np.newaxis, np.newaxis]
    if dim == 2:
        vind[saxis] = slice(None)
    vind = tuple(vind)
    vec_z = vec_z[vind + (slice(None),)]

    vnorm = np.sqrt(np.sum(vec_z**2, axis=-1))

    kpar = np.zeros_like(cubes_k["k"])
    for i, d in enumerate(["x", "y", "z"]):
        kpar += cubes_k["k" + d] * vec_z[..., i]

    mask = np.logical_not(np.isclose(vnorm, 0))
    # Broadcast to the right shape for bool indexing in kpar
    mask_b = np.broadcast_to(mask, kpar.shape)
    vnorm_b = np.broadcast_to(vnorm, kpar.shape)
    # Normalize kpar and vnorm, correctly taking care of zeros
    kpar[mask_b] /= vnorm_b[mask_b]
    kpar[~mask_b] = 0
    vec_z[mask, :] /= vnorm[mask, np.newaxis]
    vec_z[~mask] = 0

    proj_cubes = {}
    proj_cubes["k" + var + "par"] = kpar

    kvp = "k" + var + "perp"
    proj_cubes[kvp] = np.zeros_like(kpar)
    for i, d in enumerate(["x", "y", "z"]):
        # note that vec_z[..., saxis] is 0 by construction
        proj_cubes[kvp] += (cubes_k["k" + d] - kpar * vec_z[..., i]) ** 2
    proj_cubes[kvp] = np.sqrt(proj_cubes[kvp])

    if update:
        cubes_k.update(proj_cubes)

    norm_par, _ = np.histogram(proj_cubes["k" + var + "par"], bins=kbins)
    norm_perp, _ = np.histogram(proj_cubes["k" + var + "perp"], bins=kbins)

    return proj_cubes, norm_par, norm_perp


def pcube(cube, *others):
    """Compute the power associated to a Fourier space data cube

    The power is the sum of square magnitude of each component.

    Parameters:
    -----------
    cube1, cube2, ...
        (numpy.ndarray, ndim=3) data cubes for each component

    Return value:
    -------------
    pcube
        (numpy.ndarray, ndim=3) power at each point of the Fourier space
    """

    pcube = np.real(cube * np.conj(cube))
    for c in others:
        pcube += np.real(c * np.conj(c))
    return pcube


def pspectrum(pcube, kcube, kbins, norm, nbinsf):
    """Compute the power spectrum associated to a power cube

    Parameters:
    -----------
    pcube
        (numpy.ndarray, ndim=3) power cube, see `pcube`
    kcube
        (numpy.ndarray, ndim=3) wave vector magnitude, see `calc_k`
    kbins
        (numpy.array) wave vector bin edges (length nbins + 1), see `calc_k`
    norm
        (numpy.array) number of wave vectors in each bin (length nbins), see
        `calc_k`
    nbinsf
        (int) number of power bins for a 2d-bin power spectrum, computed only
        if nbinsf > 1

    Return value:
    -------------
    pspec
        (numpy.array) power spectrum (length nbins)
    kbins
        (numpy.array) vector bin edges (the input parameter)
    pspec2
        (numpy.ndarray, ndim=2 | None) 2d-bin power spectrum (shape nbins,
        nbinsf), or None if `nbinsf <= 1`
    fbins
        (numpy.array | None) power bin edges (length nbinsf + 1), or None if
        `nbinsf <= 1`
    """

    # Flatten
    k_pts = kcube.flatten()
    f_pts = pcube.flatten()

    # Drop zeros, NaNs and infinities
    mask = np.logical_and(f_pts > 0.0, np.isfinite(f_pts))
    k_pts = k_pts[mask]
    f_pts = f_pts[mask]

    fbins = None
    pspec2 = None
    if nbinsf > 1:
        # Fourier coefficients binning
        fmin = f_pts.min()
        fmax = f_pts.max()
        if fmin == fmax:
            fmin *= 0.99
            fmax *= 1.01
        fbins = fmin * (fmax / fmin) ** (np.arange(nbinsf + 1, dtype=float) / nbinsf)

        # Binned power spectrum
        pspec2, _, _ = np.histogram2d(k_pts, f_pts, bins=(kbins, fbins))
        pspec2 /= norm[:, np.newaxis]

    # Averaged power spectrum
    pow_unnorm, _ = np.histogram(k_pts, bins=kbins, weights=f_pts)
    pspec = pow_unnorm / norm

    return pspec, kbins, pspec2, fbins


# Command-line parser ----------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Compute 2D and 3D power spectra",
    epilog=textwrap.dedent(
        """
            In the output file and node name formats, you can use the formatting
            fields:
             * %(iout)d    - output number
             * %(varname)s - variable name
             * %(dim)d     - dimension (3 for cube, 2 for slices)
            """
    ),
)
parser.add_argument(
    "repo", help="RAMSES output repository", type=str, default=".", nargs="?"
)
parser.add_argument("iouts", help="output numbers", type=int, default=[1], nargs="+")
parser.add_argument(
    "-o",
    "--outfile",
    help="output file format (see below for fields)",
    type=str,
    default="pspec_%(iout)d.h5",
    nargs="?",
)
parser.add_argument(
    "-n",
    "--nodename",
    help="node name format (see below for fields)",
    type=str,
    default="/out_%(iout)05d/d%(dim)d/%(varname)s",
)


parser.add_argument("-s", "--size", help="cube size", type=float, default=1.0)
parser.add_argument(
    "-l", "--level", help="cube level (default: levelMIN)", type=int, default=0
)
parser.add_argument(
    "-k", "--kbins", help="number of wave number bins", type=int, default=100
)
parser.add_argument(
    "-f",
    "--fbins",
    help="number of Fourier bins for k-power 2D histogram (0 to disable)",
    type=int,
    default=0,
)
parser.add_argument(
    "-K", "--kbinsbig", help="number of big wave number bins", type=int, default=9
)
parser.add_argument(
    "-d", "--dkbig", help="width of the big wave number bins", type=float, default=1.0
)
parser.add_argument(
    "-S",
    "--sliceaxis",
    help="slicing axis",
    type=str,
    choices=["x", "y", "z"],
    default="z",
)

parser.add_argument(
    "-m",
    "--magnetic",
    help="MHD simulation",
    type=bool,
    default=False,
)


parser.add_argument(
    "-r",
    "--return_arrays",
    help="Return numpy array",
    type=bool,
    default=True,
)

parser.add_argument(
    "-b",
    "--bidimensional",
    help="Also compute 2D arrays",
    type=bool,
    default=True,
)



def main(arg):
    add_pspec2 = False
    if arg.fbins > 0:
        add_pspec2 = True

    # Cube level (0 means levelmin)
    clvl = arg.level

    # Slicing axis for 2D data (x = 0, y = 1, z = 2)
    try:
        saxis = {"x": 0, "y": 1, "z": 2}[arg.sliceaxis.strip()]
    except KeyError:
        parser.error("Invalid slicing axis: %r" % arg.sliceaxis)

    # Output numbers
    iouts = arg.iouts  # tools.select_outputs(arg.repo, arg.iouts)
    
    if arg.return_arrays:
        ret = {}

    for iout in iouts:

        if arg.return_arrays:
            ret[iout] = {}
        
        print(f"Output {iout}")
        print("Load data")
        levelmax = None

        # Load output ------------------------------------------------------------------
        
        ds = yt.load(f"{arg.path}/output_{iout:05}/info_{iout:05}.txt")
        

       
        fields = [("gas", "density"), 
                  ("gas", "temperature"), 
                  ("gas", "velocity_x"),
                  ("gas", "velocity_y"),
                  ("gas", "velocity_z")]
        
        if arg.magnetic:
            fields.extend([
                ("gas", "magnetic_field_x"),
                ("gas", "magnetic_field_y"),
                ("gas", "magnetic_field_z"), 
            ]
            )
        
        var_names = [f[1] for f in fields]
        
        if arg.magnetic:
            var_names[-3:] = ["Bx", "By", "Bz"]
        
      
      
        if clvl == 0:
            clvl = ds.parameters["levelmin"]
        levelmax = ds.parameters["levelmax"]

        # Extract cubes ---------------------------------------------------------------
        
        res = 2**clvl
        
        grid = ds.r[
                    ::res*1j,
                    ::res*1j,
                    ::res*1j,
                ]
        
        cubes = {var_names[i] : grid[f].v for i,f in enumerate(fields)}
 
        
        if clvl > levelmax:
            print(
                "WARNING: adjusting cube level (%d) to match data level (%d)"
                % (clvl, levelmax)
            )
            clvl = levelmax



        print("Calculate wave numbers")
        # Wave numbers -----------------------------------------------------------------
        cubes_k, kbins, knorm = calc_k(
            1 << clvl, arg.kbins, arg.kbinsbig, arg.dkbig, 3, saxis
        )
        if arg.magnetic:
            Bavg = avg_vect(cubes, "B", dim=3)
            Bavg2 = avg_vect(cubes, "B", dim=2, saxis=saxis)
            _, knorm_Bpar, knorm_Bperp = proj_B(
                cubes_k, kbins, Bavg, "B", dim=3, update=True
            )

        print("Calculate derived quantities")
        # Additional quantities --------------------------------------------------------
        cubes["logrho"] = np.log(cubes["density"])

        if arg.magnetic:
            cubes["cos_vB"] = np.zeros_like(cubes["density"])
            BB = np.zeros_like(cubes["density"])
            vv = np.zeros_like(cubes["density"])
        for a in ["x", "y", "z"]:
            cubes["kr" + a] = cubes["density"] ** (1.0 / 3.0) * cubes["velocity_" + a]
            var_names.append("kr" + a)
            if arg.magnetic:
                cubes["cos_vB"] += cubes["velocity_" + a] * cubes["B" + a]
                BB += cubes["B" + a] ** 2
                vv += cubes["velocity_" + a] ** 2
        var_names.append("logrho")
        if arg.magnetic:
            cubes["cos_vB"] /= vv * BB
            var_names.append("cos_vB")
            del vv
            del BB

        print("3D FFT")
        # 3D Fourier transform ---------------------------------------------------------
        fcubes = {}

        for v in var_names:
            fcubes[v] = fftn(cubes[v])
        fcubes.update(cubes_k)

        # Memory cleanup ---------------------------------------------------------------
        del cubes

        print("Calculate Fourier-space derived quantities")
        # Additional quantities (Fourier domain) ---------------------------------------
        helmholtz(fcubes, "velocity_", update=True)
        helmholtz(fcubes, "kr", update=True)

        print("Compute power cubes")
        # Power cubes ------------------------------------------------------------------
        pcubes = {}
        pcubes["density"] = pcube(fcubes["density"])
        pcubes["logrho"] = pcube(fcubes["logrho"])
        pcubes["velocity"] = pcube(fcubes["velocity_x"], fcubes["velocity_y"], fcubes["velocity_z"])
        pcubes["kr"] = pcube(fcubes["krx"], fcubes["kry"], fcubes["krz"])
        pcubes["velocity_c"] = pcube(fcubes["velocity_c"])
        pcubes["velocity_s"] = pcube(fcubes["velocity_sx"], fcubes["velocity_sy"], fcubes["velocity_sz"])
        pcubes["krc"] = pcube(fcubes["krc"])
        pcubes["krs"] = pcube(fcubes["krsx"], fcubes["krsy"], fcubes["krsz"])
        pcubes["velocity_z"] = pcube(fcubes["velocity_z"])
        if arg.magnetic:
            pcubes["B"] = pcube(fcubes["Bx"], fcubes["By"], fcubes["Bz"])
            pcubes["cos_vB"] = pcube(fcubes["cos_vB"])

        print("Compute 3D power spectra")
        # 3D power spectra -------------------------------------------------------------
        vspec = [
            "density",
            "logrho",
            "velocity",
            "kr",
            "velocity_c",
            "velocity_s",
            "krc",
            "krs",
            "velocity_z",
        ]
        if arg.magnetic:
            vspec += ["B", "cos_vB"]
            
        if arg.return_arrays:
                ret[iout]["3d"] = {}
        

        for v in vspec:
            pspec, kbins, pspec2, fbins = pspectrum(
                pcubes[v], cubes_k["k"], kbins, knorm, arg.fbins
            )
            if arg.magnetic:
                pspec_Bpar, _, _, _ = pspectrum(
                    pcubes[v], cubes_k["kBpar"], kbins, knorm_Bpar, 0
                )
                pspec_Bperp, kbins, _, _ = pspectrum(
                    pcubes[v], cubes_k["kBperp"], kbins, knorm_Bperp, 0
                )

            # Save 3D power spectra          
            
            params = {"iout": iout, "varname": v, "dim": 3}
            outfile = arg.outfile % params
            outpath = arg.nodename % params

            h5fd = T.open_file(outfile, mode="a")

            for n in [
                "pspec",
                "kbins",
                "pspec2",
                "fbins",
                "norm",
                "pspec_Bpar",
                "pspec_Bperp",
                "norm_Bpar",
                "norm_Bperp",
            ]:
                try:
                    h5fd.remove_node(outpath, n, recursive=True)
                except T.NoSuchNodeError:
                    pass
                
            if arg.return_arrays:
                ret[iout]["3d"][v] = {}        
                ret[iout]["3d"][v]["pspec"] = pspec
                ret[iout]["3d"][v]["kbins"] = kbins
                ret[iout]["3d"][v]["norm"] = knorm
                
                if arg.magnetic:
                    ret[iout]["3d"][v]["pspec_Bpar"] = pspec_Bpar
                    ret[iout]["3d"][v]["norm_Bpar"] = knorm_Bpar
                    ret[iout]["3d"][v]["pspec_Bperp"] = pspec_Bperp
                    ret[iout]["3d"][v]["norm_Bperp"] = knorm_Bpar
                

            h5fd.create_array(outpath, "pspec", pspec, createparents=True)
            h5fd.create_array(outpath, "kbins", kbins, createparents=True)
            if add_pspec2:
                h5fd.create_array(outpath, "pspec2", pspec2, createparents=True)
                h5fd.create_array(outpath, "fbins", fbins, createparents=True)
            h5fd.create_array(outpath, "norm", knorm, createparents=True)

            if arg.magnetic:
                h5fd.create_array(outpath, "pspec_Bpar", pspec_Bpar, createparents=True)
                h5fd.create_array(
                    outpath, "pspec_Bperp", pspec_Bperp, createparents=True
                )
                h5fd.create_array(outpath, "norm_Bpar", knorm_Bpar, createparents=True)
                h5fd.create_array(
                    outpath, "norm_Bperp", knorm_Bperp, createparents=True
                )

            try:
                h5fd.remove_node(outpath, "meta", recursive=True)
            except T.NoSuchNodeError:
                pass

            h5fd.close()

        # Memory cleanup ---------------------------------------------------------------
        del pcubes
        
        if arg.bidimensional:

            print("Calculate 2D wave numbers")
            # 2D wave numbers --------------------------------------------------------------
            cubes_k, kbins, knorm = calc_k(
                1 << clvl, arg.kbins, arg.kbinsbig, arg.dkbig, 2, saxis
            )
            if arg.magnetic:
                _, knorm_Bpar, knorm_Bperp = proj_B(
                    cubes_k, kbins, Bavg2, "B", dim=2, saxis=saxis, update=True
                )

            print("Project 3D -> 2D")
            # 3D -> 2D ---------------------------------------------------------------------
            fcubes2 = {}

            vars2D = [
                "density",
                "logrho",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "krx",
                "kry",
                "krz",
                "velocity_c",
                "velocity_sx",
                "velocity_sy",
                "velocity_sz",
                "krc",
                "krsx",
                "krsy",
                "krsz",
            ]
            if arg.magnetic:
                vars2D += ["Bx", "By", "Bz", "cos_vB"]
            for v in vars2D:
                fcubes2[v] = ifft(fcubes[v], axis=saxis)

            # Memory cleanup ---------------------------------------------------------------
            del fcubes

            print("Compute 2D power cubes")
            # 2D power cubes ---------------------------------------------------------------
            pcubes2 = {}
            pcubes2["density"] = pcube(fcubes2["density"])
            pcubes2["logrho"] = pcube(fcubes2["logrho"])
            pcubes2["velocity"] = pcube(fcubes2["velocity_x"], fcubes2["velocity_y"], fcubes2["velocity_z"])
            pcubes2["kr"] = pcube(fcubes2["krx"], fcubes2["kry"], fcubes2["krz"])
            pcubes2["velocity_c"] = pcube(fcubes2["velocity_c"])
            pcubes2["velocity_s"] = pcube(fcubes2["velocity_sx"], fcubes2["velocity_sy"], fcubes2["velocity_sz"])
            pcubes2["krc"] = pcube(fcubes2["krc"])
            pcubes2["krs"] = pcube(fcubes2["krsx"], fcubes2["krsy"], fcubes2["krsz"])

            if arg.magnetic:
                pcubes2["B"] = pcube(fcubes2["Bx"], fcubes2["By"], fcubes2["Bz"])
                pcubes2["cos_vB"] = pcube(fcubes2["cos_vB"])

            print("Compute 2D power spectra")
            # 2D power spectra -------------------------------------------------------------
            ns = 2**clvl
            f = "_%%(i)0%dd" % (np.floor(np.log10(ns)) + 1)
            
            if arg.return_arrays:
                ret[iout]["2d"] = {}
            
            
            for v in list(pcubes2.keys()):
                for i in range(ns):
                    pspec, kbins, pspec2, fbins = pspectrum(
                        pcubes2[v][:, :, i], cubes_k["k"][:, :, i], kbins, knorm, arg.fbins
                    )

                    if arg.magnetic:
                        pspec_Bpar, _, _, _ = pspectrum(
                            pcubes2[v][:, :, i],
                            cubes_k["kBpar"][:, :, i],
                            kbins,
                            knorm_Bpar,
                            0,
                        )
                        pspec_Bperp, kbins, _, _ = pspectrum(
                            pcubes2[v][:, :, i],
                            cubes_k["kBperp"][:, :, i],
                            kbins,
                            knorm_Bperp,
                            0,
                        )

                    # Save 2D power spectra
                    suff = f % {"i": i}
                    params = {"iout": iout, "varname": v, "dim": 2}
                    outfile = arg.outfile % params
                    outpath = arg.nodename % params

                    h5fd = T.open_file(outfile, mode="a")

                    for n in [
                        "pspec",
                        "kbins",
                        "pspec2",
                        "fbins",
                        "norm",
                        "pspec_Bpar",
                        "pspec_Bperp",
                        "norm_Bpar",
                        "norm_Bperp",
                    ]:
                        try:
                            h5fd.remove_node(outpath, n + suff, recursive=True)
                        except T.NoSuchNodeError:
                            pass
                        
                    if arg.return_arrays:
                        ret[iout]["2d"][v] = {}        
                        ret[iout]["2d"][v]["pspec"] = pspec
                        ret[iout]["2d"][v]["kbins"] = kbins
                        ret[iout]["2d"][v]["norm"] = knorm
                        
                        if arg.magnetic:
                            ret[iout]["3d"][v]["pspec_Bpar"] = pspec_Bpar
                            ret[iout]["3d"][v]["norm_Bpar"] = knorm_Bpar
                            ret[iout]["3d"][v]["pspec_Bperp"] = pspec_Bperp
                            ret[iout]["3d"][v]["norm_Bperp"] = knorm_Bpar
                        
                        

                    h5fd.create_array(outpath, "pspec" + suff, pspec, createparents=True)

                    if arg.magnetic:
                        h5fd.create_array(
                            outpath, "pspec_Bpar" + suff, pspec_Bpar, createparents=True
                        )
                        h5fd.create_array(
                            outpath, "pspec_Bperp" + suff, pspec_Bperp, createparents=True
                        )

                    h5fd.create_array(outpath, "kbins" + suff, kbins, createparents=True)
                    if add_pspec2:
                        h5fd.create_array(
                            outpath, "pspec2" + suff, pspec2, createparents=True
                        )
                        h5fd.create_array(
                            outpath, "fbins" + suff, fbins, createparents=True
                        )
                    h5fd.create_array(outpath, "norm" + suff, knorm, createparents=True)

                    if arg.magnetic:
                        h5fd.create_array(
                            outpath, "norm_Bpar" + suff, knorm_Bpar, createparents=True
                        )
                        h5fd.create_array(
                            outpath, "norm_Bperp" + suff, knorm_Bperp, createparents=True
                        )

                    try:
                        h5fd.remove_node(outpath, "meta", recursive=True)
                    except T.NoSuchNodeError:
                        pass


                    h5fd.close()
    return ret


if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)


def pspec(**kwargs):
    arg = parser.parse_args("1")
    for kwarg in kwargs:
        setattr(arg, kwarg, kwargs[kwarg])
    return main(arg)
