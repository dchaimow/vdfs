import numpy as np
import nibabel as nib
from scipy.optimize import root_scalar, minimize_scalar
from numba import jit
from numba.typed import List
from joblib import Parallel, delayed, parallel_backend
import os


def is_gifti(fname):
    """Tests if a file is a gifti file (based on its extension)."""
    return str(fname).endswith(".gii")


def read_surf_geometry(surf_file):
    """
    Reads vertex coordinates and faces of a freesurfer or gifti surface.
    Returns coords, faces and the freesurfer to scanner transformation. For freesurfer surfaces this
    is a translation by the c_ras stored in the surface file, gifti surface coordinates are assumed
    to already be in scanner space (identity transformation).
    """
    fs_to_scanner = np.eye(4)
    if is_gifti(surf_file):
        surf_gii = nib.load(surf_file)
        coords = surf_gii.agg_data("NIFTI_INTENT_POINTSET")
        faces = surf_gii.agg_data("NIFTI_INTENT_TRIANGLE")
    else:
        coords, faces, volume_info = nib.freesurfer.read_geometry(
            surf_file, read_metadata=True
        )
        if "cras" not in volume_info:
            raise ValueError(f"freesurfer surface {surf_file} contains no volume info (c_ras), "
                             "cannot transform it to scanner space")
        fs_to_scanner[:3, 3] = volume_info["cras"]
    return coords, faces, fs_to_scanner


def read_label_vertices(label_file):
    """
    Reads the vertex indices of a label. Either a freesurfer label file or a gifti file
    (e.g. .label.gii or .func.gii) that is interpreted as a vertex mask (all vertices with nonzero values).
    """
    if is_gifti(label_file):
        return np.flatnonzero(nib.load(label_file).agg_data())
    return nib.freesurfer.read_label(label_file)


def calc_vertex_area(coords, faces):
    """
    Calculates the area of each vertex as one third of the summed areas of all triangles that contain it.
    This is identical to the freesurfer ?h.area and ?h.area.pial files (up to float32 precision).
    Coordinates should be in mm (i.e. not in voxel grid space) and all faces of the surface should be
    included (i.e. before restricting to a label).
    """
    coords = np.asarray(coords, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area = np.zeros(coords.shape[0])
    for i in range(3):
        np.add.at(area, faces[:, i], face_area / 3)
    return area


def load_vertex_area(area_file=None, surf_file=None):
    """
    Loads vertex areas from a freesurfer area file (e.g. lh.area) or a gifti metric file. If no area file
    is provided, the vertex areas are calculated from the surface (see calc_vertex_area).
    """
    if area_file is not None:
        if is_gifti(area_file):
            return nib.load(area_file).agg_data()
        return nib.freesurfer.read_morph_data(area_file)
    coords, faces, _ = read_surf_geometry(surf_file)
    return calc_vertex_area(coords, faces)


def calc_mid_vertex_area(surf_white_file, surf_pial_file):
    """
    Calculates the vertex areas of the midthickness surface, whose vertices lie halfway between corresponding
    white and pial vertices (the intermediate surface at equi-distance depth 0.5). Used by the quadratic area
    model (see alpha_from_beta_quadratic). Note that freesurfer's ?h.area.mid is not this area, but the mean of
    ?h.area and ?h.area.pial.
    """
    coords_white, faces, _ = read_surf_geometry(surf_white_file)
    coords_pial, faces_pial, _ = read_surf_geometry(surf_pial_file)
    if coords_white.shape != coords_pial.shape or not np.array_equal(faces, faces_pial):
        raise ValueError("white and pial surfaces need to have the same vertices and triangles")
    coords_white = np.asarray(coords_white, dtype=np.float64)
    return calc_vertex_area((coords_white + coords_pial) / 2, faces)


def load_fs_surf_in_scanner_space(surf_file):
    """
    Loads a freesurfer or gifti surface in scanner space (only doing freesurfer to scanner transformation).
    """
    surf = load_fs_surf_in_grid(surf_file, np.eye(4))
    return surf


def load_fs_surf_in_grid(surf_file, grid_to_scanner, label_file=None):
    """
    Loads a freesurfer or gifti surface and transforms coordinates from freesurfer space to grid space,
    by using the grid_to_scanner transformation (e.g. affine function a functional nifti image).
    If a label file is provided, only triangles with all vertices inside the label are kept. Besides
    vertices and triangles, the returned surface contains the original index of each triangle
    ("tris_idx") and the original number of triangles ("nTrisOrig").
    """
    # load freesurfer or gifti surface
    coords, faces, fs_to_scanner = read_surf_geometry(surf_file)
    n_tris_orig = faces.shape[0]
    tris_idx = np.arange(n_tris_orig)

    # if label file provided
    if label_file is not None:
        label = read_label_vertices(label_file)
        vertex_mask = np.zeros(coords.shape[0], dtype=bool)
        vertex_mask[label] = True
        # remove all triangles that contain vertices that are not in the label
        face_mask = np.all(vertex_mask[faces],axis=1)
        faces = faces[face_mask]
        tris_idx = tris_idx[face_mask]

    surf = dict()
    surf["nVertices"] = coords.shape[0]
    surf["nTris"] = faces.shape[0]
    surf["nTrisOrig"] = n_tris_orig
    grid_to_scanner = np.matrix(grid_to_scanner)
    fs_to_voxel = grid_to_scanner.I @ fs_to_scanner
    surf["vertices"] = np.array(fs_to_voxel.dot(
        np.hstack((coords, np.ones((surf["nVertices"], 1)))).T).T[:, :3])
    surf["tris"] = np.array(faces, dtype=np.int64)
    surf["tris_idx"] = tris_idx
    return surf


def merge_surfaces(surf1, surf2):
    """
    Merges separate representations of two surfaces into one. Original triangle indices of the second
    surface are offset by the original number of triangles of the first surface.
    """
    surf = dict()
    surf["nTris"] = surf1["nTris"] + surf2["nTris"]
    surf["nVertices"] = surf1["nVertices"] + surf2["nVertices"]
    surf["vertices"] = np.array(np.concatenate((surf1["vertices"], surf2["vertices"])))
    surf["tris"] = np.concatenate((surf1["tris"], surf2["tris"] + surf1["nVertices"]))
    n_tris_orig_1 = surf1.get("nTrisOrig", surf1["nTris"])
    surf["nTrisOrig"] = n_tris_orig_1 + surf2.get("nTrisOrig", surf2["nTris"])
    surf["tris_idx"] = np.concatenate((surf1.get("tris_idx", np.arange(surf1["nTris"])),
                                       surf2.get("tris_idx", np.arange(surf2["nTris"])) + n_tris_orig_1))
    return surf


@jit(nopython=True)
def alpha_from_beta(beta, aw, ap):
    """Calculates the relative equi-distance depth alpha (distance fraction from the white surface) at which
    the volume fraction below the intermediate surface equals the equi-volume depth beta, using the local area
    of the bottom (white, aw) and top (pial, ap) vertex.

    The area is assumed to change linearly with distance, A(alpha) = aw + (ap - aw) alpha (Waehnert et al., 2014,
    NeuroImage 93, Eq. 10; surface_tools by K. Wagstyl). Then the volume fraction below alpha is
    (2 aw alpha + (ap - aw) alpha^2) / (aw + ap), and solving it for alpha gives
    alpha = (-aw + sqrt((1 - beta) aw^2 + beta ap^2)) / (ap - aw). The equivalent form used here,
    alpha = beta (aw + ap) / (aw + sqrt((1 - beta) aw^2 + beta ap^2)), is also stable for aw close to ap.
    If both areas are zero, alpha equals beta (equi-distance).

    vdfs <= 1.1 used alpha = (2 ap beta - ap beta^2 + aw beta^2) / (ap + aw). That is the inverse of the
    surface_tools function (it maps a distance fraction to a volume fraction, measured from the pial surface)
    and agrees with the correct relation only to first order in the area difference.
    """
    if ap + aw == 0:
        return beta
    denominator = aw + np.sqrt((1 - beta) * aw**2 + beta * ap**2)
    if denominator == 0:  # aw = 0 and beta = 0
        return 0.0
    return beta * (aw + ap) / denominator


@jit(nopython=True)
def alpha_from_beta_quadratic(beta, aw, ap, am):
    """Calculates the relative equi-distance depth alpha (distance fraction from the white surface) at which
    the volume fraction below the intermediate surface equals the equi-volume depth beta, assuming that the
    area changes quadratically with distance:

        A(alpha) = aw (1 - alpha)^2 + 2 am alpha (1 - alpha) + ap alpha^2

    aw, ap: vertex areas on the white and pial surface; am: middle coefficient, am = 2 a_mid - (aw + ap) / 2,
    where a_mid is the vertex area on the midthickness surface (the intermediate surface at alpha = 0.5).

    Why quadratic: vdfs moves every vertex along the straight line from its white to its pial position, so the
    edges of an intermediate triangle are linear in alpha and its (vector) area, the cross product of two edges,
    is quadratic in alpha. The scalar area is exactly quadratic when the white and pial triangles are parallel
    and very nearly so otherwise; vertex areas (sums of triangle areas) inherit this. Three areas determine the
    quadratic, and the midthickness area provides the third. For parallel surfaces the area along a normal
    changes as A(t) = A(0) (1 - 2 H t + K t^2) (mean and Gaussian curvature H, K), so the quadratic term
    matters wherever the cortex is curved in two directions; the linear-area model (alpha_from_beta) is the
    special case a_mid = (aw + ap) / 2, which is exact only for zero Gaussian curvature (e.g. a cylinder).

    The volume fraction below alpha, V(alpha) / V(1) with V(alpha) = c1 alpha + c2 alpha^2 + c3 alpha^3
    (c1 = aw, c2 = am - aw, c3 = (aw - 2 am + ap) / 3), is solved for alpha with Newton's method, starting from
    the linear-area solution. If the quadratic area is not positive throughout (degenerate geometry), the
    linear-area solution is returned.
    """
    c3 = (aw - 2 * am + ap) / 3
    linear = alpha_from_beta(beta, aw, ap)
    if abs(c3) <= 1e-12 * (abs(aw) + abs(ap)):  # linear-area model (up to rounding)
        return linear
    c1, c2 = aw, am - aw
    total = c1 + c2 + c3
    if total <= 0:
        return linear
    target = beta * total
    alpha = linear
    for _ in range(50):
        dv = (3 * c3 * alpha + 2 * c2) * alpha + c1
        if dv <= 0:
            return linear
        step = (((c3 * alpha + c2) * alpha + c1) * alpha - target) / dv
        new_alpha = min(1.0, max(0.0, alpha - step))
        if abs(new_alpha - alpha) < 1e-14:
            return new_alpha
        alpha = new_alpha
    return alpha


@jit(nopython=True)
def intermediate_plane(a1, b1, c1, a2, b2, c2, alpha_a, alpha_b, alpha_c):
    """Calculates vertices of intermediate plane triangled defined by relative depths 
    for all three vertices independently"""
    ma = a1 + alpha_a * (a2 - a1)
    mb = b1 + alpha_b * (b2 - b1)
    mc = c1 + alpha_c * (c2 - c1)
    return ma, mb, mc


@jit(nopython=True)
def equidist_plane(a1, b1, c1, a2, b2, c2, alpha):
    """The vertices of equidistant intermediate plane triangles are calculated by using the
    same relative depth for all three vertices."""
    return intermediate_plane(a1, b1, c1, a2, b2, c2, alpha, alpha, alpha)


@jit(nopython=True)
def equivol_plane(a1, b1, c1, a2, b2, c2, beta, aw_a, aw_b, aw_c, ap_a, ap_b, ap_c, am_a, am_b, am_c):
    """The vertices of equi-volume intermediate plane triangles are calculated by using relative
    depths for each vertex that depend on the local area of the bottom (white, aw) and top (pial, ap)
    vertices and the middle coefficient am of the quadratic area model (see alpha_from_beta_quadratic;
    am = (aw + ap) / 2 gives the linear-area model)."""
    alpha_a = alpha_from_beta_quadratic(beta, aw_a, ap_a, am_a)
    alpha_b = alpha_from_beta_quadratic(beta, aw_b, ap_b, am_b)
    alpha_c = alpha_from_beta_quadratic(beta, aw_c, ap_c, am_c)
    return intermediate_plane(a1, b1, c1, a2, b2, c2, alpha_a, alpha_b, alpha_c)


# Points closer than this fraction of the edge length to a triangle edge count as being on the edge.
# This accepts points on the boundary between neighbouring prisms, which rounding in the depth calculation
# can otherwise place marginally outside of all of them (#10).
EDGE_TOLERANCE = 1e-6


@jit(nopython=True)
def same_side(p1, p2, a, b):
    """Tests if two points p1 and p2 are on the same side of the line defined by a and b.
    p1 also counts as being on the same side if it lies on the line (within EDGE_TOLERANCE)."""
    cp1 = np.cross(b - a, p1 - a)
    cp2 = np.cross(b - a, p2 - a)
    if np.dot(cp1, cp2) >= 0:
        return True
    # distance of p1 from the line, |cp1| / |b - a|, compared with EDGE_TOLERANCE * |b - a|
    edge_length_sq = np.dot(b - a, b - a)
    return np.dot(cp1, cp1) <= (EDGE_TOLERANCE * edge_length_sq) ** 2


@jit(nopython=True)
def test_point_inside_triangle(p, a, b, c):
    """Tests if a point p is inside a triangle defined by vertices a, b, c (boundary included)."""
    if same_side(p, a, b, c) and same_side(p, b, a, c) and same_side(p, c, a, b):
        return True
    else:
        return False


@jit(nopython=True)
def plane_side(p, ma, mb, mc):
    """ Calculates the signed distance of a point p to a plane defined by three vertices ma, mb, mc."""
    return np.dot(p - ma, np.cross(mb - ma, mc - ma))


def find_depth_through_point(p, depth_plane_function):
    """
    This function attempts to find the depth of a point p through a prism defined by a depth plane function.
    It does so by numerically solving the intersection equation of the point with the depth plane function.
    If the intersection is inside the prism, the depth is returned, otherwise None.
    """
    def f(x):
        """ 
        Define a function that returns the signed distance of a point x to the depth plane
        as defined by the vertices returned by the supplied depth_plane_function, which is
        parametrized by a relative depth parameter x. Finding the root of this function is
        quivalent to finding the depth at which the point x intersects with the corresponding
        depth plane.
        """
        ma, mb, mc = depth_plane_function(x)
        return plane_side(p, ma, mb, mc)
    
    # we are interested in relative depth values between 0 and 1
    # by first finding the minimum and maximum of the function we can
    # determine the intervals where the function changes sign
    # so that we can use the brentq method to find the unique root in these intervals
    x_min = minimize_scalar(f, bounds=[0, 1], method="bounded").x
    x_max = minimize_scalar(lambda x: -f(x), bounds=[0, 1], method="bounded").x
    x_grid = [0, min(x_min, x_max), max(x_min, x_max), 1]
    f_x_grid = [f(x_grid[0]), f(x_grid[1]), f(x_grid[2]), f(x_grid[3])]

    for i in [1, 0, 2]:
        # if there is a sign change between two grid points, we have a root in between
        if np.sign(f_x_grid[i]) != np.sign(f_x_grid[i + 1]):
            # we find the root of the function in the interval
            x_zero = root_scalar(f, 
                                 method="brentq", 
                                 bracket=[x_grid[i], x_grid[i + 1]]).root
            # now that we have an intersection we need to check if the point is inside the 
            # triangle defined by the vertices of the depth plane
            ma, mb, mc = depth_plane_function(x_zero)
            if test_point_inside_triangle(p, ma, mb, mc):
                return x_zero
    return None


@jit(nopython=True)
def numba_calc_bounding_prisms(faces, vertices_white, vertices_pial, n_x, n_y, n_z):
    """
    This function calculates for each voxel in the grid a list of prism indices that 
    potentially contain the voxel. This is done by finding the bounding box of each prism
    in voxel space and adding the prism index to all voxels that are inside this bounding box.
    Thus, the center of a voxel can maximally be contained in one of the prisms that are
    in the list for this voxel. This is a necessary condition for the voxel to be contained
    in the prism. The actual containment is then checked in the calc_depth_from_surfaces_voxelidx
    function.
    The result is a 4D array with dimensions (n_x, n_y, n_z, n_prisms) where n_prisms is the
    maximum number of prisms that can contain a voxel. Unused entries are -1.
    """
    # we initialize the array that will hold the list of prisms indices for each voxel
    # as length 1, we will extend it as prisms are added, -1 is used as a sentinel value
    # (0 cannot be used, because it is the index of the first prism)
    bounding_prisms = np.full((n_x, n_y, n_z, 1), -1, dtype=np.int64)

    # for each prism we find the bounding box in voxel space, that is the range of 
    # voxel indices that contain the prism and add the prism index to lists of all these voxels
    for i_prism, prism_vertices_indcs in enumerate(faces):
        prism_points = np.concatenate(
            (vertices_white[prism_vertices_indcs], vertices_pial[prism_vertices_indcs]))
        
        # find bounding box in voxel space
        max_x_idx = int(np.round(np.max(prism_points[:, 0])))
        min_x_idx = int(np.round(np.min(prism_points[:, 0])))
        max_y_idx = int(np.round(np.max(prism_points[:, 1])))
        min_y_idx = int(np.round(np.min(prism_points[:, 1])))
        max_z_idx = int(np.round(np.max(prism_points[:, 2])))
        min_z_idx = int(np.round(np.min(prism_points[:, 2])))

        # get ranges of voxel indices that are inside the bounding box
        x_idx_range = np.arange(min_x_idx, max_x_idx + 1)
        y_idx_range = np.arange(min_y_idx, max_y_idx + 1)
        z_idx_range = np.arange(min_z_idx, max_z_idx + 1)

        # remove those outside of the slab
        x_idx_range = x_idx_range[(x_idx_range >= 0) & (x_idx_range < n_x)]
        y_idx_range = y_idx_range[(y_idx_range >= 0) & (y_idx_range < n_y)]
        z_idx_range = z_idx_range[(z_idx_range >= 0) & (z_idx_range < n_z)]

        # add prism index to all voxels that are inside the bounding box
        for x_idx in x_idx_range:
            for y_idx in y_idx_range:
                for z_idx in z_idx_range:
                    # find next available list index
                    # first check if the index list of any voxel is full already
                    available_indcs = np.argwhere(bounding_prisms[x_idx, y_idx, z_idx, :] == -1)
                    if available_indcs.size == 0:
                        # if full, extend the prism list by doubling its length
                        old_length = int(bounding_prisms.shape[3])
                        bounding_prisms = np.concatenate((bounding_prisms,
                                                          np.full((n_x, n_y, n_z, old_length), -1,
                                                                  dtype=np.int64)),
                                                          axis=3)
                        next_available_idx = old_length
                    else:
                        next_available_idx = int(available_indcs[0][0])
                    # add prism index to the list of the voxel
                    bounding_prisms[x_idx, y_idx, z_idx, next_available_idx] = int(i_prism)
    return bounding_prisms


def calc_depth_from_surfaces_voxelidx(voxelidx, 
                                      faces, vertices_white, vertices_pial, area_white, area_pial,
                                      bounding_prisms, p=None, method='equivol', area_mid=None):
    """
    This is the main function that calculates the depth of a voxel with coordinates given by p.
    It numerically solves an intersection equation for the voxel center with a parametrization
    of triangular planes that move between the white and pial faces of the prisms that potentially
    contain the voxel and checks wheter this intersection in inside the triangle. The parametrization is 
    either based on equi-volume or equi-distance depth. The candidate prisms are given by the bounding_prisms
    entry with 3D index voxelidx. If p is None then it is assumed that voxelidx is the voxel center
    in voxel grid coordinates. This is generally the case when we compute the depths for all voxels in a grid.
    In that case we use that grid and its resolution to determine the bounding prisms. If p is not None
    bounding_prisms should have been calculated for a grid of arbitrary resolution that contains the point p and 
    the voxelidx is the artifical voxel within wich p is contained.
    As soon as a solution is found that is inside the prism, the depth is returned.
    For method="equivol", area_mid (vertex areas of the midthickness surface) selects the quadratic area model;
    without it, the linear-area model is used (see alpha_from_beta_quadratic).
    """
    if method not in ("equivol", "equidist"):
        raise ValueError(f"method must be 'equivol' or 'equidist', not {method!r}")
    x_idx, y_idx, z_idx = voxelidx
    if p is None:
        p = np.array(voxelidx)
    prism_idx = 0
    # go through all prisms that potentially contain the voxel
    # (check the list length first, a full list has no -1 sentinel at its end)
    while (prism_idx < bounding_prisms.shape[3]
           and bounding_prisms[x_idx, y_idx, z_idx, prism_idx] != -1):
        # get vertices of the prism 
        canditate_prism_idx = bounding_prisms[x_idx, y_idx, z_idx, prism_idx]
        a1 = vertices_white[faces[canditate_prism_idx, 0], :]
        b1 = vertices_white[faces[canditate_prism_idx, 1], :]
        c1 = vertices_white[faces[canditate_prism_idx, 2], :]
        a2 = vertices_pial[faces[canditate_prism_idx, 0], :]
        b2 = vertices_pial[faces[canditate_prism_idx, 1], :]
        c2 = vertices_pial[faces[canditate_prism_idx, 2], :]

        # setup the depth plane function based on the method
        if method == "equivol":
            aw_a = area_white[faces[canditate_prism_idx, 0]]
            aw_b = area_white[faces[canditate_prism_idx, 1]]
            aw_c = area_white[faces[canditate_prism_idx, 2]]
            ap_a = area_pial[faces[canditate_prism_idx, 0]]
            ap_b = area_pial[faces[canditate_prism_idx, 1]]
            ap_c = area_pial[faces[canditate_prism_idx, 2]]
            # middle coefficient of the quadratic area model (linear-area model without midthickness areas)
            if area_mid is None:
                am_a, am_b, am_c = (aw_a + ap_a) / 2, (aw_b + ap_b) / 2, (aw_c + ap_c) / 2
            else:
                am_a = 2 * area_mid[faces[canditate_prism_idx, 0]] - (aw_a + ap_a) / 2
                am_b = 2 * area_mid[faces[canditate_prism_idx, 1]] - (aw_b + ap_b) / 2
                am_c = 2 * area_mid[faces[canditate_prism_idx, 2]] - (aw_c + ap_c) / 2

            depth_plane_function = lambda beta: equivol_plane(a1, b1, c1, a2, b2, c2, beta,
                                                              aw_a, aw_b, aw_c,
                                                              ap_a, ap_b, ap_c,
                                                              am_a, am_b, am_c)
        elif method == "equidist":
            depth_plane_function = lambda alpha: equidist_plane(a1, b1, c1, a2, b2, c2, alpha)
        # find depth through voxel center 
        # (by solving intersection equation and checking if the intersection is inside the prism)
        d = find_depth_through_point(p, depth_plane_function)

        if d is not None:
            return d, float(canditate_prism_idx)

        prism_idx = prism_idx + 1
    return np.nan, np.nan


@jit(nopython=True)
def numba_assign_calc_depth_results(results, roi, grid_size):
    """
    Helper function to assign the results of the voxel depth calculation to the correct voxel in the grid.
    """
    depths = np.empty(grid_size)
    columns = np.empty(grid_size)
    depths[:] = np.nan
    columns[:] = np.nan
    for roi_idx, result in zip(roi, results):
        depths[roi_idx[0], roi_idx[1], roi_idx[2]] = result[0]
        columns[roi_idx[0], roi_idx[1], roi_idx[2]] = result[1]
    return depths, columns


def calc_depth_from_surfaces_on_grid(surf_white, area_white, surf_pial, area_pial,
                                     n_x, n_y, n_z, method, n_jobs=32, area_mid=None):
    """
    Calculate voxel grid depths from surfaces given all surface information (assumed to be in voxel grid space).
    White and pial surface need to have the same vertices and triangles (i.e. correspond to each other),
    vertex areas need to be provided for all surface vertices. For method="equivol", area_mid (vertex areas of
    the midthickness surface, see calc_mid_vertex_area) selects the quadratic area model; without it, the
    linear-area model is used.
    Returns depths and columns, the index of the triangle (prism) that contains each voxel center. If the
    surfaces contain the original triangle indices ("tris_idx", e.g. when restricted to a label),
    these are used for the columns.
    """
    if surf_white["nVertices"] != surf_pial["nVertices"] or not np.array_equal(
            surf_white["tris"], surf_pial["tris"]):
        raise ValueError("white and pial surfaces need to have the same vertices and triangles")
    if len(area_white) != surf_white["nVertices"] or len(area_pial) != surf_pial["nVertices"]:
        raise ValueError("number of vertex areas does not match number of surface vertices")
    if area_mid is not None and len(area_mid) != surf_white["nVertices"]:
        raise ValueError("number of midthickness vertex areas does not match number of surface vertices")

    faces = surf_pial["tris"]
    vertices_white = surf_white["vertices"]
    vertices_pial = surf_pial["vertices"]

    # For each voxel, calculate a list of prism indices that potentially contain the voxel
    # so that we only to solve intersection equations for these prisms
    bounding_prisms = numba_calc_bounding_prisms(faces,
                                                 vertices_white, vertices_pial,
                                                 n_x, n_y, n_z)

    # only need to calculate depths for voxels that potentially are inside a prism
    # (lists are filled from the start, so the first entry is used if any is)
    roi = np.argwhere(bounding_prisms[:, :, :, 0] != -1)
    if len(roi) == 0:
        return np.full((n_x, n_y, n_z), np.nan), np.full((n_x, n_y, n_z), np.nan)

    # calculate depths in parallel by calling calc_depth_from_surfaces_voxelidx for each voxel
    # in the roi
    with parallel_backend("loky", inner_max_num_threads=1):
        results = List(
            Parallel(n_jobs=n_jobs)(
                delayed(
                    lambda roi_idx: calc_depth_from_surfaces_voxelidx(
                        roi_idx,
                        faces,
                        vertices_white,
                        vertices_pial,
                        area_white,
                        area_pial,
                        bounding_prisms,
                        None,
                        method,
                        area_mid)
                )(roi_idx) for roi_idx in roi))

    # take results from parallel calculation and assign them to the correct voxel in the grid
    depths, columns = numba_assign_calc_depth_results(
        results, roi, bounding_prisms.shape[:3])

    # translate prism indices into original triangle indices
    if "tris_idx" in surf_pial:
        in_prism = ~np.isnan(columns)
        columns[in_prism] = surf_pial["tris_idx"][columns[in_prism].astype(np.int64)]

    return depths, columns


def voxeldepths_from_surfaces(
    surf_white_lh_file,
    surf_pial_lh_file,
    surf_white_rh_file,
    surf_pial_rh_file,
    volume_file,
    depths_fname,
    columns_fname,
    *,
    label_lh_fname=None,
    label_rh_fname=None,
    method="equivol",
    area_model="quadratic",
    upsample_factor=None,
    n_jobs=32,
    force=False):
    """
    Calculate voxel depths from surfaces and save them to nifti files (main entry function).

    Surfaces can be freesurfer (e.g. lh.white, lh.pial) or gifti (.gii) files. Freesurfer surface
    coordinates are transformed to scanner space using the c_ras stored in the surface file, gifti
    surface coordinates are assumed to already be in scanner space.
    Vertex areas (needed for method="equivol") are calculated from the surfaces, which gives the same
    values as the freesurfer ?h.area and ?h.area.pial files. To use area files instead, use
    process_voxeldepth_from_surfaces.

    surf_white_lh_file, surf_pial_lh_file, surf_white_rh_file, surf_pial_rh_file: surface files
    volume_file: nifti file that determines the voxel grid and affine of the output
    depths_fname: output nifti file with voxel depths (0 = white, 1 = pial, NaN outside of the ribbon)
    columns_fname: output nifti file with the index of the triangle (prism) that contains each voxel center
        (if both hemispheres are used, right hemisphere triangle indices are offset by the number of
        left hemisphere triangles)
    label_lh_fname, label_rh_fname: optional freesurfer label or gifti mask (.gii, nonzero vertices) files
        that restrict the calculation to triangles inside the label. If a label is provided for only one
        hemisphere, the other hemisphere is ignored.
    method: "equivol" (equi-volume) or "equidist" (equi-distance) depths
    area_model: how the area of a column changes between white and pial surface for equi-volume depths:
        "quadratic" (default): quadratic in the distance fraction, determined by the vertex areas of the white,
            midthickness and pial surfaces. This is exact for the geometry vdfs uses (vertices moving on straight
            lines from white to pial), see alpha_from_beta_quadratic.
        "linear": linear in the distance fraction (Waehnert et al., 2014; surface_tools), exact only where the
            cortex is curved in at most one direction.
    upsample_factor: optional upsampling of the voxel grid
    n_jobs: number of parallel jobs
    force: if False and both output files exist, they are loaded instead of recalculated

    Returns the depths and columns nifti images.
    """
    return process_voxeldepth_from_surfaces(
        surf_white_lh_file, None, surf_pial_lh_file, None,
        surf_white_rh_file, None, surf_pial_rh_file, None,
        volume_file, depths_fname, columns_fname,
        label_lh_fname=label_lh_fname, label_rh_fname=label_rh_fname, method=method, area_model=area_model,
        upsample_factor=upsample_factor, n_jobs=n_jobs, force=force)


def process_voxeldepth_from_surfaces(
    surf_white_lh_file,
    area_white_lh_file,
    surf_pial_lh_file,
    area_pial_lh_file,
    surf_white_rh_file,
    area_white_rh_file,
    surf_pial_rh_file,
    area_pial_rh_file,
    volume_file,
    depths_fname,
    columns_fname,
    label_lh_fname=None,
    label_rh_fname=None,
    method="equivol",
    upsample_factor=None,
    n_jobs=32,
    force=False,
    area_model="quadratic"):
    """
    Calculate voxel depths from surfaces and vertex area files and save them to nifti files.

    Same as voxeldepths_from_surfaces, but with an area file following each surface file
    (original interface of vdfs <= 1.0, kept for compatibility and for using precomputed areas).
    Area files can be freesurfer (e.g. lh.area, lh.area.pial) or gifti metric files. If an area file is
    None, the vertex areas are calculated from the corresponding surface (identical to the freesurfer
    area files). New code should use voxeldepths_from_surfaces, which does not need area files.
    area_model: "quadratic" (default) or "linear", see voxeldepths_from_surfaces. The quadratic model also
    uses the vertex areas of the midthickness surface, which are calculated from the white and pial surfaces.
    """
    if area_model not in ("quadratic", "linear"):
        raise ValueError(f"area_model must be 'quadratic' or 'linear', not {area_model!r}")
    if (not os.path.isfile(depths_fname)
        or not os.path.isfile(columns_fname)
        or force == True):
        # load volume for wich to calculate voxel depths
        # and get voxel to scanner transformation
        volume = nib.load(volume_file)
        n_x, n_y, n_z = volume.shape[:3]
        voxel_to_scanner = volume.affine

        # optionally use upsampled grid
        if upsample_factor is not None:
            n_x = int(np.floor(n_x * upsample_factor))
            n_y = int(np.floor(n_y * upsample_factor))
            n_z = int(np.floor(n_z * upsample_factor))
            upsampled_to_voxel = np.matrix(
                [[1 / upsample_factor,                   0,                   0, -0.5 + 1 / (2 * upsample_factor)],
                 [                  0, 1 / upsample_factor,                   0, -0.5 + 1 / (2 * upsample_factor)],
                 [                  0,                   0, 1 / upsample_factor, -0.5 + 1 / (2 * upsample_factor)],
                 [                  0,                   0,                   0,                                1]])
            grid_to_scanner = voxel_to_scanner @ upsampled_to_voxel
        else:
            grid_to_scanner = voxel_to_scanner

        # load all surfaces in voxel space
        if label_lh_fname is not None or label_rh_fname is not None:
            surf_white_lh = load_fs_surf_in_grid(surf_white_lh_file, grid_to_scanner, label_lh_fname)
            surf_pial_lh = load_fs_surf_in_grid(surf_pial_lh_file, grid_to_scanner, label_lh_fname)
            surf_white_rh = load_fs_surf_in_grid(surf_white_rh_file, grid_to_scanner, label_rh_fname)
            surf_pial_rh = load_fs_surf_in_grid(surf_pial_rh_file, grid_to_scanner, label_rh_fname)
        else:
            surf_white_lh = load_fs_surf_in_grid(surf_white_lh_file, grid_to_scanner)
            surf_pial_lh = load_fs_surf_in_grid(surf_pial_lh_file, grid_to_scanner)
            surf_white_rh = load_fs_surf_in_grid(surf_white_rh_file, grid_to_scanner)
            surf_pial_rh = load_fs_surf_in_grid(surf_pial_rh_file, grid_to_scanner)


        # load area files (or calculate areas from the surfaces in mm and with all triangles)
        area_white_lh = load_vertex_area(area_white_lh_file, surf_white_lh_file)
        area_pial_lh = load_vertex_area(area_pial_lh_file, surf_pial_lh_file)
        area_white_rh = load_vertex_area(area_white_rh_file, surf_white_rh_file)
        area_pial_rh = load_vertex_area(area_pial_rh_file, surf_pial_rh_file)
        # midthickness areas for the quadratic area model
        quadratic = method == "equivol" and area_model == "quadratic"
        area_mid_lh = calc_mid_vertex_area(surf_white_lh_file, surf_pial_lh_file) if quadratic else None
        area_mid_rh = calc_mid_vertex_area(surf_white_rh_file, surf_pial_rh_file) if quadratic else None

        # merge hemispheres (but if labels are provided, only use the hemisphere with labels)
        if label_lh_fname is not None and label_rh_fname is None:
            surf_white = surf_white_lh
            surf_pial = surf_pial_lh
            area_white = area_white_lh
            area_pial = area_pial_lh
            area_mid = area_mid_lh
        elif label_rh_fname is not None and label_lh_fname is None:
            surf_white = surf_white_rh
            surf_pial = surf_pial_rh
            area_white = area_white_rh
            area_pial = area_pial_rh
            area_mid = area_mid_rh
        else:
            surf_white = merge_surfaces(surf_white_lh, surf_white_rh)
            surf_pial = merge_surfaces(surf_pial_lh, surf_pial_rh)
            area_white = np.concatenate((area_white_lh, area_white_rh), axis=0)
            area_pial = np.concatenate((area_pial_lh, area_pial_rh), axis=0)
            area_mid = np.concatenate((area_mid_lh, area_mid_rh), axis=0) if quadratic else None

        # calc voxel depths
        depths, columns = calc_depth_from_surfaces_on_grid(
            surf_white, area_white, surf_pial, area_pial, n_x, n_y, n_z, method, n_jobs, area_mid)

        # generate output nifti using affine of the used voxel grid (upsampled or not)
        xform = grid_to_scanner
        # voxelwise depths
        nii_depths = nib.nifti1.Nifti1Image(depths, xform)
        # column indices that a voxel belongs to
        nii_columns = nib.nifti1.Nifti1Image(columns, xform)

        nib.save(nii_depths, depths_fname)
        nib.save(nii_columns, columns_fname)

    else:
        # if files already exist, just load them for return
        nii_depths = nib.load(depths_fname)
        nii_columns = nib.load(columns_fname)
    return nii_depths, nii_columns


# name used in fmri_analysis before vdfs was a separate package
process_dc_voxeldepth_from_surfaces = process_voxeldepth_from_surfaces
