"""
Tests on synthetic surfaces: a flat white sheet and a flat pial sheet above it, for which depths are known.
Run with: python -m pytest tests
"""
from collections import OrderedDict

import nibabel as nib
import numpy as np
import pytest

import vdfs

GRID_SHAPE = (12, 12, 8)
Z_WHITE = 1.5
THICKNESS = 4.0
CRAS = np.array([1.0, -2.0, 3.0])


def sheet(z, scale=1.0, n=8, step=2.0, origin=(-1.31, -1.77), center=5.37):
    """Flat triangulated sheet at height z, optionally scaled in x and y around center.
    (origin and center are chosen such that no voxel center lies exactly on the boundary between two
    prisms, where tiny numerical differences, e.g. float32 gifti coordinates, could change the prism)"""
    x, y = np.meshgrid(origin[0] + step * np.arange(n), origin[1] + step * np.arange(n), indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel(), np.full(n * n, z)))
    coords[:, :2] = center + scale * (coords[:, :2] - center)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    return coords, faces.astype(np.int32)


def volume_info():
    return OrderedDict(
        head=np.array([2, 0, 20], dtype=np.int32), valid="1  # volume info valid", filename="vol.nii",
        volume=np.array([256, 256, 256]), voxelsize=np.array([1.0, 1.0, 1.0]),
        xras=np.array([-1.0, 0.0, 0.0]), yras=np.array([0.0, 0.0, -1.0]), zras=np.array([0.0, 1.0, 0.0]),
        cras=CRAS)


def write_fs(fname, coords, faces):
    """Write freesurfer surface, coordinates in scanner space are shifted by -c_ras."""
    nib.freesurfer.write_geometry(str(fname), coords - CRAS, faces, volume_info=volume_info())


def write_gii(fname, coords, faces):
    gii = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
        nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")])
    nib.save(gii, str(fname))


@pytest.fixture
def data(tmp_path):
    """Writes white/pial sheets (pial larger by factor 2) for both hemispheres, as freesurfer and gifti
    files, and a volume with identity affine. The right hemisphere sheet lies outside the grid."""
    files = {}
    for hemi, dz in (("lh", 0.0), ("rh", 100.0)):
        for surf, z, scale in (("white", Z_WHITE, 1.0), ("pial", Z_WHITE + THICKNESS, 2.0)):
            coords, faces = sheet(z + dz, scale)
            files[hemi, surf, "fs"] = tmp_path / f"{hemi}.{surf}"
            files[hemi, surf, "gii"] = tmp_path / f"{hemi}.{surf}.surf.gii"
            write_fs(files[hemi, surf, "fs"], coords, faces)
            write_gii(files[hemi, surf, "gii"], coords, faces)
            area = vdfs.calc_vertex_area(coords, faces).astype(np.float32)
            files[hemi, surf, "area"] = tmp_path / f"{hemi}.{surf}.area"
            nib.freesurfer.write_morph_data(str(files[hemi, surf, "area"]), area)
    files["volume"] = tmp_path / "volume.nii"
    nib.save(nib.Nifti1Image(np.zeros(GRID_SHAPE, dtype=np.float32), np.eye(4)), str(files["volume"]))
    files["tmp"] = tmp_path
    return files


def run(files, kind, method="equivol", name="out", **kwargs):
    depths, columns = vdfs.voxeldepths_from_surfaces(
        str(files["lh", "white", kind]), str(files["lh", "pial", kind]),
        str(files["rh", "white", kind]), str(files["rh", "pial", kind]),
        str(files["volume"]), str(files["tmp"] / f"{name}_depths.nii"),
        str(files["tmp"] / f"{name}_columns.nii"), method=method, n_jobs=2, **kwargs)
    return depths.get_fdata(), columns.get_fdata()


def interior():
    """Voxel centers (with their expected relative height) away from the sheet borders."""
    z = np.arange(GRID_SHAPE[2])
    inside = (z > Z_WHITE) & (z < Z_WHITE + THICKNESS)
    return (slice(3, 9), slice(3, 9), z[inside]), (z[inside] - Z_WHITE) / THICKNESS


def test_vertex_area_single_triangle():
    coords = np.array([[0.0, 0, 0], [2, 0, 0], [0, 3, 0]])
    np.testing.assert_allclose(vdfs.calc_vertex_area(coords, np.array([[0, 1, 2]])), [1, 1, 1])


def test_vertex_area_sums_to_surface_area():
    coords, faces = sheet(0.0, step=2.0, n=5)
    assert vdfs.calc_vertex_area(coords, faces).sum() == pytest.approx(8.0 ** 2)


def test_equidist_depths(data):
    depths, columns = run(data, "fs", method="equidist")
    idx, alpha = interior()
    np.testing.assert_allclose(depths[idx], np.broadcast_to(alpha, depths[idx].shape), atol=1e-6)
    assert not np.isnan(columns[idx]).any()
    # outside of the ribbon
    assert np.isnan(depths[:, :, 0]).all() and np.isnan(depths[:, :, 7]).all()


def test_equivol_depths(data):
    # the pial sheet is the white sheet scaled by 2 in x and y and moved up, so all intermediate sheets are
    # flat (at relative height alpha) and their area grows as (1 + alpha)^2: the exact equi-volume depth is
    # ((1 + alpha)^3 - 1) / 7, which the (default) quadratic area model reproduces
    depths, _ = run(data, "fs")
    idx, alpha = interior()
    d = depths[idx]
    assert not np.isnan(d).any()
    np.testing.assert_allclose(d, np.broadcast_to(((1 + alpha) ** 3 - 1) / 7, d.shape), atol=1e-6)


def test_equivol_depths_linear_area_model(data):
    # the linear-area model maps the equi-volume depth to the relative height via alpha_from_beta
    # (pial area 4x white area)
    depths, _ = run(data, "fs", area_model="linear")
    idx, alpha = interior()
    d = depths[idx]
    assert not np.isnan(d).any()
    alpha_est = np.vectorize(lambda b: vdfs.alpha_from_beta(b, 1.0, 4.0))(d)
    np.testing.assert_allclose(alpha_est, np.broadcast_to(alpha, d.shape), atol=1e-6)
    assert (d < np.broadcast_to(alpha, d.shape)).all()


def test_gifti_equals_freesurfer(data):
    d_fs, c_fs = run(data, "fs", name="fs")
    d_gii, c_gii = run(data, "gii", name="gii")
    np.testing.assert_allclose(d_gii, d_fs, atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(c_gii, c_fs)


def test_area_files_equal_calculated_areas(data):
    d_calc, c_calc = run(data, "fs", name="calc")
    d_file, c_file = vdfs.process_voxeldepth_from_surfaces(
        str(data["lh", "white", "fs"]), str(data["lh", "white", "area"]),
        str(data["lh", "pial", "fs"]), str(data["lh", "pial", "area"]),
        str(data["rh", "white", "fs"]), str(data["rh", "white", "area"]),
        str(data["rh", "pial", "fs"]), str(data["rh", "pial", "area"]),
        str(data["volume"]), str(data["tmp"] / "file_depths.nii"), str(data["tmp"] / "file_columns.nii"),
        n_jobs=2)
    np.testing.assert_allclose(d_file.get_fdata(), d_calc, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(c_file.get_fdata(), c_calc)
    assert vdfs.process_dc_voxeldepth_from_surfaces is vdfs.process_voxeldepth_from_surfaces


def test_label_keeps_original_triangle_indices(data):
    d_all, c_all = run(data, "fs", name="all")
    # label with the vertices of the central part of the left hemisphere sheet
    coords, _ = sheet(0.0)
    label = np.flatnonzero((coords[:, 0] > 1) & (coords[:, 0] < 10) & (coords[:, 1] > 1) & (coords[:, 1] < 10))
    for kind, label_file in (("fs", data["tmp"] / "lh.test.label"), ("gii", data["tmp"] / "lh.test.label.gii")):
        if kind == "fs":
            lines = [f"{i} {x:.3f} {y:.3f} {z:.3f} 0.0" for i, (x, y, z) in zip(label, coords[label])]
            label_file.write_text("#!ascii label\n" + f"{len(label)}\n" + "\n".join(lines) + "\n")
        else:
            mask = np.zeros(len(coords), dtype=np.int32)
            mask[label] = 1
            nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(mask)]), str(label_file))
        d_lab, c_lab = run(data, "fs", name=f"label_{kind}", label_lh_fname=str(label_file))
        in_label = ~np.isnan(d_lab)
        assert 0 < in_label.sum() < (~np.isnan(d_all)).sum()
        np.testing.assert_allclose(d_lab[in_label], d_all[in_label], atol=1e-6)
        np.testing.assert_array_equal(c_lab[in_label], c_all[in_label])


def test_first_triangle_is_used():
    # surface with a single triangle (index 0) that contains the voxel center (1, 1, 1)
    white = {"nVertices": 3, "nTris": 1, "tris": np.array([[0, 1, 2]]),
             "vertices": np.array([[0.0, 0, 0.5], [4, 0, 0.5], [0, 4, 0.5]])}
    pial = dict(white, vertices=white["vertices"] + [0, 0, 1])
    depths, columns = vdfs.calc_depth_from_surfaces_on_grid(
        white, np.ones(3), pial, np.ones(3), 3, 3, 3, "equidist", n_jobs=1)
    assert depths[1, 1, 1] == pytest.approx(0.5)
    assert columns[1, 1, 1] == 0


def test_full_prism_list_does_not_fail():
    # the only candidate prism does not contain the point and the list has no free entry left
    faces = np.array([[0, 1, 2]])
    white = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    bounding_prisms = np.zeros((1, 1, 1, 1), dtype=np.int64)
    d, c = vdfs.calc_depth_from_surfaces_voxelidx(
        (0, 0, 0), faces, white, white + [0, 0, 1], np.ones(3), np.ones(3), bounding_prisms,
        p=np.array([5.0, 5.0, 0.5]), method="equidist")
    assert np.isnan(d) and np.isnan(c)


def test_zero_area():
    assert vdfs.alpha_from_beta(0.3, 0.0, 0.0) == pytest.approx(0.3)


def test_mismatching_surfaces_are_rejected():
    white = {"nVertices": 3, "nTris": 1, "tris": np.array([[0, 1, 2]]), "vertices": np.zeros((3, 3))}
    pial = dict(white, tris=np.array([[0, 2, 1]]))
    with pytest.raises(ValueError):
        vdfs.calc_depth_from_surfaces_on_grid(white, np.ones(3), pial, np.ones(3), 2, 2, 2, "equivol")



def test_point_on_boundary_between_prisms():
    # two neighbouring prisms of a sphere mesh (grid coordinates), sharing a side face that contains the point;
    # rounding places the point marginally outside both intermediate triangles unless the boundary counts (#10)
    p = np.array([28.0, 3.0, 17.0])
    white = np.array([[28.0, 11.176604412187725, 3.2208258672652796], [28.0, 13.098956298631903, 4.318878802133362],
                      [25.89575818726388, 11.79846333781229, 3.5151299502168527],
                      [30.10424181273612, 11.79846333781229, 3.5151299502168527]])
    pial = np.array([[28.0, -1.440942278671482, 23.636445267714237], [28.0, 1.923173522605829, 25.558037903733386],
                     [24.317576827711793, -0.3526891588284933, 24.151477412879494],
                     [31.682423172288207, -0.3526891588284933, 24.151477412879494]])
    faces = np.array([[0, 1, 2], [0, 3, 1]])
    bounding_prisms = np.array([[[[0, 1]]]])
    d, c = vdfs.calc_depth_from_surfaces_voxelidx((0, 0, 0), faces, white, pial, np.ones(4), np.ones(4),
                                                  bounding_prisms, p=p, method="equidist")
    assert d == pytest.approx(0.6679147, abs=1e-6)


def test_alpha_from_beta_preserves_volume():
    # with the area changing linearly with distance, the volume fraction below alpha must equal beta
    for aw, ap in ((1.0, 4.0), (4.0, 1.0), (1.0, 1.0), (2.0, 2.0 + 1e-9), (0.0, 1.0), (1.0, 0.0)):
        for beta in np.linspace(0, 1, 11):
            alpha = vdfs.alpha_from_beta(beta, aw, ap)
            assert (2 * aw * alpha + (ap - aw) * alpha ** 2) / (aw + ap) == pytest.approx(beta, abs=1e-12)


def test_alpha_from_beta_quadratic_preserves_volume():
    # with A(alpha) = aw (1-alpha)^2 + 2 am alpha (1-alpha) + ap alpha^2 the volume fraction below alpha,
    # V(alpha) / V(1), must equal beta; for a sphere shell am = sqrt(aw ap)
    for aw, ap, am in ((1.0, 4.0, 2.0), (4.0, 1.0, 2.0), (1.0, 3.0, 1.5), (0.5, 2.0, 1.4), (2.0, 0.3, 0.6)):
        V = lambda a: aw * (a - a ** 2 + a ** 3 / 3) + am * (a ** 2 - 2 * a ** 3 / 3) + ap * a ** 3 / 3
        for beta in np.linspace(0, 1, 11):
            alpha = vdfs.alpha_from_beta_quadratic(beta, aw, ap, am)
            assert V(alpha) / V(1) == pytest.approx(beta, abs=1e-12)


def test_alpha_from_beta_quadratic_linear_special_case():
    # a midthickness area equal to the mean of white and pial area gives the linear-area model
    for aw, ap in ((1.0, 4.0), (4.0, 1.0), (0.7, 0.7)):
        for beta in np.linspace(0, 1, 11):
            assert vdfs.alpha_from_beta_quadratic(beta, aw, ap, (aw + ap) / 2) == vdfs.alpha_from_beta(beta, aw, ap)


def test_overlapping_prisms_first_candidate_wins():
    # two prisms with the same white triangle but different pial triangles both contain the point;
    # as in vdfs <= 1.1, the first candidate prism that contains the point is used
    white = np.array([[0.0, 0, 0], [4, 0, 0], [0, 4, 0], [0.0, 0, 0], [4, 0, 0], [0, 4, 0]])
    pial = white + [0, 0, 2.0]
    pial[3:] = white[3:] + [0, 0, 4.0]
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    p = np.array([1.0, 1.0, 1.0])
    for method in ("equidist", "equivol"):
        for order, (prism, depth) in (([0, 1], (0, 0.5)), ([1, 0], (1, 0.25))):
            d, c = vdfs.calc_depth_from_surfaces_voxelidx((0, 0, 0), faces, white, pial, np.ones(6), np.ones(6),
                                                          np.array([[[order]]]), p=p, method=method)
            assert c == prism and d == pytest.approx(depth)
