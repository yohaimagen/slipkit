"""Reproducible vertical strike-slip Mw 7.6 two-track InSAR synthetic case."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.spatial import Delaunay
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import SlipComponent, StrikeSlipType, TriangularFaultMesh
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.bayesian.assembler import AltarAssembler


def quadtree_sites(displacement, y_coordinates, target):
    """Keep one site per leaf; refine near the fault or at high variation."""
    displacement = np.asarray(displacement)
    if displacement.ndim == 2:
        displacement = displacement[None, ...]
    height, width = displacement.shape[-2:]

    def leaves(threshold):
        selected = []

        def visit(r, c, h, w):
            block = displacement[:, r:r+h, c:c+w]
            y_center = y_coordinates[r+(h-1)//2]
            variation = max(np.std(track) for track in block)
            if h > 1 and w > 1 and (abs(y_center) < 5. or variation > threshold):
                h0, w0 = h//2, w//2
                for dr, dh in ((0, h0), (h0, h-h0)):
                    for dc, dw in ((0, w0), (w0, w-w0)):
                        visit(r+dr, c+dc, dh, dw)
            else:
                selected.append((r+(h-1)//2, c+(w-1)//2))

        for r in range(0, height, 16):
            for c in range(0, width, 16):
                visit(r, c, min(16, height-r), min(16, width-c))
        return np.asarray(selected, dtype=int)

    low, high = 0., float(max(np.std(track) for track in displacement)*10)
    best = leaves(low)
    best_threshold = low
    for _ in range(18):
        threshold = (low+high)/2
        candidate = leaves(threshold)
        if abs(len(candidate)-target) < abs(len(best)-target):
            best, best_threshold = candidate, threshold
        if len(candidate) > target:
            low = threshold
        else:
            high = threshold
    return best, best_threshold


def sar_look_vector(heading_degrees, incidence_degrees, fault_strike_degrees=90.):
    """Return a ground-to-satellite LOS vector in fault-local coordinates."""
    heading = np.deg2rad(heading_degrees)
    incidence = np.deg2rad(incidence_degrees)
    strike = np.deg2rad(fault_strike_degrees)
    # A right-looking satellite views the ground 90 degrees clockwise from its
    # flight direction. The inverse direction is the ground-to-satellite LOS.
    look_azimuth = heading-np.pi/2
    east = np.sin(incidence)*np.sin(look_azimuth)
    north = np.sin(incidence)*np.cos(look_azimuth)
    along = east*np.sin(strike)+north*np.cos(strike)
    across = east*np.cos(strike)-north*np.sin(strike)
    return np.array([along, across, np.cos(incidence)])


def three_asperity_slip(centroids, areas, target_moment, rigidity):
    """Build three separated elliptical asperities with an exact moment."""
    x, depth = centroids[:, 0], -centroids[:, 2]
    centers = np.array([[-92., 7.], [-5., 13.], [82., 8.]])
    target_peaks = np.array([5., 4., 3.])
    along_taper = np.sin(np.pi*np.clip((x+150.)/300., 0., 1.))**.5
    depth_taper = (.35+.65*np.sin(np.pi*np.clip((depth+1.)/27., 0., 1.))**.5)
    background = .1+.4*along_taper*depth_taper
    anchors = np.array([np.argmin(((x-cx)/3.)**2+((depth-cz)/2.)**2)
                        for cx, cz in centers])
    target_mean = target_moment/(rigidity*1e6*areas.sum())

    def candidate(scale):
        widths_x = np.array([18., 16., 16.])*scale
        widths_z = np.full(3, 10.*scale)
        bumps = np.exp(-.5*((x[:, None]-centers[:, 0])/widths_x)**2
                       -.5*((depth[:, None]-centers[:, 1])/widths_z)**2)
        amplitudes = np.linalg.solve(bumps[anchors], target_peaks-background[anchors])
        return background+bumps@amplitudes

    low, high = .5, 1.5
    for _ in range(60):
        scale = (low+high)/2
        mean = np.dot(areas, candidate(scale))/areas.sum()
        if mean < target_mean:
            low = scale
        else:
            high = scale
    slip = candidate((low+high)/2)
    return slip, centers, anchors, target_peaks, (low+high)/2


def build_case(root, seed=17, observations=10000):
    started = time.monotonic()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='mw76-', dir=root))
    rng = np.random.default_rng(seed)
    depths = np.array([0., 3., 7., 12., 18., 25.])
    row_spacings = np.linspace(3., 8., len(depths))
    points = np.vstack([np.column_stack((np.linspace(-150., 150., round(300/spacing)+1),
                      np.full(round(300/spacing)+1, depth))) for depth, spacing in zip(depths, row_spacings)])
    faces = Delaunay(points).simplices.copy()
    e1, e2 = points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]]
    signed = e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]
    faces[signed < 0, 1], faces[signed < 0, 2] = faces[signed < 0, 2].copy(), faces[signed < 0, 1].copy()
    vertices = np.column_stack((points[:, 0], np.zeros(len(points)), -points[:, 1]))
    fault = TriangularFaultMesh((vertices, faces), strike_slip_type=StrikeSlipType.RIGHT_LATERAL,
                               slip_components=[SlipComponent.STRIKE_SLIP])
    areas = fault.get_areas()
    assert np.all(areas > 0) and np.isclose(areas.sum(), 300*25, rtol=1e-12)
    center = fault.get_centroids()
    rigidity = 30e9
    target_moment = 10**(1.5*7.6+9.1)
    slip, asperity_centers, asperity_anchors, target_peaks, asperity_scale = \
        three_asperity_slip(center, areas, target_moment, rigidity)
    moment = rigidity*1e6*np.dot(areas, slip)
    upper = float(slip.max()+2.)
    np.testing.assert_allclose(slip[asperity_anchors], target_peaks, atol=1e-12)
    assert np.isclose(moment, target_moment) and slip.min() >= 0 and upper > slip.max()
    if observations < 10:
        raise ValueError('At least 10 observations are required for the training/holdout split.')
    # Start with a fine raster, then select sites with the project's variance rule.
    ny = max(4, 2*round(np.sqrt(observations)/2))
    nx = 4*ny
    x, y = np.meshgrid(np.linspace(-180, 180, nx), np.linspace(-45, 45, ny))
    raster_coords = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    track_names = ('ascending', 'descending')
    headings = np.array([347., 193.])
    incidence = 34.
    look_vectors = np.vstack([sar_look_vector(heading, incidence) for heading in headings])
    np.testing.assert_allclose(np.linalg.norm(look_vectors, axis=1), 1.)
    engine = CutdeCpuEngine(observation_chunk_size=128)
    clean_rasters = []
    for name, look in zip(track_names, look_vectors):
        direction = np.tile(look, (len(raster_coords), 1))
        raster = GeodeticDataSet(raster_coords, np.zeros(len(raster_coords)), direction,
                                 np.ones(len(raster_coords)), f'synthetic-{name}-full-resolution')
        clean_rasters.append(engine.predict(fault, raster, slip).reshape(ny, nx))
    clean_rasters = np.asarray(clean_rasters)
    sites, variance_threshold = quadtree_sites(clean_rasters, y[:, 0], observations)
    rows, columns = sites.T
    site_coords = np.column_stack((x[rows, columns], y[rows, columns], np.zeros(len(rows))))
    site_count = len(site_coords)
    coords = np.tile(site_coords, (len(track_names), 1))
    direction = np.repeat(look_vectors, site_count, axis=0)
    track_index = np.repeat(np.arange(len(track_names)), site_count)
    sigma = np.full(len(coords), .01)
    clean = np.concatenate([raster[rows, columns] for raster in clean_rasters])
    dataset = GeodeticDataSet(coords, np.zeros(len(coords)), direction, sigma, 'synthetic-two-track-insar')
    dataset.data = clean+rng.normal(0., sigma)
    site_holdout = np.zeros(site_count, dtype=bool)
    for near, far in ((0., 5.), (5., 15.), (15., 45.)):
        group = np.flatnonzero((np.abs(site_coords[:, 1]) >= near) & (np.abs(site_coords[:, 1]) < far))
        site_holdout[rng.permutation(group)[:round(.2*len(group))]] = True
    holdout = np.tile(site_holdout, len(track_names))
    train = ~holdout
    training = GeodeticDataSet(coords[train], dataset.data[train], direction[train], sigma[train],
                               'synthetic-two-track-insar-train')
    problem = AltarAssembler().assemble_problem([fault], [training], engine, None, 0.)
    np.testing.assert_allclose(problem.G@slip, clean[train], rtol=1e-10, atol=1e-9)
    heldout_dataset = GeodeticDataSet(coords[holdout], dataset.data[holdout], direction[holdout],
                                      sigma[holdout], 'synthetic-two-track-insar-holdout')
    heldout_green = engine.build_kernel(fault, heldout_dataset)
    np.testing.assert_allclose(heldout_green@slip, clean[holdout], rtol=1e-10, atol=1e-9)
    case = run/'case';case.mkdir()
    np.save(case/'green.npy', problem.G)
    np.save(case/'data.npy', problem.data)
    np.save(case/'cd.npy', problem.covariance)
    np.save(case/'holdout_green.npy', heldout_green)
    np.save(case/'holdout_data.npy', dataset.data[holdout])
    np.savez(run/'synthetic.npz', vertices=vertices, faces=faces, areas_km2=areas, slip_m=slip,
             coords_km=coords, unit_vectors=direction, sigma_m=sigma, displacement_clean_m=clean,
             displacement_noisy_m=dataset.data, train=train, holdout=holdout,
             site_coords_km=site_coords, site_holdout=site_holdout, track_index=track_index,
             track_names=np.asarray(track_names), track_headings_degrees=headings,
             incidence_degrees=incidence, look_vectors=look_vectors,
             asperity_centers_km=asperity_centers,
             raster_shape=np.array([ny, nx]), quadtree_rows=rows, quadtree_cols=columns)
    characteristic = np.sqrt(2*areas)
    shallow = characteristic[-center[:, 2] < 4.]
    deep = characteristic[-center[:, 2] > 20.]
    report = dict(kind='synthetic vertical strike-slip', target_mw=7.6,
                  achieved_mw=float(2/3*(np.log10(moment)-9.1)),
                  moment_Nm=float(moment), rigidity_Pa=rigidity, length_km=300., depth_km=25.,
                  mesh_depth_rows_km=depths.tolist(), mesh_along_strike_row_spacing_km=row_spacings.tolist(),
                  triangles=fault.num_patches(), parameters=problem.G.shape[1],
                  median_shallow_characteristic_km=float(np.median(shallow)),
                  median_deep_characteristic_km=float(np.median(deep)),
                  area_weighted_mean_slip_m=float(np.dot(areas,slip)/areas.sum()),
                  max_true_slip_m=float(slip.max()), prior_lower_m=0., prior_upper_m=upper,
                  asperity_target_peak_slip_m=target_peaks.tolist(),
                  asperity_anchor_peak_slip_m=slip[asperity_anchors].tolist(),
                  asperity_centers_along_strike_depth_km=asperity_centers.tolist(),
                  asperity_width_scale=float(asperity_scale),
                  surface_locations=int(site_count), observations_per_image=int(site_count),
                  sar_images=len(track_names), observations=int(len(coords)),
                  full_resolution_grid_shape=[int(ny), int(nx)],
                  grid_spacing_km=[float(360/(nx-1)), float(90/(ny-1))],
                  quadtree_standard_deviation_threshold_m=float(variance_threshold),
                  sites_within_5km=int(np.count_nonzero(np.abs(site_coords[:, 1]) < 5)),
                  sites_5_to_15km=int(np.count_nonzero((np.abs(site_coords[:, 1]) >= 5) & (np.abs(site_coords[:, 1]) < 15))),
                  sites_15_to_45km=int(np.count_nonzero(np.abs(site_coords[:, 1]) >= 15)),
                  training_observations=int(train.sum()),
                  heldout_observations=int(holdout.sum()),
                  training_sites=int((~site_holdout).sum()), heldout_sites=int(site_holdout.sum()),
                  observation_type='two-track InSAR line-of-sight displacement',
                  fault_strike_degrees=90., sar_track_names=list(track_names),
                  sar_headings_degrees=headings.tolist(), sar_incidence_degrees=incidence,
                  sar_ground_to_satellite_look_vectors=look_vectors.tolist(),
                  noise_sigma_m=.01, seed=seed, forward_model='Cutde elastic half-space',
                  heldout_noise_rms_m=float(np.sqrt(np.mean((dataset.data[holdout]-clean[holdout])**2))),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    (run/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(path=str(run), **report)), flush=True)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--observations', type=int, default=10000)
    args = parser.parse_args()
    build_case(args.work_dir, args.seed, args.observations)
