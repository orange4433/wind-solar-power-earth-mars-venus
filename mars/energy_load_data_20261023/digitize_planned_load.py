"""Digitize visible red load ink, preserving the distinction from source telemetry."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[1] / '参考的文献' / 'InSight_Energy_Plot_extracted'
CALIBRATION = {
    'image1.png': dict(x0=195., x1=1920.5, sol0=0, sol1=1500,
                       y0=1351.5, y1=58.5, wh1=5000, width=1996, height=1470),
    'image2.png': dict(x0=163.5, x1=1910.5, sol0=1000, sol1=2100,
                       y0=1354.5, y1=67.5, wh1=900, width=1976, height=1476),
}


def coordinates(x, y, c):
    return (c['sol0'] + (x-c['x0'])*(c['sol1']-c['sol0'])/(c['x1']-c['x0']),
            (c['y0']-y)*c['wh1']/(c['y0']-c['y1']))


def select_load(name, c):
    rgb = np.asarray(Image.open(IMAGES/name).convert('RGB')).astype(float)
    assert rgb.shape[:2] == (c['height'], c['width'])
    yy, xx = np.indices(rgb.shape[:2])
    sols, wh = coordinates(xx, yy, c)
    red = (rgb[:, :, 0] > 90) & (rgb[:, :, 0] > 1.6*rgb[:, :, 1]) & (rgb[:, :, 0] > 1.6*rgb[:, :, 2])
    # Exclude titles, legend and bottom annotations; retain the entire visible load cloud.
    region = ((sols > c['sol0']+1) & (sols < 1490) & (wh > 180) &
              (wh < (2200 if name == 'image1.png' else 750)))
    removed = np.zeros_like(red)
    lines = []
    if name == 'image2.png':
        # Beyond Sol 1500 the only red marks in this band are the two SEIS reference lines.
        reference = red & (sols > 1500) & (sols < 2080) & (wh > 235) & (wh < 265)
        for upper in [True, False]:
            ry, rx = np.where(reference & ((wh > 250) if upper else (wh <= 250)))
            assert len(rx) > 300
            slope, intercept = np.polyfit(rx, ry, 1)
            residual = ry - (slope*rx+intercept)
            assert np.quantile(np.abs(residual), .99) < 2
            removed |= np.abs(yy-(slope*xx+intercept)) <= 2.2
            lines.append(dict(slope=float(slope), intercept=float(intercept), removed_half_width_px=2.2))
    mask = red & region & ~removed
    assert not (mask & (sols > 1470)).any(), 'Unexpected late red ink, possibly a reference line'
    return rgb/255, mask, removed & region, lines


def summarize(mask, removed, name, c):
    yy, xx = np.where(mask)
    sols, wh = coordinates(xx, yy, c)
    rows = []
    for start in range(c['sol0'], 1490, 10):
        select = (sols >= start) & (sols < start+10)
        if not select.any():
            continue
        ys, xs = yy[select], xx[select]
        values = wh[select]
        q10, median, q90 = np.quantile(values, [.1, .5, .9])
        row_removed = bool(removed[ys.min():ys.max()+1, xs.min():xs.max()+1].any())
        rows.append(dict(sol_bin_start=start, sol_bin_end_exclusive=start+10,
                         sol_bin_center=start+5, load_red_ink_median_wh=round(float(median), 1),
                         selected_ink_sol_min=round(float(sols[select].min()), 2),
                         selected_ink_sol_max=round(float(sols[select].max()), 2),
                         load_red_ink_p10_wh=round(float(q10), 1),
                         load_red_ink_p90_wh=round(float(q90), 1),
                         red_pixel_count=int(select.sum()), occupied_x_columns=len(np.unique(xs)),
                         reference_line_crosses_cloud=row_removed, source_image=name,
                         data_type='image_digitized_planned_load_10sol_red_pixel_distribution'))
    # Export only compact separated red components, not guesses for every overlapping marker.
    labels, count = ndi.label(mask)
    points = []
    diameter = 12 if name == 'image1.png' else 18
    min_area = 20 if name == 'image1.png' else 40
    for number, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        h, w = box[0].stop-box[0].start, box[1].stop-box[1].start
        local = labels[box] == number
        area = local.sum()
        if not (5 <= h <= diameter and 5 <= w <= diameter and area >= min_area
                and .65 <= w/h <= 1/.65 and area/(w*h) >= .55):
            continue
        expanded = (slice(max(0, box[0].start-3), box[0].stop+3),
                    slice(max(0, box[1].start-3), box[1].stop+3))
        if removed[expanded].any():
            continue
        y, x = ndi.center_of_mass(local)
        x, y = x+box[1].start, y+box[0].start
        sol, value = coordinates(x, y, c)
        if name == 'image1.png' and sol >= 1000:
            continue
        points.append(dict(sol_approx=round(sol, 2), planned_load_wh_approx=round(value, 1),
                           pixel_x=round(x, 2), pixel_y=round(y, 2), component_area_pixels=int(area),
                           source_image=name, data_type='isolated_red_component_centroid_not_original_telemetry'))
    return rows, points


def main():
    rows, points, selected, metadata = [], [], {}, {}
    for name, c in CALIBRATION.items():
        # Independent interior grid intersections: Sol 750/1500 and 1000/400 Wh.
        grid = ((1057.5, 1093., 750., 1000.) if name == 'image1.png'
                else (957.5, 782., 1500., 400.))
        sol, wh = coordinates(*grid[:2], c)
        assert abs(sol-grid[2]) < .6 and abs(wh-grid[3]) < 2
        rgb, mask, removed, lines = select_load(name, c)
        r, p = summarize(mask, removed, name, c)
        rows.extend(r); points.extend(p)
        selected[name] = (rgb, mask, removed)
        metadata[name] = dict(calibration=c, sha256=hashlib.sha256((IMAGES/name).read_bytes()).hexdigest(),
                              one_pixel_sol=(c['sol1']-c['sol0'])/(c['x1']-c['x0']),
                              one_pixel_wh=c['wh1']/(c['y0']-c['y1']), excluded_reference_lines=lines)
    all_bins = pd.DataFrame(rows)
    points = pd.DataFrame(points).sort_values('sol_approx')
    combined = all_bins[((all_bins.source_image == 'image1.png') & (all_bins.sol_bin_start < 1000)) |
                        (all_bins.source_image == 'image2.png')].sort_values('sol_bin_start')
    assert combined.sol_bin_start.is_unique
    assert combined.load_red_ink_median_wh.between(180, 2200).all()
    all_bins.to_csv(HERE/'planned_load_10sol_by_image.csv', index=False, encoding='utf-8-sig')
    combined.to_csv(HERE/'planned_load_10sol_approx.csv', index=False, encoding='utf-8-sig')
    points.to_csv(HERE/'planned_load_isolated_points_approx.csv', index=False, encoding='utf-8-sig')
    overlap = all_bins[all_bins.sol_bin_start >= 1000].pivot(index='sol_bin_start', columns='source_image', values='load_red_ink_median_wh').dropna()
    overlap['difference_wh_image2_minus_image1'] = overlap['image2.png']-overlap['image1.png']
    overlap.to_csv(HERE/'planned_load_overlap_check.csv', encoding='utf-8-sig')
    metadata['overlap_median_absolute_difference_wh'] = float(overlap.difference_wh_image2_minus_image1.abs().median())
    metadata['overlap_max_absolute_difference_wh'] = float(overlap.difference_wh_image2_minus_image1.abs().max())
    metadata['warning'] = 'Pixel-weighted image summaries, not means of original points; percentiles are not confidence intervals. No interpolation.'
    (HERE/'planned_load_digitization_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, (name, c) in zip(axes, CALIBRATION.items()):
        rgb, mask, removed = selected[name]
        ax.imshow(rgb)
        yy, xx = np.where(mask)
        ax.scatter(xx[::3], yy[::3], s=.7, c='#00bfff', alpha=.8, linewidths=0)
        p = points[points.source_image.eq(name)]
        ax.scatter(p.pixel_x, p.pixel_y, s=30, facecolors='none', edgecolors='black', linewidths=.6)
        ax.set_xlim(c['x0']-3, 1870 if name == 'image1.png' else 900)
        ax.set_ylim(c['y0']+3, 780 if name == 'image1.png' else 340)
        ax.set_title(f'{name}: cyan = selected red ink; black circles = isolated components', fontsize=12)
        ax.set_xlabel('Original image pixel x'); ax.set_ylabel('Original image pixel y')
    fig.tight_layout()
    fig.savefig(HERE/'planned_load_extraction_check.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = combined.sol_bin_center.to_numpy()
    ax.fill_between(x, combined.load_red_ink_p10_wh, combined.load_red_ink_p90_wh,
                    color='#a52a2a', alpha=.18, label='10th–90th percentile of visible red pixels')
    ax.scatter(x, combined.load_red_ink_median_wh, color='#a52a2a', s=15,
               label='10-sol red-pixel median (image approximation)')
    ax.scatter(points.sol_approx, points.planned_load_wh_approx, color='#777777', s=5, alpha=.5,
               label='Compact isolated red components')
    ax.axvline(1000, color='gray', ls='--', lw=1)
    ax.set(xlabel='InSight Sol (approximate image coordinate)', ylabel='Planned load per 3pm–3pm interval (Wh)',
           title='Digitized PPT planned load — not original telemetry', xlim=(0, 1460), ylim=(0, 2000))
    ax.legend(fontsize=9); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(HERE/'planned_load_approx.png', dpi=180)
    plt.close(fig)
    print(f'{len(combined)} ten-sol bins; {len(points)} compact isolated components')
    print('Median cross-image absolute difference:', metadata['overlap_median_absolute_difference_wh'])
    print(combined.tail(10).to_string(index=False))


if __name__ == '__main__':
    main()
