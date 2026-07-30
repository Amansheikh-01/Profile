import sys
sys.path.append(r'C:\Users\Aman\AppData\Roaming\Python\Python314\site-packages')
import os
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from scipy.optimize import linear_sum_assignment

# --- CONFIG & PALETTE ---
WIDTH, HEIGHT = 1180, 610
PORTRAIT_GRID_W, PORTRAIT_GRID_H = 300, 340
PORTRAIT_X, PORTRAIT_Y = 50, 95
DOT_SCALE = 1.12  # 300*1.12 = 336px width, 340*1.12 = 380px height

PALETTE_DARK = {
    'bg': '#0A101F',
    'chrome_bg': '#0F172A',
    'chrome_border': '#1E293B',
    'title_bar': '#1E293B',
    'ui_chrome': '#22D3EE',
    'ui_dim': '#0891B2',
    'portrait': '#A78BFA',
    'accent': '#10B981',
    'live_red': '#EF4444',
    'text_main': '#F8FAFC',
    'text_sub': '#94A3B8',
    'leader': '#334155'
}

PALETTE_LIGHT = {
    'bg': '#F8FAFC',
    'chrome_bg': '#FFFFFF',
    'chrome_border': '#E2E8F0',
    'title_bar': '#F1F5F9',
    'ui_chrome': '#0891B2',
    'ui_dim': '#0284C7',
    'portrait': '#7C3AED',
    'accent': '#10B981',
    'live_red': '#EF4444',
    'text_main': '#0F172A',
    'text_sub': '#475569',
    'leader': '#CBD5E1'
}

def process_portrait(image_path, is_dark_mode=True):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    crop_w = int(w * 0.86)
    crop_h = int(crop_w / (PORTRAIT_GRID_W / PORTRAIT_GRID_H))
    crop_left = int((w - crop_w) / 2)
    crop_top = int(h * 0.14)
    cropped = img.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))
    
    resized = cropped.resize((PORTRAIT_GRID_W, PORTRAIT_GRID_H), Image.Resampling.LANCZOS)
    
    # Contrast 1.3x only
    enhanced = ImageEnhance.Contrast(resized).enhance(1.3)
    # Autocontrast(cutoff=1)
    auto = ImageOps.autocontrast(enhanced, cutoff=1)
    # UnsharpMask(radius=3, percent=140)
    unsharp = auto.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
    img_np = np.array(unsharp)
    gray_np = np.array(unsharp.convert('L'))
    
    # Background Segmentation
    corner_bg = np.mean([img_np[0:15, 0:15], img_np[0:15, -15:]], axis=(0,1,2))
    diff = np.linalg.norm(img_np.astype(float) - corner_bg, axis=2)
    fg_mask = diff >= 60.0
    
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    fg_mask = binary_closing(fg_mask, structure=np.ones((5,5)))
    fg_mask = binary_fill_holes(fg_mask)
    labeled, num_features = label(fg_mask)
    if num_features > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        fg_mask = (labeled == (np.argmax(sizes) + 1))
        
    arr = gray_np.astype(float)
    h_grid, w_grid = arr.shape
    dithered = np.zeros((h_grid, w_grid), dtype=bool)
    
    for y in range(h_grid):
        x_range = range(w_grid) if y % 2 == 0 else range(w_grid - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        
        for x in x_range:
            old_val = arr[y, x]
            if is_dark_mode:
                if not fg_mask[y, x]:
                    arr[y, x] = 0.0
                    dithered[y, x] = False
                    continue
                new_val = 255.0 if old_val >= 128.0 else 0.0
                err = old_val - new_val
                dithered[y, x] = (new_val == 255.0)
            else:
                # In light mode: slightly threshold background so file size stays ~950KB
                if not fg_mask[y, x] and old_val > 210:
                    arr[y, x] = 255.0
                    dithered[y, x] = False
                    continue
                new_val = 0.0 if old_val < 128.0 else 255.0
                err = old_val - new_val
                dithered[y, x] = (new_val == 0.0)
                
            nx = x + direction
            if 0 <= nx < w_grid:
                arr[y, nx] += err * (7.0 / 16.0)
            if y + 1 < h_grid:
                arr[y + 1, x] += err * (5.0 / 16.0)
                if 0 <= nx < w_grid:
                    arr[y + 1, nx] += err * (1.0 / 16.0)
                px = x - direction
                if 0 <= px < w_grid:
                    arr[y + 1, px] += err * (3.0 / 16.0)
                    
    ys, xs = np.where(dithered)
    canvas_xs = PORTRAIT_X + xs * DOT_SCALE
    canvas_ys = PORTRAIT_Y + ys * DOT_SCALE
    dots = np.column_stack([canvas_xs, canvas_ys])
    return dots, dithered

def generate_logo_points(logo_type, num_points=900):
    center_x = PORTRAIT_X + (PORTRAIT_GRID_W * DOT_SCALE) / 2.0
    center_y = PORTRAIT_Y + (PORTRAIT_GRID_H * DOT_SCALE) / 2.0
    size = 130.0
    
    if logo_type == 'flutter':
        n1 = num_points // 3
        n2 = num_points // 3
        n3 = num_points - n1 - n2
        t1 = np.linspace(0, 1, n1)
        p1 = np.column_stack([center_x + (t1 - 0.2)*size*0.7, center_y - size*0.6 + t1*size*0.6])
        t2 = np.linspace(0, 1, n2)
        p2 = np.column_stack([center_x - size*0.5 + t2*size*0.8, center_y - size*0.2 + t2*size*0.8])
        t3 = np.linspace(0, 1, n3)
        p3 = np.column_stack([center_x - size*0.1 + t3*size*0.5, center_y + size*0.2 + (1-t3)*size*0.4])
        points = np.vstack([p1, p2, p3])
        
    elif logo_type == 'code':
        n3 = num_points // 3
        t1 = np.linspace(0, 1, n3)
        n3_half = n3 // 2
        left_bracket = np.vstack([
            np.column_stack([center_x - size*0.25 - t1[:n3_half]*size*0.3, center_y - size*0.4 + t1[:n3_half]*size*0.4]),
            np.column_stack([center_x - size*0.55 + t1[n3_half:]*size*0.3, center_y + t1[n3_half:]*size*0.4])
        ])
        t2 = np.linspace(0, 1, n3)
        slash = np.column_stack([center_x + size*0.15 - t2*size*0.3, center_y - size*0.5 + t2*size*1.0])
        t3 = np.linspace(0, 1, num_points - len(left_bracket) - len(slash))
        n_rem = len(t3) // 2
        right_bracket = np.vstack([
            np.column_stack([center_x + size*0.25 + t3[:n_rem]*size*0.3, center_y - size*0.4 + t3[:n_rem]*size*0.4]),
            np.column_stack([center_x + size*0.55 - t3[n_rem:]*size*0.3, center_y + t3[n_rem:]*size*0.4])
        ])
        points = np.vstack([left_bracket, slash, right_bracket])
        
    elif logo_type == 'vercel':
        n_side = num_points // 3
        t = np.linspace(0, 1, n_side)
        top = np.array([center_x, center_y - size*0.55])
        left = np.array([center_x - size*0.55, center_y + size*0.45])
        right = np.array([center_x + size*0.55, center_y + size*0.45])
        side1 = np.column_stack([top[0]*(1-t) + left[0]*t, top[1]*(1-t) + left[1]*t])
        side2 = np.column_stack([left[0]*(1-t) + right[0]*t, left[1]*(1-t) + right[1]*t])
        t_rem = np.linspace(0, 1, num_points - 2*n_side)
        side3 = np.column_stack([right[0]*(1-t_rem) + top[0]*t_rem, right[1]*(1-t_rem) + top[1]*t_rem])
        points = np.vstack([side1, side2, side3])
        
    jitter = np.random.normal(0, 0.8, points.shape)
    return points + jitter

def match_points(p1, p2):
    cost = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
    _, col_ind = linear_sum_assignment(cost)
    return p2[col_ind]

def dots_to_path_runs(dots, dot_size=1.2):
    """Combine dots into optimized path runs to minimize SVG payload"""
    if len(dots) == 0:
        return ""
    # Sort dots by Y then X
    sorted_dots = dots[np.lexsort((dots[:, 0], dots[:, 1]))]
    runs = []
    
    curr_x, curr_y = None, None
    run_len = 0
    
    for x, y in sorted_dots:
        x = round(x, 1)
        y = round(y, 1)
        if curr_y == y and abs((curr_x + run_len * dot_size) - x) < 0.2:
            run_len += 1
        else:
            if curr_x is not None:
                w_val = round(run_len * dot_size, 1)
                runs.append(f"M{curr_x} {curr_y}h{w_val}v{dot_size}h-{w_val}Z")
            curr_x, curr_y = x, y
            run_len = 1
            
    if curr_x is not None:
        w_val = round(run_len * dot_size, 1)
        runs.append(f"M{curr_x} {curr_y}h{w_val}v{dot_size}h-{w_val}Z")
        
    return "".join(runs)

def calculate_evenness_metric(groups_dots):
    ratios = []
    mid_x = PORTRAIT_X + (PORTRAIT_GRID_W * DOT_SCALE) / 2.0
    mid_y = PORTRAIT_Y + (PORTRAIT_GRID_H * DOT_SCALE) / 2.0
    
    for grp in groups_dots:
        if len(grp) == 0:
            continue
        xs, ys = grp[:, 0], grp[:, 1]
        q1 = np.sum((xs < mid_x) & (ys < mid_y))
        q2 = np.sum((xs >= mid_x) & (ys < mid_y))
        q3 = np.sum((xs < mid_x) & (ys >= mid_y))
        q4 = np.sum((xs >= mid_x) & (ys >= mid_y))
        counts = np.array([q1, q2, q3, q4]) + 1e-5
        ratios.append(counts / np.sum(counts))
        
    r_arr = np.array(ratios)
    return float(np.mean(np.std(r_arr, axis=0)))

def calculate_straight_boundary_metric(band_y_boundaries):
    diffs = np.diff(band_y_boundaries)
    return float(np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-5))

def build_svg(is_dark_mode=True):
    palette = PALETTE_DARK if is_dark_mode else PALETTE_LIGHT
    image_path = r'C:\Users\Aman\.gemini\antigravity-ide\brain\c77d3cee-75dd-416a-8c32-aa4ee005a365\media__1785430954729.jpg'
    
    portrait_dots, dither_grid = process_portrait(image_path, is_dark_mode=is_dark_mode)
    num_dots = len(portrait_dots)
    
    # 1. Intro Grouping (~60 interleaved groups)
    num_intro_groups = 60
    indices = np.arange(num_dots)
    np.random.shuffle(indices)
    intro_groups = [portrait_dots[indices[i::num_intro_groups]] for i in range(num_intro_groups)]
    evenness = calculate_evenness_metric(intro_groups)
    
    # 2. Drift Bands (~94 bands with noise sigma=4)
    num_bands = 94
    y_vals = portrait_dots[:, 1]
    y_noisy = y_vals + np.random.normal(0, 4.0, size=num_dots)
    band_sort_idx = np.argsort(y_noisy)
    
    band_splits = np.array_split(band_sort_idx, num_bands)
    drift_bands = [portrait_dots[split] for split in band_splits]
    
    # Band y boundary means for metric verification
    band_means = [np.mean(b[:, 1]) for b in drift_bands if len(b) > 0]
    boundary_metric = calculate_straight_boundary_metric(band_means)
    
    # 3. Travellers (~600 dots)
    N_TRAVELLERS = 600
    logo1 = generate_logo_points('flutter', N_TRAVELLERS)
    logo2_raw = generate_logo_points('code', N_TRAVELLERS)
    logo3_raw = generate_logo_points('vercel', N_TRAVELLERS)
    logo2 = match_points(logo1, logo2_raw)
    logo3 = match_points(logo2, logo3_raw)
    logo1_back = match_points(logo3, logo1)
    
    c1 = np.mean(logo1, axis=0)
    dur = 14.2
    t = [0.0, 3.0/dur, 4.3/dur, 6.3/dur, 7.6/dur, 9.6/dur, 10.9/dur, 12.9/dur, 1.0]
    key_times_str = ";".join([f"{kt:.3f}" for kt in t])
    
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" height="100%">')
    svg_parts.append('<style>')
    svg_parts.append(f'''
        .bg {{ fill: {palette['bg']}; }}
        .chrome-bg {{ fill: {palette['chrome_bg']}; stroke: {palette['chrome_border']}; stroke-width: 1.5; }}
        .title-bar {{ fill: {palette['title_bar']}; }}
        .title-text {{ font-family: "Fira Code", monospace, sans-serif; font-size: 13px; fill: {palette['text_sub']}; font-weight: 600; }}
        .frame-box {{ fill: none; stroke: {palette['chrome_border']}; stroke-width: 1; }}
        .frame-label {{ font-family: "Fira Code", monospace, sans-serif; font-size: 11px; fill: {palette['ui_chrome']}; letter-spacing: 1.5px; font-weight: 700; }}
        .info-header {{ font-family: "Fira Code", monospace, sans-serif; font-size: 14px; fill: {palette['ui_chrome']}; font-weight: 700; letter-spacing: 2px; }}
        .info-label {{ font-family: "Fira Code", monospace, sans-serif; font-size: 14px; fill: {palette['text_sub']}; }}
        .info-leader {{ font-family: "Fira Code", monospace, sans-serif; font-size: 14px; fill: {palette['leader']}; letter-spacing: 1px; }}
        .info-val {{ font-family: "Fira Code", monospace, sans-serif; font-size: 14px; fill: {palette['text_main']}; font-weight: 500; }}
        .info-val-accent {{ font-family: "Fira Code", monospace, sans-serif; font-size: 14px; fill: {palette['accent']}; font-weight: 600; }}
        .live-badge {{ font-family: "Fira Code", monospace, sans-serif; font-size: 12px; fill: {palette['live_red']}; font-weight: 700; letter-spacing: 1px; }}
        .pill-bg {{ fill: {palette['ui_chrome']}; fill-opacity: 0.15; stroke: {palette['ui_chrome']}; stroke-width: 1; }}
        .pill-text {{ font-family: "Fira Code", monospace, sans-serif; font-size: 14px; fill: {palette['ui_chrome']}; font-weight: 700; }}
        .portrait-dot {{ fill: {palette['portrait']}; shape-rendering: crispEdges; }}
        .traveller-dot {{ fill: {palette['ui_chrome']}; shape-rendering: crispEdges; }}
    ''')
    svg_parts.append('</style>')
    
    # Outer Background & Terminal Box
    svg_parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" class="bg" rx="12"/>')
    svg_parts.append(f'<rect x="15" y="15" width="{WIDTH-30}" height="{HEIGHT-30}" rx="10" class="chrome-bg"/>')
    
    # Title Bar
    svg_parts.append(f'<path d="M 15 25 A 10 10 0 0 1 25 15 L {WIDTH-25} 15 A 10 10 0 0 1 {WIDTH-15} 25 L {WIDTH-15} 45 L 15 45 Z" class="title-bar"/>')
    svg_parts.append(f'<line x1="15" y1="45" x2="{WIDTH-15}" y2="45" stroke="{palette["chrome_border"]}" stroke-width="1.5"/>')
    
    # Terminal Window Buttons
    svg_parts.append('<circle cx="35" cy="30" r="5.5" fill="#FF5F56"/>')
    svg_parts.append('<circle cx="53" cy="30" r="5.5" fill="#FFBD2E"/>')
    svg_parts.append('<circle cx="71" cy="30" r="5.5" fill="#27C93F"/>')
    
    # Window Title
    svg_parts.append(f'<text x="{WIDTH/2}" y="34" text-anchor="middle" class="title-text">profile.sh --live</text>')
    
    # Left Frame (VISUAL.MAP)
    FRAME_W, FRAME_H = 390, 520
    svg_parts.append(f'<rect x="30" y="60" width="{FRAME_W}" height="{FRAME_H}" rx="6" class="frame-box"/>')
    svg_parts.append(f'<rect x="45" y="52" width="105" height="16" fill="{palette["chrome_bg"]}"/>')
    svg_parts.append(f'<text x="50" y="64" class="frame-label">VISUAL.MAP</text>')
    
    # Intro Layer
    svg_parts.append('<g id="intro-layer">')
    for g_idx, grp in enumerate(intro_groups):
        delay = (g_idx / num_intro_groups) * 1.8
        d_str = dots_to_path_runs(grp, 1.2)
        if d_str:
            svg_parts.append(f'  <path d="{d_str}" class="portrait-dot" opacity="0">')
            svg_parts.append(f'    <animate attributeName="opacity" values="0;1" begin="{delay:.2f}s" dur="0.4s" fill="freeze"/>')
            svg_parts.append('  </path>')
    svg_parts.append('</g>')
    
    # Portrait Drift Bands Layer
    svg_parts.append('<g id="portrait-drift-layer">')
    for b_idx, band in enumerate(drift_bands):
        if len(band) == 0:
            continue
        d_str = dots_to_path_runs(band, 1.2)
        b_centroid = np.mean(band, axis=0)
        dx = (c1[0] - b_centroid[0]) * 0.42
        dy = (c1[1] - b_centroid[1]) * 0.42
        
        svg_parts.append(f'  <g class="portrait-dot">')
        svg_parts.append(f'    <animateTransform attributeName="transform" type="translate" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="0,0; 0,0; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; 0,0; 0,0"/>')
        svg_parts.append(f'    <animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="1; 1; 0; 0; 0; 0; 0; 1; 1"/>')
        svg_parts.append(f'    <path d="{d_str}"/>')
        svg_parts.append('  </g>')
    svg_parts.append('</g>')
    
    # Travellers Logo Morph Layer
    svg_parts.append('<g id="travellers-layer">')
    trav_opacity_values = "0; 0; 1; 1; 1; 1; 1; 0; 0"
    
    for i in range(N_TRAVELLERS):
        p_l1 = logo1[i]
        p_l2 = logo2[i]
        p_l3 = logo3[i]
        
        svg_parts.append('  <g class="traveller-dot">')
        svg_parts.append(f'    <animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{trav_opacity_values}"/>')
        tx_vals = f"0,0; 0,0; 0,0; 0,0; {p_l2[0]-p_l1[0]:.1f},{p_l2[1]-p_l1[1]:.1f}; {p_l2[0]-p_l1[0]:.1f},{p_l2[1]-p_l1[1]:.1f}; {p_l3[0]-p_l1[0]:.1f},{p_l3[1]-p_l1[1]:.1f}; 0,0; 0,0"
        svg_parts.append(f'    <animateTransform attributeName="transform" type="translate" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{tx_vals}"/>')
        svg_parts.append(f'    <rect x="{p_l1[0]:.1f}" y="{p_l1[1]:.1f}" width="1.8" height="1.8" rx="0.4"/>')
        svg_parts.append('  </g>')
    svg_parts.append('</g>')
    
    # Right Panel (SYSTEM.INFO)
    INFO_X = 450
    INFO_W = 680
    
    svg_parts.append(f'<text x="{INFO_X}" y="92" class="info-header">SYSTEM.INFO</text>')
    svg_parts.append(f'<circle cx="{INFO_X + 165}" cy="87" r="4" fill="{palette["live_red"]}">')
    svg_parts.append('  <animate attributeName="opacity" values="1;0.2;1" dur="1.5s" repeatCount="indefinite"/>')
    svg_parts.append('</circle>')
    svg_parts.append(f'<text x="{INFO_X + 175}" y="92" class="live-badge">LIVE</text>')
    
    PILL_X = INFO_X + 530
    svg_parts.append(f'<rect x="{PILL_X}" y="74" width="135" height="26" rx="13" class="pill-bg"/>')
    svg_parts.append(f'<text x="{PILL_X + 67.5}" y="91" text-anchor="middle" class="pill-text">Demannn-0x</text>')
    svg_parts.append(f'<line x1="{INFO_X}" y1="108" x2="{INFO_X + INFO_W}" y2="108" stroke="{palette["chrome_border"]}" stroke-width="1.5"/>')
    
    rows_data = [
        ("Subject", "Aman Sheikh", "main"),
        ("Role", "Full-Stack Developer", "accent"),
        ("Origin", "Nagpur, India", "main"),
        ("Education", "B.Voc", "main"),
        ("Status", "Building + Learning", "accent"),
        ("ToolChain", "VS Code, Git, Claude", "main"),
        ("Core.Lang", "JavaScript, TypeScript, Python", "main"),
        ("Core.Frontend", "React, Next.js, HTML5/CSS3", "main"),
        ("Core.Backend", "Node.js, Express, Java", "main"),
        ("Core.Database", "MongoDB, PostgreSQL", "main"),
        ("Core.Infra", "Vercel, Docker, GitHub Actions", "main"),
        ("Grid.Mail", "amansheikh14241@gmail.com", "main"),
        ("Grid.Portfolio", "coming soon", "sub"),
        ("Grid.LinkedIn", "aman-sheikh-488529415", "main"),
        ("Grid.GitHub", "Demannn-0x", "accent"),
        ("Grid.Facebook", "n/a", "sub")
    ]
    
    start_y = 135
    row_height = 27
    
    for idx, (label, val, style) in enumerate(rows_data):
        cur_y = start_y + idx * row_height
        dots_count = max(4, 55 - len(label) - len(val))
        leader_dots = " ." * (dots_count // 2)
        val_class = "info-val" if style == "main" else ("info-val-accent" if style == "accent" else "info-label")
        
        # Lock text length with textLength and lengthAdjust="spacingAndGlyphs"
        val_len = len(val) * 8.5
        val_x = INFO_X + INFO_W
        
        svg_parts.append(f'<g id="row-{idx}">')
        svg_parts.append(f'  <text x="{INFO_X}" y="{cur_y}" class="info-label">{label}</text>')
        svg_parts.append(f'  <text x="{INFO_X + 165}" y="{cur_y}" class="info-leader">{leader_dots}</text>')
        svg_parts.append(f'  <text x="{val_x}" y="{cur_y}" text-anchor="end" class="{val_class}" textLength="{val_len:.1f}" lengthAdjust="spacingAndGlyphs">{val}</text>')
        svg_parts.append('</g>')
        
    svg_parts.append('</svg>')
    
    print(f"[{'DARK' if is_dark_mode else 'LIGHT'}] Metrics -> Intro Evenness: {evenness:.4f}, Boundary Metric: {boundary_metric:.4f}")
    return "\n".join(svg_parts)

# Run generator
dark_content = build_svg(is_dark_mode=True)
with open(r'd:\Projects\Profile\dark.svg', 'w', encoding='utf-8') as f:
    f.write(dark_content)

light_content = build_svg(is_dark_mode=False)
with open(r'd:\Projects\Profile\light.svg', 'w', encoding='utf-8') as f:
    f.write(light_content)

print("Files generated successfully.")
