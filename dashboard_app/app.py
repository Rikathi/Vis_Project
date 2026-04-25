import os
import re
import base64
import io

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap

import plotly.express as px
from dash import Dash, dcc, html, Input, Output


REAL_CSV = r"D:/KLAUS/Vis_Final_Project/real_feature_output/real_features.csv"
FAKE_CSV = r"D:/KLAUS/Vis_Final_Project/fake_feature_output/fake_features.csv"

REAL_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_test_real"
FAKE_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_output_fake"

MAX_SELECTED_IMAGES = 50


# ============================================================
# Load Data
# ============================================================

real = pd.read_csv(REAL_CSV).copy()
fake = pd.read_csv(FAKE_CSV).copy()

real["type"] = "real"
fake["type"] = "fake"

df = pd.concat([real, fake], ignore_index=True).copy()

feature_cols = [c for c in df.columns if re.fullmatch(r"f\d{3}", c)]
X_raw = df[feature_cols].values
y = df["type"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

df["point_id"] = df.index.astype(str)


# ============================================================
# PCA
# ============================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100


# ============================================================
# t-SNE
# ============================================================

print("Computing 2D t-SNE...")
tsne_2d = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto",
)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

df["TSNE1"] = X_tsne_2d[:, 0]
df["TSNE2"] = X_tsne_2d[:, 1]


print("Computing 6D t-SNE...")
tsne_6d = TSNE(
    n_components=6,
    perplexity=30,
    random_state=42,
    method="exact",
    init="pca",
    learning_rate="auto",
)
X_tsne_6d = tsne_6d.fit_transform(X_scaled)

df_tsne_kde = pd.DataFrame(X_tsne_6d, columns=[f"d{i+1}" for i in range(6)])
df_tsne_kde["type"] = y


# ============================================================
# UMAP
# ============================================================

print("Computing 2D UMAP...")
umap_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)
X_umap_2d = umap_2d.fit_transform(X_scaled)

df["UMAP1"] = X_umap_2d[:, 0]
df["UMAP2"] = X_umap_2d[:, 1]


print("Computing 6D UMAP...")
umap_6d = umap.UMAP(
    n_components=6,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)
X_umap_6d = umap_6d.fit_transform(X_scaled)

df_umap_kde = pd.DataFrame(X_umap_6d, columns=[f"d{i+1}" for i in range(6)])
df_umap_kde["type"] = y


# ============================================================
# Image utilities
# ============================================================

def normalize_to_uint8(img):
    img = np.asarray(img)
    img = np.nan_to_num(img)

    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    return (img * 255).astype(np.uint8)


def numpy_slice_to_base64(img):
    img_uint8 = normalize_to_uint8(img)
    pil_img = Image.fromarray(img_uint8)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded


def resolve_image_path(row):
    if "volume_file" not in row:
        return None

    volume_file = str(row["volume_file"])

    if os.path.exists(volume_file):
        return volume_file

    filename = os.path.basename(volume_file)

    if row["type"] == "real":
        candidate = os.path.join(REAL_IMAGE_DIR, filename)
    else:
        candidate = os.path.join(FAKE_IMAGE_DIR, filename)

    if os.path.exists(candidate):
        return candidate

    return None


def load_selected_image(row):
    image_path = resolve_image_path(row)

    if image_path is None:
        return None, "Image file not found."

    try:
        arr = np.load(image_path)

        slice_idx = int(row["slice_idx"]) if "slice_idx" in row else None

        if arr.ndim == 3:
            if slice_idx is None:
                slice_idx = arr.shape[0] // 2
            slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
            img = arr[slice_idx]

        elif arr.ndim == 2:
            img = arr

        elif arr.ndim == 4:
            if slice_idx is None:
                slice_idx = arr.shape[0] // 2
            slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
            img = arr[slice_idx, :, :, 0]

        else:
            return None, f"Unsupported numpy shape: {arr.shape}"

        img_src = numpy_slice_to_base64(img)

        info = f"""
Type: {row['type']}
File: {os.path.basename(image_path)}
Slice index: {slice_idx}
PC1: {row['PC1']:.4f}
PC2: {row['PC2']:.4f}
TSNE1: {row['TSNE1']:.4f}
TSNE2: {row['TSNE2']:.4f}
UMAP1: {row['UMAP1']:.4f}
UMAP2: {row['UMAP2']:.4f}
"""

        return img_src, info

    except Exception as e:
        return None, f"Error loading image: {str(e)}"


def make_image_tile(row):
    img_src, _ = load_selected_image(row)

    if img_src is None:
        return None

    filename = os.path.basename(str(row["volume_file"])) if "volume_file" in row else "unknown"

    return html.Div(
        className="image-tile",
        children=[
            html.Img(src=img_src, className="tile-img"),
            html.Div(
                className="tile-caption",
                children=[
                    html.Div(f"{row['type']} | slice {int(row['slice_idx']) if 'slice_idx' in row else 'NA'}"),
                    html.Div(filename[:42] + "..." if len(filename) > 42 else filename),
                ],
            ),
        ],
    )


# ============================================================
# Plot functions
# ============================================================

def base_scatter(data, x, y_col, color, title, x_title, y_title):
    fig = px.scatter(
        data,
        x=x,
        y=y_col,
        color=color,
        custom_data=["point_id"],
        opacity=0.7,
        hover_data={
            x: ":.3f",
            y_col: ":.3f",
            "type": True,
            "volume_file": True if "volume_file" in data.columns else False,
            "slice_idx": True if "slice_idx" in data.columns else False,
            "point_id": False,
        },
        title=title,
    )

    fig.update_traces(marker=dict(size=6))

    fig.update_layout(
        height=560,
        clickmode="event+select",
        dragmode="lasso",
        template="plotly_white",
        margin=dict(l=35, r=15, t=55, b=35),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="",
    )

    return fig


def make_pca_plot():
    return base_scatter(
        df,
        "PC1",
        "PC2",
        "type",
        "PCA: Real vs Fake Features",
        f"PC1 ({pc1_var:.2f}%)",
        f"PC2 ({pc2_var:.2f}%)",
    )


def make_tsne_plot():
    return base_scatter(
        df,
        "TSNE1",
        "TSNE2",
        "type",
        "t-SNE: Real vs Fake Features",
        "t-SNE 1",
        "t-SNE 2",
    )


def make_umap_plot():
    return base_scatter(
        df,
        "UMAP1",
        "UMAP2",
        "type",
        "UMAP: Real vs Fake Features",
        "UMAP 1",
        "UMAP 2",
    )


def make_dbscan_plot(eps=0.65, min_samples=5):
    temp_df = df.copy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(temp_df[["PC1", "PC2"]].values)

    temp_df["cluster"] = labels
    temp_df["cluster_label"] = temp_df["cluster"].apply(
        lambda x: "noise" if x == -1 else f"cluster {x}"
    )

    fig = px.scatter(
        temp_df,
        x="PC1",
        y="PC2",
        color="cluster_label",
        symbol="type",
        custom_data=["point_id"],
        opacity=0.75,
        hover_data={
            "PC1": ":.3f",
            "PC2": ":.3f",
            "type": True,
            "cluster_label": True,
            "volume_file": True if "volume_file" in temp_df.columns else False,
            "slice_idx": True if "slice_idx" in temp_df.columns else False,
            "point_id": False,
        },
        title=f"DBSCAN on PCA Space: eps={eps}, min_samples={min_samples}",
    )

    fig.update_traces(marker=dict(size=6))

    fig.update_layout(
        height=560,
        clickmode="event+select",
        dragmode="lasso",
        template="plotly_white",
        margin=dict(l=35, r=15, t=55, b=35),
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Cluster",
    )

    return fig


def make_tsne_kde_plot():
    fig = px.scatter_matrix(
        df_tsne_kde,
        dimensions=[f"d{i+1}" for i in range(6)],
        color="type",
        opacity=0.45,
        title="t-SNE Pairwise Density View",
    )

    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=3),
    )

    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(l=35, r=15, t=55, b=35),
    )

    return fig


def make_umap_kde_plot():
    fig = px.scatter_matrix(
        df_umap_kde,
        dimensions=[f"d{i+1}" for i in range(6)],
        color="type",
        opacity=0.45,
        title="UMAP Pairwise Density View",
    )

    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=3),
    )

    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(l=35, r=15, t=55, b=35),
    )

    return fig


# ============================================================
# App layout
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    className="page-container",
    children=[
        html.Div(
            className="header",
            children=[
                html.H1("Feature Space Explorer", className="main-title"),
                html.P(
                    "Click one point for a single CT slice. Use box/lasso select to view multiple selected images.",
                    className="subtitle",
                ),
            ],
        ),

        html.Div(
            className="dashboard-layout",
            children=[
                html.Div(
                    className="left-panel",
                    children=[
                        html.Div(
                            className="control-card",
                            children=[
                                html.Div(
                                    className="control-box",
                                    children=[
                                        html.Label("Visualization Mode"),
                                        dcc.Dropdown(
                                            id="plot-mode",
                                            options=[
                                                {"label": "PCA Plot", "value": "pca"},
                                                {"label": "Perform DBSCAN on PCA", "value": "dbscan"},
                                                {"label": "t-SNE Plot", "value": "tsne"},
                                                {"label": "KDE / Pair Plot of t-SNE", "value": "tsne_kde"},
                                                {"label": "UMAP Plot", "value": "umap"},
                                                {"label": "KDE / Pair Plot of UMAP", "value": "umap_kde"},
                                            ],
                                            value="pca",
                                            clearable=False,
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="dbscan-controls",
                                    children=[
                                        html.Div(
                                            className="control-box",
                                            children=[
                                                html.Label("DBSCAN eps"),
                                                dcc.Input(
                                                    id="eps-input",
                                                    type="number",
                                                    value=0.65,
                                                    step=0.05,
                                                    min=0.01,
                                                    className="number-input",
                                                ),
                                            ],
                                        ),

                                        html.Div(
                                            className="control-box",
                                            children=[
                                                html.Label("min_samples"),
                                                dcc.Input(
                                                    id="min-samples-input",
                                                    type="number",
                                                    value=5,
                                                    step=1,
                                                    min=1,
                                                    className="number-input",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            className="plot-card",
                            children=[
                                dcc.Graph(
                                    id="main-plot",
                                    figure=make_pca_plot(),
                                    config={
                                        "displayModeBar": True,
                                        "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                html.Div(
                    className="right-panel",
                    children=[
                        html.Div(
                            className="image-card",
                            children=[
                                html.H3("Single Selected CT Slice"),
                                html.Div(
                                    id="image-info",
                                    className="image-info",
                                    children="Click a point to view one image.",
                                ),
                                html.Img(
                                    id="selected-image",
                                    className="selected-image hidden-image",
                                ),
                            ],
                        ),

                        html.Div(
                            className="cluster-card",
                            children=[
                                html.H3("Selected Region / Cluster Images"),
                                html.Div(
                                    id="selected-summary",
                                    className="selected-summary",
                                    children="Use box select or lasso select on the plot to view multiple images.",
                                ),
                                html.Div(
                                    id="selected-image-grid",
                                    className="selected-image-grid",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# Callbacks
# ============================================================

@app.callback(
    Output("main-plot", "figure"),
    Input("plot-mode", "value"),
    Input("eps-input", "value"),
    Input("min-samples-input", "value"),
)
def update_plot(plot_mode, eps, min_samples):
    if plot_mode == "pca":
        return make_pca_plot()

    if plot_mode == "dbscan":
        eps = float(eps) if eps is not None else 0.65
        min_samples = int(min_samples) if min_samples is not None else 5
        return make_dbscan_plot(eps=eps, min_samples=min_samples)

    if plot_mode == "tsne":
        return make_tsne_plot()

    if plot_mode == "tsne_kde":
        return make_tsne_kde_plot()

    if plot_mode == "umap":
        return make_umap_plot()

    if plot_mode == "umap_kde":
        return make_umap_kde_plot()

    return make_pca_plot()


@app.callback(
    Output("selected-image", "src"),
    Output("selected-image", "className"),
    Output("image-info", "children"),
    Input("main-plot", "clickData"),
)
def display_clicked_image(clickData):
    if clickData is None:
        return None, "selected-image hidden-image", "Click a point to view one image."

    try:
        point_id = clickData["points"][0]["customdata"][0]
        row = df.iloc[int(point_id)]
    except Exception:
        return (
            None,
            "selected-image hidden-image",
            "This plot is not single-image clickable. Use PCA, DBSCAN, t-SNE, or UMAP scatter plot.",
        )

    img_src, info = load_selected_image(row)

    if img_src is None:
        return None, "selected-image hidden-image", info

    return img_src, "selected-image", info


@app.callback(
    Output("selected-summary", "children"),
    Output("selected-image-grid", "children"),
    Input("main-plot", "selectedData"),
)
def display_selected_images(selectedData):
    if selectedData is None or "points" not in selectedData:
        return (
            "Use box select or lasso select on the plot to view multiple images.",
            [],
        )

    selected_points = selectedData["points"]

    point_ids = []
    for p in selected_points:
        try:
            point_ids.append(int(p["customdata"][0]))
        except Exception:
            pass

    if len(point_ids) == 0:
        return (
            "No image-linked points selected. Use PCA, DBSCAN, t-SNE, or UMAP scatter plot.",
            [],
        )

    selected_df = df.iloc[point_ids].copy()

    real_count = int((selected_df["type"] == "real").sum())
    fake_count = int((selected_df["type"] == "fake").sum())

    summary = f"""
Selected points: {len(selected_df)}
Real images: {real_count}
Fake images: {fake_count}
Showing first {min(len(selected_df), MAX_SELECTED_IMAGES)} images.
"""

    tiles = []

    for _, row in selected_df.head(MAX_SELECTED_IMAGES).iterrows():
        tile = make_image_tile(row)
        if tile is not None:
            tiles.append(tile)

    return summary, tiles


if __name__ == "__main__":
    app.run(debug=True)