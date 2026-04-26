import os
import re
import base64
import io

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap

import plotly.express as px
# from dash import Dash, dcc, html, Input, Output, ctx
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update


# ============================================================
# Paths
# ============================================================

REAL_CSV = r"D:/KLAUS/Vis_Final_Project/real_feature_output/real_features.csv"
FAKE_CSV = r"D:/KLAUS/Vis_Final_Project/fake_feature_output/fake_features.csv"

REAL_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_test_real"
FAKE_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_output_fake"

MODEL_PATH = r"D:/KLAUS/Vis_Final_Project/best_autoencoder_2d.pth"

MAX_SELECTED_IMAGES = 10
LATENT_DIM = 128
IMG_SIZE = 128
GRADCAM_CACHE = {}


# ============================================================
# Load Data
# ============================================================

real = pd.read_csv(REAL_CSV).copy()
fake = pd.read_csv(FAKE_CSV).copy()

real["type"] = "real"
fake["type"] = "fake"

df = pd.concat([real, fake], ignore_index=True).copy()

feature_cols = [c for c in df.columns if re.fullmatch(r"f\d{3}", c)]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found. Expected columns f000 to f127.")

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

df_tsne_kde = pd.DataFrame(
    X_tsne_6d,
    columns=[f"d{i+1}" for i in range(6)]
)
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

df_umap_kde = pd.DataFrame(
    X_umap_6d,
    columns=[f"d{i+1}" for i in range(6)]
)
df_umap_kde["type"] = y


# ============================================================
# Image Utilities
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
    possible_cols = ["full_path", "volume_file"]

    for col in possible_cols:
        if col in row:
            path_value = str(row[col])

            if os.path.exists(path_value):
                return path_value

            filename = os.path.basename(path_value)

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


# ============================================================
# Grad-CAM Model
# ============================================================

class ConvAutoencoder2D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )

        self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def forward(self, x):
        z = self.encode(x)
        return z


class FeatureGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.clone()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, feature_idx):
        self.model.zero_grad()

        z = self.model(x)
        target = z[0, feature_idx]
        target.backward()

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device,
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, z.detach().cpu().numpy()


# ============================================================
# Load Grad-CAM Model Once
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gradcam_model = ConvAutoencoder2D(latent_dim=LATENT_DIM).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

gradcam_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
gradcam_model.eval()

target_layer = gradcam_model.encoder[6]
gradcam_engine = FeatureGradCAM(gradcam_model, target_layer)


# ============================================================
# Grad-CAM Utilities
# ============================================================

def load_slice_for_gradcam(row, img_size=128):
    image_path = resolve_image_path(row)

    if image_path is None:
        return None, None

    arr = np.load(image_path)

    slice_idx = int(row["slice_idx"]) if "slice_idx" in row else None

    if arr.ndim == 3:
        if slice_idx is None:
            slice_idx = arr.shape[0] // 2
        slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
        sl = arr[slice_idx]

    elif arr.ndim == 2:
        sl = arr

    elif arr.ndim == 4:
        if slice_idx is None:
            slice_idx = arr.shape[0] // 2
        slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
        sl = arr[slice_idx, :, :, 0]

    else:
        return None, None

    sl = sl.astype(np.float32)

    if sl.max() > 1:
        sl = sl / 255.0

    sl = np.nan_to_num(sl)
    sl = np.clip(sl, 0, 1)

    sl_img = Image.fromarray((sl * 255).astype(np.uint8))
    sl_img = sl_img.resize((img_size, img_size), Image.BILINEAR)

    sl = np.array(sl_img).astype(np.float32) / 255.0

    x = np.expand_dims(sl, axis=0)
    x = np.expand_dims(x, axis=0)

    x = torch.tensor(x, dtype=torch.float32).to(device)

    return x, sl


def gradcam_to_base64(row, feature_idx):
    try:
        image_path = resolve_image_path(row)
        slice_idx = int(row["slice_idx"]) if "slice_idx" in row else -1

        cache_key = (image_path, slice_idx, int(feature_idx))

        if cache_key in GRADCAM_CACHE:
            return GRADCAM_CACHE[cache_key]

        x, image = load_slice_for_gradcam(row, img_size=IMG_SIZE)

        if x is None:
            return None, None

        cam, z = gradcam_engine.generate(x, feature_idx)

        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET,
        )

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        image_rgb = np.stack([image * 255] * 3, axis=-1).astype(np.uint8)
        overlay = (0.6 * image_rgb + 0.4 * heatmap).astype(np.uint8)

        pil_img = Image.fromarray(overlay)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode()
        feature_value = float(z[0, feature_idx])

        result = ("data:image/png;base64," + encoded, feature_value)

        GRADCAM_CACHE[cache_key] = result

        return result

    except Exception as e:
        print("Grad-CAM error:", e)
        return None, None


def make_image_tile(row, feature_idx=126):
    img_src, _ = load_selected_image(row)
    gradcam_src, feature_value = gradcam_to_base64(row, feature_idx)

    if img_src is None:
        return None

    if "volume_file" in row:
        filename = os.path.basename(str(row["volume_file"]))
    elif "full_path" in row:
        filename = os.path.basename(str(row["full_path"]))
    else:
        filename = "unknown"

    feature_text = ""
    if feature_value is not None:
        feature_text = f"f{feature_idx:03d} value: {feature_value:.4f}"

    return html.Div(
        className="image-tile",
        children=[
            html.Div(
                className="tile-two-images",
                children=[
                    html.Div(
                        children=[
                            html.Div("Original", className="mini-title"),
                            html.Img(src=img_src, className="tile-img"),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Div(f"Grad-CAM f{feature_idx:03d}", className="mini-title"),
                            html.Img(
                                src=gradcam_src,
                                className="tile-img" if gradcam_src is not None else "tile-img hidden-image",
                            ),
                        ],
                    ),
                ],
            ),

            html.Div(
                className="tile-caption",
                children=[
                    html.Div(f"{row['type']} | slice {int(row['slice_idx']) if 'slice_idx' in row else 'NA'}"),
                    html.Div(feature_text),
                    html.Div(filename[:52] + "..." if len(filename) > 52 else filename),
                ],
            ),
        ],
    )


# ============================================================
# Plot Functions
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
        uirevision="keep",
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
        uirevision="keep",
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
# Compare Plot Helper Functions
# ============================================================

def get_point_ids_from_selected_data(selected_data):
    if selected_data is None or "points" not in selected_data:
        return []

    point_ids = []

    for p in selected_data["points"]:
        try:
            point_ids.append(int(p["customdata"][0]))
        except Exception:
            pass

    return point_ids


def make_compare_scatter(x_col, y_col, title, selected_ids=None):
    if selected_ids is None:
        selected_ids = []

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="type",
        custom_data=["point_id"],
        opacity=0.45,
        hover_data={
            x_col: ":.3f",
            y_col: ":.3f",
            "type": True,
            "volume_file": True if "volume_file" in df.columns else False,
            "slice_idx": True if "slice_idx" in df.columns else False,
            "point_id": False,
        },
        title=title,
    )

    fig.update_traces(marker=dict(size=5))

    if len(selected_ids) > 0:
        selected_df = df.iloc[selected_ids]

        fig.add_scatter(
            x=selected_df[x_col],
            y=selected_df[y_col],
            mode="markers",
            marker=dict(
                color="red",
                size=10,
                line=dict(color="black", width=1),
            ),
            name="selected",
            customdata=selected_df[["point_id"]].values,
            hovertemplate="Selected point<br>%{x:.3f}, %{y:.3f}<extra></extra>",
        )

    fig.update_layout(
        height=360,
        clickmode="event+select",
        dragmode="lasso",
        template="plotly_white",
        margin=dict(l=35, r=15, t=50, b=35),
        legend_title="",
        uirevision="keep",
    )

    return fig


# ============================================================
# App Layout
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    className="page-container",
    children=[
        dcc.Store(id="selected-point-store", data=[]),

        html.Div(
            className="header",
            children=[
                html.H1("Feature Space Explorer", className="main-title"),
                html.P(
                    "Click one point for a single CT slice. Use box/lasso select to view selected images and Grad-CAM projections.",
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
                                                {
                                                    "label": "Compare PCA + t-SNE + UMAP",
                                                    "value": "compare_all",
                                                },
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
                            id="single-plot-container",
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

                        html.Div(
                            id="compare-plot-container",
                            className="compare-plot-card",
                            style={"display": "none"},
                            children=[
                                dcc.Graph(
                                    id="compare-pca-plot",
                                    figure=make_compare_scatter(
                                        "PC1",
                                        "PC2",
                                        "PCA Projection",
                                        [],
                                    ),
                                    config={
                                        "displayModeBar": True,
                                        "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                                    },
                                ),

                                dcc.Graph(
                                    id="compare-tsne-plot",
                                    figure=make_compare_scatter(
                                        "TSNE1",
                                        "TSNE2",
                                        "t-SNE Projection",
                                        [],
                                    ),
                                    config={
                                        "displayModeBar": True,
                                        "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                                    },
                                ),

                                dcc.Graph(
                                    id="compare-umap-plot",
                                    figure=make_compare_scatter(
                                        "UMAP1",
                                        "UMAP2",
                                        "UMAP Projection",
                                        [],
                                    ),
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
                            className="gradcam-control-card",
                            children=[
                                html.H3("Grad-CAM Feature Projection"),
                                html.Div(
                                    className="control-box",
                                    children=[
                                        html.Label("Choose latent feature"),
                                        dcc.Dropdown(
                                            id="gradcam-feature-dropdown",
                                            options=[
                                                {"label": f"f{i:03d}", "value": i}
                                                for i in range(128)
                                            ],
                                            value=126,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="gradcam-summary",
                                    className="selected-summary",
                                    children="Select a feature and then click/select points to view Grad-CAM.",
                                ),
                            ],
                        ),

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
                                html.H3("Selected Region / Cluster Images + Grad-CAM"),
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
    Output("single-plot-container", "style"),
    Output("compare-plot-container", "style"),
    Input("plot-mode", "value"),
)
def toggle_plot_mode(plot_mode):
    if plot_mode == "compare_all":
        return {"display": "none"}, {"display": "block"}

    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("main-plot", "figure"),
    Input("plot-mode", "value"),
    Input("eps-input", "value"),
    Input("min-samples-input", "value"),
)
def update_plot(plot_mode, eps, min_samples):
    if plot_mode == "pca" or plot_mode == "compare_all":
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
    Output("selected-point-store", "data"),
    Input("main-plot", "selectedData"),
    Input("compare-pca-plot", "selectedData"),
    Input("compare-tsne-plot", "selectedData"),
    Input("compare-umap-plot", "selectedData"),
    State("selected-point-store", "data"),
)
def store_selected_points(
    main_selected,
    pca_selected,
    tsne_selected,
    umap_selected,
    current_store,
):
    triggered = ctx.triggered_id

    selected_map = {
        "main-plot": main_selected,
        "compare-pca-plot": pca_selected,
        "compare-tsne-plot": tsne_selected,
        "compare-umap-plot": umap_selected,
    }

    selected_data = selected_map.get(triggered, None)

    # Extract IDs
    point_ids = get_point_ids_from_selected_data(selected_data)

    # 🚨 KEY FIX:
    # If selection becomes empty (due to redraw), DO NOT overwrite
    if point_ids is None or len(point_ids) == 0:
        return current_store if current_store is not None else []

    return point_ids


@app.callback(
    Output("compare-pca-plot", "figure"),
    Output("compare-tsne-plot", "figure"),
    Output("compare-umap-plot", "figure"),
    Input("selected-point-store", "data"),
)
def update_compare_plots(selected_ids):
    if selected_ids is None:
        selected_ids = []

    return (
        make_compare_scatter("PC1", "PC2", "PCA Projection", selected_ids),
        make_compare_scatter("TSNE1", "TSNE2", "t-SNE Projection", selected_ids),
        make_compare_scatter("UMAP1", "UMAP2", "UMAP Projection", selected_ids),
    )


@app.callback(
    Output("selected-image", "src"),
    Output("selected-image", "className"),
    Output("image-info", "children"),
    Input("main-plot", "clickData"),
    Input("compare-pca-plot", "clickData"),
    Input("compare-tsne-plot", "clickData"),
    Input("compare-umap-plot", "clickData"),
)
def display_clicked_image(main_click, pca_click, tsne_click, umap_click):
    click_map = {
        "main-plot": main_click,
        "compare-pca-plot": pca_click,
        "compare-tsne-plot": tsne_click,
        "compare-umap-plot": umap_click,
    }

    click_data = click_map.get(ctx.triggered_id)

    if click_data is None:
        return None, "selected-image hidden-image", "Click a point to view one image."

    try:
        point_id = click_data["points"][0]["customdata"][0]
        row = df.iloc[int(point_id)]
    except Exception:
        return (
            None,
            "selected-image hidden-image",
            "This plot is not single-image clickable.",
        )

    img_src, info = load_selected_image(row)

    if img_src is None:
        return None, "selected-image hidden-image", info

    return img_src, "selected-image", info


@app.callback(
    Output("selected-summary", "children"),
    Output("selected-image-grid", "children"),
    Output("gradcam-summary", "children"),
    Input("selected-point-store", "data"),
    Input("gradcam-feature-dropdown", "value"),
)
def display_selected_images(point_ids, feature_idx):
    if point_ids is None:
        point_ids = []

    if len(point_ids) == 0:
        return (
            "Use box select or lasso select on any plot to view multiple images.",
            [],
            f"Current Grad-CAM feature: f{feature_idx:03d}",
        )

    selected_df_all = df.iloc[point_ids].copy()

    half_n = MAX_SELECTED_IMAGES // 2

    real_df = selected_df_all[selected_df_all["type"] == "real"].head(half_n)
    fake_df = selected_df_all[selected_df_all["type"] == "fake"].head(half_n)

    selected_df = pd.concat([real_df, fake_df], ignore_index=False)

    real_count = int((selected_df_all["type"] == "real").sum())
    fake_count = int((selected_df_all["type"] == "fake").sum())

    shown_real_count = int((selected_df["type"] == "real").sum())
    shown_fake_count = int((selected_df["type"] == "fake").sum())

    summary = f"""
Selected points: {len(selected_df_all)}
Available real images: {real_count}
Available fake images: {fake_count}

Showing:
Real images: {shown_real_count}
Fake images: {shown_fake_count}
Total shown: {len(selected_df)}

Grad-CAM feature: f{feature_idx:03d}
"""

    gradcam_summary = f"""
Current Grad-CAM feature: f{feature_idx:03d}

In compare mode:
Select dots from PCA / t-SNE / UMAP.
The selected samples remain highlighted in red in all projections.
"""

    tiles = []

    for _, row in selected_df.iterrows():
        tile = make_image_tile(row, feature_idx=feature_idx)

        if tile is not None:
            tiles.append(tile)

    return summary, tiles, gradcam_summary


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)














# import os
# import re
# import base64
# import io

# import numpy as np
# import pandas as pd
# from PIL import Image

# import torch
# import torch.nn as nn
# import cv2

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from sklearn.manifold import TSNE
# import umap

# import plotly.express as px
# from dash import Dash, dcc, html, Input, Output


# # ============================================================
# # Paths
# # ============================================================

# REAL_CSV = r"D:/KLAUS/Vis_Final_Project/real_feature_output/real_features.csv"
# FAKE_CSV = r"D:/KLAUS/Vis_Final_Project/fake_feature_output/fake_features.csv"

# REAL_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_test_real"
# FAKE_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_output_fake"

# MODEL_PATH = r"D:/KLAUS/Vis_Final_Project/best_autoencoder_2d.pth"

# MAX_SELECTED_IMAGES = 10
# LATENT_DIM = 128
# IMG_SIZE = 128
# GRADCAM_CACHE = {}


# # ============================================================
# # Load Data
# # ============================================================

# real = pd.read_csv(REAL_CSV).copy()
# fake = pd.read_csv(FAKE_CSV).copy()

# real["type"] = "real"
# fake["type"] = "fake"

# df = pd.concat([real, fake], ignore_index=True).copy()

# feature_cols = [c for c in df.columns if re.fullmatch(r"f\d{3}", c)]

# if len(feature_cols) == 0:
#     raise ValueError("No feature columns found. Expected columns f000 to f127.")

# X_raw = df[feature_cols].values
# y = df["type"].values

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_raw)

# df["point_id"] = df.index.astype(str)


# # ============================================================
# # PCA
# # ============================================================

# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X_scaled)

# df["PC1"] = X_pca[:, 0]
# df["PC2"] = X_pca[:, 1]

# pc1_var = pca.explained_variance_ratio_[0] * 100
# pc2_var = pca.explained_variance_ratio_[1] * 100


# # ============================================================
# # t-SNE
# # ============================================================

# print("Computing 2D t-SNE...")

# tsne_2d = TSNE(
#     n_components=2,
#     perplexity=30,
#     random_state=42,
#     init="pca",
#     learning_rate="auto",
# )

# X_tsne_2d = tsne_2d.fit_transform(X_scaled)

# df["TSNE1"] = X_tsne_2d[:, 0]
# df["TSNE2"] = X_tsne_2d[:, 1]


# print("Computing 6D t-SNE...")

# tsne_6d = TSNE(
#     n_components=6,
#     perplexity=30,
#     random_state=42,
#     method="exact",
#     init="pca",
#     learning_rate="auto",
# )

# X_tsne_6d = tsne_6d.fit_transform(X_scaled)

# df_tsne_kde = pd.DataFrame(
#     X_tsne_6d,
#     columns=[f"d{i+1}" for i in range(6)]
# )
# df_tsne_kde["type"] = y


# # ============================================================
# # UMAP
# # ============================================================

# print("Computing 2D UMAP...")

# umap_2d = umap.UMAP(
#     n_components=2,
#     n_neighbors=15,
#     min_dist=0.1,
#     random_state=42,
# )

# X_umap_2d = umap_2d.fit_transform(X_scaled)

# df["UMAP1"] = X_umap_2d[:, 0]
# df["UMAP2"] = X_umap_2d[:, 1]


# print("Computing 6D UMAP...")

# umap_6d = umap.UMAP(
#     n_components=6,
#     n_neighbors=15,
#     min_dist=0.1,
#     random_state=42,
# )

# X_umap_6d = umap_6d.fit_transform(X_scaled)

# df_umap_kde = pd.DataFrame(
#     X_umap_6d,
#     columns=[f"d{i+1}" for i in range(6)]
# )
# df_umap_kde["type"] = y


# # ============================================================
# # Image Utilities
# # ============================================================

# def normalize_to_uint8(img):
#     img = np.asarray(img)
#     img = np.nan_to_num(img)

#     p1, p99 = np.percentile(img, [1, 99])
#     img = np.clip(img, p1, p99)

#     img = img - img.min()

#     if img.max() > 0:
#         img = img / img.max()

#     return (img * 255).astype(np.uint8)


# def numpy_slice_to_base64(img):
#     img_uint8 = normalize_to_uint8(img)
#     pil_img = Image.fromarray(img_uint8)

#     buffer = io.BytesIO()
#     pil_img.save(buffer, format="PNG")

#     encoded = base64.b64encode(buffer.getvalue()).decode()
#     return "data:image/png;base64," + encoded


# def resolve_image_path(row):
#     possible_cols = ["full_path", "volume_file"]

#     for col in possible_cols:
#         if col in row:
#             path_value = str(row[col])

#             if os.path.exists(path_value):
#                 return path_value

#             filename = os.path.basename(path_value)

#             if row["type"] == "real":
#                 candidate = os.path.join(REAL_IMAGE_DIR, filename)
#             else:
#                 candidate = os.path.join(FAKE_IMAGE_DIR, filename)

#             if os.path.exists(candidate):
#                 return candidate

#     return None


# def load_selected_image(row):
#     image_path = resolve_image_path(row)

#     if image_path is None:
#         return None, "Image file not found."

#     try:
#         arr = np.load(image_path)

#         slice_idx = int(row["slice_idx"]) if "slice_idx" in row else None

#         if arr.ndim == 3:
#             if slice_idx is None:
#                 slice_idx = arr.shape[0] // 2
#             slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
#             img = arr[slice_idx]

#         elif arr.ndim == 2:
#             img = arr

#         elif arr.ndim == 4:
#             if slice_idx is None:
#                 slice_idx = arr.shape[0] // 2
#             slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
#             img = arr[slice_idx, :, :, 0]

#         else:
#             return None, f"Unsupported numpy shape: {arr.shape}"

#         img_src = numpy_slice_to_base64(img)

#         info = f"""
# Type: {row['type']}
# File: {os.path.basename(image_path)}
# Slice index: {slice_idx}
# PC1: {row['PC1']:.4f}
# PC2: {row['PC2']:.4f}
# TSNE1: {row['TSNE1']:.4f}
# TSNE2: {row['TSNE2']:.4f}
# UMAP1: {row['UMAP1']:.4f}
# UMAP2: {row['UMAP2']:.4f}
# """

#         return img_src, info

#     except Exception as e:
#         return None, f"Error loading image: {str(e)}"


# # ============================================================
# # Grad-CAM Model
# # ============================================================

# class ConvAutoencoder2D(nn.Module):
#     def __init__(self, latent_dim=128):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=False),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=False),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=False),

#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=False),
#         )

#         self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
#         self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=False),

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=False),

#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=False),

#             nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         z = self.fc_enc(x)
#         return z

#     def forward(self, x):
#         z = self.encode(x)
#         return z


# class FeatureGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.activations = None
#         self.gradients = None

#         self._register_hooks()

#     def _register_hooks(self):
#         def forward_hook(module, inp, out):
#             self.activations = out.clone()

#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0].clone()

#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)

#     def generate(self, x, feature_idx):
#         self.model.zero_grad()

#         z = self.model(x)
#         target = z[0, feature_idx]
#         target.backward()

#         activations = self.activations[0]
#         gradients = self.gradients[0]

#         weights = gradients.mean(dim=(1, 2))

#         cam = torch.zeros(
#             activations.shape[1:],
#             dtype=torch.float32,
#             device=activations.device,
#         )

#         for i, w in enumerate(weights):
#             cam += w * activations[i]

#         cam = torch.relu(cam)
#         cam = cam.detach().cpu().numpy()

#         if cam.max() > 0:
#             cam = cam / cam.max()

#         return cam, z.detach().cpu().numpy()


# # ============================================================
# # Load Grad-CAM Model Once
# # ============================================================

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gradcam_model = ConvAutoencoder2D(latent_dim=LATENT_DIM).to(device)

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# gradcam_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# gradcam_model.eval()

# target_layer = gradcam_model.encoder[6]
# gradcam_engine = FeatureGradCAM(gradcam_model, target_layer)


# # ============================================================
# # Grad-CAM Utilities
# # ============================================================

# def load_slice_for_gradcam(row, img_size=128):
#     image_path = resolve_image_path(row)

#     if image_path is None:
#         return None, None

#     arr = np.load(image_path)

#     slice_idx = int(row["slice_idx"]) if "slice_idx" in row else None

#     if arr.ndim == 3:
#         if slice_idx is None:
#             slice_idx = arr.shape[0] // 2
#         slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
#         sl = arr[slice_idx]

#     elif arr.ndim == 2:
#         sl = arr

#     elif arr.ndim == 4:
#         if slice_idx is None:
#             slice_idx = arr.shape[0] // 2
#         slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
#         sl = arr[slice_idx, :, :, 0]

#     else:
#         return None, None

#     sl = sl.astype(np.float32)

#     if sl.max() > 1:
#         sl = sl / 255.0

#     sl = np.nan_to_num(sl)
#     sl = np.clip(sl, 0, 1)

#     sl_img = Image.fromarray((sl * 255).astype(np.uint8))
#     sl_img = sl_img.resize((img_size, img_size), Image.BILINEAR)

#     sl = np.array(sl_img).astype(np.float32) / 255.0

#     x = np.expand_dims(sl, axis=0)
#     x = np.expand_dims(x, axis=0)

#     x = torch.tensor(x, dtype=torch.float32).to(device)

#     return x, sl


# def gradcam_to_base64(row, feature_idx):
#     try:
#         image_path = resolve_image_path(row)
#         slice_idx = int(row["slice_idx"]) if "slice_idx" in row else -1

#         cache_key = (image_path, slice_idx, int(feature_idx))

#         if cache_key in GRADCAM_CACHE:
#             return GRADCAM_CACHE[cache_key]

#         x, image = load_slice_for_gradcam(row, img_size=IMG_SIZE)

#         if x is None:
#             return None, None

#         cam, z = gradcam_engine.generate(x, feature_idx)

#         cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

#         heatmap = cv2.applyColorMap(
#             np.uint8(255 * cam_resized),
#             cv2.COLORMAP_JET,
#         )

#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#         image_rgb = np.stack([image * 255] * 3, axis=-1).astype(np.uint8)
#         overlay = (0.6 * image_rgb + 0.4 * heatmap).astype(np.uint8)

#         pil_img = Image.fromarray(overlay)

#         buffer = io.BytesIO()
#         pil_img.save(buffer, format="PNG")

#         encoded = base64.b64encode(buffer.getvalue()).decode()
#         feature_value = float(z[0, feature_idx])

#         result = ("data:image/png;base64," + encoded, feature_value)

#         GRADCAM_CACHE[cache_key] = result

#         return result

#     except Exception as e:
#         print("Grad-CAM error:", e)
#         return None, None


# def make_image_tile(row, feature_idx=126):
#     img_src, _ = load_selected_image(row)
#     gradcam_src, feature_value = gradcam_to_base64(row, feature_idx)

#     if img_src is None:
#         return None

#     if "volume_file" in row:
#         filename = os.path.basename(str(row["volume_file"]))
#     elif "full_path" in row:
#         filename = os.path.basename(str(row["full_path"]))
#     else:
#         filename = "unknown"

#     feature_text = ""
#     if feature_value is not None:
#         feature_text = f"f{feature_idx:03d} value: {feature_value:.4f}"

#     return html.Div(
#         className="image-tile",
#         children=[
#             html.Div(
#                 className="tile-two-images",
#                 children=[
#                     html.Div(
#                         children=[
#                             html.Div("Original", className="mini-title"),
#                             html.Img(src=img_src, className="tile-img"),
#                         ],
#                     ),
#                     html.Div(
#                         children=[
#                             html.Div(f"Grad-CAM f{feature_idx:03d}", className="mini-title"),
#                             html.Img(
#                                 src=gradcam_src,
#                                 className="tile-img" if gradcam_src is not None else "tile-img hidden-image",
#                             ),
#                         ],
#                     ),
#                 ],
#             ),

#             html.Div(
#                 className="tile-caption",
#                 children=[
#                     html.Div(f"{row['type']} | slice {int(row['slice_idx']) if 'slice_idx' in row else 'NA'}"),
#                     html.Div(feature_text),
#                     html.Div(filename[:52] + "..." if len(filename) > 52 else filename),
#                 ],
#             ),
#         ],
#     )


# # ============================================================
# # Plot Functions
# # ============================================================

# def base_scatter(data, x, y_col, color, title, x_title, y_title):
#     fig = px.scatter(
#         data,
#         x=x,
#         y=y_col,
#         color=color,
#         custom_data=["point_id"],
#         opacity=0.7,
#         hover_data={
#             x: ":.3f",
#             y_col: ":.3f",
#             "type": True,
#             "volume_file": True if "volume_file" in data.columns else False,
#             "slice_idx": True if "slice_idx" in data.columns else False,
#             "point_id": False,
#         },
#         title=title,
#     )

#     fig.update_traces(marker=dict(size=6))

#     fig.update_layout(
#         height=560,
#         clickmode="event+select",
#         dragmode="lasso",
#         template="plotly_white",
#         margin=dict(l=35, r=15, t=55, b=35),
#         xaxis_title=x_title,
#         yaxis_title=y_title,
#         legend_title="",
#     )

#     return fig


# def make_pca_plot():
#     return base_scatter(
#         df,
#         "PC1",
#         "PC2",
#         "type",
#         "PCA: Real vs Fake Features",
#         f"PC1 ({pc1_var:.2f}%)",
#         f"PC2 ({pc2_var:.2f}%)",
#     )


# def make_tsne_plot():
#     return base_scatter(
#         df,
#         "TSNE1",
#         "TSNE2",
#         "type",
#         "t-SNE: Real vs Fake Features",
#         "t-SNE 1",
#         "t-SNE 2",
#     )


# def make_umap_plot():
#     return base_scatter(
#         df,
#         "UMAP1",
#         "UMAP2",
#         "type",
#         "UMAP: Real vs Fake Features",
#         "UMAP 1",
#         "UMAP 2",
#     )


# def make_dbscan_plot(eps=0.65, min_samples=5):
#     temp_df = df.copy()

#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(temp_df[["PC1", "PC2"]].values)

#     temp_df["cluster"] = labels
#     temp_df["cluster_label"] = temp_df["cluster"].apply(
#         lambda x: "noise" if x == -1 else f"cluster {x}"
#     )

#     fig = px.scatter(
#         temp_df,
#         x="PC1",
#         y="PC2",
#         color="cluster_label",
#         symbol="type",
#         custom_data=["point_id"],
#         opacity=0.75,
#         hover_data={
#             "PC1": ":.3f",
#             "PC2": ":.3f",
#             "type": True,
#             "cluster_label": True,
#             "volume_file": True if "volume_file" in temp_df.columns else False,
#             "slice_idx": True if "slice_idx" in temp_df.columns else False,
#             "point_id": False,
#         },
#         title=f"DBSCAN on PCA Space: eps={eps}, min_samples={min_samples}",
#     )

#     fig.update_traces(marker=dict(size=6))

#     fig.update_layout(
#         height=560,
#         clickmode="event+select",
#         dragmode="lasso",
#         template="plotly_white",
#         margin=dict(l=35, r=15, t=55, b=35),
#         xaxis_title="PC1",
#         yaxis_title="PC2",
#         legend_title="Cluster",
#     )

#     return fig


# def make_tsne_kde_plot():
#     fig = px.scatter_matrix(
#         df_tsne_kde,
#         dimensions=[f"d{i+1}" for i in range(6)],
#         color="type",
#         opacity=0.45,
#         title="t-SNE Pairwise Density View",
#     )

#     fig.update_traces(
#         diagonal_visible=False,
#         showupperhalf=False,
#         marker=dict(size=3),
#     )

#     fig.update_layout(
#         height=560,
#         template="plotly_white",
#         margin=dict(l=35, r=15, t=55, b=35),
#     )

#     return fig


# def make_umap_kde_plot():
#     fig = px.scatter_matrix(
#         df_umap_kde,
#         dimensions=[f"d{i+1}" for i in range(6)],
#         color="type",
#         opacity=0.45,
#         title="UMAP Pairwise Density View",
#     )

#     fig.update_traces(
#         diagonal_visible=False,
#         showupperhalf=False,
#         marker=dict(size=3),
#     )

#     fig.update_layout(
#         height=560,
#         template="plotly_white",
#         margin=dict(l=35, r=15, t=55, b=35),
#     )

#     return fig


# # ============================================================
# # App Layout
# # ============================================================

# app = Dash(__name__)

# app.layout = html.Div(
#     className="page-container",
#     children=[
#         html.Div(
#             className="header",
#             children=[
#                 html.H1("Feature Space Explorer", className="main-title"),
#                 html.P(
#                     "Click one point for a single CT slice. Use box/lasso select to view selected images and Grad-CAM projections.",
#                     className="subtitle",
#                 ),
#             ],
#         ),

#         html.Div(
#             className="dashboard-layout",
#             children=[
#                 html.Div(
#                     className="left-panel",
#                     children=[
#                         html.Div(
#                             className="control-card",
#                             children=[
#                                 html.Div(
#                                     className="control-box",
#                                     children=[
#                                         html.Label("Visualization Mode"),
#                                         dcc.Dropdown(
#                                             id="plot-mode",
#                                             options=[
#                                                 {"label": "PCA Plot", "value": "pca"},
#                                                 {"label": "Perform DBSCAN on PCA", "value": "dbscan"},
#                                                 {"label": "t-SNE Plot", "value": "tsne"},
#                                                 {"label": "KDE / Pair Plot of t-SNE", "value": "tsne_kde"},
#                                                 {"label": "UMAP Plot", "value": "umap"},
#                                                 {"label": "KDE / Pair Plot of UMAP", "value": "umap_kde"},
#                                             ],
#                                             value="pca",
#                                             clearable=False,
#                                         ),
#                                     ],
#                                 ),

#                                 html.Div(
#                                     className="dbscan-controls",
#                                     children=[
#                                         html.Div(
#                                             className="control-box",
#                                             children=[
#                                                 html.Label("DBSCAN eps"),
#                                                 dcc.Input(
#                                                     id="eps-input",
#                                                     type="number",
#                                                     value=0.65,
#                                                     step=0.05,
#                                                     min=0.01,
#                                                     className="number-input",
#                                                 ),
#                                             ],
#                                         ),

#                                         html.Div(
#                                             className="control-box",
#                                             children=[
#                                                 html.Label("min_samples"),
#                                                 dcc.Input(
#                                                     id="min-samples-input",
#                                                     type="number",
#                                                     value=5,
#                                                     step=1,
#                                                     min=1,
#                                                     className="number-input",
#                                                 ),
#                                             ],
#                                         ),
#                                     ],
#                                 ),
#                             ],
#                         ),

#                         html.Div(
#                             className="plot-card",
#                             children=[
#                                 dcc.Graph(
#                                     id="main-plot",
#                                     figure=make_pca_plot(),
#                                     config={
#                                         "displayModeBar": True,
#                                         "modeBarButtonsToAdd": ["lasso2d", "select2d"],
#                                     },
#                                 ),
#                             ],
#                         ),
#                     ],
#                 ),

#                 html.Div(
#                     className="right-panel",
#                     children=[
#                         html.Div(
#                             className="gradcam-control-card",
#                             children=[
#                                 html.H3("Grad-CAM Feature Projection"),
#                                 html.Div(
#                                     className="control-box",
#                                     children=[
#                                         html.Label("Choose latent feature"),
#                                         dcc.Dropdown(
#                                             id="gradcam-feature-dropdown",
#                                             options=[
#                                                 {"label": f"f{i:03d}", "value": i}
#                                                 for i in range(128)
#                                             ],
#                                             value=126,
#                                             clearable=False,
#                                         ),
#                                     ],
#                                 ),
#                                 html.Div(
#                                     id="gradcam-summary",
#                                     className="selected-summary",
#                                     children="Select a feature and then click/select points to view Grad-CAM.",
#                                 ),
#                             ],
#                         ),

#                         html.Div(
#                             className="image-card",
#                             children=[
#                                 html.H3("Single Selected CT Slice"),
#                                 html.Div(
#                                     id="image-info",
#                                     className="image-info",
#                                     children="Click a point to view one image.",
#                                 ),
#                                 html.Img(
#                                     id="selected-image",
#                                     className="selected-image hidden-image",
#                                 ),
#                             ],
#                         ),

#                         html.Div(
#                             className="cluster-card",
#                             children=[
#                                 html.H3("Selected Region / Cluster Images + Grad-CAM"),
#                                 html.Div(
#                                     id="selected-summary",
#                                     className="selected-summary",
#                                     children="Use box select or lasso select on the plot to view multiple images.",
#                                 ),
#                                 html.Div(
#                                     id="selected-image-grid",
#                                     className="selected-image-grid",
#                                 ),
#                             ],
#                         ),
#                     ],
#                 ),
#             ],
#         ),
#     ],
# )


# # ============================================================
# # Callbacks
# # ============================================================

# @app.callback(
#     Output("main-plot", "figure"),
#     Input("plot-mode", "value"),
#     Input("eps-input", "value"),
#     Input("min-samples-input", "value"),
# )
# def update_plot(plot_mode, eps, min_samples):
#     if plot_mode == "pca":
#         return make_pca_plot()

#     if plot_mode == "dbscan":
#         eps = float(eps) if eps is not None else 0.65
#         min_samples = int(min_samples) if min_samples is not None else 5
#         return make_dbscan_plot(eps=eps, min_samples=min_samples)

#     if plot_mode == "tsne":
#         return make_tsne_plot()

#     if plot_mode == "tsne_kde":
#         return make_tsne_kde_plot()

#     if plot_mode == "umap":
#         return make_umap_plot()

#     if plot_mode == "umap_kde":
#         return make_umap_kde_plot()

#     return make_pca_plot()


# @app.callback(
#     Output("selected-image", "src"),
#     Output("selected-image", "className"),
#     Output("image-info", "children"),
#     Input("main-plot", "clickData"),
# )
# def display_clicked_image(clickData):
#     if clickData is None:
#         return None, "selected-image hidden-image", "Click a point to view one image."

#     try:
#         point_id = clickData["points"][0]["customdata"][0]
#         row = df.iloc[int(point_id)]
#     except Exception:
#         return (
#             None,
#             "selected-image hidden-image",
#             "This plot is not single-image clickable. Use PCA, DBSCAN, t-SNE, or UMAP scatter plot.",
#         )

#     img_src, info = load_selected_image(row)

#     if img_src is None:
#         return None, "selected-image hidden-image", info

#     return img_src, "selected-image", info


# @app.callback(
#     Output("selected-summary", "children"),
#     Output("selected-image-grid", "children"),
#     Output("gradcam-summary", "children"),
#     Input("main-plot", "selectedData"),
#     Input("gradcam-feature-dropdown", "value"),
# )
# def display_selected_images(selectedData, feature_idx):
#     if selectedData is None or "points" not in selectedData:
#         return (
#             "Use box select or lasso select on the plot to view multiple images.",
#             [],
#             f"Current Grad-CAM feature: f{feature_idx:03d}",
#         )

#     selected_points = selectedData["points"]

#     point_ids = []

#     for p in selected_points:
#         try:
#             point_ids.append(int(p["customdata"][0]))
#         except Exception:
#             pass

#     if len(point_ids) == 0:
#         return (
#             "No image-linked points selected. Use PCA, DBSCAN, t-SNE, or UMAP scatter plot.",
#             [],
#             f"Current Grad-CAM feature: f{feature_idx:03d}",
#         )

#     # selected_df = df.iloc[point_ids].copy()

#     # real_count = int((selected_df["type"] == "real").sum())
#     # fake_count = int((selected_df["type"] == "fake").sum())

#     selected_df_all = df.iloc[point_ids].copy()

#     real_df = selected_df_all[selected_df_all["type"] == "real"].head(MAX_SELECTED_IMAGES // 2)
#     fake_df = selected_df_all[selected_df_all["type"] == "fake"].head(MAX_SELECTED_IMAGES // 2)

#     selected_df = pd.concat([real_df, fake_df], ignore_index=False)

#     real_count = int((selected_df_all["type"] == "real").sum())
#     fake_count = int((selected_df_all["type"] == "fake").sum())

#     shown_real_count = int((selected_df["type"] == "real").sum())
#     shown_fake_count = int((selected_df["type"] == "fake").sum())

#     summary = f"""
# Selected points: {len(selected_df_all)}
# Available real images: {real_count}
# Available fake images: {fake_count}

# Showing:
# Real images: {shown_real_count}
# Fake images: {shown_fake_count}
# Total shown: {len(selected_df)}

# Grad-CAM feature: f{feature_idx:03d}
# """

#     gradcam_summary = f"""
# Current Grad-CAM feature: f{feature_idx:03d}
# For every selected image, the dashboard shows:
# 1. Original CT slice
# 2. Grad-CAM overlay for f{feature_idx:03d}
# """

#     tiles = []

#     for _, row in selected_df.iterrows():
#         tile = make_image_tile(row, feature_idx=feature_idx)

#         if tile is not None:
#             tiles.append(tile)

#     return summary, tiles, gradcam_summary


# if __name__ == "__main__":
#     app.run(debug=False, host="127.0.0.1", port=8050)



