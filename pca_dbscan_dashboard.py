# ============================================================
# Interactive PCA + DBSCAN Dashboard
# Real vs Fake Feature Space + Click-to-View Image Slice
# ============================================================

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

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output


# ============================================================
# 1. Paths
# ============================================================

REAL_CSV = r"D:/KLAUS/Vis_Final_Project/real_feature_output/real_features.csv"
FAKE_CSV = r"D:/KLAUS/Vis_Final_Project/fake_feature_output/fake_features.csv"

REAL_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_test_real"
FAKE_IMAGE_DIR = r"D:/KLAUS/Vis_Final_Project/gridimages2_npy_output_fake"


# ============================================================
# 2. Load feature CSV files
# ============================================================

real = pd.read_csv(REAL_CSV).copy()
fake = pd.read_csv(FAKE_CSV).copy()

real["type"] = "real"
fake["type"] = "fake"

df = pd.concat([real, fake], ignore_index=True).copy()

feature_cols = [c for c in df.columns if re.fullmatch(r"f\d{3}", c)]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found. Expected columns like f000, f001, ..., f127.")

print(f"Total samples: {len(df)}")
print(f"Feature columns found: {len(feature_cols)}")


# ============================================================
# 3. PCA
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

df["point_id"] = df.index.astype(str)

explained_var = pca.explained_variance_ratio_
pc1_var = explained_var[0] * 100
pc2_var = explained_var[1] * 100


# ============================================================
# 4. DBSCAN function
# ============================================================

def perform_dbscan(eps=0.65, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df[["PC1", "PC2"]].values)
    return labels


# ============================================================
# 5. Convert numpy slice to image for browser display
# ============================================================

def normalize_to_uint8(img):
    img = np.asarray(img)

    img = np.nan_to_num(img)

    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    img = (img * 255).astype(np.uint8)
    return img


def numpy_slice_to_base64(img):
    img_uint8 = normalize_to_uint8(img)

    pil_img = Image.fromarray(img_uint8)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded


def resolve_image_path(row):
    """
    Uses volume_file from CSV.
    If the CSV stores only filename, this will search inside real/fake folder.
    """

    if "volume_file" not in row:
        return None

    volume_file = str(row["volume_file"])

    # Case 1: full valid path
    if os.path.exists(volume_file):
        return volume_file

    # Case 2: filename only
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

        # If arr is 3D volume
        if arr.ndim == 3:
            if slice_idx is None:
                slice_idx = arr.shape[0] // 2

            slice_idx = max(0, min(slice_idx, arr.shape[0] - 1))
            img = arr[slice_idx]

        # If arr is already 2D image
        elif arr.ndim == 2:
            img = arr

        # If arr has channel dimension
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
        """

        return img_src, info

    except Exception as e:
        return None, f"Error loading image: {str(e)}"


# ============================================================
# 6. Plot functions
# ============================================================

def make_pca_plot():
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="type",
        custom_data=["point_id"],
        opacity=0.65,
        hover_data={
            "PC1": ":.3f",
            "PC2": ":.3f",
            "type": True,
            "volume_file": True if "volume_file" in df.columns else False,
            "slice_idx": True if "slice_idx" in df.columns else False,
            "point_id": False,
        },
        title="PCA: Real vs Fake Features",
    )

    fig.update_traces(marker=dict(size=7))

    fig.update_layout(
        height=700,
        clickmode="event+select",
        xaxis_title=f"PC1 ({pc1_var:.2f}% variance)",
        yaxis_title=f"PC2 ({pc2_var:.2f}% variance)",
        legend_title="Type",
        template="plotly_white",
    )

    return fig


def make_dbscan_plot(eps=0.65, min_samples=5):
    temp_df = df.copy()
    temp_df["cluster"] = perform_dbscan(eps=eps, min_samples=min_samples)
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

    fig.update_traces(marker=dict(size=7))

    fig.update_layout(
        height=700,
        clickmode="event+select",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Cluster",
        template="plotly_white",
    )

    return fig


# ============================================================
# 7. Dash App
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "20px",
        "backgroundColor": "#f8f9fb",
    },
    children=[
        html.H1(
            "Interactive PCA + DBSCAN Feature Dashboard",
            style={"textAlign": "center", "marginBottom": "5px"},
        ),

        html.P(
            "Click any point in the PCA or DBSCAN plot to view the corresponding CT slice.",
            style={"textAlign": "center", "color": "#555"},
        ),

        html.Div(
            style={
                "display": "flex",
                "gap": "20px",
                "marginTop": "20px",
            },
            children=[
                html.Div(
                    style={
                        "width": "72%",
                        "backgroundColor": "white",
                        "padding": "15px",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "15px",
                                "alignItems": "center",
                                "marginBottom": "15px",
                            },
                            children=[
                                html.Div(
                                    style={"width": "35%"},
                                    children=[
                                        html.Label("Plot Mode"),
                                        dcc.Dropdown(
                                            id="plot-mode",
                                            options=[
                                                {
                                                    "label": "Show PCA Plot",
                                                    "value": "pca",
                                                },
                                                {
                                                    "label": "Perform DBSCAN on PCA",
                                                    "value": "dbscan",
                                                },
                                            ],
                                            value="pca",
                                            clearable=False,
                                        ),
                                    ],
                                ),

                                html.Div(
                                    style={"width": "20%"},
                                    children=[
                                        html.Label("DBSCAN eps"),
                                        dcc.Input(
                                            id="eps-input",
                                            type="number",
                                            value=0.65,
                                            step=0.05,
                                            min=0.01,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),

                                html.Div(
                                    style={"width": "20%"},
                                    children=[
                                        html.Label("min_samples"),
                                        dcc.Input(
                                            id="min-samples-input",
                                            type="number",
                                            value=5,
                                            step=1,
                                            min=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        dcc.Graph(
                            id="main-plot",
                            figure=make_pca_plot(),
                            config={"displayModeBar": True},
                        ),
                    ],
                ),

                html.Div(
                    style={
                        "width": "28%",
                        "backgroundColor": "white",
                        "padding": "15px",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3("Selected Image"),

                        html.Div(
                            id="image-info",
                            style={
                                "whiteSpace": "pre-line",
                                "fontSize": "13px",
                                "color": "#333",
                                "marginBottom": "10px",
                            },
                            children="Click a point to view its image.",
                        ),

                        html.Img(
                            id="selected-image",
                            style={
                                "width": "100%",
                                "border": "1px solid #ddd",
                                "borderRadius": "8px",
                                "display": "none",
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# 8. Callbacks
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

    eps = float(eps) if eps is not None else 0.65
    min_samples = int(min_samples) if min_samples is not None else 5

    return make_dbscan_plot(eps=eps, min_samples=min_samples)


@app.callback(
    Output("selected-image", "src"),
    Output("selected-image", "style"),
    Output("image-info", "children"),
    Input("main-plot", "clickData"),
)
def display_clicked_image(clickData):
    if clickData is None:
        return (
            None,
            {
                "width": "100%",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "display": "none",
            },
            "Click a point to view its image.",
        )

    point_id = clickData["points"][0]["customdata"][0]
    row = df.iloc[int(point_id)]

    img_src, info = load_selected_image(row)

    if img_src is None:
        return (
            None,
            {
                "width": "100%",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "display": "none",
            },
            info,
        )

    return (
        img_src,
        {
            "width": "100%",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "display": "block",
        },
        info,
    )


# ============================================================
# 9. Run
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)