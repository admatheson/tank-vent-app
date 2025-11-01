import streamlit as st
import numpy as np
import pyvista as pv
from stpyvista import stpyvista

st.set_page_config(page_title="C-NLOPB Tank Vent (ft)", layout="wide")

st.title("C-NLOPB Tank Ventilation – Imperial (ft)")
st.markdown("**12 Air Changes per Hour (ACH) – SOR/96-118**")

col1, col2 = st.columns(2)

with col1:
    orientation = st.selectbox("Tank Orientation", ["Vertical", "Horizontal"])
    if orientation == "Vertical":
        diameter_ft = st.number_input("Diameter (ft)", 0.3, 160.0, 10.0)
        height_ft = st.number_input("Height (ft)", 0.3, 330.0, 33.0)
        length_ft = height_ft
    else:
        length_ft = st.number_input("Length (ft)", 0.3, 330.0, 50.0)
        diameter_ft = st.number_input("Diameter (ft)", 0.3, 65.0, 13.0)
        height_ft = diameter_ft

    volume_ft3 = 3.14159 * (diameter_ft/2)**2 * (height_ft if orientation == "Vertical" else length_ft)
    flow_cfh = volume_ft3 * 12

    st.metric("Tank Volume", f"{volume_ft3:,.1f} ft³")
    st.metric("Required Flow Rate", f"{flow_cfh:,.0f} CFH")

with col2:
    L = length_ft if orientation == "Horizontal" else height_ft
    H = height_ft if orientation == "Vertical" else diameter_ft
    inlet = np.array([
        st.slider("Inlet X (ft)", -L/2, L/2, 0.0, 0.3),
        st.slider("Inlet Y (ft)", -diameter_ft/2, diameter_ft/2, 0.0, 0.3),
        st.slider("Inlet Z (ft)", -H/2, H/2, -H/3, 0.3)
    ])
    outlet = np.array([
        st.slider("Outlet X (ft)", -L/2, L/2, 0.0, 0.3),
        st.slider("Outlet Y (ft)", -diameter_ft/2, diameter_ft/2, 0.3),
        st.slider("Outlet Z (ft)", -H/2, H/2, H/3, 0.3)
    ])

# Cell-centered velocity
def field(i, o, res=25):
    n = res - 1
    if n < 1: n = 1
    x = np.linspace(-L/2 + L/(2*n), L/2 - L/(2*n), n)
    y = np.linspace(-diameter_ft/2 + diameter_ft/(2*n), diameter_ft/2 - diameter_ft/(2*n), n)
    z = np.linspace(-H/2 + H/(2*n), H/2 - H/(2*n), n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    V = np.zeros_like(P)
    for p, s in zip([i, o], [1, -1]):
        v = P - p
        r = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        V += s * v / (r**3)
    
    grid = pv.ImageData()
    grid.dimensions = (n+1, n+1, n+1)
    grid.spacing = (L/n, diameter_ft/n, H/n)
    grid.origin = (-L/2, -diameter_ft/2, -H/2)
    grid.cell_data["velocity"] = V.astype(np.float32)
    grid.set_active_vectors("velocity")
    return grid

g = field(inlet, outlet)
streamlines = g.streamlines_from_source(pv.PointSet(inlet + np.random.randn(80,3)*0.2), max_time=50)

st.subheader("3D Air Flow Simulation")
plotter = pv.Plotter()
cyl = pv.Cylinder(radius=diameter_ft/2, height=height_ft if orientation=="Vertical" else length_ft, resolution=40)
if orientation == "Horizontal":
    cyl.rotate_z(90)
plotter.add_mesh(cyl, color="lightblue", opacity=0.2)

# FIXED: Use n_lines and get_line(i)
for i in range(streamlines.n_lines):
    line = streamlines.get_line(i)
    if line.n_points > 1:
        plotter.add_lines(line.points, color="orange", width=2)

plotter.add_points(inlet, color="red", point_size=20)
plotter.add_points(outlet, color="blue", point_size=20)
stpyvista(plotter, height=500, key="flow")

# Coverage
cov = np.mean(np.linalg.norm(g["velocity"], axis=1) > 0.1)
st.metric("Airflow Coverage", f"{cov*100:.1f}%")

st.subheader("Download Report")
report = f"""
C-NLOPB Ventilation Report (Imperial)
====================================
Tank Type: {orientation} Cylindrical
Dimensions: Length={length_ft:.1f} ft, Diameter={diameter_ft:.1f} ft, Height={height_ft:.1f} ft
Volume: {volume_ft3:.1f} ft³
Required Flow Rate: {flow_cfh:.0f} CFH (12 ACH)

Inlet Location (ft): {inlet.round(2)}
Outlet Location (ft): {outlet.round(2)}
Airflow Coverage: {cov*100:.1f}%

Compliant: {'Yes' if cov >= 0.85 else 'No – Improve placement'}
"""
st.download_button("Download Report (TXT)", report, "cnlopb_report_imperial.txt")
