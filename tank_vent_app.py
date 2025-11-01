
import streamlit as st
import numpy as np
import pyvista as pv
from stpyvista import stpyvista

st.set_page_config(page_title="C-NLOPB Tank Vent", layout="wide")
pv.global_theme.background = 'white'

def vol(o, L, D, H):
    return np.pi * (D/2)**2 * (H if o=="Vertical" else L)

st.title("C-NLOPB Tank Ventilation")
c1, c2 = st.columns(2)
with c1:
    o = st.selectbox("Orientation", ["Vertical", "Horizontal"])
    if o=="Vertical":
        D = st.number_input("Diameter (m)",0.1,50.0,3.0)
        H = st.number_input("Height (m)",0.1,100.0,10.0)
        L = H
    else:
        L = st.number_input("Length (m)",0.1,100.0,15.0)
        D = st.number_input("Diameter (m)",0.1,20.0,4.0)
        H = D
    V = vol(o,L,D,H)
    st.metric("Volume",f"{V:,.1f} m³")
    st.metric("Flow (12 ACH)",f"{V*12:,.0f} m³/h")

with c2:
    LL = L if o=="Horizontal" else H
    HH = H if o=="Vertical" else D
    inlet = np.array([
        st.slider("Inlet X",-LL/2,LL/2,0.0,0.1),
        st.slider("Inlet Y",-D/2,D/2,0.0,0.1),
        st.slider("Inlet Z",-HH/2,HH/2,-HH/3,0.1)
    ])
    outlet = np.array([
        st.slider("Outlet X",-LL/2,LL/2,0.0,0.1),
        st.slider("Outlet Y",-D/2,D/2,0.1),
        st.slider("Outlet Z",-HH/2,HH/2,HH/3,0.1)
    ])

def field(i,o,res=25):
    x,y,z = [np.linspace(-a/2,a/2,res) for a in (LL,D,HH)]
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    P = np.column_stack((X.ravel(),Y.ravel(),Z.ravel()))
    V = np.zeros_like(P)
    for p,s in zip([i,o],[1,-1]):
        r = np.linalg.norm(P-p,axis=1,keepdims=True)+1e-6
        V += s*(P-p)/(r**3)
    g = pv.UniformGrid()
    g.dimensions = (res,res,res)
    g.spacing = (LL/(res-1),D/(res-1),HH/(res-1))
    g.origin = (-LL/2,-D/2,-HH/2)
    g.cell_data["V"] = V.astype(np.float32)
    g.set_active_vectors("V")
    return g

g = field(inlet,outlet)
s = g.streamlines_from_source(pv.PointSet(inlet+np.random.randn(80,3)*0.05),max_time=50)
p = pv.Plotter()
p.add_mesh(pv.Cylinder(radius=D/2,height=H if o=="Vertical" else L,resolution=40).rotate_z(90 if o=="Horizontal" else 0),color='lightblue',opacity=0.15)
p.add_mesh(s,line_width=2,cmap='turbo')
p.add_points(inlet,color='red',point_size=15)
p.add_points(outlet,color='blue',point_size=15)
stpyvista(p,height=500,key="flow")

cov = np.mean(np.linalg.norm(g["V"],axis=1)>0.03)
st.metric("Coverage",f"{cov*100:.1f}%")