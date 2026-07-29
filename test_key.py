import streamlit as st
import pydeck as pdk
try:
    st.pydeck_chart(pdk.Deck(), key="test_key")
    print("SUCCESS")
except Exception as e:
    print("FAILED:", type(e))
