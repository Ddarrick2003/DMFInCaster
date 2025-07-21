import streamlit as st

# Set page config and favicon/logo
def set_page_config():
    st.set_page_config(
        page_title="FinCaster",
        page_icon="🟢",  # You can replace this with an emoji or custom icon later
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Inject custom CSS for modern light mode UI
def inject_custom_css():
    st.markdown("""
    <style>
        /* Hide Streamlit sidebar toggle completely */
        [data-testid="collapsedControl"] {
            display: none;
        }

        /* Make top padding tighter */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* Modern card-like visuals */
        .stApp {
            background-color: #F7FAF9;
        }

        /* Hide default hamburger and footer */
        header, footer {visibility: hidden;}

        /* Font overrides */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        /* Section title styling */
        h1, h2, h3 {
            color: #1F4E3D;
        }
    </style>
    """, unsafe_allow_html=True)
