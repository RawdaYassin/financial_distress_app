"""
MENA Financial Distress Analyzer
=================================
Clean client-facing deployment with rich SHAP explanations.
Author: Financial AI Research Team
Date:   2026-02-18

UPDATED:
  - Models are downloaded automatically from Google Drive on first run
    using gdown (pip install gdown).  Files are cached in a local
    .model_cache/ folder so they are only downloaded once.
  - model_metadata.json dependency removed entirely.  The feature list
    is sourced from SELECTED_FEATURES in this file (matches train_all_models.py).
  - Handles both raw-estimator pickles (complete_modeling_pipeline.py output)
    and legacy dict-wrapped formats.

HOW TO CONFIGURE GOOGLE DRIVE:
  1. Open the folder: https://drive.google.com/drive/folders/1VrvhaEWcI2TrU0tWKfgzW5TCVJlL7QAC
  2. Right-click each file → "Get link" → copy the file ID
     (the long string between /d/ and /view in the shareable URL)
  3. Paste the IDs into GDRIVE_FILE_IDS below.
  4. Make sure each file is shared as "Anyone with the link can view".
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="MENA Financial Distress Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-Powered Financial Analysis for MENA Companies"},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, .stDeployButton { display: none; }
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        color: #1a1a2e;
    }
    .page-title {
        font-size: 2.4rem; font-weight: 800; color: #1e3c72;
        text-align: center; padding: 2rem 0 0.35rem; letter-spacing: -0.5px;
    }
    .page-subtitle {
        text-align: center; color: #5a6a85; font-size: 1.05rem;
        padding-bottom: 1.6rem; border-bottom: 1px solid #e4e8f0; margin-bottom: 2rem;
    }
    .sec-title {
        font-size: 1.25rem; font-weight: 700; color: #1e3c72;
        margin: 2.2rem 0 1rem; padding-bottom: 0.45rem;
        border-bottom: 3px solid #2a5298;
    }
    .co-banner {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 55%, #4776c8 100%);
        color: white; padding: 1.8rem 2.4rem; border-radius: 16px;
        margin: 1.2rem 0 1.8rem; box-shadow: 0 6px 24px rgba(30,60,114,0.22);
    }
    .co-banner h2 { margin: 0; font-size: 1.75rem; font-weight: 800; }
    .co-banner p  { margin: 0.35rem 0 0; opacity: 0.82; font-size: 0.97rem; }
    .filter-panel {
        background: #f5f7ff; border: 1px solid #d8dff5;
        border-radius: 14px; padding: 1.6rem 2rem 1.3rem; margin-bottom: 1.6rem;
    }
    .verdict-healthy {
        background: linear-gradient(135deg,#e6faf2,#cef4e4);
        border: 2px solid #27ae60; border-radius: 14px;
        padding: 1.8rem; text-align: center;
    }
    .verdict-distress {
        background: linear-gradient(135deg,#fdf0f0,#fddede);
        border: 2px solid #e74c3c; border-radius: 14px;
        padding: 1.8rem; text-align: center;
    }
    .verdict-label { font-size: 1.45rem; font-weight: 800; margin: 0; }
    .verdict-sub   { font-size: 0.88rem; margin: 0.4rem 0 0; opacity: 0.72; }
    .risk-badge {
        border-radius: 14px; padding: 1.8rem; text-align: center; color: white;
    }
    .risk-badge h4 {
        margin: 0; font-size: 0.82rem; opacity: 0.82;
        font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
    }
    .risk-badge h2 { margin: 0.45rem 0 0; font-size: 1.85rem; font-weight: 800; }
    .risk-badge p  { margin: 0.25rem 0 0; font-size: 0.8rem; opacity: 0.8; }
    .exp-card {
        border-radius: 14px; padding: 1.5rem 2.0rem; margin: 0.2rem 0 0.5rem;
        line-height: 1.8; font-size: 0.98rem; color: #1a1a2e;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .exp-card-critical { border-left: 5px solid #e74c3c; background: #fdf5f5; }
    .exp-card-high     { border-left: 5px solid #e67e22; background: #fef8f0; }
    .exp-card-medium   { border-left: 5px solid #f39c12; background: #fefdf0; }
    .exp-card-ok       { border-left: 5px solid #27ae60; background: #f0faf5; }
    .exp-card-info     { border-left: 4px solid #2a5298; background: #f5f7ff; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.7rem 0 1.1rem; }
    .chip {
        display: inline-block; padding: 0.28rem 0.78rem;
        border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    }
    .chip-g { background:#eafaf1; color:#1a7a4a; border:1px solid #abebc6; }
    .chip-r { background:#fdedec; color:#c0392b; border:1px solid #f5b7b1; }
    .chip-a { background:#fef9e7; color:#7a5200; border:1px solid #f9e79f; }
    .info-card {
        background: white; padding: 1.6rem 1.9rem; border-radius: 14px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 3px 14px rgba(0,0,0,0.05); margin: 0.8rem 0;
    }
    .chart-note { font-size: 0.85rem; color: #6b7a96; margin: 0 0 0.6rem; line-height: 1.5; }
    .hdiv { border: none; border-top: 1px solid #e4e8f0; margin: 1.8rem 0; }
    .stButton > button {
        background: linear-gradient(135deg,#1e3c72,#2a5298);
        color: white; font-weight: 700; font-size: 0.97rem;
        padding: 0.7rem 1.8rem; border: none; border-radius: 10px;
        box-shadow: 0 4px 12px rgba(30,60,114,0.28);
        transition: all 0.2s; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(30,60,114,0.38);
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #1e3c72; }
    div[data-testid="stMetricLabel"] {
        font-size: 0.77rem; color: #5a6a85;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; }
    .disclaimer-text {
        font-size: 0.8rem; color: #8a9ab5;
        border-top: 1px solid #e0e4f0; padding-top: 0.7rem; margin-top: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# GOOGLE DRIVE CONFIGURATION
# ============================================================================
# Instructions:
#   1. Open: https://drive.google.com/drive/folders/1VrvhaEWcI2TrU0tWKfgzW5TCVJlL7QAC
#   2. Right-click each file → Share → "Anyone with the link" (Viewer)
#   3. Click "Copy link" — the file ID is the string between /d/ and /view
#      e.g. https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
#   4. Paste each file ID below.
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_FILE_IDS = {
    "final_production_model.pkl": "1tSEO80eYnx70FrPx1rTlYKoHPtjafsUx",
    "scaler.pkl":                 "1CI6vkBkh1oKdg6tGCXT-5fxccbuUcP_l",
    "fairness_report.json":       "1OCbRymlXsx1g6U5w0wqHbffLdWU_iipB",
}
# https://drive.google.com/file/d/1CI6vkBkh1oKdg6tGCXT-5fxccbuUcP_l/view?usp=sharing
# https://drive.google.com/file/d/1tSEO80eYnx70FrPx1rTlYKoHPtjafsUx/view?usp=drive_link

# Files are cached here after first download — no re-download on restart
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache")


def _ensure_gdown():
    """Import gdown, auto-installing it if missing."""
    try:
        import gdown
        return gdown
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import gdown
        return gdown


def _download_from_drive(file_id: str, dest_path: str) -> bool:
    """
    Download one file from Google Drive. Returns True on success.

    Uses gdown with fuzzy=True so it handles both raw IDs and full share URLs.
    The `use_cookies=False` flag avoids the virus-scan confirmation wall that
    Google shows for large files when cookies aren't present.
    """
    try:
        gdown = _ensure_gdown()
        # Build a clean direct-download URL from the bare file ID
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(
            url,
            dest_path,
            quiet=False,
            fuzzy=True,
            use_cookies=False,  # bypass large-file virus-scan confirmation
        )
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception as e:
        st.warning(f"Download failed for `{os.path.basename(dest_path)}`: {e}")
        return False


def _get_model_file(filename: str) -> str | None:
    """
    Return a local path to `filename`, downloading from Google Drive if needed.

    Search order:
      1. Same directory as app.py  — allows manual file placement / local dev
      2. .model_cache/ directory   — previously auto-downloaded files
      3. Download from Google Drive using the configured file ID
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Local override — file placed next to app.py
    local = os.path.join(app_dir, filename)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local

    # 2. Cache hit
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        return cached

    # 3. Download from Drive
    file_id = GDRIVE_FILE_IDS.get(filename, "")
    if not file_id or file_id.startswith("REPLACE_WITH"):
        return None  # ID not yet configured

    st.info(f"Downloading `{filename}` from Google Drive… (first run only)")
    success = _download_from_drive(file_id, cached)
    return cached if success else None


# ============================================================================
# COMPANIES DATABASE
# ============================================================================

MENA_COMPANIES = {
    "Saudi Arabia": [
        {"name": "Al Rajhi Bank",                    "ticker": "1120.SR", "sector": "Banking"},
        {"name": "Saudi National Bank",               "ticker": "1180.SR", "sector": "Banking"},
        {"name": "Riyad Bank",                        "ticker": "1010.SR", "sector": "Banking"},
        {"name": "Saudi Awwal Bank",                  "ticker": "1140.SR", "sector": "Banking"},
        {"name": "Alinma Bank",                       "ticker": "1150.SR", "sector": "Banking"},
        {"name": "Bank AlJazira",                     "ticker": "1020.SR", "sector": "Banking"},
        {"name": "Arab National Bank",                "ticker": "1080.SR", "sector": "Banking"},
        {"name": "Banque Saudi Fransi",               "ticker": "1050.SR", "sector": "Banking"},
        {"name": "Saudi Re",                          "ticker": "8200.SR", "sector": "Insurance"},
        {"name": "Tawuniya",                          "ticker": "8010.SR", "sector": "Insurance"},
        {"name": "Bupa Arabia",                       "ticker": "8210.SR", "sector": "Insurance"},
        {"name": "Medgulf",                           "ticker": "8030.SR", "sector": "Insurance"},
        {"name": "Walaa Insurance",                   "ticker": "8250.SR", "sector": "Insurance"},
        {"name": "Saudi Aramco",                      "ticker": "2222.SR", "sector": "Energy"},
        {"name": "SABIC",                             "ticker": "2010.SR", "sector": "Petrochemicals"},
        {"name": "Saudi Electricity",                 "ticker": "5110.SR", "sector": "Utilities"},
        {"name": "ACWA Power",                        "ticker": "2082.SR", "sector": "Utilities"},
        {"name": "Saudi Kayan",                       "ticker": "2350.SR", "sector": "Petrochemicals"},
        {"name": "Yanbu Petrochemical",               "ticker": "2290.SR", "sector": "Petrochemicals"},
        {"name": "Advanced Petrochemical",            "ticker": "2330.SR", "sector": "Petrochemicals"},
        {"name": "Petro Rabigh",                      "ticker": "2380.SR", "sector": "Petrochemicals"},
        {"name": "SIIG",                              "ticker": "2250.SR", "sector": "Petrochemicals"},
        {"name": "Tasnee",                            "ticker": "2060.SR", "sector": "Petrochemicals"},
        {"name": "Saudi International Petrochemical", "ticker": "2310.SR", "sector": "Petrochemicals"},
        {"name": "Nama Chemicals",                    "ticker": "2210.SR", "sector": "Petrochemicals"},
        {"name": "Methanol Chemicals",                "ticker": "2001.SR", "sector": "Petrochemicals"},
        {"name": "Saudi Aramco Base Oil",             "ticker": "2370.SR", "sector": "Petrochemicals"},
        {"name": "Maaden",                            "ticker": "1211.SR", "sector": "Mining"},
        {"name": "National Industrialization",        "ticker": "2060.SR", "sector": "Industrial"},
        {"name": "Saudi Steel Pipe",                  "ticker": "1320.SR", "sector": "Industrial"},
        {"name": "Filling & Packing",                 "ticker": "2100.SR", "sector": "Industrial"},
        {"name": "Saudi Paper",                       "ticker": "2300.SR", "sector": "Industrial"},
        {"name": "Saudi Telecom (STC)",               "ticker": "7010.SR", "sector": "Telecom"},
        {"name": "Mobily",                            "ticker": "7020.SR", "sector": "Telecom"},
        {"name": "Zain Saudi",                        "ticker": "7030.SR", "sector": "Telecom"},
        {"name": "Atheeb Telecom",                    "ticker": "7040.SR", "sector": "Telecom"},
        {"name": "Almarai",                           "ticker": "2280.SR", "sector": "Food & Beverages"},
        {"name": "Savola Group",                      "ticker": "2050.SR", "sector": "Food & Beverages"},
        {"name": "Herfy Food",                        "ticker": "6002.SR", "sector": "Food Services"},
        {"name": "Nadec",                             "ticker": "6010.SR", "sector": "Food & Beverages"},
        {"name": "Anaam Holding",                     "ticker": "4061.SR", "sector": "Food & Beverages"},
        {"name": "Tanmiah Food",                      "ticker": "6060.SR", "sector": "Food & Beverages"},
        {"name": "Halwani Bros",                      "ticker": "6001.SR", "sector": "Food & Beverages"},
        {"name": "Jarir Marketing",                   "ticker": "4190.SR", "sector": "Retail"},
        {"name": "Extra Stores",                      "ticker": "4003.SR", "sector": "Retail"},
        {"name": "Fawaz Alhokair",                    "ticker": "4240.SR", "sector": "Retail"},
        {"name": "Al Othaim Markets",                 "ticker": "4001.SR", "sector": "Retail"},
        {"name": "Arabian Centres",                   "ticker": "4321.SR", "sector": "Retail"},
        {"name": "Bindawood Holding",                 "ticker": "4161.SR", "sector": "Retail"},
        {"name": "Mouwasat Medical",                  "ticker": "4002.SR", "sector": "Healthcare"},
        {"name": "Dallah Healthcare",                 "ticker": "4004.SR", "sector": "Healthcare"},
        {"name": "Dr Sulaiman Al Habib",              "ticker": "4013.SR", "sector": "Healthcare"},
        {"name": "Nahdi Medical",                     "ticker": "4164.SR", "sector": "Healthcare"},
        {"name": "Dar Al Arkan",                      "ticker": "4320.SR", "sector": "Real Estate"},
        {"name": "Emaar Economic City",               "ticker": "4220.SR", "sector": "Real Estate"},
        {"name": "Jabal Omar",                        "ticker": "4250.SR", "sector": "Real Estate"},
        {"name": "Retal Urban",                       "ticker": "4322.SR", "sector": "Real Estate"},
        {"name": "Taiba Holding",                     "ticker": "4090.SR", "sector": "Real Estate"},
        {"name": "Saudi Real Estate",                 "ticker": "4020.SR", "sector": "Real Estate"},
        {"name": "Saudi Cement",                      "ticker": "3020.SR", "sector": "Cement"},
        {"name": "Yamama Cement",                     "ticker": "3030.SR", "sector": "Cement"},
        {"name": "Qassim Cement",                     "ticker": "3040.SR", "sector": "Cement"},
        {"name": "Yanbu Cement",                      "ticker": "3060.SR", "sector": "Cement"},
        {"name": "Saudi Oger",                        "ticker": "4146.SR", "sector": "Construction"},
        {"name": "Astra Industrial",                  "ticker": "1212.SR", "sector": "Construction"},
        {"name": "Bahri",                             "ticker": "4030.SR", "sector": "Transportation"},
        {"name": "Saudi Airlines Catering",           "ticker": "6004.SR", "sector": "Services"},
        {"name": "SAPTCO",                            "ticker": "4031.SR", "sector": "Transportation"},
    ],
    "Qatar": [
        {"name": "Qatar National Bank",              "ticker": "QNBK.QA", "sector": "Banking"},
        {"name": "Qatar Islamic Bank",               "ticker": "QIBK.QA", "sector": "Banking"},
        {"name": "Commercial Bank Qatar",            "ticker": "CBQK.QA", "sector": "Banking"},
        {"name": "Doha Bank",                        "ticker": "DHBK.QA", "sector": "Banking"},
        {"name": "Masraf Al Rayan",                  "ticker": "MARK.QA", "sector": "Banking"},
        {"name": "Qatar International Islamic Bank", "ticker": "QIIK.QA", "sector": "Banking"},
        {"name": "Ahli Bank",                        "ticker": "ABQK.QA", "sector": "Banking"},
        {"name": "Qatar Insurance",                  "ticker": "QATI.QA", "sector": "Insurance"},
        {"name": "Qatar General Insurance",          "ticker": "QGRI.QA", "sector": "Insurance"},
        {"name": "Industries Qatar",                 "ticker": "IQCD.QA", "sector": "Industrial"},
        {"name": "Qatar Fuel (Woqod)",               "ticker": "QFLS.QA", "sector": "Energy"},
        {"name": "Nakilat",                          "ticker": "QGTS.QA", "sector": "Transportation"},
        {"name": "Qatar Electricity & Water",        "ticker": "QEWS.QA", "sector": "Utilities"},
        {"name": "Qatar Aluminum",                   "ticker": "QAMC.QA", "sector": "Industrial"},
        {"name": "Gulf International Services",      "ticker": "GISS.QA", "sector": "Industrial"},
        {"name": "Ooredoo",                          "ticker": "ORDS.QA", "sector": "Telecom"},
        {"name": "Vodafone Qatar",                   "ticker": "VFQS.QA", "sector": "Telecom"},
        {"name": "Barwa Real Estate",                "ticker": "BRES.QA", "sector": "Real Estate"},
        {"name": "Ezdan Holding",                    "ticker": "ERES.QA", "sector": "Real Estate"},
        {"name": "UDC",                              "ticker": "UDCD.QA", "sector": "Real Estate"},
        {"name": "Qatari Investors Group",           "ticker": "QIGD.QA", "sector": "Real Estate"},
        {"name": "Widam Food",                       "ticker": "WDAM.QA", "sector": "Food & Beverages"},
        {"name": "Salam International",              "ticker": "SIIS.QA", "sector": "Services"},
    ],
    "Kuwait": [
        {"name": "National Bank of Kuwait",         "ticker": "NBK.KW",     "sector": "Banking"},
        {"name": "Boubyan Bank",                    "ticker": "BOUBYAN.KW", "sector": "Banking"},
        {"name": "Gulf Bank",                       "ticker": "GBK.KW",     "sector": "Banking"},
        {"name": "Burgan Bank",                     "ticker": "BURG.KW",    "sector": "Banking"},
        {"name": "Commercial Bank Kuwait",          "ticker": "CBK.KW",     "sector": "Banking"},
        {"name": "Kuwait International Bank",       "ticker": "KIB.KW",     "sector": "Banking"},
        {"name": "Kuwait Projects Company (KIPCO)", "ticker": "KPROJ.KW",   "sector": "Investment"},
        {"name": "Zain Kuwait",                     "ticker": "ZAIN.KW",    "sector": "Telecom"},
        {"name": "Ooredoo Kuwait",                  "ticker": "OOREDOO.KW", "sector": "Telecom"},
        {"name": "STC Kuwait",                      "ticker": "STC.KW",     "sector": "Telecom"},
        {"name": "Americana Group",                 "ticker": "FOOD.KW",    "sector": "Food & Beverages"},
        {"name": "National Industries Group",       "ticker": "NIND.KW",    "sector": "Industrial"},
        {"name": "Kuwait Cement",                   "ticker": "KCEM.KW",    "sector": "Construction Materials"},
        {"name": "Kuwait Real Estate",              "ticker": "KRE.KW",     "sector": "Real Estate"},
        {"name": "Sokouk Holding",                  "ticker": "SOKOUK.KW",  "sector": "Real Estate"},
    ],
    "Egypt": [
        {"name": "Commercial International Bank",   "ticker": "COMI.CA", "sector": "Banking"},
        {"name": "Abu Dhabi Islamic Bank Egypt",    "ticker": "ADIB.CA", "sector": "Banking"},
        {"name": "Credit Agricole Egypt",           "ticker": "CIEB.CA", "sector": "Banking"},
        {"name": "Egyptian Gulf Bank",              "ticker": "EGBE.CA", "sector": "Banking"},
        {"name": "Telecom Egypt",                   "ticker": "ETEL.CA", "sector": "Telecom"},
        {"name": "Eastern Company",                 "ticker": "EAST.CA", "sector": "Consumer Goods"},
        {"name": "Juhayna Food",                    "ticker": "JUFO.CA", "sector": "Food & Beverages"},
        {"name": "Edita Food",                      "ticker": "EFID.CA", "sector": "Food & Beverages"},
        {"name": "Arabian Food Industries (Domty)", "ticker": "DOMT.CA", "sector": "Food & Beverages"},
        {"name": "Cairo Poultry",                   "ticker": "POUL.CA", "sector": "Food & Beverages"},
        {"name": "Ismailia Misr Poultry",           "ticker": "ISPH.CA", "sector": "Food & Beverages"},
        {"name": "Oriental Weavers",                "ticker": "ORWE.CA", "sector": "Manufacturing"},
        {"name": "Egyptian Iron & Steel",           "ticker": "ESRS.CA", "sector": "Manufacturing"},
        {"name": "Talaat Moustafa",                 "ticker": "TMGH.CA", "sector": "Real Estate"},
        {"name": "Palm Hills",                      "ticker": "PHDC.CA", "sector": "Real Estate"},
        {"name": "Sixth of October",                "ticker": "OCDI.CA", "sector": "Real Estate"},
        {"name": "Orascom Development",             "ticker": "ORAS.CA", "sector": "Real Estate"},
        {"name": "Suez Cement",                     "ticker": "SUCE.CA", "sector": "Cement"},
        {"name": "Arabian Cement",                  "ticker": "ARCC.CA", "sector": "Cement"},
        {"name": "Sinai Cement",                    "ticker": "SCEM.CA", "sector": "Cement"},
        {"name": "Alexandria Mineral Oils (AMOC)",  "ticker": "AMOC.CA", "sector": "Petrochemicals"},
        {"name": "Sidi Kerir Petrochemicals",       "ticker": "SKPC.CA", "sector": "Petrochemicals"},
        {"name": "Fawry",                           "ticker": "FWRY.CA", "sector": "Technology"},
        {"name": "e-finance",                       "ticker": "EFIH.CA", "sector": "Technology"},
        {"name": "Raya Contact Center",             "ticker": "RAYA.CA", "sector": "Technology"},
    ],
    "Bahrain": [
        {"name": "National Bank of Bahrain", "ticker": "NBB.BH",     "sector": "Banking"},
        {"name": "BBK",                      "ticker": "BBK.BH",     "sector": "Banking"},
        {"name": "Al Salam Bank",            "ticker": "SALAM.BH",   "sector": "Banking"},
        {"name": "Ithmaar Bank",             "ticker": "ITHMR.BH",   "sector": "Banking"},
        {"name": "Bahrain Islamic Bank",     "ticker": "BISB.BH",    "sector": "Banking"},
        {"name": "Investcorp",               "ticker": "INVCORP.BH", "sector": "Investment"},
        {"name": "GFH Financial",            "ticker": "GFH.BH",     "sector": "Investment"},
        {"name": "Aluminium Bahrain (Alba)", "ticker": "ALBH.BH",    "sector": "Manufacturing"},
        {"name": "Zain Bahrain",             "ticker": "ZAINBH.BH",  "sector": "Telecom"},
        {"name": "Seef Properties",          "ticker": "SEEF.BH",    "sector": "Real Estate"},
        {"name": "Bahrain Duty Free",        "ticker": "DUTYF.BH",   "sector": "Retail"},
    ],
}


# ============================================================================
# FEATURES — single source of truth (matches train_all_models.py exactly)
# ============================================================================

SELECTED_FEATURES = [
    "Market_Cap_USD", "Low", "Volume", "Daily_Return_%", "Price_Range_%",
    "ROA_%", "Equity_Ratio", "Asset_Turnover", "OCF_to_Debt", "Altman_Z_Score",
    "Death_Cross", "True_Range", "RSI_14", "US_10Y",
    "Oil_Volatility_20D", "Oil_Below_60", "Oil_Below_40", "Brent_Change_%",
    "VIX_Change_%", "Very_High_VIX", "Strong_Dollar",
    "SAR_USD", "KWD_USD", "QAR_USD_Volatility_20D", "BHD_USD_Volatility_20D",
    "Gulf_Crisis_End", "Is_Month_End",
    "Very_High_Governance_Risk", "Has_Controversy", "Poor_Governance",
    "Operating Cf_M", "Free Cf_M", "Debt_to_Equity", "ROE_%", "Net_Profit_Margin_%",
    "Volatility_20", "ROC_20",
    "Egypt_FX_Crisis", "EGP_USD_Change_%", "Pandemic_Recession",
    "Environment_Score", "Social_Score",
    "Young_Company", "Low_Institutional_Ownership",
    "Month_x",
]

FEATURE_LABELS = {
    "Market_Cap_USD":              "Company Size (Market Value)",
    "Low":                         "Daily Low Price",
    "Volume":                      "Trading Activity (Volume)",
    "Daily_Return_%":              "Daily Price Change",
    "Price_Range_%":               "Daily Price Spread",
    "ROA_%":                       "Return on Assets",
    "Equity_Ratio":                "Equity Ratio",
    "Asset_Turnover":              "Asset Efficiency",
    "OCF_to_Debt":                 "Cash Flow vs Debt Coverage",
    "Altman_Z_Score":              "Financial Health Score",
    "Death_Cross":                 "Bearish Price Signal",
    "True_Range":                  "Daily Price Volatility",
    "RSI_14":                      "Price Momentum",
    "US_10Y":                      "US Interest Rate",
    "Oil_Volatility_20D":          "Oil Price Stability",
    "Oil_Below_60":                "Low Oil Price Indicator",
    "Oil_Below_40":                "Very Low Oil Price Indicator",
    "Brent_Change_%":              "Oil Price Change",
    "VIX_Change_%":                "Market Uncertainty Change",
    "Very_High_VIX":               "Extreme Market Uncertainty",
    "Strong_Dollar":               "Strong US Dollar",
    "SAR_USD":                     "Saudi Riyal Rate",
    "KWD_USD":                     "Kuwaiti Dinar Rate",
    "QAR_USD_Volatility_20D":      "Qatar Riyal Stability",
    "BHD_USD_Volatility_20D":      "Bahraini Dinar Stability",
    "Gulf_Crisis_End":             "Post-Gulf Crisis Period",
    "Is_Month_End":                "Month-End Period",
    "Very_High_Governance_Risk":   "High Governance Risk",
    "Has_Controversy":             "Company Controversy",
    "Poor_Governance":             "Governance Concerns",
    "Operating Cf_M":              "Operating Cash Flow",
    "Free Cf_M":                   "Free Cash Flow",
    "Debt_to_Equity":              "Debt Relative to Equity",
    "ROE_%":                       "Return on Equity",
    "Net_Profit_Margin_%":         "Net Profit Margin",
    "Volatility_20":               "Price Stability (20-Day)",
    "ROC_20":                      "Price Trend (20-Day)",
    "Egypt_FX_Crisis":             "Egypt Currency Pressure",
    "EGP_USD_Change_%":            "Egyptian Pound Change",
    "Pandemic_Recession":          "Pandemic Period",
    "Environment_Score":           "Environmental Score",
    "Social_Score":                "Social Score",
    "Young_Company":               "Early-Stage Company",
    "Low_Institutional_Ownership": "Low Institutional Ownership",
    "Month_x":                     "Month of Year",
}

FEATURE_CATEGORIES = {
    "Financial Health": [
        "ROA_%", "Equity_Ratio", "Asset_Turnover", "OCF_to_Debt", "Altman_Z_Score",
        "Operating Cf_M", "Free Cf_M", "Debt_to_Equity", "ROE_%", "Net_Profit_Margin_%",
    ],
    "Market & Price": [
        "Market_Cap_USD", "Low", "Volume", "Daily_Return_%", "Price_Range_%",
        "True_Range", "Volatility_20", "ROC_20", "Death_Cross", "RSI_14",
    ],
    "Oil & Global Macro": [
        "Oil_Volatility_20D", "Oil_Below_60", "Oil_Below_40",
        "Brent_Change_%", "US_10Y", "VIX_Change_%", "Very_High_VIX", "Strong_Dollar",
    ],
    "Regional Factors": [
        "SAR_USD", "KWD_USD", "QAR_USD_Volatility_20D", "BHD_USD_Volatility_20D",
        "Gulf_Crisis_End", "Egypt_FX_Crisis", "EGP_USD_Change_%", "Pandemic_Recession",
    ],
    "Governance & ESG": [
        "Very_High_Governance_Risk", "Has_Controversy", "Poor_Governance",
        "Environment_Score", "Social_Score", "Young_Company", "Low_Institutional_Ownership",
    ],
    "Timing": ["Is_Month_End", "Month_x"],
}


# ============================================================================
# MODEL LOADING — Google Drive download with local cache
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_resources():
    """
    Load model + scaler. No metadata file required.

    The pipeline (complete_modeling_pipeline.py) saves the model as a raw estimator:
        pickle.dump(best_model, f)
    We also handle the legacy dict-wrapped format just in case.
    """
    # ── Model ─────────────────────────────────────────────────────────────
    model_path = _get_model_file("final_production_model.pkl")
    if model_path is None:
        st.error(
            "**`final_production_model.pkl` could not be found or downloaded.**\n\n"
            "Please add the Google Drive file ID to `GDRIVE_FILE_IDS['final_production_model.pkl']`"
            " in the script, then restart the app.\n\n"
            "Alternatively, drop the file next to `app.py` for a local override."
        )
        st.stop()

    try:
        with open(model_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        st.error(f"Could not unpickle `final_production_model.pkl`.\n\nDetails: {e}")
        st.stop()

    # Raw estimator (pipeline output) or legacy dict-wrapped format
    if hasattr(raw, "predict_proba"):
        model = raw
    elif isinstance(raw, dict):
        model = (
            raw.get("final_model")
            or raw.get("best_model")
            or next((v for v in raw.values() if hasattr(v, "predict_proba")), None)
        )
        if model is None:
            st.error(
                f"The model pickle is a dict but no estimator was found inside it.\n\n"
                f"Keys: `{list(raw.keys())}`"
            )
            st.stop()
    else:
        st.error(f"Unexpected model type: `{type(raw)}`. Expected a fitted scikit-learn estimator.")
        st.stop()

    # ── Scaler ────────────────────────────────────────────────────────────
    scaler_path = _get_model_file("scaler.pkl")
    if scaler_path is None:
        st.error(
            "**`scaler.pkl` could not be found or downloaded.**\n\n"
            "Please add the Google Drive file ID to `GDRIVE_FILE_IDS['scaler.pkl']`"
            " in the script, then restart the app.\n\n"
            "Alternatively, drop the file next to `app.py` for a local override."
        )
        st.stop()

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Could not unpickle `scaler.pkl`.\n\nDetails: {e}")
        st.stop()

    # ── SHAP explainer (optional) ─────────────────────────────────────────
    explainer = None
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        pass  # SHAP charts won't render, but the rest of the app still works

    return model, scaler, explainer


@st.cache_data(show_spinner=False)
def load_fairness_report():
    path = _get_model_file("fairness_report.json")
    if path is None:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================================
# DATA FETCHING & METRICS
# ============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_company_data(ticker, period="2y"):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)
        if len(hist) == 0:
            return None
        return {
            "history":       hist,
            "financials":    stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cash_flow":     stock.cashflow,
            "info":          stock.info,
        }
    except Exception:
        return None


def calculate_all_metrics(data):
    try:
        bs, fin, cf, info, hist = (
            data["balance_sheet"], data["financials"],
            data["cash_flow"],     data["info"], data["history"],
        )

        def gv(series, key, default=0):
            try:
                return float(series[key]) if key in series.index else default
            except Exception:
                return default

        lbs  = bs.iloc[:,0]  if len(bs.columns)  > 0 else pd.Series(dtype=float)
        lfin = fin.iloc[:,0] if len(fin.columns) > 0 else pd.Series(dtype=float)
        lcf  = cf.iloc[:,0]  if len(cf.columns)  > 0 else pd.Series(dtype=float)

        assets   = gv(lbs,  "Total Assets",             info.get("totalAssets", 0))
        equity   = gv(lbs,  "Total Stockholder Equity", info.get("totalStockholderEquity", 0))
        curr_a   = gv(lbs,  "Current Assets",     0)
        curr_l   = gv(lbs,  "Current Liabilities", 0)
        retained = gv(lbs,  "Retained Earnings",   0)
        debt     = gv(lbs,  "Total Debt", 0) or (
                   gv(lbs,  "Long Term Debt", 0) + gv(lbs, "Short Long Term Debt", 0))
        net_inc  = gv(lfin, "Net Income",    0)
        revenue  = gv(lfin, "Total Revenue", 0)
        ebit     = gv(lfin, "EBIT",          0)
        ocf      = gv(lcf,  "Total Cash From Operating Activities", 0)
        capex    = gv(lcf,  "Capital Expenditures", 0)
        fcf      = gv(lcf,  "Free Cash Flow", 0) or (ocf - abs(capex) if ocf else 0)

        mkt_cap = info.get("marketCap", 0) or 0
        rets    = hist["Close"].pct_change().dropna()
        lat_ret = float(rets.iloc[-1] * 100) if len(rets) > 0 else 0
        vol_20  = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets) >= 20 else 0

        pr_pct = 0.0
        if len(hist) > 0 and hist["Open"].iloc[-1] > 0:
            pr_pct = (hist["High"].iloc[-1] - hist["Low"].iloc[-1]) / hist["Open"].iloc[-1] * 100

        tr = 0.0
        if len(hist) > 1:
            tr = float(pd.concat([
                hist["High"] - hist["Low"],
                (hist["High"] - hist["Close"].shift(1)).abs(),
                (hist["Low"]  - hist["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1).iloc[-1])

        ma50  = hist["Close"].rolling(50).mean()
        ma200 = hist["Close"].rolling(200).mean()
        dc = int(
            len(ma50) > 0 and len(ma200) > 0 and
            not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]) and
            ma50.iloc[-1] < ma200.iloc[-1]
        )

        delta = hist["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_s = 100 - (100 / (1 + rs))
        rsi14 = float(rsi_s.iloc[-1]) if len(rsi_s) > 0 and not pd.isna(rsi_s.iloc[-1]) else 50

        roc20 = 0.0
        if len(hist) > 20 and hist["Close"].iloc[-21] != 0:
            roc20 = (hist["Close"].iloc[-1] - hist["Close"].iloc[-21]) / hist["Close"].iloc[-21] * 100

        eq_r = equity  / assets  if assets  > 0 else 0
        at   = revenue / assets  if assets  > 0 else 0
        o2d  = ocf     / debt    if debt    > 0 else 0
        d2e  = debt    / equity  if equity  > 0 else 0
        roe  = net_inc / equity  * 100 if equity  > 0 else 0
        roa  = net_inc / assets  * 100 if assets  > 0 else 0
        npm  = net_inc / revenue * 100 if revenue > 0 else 0

        wc = curr_a - curr_l
        z  = (
            1.2*wc/assets + 1.4*retained/assets + 3.3*ebit/assets +
            (0.6*equity/debt if debt > 0 else 0) + 1.0*revenue/assets
        ) if assets > 0 else 0.0

        today = datetime.now()

        return {
            "Market_Cap_USD": mkt_cap,
            "Low":            float(hist["Low"].iloc[-1]) if len(hist) > 0 else 0,
            "Volume":         float(hist["Volume"].mean()) if len(hist) > 0 else 0,
            "Daily_Return_%": lat_ret if not np.isnan(lat_ret) else 0,
            "Price_Range_%":  pr_pct,
            "ROA_%":          roa,
            "Equity_Ratio":   eq_r,
            "Asset_Turnover": at,
            "OCF_to_Debt":    o2d,
            "Altman_Z_Score": z,
            "Death_Cross":    dc,
            "True_Range":     tr,
            "RSI_14":         rsi14,
            "US_10Y": 0, "Oil_Volatility_20D": 0, "Oil_Below_60": 0, "Oil_Below_40": 0,
            "Brent_Change_%": 0, "VIX_Change_%": 0, "Very_High_VIX": 0, "Strong_Dollar": 0,
            "SAR_USD": 0, "KWD_USD": 0, "QAR_USD_Volatility_20D": 0, "BHD_USD_Volatility_20D": 0,
            "Gulf_Crisis_End": 0,
            "Is_Month_End":   int(today.day >= 25),
            "Very_High_Governance_Risk": 0, "Has_Controversy": 0, "Poor_Governance": 0,
            "Operating Cf_M": ocf / 1e6,
            "Free Cf_M":      fcf / 1e6,
            "Debt_to_Equity": d2e,
            "ROE_%":          roe,
            "Net_Profit_Margin_%": npm,
            "Volatility_20":  float(vol_20) if not np.isnan(vol_20) else 0,
            "ROC_20":         roc20,
            "Egypt_FX_Crisis": 0, "EGP_USD_Change_%": 0, "Pandemic_Recession": 0,
            "Environment_Score": 50, "Social_Score": 50,
            "Young_Company": 0, "Low_Institutional_Ownership": 0,
            "Month_x":        today.month,
        }
    except Exception:
        return None


def make_prediction(model, scaler, metrics):
    X  = pd.DataFrame([[metrics.get(f, 0) for f in SELECTED_FEATURES]], columns=SELECTED_FEATURES)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0, 1]
    risk = (
        "Critical" if prob >= 0.7 else
        "High"     if prob >= 0.5 else
        "Medium"   if prob >= 0.3 else
        "Low"
    )
    return pred, prob, risk


# ============================================================================
# CHARTS
# ============================================================================

def create_price_chart(hist, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"], name="Closing Price",
        line=dict(color="#1e3c72", width=2.5),
        fill="tozeroy", fillcolor="rgba(30,60,114,0.07)"
    ))
    ma50 = hist["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=hist.index, y=ma50, name="50-Day Average",
        line=dict(color="#e67e22", width=1.8, dash="dash")
    ))
    if len(hist) >= 200:
        ma200 = hist["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma200, name="200-Day Average",
            line=dict(color="#c0392b", width=1.8, dash="dot")
        ))
    fig.update_layout(
        title=dict(text=f"{name}  —  Price History", font=dict(size=14, color="#1e3c72")),
        height=400, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=46, b=10),
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def create_gauge(value, title):
    colour = "#27ae60" if value < 33 else "#f39c12" if value < 66 else "#c0392b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 24, "color": "#1e3c72"}},
        title={"text": title, "font": {"size": 12, "color": "#5a6a85"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar":  {"color": colour, "thickness": 0.28},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "#eafaf1"},
                {"range": [33, 66], "color": "#fef9e7"},
                {"range": [66, 100], "color": "#fdedec"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=18, r=18, t=36, b=8), paper_bgcolor="white")
    return fig


def _labels(features):
    return [FEATURE_LABELS.get(f, f) for f in features]


def shap_category_chart(explainer, X_scaled, name):
    vals     = explainer.shap_values(X_scaled)[0]
    feat_idx = {f: i for i, f in enumerate(SELECTED_FEATURES)}
    cat_vals = {
        cat: float(np.sum(vals[[feat_idx[f] for f in flist if f in feat_idx]]))
        for cat, flist in FEATURE_CATEGORIES.items()
    }
    cats    = list(cat_vals.keys())
    totals  = list(cat_vals.values())
    colours = ["#c0392b" if v > 0 else "#27ae60" for v in totals]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(cats, totals, color=colours, width=0.52, zorder=3)
    ax.axhline(0, color="#888", linewidth=0.8, zorder=2)
    for b, v in zip(bars, totals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (max(abs(t) for t in totals) * 0.03) * (1 if v >= 0 else -1),
            f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
            fontsize=9.5, color="#1e3c72", fontweight="600"
        )
    ax.set_ylabel("Impact on risk score", fontsize=11, color="#5a6a85")
    ax.set_title(f"Risk Contribution by Category — {name}", fontsize=13,
                 fontweight="bold", color="#1e3c72", pad=14)
    ax.tick_params(axis="x", labelsize=10, rotation=12)
    ax.tick_params(axis="y", labelsize=9, colors="#888")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ddd"); ax.spines["left"].set_color("#ddd")
    ax.set_facecolor("white"); ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Raises distress risk"),
        mpatches.Patch(color="#27ae60", label="Lowers distress risk"),
    ], fontsize=9.5, framealpha=0, loc="upper right")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    return fig


def shap_direction_chart(explainer, X_scaled, name):
    vals   = explainer.shap_values(X_scaled)[0]
    labels = _labels(SELECTED_FEATURES)
    df = pd.DataFrame({"Factor": labels, "Impact": vals})
    df = df.reindex(df["Impact"].abs().sort_values(ascending=False).index).head(14)
    df = df.sort_values("Impact")
    colours = ["#c0392b" if v > 0 else "#27ae60" for v in df["Impact"]]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(df["Factor"], df["Impact"], color=colours, height=0.62)
    ax.axvline(0, color="#555", linewidth=0.9)
    ax.set_xlabel("Effect on risk score  (red = increases risk, green = reduces risk)",
                  fontsize=10.5, color="#5a6a85")
    ax.set_title(f"Factors Raising vs Reducing Risk — {name}", fontsize=13,
                 fontweight="bold", color="#1e3c72", pad=14)
    ax.tick_params(axis="y", labelsize=9.5); ax.tick_params(axis="x", labelsize=9, colors="#888")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ddd"); ax.spines["left"].set_visible(False)
    ax.set_facecolor("white"); ax.xaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Increases distress risk"),
        mpatches.Patch(color="#27ae60", label="Reduces distress risk"),
    ], fontsize=9.5, framealpha=0)
    fig.patch.set_facecolor("white"); plt.tight_layout()
    return fig


def shap_importance_chart(explainer, X_scaled, name):
    vals   = np.abs(explainer.shap_values(X_scaled)[0])
    labels = _labels(SELECTED_FEATURES)
    df = (pd.DataFrame({"Factor": labels, "Influence": vals})
            .sort_values("Influence", ascending=True).tail(12))
    med = df["Influence"].median()
    colours = ["#1e3c72" if v >= med else "#90aad4" for v in df["Influence"]]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.barh(df["Factor"], df["Influence"], color=colours, height=0.62)
    for b, v in zip(bars, df["Influence"]):
        ax.text(v + df["Influence"].max() * 0.012,
                b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9, color="#1e3c72")
    ax.set_xlabel("Overall influence on the prediction (higher = more important)",
                  fontsize=10.5, color="#5a6a85")
    ax.set_title(f"Most Influential Factors — {name}", fontsize=13,
                 fontweight="bold", color="#1e3c72", pad=14)
    ax.tick_params(axis="y", labelsize=9.5); ax.tick_params(axis="x", labelsize=9, colors="#888")
    for spine in ["top", "right", "left"]: ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ddd"); ax.set_facecolor("white")
    ax.xaxis.grid(True, color="#f0f0f0", zorder=0)
    fig.patch.set_facecolor("white"); plt.tight_layout()
    return fig


def shap_waterfall_chart(explainer, X_scaled, name):
    exp = shap.Explanation(
        values=explainer.shap_values(X_scaled)[0],
        base_values=explainer.expected_value,
        data=X_scaled[0],
        feature_names=_labels(SELECTED_FEATURES),
    )
    fig, ax = plt.subplots(figsize=(10, 6.8))
    shap.waterfall_plot(exp, show=False, max_display=15)
    ax.set_title(f"Step-by-Step Breakdown — {name}", fontsize=13,
                 fontweight="bold", color="#1e3c72", pad=14)
    plt.tight_layout()
    return fig


def create_fairness_chart(results_df, group_col, group_label):
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"Accuracy (F1-Score) by {group_label}", f"Reliability (AUC) by {group_label}",
        "Precision vs Recall", f"Number of Cases by {group_label}",
    ))
    mean_f1  = results_df["F1_Score"].mean()
    mean_auc = results_df["AUC_ROC"].mean()
    fig.add_trace(go.Bar(x=results_df["F1_Score"],  y=results_df[group_col], orientation="h",
                         marker_color="#2a5298", showlegend=False), row=1, col=1)
    fig.add_vline(x=mean_f1,  line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"Avg {mean_f1:.3f}",  row=1, col=1)
    fig.add_trace(go.Bar(x=results_df["AUC_ROC"], y=results_df[group_col], orientation="h",
                         marker_color="#27ae60", showlegend=False), row=1, col=2)
    fig.add_vline(x=mean_auc, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"Avg {mean_auc:.3f}", row=1, col=2)
    fig.add_trace(go.Scatter(
        x=results_df["Precision"], y=results_df["Recall"],
        mode="markers+text", text=results_df[group_col], textposition="top center",
        marker=dict(size=results_df["Samples"] / results_df["Samples"].max() * 28 + 8,
                    color="#9b59b6", opacity=0.65),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Bar(x=results_df["Samples"], y=results_df[group_col], orientation="h",
                         marker_color="#e67e22", showlegend=False), row=2, col=2)
    fig.update_layout(height=600, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color="#1e3c72", size=11))
    return fig


# ============================================================================
# EXPLANATION RENDERERS
# ============================================================================

def render_shap_narrative(shap_vals, pred, prob):
    labels  = _labels(SELECTED_FEATURES)
    df      = pd.DataFrame({"Factor": labels, "SHAP": shap_vals})
    df_abs  = df.reindex(df["SHAP"].abs().sort_values(ascending=False).index)
    top_pos = df_abs[df_abs["SHAP"] > 0].head(3)["Factor"].tolist()
    top_neg = df_abs[df_abs["SHAP"] < 0].head(3)["Factor"].tolist()
    top_all = df_abs.head(1)["Factor"].tolist()

    pos_text = ", ".join(f"**{f}**" for f in top_pos) if top_pos else "no strong positive contributors"
    neg_text = ", ".join(f"**{f}**" for f in top_neg) if top_neg else "no strong negative contributors"
    top_text = f"**{top_all[0]}**" if top_all else "an unknown factor"

    st.markdown(
        '<div class="exp-card exp-card-info">'
        '<p style="font-size:1.5rem;font-weight:800;color:#1f2937;margin:0;">'
        'What the AI analysis found</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"The model reached its distress classification primarily because of {top_text}, "
                    f"which had the single largest influence on the outcome.")
        st.markdown(f"The factors most strongly **raising** the distress signal are: {pos_text}. "
                    f"These indicate areas of financial or market stress weighted heavily by the model.")
        st.markdown(f"On the positive side, the following factors are **working in the company's favour** "
                    f"and reducing the risk score: {neg_text}. Without these, the assessed risk would be even higher.")
    else:
        st.markdown(f"The model classified this company as financially healthy, with {top_text} "
                    f"being the most influential factor in that conclusion.")
        st.markdown(f"The factors most strongly **supporting the healthy classification** are: {neg_text}. "
                    f"These reflect positive financial and market signals the model found reassuring.")
        st.markdown(f"Some factors do introduce minor risk into the picture: {pos_text}. "
                    f"However, these are outweighed by the positive signals and do not change the overall outcome.")

    st.markdown(
        '<p class="disclaimer-text">The charts below show this in visual form — from a broad '
        'category view down to individual factors.</p>',
        unsafe_allow_html=True,
    )


def render_explanation(name, period_label, pred, prob, risk, metrics):
    pct  = f"{prob:.1%}"
    z    = metrics.get("Altman_Z_Score", 0)
    roe  = metrics.get("ROE_%", 0)
    npm  = metrics.get("Net_Profit_Margin_%", 0)
    de   = metrics.get("Debt_to_Equity", 0)
    ocf  = metrics.get("Operating Cf_M", 0)
    fcf  = metrics.get("Free Cf_M", 0)
    rsi  = metrics.get("RSI_14", 50)
    vol  = metrics.get("Volatility_20", 0)
    roc  = metrics.get("ROC_20", 0)
    dc   = metrics.get("Death_Cross", 0)
    eq_r = metrics.get("Equity_Ratio", 0)

    chips = []
    if   z > 2.99: chips.append(("chip-g", f"Financial Health Score: {z:.2f}  —  Strong"))
    elif z > 1.81: chips.append(("chip-a", f"Financial Health Score: {z:.2f}  —  Caution"))
    else:          chips.append(("chip-r", f"Financial Health Score: {z:.2f}  —  Weak"))
    if roe > 15:   chips.append(("chip-g", f"Return on Equity: {roe:.1f}%  —  Strong"))
    elif roe > 5:  chips.append(("chip-a", f"Return on Equity: {roe:.1f}%  —  Moderate"))
    else:          chips.append(("chip-r", f"Return on Equity: {roe:.1f}%  —  Low"))
    if de < 1.0:   chips.append(("chip-g", f"Debt Level: {de:.2f}x  —  Low"))
    elif de < 2.5: chips.append(("chip-a", f"Debt Level: {de:.2f}x  —  Moderate"))
    else:          chips.append(("chip-r", f"Debt Level: {de:.2f}x  —  High"))
    if ocf > 0:    chips.append(("chip-g", f"Cash Generation: +${ocf:.0f}M"))
    else:          chips.append(("chip-r", f"Cash Generation: ${ocf:.0f}M"))
    if dc == 1:    chips.append(("chip-r", "Bearish Price Signal Active"))
    if rsi < 35:   chips.append(("chip-a", f"Price Momentum: {rsi:.0f}  —  Low"))
    elif rsi > 65: chips.append(("chip-a", f"Price Momentum: {rsi:.0f}  —  Elevated"))
    else:          chips.append(("chip-g", f"Price Momentum: {rsi:.0f}  —  Stable"))

    chips_html = (
        '<div class="chip-row">'
        + "".join(f'<span class="chip {c}">{l}</span>' for c, l in chips)
        + "</div>"
    )

    if pred == 1 and prob >= 0.7:
        card_mod = "exp-card-critical"; hdr_color = "#c0392b"
        hdr  = "Critical Risk — Immediate Attention Required"
        summ = (f"The analysis assigns a **{pct}** probability of financial distress to **{name}** "
                f"over the **{period_label}** period. This is the highest severity classification, "
                f"indicating serious financial strain that requires prompt action.")
        fin  = (f"The overall financial health score stands at **{z:.2f}** "
                + ("— placing it in a zone historically associated with serious financial stress. " if z <= 1.81
                   else "— sitting in an uncertain middle zone where risk is elevated. ")
                + f"Return on equity of **{roe:.1f}%** against a net profit margin of **{npm:.1f}%** "
                + ("signals the company is eroding shareholder value. " if roe < 0
                   else "reflects fragile profitability with no buffer for setbacks. ")
                + f"Cash generation from operations is **${ocf:.0f}M** — "
                + ("the company is burning through cash, which is not sustainable. " if ocf <= 0
                   else "positive but insufficient to offset the other distress signals. ")
                + f"Debt stands at **{de:.2f}x** equity, "
                + ("making it highly dependent on external financing." if de > 2.5
                   else "adding financial pressure to an already stressed picture."))
        mkt  = (f"Market behaviour supports the concern. Annualised price volatility of **{vol:.1f}%** "
                + ("indicates extreme instability. " if vol > 60
                   else "reflects elevated price behaviour. " if vol > 30
                   else "is moderate but consistent with underlying stress. ")
                + ("A bearish price signal is currently active. " if dc == 1 else "")
                + f"Short-term price momentum stands at **{roc:.1f}%** over the past month"
                + (", pointing to accelerating deterioration." if roc < -10
                   else ", showing continued price decline." if roc < 0
                   else ", showing some resilience despite the broader stress signals."))
        acts = ["Review the company's cash position and near-term financial obligations immediately.",
                "Assess whether outstanding debts can be refinanced or renegotiated before maturity.",
                "Examine whether operational cost reductions could restore positive cash generation.",
                "Increase monitoring frequency significantly until the risk picture improves.",
                "Consider whether the current level of exposure to this company is appropriate."]

    elif pred == 1 and prob >= 0.5:
        card_mod = "exp-card-high"; hdr_color = "#c0640c"
        hdr  = "High Risk — Close Monitoring Recommended"
        summ = (f"**{name}** has been flagged as financially distressed, with a **{pct}** probability "
                f"of distress over the **{period_label}** period. This signals meaningful financial "
                f"weakness that warrants close and proactive attention.")
        fin  = (f"The financial health score of **{z:.2f}** "
                + ("is in the distress zone, indicating structural weaknesses. " if z <= 1.81
                   else "sits in an uncertain range — concerning alongside other signals. ")
                + f"Return on equity of **{roe:.1f}%** and net profit margin of **{npm:.1f}%** indicate "
                + ("declining earnings quality. " if roe < 5 else "profitability that has weakened but not collapsed. ")
                + f"Operating cash generation of **${ocf:.0f}M** — "
                + ("insufficient to service debt or invest at the required level. " if ocf < 0
                   else "positive, but the combination with higher debt still raises concern. ")
                + f"Equity ratio of **{eq_r:.2f}** suggests "
                + ("limited financial flexibility." if eq_r < 0.3 else "a moderate equity base."))
        mkt  = (f"Market signals are broadly consistent with the financial picture. "
                f"Volatility of **{vol:.1f}%** "
                + ("is elevated. " if vol > 30 else "is moderate. ")
                + ("A bearish price signal is active. " if dc == 1
                   else "No major bearish price signals are currently active. ")
                + f"Price momentum over the past month is **{roc:.1f}%**, "
                + ("indicating continued downward pressure." if roc < 0 else "showing some near-term stability."))
        acts = ["Conduct a detailed cash flow and liquidity review covering the next 12 months.",
                "Identify and review any debt covenants or obligations with approaching deadlines.",
                "Assess operational efficiency — where can costs be reduced or working capital improved?",
                "Monitor the next quarterly results closely for early signs of improvement.",
                "Review whether the current level of exposure is appropriate given the risk profile."]

    elif prob >= 0.3:
        card_mod = "exp-card-medium"; hdr_color = "#9a6f00"
        hdr  = "Moderate Risk — Worth Monitoring"
        summ = (f"**{name}** is not currently classified as distressed, but the analysis assigns a "
                f"**{pct}** probability of distress over the **{period_label}** period. "
                f"The company appears stable today, but early-warning signals recommend ongoing observation.")
        fin  = (f"The financial health score of **{z:.2f}** "
                + ("is comfortably above the safe threshold. " if z > 2.99
                   else "sits in an intermediate range — no immediate red flags but some stress to watch. ")
                + f"Return on equity of **{roe:.1f}%** with net profit margin of **{npm:.1f}%** are "
                + ("adequate but not strong enough to cushion against any earnings shock. " if 5 <= roe <= 15
                   else "on the weaker side. ")
                + f"Operating cash flow of **${ocf:.0f}M** and free cash flow of **${fcf:.0f}M** "
                + ("are both positive — providing some financial headroom." if ocf > 0 and fcf > 0
                   else "are under some pressure, worth watching."))
        mkt  = (f"Market signals are mixed. Price volatility of **{vol:.1f}%** "
                + ("is elevated. " if vol > 30 else "is contained. ")
                + f"Price momentum over the past month is **{roc:.1f}%**, "
                + ("showing a modest downward drift. " if roc < 0 else "showing positive momentum. ")
                + ("No bearish price signals are currently active." if dc == 0
                   else "A bearish price signal has appeared."))
        acts = ["Continue monitoring on a monthly basis.",
                "Track the next earnings release for any signs of deterioration.",
                "Stay aware of broader sector conditions — particularly oil prices and regional economic shifts.",
                "No immediate action required, but flag as a company to watch."]

    else:
        card_mod = "exp-card-ok"; hdr_color = "#1a7a4a"
        hdr  = "Low Risk — Financially Sound"
        summ = (f"**{name}** is in good financial health. The analysis assigns only a **{pct}** probability "
                f"of distress over the **{period_label}** period — placing it firmly in the low-risk category.")
        fin  = (f"The financial health score of **{z:.2f}** "
                + ("is comfortably above the safe threshold, reflecting a robust balance sheet. " if z > 2.99
                   else "is positive and does not raise any concerns. ")
                + f"Return on equity of **{roe:.1f}%** reflects "
                + ("strong returns for shareholders. " if roe > 15 else "healthy profitability. ")
                + f"Net profit margin of **{npm:.1f}%** demonstrates solid earnings performance. "
                + f"Operating cash flow of **${ocf:.0f}M** and free cash flow of **${fcf:.0f}M** confirm "
                + "the company is generating real cash — a key sign of sustainable financial strength. "
                + f"With debt at **{de:.2f}x** equity, the balance sheet "
                + ("is conservatively positioned." if de < 1.0 else "is in a manageable position."))
        mkt  = (f"Market signals are broadly positive. Price volatility of **{vol:.1f}%** "
                + ("is low, consistent with stable investor sentiment. " if vol < 20
                   else "is moderate and not a concern given the underlying fundamentals. ")
                + ("No bearish price signals are active. " if dc == 0
                   else "A bearish price signal has technically appeared, but is not corroborated by "
                        "the fundamentals — likely short-term noise. ")
                + f"Price momentum over the past month is **{roc:.1f}%**"
                + (", showing positive near-term performance." if roc > 0
                   else ", showing mild softness which is not unusual for a financially healthy company."))
        acts = ["No immediate action required — the company is in good financial standing.",
                "Continue with standard periodic monitoring.",
                "This company can serve as a useful benchmark for peer comparisons within the sector.",
                "Keep an eye on broader macro conditions (oil prices, interest rates, regional currency moves)."]

    st.markdown(
        f'<div class="exp-card {card_mod}">'
        f'<div style="font-size:1.5rem;font-weight:800;color:{hdr_color};'
        f'margin-bottom:0;line-height:1.2;">{hdr}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Summary**");          st.markdown(summ)
    st.markdown("**Financial Health**"); st.markdown(fin)
    st.markdown("**Market Behaviour**"); st.markdown(mkt)
    st.markdown("**Signal Summary**");   st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown("**Recommended Next Steps**")
    for a in acts:
        st.markdown(f"- {a}")
    st.markdown(
        '<p class="disclaimer-text">This analysis is produced by an AI model trained on historical '
        'data from companies listed across MENA markets. It is intended to support informed '
        'decision-making and does not constitute financial or investment advice.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="page-title">MENA Financial Distress Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">AI-powered financial health assessment across MENA markets</div>',
        unsafe_allow_html=True,
    )

    model, scaler, explainer = load_model_resources()

    with st.sidebar:
        st.markdown("### Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy",    "86.9%", help="Average accuracy across all countries and sectors")
        c2.metric("Distress Prediction", "88.2%", help="Recall —  catching distress events early")
        st.caption("Trained on data from 5 countries and 20+ sectors")
        st.markdown("---")
        st.markdown("### Fairness Report")
        run_ethical = st.checkbox("Show consistency analysis", value=False,
                                  help="View how evenly the model performs across countries and sectors")

    tab1, tab2 = st.tabs(["Company Analysis", "Fairness & Consistency"])

    with tab1:
        st.markdown(
            """<div class="filter-panel" style="background-color:#f9fafb;border:1px solid #e5e7eb;
            border-radius:10px;padding:1rem;margin-bottom:1rem;">
            <div style="font-size:1.1rem;font-weight:800;color:#4b5563;margin-bottom:0.8rem;">
            Select a Company to Analyse</div>""",
            unsafe_allow_html=True,
        )

        period_options = {
            "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
            "1 Year": "1y", "2 Years": "2y", "3 Years": "3y",
            "5 Years": "5y", "Maximum Available": "max",
        }

        fc1, fc2, fc3, fc4 = st.columns([1.1, 1.1, 1.5, 1.1])
        with fc1:
            st.markdown("**Country**")
            country = st.selectbox("Country", list(MENA_COMPANIES.keys()),
                                   label_visibility="collapsed", key="sel_country")
        all_co  = MENA_COMPANIES[country]
        sectors = ["All Sectors"] + sorted(set(c["sector"] for c in all_co))
        with fc2:
            st.markdown("**Sector**")
            sel_sector = st.selectbox("Sector", sectors,
                                      label_visibility="collapsed", key="sel_sector")
        filtered = all_co if sel_sector == "All Sectors" else [c for c in all_co if c["sector"] == sel_sector]
        with fc3:
            st.markdown("**Company**")
            sel_name = st.selectbox("Company", [c["name"] for c in filtered],
                                    label_visibility="collapsed", key="sel_company")
        with fc4:
            st.markdown("**Period**")
            sel_period_label = st.selectbox("Period", list(period_options.keys()),
                                            index=3, label_visibility="collapsed", key="sel_period")

        period  = period_options[sel_period_label]
        co_info = next(c for c in filtered if c["name"] == sel_name)

        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([2.5, 1, 2.5])
        with btn_col:
            analyze = st.button("Run Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if not analyze:
            st.markdown('<div class="sec-title">About This Tool</div>', unsafe_allow_html=True)
            st.markdown("""<div class="info-card">
                <p style="margin:0 0 0.65rem;font-size:1.02rem;">
                    This tool uses artificial intelligence to assess the financial health of
                    publicly listed companies across the MENA region. Select a country, sector,
                    and company above, then click <strong>Run Analysis</strong>.
                </p>
                <p style="margin:0;color:#5a6a85;font-size:0.93rem;">
                    The analysis draws on financial data, market behaviour, and regional risk factors —
                    evaluated using a model trained on historical data from five countries and
                    over twenty industry sectors.
                </p></div>""", unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Coverage</div>', unsafe_allow_html=True)
            total = sum(len(c) for c in MENA_COMPANIES.values())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Companies Covered", total)
            m2.metric("Countries",         len(MENA_COMPANIES))
            m3.metric("Industry Sectors",  "20+")
            m4.metric("Data",              "Live Market")
            for ctry, comps in MENA_COMPANIES.items():
                with st.expander(f"{ctry}  —  {len(comps)} companies"):
                    grp = {}
                    for c in comps: grp.setdefault(c["sector"], []).append(c["name"])
                    for sec, names in sorted(grp.items()):
                        st.markdown(f"**{sec}**")
                        for n in names: st.markdown(f"&emsp;• {n}")
            return

        with st.spinner(f"Retrieving data for {sel_name}…"):
            data = fetch_company_data(co_info["ticker"], period)
        if not data or len(data["history"]) == 0:
            st.error(f"Could not retrieve data for {sel_name}. Try a shorter period or different company.")
            st.stop()

        with st.spinner("Running analysis…"):
            metrics = calculate_all_metrics(data)
        if not metrics:
            st.error("An error occurred while calculating metrics. Please try another company.")
            st.stop()

        pred, prob, risk = make_prediction(model, scaler, metrics)

        st.markdown(f"""<div class="co-banner">
            <h2>{sel_name}</h2>
            <p>{co_info["sector"]} &nbsp;·&nbsp; {country} &nbsp;·&nbsp;
               {co_info["ticker"]} &nbsp;·&nbsp; {sel_period_label}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Assessment Results</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            v_cls = "verdict-distress" if pred == 1 else "verdict-healthy"
            v_col = "#c0392b"          if pred == 1 else "#1a7a4a"
            v_lbl = "Financial Distress Detected" if pred == 1 else "Financially Healthy"
            v_sub = ("The analysis has flagged this company as being under financial stress."
                     if pred == 1 else "No significant financial stress detected.")
            st.markdown(f'<div class="{v_cls}"><p class="verdict-label" style="color:{v_col};">'
                        f'{v_lbl}</p><p class="verdict-sub">{v_sub}</p></div>',
                        unsafe_allow_html=True)
        with r2:
            st.plotly_chart(create_gauge(prob * 100, "Distress Probability"), use_container_width=True)
        with r3:
            risk_colors = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e67e22", "Critical": "#c0392b"}
            risk_desc   = {"Low": "No significant concern.", "Medium": "Some signals to monitor.",
                           "High": "Requires close attention.", "Critical": "Immediate action needed."}
            st.markdown(f'<div class="risk-badge" style="background:{risk_colors[risk]};">'
                        f'<h4>Risk Level</h4><h2>{risk}</h2><p>{risk_desc[risk]}</p></div>',
                        unsafe_allow_html=True)
        with r4:
            conf = (1 - abs(prob - 0.5) * 2) * 100
            st.plotly_chart(create_gauge(conf, "Result Confidence"), use_container_width=True)

        st.markdown('<div class="sec-title">Detailed Assessment</div>', unsafe_allow_html=True)
        render_explanation(sel_name, sel_period_label, pred, prob, risk, metrics)

        st.markdown('<div class="sec-title">Stock Price History</div>', unsafe_allow_html=True)
        st.plotly_chart(create_price_chart(data["history"], sel_name), use_container_width=True)

        st.markdown('<div class="sec-title">What Is Driving This Result?</div>', unsafe_allow_html=True)
        if explainer is not None:
            X_df     = pd.DataFrame([[metrics.get(f, 0) for f in SELECTED_FEATURES]],
                                    columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(X_df)
            try:
                shap_vals = explainer.shap_values(X_scaled)[0]
                render_shap_narrative(shap_vals, pred, prob)

                st.markdown('<p class="chart-note">This chart groups all factors into categories and '
                            'shows whether each is raising or lowering the assessed risk.</p>',
                            unsafe_allow_html=True)
                st.pyplot(shap_category_chart(explainer, X_scaled, sel_name),
                          use_container_width=True); plt.close()

                st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
                st.markdown('<p class="chart-note">This chart shows the top individual factors and '
                            'whether each one is increasing or reducing the risk score.</p>',
                            unsafe_allow_html=True)
                st.pyplot(shap_direction_chart(explainer, X_scaled, sel_name),
                          use_container_width=True); plt.close()

                st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
                st.markdown('<p class="chart-note">This chart ranks the top factors by overall '
                            'influence on the result, regardless of direction.</p>',
                            unsafe_allow_html=True)
                st.pyplot(shap_importance_chart(explainer, X_scaled, sel_name),
                          use_container_width=True); plt.close()

                st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
                with st.expander("View step-by-step breakdown"):
                    st.markdown('<p class="chart-note">This chart shows how the model built up its '
                                'final score step by step, starting from a baseline.</p>',
                                unsafe_allow_html=True)
                    st.pyplot(shap_waterfall_chart(explainer, X_scaled, sel_name),
                              use_container_width=True); plt.close()

            except Exception as e:
                st.info(f"Factor analysis could not be generated: {e}")
        else:
            st.markdown("""<div class="info-card">
                <p style="margin:0;color:#5a6a85;">
                    Detailed factor analysis requires the base XGBoost model file.
                </p></div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Download Report</div>', unsafe_allow_html=True)
        report_df = pd.DataFrame([{
            "Company": sel_name, "Ticker": co_info["ticker"],
            "Country": country,  "Sector": co_info["sector"],
            "Period":  sel_period_label, "Date": datetime.now().strftime("%Y-%m-%d"),
            "Result":  "Distressed" if pred == 1 else "Healthy",
            "Probability": f"{prob:.4f}", "Risk Level": risk,
            **{FEATURE_LABELS.get(k, k): (f"{v:.4f}" if isinstance(v, float) else v)
               for k, v in metrics.items()},
        }])
        st.download_button(
            "Download Full Report (CSV)",
            report_df.to_csv(index=False).encode("utf-8"),
            f"{sel_name.replace(' ', '_')}_distress_report.csv",
            "text/csv", use_container_width=True,
        )

    # ── Fairness tab ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-title">Model Fairness & Consistency</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
            <p style="margin:0 0 0.6rem;font-size:1.01rem;">
                A responsible AI model should perform consistently — not significantly more or less
                accurate for companies from one country or sector versus another.
            </p>
            <p style="margin:0;color:#5a6a85;font-size:0.91rem;">
                Fairness is assessed by comparing performance across all five countries and all
                industry sectors. A well-designed model shows minimal variation between groups.
            </p></div>""", unsafe_allow_html=True)

        if not run_ethical:
            st.markdown("""<div class="info-card" style="margin-top:1.2rem;">
                <p style="margin:0 0 0.5rem;"><strong>How to view the fairness analysis</strong></p>
                <p style="margin:0;color:#5a6a85;">
                    Enable <strong>"Show consistency analysis"</strong> in the sidebar.
                </p></div>""", unsafe_allow_html=True)
            return

        report = load_fairness_report()
        if report is None:
            st.warning("The fairness report could not be loaded. "
                       "Add `fairness_report.json` file ID to `GDRIVE_FILE_IDS`, or drop "
                       "the file next to `app.py`.")
            return

        overall = report.get("overall_assessment", {})
        is_pass = "PASSED" in overall.get("status", "").upper()

        if is_pass:
            st.markdown("""<div style="background-color:#ecfdf5;border-left:5px solid #10b981;
                padding:1.25rem;border-radius:8px;margin-bottom:1.5rem;">
                <div style="font-size:1.1rem;font-weight:700;color:#065f46;">
                    Consistency Validation Passed</div>
                <p style="margin:0.5rem 0 0;font-size:0.95rem;color:#065f46;opacity:0.9;">
                    The model performs consistently across all countries and sectors.
                </p></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background-color:#fffbeb;border-left:5px solid #f59e0b;
                padding:1.25rem;border-radius:8px;margin-bottom:1.5rem;">
                <div style="font-size:1.1rem;font-weight:700;color:#92400e;">
                    Performance Variation Detected</div>
                <p style="margin:0.5rem 0 0;font-size:0.95rem;color:#92400e;opacity:0.9;">
                    Minor inconsistencies were found in some groups. See details below.
                </p></div>""", unsafe_allow_html=True)

        if overall.get("assessment"):
            st.markdown(
                f'<div style="padding:1rem 0;border-top:1px solid #f3f4f6;">'
                f'<span style="font-weight:700;color:#374151;">Finding:</span> '
                f'<span style="color:#4b5563;">{overall["assessment"]}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border:none;border-top:1px solid #e5e7eb;margin:2rem 0;">',
                    unsafe_allow_html=True)

        fm   = report.get("fairness_metrics", {})
        c_fm = fm.get("country", {})
        if c_fm:
            st.markdown('<div class="sec-title">Performance by Country</div>', unsafe_allow_html=True)
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            cc1.metric("Accuracy Variation",    f"{c_fm.get('f1_std', 0):.4f}")
            cc2.metric("Accuracy Gap",          f"{c_fm.get('f1_range', 0):.4f}")
            cc3.metric("Reliability Variation", f"{c_fm.get('auc_std', 0):.4f}")
            cc4.metric("Reliability Gap",       f"{c_fm.get('auc_range', 0):.4f}")
            cc5.metric("Status", "Passed" if c_fm.get("is_fair") else "Review needed")
            cdata = report.get("country_analysis", [])
            if cdata:
                st.plotly_chart(create_fairness_chart(pd.DataFrame(cdata), "Country", "Country"),
                                use_container_width=True)
                with st.expander("View country-level figures"):
                    dc2 = pd.DataFrame(cdata).copy()
                    for col in ["Distress_Rate", "F1_Score", "AUC_ROC", "Precision", "Recall", "Accuracy"]:
                        if col in dc2.columns:
                            dc2[col] = (dc2[col] * 100).round(2).astype(str) + "%"
                    dc2.columns = [c.replace("_", " ") for c in dc2.columns]
                    st.dataframe(dc2, use_container_width=True)

        s_fm = fm.get("sector", {})
        if s_fm:
            st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Performance by Sector</div>', unsafe_allow_html=True)
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Accuracy Variation",    f"{s_fm.get('f1_std', 0):.4f}")
            sc2.metric("Accuracy Gap",          f"{s_fm.get('f1_range', 0):.4f}")
            sc3.metric("Reliability Variation", f"{s_fm.get('auc_std', 0):.4f}")
            sc4.metric("Reliability Gap",       f"{s_fm.get('auc_range', 0):.4f}")
            sc5.metric("Status", "Passed" if s_fm.get("is_fair") else "Review needed")
            sdata = report.get("sector_analysis", [])
            if sdata:
                st.plotly_chart(create_fairness_chart(pd.DataFrame(sdata), "Sector", "Sector"),
                                use_container_width=True)
                with st.expander("View sector-level figures"):
                    ds2 = pd.DataFrame(sdata).copy()
                    for col in ["Distress_Rate", "F1_Score", "AUC_ROC", "Precision", "Recall", "Accuracy"]:
                        if col in ds2.columns:
                            ds2[col] = (ds2[col] * 100).round(2).astype(str) + "%"
                    ds2.columns = [c.replace("_", " ") for c in ds2.columns]
                    st.dataframe(ds2, use_container_width=True)

        st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
        st.markdown("### Conclusion")
        if overall.get("country_fair") and overall.get("sector_fair"):
            st.success("The model produces consistent results across all MENA countries and industry sectors.")
        elif overall.get("country_fair") or overall.get("sector_fair"):
            st.warning("Minor variation was found in one dimension. Results are broadly consistent.")
        else:
            st.error("Notable performance variation detected. Results for affected groups should be "
                     "interpreted with additional care.")

        ts = report.get("timestamp", "")
        if ts:
            st.caption(f"Report generated: {ts[:19].replace('T', ' ')}")


if __name__ == "__main__":
    main()