
import streamlit as st
from datetime import datetime
import pytz

def is_market_open():
    """Checks if the Indian stock market is open."""
    IST = pytz.timezone('Asia/Kolkata')
    now = datetime.now(IST)
    is_weekday = 0 <= now.weekday() <= 4  # Monday (0) to Friday (4)
    market_open_time = now.replace(hour=9, minute=15, second=0).time()
    market_close_time = now.replace(hour=15, minute=30, second=0).time()
    is_market_hours = market_open_time <= now.time() <= market_close_time
    return is_weekday and is_market_hours

def display_market_status():
    """Checks market status and displays a corresponding banner in the UI."""
    # This check prevents the banner from showing up on the login page
    if 'auth_manager' in st.session_state and st.session_state.auth_manager.is_logged_in():
        if is_market_open():
            st.info("📈 **Market: Open**", icon="✅")
        else:
            st.warning("🌙 **Market: Closed**", icon="ℹ️")