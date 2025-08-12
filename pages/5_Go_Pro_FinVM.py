import streamlit as st
from components.auth import AuthManager
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Go Pro - FinVM",
    page_icon="👑",
    layout="wide"
)

# --- Authentication Check ---
auth_manager = st.session_state.get('auth_manager')
if not auth_manager or not auth_manager.is_logged_in():
    st.error("Please login to view subscription options.")
    st.stop()

user_info = auth_manager.get_current_user()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .sub-header {
        background: linear-gradient(90deg, #ffc107, #ffeb3b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .pro-card {
        background: linear-gradient(145deg, #0d1b3e, #301934); /* Royal/Navy Blue Gradient */
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 8px 12px rgba(255, 193, 7, 0.15);
        text-align: center;
    }
    .feature-list {
        text-align: left;
        margin-top: 1.5rem;
        padding-left: 20px;
    }
    .feature-list li {
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    .price {
        font-size: 2.5rem;
        font-weight: bold;
        color: #9400D3);
        margin: 1rem 0;
    }
    .price-sub {
        font-size: 1rem;
        color: #fafafa;
    }
    .qr-container {
        text-align: center;
        background: #301934;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Page Content ---

st.markdown('<h1 class="sub-header">👑 Go Pro and Supercharge Your Insights</h1>', unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center;'>Unlock exclusive AI-powered tools and personalized analysis to elevate your investment strategy.</h4>",
    unsafe_allow_html=True)

st.markdown("---")

# --- Main Subscription Card ---
col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:
    pro_card_html = """
    <div class="pro-card">
        <h2>FinVM Pro</h2>
        <p>The ultimate toolkit for the data-driven investor.</p>
        <div class="price">
            ₹499 <span class="price-sub">/ month</span>
        </div>
        <ul class="feature-list">
            <li><strong>Advanced AI Models:</strong> Access to our next-gen predictive models for higher accuracy.</li>
            <li><strong>Personalized Portfolio Manager:</strong> Receive automated rebalancing suggestions and risk-optimization reports.</li>
            <li><strong>Expanded Stock Universe:</strong> Analyze stocks beyond the NIFTY 50, including Mid-Caps and Small-Caps.</li>
            <li><strong>Real-time AI Alerts:</strong> Get instant notifications on market-moving events and portfolio risks.</li>
            <li><strong>Exclusive Research:</strong> Unlock in-depth market reports and sentiment trend analysis.</li>
        </ul>
    </div>
    """
    st.markdown(pro_card_html, unsafe_allow_html=True)


st.markdown("---")

# --- Payment Section ---
st.header("Ready to Go Pro?")

pay_col1, pay_col2 = st.columns([1.2, 1])

with pay_col1:
    st.subheader("How to Subscribe")
    st.markdown("""
    1.  **Scan the QR Code:** Open your favorite UPI payment app (like Google Pay, PhonePe, or Paytm).
    2.  **Enter the Amount:** Make a payment of **₹499**.
    3.  **Confirm Payment:** Complete the transaction.
    4.  **Activation:** After a successful payment, your Pro plan will be activated within 2 hours.

    *PLEASE SCAN THE QR!*
    """)

    st.info(
        "After payment, please email your transaction ID to `activation@finvm.com` if your account isn't upgraded automatically.")

with pay_col2:
    st.subheader("Scan to Pay")
    try:
        qr_image_path = Path("qr_code.png")
        if qr_image_path.exists():
            st.image(str(qr_image_path), use_container_width=True)
        else:
            st.error("QR Code image (`qr_code.png`) not found. Please add it to your project's root directory.")
    except Exception as e:
        st.error(f"Could not load QR code image: {e}")