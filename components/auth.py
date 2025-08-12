import streamlit as st
import hashlib
from components.database import get_connection


class AuthManager:
    def __init__(self):
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
            st.session_state.user_info = None

    def hash_password(self, password):
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, username, password, avatar=None):
        """Create a new user in Supabase."""
        sb = get_connection()
        hashed_password = self.hash_password(password)

        # This logic is correct. It checks if the `data` list is non-empty.
        existing = sb.table("users").select("username").eq("username", username).execute()
        if existing.data:
            st.error("Username already exists. Please choose another.")
            return False

        try:
            sb.table("users").insert({
                "username": username,
                "password_hash": hashed_password,
                "avatar": avatar
            }).execute()
            st.success("Account created successfully! Please log in.")
            return True
        except Exception as e:
            st.error(f"Error during signup: {e}")
            return False

    def login(self, username, password):
        """Authenticate user against Supabase."""
        sb = get_connection()
        hashed_password = self.hash_password(password)

        res = sb.table("users").select("username, avatar, password_hash").eq("username", username).execute()

        if res.data and res.data[0]["password_hash"] == hashed_password:
            user_data = res.data[0]
            st.session_state.logged_in = True
            st.session_state.user_info = {
                "username": user_data["username"],
                "avatar": user_data["avatar"]
            }
            return True

        st.error("Invalid username or password.")
        return False

    def logout(self):
        """Clear session."""
        st.session_state.logged_in = False
        st.session_state.user_info = None

    def is_logged_in(self):
        return st.session_state.get("logged_in", False)

    def get_current_user(self):
        return st.session_state.get("user_info")

    def get_user_avatar(self, username):
        """Fetches an avatar for any user."""
        sb = get_connection()

        res = sb.table("users").select("avatar").eq("username", username).execute()

        return res.data[0]["avatar"] if res.data else "👤"