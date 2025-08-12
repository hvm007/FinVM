from supabase import create_client
import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# Get the variables from the environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set in your environment or a .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_connection():
    """Return Supabase client for compatibility with old code."""
    return supabase