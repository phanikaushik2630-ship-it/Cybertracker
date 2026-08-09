import sys
import os

# Add root folder to sys.path so imports resolve seamlessly on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app
