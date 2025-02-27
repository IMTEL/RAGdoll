import sys
import os

# Add the project root directory (assuming tests/ is one level down) to sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)