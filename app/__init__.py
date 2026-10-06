# Q-ORBIT frontend package.
# This file makes `app/` a regular package so `from app.space_theme import ...`
# always resolves to this directory — even when Streamlit's script folder
# (which contains app.py) sits earlier on sys.path and would otherwise let
# the *file* app.py shadow the *package* app/.
