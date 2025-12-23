"""
Update streamlit-app.py to use get_data_file() for all pd.read_csv calls.
This allows the app to work with both full data and sample data.
"""

def update_streamlit_app():
    """Update all pd.read_csv calls in streamlit-app.py."""
    
    with open('streamlit-app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all pd.read_csv calls with get_data_file()
    replacements = [
        ("pd.read_csv('caracteristics.csv'", "pd.read_csv(get_data_file('caracteristics.csv')"),
        ("pd.read_csv('places.csv'", "pd.read_csv(get_data_file('places.csv')"),
        ("pd.read_csv('users.csv'", "pd.read_csv(get_data_file('users.csv')"),
        ("pd.read_csv('vehicles.csv'", "pd.read_csv(get_data_file('vehicles.csv')"),
        ("pd.read_csv('holidays.csv'", "pd.read_csv(get_data_file('holidays.csv')"),
        ("pd.read_csv('acc.csv'", "pd.read_csv(get_data_file('acc.csv')"),
        ("pd.read_csv('master_acc.csv'", "pd.read_csv(get_data_file('master_acc.csv')"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Write back
    with open('streamlit-app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated streamlit-app.py to use get_data_file()")
    print(f"   Made {sum(content.count(new) for old, new in replacements)} replacements")

if __name__ == "__main__":
    update_streamlit_app()
