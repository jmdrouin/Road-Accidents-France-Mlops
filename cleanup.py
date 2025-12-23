#!/usr/bin/env python3
"""Clean up streamlit-app.py by removing everything after the footer"""

with open('streamlit-app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the last occurrence of the footer closing
footer_marker = '    unsafe_allow_html=True\n)'
last_pos = content.rfind(footer_marker)

if last_pos != -1:
    # Keep content up to and including the footer
    clean_content = content[:last_pos + len(footer_marker)]
    
    with open('streamlit-app.py', 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    print(f"✅ Cleaned! Kept {len(clean_content)} characters up to footer")
    print(f"   Original: {len(content)} characters")
    print(f"   Removed: {len(content) - len(clean_content)} characters")
else:
    print("❌ Footer marker not found!")
