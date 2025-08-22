#!/usr/bin/env python3
"""
Script to fix remaining chart functions that still use old limits[0] pattern
"""

def fix_limits_pattern():
    # Read the file
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find and replace
    old_pattern = '''    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")'''
    
    new_pattern = '''    # Use helper function to add fixed control limits and percentage-based zoom
    try:
        zoom_pct = limits if isinstance(limits, (int, float)) else 10
        fig = get_control_limits_and_apply_zoom(fig, selected_tech_layer, selected_chart_mpx, zoom_pct, "Chart")
    except:
        fig = get_control_limits_and_apply_zoom(fig, selected_tech_layer, selected_chart_mpx, 10, "Chart")'''
    
    # Count occurrences before replacement
    count_before = content.count('limits[0]')
    
    # Apply the replacement
    content = content.replace(old_pattern, new_pattern)
    
    # Count occurrences after replacement
    count_after = content.count('limits[0]')
    
    # Write back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed limit patterns: {count_before} -> {count_after} occurrences of 'limits[0]'")
    return count_before, count_after

if __name__ == "__main__":
    fix_limits_pattern()
