# Theme color palettes
def get_theme_colors(mode='light'):
    if mode == 'dark':
        return {
            'actual': '#00ff99',
            'forecast': '#ffaa00',
            'conf_fill': 'rgba(255, 170, 0, 0.2)',
            'template': 'plotly_dark',
            'bg': '#1e1e1e',
            'text': '#ffffff'
        }
    else:
        return {
            'actual': '#28a745',
            'forecast': '#ff5733',
            'conf_fill': 'rgba(255, 87, 51, 0.2)',
            'template': 'plotly_white',
            'bg': '#ffffff',
            'text': '#000000'
        }
