def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF5733")

    Returns:
        Tuple of RGB values (0-1 range)
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
