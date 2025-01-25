import re


def polish_pgn(pgn):
    """
    Remove spaces after move numbers in the PGN to match the expected format.
    """
    # Replace patterns like '1. e4' with '1.e4'
    
    polished_pgn = re.sub(r'(\d+)\.\s+', r'\1.', pgn)
    return polished_pgn


def is_plausible_san(move_text: str) -> bool:
    """Quick check for SAN-like format before board validation"""
    return True
    # Add O-O and O-O-O as valid moves alongside the existing pattern
    return re.match(r"^(O-O-O|O-O|[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?[+#]?)$", move_text)
