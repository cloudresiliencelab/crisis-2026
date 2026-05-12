"""
Permission tag -> codebook section name(s) for permission-restricted matching.

Section names must match Codebook.csv header rows exactly. Used by
validate_gold_set.py (gold-set validation) and annotate_all_snippets.py
(sensitivity analysis). Unknown permissions fall back to all sections.
"""

PERMISSION_TO_SECTIONS: dict[str, list[str]] = {
    "location": ["Location"],
    "storage": ["Storage / File"],
    "file": ["Storage / File"],
    "instance": ["Instance"],
    "bluetooth": ["Bluetooth"],
    "camera": ["Camera"],
    "overlay": ["Overlay"],
    "disturb mode": ["Disturb Mode"],
    "phone number": ["Phone Number"],
    "filter calls": ["Filter Calls"],
    "sms": ["SMS"],
    "contact": ["Contact"],
    "body sensor": ["Body Sensor"],
    "call logs": ["Call Logs"],
    "microphone": ["Microphone"],
    "clipboard": ["Clipboard"],
    "metadata": ["Metadata"],
    "network data": ["Network Data"],
    "accessibility": ["Accessibility"],
}
