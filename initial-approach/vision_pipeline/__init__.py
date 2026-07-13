"""Vision pipeline package - self-contained multi-step image analysis.

Runs: Scene Parser -> Detail Identifier -> Detail Extractor (Florence-2)
      -> Crop Tool -> Detail Focusser

All modules are local to the council package - no external llm_council dependency.
"""
