"""Vendored VITS model from rhasspy/piper (piper_train.vits).

Only the modules needed to unpickle and run inference on the VITS
checkpoint are included.  Training-only modules (dataset, losses,
mel_processing) are stubbed out so the import chain succeeds.

Source: https://github.com/rhasspy/piper  (src/python/piper_train/vits/)
License: MIT
"""
