import os

# Some conda/Windows environments ship two OpenMP runtimes (MKL + PyTorch) and abort on import
# order; this is a harmless, test-only workaround. It is a no-op everywhere else (e.g. CI).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
