from __future__ import annotations

try:
    from tqdm.auto import tqdm as tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments
    class _SilentProgress:
        def __init__(self, iterable, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_kwargs) -> None:
            return None

    def tqdm(iterable, **kwargs):
        return _SilentProgress(iterable, **kwargs)

