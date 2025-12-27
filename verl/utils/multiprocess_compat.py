"""
Compatibility helpers for multiprocess resource tracker issues.

This module provides a no-op patch to avoid import errors in environments
where the underlying multiprocess resource tracker is either absent or
unreliable. If you hit semaphore/shared memory tracker warnings, extend
this shim accordingly.
"""


def patch_multiprocess_resource_tracker() -> None:
    """
    Apply a best-effort patch to multiprocess.resource_tracker.

    Currently implemented as a no-op to satisfy imports. Extend this
    if you need to adjust tracker behavior for your platform.
    """
    try:
        import multiprocess.resource_tracker as rt  # noqa: F401
    except Exception:
        return

