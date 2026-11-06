"""LEGACY (HIVE 1.x) optional PyImageJ/Fiji bridge.

Deliberately isolated from the analysis pipeline: PyImageJ starts a JVM and
resolves Maven artifacts on first run, which is slow and fails in ways that
have nothing to do with image analysis. Nothing in HIVE depends on it; it
exists so a Fiji installation can be reached when one is available.
"""


def start_imagej(fiji_path="/Applications/Fiji", add_legacy=False):
    """Start a headless ImageJ/Fiji gateway.

    Args:
        fiji_path: Path to the Fiji installation.
        add_legacy: Include the ImageJ 1.x legacy layer. Off by default --
            the legacy layer is incompatible with true headless operation.

    Returns:
        An initialised PyImageJ gateway.

    Raises:
        ImportError: If ``pyimagej`` is not installed (imported lazily).

    Note:
        The first call downloads Maven dependencies and can take minutes.
    """
    import imagej
    ij = imagej.init(fiji_path, mode="headless", add_legacy=add_legacy)
    return ij


def pyimagej_status(fiji_path="/Applications/Fiji"):
    """Report whether the PyImageJ bridge can start, without raising.

    Args:
        fiji_path: Path to the Fiji installation.

    Returns:
        ``(ok, message)`` -- ``(True, version string)`` on success, or
        ``(False, error text)``. Broad by design: this is a probe used by the
        CLI's ``--pyimagej-check`` flag, where any failure is a plain "not
        available" answer rather than a crash.
    """
    try:
        ij = start_imagej(fiji_path=fiji_path, add_legacy=False)
        return True, f"PyImageJ/Fiji started. Version: {ij.getVersion()}"
    except Exception as exc:
        return False, str(exc)
