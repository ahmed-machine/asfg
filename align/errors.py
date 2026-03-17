"""Custom exception hierarchy for the alignment pipeline."""


class AlignmentError(Exception):
    """Base exception for alignment pipeline failures."""


class InsufficientDataError(AlignmentError):
    """Not enough data (matches, GCPs, overlap) to proceed."""


class CoarseOffsetError(AlignmentError):
    """Coarse offset detection failed."""


class WarpError(AlignmentError):
    """Warp application failed."""


class AlreadyAlignedError(AlignmentError):
    """Images are already well-aligned; no correction needed."""


class UserAbortError(AlignmentError):
    """User declined to proceed."""
