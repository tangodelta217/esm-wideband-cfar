"""Utilities for tracking spectral peaks over consecutive frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


@dataclass
class _TrackState:
    track_id: int
    bins: List[int] = field(default_factory=list)
    freqs: List[float] = field(default_factory=list)
    start_frame: int = 0
    last_frame: int = 0
    gap_count: int = 0
    closed: bool = False
    matched_in_frame: bool = False

    def snapshot(self) -> dict:
        return {
            "id": self.track_id,
            "bins": list(self.bins),
            "freqs_hz": list(self.freqs),
            "start_frame": self.start_frame,
            "last_frame": self.last_frame,
            "length": len(self.bins),
            "gap_count": self.gap_count,
        }


class PeakTracker:
    """Track spectral peaks across frames using a simple nearest-neighbour association."""

    def __init__(
        self,
        sample_rate_hz: float,
        fft_size: int,
        max_gap: int = 2,
        bin_tolerance: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        sample_rate_hz:
            Sample rate used to convert FFT bins to absolute frequency.
        fft_size:
            FFT size corresponding to the bin spacing.
        max_gap:
            Maximum number of consecutive frames a track may go unmatched before closing.
        bin_tolerance:
            Maximum distance (in bins) allowed when associating detections to existing tracks.
        """
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if fft_size <= 0:
            raise ValueError("fft_size must be positive")
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        if bin_tolerance < 0:
            raise ValueError("bin_tolerance must be non-negative")

        self.sample_rate_hz = float(sample_rate_hz)
        self.fft_size = int(fft_size)
        self.max_gap = int(max_gap)
        self.bin_tolerance = int(bin_tolerance)

        self._tracks: List[_TrackState] = []
        self._closed_buffer: List[dict] = []
        self._next_id = 1
        self._frame_index = -1

    def bin_to_hz(self, bin_index: int) -> float:
        """Convert a bin index into a frequency in Hz, assuming a centered spectrum."""
        return (bin_index / self.fft_size - 0.5) * self.sample_rate_hz

    def update(self, detections: Iterable[int]) -> Tuple[List[dict], List[dict]]:
        """Update tracker state with detections from the next frame.

        Parameters
        ----------
        detections:
            Iterable of integer FFT bin indices representing detected peaks.

        Returns
        -------
        active_tracks:
            List of dictionaries describing tracks that remain active after the update.
        closed_tracks:
            Tracks that were closed during this update step.
        """
        self._frame_index += 1
        current_frame = self._frame_index

        unique_bins = sorted(set(int(b) for b in detections))
        for track in self._tracks:
            track.matched_in_frame = False

        # Associate detections to existing tracks.
        for bin_idx in unique_bins:
            best_track: _TrackState | None = None
            best_delta = None
            for track in self._tracks:
                if track.closed:
                    continue
                if track.matched_in_frame:
                    continue
                delta = abs(bin_idx - track.bins[-1]) if track.bins else None
                if delta is None:
                    continue
                if delta <= self.bin_tolerance and (best_track is None or delta < best_delta):
                    best_track = track
                    best_delta = delta
            if best_track is not None:
                self._append_detection(best_track, bin_idx, current_frame)
            else:
                self._create_track(bin_idx, current_frame)

        # Update gap counters and close tracks as needed.
        active_tracks: List[dict] = []
        closed_tracks: List[dict] = []
        for track in self._tracks:
            if track.closed:
                continue
            if track.matched_in_frame:
                track.gap_count = 0
            else:
                track.gap_count += 1
            if track.gap_count > self.max_gap:
                track.closed = True
                snapshot = track.snapshot()
                self._closed_buffer.append(snapshot)
                closed_tracks.append(snapshot)
            else:
                active_tracks.append(track.snapshot())

        return active_tracks, closed_tracks

    def flush(self) -> List[dict]:
        """Return and clear tracks that have been closed so far."""
        flushed = list(self._closed_buffer)
        self._closed_buffer.clear()
        return flushed

    def _create_track(self, bin_idx: int, frame_idx: int) -> None:
        track = _TrackState(
            track_id=self._next_id,
            start_frame=frame_idx,
            last_frame=frame_idx,
        )
        self._next_id += 1
        self._tracks.append(track)
        self._append_detection(track, bin_idx, frame_idx)

    def _append_detection(self, track: _TrackState, bin_idx: int, frame_idx: int) -> None:
        freq_hz = self.bin_to_hz(bin_idx)
        track.bins.append(int(bin_idx))
        track.freqs.append(freq_hz)
        track.last_frame = frame_idx
        track.matched_in_frame = True
