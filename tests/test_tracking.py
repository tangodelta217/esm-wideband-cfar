"""Tests for the PeakTracker implementation."""

import pytest

from src.cfar.tracking import PeakTracker


@pytest.fixture
def tracker() -> PeakTracker:
    """Default tracker for a 2-MHz capture with 1024-point FFTs."""
    return PeakTracker(sample_rate_hz=2e6, fft_size=1024, max_gap=2, bin_tolerance=2)


def test_tracker_freq_conversion(tracker: PeakTracker):
    """Verify that FFT bin to frequency conversion is correct for a centered spectrum."""
    assert tracker.bin_to_hz(0) == -1e6
    assert tracker.bin_to_hz(512) == 0.0
    assert tracker.bin_to_hz(1023) == (1023 / 1024 - 0.5) * 2e6
    assert tracker.bin_to_hz(1024) == 1e6


def test_tracker_basic_tracking(tracker: PeakTracker):
    """Test basic tracking of a single, continuous signal without gaps."""
    detections_per_frame = [[512], [513], [512], [514]]
    for frame_idx, detections in enumerate(detections_per_frame):
        active, closed = tracker.update(detections)
        assert not closed
        assert len(active) == 1
        assert active[0]["id"] == 1
        assert active[0]["length"] == frame_idx + 1
        assert active[0]["last_frame"] == frame_idx

    # Final state
    final_active, final_closed = tracker.update([])
    assert not final_closed
    assert len(final_active) == 1
    assert final_active[0]["bins"] == [512, 513, 512, 514]


def test_tracker_with_gaps_and_track_closing(tracker: PeakTracker):
    """Test that a track is maintained through gaps shorter than max_gap and closed otherwise."""
    # Frame 0: New track
    tracker.update([100])
    # Frame 1: Gap
    tracker.update([])
    # Frame 2: Gap
    active, closed = tracker.update([])
    assert not closed
    assert len(active) == 1
    assert active[0]["gap_count"] == 2

    # Frame 3: Gap exceeds max_gap, track should be closed
    active, closed = tracker.update([])
    assert not active
    assert len(closed) == 1
    assert closed[0]["id"] == 1
    assert closed[0]["length"] == 1
    assert closed[0]["last_frame"] == 0

    # Frame 4: A new detection should create a new track
    active, closed = tracker.update([100])
    assert not closed
    assert len(active) == 1
    assert active[0]["id"] == 2  # New ID


def test_tracker_multiple_targets(tracker: PeakTracker):
    """Test the tracker's ability to create and update multiple simultaneous tracks."""
    tracker.update([100, 800])
    active, _ = tracker.update([101, 802, 500])

    assert len(active) == 3
    ids = {t["id"] for t in active}
    assert ids == {1, 2, 3}

    # Check that tracks were updated correctly
    track1 = next(t for t in active if t["id"] == 1)
    track2 = next(t for t in active if t["id"] == 2)
    track3 = next(t for t in active if t["id"] == 3)

    assert track1["bins"] == [100, 101]
    assert track2["bins"] == [800, 802]
    assert track3["bins"] == [500]


def test_tracker_flush_clears_buffer(tracker: PeakTracker):
    """Verify that flush() returns closed tracks and clears the internal buffer."""
    tracker.update([100])
    tracker.update([])
    tracker.update([])
    tracker.update([])  # Close track 1

    assert len(tracker._closed_buffer) == 1
    flushed = tracker.flush()
    assert len(flushed) == 1
    assert flushed[0]["id"] == 1
    assert not tracker._closed_buffer

    # A second flush should return nothing
    assert not tracker.flush()
