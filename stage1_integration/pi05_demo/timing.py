"""Host event ordering and sampling checks; device clocks remain separate."""
import numpy as np

TIMING_FIELDS = (
    "state_bridge_receive_s", "state_client_receive_s", "gripper_bridge_read_s",
    "gello_read_s", "command_send_s", "command_enqueue_s", "command_ack_s",
)


def validate_timing(timing, sample_time):
    values = np.asarray([timing[key] for key in TIMING_FIELDS], float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite host event time")
    state, received, gripper, gello, sent, enqueued, ack = values
    if not state <= received <= gello <= sample_time or gripper > received:
        raise ValueError("invalid observation event time order")
    if not sample_time <= sent <= enqueued <= ack:
        raise ValueError("invalid command event time order")


def validate_camera_metadata(metadata, received, sample_time):
    stamp = float(metadata["receive_monotonic_s"])
    ready = float(metadata["ready_monotonic_s"])
    if not np.isfinite([stamp, ready]).all() or abs(stamp - received) > 1e-9:
        raise ValueError("camera receive time mismatch")
    if not stamp <= ready <= sample_time:
        raise ValueError("invalid camera receive/ready time order")
    source = metadata.get("source_time_s")
    if source is not None and (not np.isfinite(source) or source < 0):
        raise ValueError("invalid camera device frame time")
    if not isinstance(metadata["source_clock"], str) or not metadata["source_clock"]:
        raise ValueError("camera source clock must be named")
    frame_id = metadata.get("frame_id", -1)
    if not isinstance(frame_id, (int, np.integer)) or frame_id < -1:
        raise ValueError("invalid camera frame id")


def distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return {}
    return {key: float(value) for key, value in zip(
        ("min", "p50", "p95", "p99", "max"), np.quantile(values, [0, .5, .95, .99, 1]))}


def audit_timing(episode, fps):
    """Fixed-FPS training must not silently turn very irregular samples into a grid."""
    times = np.asarray(episode["timestamp"], float)
    dt = np.diff(times)
    if not np.isfinite(times).all() or np.any(dt <= 0):
        raise ValueError("invalid episode timing")
    period = 1. / fps
    # A provisional data-quality bound, not an exposure-sync or hardware guarantee.
    if len(dt) and np.any(np.abs(dt - period) > period * .5):
        raise ValueError("irregular control timing: interval differs from nominal FPS by over 50%")
    grid_error = times - times[0] - np.arange(len(times)) * period if len(times) else np.array([])
    if len(times) and np.max(np.abs(grid_error)) > period:
        raise ValueError("control timing drifts by more than one frame from fixed FPS")
    report = {"timing_version": int(episode.attrs.get("timing_version", 1)),
              "interval_ms": distribution(dt * 1000),
              "max_grid_error_ms": float(np.max(np.abs(grid_error)) * 1000) if len(times) else 0.,
              "exposure_alignment_verified": False, "cameras": {}}
    for key in ("main", "wrist"):
        receive = np.asarray(episode[f"camera_time/{key}"])
        details = {"host_receive_age_ms": distribution((times - receive) * 1000),
                   "repeated_receive_timestamp_count": int(np.count_nonzero(np.diff(receive) == 0)),
                   "source_clock": episode.attrs.get(f"camera_source_clock/{key}", "unavailable")}
        if f"camera_frame_id/{key}" in episode:
            ids = np.asarray(episode[f"camera_frame_id/{key}"])
            details["repeated_frame_id_count"] = int(np.count_nonzero((ids[1:] >= 0) & (np.diff(ids) == 0)))
            details["missing_frame_id_count"] = int(np.count_nonzero(ids < 0))
        report["cameras"][key] = details
    report["host_receive_skew_ms"] = distribution(
        np.abs(np.asarray(episode["camera_time/main"]) - np.asarray(episode["camera_time/wrist"])) * 1000)
    if report["timing_version"] == 2:
        report["command_ack_round_trip_ms"] = distribution(
            (np.asarray(episode["timing/command_ack_s"]) - np.asarray(episode["timing/command_send_s"])) * 1000)
    return report
