"""Offline episode contact labels for supervising a causal phase predictor."""

import numpy as np


def contact_phase_alpha(timestamps, contact, episode_ranges, precontact_seconds):
    """Label every contact onset with a preceding linear alpha ramp.

    Future contact times are used only to construct training targets. Inputs
    to the policy and renderer remain the recorded current observations.
    Phase IDs are 0=free, 1=pre-contact, 2=contact. The left ramp endpoint
    belongs to pre-contact with alpha zero; confirmed contact has alpha one.
    """
    duration = float(precontact_seconds)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("precontact_seconds must be finite and non-negative")
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    contact = np.asarray(contact, dtype=bool).reshape(-1)
    if times.shape != contact.shape or not np.isfinite(times).all():
        raise ValueError("Contact labels require one finite timestamp per frame")
    phase = np.where(contact, 2, 0).astype(np.int64)
    alpha = contact.astype(np.float32)
    if duration == 0:
        return phase, alpha
    for rows in episode_ranges:
        start, stop = int(rows.start), int(rows.stop)
        local_contact = contact[start:stop]
        if not len(local_contact):
            continue
        local_time = times[start:stop] - times[start]
        onsets = np.flatnonzero(local_contact & ~np.r_[False, local_contact[:-1]])
        for onset in onsets:
            remaining = local_time[onset] - local_time
            # Recorded float32 seconds can move an exact boundary by a few
            # ulps. Preserve that endpoint without rounding the ramp duration.
            precontact = (~local_contact) & (remaining > 0) & (remaining <= duration + 1e-6)
            ramp = np.clip(1.0 - remaining[precontact] / duration, 0.0, 1.0).astype(np.float32)
            local_phase = phase[start:stop]
            local_alpha = alpha[start:stop]
            local_phase[precontact] = 1
            # If two short contact bouts overlap in their preceding windows,
            # the nearest upcoming onset determines each free frame's ramp.
            local_alpha[precontact] = np.maximum(local_alpha[precontact], ramp)
    return phase, alpha
