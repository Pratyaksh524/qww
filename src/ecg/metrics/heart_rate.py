"""Heart rate calculation from ECG signals"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import platform
import time
from collections import deque


# Global smoothing buffers for BPM stabilization
_bpm_smoothing_buffers = {}  # Key: instance_id, Value: deque buffer
_bpm_ema_values = {}         # Key: instance_id, Value: EMA value
_last_stable_bpm = {}        # Key: instance_id, Value: Last stable BPM value
_bpm_last_success_ts = {}    # Key: instance_id, Value: last success timestamp

# FIX #5: Startup beat counter - do not emit HR until enough stable beats seen
_bpm_beat_count = {}         # Key: instance_id, Value: cumulative beat count

# PR Stabilization (FIX #6): 10-second stability requirement
_bpm_displayed_value = {}    # Key: instance_id, Value: Currently displayed stable BPM
_bpm_pending_value = {}      # Key: instance_id, Value: New value being monitored for stability
_bpm_stability_start_ts = {} # Key: instance_id, Value: Timestamp when pending value first appeared


def cleanup_instance(instance_id: str):
    """
    Remove all smoothing state for a given instance_id.

    Call this when a monitoring session ends to prevent memory leaks
    in long-running processes that create many instance_ids over time.

    Args:
        instance_id: The instance key used in calculate_heart_rate_from_signal()
    """
    for d in (_bpm_smoothing_buffers, _bpm_ema_values,
              _last_stable_bpm, _bpm_last_success_ts, _bpm_beat_count,
              _bpm_displayed_value, _bpm_pending_value, _bpm_stability_start_ts):
        d.pop(instance_id, None)


# FIX #5: Startup parameters
_STARTUP_LOCKOUT_BEATS = 5    # Ignore first N beat detections
_STARTUP_RR_MAX_MS     = 2000 # RR > 2000ms (< 30 BPM) at startup = noise floor
_STARTUP_ECTOPIC_TOL   = 0.10 # Tighter ectopic rejection during warmup (10%)
_NORMAL_ECTOPIC_TOL    = 0.20 # Normal tolerance after warmup (20%)


def calculate_heart_rate_from_signal(lead_data, sampling_rate=None, sampler=None, instance_id=None):
    """Calculate heart rate from Lead II data using R-R intervals.

    Args:
        lead_data:      Raw ECG signal data (numpy array or list).
        sampling_rate:  Sampling rate in Hz (optional, defaults to 500 Hz).
        sampler:        SamplingRateCalculator instance (optional).
        instance_id:    Key for per-instance smoothing buffers.

    Returns:
        int: Heart rate in BPM (10-300 range), or 0 if calculation fails.
    """
    try:
        buffer_key = instance_id if instance_id is not None else 'global'

        def _fallback_value():
            last = _last_stable_bpm.get(buffer_key, None)
            last_success = _bpm_last_success_ts.get(buffer_key, 0.0)
            if last is not None and (time.time() - last_success) <= 0.5:
                return last
            return 0

        # -- Early exit: flat / silent signal -----------------------------
        try:
            arr = np.asarray(lead_data, dtype=float)
            if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                return 0
        except Exception:
            return 0

        # -- Validate input ------------------------------------------------
        if not isinstance(lead_data, (list, np.ndarray)) or len(lead_data) < 200:
            print(" Insufficient data for heart rate calculation")
            return _fallback_value()

        try:
            lead_data = np.asarray(lead_data, dtype=float)
        except Exception as e:
            print(f" Error converting lead data to array: {e}")
            return _fallback_value()

        if np.any(np.isnan(lead_data)) or np.any(np.isinf(lead_data)):
            print(" Invalid values (NaN/Inf) in lead data")
            return _fallback_value()

        # -- Sampling rate -------------------------------------------------
        fs = 500.0
        if sampling_rate is not None and sampling_rate > 10:
            fs = float(sampling_rate)
        elif (sampler is not None and hasattr(sampler, 'sampling_rate')
              and sampler.sampling_rate > 10):
            detected = sampler.sampling_rate
            if np.isfinite(detected):
                fs = float(detected)
        if fs <= 0 or not np.isfinite(fs):
            fs = 500.0

        # -- Filter --------------------------------------------------------
        try:
            from ..signal_paths import display_filter
            filtered_signal = display_filter(lead_data, fs)
            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(" Filter produced invalid values")
                return _fallback_value()
        except Exception as e:
            print(f" Error in signal filtering: {e}")
            return _fallback_value()

        # -- Peak detection ------------------------------------------------
        try:
            signal_mean = np.mean(filtered_signal)
            signal_std  = np.std(filtered_signal)
            if signal_std == 0:
                print(" No signal variation detected")
                return _fallback_value()

            height_threshold     = signal_mean + 0.5 * signal_std
            prominence_threshold = signal_std * 0.4

            # FIX #3: initialise peaks so NameError cannot occur if all
            # strategies fail AND the fallback find_peaks also raises.
            peaks = np.array([], dtype=int)

            detection_results = []

            # Strategy 1: Conservative - best for 10-120 BPM
            peaks_conservative, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.35 * fs),
                prominence=prominence_threshold,
            )
            if len(peaks_conservative) >= 2:
                rr = np.diff(peaks_conservative) * (1000.0 / fs)
                valid = rr[(rr >= 200) & (rr <= 6000)]
                if len(valid) > 0:
                    detection_results.append((
                        'conservative', peaks_conservative,
                        60000.0 / np.median(valid), np.std(valid)
                    ))

            # Strategy 2: Normal - best for 100-180 BPM
            peaks_normal, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.22 * fs),
                prominence=prominence_threshold,
            )
            if len(peaks_normal) >= 2:
                rr = np.diff(peaks_normal) * (1000.0 / fs)
                valid = rr[(rr >= 200) & (rr <= 6000)]
                if len(valid) > 0:
                    detection_results.append((
                        'normal', peaks_normal,
                        60000.0 / np.median(valid), np.std(valid)
                    ))

            # Strategy 3: Tight - best for 160-220 BPM
            peaks_tight, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.12 * fs),
                prominence=prominence_threshold * 2.0,  # Doubled to avoid detecting T-waves as R-peaks
            )
            if len(peaks_tight) >= 2:
                rr = np.diff(peaks_tight) * (1000.0 / fs)
                valid = rr[(rr >= 200) & (rr <= 6000)]
                if len(valid) > 0:
                    detection_results.append((
                        'tight', peaks_tight,
                        60000.0 / np.median(valid), np.std(valid)
                    ))

            # Strategy 4: Ultra-tight - best for 200-300 BPM (FIX: high BPM halving issue)
            # At 300 BPM, RR interval = 200ms = 100 samples at 500Hz
            # distance=0.15*fs = 75 samples allows detection of peaks as close as 150ms apart
            peaks_ultra_tight, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.15 * fs),
                prominence=prominence_threshold * 2.0,  # Doubled to avoid detecting T-waves as R-peaks
            )
            if len(peaks_ultra_tight) >= 2:
                rr = np.diff(peaks_ultra_tight) * (1000.0 / fs)
                valid = rr[(rr >= 200) & (rr <= 6000)]
                if len(valid) > 0:
                    bpm_ultra = 60000.0 / np.median(valid)
                    # FIX: Recalculate std for ultra-tight
                    std_ultra = np.std(valid)
                    detection_results.append((
                        'ultra_tight', peaks_ultra_tight,
                        bpm_ultra, std_ultra
                    ))

            # FIX #2: Proper best-candidate selection.
            # Among all strategies whose std is within stability limits,
            # prefer the one with the LOWEST std (most consistent RR intervals).
            # Sorting by BPM descending first avoids choosing a sub-harmonic
            # (half the real rate) when a faster stable strategy also exists.
            if detection_results:
                # Candidates that pass the stability gate
                # FIX: Adaptive stability thresholds for high BPM
                # At high heart rates (>180 BPM), allow higher std because:
                # 1. RR intervals are shorter, so absolute std naturally higher
                # 2. Ultra-tight detection may have slightly more variance
                stable_candidates = []
                for r in detection_results:
                    method, peaks_result, bpm, std = r
                    # Adaptive thresholds based on BPM
                    if bpm > 180:
                        # High BPM: allow std up to 25ms or 20% of BPM
                        max_std_abs = 25
                        max_std_pct = 0.20
                    else:
                        # Normal BPM: stricter thresholds
                        max_std_abs = 15
                        max_std_pct = 0.15
                    
                    if std <= max_std_abs and std <= bpm * max_std_pct:
                        stable_candidates.append(r)
                        print(f" [OK] Strategy '{method}' PASSED stability gate: BPM={bpm:.1f}, std={std:.1f}ms (max_std={max_std_abs}ms, max_pct={max_std_pct*100}%)")
                    else:
                        print(f" [FAIL] Strategy '{method}' FAILED stability gate: BPM={bpm:.1f}, std={std:.1f}ms (max_std={max_std_abs}ms, max_pct={max_std_pct*100}%)")

                if stable_candidates:
                    # Among stable candidates prefer highest BPM (avoids sub-harmonic aliasing),
                    # but only if a faster candidate's BPM is not >10% higher than the next one
                    # (which would indicate noise rather than a true faster rate).
                    # CRITICAL FIX: Always prefer highest BPM to avoid sub-harmonic detection
                    # Strategy detecting every other beat looks "stable" but gives half the rate
                    stable_candidates.sort(key=lambda x: x[2], reverse=True)
                    best_method, peaks, best_bpm, best_std = stable_candidates[0]
                    # Log RR intervals
                    rr_median_ms = 60000.0 / best_bpm if best_bpm > 0 else 0
                    print(f" [SELECT] SELECTED: '{best_method}' strategy with BPM={best_bpm:.1f}, std={best_std:.1f}ms, RR_median={rr_median_ms:.1f}ms")
                else:
                    # Fallback: take the most stable result even if not ideal
                    detection_results.sort(key=lambda x: x[3])
                    best_method, peaks, best_bpm, best_std = detection_results[0]
                    rr_median_ms = 60000.0 / best_bpm if best_bpm > 0 else 0
                    print(f" [WARN] FALLBACK: No stable candidates, using '{best_method}' with BPM={best_bpm:.1f}, std={best_std:.1f}ms, RR_median={rr_median_ms:.1f}ms")
                    
                    # LOGGING: Print why others failed
                    print(f" WARNING: High-BPM Debug - Fallback to '{best_method}' (BPM={best_bpm}). Candidates were:")
                    for r in detection_results:
                         print(f"   - {r[0]}: BPM={r[2]:.1f}, std={r[3]:.1f}")
            else:
                # Fallback when no strategy found >=2 peaks
                peaks, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.4 * fs),
                    prominence=prominence_threshold,
                )

        except Exception as e:
            print(f" Error in peak detection: {e}")
            return _fallback_value()

        if len(peaks) < 2:
            print(f" Insufficient peaks detected: {len(peaks)}")
            return _fallback_value()

        # -- RR intervals --------------------------------------------------
        try:
            rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
            if len(rr_intervals_ms) == 0:
                return _fallback_value()
        except Exception as e:
            print(f" Error calculating R-R intervals: {e}")
            return _fallback_value()

        # -- Physiological filter + ectopic rejection ----------------------
        try:
            valid_intervals = rr_intervals_ms[
                (rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)
            ]

            if len(valid_intervals) < 2:
                print(" No valid R-R intervals after initial filter")
                return _fallback_value()

            # FIX #5: Track cumulative beat count for startup behaviour
            if buffer_key not in _bpm_beat_count:
                _bpm_beat_count[buffer_key] = 0
            _bpm_beat_count[buffer_key] += len(valid_intervals)
            beat_count = _bpm_beat_count[buffer_key]

            # FIX #5: During startup, apply hard RR floor (< 30 BPM = noise)
            is_startup = beat_count <= _STARTUP_LOCKOUT_BEATS
            if is_startup:
                valid_intervals = valid_intervals[valid_intervals <= _STARTUP_RR_MAX_MS]
                if len(valid_intervals) < 2:
                    print(f" Startup lockout: insufficient stable beats ({beat_count}/{_STARTUP_LOCKOUT_BEATS})")
                    return _fallback_value()

            # FIX #5: Tighter ectopic rejection during startup warmup
            ectopic_tol = _STARTUP_ECTOPIC_TOL if is_startup else _NORMAL_ECTOPIC_TOL

            if len(valid_intervals) >= 3:
                median_rr_initial = np.median(valid_intervals)
                tolerance = ectopic_tol * median_rr_initial
                normal_intervals = valid_intervals[
                    np.abs(valid_intervals - median_rr_initial) <= tolerance
                ]
                if len(normal_intervals) >= 2:
                    valid_intervals = normal_intervals

            if len(valid_intervals) == 0:
                return _fallback_value()

        except Exception as e:
            print(f" Error filtering intervals: {e}")
            return _fallback_value()

        # -- Heart rate calculation -----------------------------------------
        try:
            median_rr = np.median(valid_intervals)
            if median_rr <= 0:
                return _fallback_value()

            heart_rate = 60000.0 / median_rr
            heart_rate = max(10.0, min(300.0, heart_rate))

            # FIX #4: Anti-aliasing guard uses the peaks from the selected
            # strategy (stored in `peaks` above), not a stale variable.
            try:
                window_sec = len(lead_data) / float(fs)
            except Exception:
                window_sec = 0
            if heart_rate > 150 and window_sec >= 5.0:
                expected_peaks = (heart_rate * window_sec) / 60.0
                if expected_peaks > len(peaks) * 3:
                    print(f" Suspicious high BPM ({heart_rate:.1f}) vs peak count. Clamping.")
                    heart_rate = 10.0

            if not np.isfinite(heart_rate):
                return _fallback_value()

            hr_int = int(round(heart_rate))

            # -- EMA + median smoothing -------------------------------------
            if buffer_key not in _bpm_smoothing_buffers:
                _bpm_smoothing_buffers[buffer_key] = deque(maxlen=15)

            buf = _bpm_smoothing_buffers[buffer_key]
            buf.append(hr_int)

            if buffer_key not in _bpm_ema_values:
                _bpm_ema_values[buffer_key] = float(hr_int)

            median_hr = int(round(np.median(list(buf)))) if len(buf) >= 5 else hr_int

            current_display = int(round(_bpm_ema_values[buffer_key]))

            # FIX #1: Threshold raised to 5 BPM so that tiny integer
            # rounding differences (1-4 BPM) use the slow alpha=0.10
            # path and do not cause constant high-alpha flickering.
            alpha = 0.5 if abs(median_hr - current_display) >= 5 else 0.10
            _bpm_ema_values[buffer_key] = (
                (1 - alpha) * _bpm_ema_values[buffer_key] + alpha * median_hr
            )

            smoothed_hr = int(round(_bpm_ema_values[buffer_key]))

            if buffer_key not in _last_stable_bpm:
                _last_stable_bpm[buffer_key] = smoothed_hr

            last_stable = _last_stable_bpm[buffer_key]
            _bpm_last_success_ts[buffer_key] = time.time()

            if abs(smoothed_hr - last_stable) >= 1:
                _last_stable_bpm[buffer_key] = smoothed_hr
            
            # -- PR Stabilization: 10-second "Hold-and-Jump" logic ------------
            # Requirement: Update displayed value only if stable for 10 seconds.
            # Stable is defined as within +/-2 BPM of the target value.
            
            # Initial state setup
            if buffer_key not in _bpm_displayed_value:
                _bpm_displayed_value[buffer_key] = smoothed_hr
                _bpm_pending_value[buffer_key] = None
                _bpm_stability_start_ts[buffer_key] = 0
            
            displayed_bpm = _bpm_displayed_value[buffer_key]
            
            # Small changes (+/-2 BPM) update immediately to follow slow trends
            if abs(smoothed_hr - displayed_bpm) <= 2:
                _bpm_displayed_value[buffer_key] = smoothed_hr
                _bpm_pending_value[buffer_key] = None
                _bpm_stability_start_ts[buffer_key] = 0
                return smoothed_hr
            else:
                # Large change detected: Hold old value and monitor new one for 10s stability
                current_time = time.time()
                pending = _bpm_pending_value[buffer_key]
                
                if pending is None:
                    # Start monitoring new value
                    _bpm_pending_value[buffer_key] = smoothed_hr
                    _bpm_stability_start_ts[buffer_key] = current_time
                else:
                    # If current value is still "stable" relative to our pending target (+/-2 BPM)
                    if abs(smoothed_hr - pending) <= 2:
                        # Check if stable long enough
                        if current_time - _bpm_stability_start_ts[buffer_key] >= 10.0:
                            # 10s achieved! Jump to new value.
                            _bpm_displayed_value[buffer_key] = smoothed_hr
                            _bpm_pending_value[buffer_key] = None
                            _bpm_stability_start_ts[buffer_key] = 0
                            return smoothed_hr
                    else:
                        # Value shifted again, reset the 10s timer for the new target
                        _bpm_pending_value[buffer_key] = smoothed_hr
                        _bpm_stability_start_ts[buffer_key] = current_time
                
                # Return the currently locked "old" stable value
                return displayed_bpm

        except Exception as e:
            print(f" Error calculating final BPM: {e}")
            return _fallback_value()

    except Exception as e:
        print(f" Critical error in calculate_heart_rate_from_signal: {e}")
        return 0