"""ECG metrics display update functions — unified & corrected.

Changes vs previous version
────────────────────────────
FIX-D1: Added 'rr_interval' label support — RR was calculated but never rendered.
FIX-D2: Added 'p_duration' label support — p_duration was passed but silently dropped
         because the key was never in metric_labels.
FIX-D3: QT/QTc display format now robust — shows "QT/QTc" or just "QTc" if QT missing,
         and never crashes on None values.
FIX-D4: get_current_metrics_from_labels now returns 'rr_interval' and 'qt_interval'
         keys so dashboard / report code receives complete data.
FIX-D5: Throttle remains 0.3 s; force_immediate path properly resets last_update_ts=0
         before calling (handled in twelve_lead_test.py wrapper, unchanged here).
"""

import time
from typing import Dict, Optional


def update_ecg_metrics_display(
        metric_labels: Dict,
        heart_rate: int,
        pr_interval: int,
        qrs_duration: int,
        p_duration: int,
        qt_interval: Optional[float] = None,
        qtc_interval: Optional[int] = None,
        qtcf_interval: Optional[int] = None,
        last_update_ts: Optional[float] = None,
        rr_interval: Optional[float] = None,       # FIX-D1: new param
        skip_heart_rate: bool = False,              # HolterBPM: bypass old HR path
) -> float:
    """Update the ECG metrics display in the UI.

    Supported metric_labels keys:
        'heart_rate'   – BPM (int)
        'rr_interval'  – RR interval in ms (float)   ← FIX-D1
        'pr_interval'  – PR interval in ms (int)
        'qrs_duration' – QRS duration in ms (int)
        'p_duration'   – P-wave duration in ms (int) ← FIX-D2
        'qtc_interval' – Shows "QT/QTc" or "QTc" text
        'time_elapsed' – Timer (updated separately, not touched here)

    Returns:
        Updated timestamp (float).
    """
    try:
        current_time = time.time()
        # Throttle: max one display refresh per 0.3 s
        # last_update_ts=None (or 0.0) means "force now"
        if last_update_ts and current_time - last_update_ts < 0.3:
            return last_update_ts

        if not metric_labels:
            return current_time

        # ── BPM ──────────────────────────────────────────────────────────────
        # skip_heart_rate=True → controlled exclusively by HolterBPMController
        if not skip_heart_rate:
            if 'heart_rate' in metric_labels:
                hr_val = int(round(heart_rate)) if isinstance(heart_rate, (int, float)) else 0
                metric_labels['heart_rate'].setText(f"{hr_val:3d}")

        # ── RR Interval ───────────────────────────────────────────────────── FIX-D1
        if 'rr_interval' in metric_labels:
            if rr_interval is not None and rr_interval > 0:
                rr_val = int(round(rr_interval))
                metric_labels['rr_interval'].setText(f"{rr_val}")
            else:
                metric_labels['rr_interval'].setText("--")

        # ── PR Interval ───────────────────────────────────────────────────────
        if 'pr_interval' in metric_labels:
            pr_val = int(round(pr_interval)) if isinstance(pr_interval, (int, float)) else 0
            if pr_val > 0:
                metric_labels['pr_interval'].setText(f"{pr_val:3d}")
            else:
                metric_labels['pr_interval'].setText("  0")

        # ── QRS Duration ──────────────────────────────────────────────────────
        if 'qrs_duration' in metric_labels:
            qrs_val = int(round(qrs_duration)) if isinstance(qrs_duration, (int, float)) else 0
            if qrs_val > 0:
                metric_labels['qrs_duration'].setText(f"{qrs_val:3d}")
            else:
                metric_labels['qrs_duration'].setText("  0")

        # ── P Duration ────────────────────────────────────────────────────── FIX-D2
        if 'p_duration' in metric_labels:
            if isinstance(p_duration, (int, float)) and p_duration > 0:
                metric_labels['p_duration'].setText(f"{int(round(p_duration))}")
            else:
                metric_labels['p_duration'].setText("--")

        # ── ST (legacy key — keep at 0, ST elevation is separate) ────────────
        if 'st_interval' in metric_labels:
            metric_labels['st_interval'].setText("0")

        # ── QT / QTc ─────────────────────────────────────────────────────── FIX-D3
        if 'qtc_interval' in metric_labels:
            parts = []
            qt_ok  = qt_interval  is not None and isinstance(qt_interval,  (int, float)) and qt_interval  > 0
            qtc_ok = qtc_interval is not None and isinstance(qtc_interval, (int, float)) and qtc_interval > 0

            if qt_ok:
                parts.append(f"{int(round(qt_interval))}")
            if qtc_ok:
                parts.append(f"{int(round(qtc_interval))}")

            display_text = "/".join(parts) if parts else "0"
            metric_labels['qtc_interval'].setText(display_text)

        return current_time

    except Exception as e:
        print(f" ⚠️ update_ecg_metrics_display error: {e}")
        return last_update_ts if last_update_ts else time.time()


def get_current_metrics_from_labels(
        metric_labels: Dict,
        data: list,
        last_heart_rate: Optional[int] = None,
        sampler=None,
) -> Dict[str, str]:
    """Get current ECG metrics for dashboard / report from UI labels.

    FIX-D4: Returns 'rr_interval' and 'qt_interval' keys that were
             previously missing — dashboard and report code now receive
             the complete metric set.

    Returns:
        Dict[str, str] — all values as strings (empty string = not available).
    """
    try:
        metrics: Dict[str, str] = {}

        # ── Signal quality check ──────────────────────────────────────────────
        has_real_signal = False
        if len(data) > 1:
            import numpy as np
            lead_ii = data[1]
            if (len(lead_ii) >= 100
                    and not np.all(lead_ii == 0)
                    and np.std(lead_ii) >= 0.1):
                has_real_signal = True

        # ── BPM ──────────────────────────────────────────────────────────────
        if metric_labels and 'heart_rate' in metric_labels:
            hr_text = (metric_labels['heart_rate'].text()
                       .replace('BPM', '').replace('bpm', '').strip())
            if hr_text and hr_text not in ('00', '--', '0', ''):
                metrics['heart_rate'] = hr_text
            elif has_real_signal and last_heart_rate and last_heart_rate > 0:
                metrics['heart_rate'] = str(last_heart_rate)
            else:
                metrics['heart_rate'] = "0"
        elif has_real_signal and last_heart_rate and last_heart_rate > 0:
            metrics['heart_rate'] = str(last_heart_rate)
        else:
            metrics['heart_rate'] = "0"

        if not metric_labels:
            return metrics

        # ── RR Interval ───────────────────────────────────────────────────── FIX-D4
        if 'rr_interval' in metric_labels:
            metrics['rr_interval'] = (
                metric_labels['rr_interval'].text()
                .replace('ms', '').strip()
            )
        else:
            metrics['rr_interval'] = ""

        # ── PR Interval ───────────────────────────────────────────────────────
        if 'pr_interval' in metric_labels:
            metrics['pr_interval'] = (
                metric_labels['pr_interval'].text()
                .replace('ms', '').strip()
            )

        # ── QRS Duration ──────────────────────────────────────────────────────
        if 'qrs_duration' in metric_labels:
            metrics['qrs_duration'] = (
                metric_labels['qrs_duration'].text()
                .replace('ms', '').strip()
            )

        # ── P Duration ────────────────────────────────────────────────────────
        if 'p_duration' in metric_labels:
            metrics['p_duration'] = (
                metric_labels['p_duration'].text()
                .replace('ms', '').strip()
            )

        # ── ST (legacy) ───────────────────────────────────────────────────────
        if 'st_interval' in metric_labels:
            metrics['st_interval'] = (
                metric_labels['st_interval'].text().strip()
                .replace('ms', '').replace('mV', '').strip()
            )

        # ── QT / QTc ─────────────────────────────────────────────────────────
        if 'qtc_interval' in metric_labels:
            raw = metric_labels['qtc_interval'].text().strip().replace('ms', '')
            metrics['qtc_interval'] = raw
            # FIX-D4: also split into qt_interval / qtc_interval if "QT/QTc" format
            if '/' in raw:
                parts = raw.split('/')
                metrics['qt_interval']  = parts[0].strip()
                metrics['qtc_interval'] = parts[1].strip()
            else:
                metrics['qt_interval'] = ""

        # ── Time elapsed ──────────────────────────────────────────────────────
        if 'time_elapsed' in metric_labels:
            metrics['time_elapsed'] = metric_labels['time_elapsed'].text()

        # ── Sampling rate ─────────────────────────────────────────────────────
        if sampler and hasattr(sampler, 'sampling_rate') and sampler.sampling_rate > 0:
            metrics['sampling_rate'] = f"{sampler.sampling_rate:.1f}"
        else:
            metrics['sampling_rate'] = "--"

        return metrics

    except Exception as e:
        print(f" ⚠️ get_current_metrics_from_labels error: {e}")
        return {}