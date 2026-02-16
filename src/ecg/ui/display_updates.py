"""ECG metrics display update functions"""
import time
from typing import Dict, Optional


def update_ecg_metrics_display(metric_labels: Dict, heart_rate: int, pr_interval: int, 
                               qrs_duration: int, p_duration: int, 
                               qt_interval: Optional[int] = None, 
                               qtc_interval: Optional[int] = None, 
                               qtcf_interval: Optional[int] = None,
                               last_update_ts: Optional[float] = None) -> float:
    """Update the ECG metrics display in the UI (dashboard: BPM, PR, QRS, P, QT/QTc, timer only)
    
    Args:
        metric_labels: Dictionary of metric label widgets
        heart_rate: Heart rate in BPM
        pr_interval: PR interval in ms
        qrs_duration: QRS duration in ms
        p_duration: P-wave duration in ms
        qt_interval: QT interval in ms (optional)
        qtc_interval: QTc interval in ms (optional)
        qtcf_interval: QTcF interval in ms (optional)
        last_update_ts: Last update timestamp (for throttling)
    
    Returns:
        Updated timestamp
    """
    try:
        # Throttle updates to every 0.3 seconds for faster updates (within 10 seconds requirement)
        # Allow immediate update on first call (when last_update_ts is None)
        current_time = time.time()
        if last_update_ts is not None and current_time - last_update_ts < 0.3:
            return last_update_ts
        
        if metric_labels:
            if 'heart_rate' in metric_labels:
                # Fixed-width formatting (3 digits) to prevent text shifting
                metric_labels['heart_rate'].setText(f"{heart_rate:3d}")
            if 'pr_interval' in metric_labels:
                # Round to integer and use fixed-width formatting (3 digits) to prevent text shifting
                pr_val = int(round(pr_interval)) if isinstance(pr_interval, (int, float)) else pr_interval
                metric_labels['pr_interval'].setText(f"{pr_val:3d}")
            if 'qrs_duration' in metric_labels:
                # Round to integer and use fixed-width formatting (2 digits) to prevent text shifting
                qrs_val = int(round(qrs_duration)) if isinstance(qrs_duration, (int, float)) else qrs_duration
                metric_labels['qrs_duration'].setText(f"{qrs_val:2d}")
            if 'st_interval' in metric_labels:
                # ST Interval (now separate from P duration)
                metric_labels['st_interval'].setText("0") # Default to 0 until ST calc is restored
            
            if 'p_duration' in metric_labels:
                # P-wave duration
                if isinstance(p_duration, (int, float)) and p_duration > 0:
                    p_val = int(round(p_duration))
                    metric_labels['p_duration'].setText(f"{p_val}")
                else:
                    metric_labels['p_duration'].setText("0")
            if 'qtc_interval' in metric_labels:
                # Display only QT/QTc on live pages as requested
                display_text = ""
                if qt_interval is not None and qt_interval > 0:
                    display_text += f"{int(round(qt_interval))}"
                
                if qtc_interval is not None and qtc_interval > 0:
                    if display_text: display_text += "/"
                    display_text += f"{int(round(qtc_interval))}"
                
                # QTcF removed from live page display but kept for reports
                
                if display_text:
                    metric_labels['qtc_interval'].setText(display_text)
                else:
                    metric_labels['qtc_interval'].setText("0")
        
        return current_time
    except Exception as e:
        print(f"Error updating ECG metrics: {e}")
        return last_update_ts if last_update_ts else time.time()


def get_current_metrics_from_labels(metric_labels: Dict, data: list, 
                                    last_heart_rate: Optional[int] = None,
                                    sampler=None) -> Dict[str, str]:
    """Get current ECG metrics for dashboard display from UI labels
    
    Args:
        metric_labels: Dictionary of metric label widgets
        data: ECG data buffers (for signal validation)
        last_heart_rate: Last calculated heart rate (fallback)
        sampler: SamplingRateCalculator instance
    
    Returns:
        Dictionary of metric values as strings
    """
    try:
        metrics = {}
        
        # Check if we have real signal data
        has_real_signal = False
        if len(data) > 1:  # Lead II data available
            import numpy as np
            lead_ii_data = data[1]
            if len(lead_ii_data) >= 100 and not np.all(lead_ii_data == 0) and np.std(lead_ii_data) >= 0.1:
                has_real_signal = True
        
        # Get current heart rate - use same value displayed on 12-lead test page (unsmoothed)
        # Priority: Get from metric_labels (what's actually displayed on 12-lead page)
        if metric_labels and 'heart_rate' in metric_labels:
            hr_text = metric_labels['heart_rate'].text().strip()
            # Extract numeric value (remove "BPM" or "bpm" suffix and spaces)
            hr_text = hr_text.replace(' BPM', '').replace(' bpm', '').replace('BPM', '').replace('bpm', '').strip()
            if hr_text and hr_text not in ('00', '--', '0', ''):
                metrics['heart_rate'] = hr_text
            elif has_real_signal and last_heart_rate and last_heart_rate > 0:
                metrics['heart_rate'] = f"{last_heart_rate}"
            else:
                metrics['heart_rate'] = "0"
        elif has_real_signal and last_heart_rate and last_heart_rate > 0:
            metrics['heart_rate'] = f"{last_heart_rate}"
        else:
            metrics['heart_rate'] = "0"
        
        # Get other metrics from UI labels (these should be zero if reset properly)
        if metric_labels:
            if 'pr_interval' in metric_labels:
                metrics['pr_interval'] = metric_labels['pr_interval'].text().replace(' ms', '')
            if 'qrs_duration' in metric_labels:
                metrics['qrs_duration'] = metric_labels['qrs_duration'].text().replace(' ms', '')
            if 'st_interval' in metric_labels:
                metrics['st_interval'] = metric_labels['st_interval'].text().strip().replace(' ms', '').replace(' mV', '').replace(' mV mV', '')
            if 'qtc_interval' in metric_labels:
                metrics['qtc_interval'] = metric_labels['qtc_interval'].text().strip().replace(' ms', '')
            if 'time_elapsed' in metric_labels:
                metrics['time_elapsed'] = metric_labels['time_elapsed'].text()
        
        # Get sampling rate
        if sampler and hasattr(sampler, 'sampling_rate') and sampler.sampling_rate > 0:
            metrics['sampling_rate'] = f"{sampler.sampling_rate:.1f}"
        else:
            metrics['sampling_rate'] = "--"
        
        return metrics
    except Exception as e:
        print(f"Error getting current metrics: {e}")
        return {}
