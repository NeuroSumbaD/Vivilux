"""Finite capture of 5000 samples while toggling a phase shifter every ~1000 samples.

Captures exactly 5000 samples per channel at the user-specified rate and toggles
the phase shifter DAC output via the netlist interface after each 1000-sample
block (at ~1000, ~2000, ~3000, ~4000). Saves data and plots for inspection.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt


import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader

from board_config_6x6_v2 import netlist
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finite capture of 5000 samples with phase-shifter toggling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="Dev1", help="NI-DAQ device name")
    parser.add_argument("--channels", nargs="+", type=int, default=[0], help="AI channels to record")
    parser.add_argument("--rate", type=int, default=250_000, help="Sample rate (Hz)")
    parser.add_argument("--outfile", default="phase_shift_5000samp_capture.npz", help="Output file (.npz)")
    parser.add_argument("--maxval", type=float, default=1.3, help="AI max expected voltage (V)")
    parser.add_argument("--terminal", default="RSE", choices=["RSE", "NRSE", "DIFF"], help="AI terminal config")
    parser.add_argument("--port", default="COM5", help="Serial port for laser controller")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--stabilize", type=float, default=2.0, help="Pre-capture wait for lasers to stabilize (s)")
    parser.add_argument("--ps-pin", type=str, default="3_1_i", help="Netlist pin name for phase shifter control")
    parser.add_argument("--ps-voltage", type=float, default=3.0, help="Phase shifter ON voltage (V)")
    args = parser.parse_args()

    TOTAL_SAMPLES = 5000
    BLOCK = 1000  # toggle after each block (at 1000, 2000, 3000, 4000)

    channels = args.channels
    num_ch = len(channels)
    rate = int(args.rate)

    print("Initializing DAC and waiting to stabilize...")
    time.sleep(0.1)

    # ----------------------- NI-DAQmx task setup --------------------------
    term_map = {
        "RSE": TerminalConfiguration.RSE,
        "NRSE": TerminalConfiguration.NRSE,
        "DIFF": TerminalConfiguration.DIFF,
    }
    term_cfg = term_map[args.terminal.upper()]

    # Use the netlist context manager for proper setup/teardown
    with netlist:
        # Initialize phase shifter to 0 V (OFF) and allow to stabilize
        try:
            netlist[args.ps_pin].vout(0.0)
        except Exception as e:
            print(f"Warning: failed to initialize {args.ps_pin} to 0 V: {e}")

        if args.stabilize > 0:
            time.sleep(args.stabilize)

        task = nidaqmx.Task()
        try:
            for ch in channels:
                task.ai_channels.add_ai_voltage_chan(
                    f"{args.device}/ai{ch}",
                    min_val=0.0,
                    max_val=float(args.maxval),
                    terminal_config=term_cfg,
                )

            # Ensure the input buffer can hold the whole record so we can read in blocks safely
            try:
                task.in_stream.input_buf_size = TOTAL_SAMPLES
            except Exception:
                # Some devices clamp this; as long as we read promptly in small blocks it is fine
                pass

            task.timing.cfg_samp_clk_timing(
                rate=rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=TOTAL_SAMPLES,
            )

            reader = AnalogMultiChannelReader(task.in_stream)

            # Preallocate storage
            data = np.empty((num_ch, TOTAL_SAMPLES), dtype=np.float64)

            # Track where we toggled (sample indices based on block boundaries)
            toggle_indices = []
            # Track host-side timing: command issued and vout() returned
            host_toggle_cmd_indices = []
            host_toggle_done_indices = []

            # Start acquisition
            print(f"Starting finite acquisition: {TOTAL_SAMPLES} samples/ch @ {rate} Hz")
            task.start()
            t0 = time.perf_counter()  # host reference time near start of sampling

            # Read in blocks of 1000 samples per channel and toggle after each block (except after the last)
            current_on = False  # Phase shifter state: False=0V, True=ON voltage
            for i in range(TOTAL_SAMPLES // BLOCK):
                start = i * BLOCK
                end = start + BLOCK
                tmp = np.empty((num_ch, BLOCK), dtype=np.float64)
                # Blocking read of exactly BLOCK samples per channel
                reader.read_many_sample(tmp, number_of_samples_per_channel=BLOCK, timeout=2.0)
                data[:, start:end] = tmp

                # Toggle after completing blocks 0..3 (not after the final block)
                if i < (TOTAL_SAMPLES // BLOCK) - 1:
                    # Record host timing around the toggle
                    t_cmd = time.perf_counter()
                    current_on = not current_on
                    level = args.ps_voltage if current_on else 0.0
                    try:
                        netlist[args.ps_pin].vout(level)
                    except Exception as e:
                        print(f"Warning: failed to set {args.ps_pin} to {level} V: {e}")
                    t_done = time.perf_counter()

                    toggle_indices.append(end)  # commanded at block boundary (idealized)
                    # Convert host timestamps to sample indices
                    host_toggle_cmd_indices.append(int(round((t_cmd - t0) * rate)))
                    host_toggle_done_indices.append(int(round((t_done - t0) * rate)))

            # Make sure task completed
            task.wait_until_done(timeout=5.0)
            print("Acquisition complete. Saving and plotting...")

        except nidaqmx.errors.DaqError as e:
            print(f"DAQ Error: {e}")
            return
        finally:
            try:
                task.stop()
                task.close()
            except Exception:
                pass

            # Return phase shifter to 0 V before closing context
            try:
                netlist[args.ps_pin].vout(0.0)
            except Exception:
                pass

    # -------------------------- Save & plot -------------------------------
    t = np.arange(TOTAL_SAMPLES) / rate
    np.savez(
        args.outfile,
        data=data,
        time=t,
        channels=np.array(channels, dtype=int),
        rate=rate,
        ps_pin=args.ps_pin,
        ps_voltage=float(args.ps_voltage),
        toggles=np.array(toggle_indices, dtype=int),
        host_toggle_cmd=np.array(host_toggle_cmd_indices, dtype=int),
        host_toggle_done=np.array(host_toggle_done_indices, dtype=int),
    )
    print(f"Saved capture to {args.outfile}")

    # Print latency stats if available
    if len(toggle_indices) and len(host_toggle_cmd_indices) == len(toggle_indices):
        ti = np.array(toggle_indices)
        hc = np.array(host_toggle_cmd_indices)
        hd = np.array(host_toggle_done_indices)
        cmd_lat_samp = hc - ti
        done_lat_samp = hd - ti
        cmd_lat_ms = (cmd_lat_samp / rate) * 1000.0
        done_lat_ms = (done_lat_samp / rate) * 1000.0
        def fmt_stats(a: np.ndarray) -> str:
            return f"min/med/max = {np.min(a):.3f}/{np.median(a):.3f}/{np.max(a):.3f}"
        print("Host command latency relative to block boundary:")
        print(f"  samples: {fmt_stats(cmd_lat_samp)}")
        print(f"  ms:      {fmt_stats(cmd_lat_ms)}")
        print("Host done (vout returned) latency relative to block boundary:")
        print(f"  samples: {fmt_stats(done_lat_samp)}")
        print(f"  ms:      {fmt_stats(done_lat_ms)}")

    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(channels):
        plt.plot(t, data[i], label=f"ai{ch}")
    # Add red dashed vertical lines at ideal block-boundary toggle times
    if len(toggle_indices) > 0:
        toggle_times = np.array(toggle_indices, dtype=float) / rate
        for j, tt in enumerate(toggle_times):
            if j == 0:
                plt.axvline(tt, color="red", linestyle="--", alpha=0.8, label="toggle")
            else:
                plt.axvline(tt, color="red", linestyle="--", alpha=0.8)
    # Add host-side measured timing markers
    if 'host_toggle_cmd_indices' in locals() and len(host_toggle_cmd_indices) > 0:
        cmd_times = np.array(host_toggle_cmd_indices, dtype=float) / rate
        for j, tt in enumerate(cmd_times):
            if j == 0:
                plt.axvline(tt, color="orange", linestyle=":", alpha=0.9, label="host cmd")
            else:
                plt.axvline(tt, color="orange", linestyle=":", alpha=0.9)
    if 'host_toggle_done_indices' in locals() and len(host_toggle_done_indices) > 0:
        done_times = np.array(host_toggle_done_indices, dtype=float) / rate
        for j, tt in enumerate(done_times):
            if j == 0:
                plt.axvline(tt, color="green", linestyle=":", alpha=0.9, label="host done")
            else:
                plt.axvline(tt, color="green", linestyle=":", alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Phase shift toggling every ~{BLOCK} samples @ {rate} Hz")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
