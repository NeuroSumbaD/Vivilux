"""Finite capture of 5000 samples while toggling lasers every ~1000 samples.

This version performs a finite acquisition of exactly 5000 samples per channel
at the user-specified sample rate. During the acquisition it toggles the lasers
after roughly each 1000-sample block (i.e., at ~1000, ~2000, ~3000, ~4000).
Data are saved to disk and plotted at the end.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import serial

import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finite capture of 5000 samples with laser toggling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="Dev1", help="NI-DAQ device name")
    parser.add_argument("--channels", nargs="+", type=int, default=[0], help="AI channels to record")
    parser.add_argument("--rate", type=int, default=250_000, help="Sample rate (Hz)")
    parser.add_argument("--outfile", default="lasers_5000samp_capture.npz", help="Output file (.npz)")
    parser.add_argument("--maxval", type=float, default=1.3, help="AI max expected voltage (V)")
    parser.add_argument("--terminal", default="RSE", choices=["RSE", "NRSE", "DIFF"], help="AI terminal config")
    parser.add_argument("--port", default="COM5", help="Serial port for laser controller")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--stabilize", type=float, default=2.0, help="Pre-capture wait for lasers to stabilize (s)")
    args = parser.parse_args()

    TOTAL_SAMPLES = 5000
    BLOCK = 1000  # toggle after each block (at 1000, 2000, 3000, 4000)

    channels = args.channels
    num_ch = len(channels)
    rate = int(args.rate)

    # ---------------- Serial setup & initial configuration -----------------
    ser = serial.Serial(args.port, args.baud, timeout=0.01)
    time.sleep(0.1)

    # Mirror original init sequence; ignore responses to avoid blocking
    init_cmds = [
        b"W C AAAAAAAA\n",
        b"W 10 AAAAAAAA\n",
        b"W 14 AAAAAAAA\n",
        b"W 18 AAAAAAAA\n",
        b"W 8 F000000F\n",  # Power/enable group (per original script)
        b"W 4 F0000000\n",  # Ensure PWM OFF before capture starts
    ]
    for cmd in init_cmds:
        ser.write(cmd)
        try:
            _ = ser.readline()
        except Exception:
            pass

    print("Configured serial for laser control. Waiting to stabilize...")
    if args.stabilize > 0:
        time.sleep(args.stabilize)

    # ----------------------- NI-DAQmx task setup --------------------------
    term_map = {
        "RSE": TerminalConfiguration.RSE,
        "NRSE": TerminalConfiguration.NRSE,
        "DIFF": TerminalConfiguration.DIFF,
    }
    term_cfg = term_map[args.terminal.upper()]

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

        # Start acquisition
        print(f"Starting finite acquisition: {TOTAL_SAMPLES} samples/ch @ {rate} Hz")
        task.start()

        # Read in blocks of 1000 samples per channel and toggle after each block (except after the last)
        current_on = False  # PWM state we control via reg 4
        for i in range(TOTAL_SAMPLES // BLOCK):
            start = i * BLOCK
            end = start + BLOCK
            tmp = np.empty((num_ch, BLOCK), dtype=np.float64)
            # Blocking read of exactly BLOCK samples per channel
            reader.read_many_sample(tmp, number_of_samples_per_channel=BLOCK, timeout=2.0)
            data[:, start:end] = tmp

            # Toggle after completing block 0..3 (not after the final block)
            if i < (TOTAL_SAMPLES // BLOCK) - 1:
                # Flip PWM state quickly without waiting for a response
                current_on = not current_on
                ser.write(b"W 4 F000000F\n" if current_on else b"W 4 F0000000\n")

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
        try:
            ser.close()
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
    )
    print(f"Saved capture to {args.outfile}")

    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(channels):
        plt.plot(t, data[i], label=f"ai{ch}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Laser toggling every ~{BLOCK} samples @ {rate} Hz")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
