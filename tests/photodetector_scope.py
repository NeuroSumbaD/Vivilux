'''Non-blocking real-time oscilloscope for NI DAQ with continuous acquisition.

This approach uses:
- Continuous hardware-buffered acquisition (runs in background)
- Manual canvas updates (non-blocking)
- Separate data collection and rendering threads
- Automatic adjustment to match system capabilities
'''

import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import argparse
import time
import threading
from collections import deque


class RealtimeScope:
    """Non-blocking real-time oscilloscope with continuous acquisition."""
    
    def __init__(self, device_name="Dev1", channels=[0], target_fps=30, 
                 buffer_duration=2.0, sample_rate=10000):
        """
        Parameters
        ----------
        device_name : str
            NI DAQ device name (e.g., "Dev1")
        channels : list of int
            AI channel numbers to read
        target_fps : float
            Target display refresh rate in FPS (default: 30)
        buffer_duration : float
            How many seconds of data to display (default: 2.0)
        sample_rate : int
            DAQ sample rate in Hz (default: 10000)
        """
        self.device_name = device_name
        self.channels = channels
        self.target_fps = target_fps
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        
        # Calculate buffer size
        self.buffer_samples = int(buffer_duration * sample_rate)
        
        # Circular buffer for continuous data (using deque for efficiency)
        self.data_buffer = {ch: deque(maxlen=self.buffer_samples) for ch in channels}
        self.time_buffer = deque(maxlen=self.buffer_samples)
        
        # Thread control
        self.running = False
        self.acquisition_thread = None
        self.task = None
        self.buffer_lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.last_render_time = time.time()
        self.render_times = deque(maxlen=50)
        self.data_read_times = deque(maxlen=50)
        self.samples_acquired = 0
        self.start_time = None
        
        # Adaptive parameters
        self.read_chunk_size = min(1000, sample_rate // 10)  # Start with 100ms chunks
        self.consecutive_slow_renders = 0
        self.last_adjustment_time = 0
        
        # Setup plot
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax, self.ax_stats) = plt.subplots(2, 1, figsize=(12, 8), 
                                                            gridspec_kw={'height_ratios': [4, 1]})
        self.lines = {}
        colors = plt.cm.tab10(range(len(channels)))
        for idx, ch in enumerate(channels):
            line, = self.ax.plot([], [], label=f'AI{ch}', color=colors[idx], linewidth=1.5)
            self.lines[ch] = line
        
        self.ax.set_xlabel('Time (s)', fontsize=11)
        self.ax.set_ylabel('Voltage (V)', fontsize=11)
        self.ax.set_title(f'Real-time Photodetector Scope - {target_fps} FPS target (Continuous)', fontsize=12)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, buffer_duration)
        self.ax.set_ylim(0, 1.3)
        
        # Stats display
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.5, '', fontsize=10, 
                                             verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Print configuration
        print(f"╔{'═'*70}╗")
        print(f"║ Oscilloscope Configuration (Continuous Acquisition){' '*18}║")
        print(f"╠{'═'*70}╣")
        print(f"║ Device: {device_name:<60} ║")
        print(f"║ Channels: {str(channels):<58} ║")
        print(f"╠{'═'*70}╣")
        print(f"║ ACQUISITION:{' '*57}║")
        print(f"║   Sample rate: {sample_rate:>8} Hz{' '*45}║")
        print(f"║   Read chunk:  {self.read_chunk_size:>8} samples (~{self.read_chunk_size/sample_rate*1000:.1f}ms){' '*33}║")
        print(f"╠{'═'*70}╣")
        print(f"║ DISPLAY:{' '*61}║")
        print(f"║   Target FPS:     {target_fps:>5.1f} fps  ({1000/target_fps:.1f} ms/frame){' '*27}║")
        print(f"║   Buffer duration: {buffer_duration:>4.1f} s  ({self.buffer_samples} samples){' '*28}║")
        print(f"╠{'═'*70}╣")
        print(f"║ MODE: Continuous hardware-buffered acquisition{' '*23}║")
        print(f"║       Non-blocking rendering with manual canvas updates{' '*15}║")
        print(f"║       Auto-adaptive read chunk sizing{' '*33}║")
        print(f"╚{'═'*70}╝")
        
    def _acquisition_loop(self):
        """Background thread for continuous data acquisition."""
        try:
            # Setup continuous acquisition task
            self.task = nidaqmx.Task()
            
            # Add channels
            for ch in self.channels:
                self.task.ai_channels.add_ai_voltage_chan(
                    f"{self.device_name}/ai{ch}",
                    min_val=0.0,
                    max_val=1.3,
                    terminal_config=TerminalConfiguration.RSE
                )
            
            # Configure continuous acquisition
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS
            )
            
            # Start task
            self.task.start()
            print("✓ Continuous acquisition started")
            
            # Continuous read loop
            while self.running:
                read_start = time.time()
                try:
                    # Read available data
                    data = self.task.read(
                        number_of_samples_per_channel=self.read_chunk_size,
                        timeout=1.0
                    )
                    
                    # Handle single vs multi-channel
                    if len(self.channels) == 1:
                        data = [data]
                    
                    # Calculate timestamps
                    current_time = time.time() - self.start_time
                    num_samples = len(data[0]) if isinstance(data[0], list) else len(data)
                    timestamps = np.linspace(
                        current_time - (num_samples / self.sample_rate),
                        current_time,
                        num_samples
                    )
                    
                    # Update buffer (thread-safe)
                    with self.buffer_lock:
                        for idx, ch in enumerate(self.channels):
                            ch_data = data[idx]
                            self.data_buffer[ch].extend(ch_data)
                        self.time_buffer.extend(timestamps)
                        self.samples_acquired += num_samples
                    
                    # Track read time
                    read_time = time.time() - read_start
                    self.data_read_times.append(read_time * 1000)
                    
                except nidaqmx.errors.DaqError as e:
                    if self.running:
                        print(f"DAQ Error: {e}")
                        break
                        
        except Exception as e:
            print(f"Acquisition error: {e}")
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
            print("✓ Acquisition stopped")
    
    def _render_frame(self):
        """Render one frame manually (non-blocking)."""
        # Check if matplotlib backend is ready
        if not plt.fignum_exists(self.fig.number):
            self.running = False
            return False
        
        render_start = time.time()
        
        # Get data snapshot (thread-safe)
        with self.buffer_lock:
            if len(self.time_buffer) == 0:
                return True  # No data yet, keep running
            
            # Convert deque to numpy arrays
            times = np.array(self.time_buffer)
            data_arrays = {ch: np.array(self.data_buffer[ch]) for ch in self.channels}
        
        # Update plot lines
        for ch in self.channels:
            self.lines[ch].set_data(times, data_arrays[ch])
        
        # Calculate dynamic x-axis limits (rolling window)
        if len(times) > 0:
            current_time = times[-1]
            x_min = max(0, current_time - self.buffer_duration)
            x_max = current_time
            self.ax.set_xlim(x_min, x_max)
        
        # Calculate dynamic y-axis limits
        all_values = np.concatenate(list(data_arrays.values()))
        if len(all_values) > 0:
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            y_range = y_max - y_min
            
            if y_range < 0.01:  # Handle nearly constant signals
                y_center = (y_max + y_min) / 2
                y_min_adj = max(0, y_center - 0.05)
                y_max_adj = min(1.3, y_center + 0.05)
            else:
                padding = y_range * 0.1
                y_min_adj = max(0, y_min - padding)
                y_max_adj = min(1.3, y_max + padding)
            
            self.ax.set_ylim(y_min_adj, y_max_adj)
        
        # Update stats
        render_time = (time.time() - render_start) * 1000
        self.render_times.append(render_time)
        
        elapsed = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_render = np.mean(self.render_times) if self.render_times else 0
        avg_read = np.mean(self.data_read_times) if self.data_read_times else 0
        
        # Calculate data rate
        data_rate = self.samples_acquired / elapsed if elapsed > 0 else 0
        buffer_fill = len(self.time_buffer) / self.buffer_samples * 100
        
        stats = (f"Performance (Frame {self.frame_count}, {elapsed:.1f}s):\n"
                f"  Target: {self.target_fps:.1f} FPS  |  Actual: {actual_fps:.1f} FPS  |  "
                f"  Render: {avg_render:.2f}ms  |  Data read: {avg_read:.2f}ms\n"
                f"  Sample rate: {data_rate:.0f} Hz  |  Chunk: {self.read_chunk_size} samples  |  "
                f"  Buffer: {buffer_fill:.1f}%  |  Total: {self.samples_acquired}")
        
        self.stats_text.set_text(stats)
        
        # Auto-adjust chunk size if falling behind
        time_since_adjust = time.time() - self.last_adjustment_time
        target_frame_time = 1000 / self.target_fps
        
        if render_time > target_frame_time * 0.8 and time_since_adjust > 2.0:
            # Rendering too slow, reduce chunk size
            if self.read_chunk_size > 100:
                old_chunk = self.read_chunk_size
                self.read_chunk_size = max(100, int(self.read_chunk_size * 0.75))
                print(f"\n⚡ AUTO-ADJUST: Reduced chunk size {old_chunk} → {self.read_chunk_size} samples")
                self.last_adjustment_time = time.time()
                self.consecutive_slow_renders = 0
        elif render_time < target_frame_time * 0.5 and time_since_adjust > 5.0:
            # Rendering fast, can increase chunk size for efficiency
            if self.read_chunk_size < self.sample_rate // 2:
                old_chunk = self.read_chunk_size
                self.read_chunk_size = min(self.sample_rate // 2, int(self.read_chunk_size * 1.25))
                print(f"\n⚡ AUTO-ADJUST: Increased chunk size {old_chunk} → {self.read_chunk_size} samples")
                self.last_adjustment_time = time.time()
        
        # Manually update canvas (non-blocking)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except:
            return False
        
        self.frame_count += 1
        self.last_render_time = time.time()
        
        return True
    
    def start(self):
        """Start the oscilloscope with continuous acquisition."""
        self.running = True
        self.start_time = time.time()
        
        # Start acquisition thread
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()
        
        # Allow acquisition to start
        time.sleep(0.1)
        
        print("\n✓ Starting non-blocking render loop...")
        print("  Close the window to stop\n")
        
        # Manual render loop
        target_frame_interval = 1.0 / self.target_fps
        
        try:
            while self.running:
                # Calculate time to next frame
                time_since_last = time.time() - self.last_render_time
                if time_since_last >= target_frame_interval:
                    if not self._render_frame():
                        break
                else:
                    # Sleep briefly to avoid busy-waiting
                    time.sleep(0.001)
                
                # Process GUI events
                plt.pause(0.001)
                
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop and cleanup."""
        if not self.running:
            return
            
        print("\n✓ Shutting down...")
        self.running = False
        
        # Wait for acquisition thread
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=2.0)
        
        # Close plot
        try:
            plt.close(self.fig)
        except:
            pass
        
        print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time NI DAQ Oscilloscope (Continuous Acquisition with Non-blocking Rendering)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 30 FPS with 2s buffer
  python photodetector_scope.py --channels 0 1 2 3
  
  # Fast 60 FPS with short buffer
  python photodetector_scope.py --channels 0 1 --fps 60 --buffer 1.0
  
  # Slow 10 FPS with long buffer
  python photodetector_scope.py --channels 0 --fps 10 --buffer 5.0
  
  # High sample rate
  python photodetector_scope.py --channels 0 1 --rate 50000 --fps 30
        """)
    
    parser.add_argument('--device', default='Dev1', 
                        help='Device name (default: Dev1)')
    parser.add_argument('--channels', nargs='+', type=int, default=[0], 
                        help='AI channels to plot (default: 0)')
    parser.add_argument('--fps', type=float, default=30, 
                        help='Target display FPS (default: 30)')
    parser.add_argument('--buffer', type=float, default=2.0, 
                        help='Buffer duration in seconds (default: 2.0)')
    parser.add_argument('--rate', type=int, default=10000, 
                        help='DAQ sample rate in Hz (default: 10000)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.fps <= 0 or args.fps > 240:
        print("ERROR: FPS must be between 0 and 240")
        return
    
    if args.buffer < 0.1 or args.buffer > 60:
        print("ERROR: Buffer duration must be between 0.1 and 60 seconds")
        return
    
    if args.rate < 1000 or args.rate > 250000:
        print("ERROR: Sample rate must be between 1000 and 250000 Hz")
        return
    
    print(f"Starting oscilloscope...")
    
    scope = RealtimeScope(
        device_name=args.device,
        channels=args.channels,
        target_fps=args.fps,
        buffer_duration=args.buffer,
        sample_rate=args.rate
    )
    
    try:
        scope.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        scope.stop()


if __name__ == "__main__":
    main()