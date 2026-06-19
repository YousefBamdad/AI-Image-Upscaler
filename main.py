import os
import torch
from ui import UpscalerApp

# 1. Automatic CPU core hardware detection and PyTorch multi-threading optimization
total_cores = os.cpu_count()

# Reserve threads so the OS doesn't freeze during heavy tensor calculations
optimal_threads = max(1, total_cores - 1) if total_cores else 4
torch.set_num_threads(optimal_threads)

# 2. Start Application Runtime
if __name__ == "__main__":
    app = UpscalerApp()

    # Send hardware initialization specs directly into the UI log view
    app.add_log(f"[Hardware Setup] System detected {total_cores} logical CPU cores.")
    app.add_log(f"[Hardware Setup] PyTorch threads successfully optimized to use: {optimal_threads}")
    app.add_log("-" * 65)

    app.mainloop()