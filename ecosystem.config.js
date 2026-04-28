// ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "train-teutonic",
      
      // Launch torchrun via Python interpreter
      script: "torchrun",
      
      // Args: torchrun flags + your script + config file
      args: [
        "--nproc_per_node=2",           // Number of GPUs
        "--master_port=29500",          // Avoid port conflicts
        "--rdzv_backend=c10d",          // Recommended for PyTorch DDP
        "train.py",                     // Your training script
        "config.yaml"                   // Pass your YAML config
      ].join(" "),
      
      interpreter: "none", // torchrun is already executable via PATH
      
      // Environment variables
      env: {
        // GPU visibility
        CUDA_VISIBLE_DEVICES: "0,1",
        
        // PyTorch distributed settings
        MASTER_ADDR: "localhost",
        MASTER_PORT: "29500",
        
        // Optional: override config via env vars (HfArgumentParser supports this)
        // MODEL_PATH: "sniper918/Teutonic-III-vxxiv",
        // LEARNING_RATE: "1e-6",
        
        // WandB authentication (if not using wandb login)
        // WANDB_API_KEY: "your-key-here",
        
        // Python buffering for real-time logs
        PYTHONUNBUFFERED: "1",
        
        // NCCL settings for stability
        NCCL_DEBUG: "WARN",              // Reduce noise; use "INFO" for debugging
        NCCL_IB_DISABLE: "1",            // Disable InfiniBand if not available
        NCCL_SOCKET_IFNAME: "lo",        // Use loopback for local multi-GPU
      },
      
      // Process management
      instances: 1,                      // ONE torchrun process (spawns workers internally)
      autorestart: true,                 // Restart on crash (optional)
      restart_delay: 10000,              // Wait 10s before restart
      
      // Logging - PM2 handles stdout/stderr
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      out_file: `logs/pm2-out.log`,      // Combined output from all ranks
      error_file: `logs/pm2-err.log`,    // Errors only
      merge_logs: true,                  // Merge logs from all worker processes
      
      // Optional: max memory restart
      max_memory_restart: "20G",         // Restart if any worker exceeds 20GB
      
      // Watch mode (disable for training!)
      watch: false,
      
      // Graceful shutdown
      kill_timeout: 60000,               // Wait 60s for cleanup on stop
    },
  ],
};