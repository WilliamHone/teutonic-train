module.exports = {
  apps: [
    {
      name: "train-teutonic",
      script: "torchrun",
      args: "--nproc_per_node=2 train.py",
      interpreter: "python",
      // Ensure environment variables for GPU usage are passed
      env: {
        CUDA_VISIBLE_DEVICES: "0,1", // Use specific GPUs
        MASTER_PORT: "29500", // Free port for rendezvous
      },
      // Important for handling multiple processes
      instances: 1, 
      autorestart: false, // Don't restart, let torchrun handle failure
    },
  ],
};
