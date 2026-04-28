mkdir -p /teutonic/teutonic_dataset

seq -f "%06g" 1368 1369 | \
xargs -n 1 -P 8 -I {} \
wget -q --show-progress -c --tries=10 --timeout=30 \
  -O ~/teutonic/teutonic_dataset/shard_{}.npy \
  "https://s3.hippius.com/teutonic-sn3/dataset/v2/shards/shard_{}.npy"
