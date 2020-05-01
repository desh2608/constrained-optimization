# Run penalty method for different values of delta
for delta in `seq -f "%f" 1 0.1 10`; do
  ./penalty.py --delta $delta
done
