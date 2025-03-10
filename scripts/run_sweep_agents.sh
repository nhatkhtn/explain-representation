for ((i = 1; i <= 50; i++));
do
    echo "Running the "$i"-th round"
    for ((j = 1; j <= 10; j++));
    do
        # wandb agent nhathcmus/sweep-exrep-downstream1/vlt460yb --count 1
        nohup wandb agent nhathcmus/sweep-exrep-downstream1/tq0xzeuv --count 1 & 
        pid=$!
        echo "PID is "$pid""
    done
    sleep 360s
    kill $(pgrep -f run_train.py)
done