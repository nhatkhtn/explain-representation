for ((i = 1; i <= 100; i++));
do
    echo "Running the "$i"-th round"
    for ((j = 1; j <= 5; j++));
    do
        # wandb agent nhathcmus/sweep-exrep-downstream1/vlt460yb --count 1
        nohup wandb agent nhathcmus/sweep-exrep-downstream1/2jtrm4yx --count 1 & 
        pid=$!
        echo "PID is "$pid""
    done
    sleep 90s
    kill $(pgrep -f run_train.py)
done