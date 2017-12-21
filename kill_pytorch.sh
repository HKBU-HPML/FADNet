PID=`ps -aux | grep "python main.py" | awk '{print $2}'`
echo ${PID}
kill -9 $PID
