import datetime

# Define today
start_date = str(datetime.date.today())
start_date.replace('/', '_')

# Check to see if the log file has been run today
with open('log.txt', mode='r') as file:
    log = file.readline()
    while(log):
        if log.split()[0] == start_date:
            quit()
        log = file.readline()

# Append the successful daily update to the log
with open('log.txt', mode='a') as file:
    file.write('%s - Stock Prediction Daily Update Completed\n' % (start_date))
