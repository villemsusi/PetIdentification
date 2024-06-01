
def write_logs(data):
    with open("logs/capture_log.txt", "a+") as f:
        f.write(str(data) + "\n")

def read_logs(amount):
    with open("logs/capture_log.txt", "r+") as f:
        log_list = f.readlines()
    return log_list[-amount:][::-1]

