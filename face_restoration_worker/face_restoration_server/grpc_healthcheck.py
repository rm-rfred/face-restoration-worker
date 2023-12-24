import grpc

TIMEOUT = 30

try:
    channel = grpc.insecure_channel("localhost:13000")
    grpc.channel_ready_future(channel).result(timeout=TIMEOUT)
    channel.close()
except grpc.FutureTimeoutError:
    exit(1)

exit(0)
