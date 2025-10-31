import random
import json
import time
import datetime
from kafka import KafkaProducer
import redis

# Kafka configuration
bootstrap_servers = ['localhost:9092']
topic = 'lambda'

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Redis configuration
keeper = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # ensures returned values are strings instead of bytes
)


# Generate a new NSR
def generate_nsr():
    nsr_id = int(time.time())
    nsr = {
        "id": nsr_id,
        "QoS": {
            # Latency in mili second
            "L_max_int": random.uniform(1, 10),
            "L_max_ext": random.uniform(1, 10),
            # Throughput in mbps
            "Phi_min_int": random.uniform(50, 200),
            "Phi_min_ext": random.uniform(50, 200),
            # packet loss in persentage (0,1)
            "P_max_int": random.uniform(0, 0.01),
            "P_max_ext": random.uniform(0, 0.01)
        },
        "T0": random.randint(5, 15) * 60,  # Lifespan in minutes
        "revenue": random.randint(1,100)  
    }
    return nsr

# Continuous NSR generation
# TODO make a specific distribution for NSR generating instead of random.
print("NSR Generator started", datetime.datetime.now())
while True:
    if random.random() < 0.3:  # 30% chance per second
        nsr = generate_nsr()
        producer.send(topic, nsr)
        
        # update redis lambda queue infourmation
        if keeper.get('q_size'):
            keeper.set('q_size', int(keeper.get())+ 1)
            keeper.set('sum_of_revenue', int(keeper.get('sum_of_revenue')) + nsr["revenue"])
            # TODO Add sum of QoS parameter 

        else:
            keeper.set('q_size',1)
            keeper.set('sum_of_revenue',0)


            

        print(f"Sent NSR_{nsr['id']} to lambda topic at {time.strftime('%H:%M:%S CEST, %B %d, %Y')}, T0={nsr['T0']:.1f}s")
    time.sleep(1)  # 1 second interval

# if __name__ == "__main__":
#     nsr = generate_nsr()
#     print (nsr[])