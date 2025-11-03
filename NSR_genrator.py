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
    nsr_id = int(time.time() * 1000)
    nsr = {
        "id": nsr_id,
        "QoS": {
            # Latency in mili second
            "L_max_int": round(random.uniform(1, 10), 2),
            "L_max_ext": round(random.uniform(1, 10), 2),
            # Throughput in mbps
            "Phi_min_int": round(random.uniform(50, 200), 2),
            "Phi_min_ext": round(random.uniform(50, 200), 2),
            # packet loss in persentage (0,1)
            "P_max_int": round(random.uniform(0, 0.01), 4),
            "P_max_ext": round(random.uniform(0, 0.01), 4)
        },
        "T0": random.randint(5, 15) * 60,  # Lifespan in minutes
        "revenue": random.randint(10, 100),
        "arrival_time": int(time.time()) 
    }
    return nsr

# Continuous NSR generation
# TODO make a specific distribution for NSR generating instead of random.
print("NSR Generator started", datetime.datetime.now())
while True:
    if random.random() < 0.3:  # 30% chance per second
        nsr = generate_nsr()
        producer.send(topic, nsr)
        
        # Update Redis queue stats
        q_size = keeper.incr('q_size')

       if q_size == 1:
            keeper.set('sum_revenue', 0)
            keeper.set('sum_L_max_int', 0.0)
            keeper.set('sum_L_max_ext', 0.0)
            keeper.set('sum_Phi_min_int', 0.0)
            keeper.set('sum_Phi_min_ext', 0.0)
            keeper.set('sum_P_max_int', 0.0)
            keeper.set('sum_P_max_ext', 0.0)
            

        # Add current NSR's values
        keeper.incrby('sum_revenue', nsr['revenue'])
        keeper.incrbyfloat('sum_L_max_int', nsr['QoS']['L_max_int'])
        keeper.incrbyfloat('sum_L_max_ext', nsr['QoS']['L_max_ext'])
        keeper.incrbyfloat('sum_Phi_min_int', nsr['QoS']['Phi_min_int'])
        keeper.incrbyfloat('sum_Phi_min_ext', nsr['QoS']['Phi_min_ext'])
        keeper.incrbyfloat('sum_P_max_int', nsr['QoS']['P_max_int'])
        keeper.incrbyfloat('sum_P_max_ext', nsr['QoS']['P_max_ext'])


            

        print(f"NSR_{nsr['id']} → Kafka | Queue: {q_size} | Rev: {nsr['revenue']}")
        
    time.sleep(1)  # 1 second interval
