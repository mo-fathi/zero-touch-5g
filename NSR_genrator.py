import random
import json
import time
import datetime
from kafka import KafkaProducer

# Kafka configuration
bootstrap_servers = ['localhost:9092']
topic = 'lambda'

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
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
    }
    return nsr

# Continuous NSR generation
print("NSR Generator started", datetime.datetime.now())
while True:
    if random.random() < 0.3:  # 30% chance per second
        nsr = generate_nsr()
        producer.send(topic, nsr)
        print(f"Sent NSR_{nsr['id']} to lambda topic at {time.strftime('%H:%M:%S CEST, %B %d, %Y')}, T0={nsr['T0']:.1f}s")
    time.sleep(1)  # 1 second interval