import random
import json
import time
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
    nsr_id = random.randint(1, 1000)
    bw_required = random.uniform(10, 50)  # Bandwidth in MHz
    nsr = {
        "id": nsr_id,
        "QoS": {
            "L_max_int": random.uniform(1, 10),
            "L_max_ext": random.uniform(1, 10),
            "Phi_min_int": random.uniform(50, 200),
            "Phi_min_ext": random.uniform(50, 200),
            "P_max_int": random.uniform(0, 0.01),
            "P_max_ext": random.uniform(0, 0.01)
        },
        "T0": random.uniform(10, 50),  # Lifespan in seconds
    }
    return nsr

# Continuous NSR generation
print("NSR Generator started at 11:13 PM CEST, August 30, 2025...")
while True:
    if random.random() < 0.3:  # 30% chance per second
        nsr = generate_nsr()
        producer.send(topic, nsr)
        print(f"Sent NSR_{nsr['id']} to lambda topic at {time.strftime('%H:%M:%S CEST, %B %d, %Y')}, T0={nsr['T0']:.1f}s")
    time.sleep(1)  # 1 second interval