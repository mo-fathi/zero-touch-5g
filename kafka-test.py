from kafka import KafkaProducer

producer = KafkaProducer(
   bootstrap_servers='localhost:9092',
)

producer.send('lambda', b'test')
producer.flush()
print('Published message')