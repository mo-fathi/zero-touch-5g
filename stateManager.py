




if __name__ == "__main__":
    print("lambda state manager started at: ", datetime.datetime.now())






# Admission Control cycle:
    try:
        for message in consumer:
            if message:
                print(f"Received message: {message.value}")
                nsr = message.value
                admission_control(nsr)
                
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        consumer.close()