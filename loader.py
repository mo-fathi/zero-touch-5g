import json
from kafka import KafkaConsumer
import subprocess
import time
import threading

# Kafka configuration
BOOTSTRAP_SERVERS = ['localhost:9092']

# Helm chart paths (adjust as needed)
OAI_CORE_CHART_PATH = './charts/oai-5g-core/oai-5g-basic'  # Path to OAI core chart
OAI_GNB_CHART_PATH = './charts/oai-gnb'  # Path to gNB chart
OAI_UE_CHART_PATH = './charts/oai-ue'    # Path to UE chart

def uninstall_nsr(namespace, nsr_id, t0_seconds):
    """Wait for T0 seconds and then uninstall Helm releases and delete namespace."""
    print(f"Scheduled uninstallation for NSR {nsr_id} in {t0_seconds} seconds.")
    time.sleep(t0_seconds)
    try:
        # Uninstall Helm releases in reverse order (UEs, gNB, core)
        subprocess.check_call([
            "helm", "uninstall", "oai-ues", "--namespace", namespace
        ])
        print(f"Uninstalled oai-ues from {namespace}.")
        subprocess.check_call([
            "helm", "uninstall", "oai-gnb", "--namespace", namespace
        ])
        print(f"Uninstalled oai-gnb from {namespace}.")
        subprocess.check_call([
            "helm", "uninstall", "oai-core", "--namespace", namespace
        ])
        print(f"Uninstalled oai-core from {namespace}.")
        # Delete the namespace
        subprocess.check_call(["kubectl", "delete", "namespace", namespace])
        print(f"Namespace {namespace} deleted.")
    except subprocess.CalledProcessError as e:
        print(f"Error during uninstallation for NSR {nsr_id}: {e}")

def deploy_nsr(nsr):
    nsr_id = nsr.get('nsr_id', 'default')  # Extract nsr_id or use default
    namespace = f"nsr-{nsr_id}"
    # Get T0 from NSR (in seconds) or default to 3600s (1 hour)
    t0_seconds = nsr.get('t0_seconds', 3600)

    try:
        # Step 1: Create a new namespace
        subprocess.check_call(["kubectl", "create", "namespace", namespace])
        print(f"Namespace {namespace} created.")

        # Step 2: Deploy OAI Core using Helm
        subprocess.check_call([
            "helm", "install", "oai-core", OAI_CORE_CHART_PATH,
            "--namespace", namespace
        ])
        print("OAI Core deployed.")

        # Wait for core pods to be ready
        subprocess.check_call([
            "kubectl", "wait", "--namespace", namespace,
            "--for=condition=ready", "pod", "-l", "app.kubernetes.io/instance=oai-core",
            "--timeout=5m"
        ])

        # Step 3: Deploy gNB using Helm
        subprocess.check_call([
            "helm", "install", "oai-gnb", OAI_GNB_CHART_PATH,
            "--namespace", namespace
        ])
        print("OAI gNB deployed.")

        # Wait for gNB pods to be ready
        subprocess.check_call([
            "kubectl", "wait", "--namespace", namespace,
            "--for=condition=ready", "pod", "-l", "app.kubernetes.io/instance=oai-gnb",
            "--timeout=5m"
        ])

        # Step 4: Deploy UEs using Helm
        subprocess.check_call([
            "helm", "install", "oai-ues", OAI_UE_CHART_PATH,
            "--namespace", namespace
        ])
        print("OAI UEs deployed.")

        # Wait for UE pods to be ready
        subprocess.check_call([
            "kubectl", "wait", "--namespace", namespace,
            "--for=condition=ready", "pod", "-l", "app.kubernetes.io/instance=oai-ues",
            "--timeout=5m"
        ])

        print(f"Deployment for NSR {nsr_id} completed in namespace {namespace}.")

        # Schedule uninstallation after T0 seconds
        threading.Thread(
            target=uninstall_nsr,
            args=(namespace, nsr_id, t0_seconds),
            daemon=True
        ).start()

    except subprocess.CalledProcessError as e:
        print(f"Error during deployment for NSR {nsr_id}: {e}")
        # Optional: Cleanup on failure
        subprocess.call(["kubectl", "delete", "namespace", namespace])

if __name__ == "__main__":
    consumer = KafkaConsumer(
        'deploy',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='nsr-deploy-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print("Starting Kafka consumer for 'deploy' topic...")
    for message in consumer:
        nsr = message.value
        print(f"Received accepted NSR: {nsr}")
        deploy_nsr(nsr)