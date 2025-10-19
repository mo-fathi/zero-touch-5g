#!/usr/bin/env python3
from kubernetes import client, config
import math

def parse_quantity(quantity_str):
    """
    Convert Kubernetes quantity strings (e.g. '100m', '2Gi') to float values.
    CPU: returns cores as float
    Memory: returns bytes as float
    """
    quantity_str = str(quantity_str).strip()
    if quantity_str.endswith('m'):  # CPU in millicores
        return float(quantity_str[:-1]) / 1000
    elif quantity_str.endswith('n'): # CPU in nanosecond
        return float(quantity_str[:-1]) / 1_000_000_000.0
    elif quantity_str.endswith('Ki'):
        return float(quantity_str[:-2]) * 1024
    elif quantity_str.endswith('Mi'):
        return float(quantity_str[:-2]) * 1024 ** 2
    elif quantity_str.endswith('Gi'):
        return float(quantity_str[:-2]) * 1024 ** 3
    elif quantity_str.endswith('Ti'):
        return float(quantity_str[:-2]) * 1024 ** 4
    else:
        try:
            return float(quantity_str)
        except ValueError:
            return 0.0

def get_realtime_resources():
    # Load configuration
    try:
        config.load_kube_config()        # local dev
    except:
        config.load_incluster_config()   # inside cluster

    v1 = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()

    nodes = v1.list_node().items
    total_alloc_cpu = total_alloc_mem = 0.0
    total_used_cpu = total_used_mem = 0.0

    # Get metrics from Metrics API (requires metrics-server)
    #TODO just worker nodes should be add for this metrics tupple
    metrics = custom_api.list_cluster_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        plural="nodes"
    )


    usage_by_node = {
        item["metadata"]["name"]: item["usage"]
        for item in metrics["items"]
    }

    for node in nodes:
        labels = node.metadata.labels or {}
        name = node.metadata.name


        # Skip control-plane / master nodes
        if (
            "node-role.kubernetes.io/control-plane" in labels
            or "node-role.kubernetes.io/master" in labels
        ):
            continue

        alloc = node.status.allocatable
        alloc_cpu = parse_quantity(alloc["cpu"])
        alloc_mem = parse_quantity(alloc["memory"])

        total_alloc_cpu += alloc_cpu
        total_alloc_mem += alloc_mem

        if name in usage_by_node:
            usage = usage_by_node[name]
            used_cpu = parse_quantity(usage["cpu"])
            used_mem = parse_quantity(usage["memory"])
        else:
            used_cpu = used_mem = 0.0

        total_used_cpu += used_cpu
        total_used_mem += used_mem

    # print (total_alloc_cpu)
    # print (total_used_cpu)
    # exit (0)
    free_cpu = total_alloc_cpu - total_used_cpu
    free_mem = total_alloc_mem - total_used_mem

    print("=== Kubernetes Cluster Real-Time Resources ===")
    print(f"Allocatable CPU cores: {total_alloc_cpu:.2f}")
    print(f"Used CPU cores:        {total_used_cpu:.2f}")
    print(f"Free CPU cores:        {free_cpu:.2f}\n")

    print(f"Allocatable Memory:    {total_alloc_mem / (1024 ** 3):.2f} GiB")
    print(f"Used Memory:           {total_used_mem / (1024 ** 3):.2f} GiB")
    print(f"Free Memory:           {free_mem / (1024 ** 3):.2f} GiB")

if __name__ == "__main__":
    get_realtime_resources()
