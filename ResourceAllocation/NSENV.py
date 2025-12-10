import gymnasium as gym
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Box
import numpy as np
import random
from typing import Any





class NetSliceEnv(gym.Env):

    metadata = {"render_modes": []}


    def __init__(self, max_slices: int = 10, nf_num: int = 8, slice_feature_dim: int = 2, nf_feature_dim: int = 8, qos_params: int = 6):
        super().__init__()

        self.max_slices = max_slices
        self.nf_num = nf_num
        self.ns_feature_dim = slice_feature_dim
        # It comes from: 2 (cpu,mem) * 4 (request, limit, min, usage)
        self.nf_feature_dim = nf_feature_dim
        self.qos_params = qos_params
        
        self.current_step = 0

        # Define Observation space for RA agent
        self.observation_space = GymDict({
            # Remaining Resource of the cluster (cpu, mem, and radio bandwidth)
            # 6 shape as: (cpu_cap, cpu_used, mem_cap, mem_used, bw_cap, bw_used)
            "cluster": Box(low=0.0, high=1e6, shape=(self.qos_params,), dtype=np.float32),

            # QoS feature of Network slices
            # qos_params (e.g. int_latency, ext_latency, int_throughput, ext_throughput, int_packet_loss, ext_packet_loss)
            # qos_params * 2 for current and required for each one.
            "QoS": Box(low=0.0, high=1e6, shape=(self.max_slices, self.qos_params * 2), dtype=np.float32),

            # Network Function Resources Information of each network slice
            "nf_features": Box(low=0.0, high=1e6, shape=(self.max_slices, self.nf_num, self.nf_feature_dim), dtype=np.float32),

            # Network Slice Resource Information (Bandwidth)
            # TODO we can add number of UEs of network slices
            "slice_features": Box(low=0.0, high=1e6, shape=(self.max_slices, self.ns_feature_dim), dtype=np.float32),

            # To show agent wich Network Slices are masked (does not exist)
            "mask": Box(low=0, high=1, shape=(self.max_slices,), dtype=np.float32),

        })

        # Define Action Space for RA agent
        action_dim = self.max_slices * (2 * self.nf_num + 1) # 2 (cpu/mem) + 1 BW
        # Deltas for resources
        # action = {slice_1_cpu_NF1, ..., slice_1_cpu_NFmax, slice_1_mem_NF1, ..., slice_1_mem_NFmax, slice_1_BW,
        #           slice_2_cpu_NF1, ..., slice_2_cpu_NFmax, slice_2_mem_NF1, ..., slice_2_mem_NFmax, slice_2_BW,
        #           ...
        #           ...
        #           ...
        #           slice_max_cpu_NF1, ..., slice_max_cpu_NFmax, slice_max_mem_NF1, ..., slice_max_mem_NFmax, slice_max_BW}
        self.action_space = Box (low=-5, high=5.0, shape=(action_dim,), dtype=np.float32)


        # Create cluster simulator
        self.simulator = ClusterSimulator(
            max_slices = self.max_slices,
            nf_num = self.nf_num
        )


    def reset(self, *,  seed=None, options = None):
        super().reset(seed=seed)

        self.current_step = 0
        self.simulator.reset()

        initial_slices = random.randint(2, self.max_slices // 2 + 1)
        for _ in range(initial_slices):
            self.simulator.add_slice()
        self.simulator.update_loads()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        
        active = len(self.simulator.active_slices)

        # Unpack and mask actions
        cpu_deltas = np.zeros((self.max_slices, self.nf_num))
        mem_deltas = np.zeros((self.max_slices, self.nf_num))
        bw_deltas = np.zeros(self.max_slices)

        idx = 0
        for i in range(self.max_slices):
            cpu_deltas[i] = action[idx:idx + self.nf_num]
            idx += self.nf_num
            mem_deltas[i] = action[idx:idx + self.nf_num]
            idx += self.nf_num
            bw_deltas[i] = action[idx]
            idx += 1

        mask_vec = np.zeros(self.max_slices)
        mask_vec[:active] = 1.0
        for i in range(self.max_slices):
            if mask_vec[i] == 0:
                cpu_deltas[i] = 0
                mem_deltas[i] = 0
                bw_deltas[i] = 0

        for i in range(active):
            self.simulator.apply_delta(i, cpu_deltas[i], mem_deltas[i], bw_deltas[i])

        # Dynamic arrival/departure
        # TODO make it better, for example base on time
        if random.random() < 0.07 and active < self.max_slices:
            self.simulator.add_slice()
        if random.random() < 0.04 and active > 1:
            self.simulator.remove_slice()

        self.simulator.update_loads()

        reward = self._compute_reward()
        info["active_slices"] = len(self.simulator.active_slices)

        self.current_step += 1
        if self.current_step >= 1000:
            terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):

        cluster = np.zeros(self.qos_params, dtype=np.float32)
        qos = np.zeros((self.max_slices, self.qos_params * 2), dtype=np.float32)
        nf_features = np.zeros((self.max_slices, self.nf_num, self.nf_feature_dim), dtype=np.float32)
        slice_features  = np.zeros((self.max_slices, self.ns_feature_dim), dtype=np.float32) 
        mask = np.zeros(self.max_slices, dtype=np.float32)
        

        active = len(self.simulator.active_slices)
        mask[:active] = 1.0

        used_cpu = used_mem = used_bw = 0.0
        

        simulated_qos = self.simulate_qos(self.simulator.active_slices) 

        for i in range(active):
            s = self.simulator.active_slices[i]
            num_nfs = len(s["nfs"])
            # nf_mask[i, :num_nfs] = 1.0

            total_req_cpu = total_limit_cpu = total_min_cpu = total_used_cpu = 0.0
            total_req_mem = total_limit_mem = total_min_mem = total_used_mem = 0.0

            for j in range(num_nfs):
                nf = s["nfs"][j]
                nf_features[i, j] = np.array([
                    nf["requested_cpu"],
                    nf["limited_cpu"],
                    nf["min_cpu"],
                    nf["cpu_usage"],
                    nf["requested_mem"],
                    nf["limited_mem"],
                    nf["min_mem"],
                    nf["mem_usage"],
                ], dtype=np.float32)

                total_req_cpu += nf["requested_cpu"]
                total_limit_cpu += nf["limited_cpu"]
                total_min_cpu += nf["min_cpu"]
                total_used_cpu += nf["cpu_usage"]
                total_req_mem += nf["requested_mem"]
                total_limit_mem += nf["limited_mem"]
                total_min_mem += nf["min_mem"]
                total_used_mem += nf["mem_usage"]

                used_cpu += max(nf["cpu_usage"], nf["requested_cpu"])
                used_mem += max(nf["mem_usage"], nf["requested_mem"])

            used_bw += max(s["bw_usage"], s["allocated_bw"])

            slice_qos = np.array([
                    simulated_qos[i]["int_latency"],
                    simulated_qos[i]["int_loss"],
                    simulated_qos[i]["int_throughput"],
                    simulated_qos[i]["ext_latency"],
                    simulated_qos[i]["ext_loss"],
                    simulated_qos[i]["ext_throughput"],
                    s["target_int_latency_ms"],
                    s["target_int_loss"],
                    s["target_int_throughput"],
                    s["target_ext_latency_ms"],
                    s["target_ext_loss"],
                    s["target_ext_throughput"],
            ], dtype=np.float32)

            qos[i] = slice_qos

            slice_features[i] = np.array([
                s["allocated_bw"],
                s["bw_usage"],
            ], dtype=np.float32)

        cluster = np.array([
            self.simulator.total_capacity_cpu,
            used_cpu,
            self.simulator.total_capacity_mem,
            used_mem,
            self.simulator.total_capacity_bw,
            used_bw,
        ], dtype=np.float32)



        return {
            "cluster": cluster,
            "QoS": qos,
            "nf_features": nf_features,
            "slice_features": slice_features,
            "mask": mask,
        }
        
    def _compute_reward(self):
        reward = 0.0
        
        # get QoS parameters
        qos = self.simulate_qos(self.simulator.active_slices)
        
        # total_used_cpu and total_used_mem are sum of max(usage, requested) for each NF
        # total_requested_cpu and total_requested_mem are sum of requested for each NF
        # total_used_bw is sum of used bw for each slice
        # total_allocated_bw is sum of allocated bw for each slice
        total_used_cpu = total_requested_cpu = total_used_mem = total_requested_mem = total_used_bw = total_allocated_bw = 0.0

        # Violations and over-allocations
        # TODO using over provisioning parameters in reward
        total_cpu_over = total_mem_over = total_bw_over = 0.0

        slice_idx = 0
        for s in self.simulator.active_slices:

            for nf in s["nfs"]:
                total_cpu_over += max(0.0, nf["requested_cpu"] - nf["cpu_usage"] )
                total_mem_over += max(0.0, nf["requested_mem"] - nf["mem_usage"] )

                total_used_cpu += max(nf["cpu_usage"], nf["requested_cpu"])
                total_requested_cpu += nf["requested_cpu"]
                total_used_mem += max(nf["mem_usage"], nf["requested_mem"])
                total_requested_mem += nf["requested_mem"]

            total_used_bw += s["bw_usage"]
            total_allocated_bw += s["allocated_bw"]

            # Compute Reward/Penalty based on QoS
            qos_ok = all([
                        qos[slice_idx]["qos_satisfied"]["int_latency"],
                        qos[slice_idx]["qos_satisfied"]["int_loss"],
                        qos[slice_idx]["qos_satisfied"]["int_throughput"],
                        qos[slice_idx]["qos_satisfied"]["ext_latency"],
                        qos[slice_idx]["qos_satisfied"]["ext_loss"],
                        qos[slice_idx]["qos_satisfied"]["ext_throughput"]                 
                        ])

            if qos_ok:
                reward += 25.0
            else:
                reward -= 40.0

            # Specific QoS penalties
            if not qos[slice_idx]["qos_satisfied"]["int_latency"]:
                reward -= 10.0 * (qos[slice_idx]["int_latency"] - s["target_int_latency_ms"] * 1.1) / s["target_int_latency_ms"]
            if not qos[slice_idx]["qos_satisfied"]["int_loss"]:
                reward -= 5.0 * (qos[slice_idx]["int_loss"] - s["target_int_loss"] * 1.1) / s["target_int_loss"]
            if not qos[slice_idx]["qos_satisfied"]["int_throughput"]:
                reward -= 10.0 * (s["target_int_throughput"] * 0.9 - qos[slice_idx]["int_throughput"]) / s["target_int_throughput"]
            if not qos[slice_idx]["qos_satisfied"]["ext_latency"]:
                reward -= 10.0 * (qos[slice_idx]["ext_latency"] - s["target_ext_latency_ms"] * 1.1) / s["target_ext_latency_ms"]
            if not qos[slice_idx]["qos_satisfied"]["ext_loss"]:
                reward -= 5.0 * (qos[slice_idx]["ext_loss"] - s["target_ext_loss"] * 1.1) / s["target_ext_loss"]
            if not qos[slice_idx]["qos_satisfied"]["ext_throughput"]:
                reward -= 10.0 * (s["target_ext_throughput"] * 0.9 - qos[slice_idx]["ext_throughput"]) / s["target_ext_throughput"]

            slice_idx += 1

        # Global efficiency bonuses
        # TODO prevent to assign requested < usage
        if total_requested_cpu > 0:
            cpu_util = total_used_cpu / total_requested_cpu
            reward += 20.0 * cpu_util
        if total_requested_mem > 0:
            mem_util = total_used_mem / total_requested_mem
            reward += 10.0 * mem_util
        if total_allocated_bw > 0:
            bw_util = total_used_bw / total_allocated_bw
            reward += 15.0 * bw_util

        # Penalties for cost of allocations
        reward -= 0.03 * total_requested_cpu
        reward -= 0.01 * total_requested_mem
        reward -= 0.02 * total_allocated_bw

        return reward

        
    def _get_info(self):
        pass
    
    def simulate_qos(self, slices):
        """
        Simulates QoS parameters for network slices based on resource allocations.
        
        Args:
        slices (list of dict): Each dict represents a slice with:
            - 'allocated_bw': Allocated bandwidth (bps)
            - 'bw_usage': Bandwidth usage (bps)
            - 'target_int_latency_ms': Target internal latency (ms)
            - 'target_int_loss': Target internal loss (0-1)
            - 'target_int_throughput': Target internal throughput (bps)
            - 'target_ext_latency_ms': Target external latency (ms)
            - 'target_ext_loss': Target external loss (0-1)
            - 'target_ext_throughput': Target external throughput (bps)
            - 'nfs': list of dicts, each NF with:
                - 'requested_cpu': Requested CPU (cores)
                - 'limited_cpu': Limited CPU (cores)
                - 'min_cpu': Minimum CPU (cores)
                - 'cpu_usage': CPU usage (cores)
                - 'requested_mem': Requested memory (MB)
                - 'limited_mem': Limited memory (MB)
                - 'min_mem': Minimum memory (MB)
                - 'mem_usage': Memory usage (MB)
        
        Returns:
        list of dict: Each dict with computed QoS for the slice and satisfaction status.
        """
        
        # Constants (adjust based on your simulation)
        CPU_CLOCK_SPEED_GHZ = 2.0  # CPU frequency
        CYCLES_PER_PACKET = 2000   # Cycles needed to process one packet
        PACKET_SIZE_BYTES = 1500   # Average packet size
        BUFFER_FRACTION = 0.7      # Fraction of memory for buffers
        MISS_LATENCY_MS = 0.5      # Memory miss latency
        PROPAGATION_DELAY_MS = 5   # Base propagation delay
        EFFICIENCY_FACTOR = 0.9    # Bandwidth efficiency
        BASE_LOSS = 0.001          # Base wireless loss
        HIGH_LATENCY_PENALTY = 1000  # Penalty for overload (ms)
        
        results = []
        
        for s in slices:
            allocated_bw = s['allocated_bw']
            bw_usage = s['bw_usage']
            nfs = s['nfs']
            
            # External QoS (bandwidth-driven)
            ext_throughput = min(allocated_bw, bw_usage) * EFFICIENCY_FACTOR
            
            transmission_delay_sec = (PACKET_SIZE_BYTES * 8) / allocated_bw if allocated_bw > 0 else HIGH_LATENCY_PENALTY / 1000
            if bw_usage < allocated_bw:
                queuing_delay_sec = 1 / (allocated_bw - bw_usage) if (allocated_bw - bw_usage) > 0 else HIGH_LATENCY_PENALTY / 1000
            else:
                queuing_delay_sec = HIGH_LATENCY_PENALTY / 1000
            ext_latency = PROPAGATION_DELAY_MS + 1000 * (transmission_delay_sec + queuing_delay_sec)
            
            if bw_usage <= allocated_bw:
                ext_loss = BASE_LOSS
            else:
                ext_loss = (bw_usage - allocated_bw) / bw_usage + BASE_LOSS
            
            # Internal QoS (aggregate over NFs)
            int_throughput = float('inf')
            int_latency = 0.0
            int_loss = 0.0  # We'll average losses
            
            for nf in nfs:
                # Effective allocations
                effective_cpu = max(nf['min_cpu'], min(nf['requested_cpu'], nf['limited_cpu']))
                effective_mem = max(nf['min_mem'], min(nf['requested_mem'], nf['limited_mem']))
                
                # Service rate μ (packets/sec)
                mu = (effective_cpu * CPU_CLOCK_SPEED_GHZ * 1e9) / CYCLES_PER_PACKET
                
                # Adjust for overload
                if nf['cpu_usage'] > effective_cpu:
                    mu *= (effective_cpu / nf['cpu_usage'])
                
                # Arrival rate λ (from bw_usage, assume slice bw_usage is input rate)
                lambda_rate = bw_usage / (PACKET_SIZE_BYTES * 8)  # packets/sec
                
                # Throughput per NF
                nf_throughput = min(mu * (PACKET_SIZE_BYTES * 8), bw_usage)  # bps
                if nf['mem_usage'] > effective_mem:
                    nf_throughput *= (effective_mem / nf['mem_usage'])
                int_throughput = min(int_throughput, nf_throughput)
                
                # Latency per NF
                processing_delay_sec = (PACKET_SIZE_BYTES * 8) / (mu * (PACKET_SIZE_BYTES * 8)) if mu > 0 else HIGH_LATENCY_PENALTY / 1000
                if lambda_rate < mu:
                    queuing_delay_sec = 1 / (mu - lambda_rate)
                else:
                    queuing_delay_sec = HIGH_LATENCY_PENALTY / 1000
                
                stall_delay_ms = 0
                if nf['mem_usage'] > 0.8 * effective_mem:
                    miss_rate = max(0, (nf['mem_usage'] - effective_mem) / nf['mem_usage'])
                    stall_delay_ms = miss_rate * MISS_LATENCY_MS
                
                nf_latency = 1000 * (processing_delay_sec + queuing_delay_sec) + stall_delay_ms
                int_latency += nf_latency
                
                # Loss per NF (approximate M/M/1/K)
                buffer_size_packets = (effective_mem * BUFFER_FRACTION * 1e6 * 8) / (PACKET_SIZE_BYTES * 8)  # bits to packets
                K = max(1, int(buffer_size_packets))
                rho = lambda_rate / mu if mu > 0 else 1
                if rho != 1:
                    loss = (rho ** K * (1 - rho)) / (1 - rho ** (K + 1))
                else:
                    loss = 1 / (K + 1)
                if nf['cpu_usage'] > effective_cpu or nf['mem_usage'] > effective_mem:
                    overload_loss = max(0, max((nf['cpu_usage'] - effective_cpu) / nf['cpu_usage'], (nf['mem_usage'] - effective_mem) / nf['mem_usage']))
                    loss = min(1, loss + overload_loss)
                int_loss += loss / len(nfs)  # Average
            
            # Add base propagation to int_latency
            int_latency += PROPAGATION_DELAY_MS
            
            # Check satisfaction
            # TODO consider some error margin for example 10%
            qos_satisfied = {
                'int_latency': int_latency <= s['target_int_latency_ms'],
                'int_loss': int_loss <= s['target_int_loss'],
                'int_throughput': int_throughput >= s['target_int_throughput'],
                'ext_latency': ext_latency <= s['target_ext_latency_ms'],
                'ext_loss': ext_loss <= s['target_ext_loss'],
                'ext_throughput': ext_throughput >= s['target_ext_throughput']
            }
            
            results.append({
                'int_latency': int_latency,
                'int_loss': int_loss,
                'int_throughput': int_throughput,
                'ext_latency': ext_latency,
                'ext_loss': ext_loss,
                'ext_throughput': ext_throughput,
                'qos_satisfied': qos_satisfied
            })
        
        return results




class ClusterSimulator():


    def __init__(self, max_slices: int = 10, nf_num: int = 8):
        self.total_capacity_cpu = 200.0      # cores
        self.total_capacity_mem = 1024.0     # GiB
        self.total_capacity_bw = 1000.0      # MHz

        self.total_remaining_cpu = self.total_capacity_cpu
        self.total_remaining_mem = self.total_capacity_mem
        self.total_remaining_bw = self.total_capacity_bw


        self.active_slices: list[dict] = []

        self.max_slices = max_slices
        self.nf_num = nf_num


    def reset(self):
        self.active_slices.clear()

        self.total_remaining_cpu = self.total_capacity_cpu
        self.total_remaining_mem = self.total_capacity_mem
        self.total_remaining_bw = self.total_capacity_bw

    def add_slice(self):
        if len(self.active_slices) >= self.max_slices:
            return
        nfs = []
        current_remaining_cpu = self.remaining_cpu
        current_remaining_mem = self.remaining_mem
        current_remaining_bw = self.remaining_bw
        # The min_cpu and min_mem are the minimum amount of resource that a Network Function needs to run.
        # They set at the creation time and will not change during the running. They're like resources.requests.

        # The requested_cpu and requested_mem are the current actual resources assigned to the Network Functions. They will change by Resource Allocation agent.
        # The limited_cpu and limited_mem are the maximum resources that a Network Function can use. They're like resources.limits. They will also change by Resource Allocation agent.
        # In this simulation the limited_cpu and limited_mem are coefficient * requested_cpu and requested_mem respectively.
        for _ in range(self.nf_num):
            min_cpu = random.uniform(0.5, 2.0)
            min_mem = random.uniform(2.0, 16.0)
            requested_cpu = random.uniform(1.0, 4.0)
            requested_mem = random.uniform(4.0, 32.0)
            limited_cpu = requested_cpu * random.uniform(1.2, 1.5)
            limited_mem = requested_mem * random.uniform(1.2, 1.5)
            cpu_usage = random.uniform(min_cpu, limited_cpu)
            mem_usage = random.uniform(min_mem, limited_mem)
            if(
                current_remaining_cpu < max(cpu_usage, requested_cpu) and
                current_remaining_mem < max(mem_usage, requested_mem)
            ):
                return
            nfs.append({
                "min_cpu": min_cpu,
                "min_mem": min_mem,
                "requested_cpu": requested_cpu,
                "requested_mem": requested_mem,
                "limited_cpu": limited_cpu,
                "limited_mem": limited_mem,
                "cpu_usage": cpu_usage,
                "mem_usage": mem_usage,
            })
            current_remaining_cpu -= max(cpu_usage, requested_cpu)
            current_remaining_mem -= max(mem_usage, requested_mem)

        min_bw = random.uniform(20.0, 70.0)
        allocated_bw = random.uniform(min_bw, 150.0)
        bw_usage = random.uniform(min_bw, allocated_bw)

        if current_remaining_bw < allocated_bw:
            return
        self.active_slices.append({
            "nfs": nfs,
            "min_bw": min_bw,
            "allocated_bw": allocated_bw,
            "bw_usage": bw_usage,
            "target_int_latency_ms": random.uniform(5.0, 20.0),
            "target_int_loss": random.uniform(0.001, 0.01),
            "target_int_throughput": random.uniform(100.0, 500.0),  # Mbps
            "target_ext_latency_ms": random.uniform(10.0, 50.0),
            "target_ext_loss": random.uniform(0.001, 0.01),
            "target_ext_throughput": random.uniform(50.0, 200.0),  # Mbps
        })

        self.total_remaining_cpu = current_remaining_cpu
        self.total_remaining_mem = current_remaining_mem
        self.total_remaining_bw = current_remaining_bw - allocated_bw

    def remove_slice(self, idx: int | None = None):
        if not self.active_slices:
            return
        if idx is None:
            idx = random.randint(0, len(self.active_slices) - 1)

        for nf in self.active_slices[idx]["nfs"]:
            self.remaining_cpu += max(nf["cpu_usage"], nf["requested_cpu"])
            self.remaining_mem += max(nf["mem_usage"], nf["requested_mem"])
        self.remaining_bw += self.active_slices[idx]["allocated_bw"]
        
        del self.active_slices[idx]

    def update_loads(self):
        # Simulate changes in resource usage for each slice and its NFs
        for s in self.active_slices:
            for nf in s["nfs"]:
                cpu_change = random.gauss(0, 0.2)
                mem_change = random.gauss(0, 1.0)
                if cpu_change > 0:
                    if nf["cpu_usage"] > nf["requested_cpu"] and cpu_change < self.remaining_cpu:
                        nf["cpu_usage"] = np.clip(nf["cpu_usage"] + cpu_change, nf["min_cpu"], nf["limited_cpu"])
                        self.remaining_cpu -= cpu_change
                    elif nf["cpu_usage"] < nf["requested_cpu"] and cpu_change + nf["cpu_usage"] - nf["requested_cpu"] < self.remaining_cpu:
                        nf["cpu_usage"] = np.clip(nf["cpu_usage"] + cpu_change, nf["min_cpu"], nf["limited_cpu"])
                        self.remaining_cpu -= cpu_change + nf["cpu_usage"] - nf["requested_cpu"]
                else:
                    nf["cpu_usage"] = np.clip(nf["cpu_usage"] + cpu_change, nf["min_cpu"], nf["limited_cpu"])
                    self.remaining_cpu -= cpu_change
                
                if mem_change > 0:
                    if nf["mem_usage"] > nf["requested_mem"] and mem_change < self.remaining_mem:
                        nf["mem_usage"] = np.clip(nf["mem_usage"] + mem_change, nf["min_mem"], nf["limited_mem"])
                        self.remaining_mem -= mem_change
                    elif nf["mem_usage"] < nf["requested_mem"] and mem_change + nf["mem_usage"] - nf["requested_mem"] < self.remaining_mem:
                        nf["mem_usage"] = np.clip(nf["mem_usage"] + mem_change, nf["min_mem"], nf["limited_mem"])
                        self.remaining_mem -= mem_change + nf["mem_usage"] - nf["requested_mem"]
                else:
                    nf["cpu_usage"] = np.clip(nf["cpu_usage"] + cpu_change, nf["min_cpu"], nf["limited_cpu"])
                    self.remaining_cpu -= cpu_change

            s["bw_usage"] = np.clip(s["bw_usage"] + random.gauss(0, 10.0), s["min_bw"], s["allocated_bw"])

    def apply_delta(self, idx: int, cpu_deltas: np.ndarray, mem_deltas: np.ndarray, delta_bw: float):
        # Apply resource allocation deltas to the specified slice
        if 0 <= idx < len(self.active_slices):
            s = self.active_slices[idx]
            for j in range(self.nf_num):
                nf = s["nfs"][j]

                if cpu_deltas[j] > 0:
                    if nf["requested_cpu"] >= nf["cpu_usage"] and cpu_deltas[j] < self.total_remaining_cpu:
                        nf["requested_cpu"] = np.clip(nf["requested_cpu"] + cpu_deltas[j], nf["min_cpu"] * 0.8, np.inf)
                        self.remaining_cpu -= cpu_deltas[j]
                    elif nf["requested_cpu"] < nf["cpu_usage"] and cpu_deltas[j] + nf["requested_cpu"] - nf["cpu_usage"] < self.total_remaining_cpu:
                        nf["requested_cpu"] = np.clip(nf["requested_cpu"] + cpu_deltas[j], nf["min_cpu"] * 0.8, np.inf)
                        self.remaining_cpu -= cpu_deltas[j] + nf["requested_cpu"] - nf["cpu_usage"]
                else:
                    nf["requested_cpu"] = np.clip(nf["requested_cpu"] + cpu_deltas[j], nf["min_cpu"] * 0.8, np.inf)
                    self.remaining_cpu -= cpu_deltas[j]

                if mem_deltas[j] > 0:
                    if nf["requested_mem"] >= nf["mem_usage"] and mem_deltas[j] < self.total_remaining_mem:
                        nf["requested_mem"] = np.clip(nf["requested_mem"] + mem_deltas[j], nf["min_mem"] * 0.8, np.inf)
                        self.remaining_mem -= mem_deltas[j]
                    elif nf["requested_mem"] < nf["mem_usage"] and mem_deltas[j] + nf["requested_mem"] - nf["mem_usage"] < self.total_remaining_mem:
                        nf["requested_mem"] = np.clip(nf["requested_mem"] + mem_deltas[j], nf["min_mem"] * 0.8, np.inf)
                        self.remaining_mem -= mem_deltas[j] + nf["requested_mem"] - nf["mem_usage"]
                else:
                    nf["requested_mem"] = np.clip(nf["requested_mem"] + mem_deltas[j], nf["min_mem"] * 0.8, np.inf)
                    self.remaining_mem -= mem_deltas[j]
            if delta_bw > 0 
                if delta_bw <= self.total_remaining_bw:
                    s["allocated_bw"] = np.clip(s["allocated_bw"] + delta_bw, s["min_bw"] * 0.8, self.total_capacity_bw)
                    self.remaining_bw -= delta_bw
            else:
                s["allocated_bw"] = np.clip(s["allocated_bw"] + delta_bw, s["min_bw"] * 0.8, self.total_capacity_bw)
                self.remaining_bw -= delta_bw
    

        remaining_cpu = remaining_mem = remaining_bw = 0.0

        for s in self.simulator.active_slices:
            for nf in s["nfs"]:
                remaining_cpu += max (nf["cpu_usage"] , nf["requested_cpu"])
                remaining_mem += max (nf["mem_usage"] , nf["requested_mem"])
            remaining_bw += s["allocated_bw"]

        remaining_cpu = self.simulator.total_capacity_cpu - remaining_cpu
        remaining_mem = self.simulator.total_capacity_mem - remaining_mem
        remaining_bw = self.simulator.total_capacity_bw - remaining_bw

        return {
            "remaining_cpu": remaining_cpu,
            "remaining_mem": remaining_mem,
            "remaining_bw": remaining_bw,
        }