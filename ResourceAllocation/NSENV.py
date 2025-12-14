import gymnasium as gym
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Box
import numpy as np
import random
from typing import Any,List, Dict
import math





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
            "cluster": Box(low=0.0, high=1e4, shape=(self.qos_params,), dtype=np.float32),

            # QoS feature of Network slices
            # qos_params (e.g. int_latency, ext_latency, int_throughput, ext_throughput, int_packet_loss, ext_packet_loss)
            # qos_params * 2 for current and required for each one.
            "QoS": Box(low=0.0, high=1e4, shape=(self.max_slices, self.qos_params * 2), dtype=np.float32),

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
        self.action_space = Box (low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32)


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


        self.simulator.update_loads(qos_state= self.simulate_qos(self.simulator.active_slices))



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


        self.simulator.update_loads(qos_state = self.simulate_qos(self.simulator.active_slices))


        reward = self._compute_reward()
        info["active_slices"] = len(self.simulator.active_slices)
        metrics = self._get_metrics()
        info.update(metrics)

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

                total_used_cpu += min(nf["cpu_usage"], nf["requested_cpu"])
                total_requested_cpu += nf["requested_cpu"]
                total_used_mem += min(nf["mem_usage"], nf["requested_mem"])
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

            total_bw_over += max(0.0, s["allocated_bw"] - s["bw_usage"] )

            if qos_ok:
                reward += 200.0
            else:
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

        # Penalties for over-provisioning
        reward -= 0.05 * total_cpu_over
        reward -= 0.02 * total_mem_over
        reward -= 0.04 * total_bw_over


        # Penalties for cost of allocations
        reward -= 0.03 * total_requested_cpu
        reward -= 0.01 * total_requested_mem
        reward -= 0.02 * total_allocated_bw

        return reward

        
    def _get_info(self):
        pass
    

    def _get_metrics(self):
        qos = self.simulate_qos(self.simulator.active_slices)
        
        total_used_cpu = total_requested_cpu = total_used_mem = total_requested_mem = total_used_bw = total_allocated_bw = 0.0
        total_cpu_over = total_mem_over = total_bw_over = 0.0
        
        num_ok = num_int_ok = num_ext_ok = 0
        num_active = len(self.simulator.active_slices)
        
        sum_int_latency = sum_int_loss = sum_int_throughput = 0.0
        sum_ext_latency = sum_ext_loss = sum_ext_throughput = 0.0

        total_cluster_cpu_used = total_cluster_mem_used = total_cluster_bw_used = 0.0
        total_cluster_cpu_requested = total_cluster_mem_requested = total_cluster_bw_allocated = 0.0
        
        for i, s in enumerate(self.simulator.active_slices):
            for nf in s["nfs"]:
                total_cpu_over += max(0.0, nf["requested_cpu"] - nf["cpu_usage"])
                total_mem_over += max(0.0, nf["requested_mem"] - nf["mem_usage"])
                
                total_used_cpu += min(nf["cpu_usage"], nf["requested_cpu"])
                total_requested_cpu += nf["requested_cpu"]
                total_used_mem += min(nf["mem_usage"], nf["requested_mem"])
                total_requested_mem += nf["requested_mem"]

                total_cluster_cpu_used += nf["cpu_usage"]
                total_cluster_cpu_requested += nf["requested_cpu"]
                total_cluster_mem_used += nf["mem_usage"]
                total_cluster_mem_requested += nf["requested_mem"]
            
            total_used_bw += s["bw_usage"]
            total_allocated_bw += s["allocated_bw"]
            total_bw_over += max(0.0, s["allocated_bw"] - s["bw_usage"])

            total_cluster_bw_allocated += s["allocated_bw"]
            total_cluster_bw_used += s["bw_usage"]
            
            slice_qos_ok = all(qos[i]["qos_satisfied"].values())
            if slice_qos_ok:
                num_ok += 1
            
            if qos[i]["int_sla_ok"]:
                num_int_ok += 1
            if qos[i]["ext_sla_ok"]:
                num_ext_ok += 1
            
            sum_int_latency += qos[i]["int_latency"]
            sum_int_loss += qos[i]["int_loss"]
            sum_int_throughput += qos[i]["int_throughput"]
            sum_ext_latency += qos[i]["ext_latency"]
            sum_ext_loss += qos[i]["ext_loss"]
            sum_ext_throughput += qos[i]["ext_throughput"]
        
        cpu_util = total_used_cpu / total_requested_cpu if total_requested_cpu > 0 else 0.0
        mem_util = total_used_mem / total_requested_mem if total_requested_mem > 0 else 0.0
        bw_util = total_used_bw / total_allocated_bw if total_allocated_bw > 0 else 0.0
        
        fraction_qos_ok = num_ok / num_active if num_active > 0 else 0.0
        fraction_int_ok = num_int_ok / num_active if num_active > 0 else 0.0
        fraction_ext_ok = num_ext_ok / num_active if num_active > 0 else 0.0
        
        avg_int_latency = sum_int_latency / num_active if num_active > 0 else 0.0
        avg_int_loss = sum_int_loss / num_active if num_active > 0 else 0.0
        avg_int_throughput = sum_int_throughput / num_active if num_active > 0 else 0.0
        avg_ext_latency = sum_ext_latency / num_active if num_active > 0 else 0.0
        avg_ext_loss = sum_ext_loss / num_active if num_active > 0 else 0.0
        avg_ext_throughput = sum_ext_throughput / num_active if num_active > 0 else 0.0
        
        return {
            "active_slices": num_active,
            "cluster_cpu_requested": total_cluster_cpu_requested,
            "cluster_cpu_used": total_cluster_cpu_used,
            "cluster_mem_requested": total_cluster_mem_requested,
            "cluster_mem_used": total_cluster_mem_used,
            "cluster_bw_allocated": total_cluster_bw_allocated,
            "cluster_bw_used": total_cluster_bw_used,
            "cpu_util": cpu_util,
            "mem_util": mem_util,
            "bw_util": bw_util,
            "fraction_qos_ok": fraction_qos_ok,
            "fraction_int_ok": fraction_int_ok,
            "fraction_ext_ok": fraction_ext_ok,
            "total_cpu_over": total_cpu_over,
            "total_mem_over": total_mem_over,
            "total_bw_over": total_bw_over,
            "avg_int_latency": avg_int_latency,
            "avg_int_loss": avg_int_loss,
            "avg_int_throughput": avg_int_throughput,
            "avg_ext_latency": avg_ext_latency,
            "avg_ext_loss": avg_ext_loss,
            "avg_ext_throughput": avg_ext_throughput,
        }

        


    def simulate_qos(self, slices) -> List[Dict]:
        """
        Compute internal and external QoS metrics for a list of slices.

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
            list of dict: Each dict with computed QoS and satisfaction flags:
                - 'int_latency_ms', 'int_loss', 'int_throughput_bps'
                - 'ext_latency_ms', 'ext_loss', 'ext_throughput_bps'
                - 'int_sla_ok' (bool), 'ext_sla_ok' (bool)
                - 'per_nf' : list of per-NF diagnostics (mu, lambda, rho, P_block, T_ms, throughput_bps)
        """

        # ---- Tunable constants ----
        PACKET_SIZE_BYTES = 1500                 # average packet size in bytes
        PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8
        ALPHA_CPU = 1000.0                        # packets/sec per CPU core (tune to your NFs)
        PHI_BUFFER = 0.5                          # fraction of requested_mem used as packet buffer
        PROP_DELAY_PER_HOP_S = 0.001              # propagation/switching delay per internal hop (s)
        EXTERNAL_LINK_CAPACITY_BPS = 100e6        # default external capacity (bps) (tunable)
        EXTERNAL_PROP_DELAY_S = 0.020             # external propagation delay (s) (tunable)
        EXTERNAL_BASE_LOSS = 0.0                  # base non-congestion external loss (e.g. physical loss)
        LARGE_DELAY_S = 10.0                      # used when overloaded (s) - large but finite
        QOS_ERROR_MARGIN = 0.10                   # margin for considering QoS violations

        results = []

        for s in slices:
            allocated_bw = float(s.get('allocated_bw', 0.0)) * 1e6  # capacity for slice (bps)
            bw_usage = float(s.get('bw_usage', 0.0)) * 1e6           # offered (bps)
            # compute offered packet-rate (packets/s)
            lambda_slice_pkts = (bw_usage / PACKET_SIZE_BITS) if PACKET_SIZE_BITS > 0 else 0.0

            nfs = s.get('nfs', [])
            per_nf = []

            # per-NF computations
            nf_throughputs_bps = []
            nf_block_probs = []
            nf_system_times_s = []

            for nf in nfs:
                requested_cpu = max(0.0, float(nf.get('requested_cpu', 0.0)))
                requested_mem_mb = max(0.0, float(nf.get('requested_mem', 0.0)))

                # service rate (pkts/s)
                mu = ALPHA_CPU * requested_cpu

                # arrival rate into this NF - we assume the slice offered pkts traverse the NF (f_j = 1)
                lam = lambda_slice_pkts

                # buffer size K derived from memory: use PHI_BUFFER portion for packets
                buffer_bytes = (requested_mem_mb * 1024 * 1024) * PHI_BUFFER
                pkt_size_bytes = PACKET_SIZE_BYTES if PACKET_SIZE_BYTES > 0 else 1
                K = int(max(0, math.floor(buffer_bytes / pkt_size_bytes)))

                # traffic intensity
                rho = (lam / mu) if mu > 0 else float('inf')

                # blocking probability P_block using M/M/1/K (rho != 1)
                if mu <= 0:
                    P_block = 1.0
                else:
                    if rho == 1.0:
                        # limiting value
                        P_block = 1.0 / (K + 1) if K >= 0 else 1.0
                    else:
                        # handle rho very close to 1 carefully
                        if rho < 1.0:
                            # compute (1-rho) * rho^K / (1 - rho^(K+1))
                            # guard against rho**(K+1) under/overflow by using math.pow
                            try:
                                numerator = (1.0 - rho) * math.pow(rho, K)
                                denom = 1.0 - math.pow(rho, K + 1)
                                P_block = numerator / denom if denom != 0 else 1.0
                            except OverflowError:
                                # when rho^K under/overflows, resort to boundary
                                P_block = 0.0 if rho < 1.0 and K == 0 else min(1.0, math.exp(-abs(K)))
                        else:
                            # rho > 1: extremely overloaded, many arrivals are blocked if buffer finite
                            # Use approx: most packets blocked -> set P_block close to 1 but bounded
                            if K == 0:
                                P_block = 1.0
                            else:
                                # approximate with geometric tail
                                try:
                                    P_block = min(0.999999, math.pow(rho / (1.0 + rho), K))
                                except OverflowError:
                                    P_block = 0.999999

                # mean system time T_j (s) approx
                if mu > lam and lam >= 0:
                    # approximate M/M/1 mean system time
                    try:
                        T = 1.0 / (mu - lam)
                    except ZeroDivisionError:
                        T = LARGE_DELAY_S
                else:
                    T = LARGE_DELAY_S

                # throughput accepted (pkts/s) and convert to bps
                accepted_pkts_per_s = lam * (1.0 - P_block)
                throughput_bps = accepted_pkts_per_s * PACKET_SIZE_BITS

                per_nf.append({
                    'mu_pkts_s': mu,
                    'lambda_pkts_s': lam,
                    'rho': rho,
                    'K_packets': K,
                    'P_block': P_block,
                    'T_s': T,
                    'throughput_bps': throughput_bps
                })

                nf_throughputs_bps.append(throughput_bps)
                nf_block_probs.append(P_block)
                nf_system_times_s.append(T)

            # Compose internal throughput: limited by slice allocated_bw and bottleneck NF throughput
            min_nf_throughput_bps = min(nf_throughputs_bps) if nf_throughputs_bps else float('inf')
            int_throughput_bps = min(allocated_bw if allocated_bw > 0 else float('inf'),
                                    min_nf_throughput_bps if min_nf_throughput_bps != float('inf') else float('inf'))

            # Internal latency: sum NF system times + per-internal-hop tx and prop delays
            sum_nf_system_time_s = sum(nf_system_times_s)
            num_internal_hops = max(0, len(nfs) - 1)   # packets typically traverse (nfs-1) internal links
            tx_delay_per_hop_s = (PACKET_SIZE_BITS / allocated_bw) if allocated_bw > 0 else LARGE_DELAY_S
            internal_link_prop_s = PROP_DELAY_PER_HOP_S
            total_internal_tx_and_prop_s = num_internal_hops * (tx_delay_per_hop_s + internal_link_prop_s)
            int_latency_s = sum_nf_system_time_s + total_internal_tx_and_prop_s
            int_latency_ms = int_latency_s * 1000.0

            # Internal loss: combine NF blocking and link drops
            # link drop approximated as congestion drop if bw_usage > allocated_bw
            if bw_usage > 0 and allocated_bw > 0:
                link_drop_per_hop = max(0.0, 1.0 - (allocated_bw / bw_usage)) if bw_usage > 0 else 0.0
                # clamp
                link_drop_per_hop = min(max(link_drop_per_hop, 0.0), 1.0)
            else:
                link_drop_per_hop = 0.0

            # combined delivered probability = product over (1 - P_block_j) * product over (1 - link_drop)
            prod_nf_success = 1.0
            for pb in nf_block_probs:
                prod_nf_success *= max(0.0, 1.0 - pb)
            prod_links_success = math.pow(max(0.0, 1.0 - link_drop_per_hop), num_internal_hops)
            delivered_prob = prod_nf_success * prod_links_success
            int_loss = 1.0 - delivered_prob
            int_loss = min(max(int_loss, 0.0), 1.0)

            # External path: extend with an external link (use EXTERNAL_LINK_CAPACITY_BPS)
            external_link_capacity = EXTERNAL_LINK_CAPACITY_BPS
            # external link drop due to congestion
            if bw_usage > 0 and external_link_capacity > 0:
                ext_link_drop = max(0.0, 1.0 - (external_link_capacity / bw_usage)) if bw_usage > 0 else 0.0
                ext_link_drop = min(max(ext_link_drop, 0.0), 1.0)
            else:
                ext_link_drop = 0.0

            # external throughput limited by internal throughput and external capacity
            ext_throughput_bps = min(int_throughput_bps if int_throughput_bps is not None else 0.0, external_link_capacity)

            # external latency = internal latency + external tx + external prop
            external_tx_s = (PACKET_SIZE_BITS / external_link_capacity) if external_link_capacity > 0 else LARGE_DELAY_S
            ext_latency_s = int_latency_s + external_tx_s + EXTERNAL_PROP_DELAY_S
            ext_latency_ms = ext_latency_s * 1000.0

            # external loss composes internal loss and external link drops and a base external loss
            ext_loss = 1.0 - ((1.0 - int_loss) * (1.0 - ext_link_drop) * (1.0 - EXTERNAL_BASE_LOSS))
            ext_loss = min(max(ext_loss, 0.0), 1.0)

            # SLA satisfaction flags (interpretation: throughput >= target, latency <= target, loss <= target)
            int_target_latency_ms = float(s.get('target_int_latency_ms', float('inf')))
            int_target_loss = float(s.get('target_int_loss', 1.0))
            int_target_throughput = float(s.get('target_int_throughput', 0.0))

            ext_target_latency_ms = float(s.get('target_ext_latency_ms', float('inf')))
            ext_target_loss = float(s.get('target_ext_loss', 1.0))
            ext_target_throughput = float(s.get('target_ext_throughput', 0.0))


            qos_satisfied = {
                'int_latency': int_latency_ms <= int_target_latency_ms * (1.0 + QOS_ERROR_MARGIN),
                'int_loss': int_loss <= int_target_loss * (1.0 + QOS_ERROR_MARGIN),
                'int_throughput': int_throughput_bps >= int_target_throughput  * (1.0 - QOS_ERROR_MARGIN),
                'ext_latency': ext_latency_ms <= ext_target_latency_ms * (1.0 + QOS_ERROR_MARGIN),
                'ext_loss': ext_loss <= ext_target_loss * (1.0 + QOS_ERROR_MARGIN),
                'ext_throughput': ext_throughput_bps >= ext_target_throughput * (1.0 - QOS_ERROR_MARGIN),
            }

            int_sla_ok = all(qos_satisfied[k] for k in ['int_latency', 'int_loss', 'int_throughput'])
            ext_sla_ok = all(qos_satisfied[k] for k in ['ext_latency', 'ext_loss', 'ext_throughput'])

            results.append({
                'int_latency': float(int_latency_ms),
                'int_loss': float(int_loss),
                'int_throughput': float(int_throughput_bps) / 1e6,  # convert to Mbps
                'ext_latency': float(ext_latency_ms),
                'ext_loss': float(ext_loss),
                'ext_throughput': float(ext_throughput_bps) / 1e6,  # convert to Mbps
                'int_sla_ok': bool(int_sla_ok),
                'ext_sla_ok': bool(ext_sla_ok),
                'qos_satisfied': qos_satisfied,
                'per_nf': per_nf,
                # helpful diagnostics
                'num_internal_hops': num_internal_hops,
                'link_drop_per_internal_hop': float(link_drop_per_hop),
                'external_link_drop': float(ext_link_drop),
                'lambda_slice_pkts': float(lambda_slice_pkts)
            })

        return results





class ClusterSimulator():


    def __init__(self, max_slices: int = 10, nf_num: int = 8):
        self.total_capacity_cpu = 200.0      # cores
        self.total_capacity_mem = 1024.0     # GiB
        self.total_capacity_bw = 1000.0      # MHz

        
        # self.remaining_cpu = self.total_capacity_cpu
        # self.remaining_mem = self.total_capacity_mem
        # self.remaining_bw = self.total_capacity_bw


        self.active_slices: list[dict] = []

        self.max_slices = max_slices
        self.nf_num = nf_num


    def reset(self):
        self.active_slices.clear()

        # self.remaining_cpu = self.total_capacity_cpu
        # self.remaining_mem = self.total_capacity_mem
        # self.remaining_bw = self.total_capacity_bw

    def add_slice(self):
        if len(self.active_slices) >= self.max_slices:
            return
        nfs = []

        # The min_cpu and min_mem are the minimum amount of resource that a Network Function needs to run.
        # They set at the creation time and will not change during the running. They're like resources.requests.

        # The requested_cpu and requested_mem are the current actual resources assigned to the Network Functions. They will change by Resource Allocation agent.
        # The limited_cpu and limited_mem are the maximum resources that a Network Function can use. They're like resources.limits. They will also change by Resource Allocation agent.
        # In this simulation the limited_cpu and limited_mem are coefficient * requested_cpu and requested_mem respectively.
        for _ in range(self.nf_num):
            rem_resources = self.get_remaining_resources()
            min_cpu = random.uniform(0.5, 2.0)
            min_mem = random.uniform(2.0, 16.0)
            requested_cpu = random.uniform(1.0, 4.0)
            requested_mem = random.uniform(4.0, 32.0)
            limited_cpu = requested_cpu * random.uniform(1.2, 1.5)
            limited_mem = requested_mem * random.uniform(1.2, 1.5)
            cpu_usage = random.uniform(min_cpu, limited_cpu)
            mem_usage = random.uniform(min_mem, limited_mem)
            if(
                rem_resources["remaining_cpu"] < max(cpu_usage, requested_cpu) or
                rem_resources["remaining_mem"] < max(mem_usage, requested_mem)
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

        min_bw = random.uniform(50.0, 80.0)
        allocated_bw = random.uniform(min_bw, 150.0)
        bw_usage = random.uniform(min_bw, allocated_bw)

        rem_resources = self.get_remaining_resources()
        if rem_resources["remaining_bw"] < allocated_bw:
            return
        
        # Add the new slice
        self.active_slices.append({
            "nfs": nfs,
            "min_bw": min_bw,
            "allocated_bw": allocated_bw,
            "bw_usage": bw_usage,
            "target_int_latency_ms": random.uniform(5.0, 20.0),
            "target_int_loss": random.uniform(0.001, 0.02),
            "target_int_throughput": random.uniform(100.0, 500.0),  # Mbps
            "target_ext_latency_ms": random.uniform(10.0, 50.0),
            "target_ext_loss": random.uniform(0.01, 0.03),
            "target_ext_throughput": random.uniform(50.0, 200.0),  # Mbps
        })




    def remove_slice(self, idx: int | None = None):
        if not self.active_slices:
            return
        if idx is None:
            idx = random.randint(0, len(self.active_slices) - 1)
        
        del self.active_slices[idx]

    def update_loads(self, qos_state):
        # Simulate changes in resource usage for each slice and its NFs
        slice_idx = 0
        for s in self.active_slices:
            for nf in s["nfs"]:
                rem_resources = self.get_remaining_resources()
                # updating cpu
                nf["cpu_usage"] = np.clip(nf["cpu_usage"] + random.gauss(0, 0.2), nf["min_cpu"], min(nf["limited_cpu"], max(nf["requested_cpu"],nf["cpu_usage"]) + rem_resources["remaining_cpu"]))

                # updating mem
                nf["mem_usage"] = np.clip(nf["mem_usage"] + random.gauss(0, 0.2), nf["min_mem"], min(nf["limited_mem"], max(nf["requested_mem"],nf["mem_usage"]) + rem_resources["remaining_mem"]))


            if( not qos_state[slice_idx]["qos_satisfied"]["ext_throughput"] ):
                s["bw_usage"] = np.clip(s["bw_usage"] + random.gauss(15.0, 2.0), s["min_bw"], s["allocated_bw"])
            else:
                s["bw_usage"] = np.clip(s["bw_usage"] + random.gauss(0, 10.0), s["min_bw"], s["allocated_bw"])

            slice_idx += 1


    def apply_delta(self, idx: int, cpu_deltas: np.ndarray, mem_deltas: np.ndarray, delta_bw: float):
        # Apply resource allocation deltas to the specified slice
        if 0 <= idx < len(self.active_slices):
            s = self.active_slices[idx]
            for j in range(self.nf_num):
                nf = s["nfs"][j]
                rem_resources = self.get_remaining_resources()

                # update cpu 
                nf["requested_cpu"] = np.clip(nf["requested_cpu"] + cpu_deltas[j], nf["min_cpu"],max(nf["requested_cpu"],nf["cpu_usage"])+ rem_resources["remaining_cpu"]) 
                nf["limited_cpu"] = nf["requested_cpu"] * random.uniform(1.2, 1.5)

                # update mem
                nf["requested_mem"] = np.clip(nf["requested_mem"] + mem_deltas[j], nf["min_mem"], nf["requested_mem"] + rem_resources["remaining_mem"])
                nf["limited_mem"] = nf["requested_mem"] * random.uniform(1.2, 1.5)
            
            rem_resources = self.get_remaining_resources()
            # update bw
            s["allocated_bw"] = np.clip(s["allocated_bw"] + delta_bw, s["min_bw"], s["allocated_bw"] + rem_resources["remaining_bw"] )


    def get_remaining_resources(self):
        total_cpu = 0.0
        total_mem = 0.0
        total_bw = 0.0
        for i in range(len(self.active_slices)):
            total_cpu += sum(max(nf["requested_cpu"],nf["cpu_usage"]) for nf in self.active_slices[i]["nfs"])   
            total_mem += sum(max(nf["requested_mem"],nf["mem_usage"]) for nf in self.active_slices[i]["nfs"])
            total_bw += self.active_slices[i]["allocated_bw"]

        return({
            "remaining_cpu": self.total_capacity_cpu - total_cpu,
            "remaining_mem": self.total_capacity_mem - total_mem,
            "remaining_bw": self.total_capacity_bw  - total_bw
        })

if __name__ == "__main__":
    env = NetSliceEnv()
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Info: {info}")
        print(f"Observation: {obs}")
        print(env.simulator.get_remaining_resources())

        input("Press Enter to continue...")